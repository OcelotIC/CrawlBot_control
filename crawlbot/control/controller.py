"""
WholeBodyController — the two-stage controller (spec §4): centroidal NMPC at
10 Hz, whole-body QP at 100 Hz, and the AOCS, driven as one block.

Contract (docs/architecture/unified_planner_architecture.md §4, not the QP's
current 40-argument signature): measurements come in through ``SensorSuite``,
references are queried BY TIME from a ``ReferenceSource`` (``com_at``,
``L_com_at``, ...), and the output of each stage is a typed record —
``NMPCPlan`` for stage 1 — so that the ROS 2 port can map each boundary to a
message. The solvers' own signatures stay internal to this module.

Built incrementally by the refactor/sim-loop-split chantier; every block is
moved verbatim from ``sim_loop.py`` (same expressions, same order), so the
floating-point sequence is unchanged.
"""

import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from crawlbot.core.robot_interface import contact_jacobians


@dataclass
class DiagHooks:
    """Runtime-only diagnostic switches (not config fields), shared by the loop
    and the controller. ``SimulationLoop._diag_*`` are properties onto these.

    pure_pd        strip every feedforward entering the QP (a_com_ff, a_torso_ff,
                   λ_ref) and the NMPC's L_com_ref -> PD feedback only
    freeze_ref     hold r_b_ref / R_b_ref at their first sample
    disable_aocs   force τ_w = 0 every QP sub-step
    lock_arm_joints  zero arm torques, and qvel[arm] after every plant step
    """
    pure_pd: bool = False
    freeze_ref: bool = False
    disable_aocs: bool = False
    lock_arm_joints: bool = False


def ee_data(rs, arm):
    """Return (J_ee, Jdq_ee, oMf_ee) for the given arm."""
    if arm == 'b':
        return rs.J_tool_b, rs.Jdot_dq_tool_b, rs.oMf_tool_b
    else:
        return rs.J_tool_a, rs.Jdot_dq_tool_a, rs.oMf_tool_a


@dataclass
class ControlIntent:
    """What the orchestrator asks of the controller for one NMPC tick.

    ``contact`` is the tick's ContactConfig (which arms are welded);
    ``contact_nmpc`` the same phase with the stance anchor positions;
    ``stance_anchors`` = (anchor A, anchor B), structure frame.

    The four mode fields at the bottom are the caller-side decisions that
    unified_planner_architecture.md §4.3 removes from the controller boundary
    (settling becomes a reference that says "stay here"). They are carried
    explicitly here — not hidden — until that planner exists.
    """
    t: float
    phase: str
    step_idx: int
    contact: Any
    contact_nmpc: Any
    stance_anchors: tuple
    swing_arm: str
    ss_end: float
    settle_mode: bool = False
    passivity_hold: bool = False
    passivity_override: Optional[bool] = None
    ds_centroidal_active: bool = False


@dataclass
class NMPCPlan:
    """Stage-1 output for one NMPC tick — the NMPC→QP contract.

    ``rp, vp, lr, af``: first-step CoM position / velocity reference, contact
    wrench reference λ (12,), CoM feedforward acceleration. ``*_k0/_k1``: the
    first two knots of the planned CoM trajectory, interpolated at QP rate.
    ``rs`` is the robot state the solve was posed from; ``L_com_now`` a copy of
    its L_com (seed of the caller's L_com carry). ``cref_r/cref_v`` the CoM
    reference the NMPC tracked; ``L_com_ref`` / ``t_mid`` what it was asked
    for momentum (telemetry). ``ok / status_code / cost / info / t_ms``: solve
    outcome (status 0 ok, 1 max_iter, 2 infeasible).
    """
    rs: Any
    L_com_now: np.ndarray
    cref_r: np.ndarray
    cref_v: np.ndarray
    rp: np.ndarray
    vp: np.ndarray
    lr: np.ndarray
    af: np.ndarray
    rp_k0: np.ndarray
    rp_k1: np.ndarray
    vp_k0: np.ndarray
    vp_k1: np.ndarray
    ok: bool
    status_code: int
    cost: float
    info: Optional[Any]
    t_ms: float
    L_com_ref: np.ndarray
    t_mid: float


@dataclass
class QPCarry:
    """State threaded through the QP sub-steps of ONE NMPC tick.

    Re-created by ``begin_tracking`` at every NMPC tick. The AOCS history is
    NOT here: the AOCS owns it (``AttitudeController.reset_for_nmpc_tick``).
    ``tau_w_last`` / ``transport_mag_last`` are the tick's last AOCS outputs,
    for telemetry. ``lr`` / ``af`` start as the plan's and are zeroed in place
    of the plan's by ``pure_pd``; ``hw`` is the loop's h_w carry (the same
    array object).
    """
    tau_last: np.ndarray
    tau_w_last: np.ndarray
    transport_mag_last: float
    qp_ok: bool
    hw: np.ndarray
    lr: np.ndarray
    af: np.ndarray
    lambda_qp_sol: Any = None
    p_torso_ref_used: Any = None


@dataclass
class TrackOut:
    """One QP sub-step: the command, and what telemetry reads.

    ``tau`` joint torques to apply (clipped, lock-zeroed); ``tau_w`` wheel
    torques (the RWA is always present). The rest is for
    the loop's diagnostic traces: ``tau_raw`` is the QP torque BEFORE the
    clip (the dock-work and physics traces read that one).
    """
    tau: np.ndarray
    tau_w: Optional[np.ndarray]
    tau_raw: np.ndarray
    rs: Any
    qdd_t: np.ndarray
    lambda_qp: np.ndarray
    qp_ok: bool
    passivity_active: bool
    rp_interp: np.ndarray
    p_torso_ref_used: np.ndarray


class WholeBodyController:
    """NMPC + whole-body QP + AOCS; measurements in, commands out."""

    def __init__(self, cfg, robot, sensors, nmpc, qp, mapping, aocs, gmo,
                 diag, n_qp_per_nmpc):
        self._cfg = cfg
        self._robot = robot
        self._sensors = sensors
        self._nmpc = nmpc
        self._qp = qp
        self._mapping = mapping
        self._aocs = aocs
        self._gmo = gmo
        self._diag = diag
        self._n_qp_per_nmpc = n_qp_per_nmpc
        # ── Mapping-layer state (was on SimulationLoop) ────────────────
        # M7 EE-bisection follow-up: torso linear position at SS entry
        # (set in _setup_torso_for_step). Read by _step() when
        # cfg.mapping_bypass_in_ss is True; otherwise unused.
        self._ss_entry_p_torso: Optional[np.ndarray] = None
        # Option A (T12 fix, 2026-04-22): post-dock blend state for
        # the DS torso position reference. Populated at weld
        # activation; cleared on SS entry. See
        # cfg.ds_ramp_duration_s and Misc/reports/architecture/M7_T12_MEMO.md §5.
        self._ds_ramp_t_start: Optional[float] = None
        self._ds_ramp_p_start: Optional[np.ndarray] = None
        self._ds_ramp_p_end: Optional[np.ndarray] = None
        # _diag_freeze_ref: first-sample r_b_ref / R_b_ref held by the hook.
        self._diag_frozen_r_b_ref: Optional[np.ndarray] = None
        self._diag_frozen_R_b_ref: Optional[np.ndarray] = None
        # Most recent CoMToTorsoMapping δ(q) output; populated by track()
        # only on cycles where the live mapping was invoked (i.e. NOT
        # under mapping_bypass_in_ss). None otherwise.
        self._last_mapping_delta: Optional[np.ndarray] = None
        # Auxiliary: δ(q_current) computed alongside the live mapping
        # (which uses q_planned during SS) for the planned-vs-current
        # mass-distribution diagnostic. Not used in control.
        self._last_mapping_delta_current: Optional[np.ndarray] = None
        # F-RATE: cache for mapping outputs at NMPC rate (10 Hz). The
        # mapping (δ(q), δ̇(q,q̇)) is recomputed once per NMPC tick and
        # reused across the WBC sub-steps. Avoids the q_current → δ → r_b
        # → q feedback loop running at 100 Hz.
        # F-SAT: previous-cycle r_b_ref kept so per-WBC-tick increment
        # can be saturated against the physical body-acceleration bound.
        self._mapping_nmpc_tick: int = -1
        self._mapping_cache_delta: Optional[np.ndarray] = None
        self._mapping_cache_delta_dot: Optional[np.ndarray] = None
        self._last_r_b_ref_out: Optional[np.ndarray] = None
        # F-SAT telemetry: counts of cycles where the increment was
        # clipped, max clip magnitude, per-step bookkeeping.
        self._sat_total_calls: int = 0
        self._sat_clipped_calls: int = 0
        self._sat_max_clip_mm: float = 0.0

    def plan(self, intent, refs):
        """Stage 1 — one centroidal NMPC solve (dt_nmpc = 0.1 s).

        ``refs`` is the ReferenceSource (``com_at``, ``L_com_at``);
        Returns the ``NMPCPlan`` the QP stage consumes.
        """
        cfg = self._cfg
        t, phase, step_idx = intent.t, intent.phase, intent.step_idx
        cc_nmpc, settle_mode = intent.contact_nmpc, intent.settle_mode
        cref_r, cref_v = refs.com_at(t, settle_mode)

        # Robot state in structure frame.
        # Extract structure angular velocity for non-inertial corrections.
        omega_s = self._sensors.omega_struct()
        pq, pv = self._sensors.joint_state()
        rs = self._robot.update(pq, pv, omega_struct=omega_s)
        # Seed for the caller's L_com_prev carry, copied here as it was.
        L_com_now = rs.L_com.copy()

        # Contact config from constant structure-frame anchors (no live reading)
        # ══════════════════════════════════════════════════════════════════
        # STAGE 1 — centroidal NMPC  (dt_nmpc = 0.1 s, once per _step)
        # Contact config, warm start, solve, and the shifted-fallback path when the
        # solve fails. Produces x_plan / u_plan; everything after consumes them.
        # The fallback branch is canonical-unreached BY DESIGN — the canonical run
        # never has nmpc_ok False.
        # ══════════════════════════════════════════════════════════════════
        # NMPC — plans robot motion only; AOCS manages wheels independently.
        nmpc_ok = True
        nmpc_status_code = 0  # 0=ok, 1=max_iter, 2=infeasible
        nmpc_cost_val = np.inf
        t_nmpc_start = time.perf_counter()
        # M3: pass current hw (wheel momentum) so NMPC can compute c_simple
        # for the conservation-law box.
        # M5: pass L_com_ref from the TorsoPlanner — nonzero during SS
        # when the torso is rotating. Prevents the NMPC from treating
        # intentional rotation as a disturbance to be cancelled.
        hw_for_nmpc = self._sensors.wheel_momentum()
        # Query L_com_ref at the horizon midpoint to track the
        # planner's rotation phase reasonably.
        t_mid = t + 0.5 * cfg.nmpc_N * cfg.nmpc_dt
        L_com_ref_nmpc = refs.L_com_at(t_mid)
        if self._diag.pure_pd:
            L_com_ref_nmpc = np.zeros(3)
        L_com_ref_asked = L_com_ref_nmpc.copy()   # telemetry, as captured before
        info_n = None
        # NMPC warm-start diagnosis: tag each solve with locomotion-step
        # index and phase. Read by NMPCSolver.solve when populating
        # step_log. Does not affect control logic.
        try:
            self._nmpc._nmpc.diag_label = (
                f"step{int(step_idx):02d}/{phase}/t={float(t):.2f}")
        except Exception:
            pass
        try:
            rp, vp, _, lr, info_n = self._nmpc.solve(
                r_com=rs.r_com, v_com=rs.v_com, L_com=rs.L_com,
                r_com_ref=cref_r, v_com_ref=cref_v,
                contact_config=cc_nmpc, warm_start=True,
                hw_current=hw_for_nmpc,
                L_com_ref=L_com_ref_nmpc)
            af = self._nmpc.compute_feedforward_acceleration(lr)
            nmpc_ok = info_n.success
            nmpc_cost_val = float(info_n.cost) if np.isfinite(info_n.cost) else np.inf
            if not info_n.success:
                nmpc_status_code = 2 if 'infeasib' in info_n.status.lower() else 1
        except Exception:
            nmpc_ok = False
            nmpc_status_code = 2
            af = np.zeros(3)
        t_nmpc_ms = (time.perf_counter() - t_nmpc_start) * 1000

        # ── M5 Fix 2: infeasibility fallback via receding-horizon shift ─
        # On NMPC failure, do NOT jump to cref.r_com (creates a reference
        # discontinuity that saturates actuators). Instead, warm-shift
        # the previous feasible trajectory by one NMPC step. We do NOT
        # update _last_x_opt in place — repeated failures re-shift the
        # same last-successful trajectory, so the fallback never drifts
        # more than ~1 NMPC step from a real plan.
        x_plan, u_plan, _ = self._nmpc.get_last_trajectory()
        if not nmpc_ok:
            x_shift, u_shift = self._nmpc.get_shifted_fallback()
            if x_shift is not None and u_shift is not None:
                x_plan = x_shift
                u_plan = u_shift
                rp = x_plan[0:3, 1]
                vp = x_plan[3:6, 1]
                lr = u_plan[:, 0]
                af = self._nmpc.compute_feedforward_acceleration(lr)
            else:
                # No previous solve — only possible on the very first
                # NMPC call. Use the reference-level CoM as a last resort
                # (pre-planner if active, else the TorsoPlanner cref).
                rp = np.asarray(cref_r, dtype=float).copy()
                vp = np.asarray(cref_v, dtype=float).copy()
                lr = np.zeros(12)
                af = np.zeros(3)

        # ── M5 Fix 1b: interpolate across QP sub-steps ─────────────────
        # Cache the full trajectory's first two knots for linear
        # interpolation inside the inner QP loop. The control u_0 is
        # piecewise constant over [t, t+dt_nmpc] and is NOT interpolated.
        if x_plan is not None:
            rp_k0 = x_plan[0:3, 0].copy()
            rp_k1 = x_plan[0:3, 1].copy()
            vp_k0 = x_plan[3:6, 0].copy()
            vp_k1 = x_plan[3:6, 1].copy()
        else:
            rp_k0 = rp.copy()
            rp_k1 = rp.copy()
            vp_k0 = vp.copy()
            vp_k1 = vp.copy()


        return NMPCPlan(
            rs=rs, L_com_now=L_com_now, cref_r=cref_r, cref_v=cref_v,
            rp=rp, vp=vp, lr=lr, af=af,
            rp_k0=rp_k0, rp_k1=rp_k1, vp_k0=vp_k0, vp_k1=vp_k1,
            ok=nmpc_ok, status_code=nmpc_status_code, cost=nmpc_cost_val,
            info=info_n, t_ms=t_nmpc_ms,
            L_com_ref=L_com_ref_asked, t_mid=t_mid)

    # ── Stage 2 — whole-body QP (dt_qp = 0.01 s) ──────────────────────────

    def begin_tracking(self, plan, hw):
        """Fresh per-NMPC-tick carry for the QP sub-steps."""
        self._aocs.reset_for_nmpc_tick(plan.rs)
        return QPCarry(
            tau_last=np.zeros(self._robot.n_joints),
            tau_w_last=np.zeros(3),
            transport_mag_last=0.0,
            qp_ok=True,
            hw=hw, lr=plan.lr, af=plan.af)

    def track(self, carry, qs, tq, intent, refs, plan):
        """One QP sub-step: measure, build the torso / swing references,
        solve the whole-body QP, clip, and compute the AOCS wheel torque.

        Writes nothing to the plant — returns the command (``TrackOut``).
        """
        cfg = self._cfg
        qp = self._qp
        t, phase, step_idx = intent.t, intent.phase, intent.step_idx
        cc_ss, cc_nmpc = intent.contact, intent.contact_nmpc
        swing_arm, ss_end = intent.swing_arm, intent.ss_end
        settle_mode = intent.settle_mode
        passivity_hold = intent.passivity_hold
        passivity_override = intent.passivity_override
        ds_centroidal_active = intent.ds_centroidal_active
        rp_k0, rp_k1, vp_k0, vp_k1 = plan.rp_k0, plan.rp_k1, plan.vp_k0, plan.vp_k1
        lr, af, hw = carry.lr, carry.af, carry.hw
        qp_ok = carry.qp_ok

        omega_s = self._sensors.omega_struct()
        pq, pv = self._sensors.joint_state()
        rs = self._robot.update(pq, pv, omega_struct=omega_s)
        Jc, Jdc = contact_jacobians(
            rs, cc_ss.active_contacts[0], cc_ss.active_contacts[1])

        # M5 Fix 1b: linear interpolation along the NMPC trajectory
        # knot 0 -> knot 1 across the 10 QP sub-steps. alpha goes
        # 0, 0.1, 0.2, ..., 0.9 — at qs=0 we target the current
        # state (knot 0, matches rs), at qs=9 we target 90 % of the
        # way to knot 1. This matches the time parameterisation
        # tq = t + qs*dt_qp for the NMPC reference at that time,
        # avoiding the staircase reference that otherwise creates
        # impulsive torques through the mapping.
        alpha_interp = qs / self._n_qp_per_nmpc
        rp_interp = (1.0 - alpha_interp) * rp_k0 + alpha_interp * rp_k1
        vp_interp = (1.0 - alpha_interp) * vp_k0 + alpha_interp * vp_k1

        # Torso reference (structure frame — no struct pose needed at QP rate).
        #
        # M7 v19: mapping-based torso reference in BOTH SS and DS,
        # but SS uses q_planned (quintic interp between step start
        # and end arm configurations) while DS uses q_current.
        # Rationale:
        #   - v12 (mapping + δ̇, q_current everywhere) had SS torso
        #     peak 22 mm — the mapping's δ(q) term implicitly
        #     anticipates arm-induced base drift.
        #   - v16..v18 (TorsoPlanner direct in SS, no mapping) lost
        #     that compensation and peaked at 120+ mm.
        #   - v19 restores the mapping in SS but feeds it q_planned
        #     so it becomes a feedforward term on the nominal
        #     trajectory, not a feedback term on arm tracking error.
        # Cap the query time to ss_end - ε so reference_at() returns
        # the quintic's terminal pose (p_t1, v=0, a=0) during the
        # post-T_step margin/hold window instead of falling through
        # to _hold_reference() which holds p_t0 (initial) — see
        # TorsoPlanner.reference_at().
        tr = refs.torso_at(tq, phase, ss_end)
        if (phase == 'SS' and cfg.mapping_bypass_in_ss
                and self._ss_entry_p_torso is not None):
            # Diagnostic bypass: freeze the linear torso reference at
            # its SS-entry value; angular reference still from
            # TorsoPlanner. Mapping is not called this tick.
            p_torso_ref_used = self._ss_entry_p_torso.copy()
            v_torso_ref_used = np.concatenate([np.zeros(3), tr.v[3:6]])
            a_torso_ff_used = np.concatenate([np.zeros(3), tr.a[3:6]])
        elif (phase in ('SS', 'DS') and self._mapping is not None
                and cfg.use_m2_stack
                and not (cfg.ss_two_task_mode and phase == 'SS')):
            # Phase-2.1 two-task: in SS the torso-pose task is fed the RAW
            # TorsoPlanner quintic+SLERP (the `else` branch below, tr.p/v/a)
            # — NO CoMToTorsoMapping δ. (DS still uses the mapping.)
            af_for_mapping = np.zeros(3) if self._diag.pure_pd else af
            # Planned-vs-current diag (commit 64479ab) confirmed
            # q_current is the right q-source for the δ term. But
            # q_current at the WBC rate (100 Hz) closes a mapping
            # feedback loop that oscillates r_b_ref by up to
            # 237 mm/tick on large swings (commit 1b5b841).
            # F-RATE: recompute δ(q) and δ̇(q,q̇) ONCE per NMPC tick
            # (10 Hz). The (m_total/m_b)·rp_interp term still varies
            # smoothly at WBC rate via the existing interpolation.
            ratio = self._mapping.ratio
            m_b = self._mapping.m_b
            # CLEANUP-14: use_local_delta_mapping was False on the canonical, so
            # only the cached-delta path below ever ran. The loop-free D_local
            # variant and its flag were removed.
            if qs == 0 or self._mapping_cache_delta is None:
                self._mapping_cache_delta = np.asarray(
                    self._mapping.compute_delta(rs.q),
                    dtype=float).copy()
                self._mapping_cache_delta_dot = np.asarray(
                    self._mapping.compute_delta_dot(rs.q, rs.v),
                    dtype=float).copy()
            _delta_q = self._mapping_cache_delta
            _delta_dot = self._mapping_cache_delta_dot
            r_b_ref_m = ratio * rp_interp - _delta_q / m_b
            v_b_ref_m = ratio * vp_interp - _delta_dot / m_b
            a_b_ff_m = ratio * af_for_mapping
            # F-SAT: cap the per-WBC-tick r_b_ref increment at the
            # *planned* torso-reference velocity (|v_b_ref_m|, already
            # feasibility-bounded by the pre-planner CoM trajectory)
            # plus a jitter slack. This clips the multi-m/s cross-tick
            # δ(q_current) jitter the limiter exists for, while letting
            # the reference advance at its commanded rate. The previous
            # cap, (f_max/m_b)·dt²·2 ≈ 0.125mm/tick, was the body's
            # 2-tick *startup* distance — it throttled sustained motion,
            # so the torso reference advanced only ~0.125mm × n_ticks and
            # never reached the dock on large steps (T15 step 2: needed
            # ~590mm, old cap allowed ~200mm), stranding the swing arm.
            if self._last_r_b_ref_out is not None and phase == 'SS':
                v_ref_ff = float(np.linalg.norm(v_b_ref_m))
                threshold = ((v_ref_ff + cfg.fsat_jitter_margin)
                             * cfg.dt_qp)
                delta_rb = r_b_ref_m - self._last_r_b_ref_out
                nrm = float(np.linalg.norm(delta_rb))
                self._sat_total_calls += 1
                if nrm > threshold and nrm > 0.0:
                    r_b_ref_m = (self._last_r_b_ref_out
                                 + threshold * delta_rb / nrm)
                    self._sat_clipped_calls += 1
                    self._sat_max_clip_mm = max(
                        self._sat_max_clip_mm,
                        (nrm - threshold) * 1000.0)
            self._last_r_b_ref_out = r_b_ref_m.copy()
            # Telemetry (still records δ used by the WBC; q_current
            # comparison still computed for the planned-vs-current
            # diagnostic, unchanged).
            self._last_mapping_delta = self._mapping_cache_delta.copy()
            try:
                self._last_mapping_delta_current = np.asarray(
                    self._mapping.compute_delta(rs.q),
                    dtype=float).copy()
            except Exception:
                self._last_mapping_delta_current = None
            # Option A: post-dock blend of the DS torso linear
            # position reference from the SS-exit pose to the live
            # mapping output over cfg.ds_ramp_duration_s. Quintic
            # shape s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5.
            # Orientation reference (tr.R below) is not blended.
            if phase == 'DS':
                T_ramp = cfg.ds_ramp_duration_s
                if (T_ramp > 0.0
                        and self._ds_ramp_t_start is not None
                        and self._ds_ramp_p_start is not None):
                    tau_blend = (tq - self._ds_ramp_t_start) / T_ramp
                    if tau_blend <= 0.0:
                        r_b_ref_m = self._ds_ramp_p_start.copy()
                    elif tau_blend < 1.0:
                        s_blend = (10.0 * tau_blend ** 3
                                   - 15.0 * tau_blend ** 4
                                   + 6.0 * tau_blend ** 5)
                        r_b_ref_m = ((1.0 - s_blend) * self._ds_ramp_p_start
                                     + s_blend * r_b_ref_m)
            p_torso_ref_used = r_b_ref_m
            v_torso_ref_used = np.concatenate([v_b_ref_m, tr.v[3:6]])
            a_torso_ff_used = np.concatenate([a_b_ff_m, tr.a[3:6]])
        else:
            p_torso_ref_used = tr.p
            v_torso_ref_used = tr.v
            a_torso_ff_used = tr.a

        if self._diag.freeze_ref:
            # Freeze r_b_ref / R_b_ref to the first sample taken.
            # Used to probe PD stability at a STATIC target.
            if self._diag_frozen_r_b_ref is None:
                self._diag_frozen_r_b_ref = p_torso_ref_used.copy()
                self._diag_frozen_R_b_ref = tr.R.copy()
            p_torso_ref_used = self._diag_frozen_r_b_ref.copy()
            R_b_ref_frozen = self._diag_frozen_R_b_ref.copy()
            v_torso_ref_used = np.zeros(6)
            a_torso_ff_used = np.zeros(6)

        if self._diag.pure_pd:
            # Strip all feedforward terms entering the QP.
            a_torso_ff_used = np.zeros(6)
            lr = np.zeros(12)
            af = np.zeros(3)
        R_torso_ref_used = (self._diag_frozen_R_b_ref
                            if self._diag.freeze_ref else tr.R)
        tkw = dict(
            J_torso=rs.J_torso, Jdot_dq_torso=rs.Jdot_dq_torso,
            p_torso=rs.oMf_torso.translation,
            R_torso=rs.oMf_torso.rotation,
            p_torso_ref=p_torso_ref_used, R_torso_ref=R_torso_ref_used,
            v_torso_ref=v_torso_ref_used, a_torso_ff=a_torso_ff_used)

        ek = {}
        if phase == 'SS':
            # M7: SwingPlanner plans over [0, T_step] (same horizon
            # as the torso planner). Its quintic timing `tau =
            # clip((t - t_start)/T, 0, 1)` naturally clamps past
            # T_step to the terminal reference (p=p_dock, v=0, a=0),
            # which is exactly what the margin/hold windows need —
            # no separate EXT approach-velocity reference is
            # required.
            sr = refs.swing_at(tq, phase, ss_end)
            if sr.is_swinging and sr.swing_arm == swing_arm:
                J_ee, Jdq_ee, oMf_ee = ee_data(rs, swing_arm)
                ek = dict(J_ee=J_ee, Jdot_dq_ee=Jdq_ee,
                          p_ee=oMf_ee.translation, R_ee=oMf_ee.rotation,
                          p_ee_ref=sr.p_ee, R_ee_ref=sr.R_ee,
                          v_ee_ref=np.concatenate([sr.v_ee, sr.omega_ee]),
                          a_ee_ff=np.concatenate([sr.a_ee, sr.alpha_ee]))

        # M2: enable passivity inequality during DS (settling) when the
        # reworked task stack is active. settle_mode already bypasses
        # torso/EE tasks; passivity just adds dq^T*tau_q + 2α*T ≤ 0.
        # M7: passivity active during DS (energy-based settle) and
        # during the SS convergence-hold window (passivity_hold=True).
        # Main loop engages hold mode once the trajectory has ended
        # without docking, so the system dissipates residual energy
        # while the EE closes on the target.
        passivity_active = bool(
            cfg.use_m2_stack and (phase == 'DS' or passivity_hold))
        if passivity_override is not None:
            passivity_active = bool(passivity_override)

        # WBC slack/tube diagnosis: tag each QP solve with locomotion
        # step + phase + sub-step. Read by WholeBodyQP.solve when
        # populating hw_slack_log. Does not affect control logic.
        try:
            qp.diag_label = (
                f"step{int(step_idx):02d}/{phase}/qs={int(qs):02d}")
        except Exception:
            pass
        # Piste A LOT A: envelope-coupled passivity work-budget. Per tick,
        # W_budget = β·α·max(0, τ_w_max − ‖Ḣ_s(lambda_ref)‖∞), where
        # Ḣ_s(lambda_ref) = Σ r_Cj×f_j_ref + τ_j_ref is the NMPC-EXACT
        # origin-referenced planned momentum-rate (the same quantity the
        # NMPC envelope path constraint enforces). A pre-solve scalar that
        # opens the strict passivity RHS only by the planned envelope
        # headroom. β=0 ⇒ None ⇒ strict (byte-identical).
        try:
            qdd_t_qp, qdd_qp, lambda_qp_sol, tau, _ = qp.solve(
                dq_t=rs.dq_torso,
                q=rs.q_joints, dq=rs.dq_joints,
                r_com_ref=rp_interp, v_com_ref=vp_interp,
                lambda_ref=lr, a_com_ff=af,
                H_robot=rs.H, C_robot=rs.C,
                J_com=rs.J_com, Jdot_dq_com=rs.Jdot_dq_com,
                contact_config=cc_ss, J_contacts=Jc, Jdot_dq_contacts=Jdc,
                hw_current=hw,
                hw_min=-cfg.hw_qp_tight, hw_max=cfg.hw_qp_tight,
                r_com=rs.r_com, L_com_current=rs.L_com,
                settle_mode=settle_mode,
                passivity_active=passivity_active,
                ds_centroidal_active=ds_centroidal_active,
                # strict passivity RHS (the Piste-A budget was removed)
                passivity_W_budget=None,
                **tkw, **ek)
        except Exception as _qp_exc:
            # Surface QP failures (silent swallow -> zero torque is
            # dangerous). Fall back to zero torque of the correct
            # actuator dimension.
            print(f"  [QP-FAIL] {phase} t={t:.2f} qs={qs}: "
                  f"{type(_qp_exc).__name__}: {_qp_exc}")
            tau = np.zeros(self._robot.n_joints)
            lambda_qp_sol = np.zeros(12)
            qdd_t_qp = np.zeros(6)
            qdd_qp = np.zeros(self._robot.model.nv)
            qp_ok = False

        tau_raw = tau

        tau = np.clip(tau, -cfg.tau_max, cfg.tau_max)
        tau_last = tau.copy()
        if self._diag.lock_arm_joints:
            # Diagnostic: zero all joint torques so the arms hold
            # their initial config (qvel[arm] is forced to 0 after
            # every mj_step below).
            tau = np.zeros_like(tau)
            tau_last = tau.copy()

        # AOCS: reaction wheel torque command.
        tau_w_cmd, transport_mag_last = self._aocs.command(
            phase=phase, rs=rs, lambda_qp_sol=lambda_qp_sol,
            cc_nmpc=cc_nmpc,
            stance_anchors=intent.stance_anchors)
        tau_w_last = tau_w_cmd.copy()

        carry.lr, carry.af = lr, af
        carry.qp_ok = qp_ok
        carry.tau_last, carry.tau_w_last = tau_last, tau_w_last
        carry.transport_mag_last = transport_mag_last
        carry.lambda_qp_sol = lambda_qp_sol
        carry.p_torso_ref_used = p_torso_ref_used
        return TrackOut(
            tau=tau, tau_w=tau_w_cmd, tau_raw=tau_raw, rs=rs,
            qdd_t=qdd_t_qp, lambda_qp=lambda_qp_sol, qp_ok=qp_ok,
            passivity_active=passivity_active, rp_interp=rp_interp,
            p_torso_ref_used=p_torso_ref_used)

    def after_step(self, carry, tau):
        """Post-physics-step estimation: GMO update and the h_w carry."""
        omega_s_post = self._sensors.omega_struct()
        rs2 = self._robot.update(
            *self._sensors.joint_state(),
            omega_struct=omega_s_post)

        # GMO update (100 Hz, after physics step)
        tau_applied = np.zeros(self._robot.model.nv)
        tau_applied[6:6 + self._robot.n_joints] = tau
        self._gmo.update(rs2.H, rs2.v, rs2.C_matrix, tau_applied)

        hw = self._sensors.wheel_momentum()
        # Do NOT clip hw here. The QP's hw safety constraint is
        # soft (slack variables with heavy quadratic penalty), so
        # the QP stays feasible even when physical hw is beyond
        # the box, and actively generates the maximum corrective
        # wrench it can.

        carry.hw = hw

    # ── Events from the orchestrator (reference shaping state) ────────────

    def on_ss_entry(self, p_torso_entry):
        """SS entry: record the torso position and reset the post-dock DS
        blend (Option A); the next dock re-arms it."""
        self._ss_entry_p_torso = p_torso_entry.copy()
        self._ds_ramp_t_start = None
        self._ds_ramp_p_start = None
        self._ds_ramp_p_end = None

    def on_dock(self, t):
        """Weld activation: start the post-dock DS blend from the SS-entry
        torso position (Option A; see Misc/reports/architecture/M7_T12_MEMO.md §5).
        The blend endpoint is not stored — the DS torso reference recomputes
        the live mapping output each tick and blends it against the start."""
        self._ds_ramp_t_start = float(t)
        if self._ss_entry_p_torso is not None:
            self._ds_ramp_p_start = (
                self._ss_entry_p_torso.copy())

    # ── DS passivity settle (inter-step / setup) ──────────────────────────

    def begin_settle(self):
        """Entry of an inter-step DS settle: seed the AOCS history."""
        self._aocs.reset_for_settle()

    def settle(self, rs, cc_ds, hw_current, fallback_Kd):
        """One tick of the passivity-constrained settle QP (NMPC bypassed).

        Returns ``(tau, lambda_qp_sol, tau_w_applied, wheel_cmd)``:
        ``wheel_cmd`` is what to write to the wheel actuators — 0.0 (AOCS off
        in the inter-step settle) or ``tau_w_applied``. The AOCS keeps its
        own ω_s history (``begin_settle`` seeds it).
        """
        cfg = self._cfg
        Jc, Jdc = contact_jacobians(rs, True, True)

        # c_curr (J2): refresh the QP's hw_current parameter to the LIVE
        # wheel momentum this tick. With the AOCS active in this loop the
        # wheels move, so the entry-frozen hw_current goes stale and the
        # QP momentum-safety box loses C5 margin. Reading qvel[6:9] here
        # (same source as the _step per-tick refresh) and passing it as
        # the frozen hw parameter mirrors c_simple(k): a fresh numeric
        # value per solve, frozen during the solve, NOT a decision
        # variable (non-co-integration audit — the QP decision vector
        # {qdd_t, qdd, λ, τ_q, slack} is unchanged). Flag OFF ⇒ the
        # entry-frozen value (byte-identical). At k=0 qvel is unchanged
        # since entry ⇒ the first solve is identical either way.
        if cfg.interstep_hw_refresh:
            hw_for_qp = self._sensors.wheel_momentum()
        else:
            hw_for_qp = hw_current

        try:
            # lambda_qp_sol captured for logging-only (was `_` before
            # Phase B). No control change — value is not consumed
            # downstream of this branch.
            _, _, lambda_qp_sol, tau, _ = self._qp.solve(
                dq_t=rs.dq_torso,
                q=rs.q_joints, dq=rs.dq_joints,
                r_com_ref=rs.r_com, v_com_ref=np.zeros(3),
                lambda_ref=np.zeros(12), a_com_ff=np.zeros(3),
                H_robot=rs.H, C_robot=rs.C,
                J_com=rs.J_com, Jdot_dq_com=rs.Jdot_dq_com,
                contact_config=cc_ds,
                J_contacts=Jc, Jdot_dq_contacts=Jdc,
                hw_current=hw_for_qp,
                hw_min=cfg.hw_min, hw_max=cfg.hw_max,
                r_com=rs.r_com, L_com_current=rs.L_com,
                J_torso=rs.J_torso,
                Jdot_dq_torso=rs.Jdot_dq_torso,
                p_torso=rs.oMf_torso.translation.copy(),
                R_torso=rs.oMf_torso.rotation.copy(),
                p_torso_ref=rs.oMf_torso.translation.copy(),
                R_torso_ref=rs.oMf_torso.rotation.copy(),
                v_torso_ref=np.zeros(6),
                a_torso_ff=np.zeros(6),
                settle_mode=True,
                passivity_active=True,
                settle_alpha_wrench=(cfg.interstep_settle_alpha_wrench
                                     if cfg.interstep_settle_alpha_wrench > 0
                                     else None))
            tau = np.clip(tau, -cfg.tau_max, cfg.tau_max)
        except Exception:
            tau = -fallback_Kd * rs.dq_joints
            tau = np.clip(tau, -cfg.tau_max, cfg.tau_max)
            lambda_qp_sol = np.zeros(12)

        # AOCS re-activation (free-floating invariant — J2 step 4a).
        # The structure is free-floating every tick, so the AOCS must
        # run in DS too. Flag ON (default): run the canonical
        # legacy_pid_numerical AOCS with the DS wrench FF from the
        # settle-QP wrench (lambda_qp_sol). Flag OFF: the legacy
        # hardcoded-zero path, byte-identical to the pre-fix loop.
        tau_w_applied = np.zeros(3)
        if cfg.aocs_active_in_interstep:
            tau_w_applied = self._aocs.command_interstep(
                rs, cc_ds, lambda_qp_sol)
            wheel_cmd = tau_w_applied
        else:
            wheel_cmd = 0.0
        return tau, lambda_qp_sol, tau_w_applied, wheel_cmd
