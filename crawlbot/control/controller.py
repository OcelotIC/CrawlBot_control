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


class WholeBodyController:
    """NMPC + whole-body QP + AOCS; measurements in, commands out."""

    def __init__(self, cfg, robot, sensors, nmpc, has_rwa, diag):
        self._cfg = cfg
        self._robot = robot
        self._sensors = sensors
        self._nmpc = nmpc
        self._has_rwa = has_rwa
        self._diag = diag

    def plan(self, t, *, phase, step_idx, cc_nmpc, settle_mode, refs, hw):
        """Stage 1 — one centroidal NMPC solve (dt_nmpc = 0.1 s).

        ``refs`` is the ReferenceSource (``com_at``, ``L_com_at``);
        ``cc_nmpc`` the contact configuration with the stance anchors;
        ``hw`` the loop's h_w carry (used only without reaction wheels).
        Returns the ``NMPCPlan`` the QP stage consumes.
        """
        cfg = self._cfg
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
        if self._has_rwa:
            hw_for_nmpc = self._sensors.wheel_momentum()
        else:
            hw_for_nmpc = hw.copy()
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
