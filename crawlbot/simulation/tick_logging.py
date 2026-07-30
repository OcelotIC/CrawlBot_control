"""Per-tick recorders — the half of the loop that only writes the log.

`SimulationLoop` is the controller; this is its telemetry. The two are separated
because they fail differently: a mistake here corrupts a plot, a mistake there
corrupts the robot. Keeping them apart means a reader chasing a control question
never has to page through 400 lines of `log.*.append`, and a reader chasing a
missing channel knows exactly which file to open.

Not to be confused with `logging.py`, which defines `SimLog` — the *container*
these write into, plus environment capture. This module is the *writers*.

Contents:
    TickState         what `_step` produces that its recorder consumes
    TickLoggingMixin  `_log_ds_tick` (double support) and `_log_ss_tick`
                      (single support), mixed into SimulationLoop

Both recorders are pure with respect to control: they read `self`, the MuJoCo /
Pinocchio state and their arguments, and write only to `log`. Two calls do mutate
— `mujoco.mj_forward` and `robot.update` — but both only recompute derived
quantities from unchanged `qpos`/`qvel`. That property is what lets the canonical
gate settle any change here: if the exported artifact is byte-identical, the
control path provably did not move.

The mixin resolves three calls on the final class rather than here —
`_get_ee_data`, `_gripper_distance`, `_swing_query_time`. They are geometry
queries owned by the loop; duplicating them would be worse than the coupling.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    import mujoco
except ImportError:
    mujoco = None

try:
    import pinocchio as pin
except ImportError:
    pin = None

from crawlbot.core.state_conversions import (
    mujoco_to_pinocchio, quat_wxyz_to_euler_deg)


@dataclass
class TickState:
    """Everything `_step` produces that its logging tail consumes.

    `_step` ends with ~190 lines that only *record* what the tick did — no
    control decision is taken there. Extracting that into `_log_ss_tick` (the
    single-support counterpart of the long-standing `_log_ds_tick`) needs the
    values to cross one boundary, and passing them positionally would mean a
    29-argument signature: the same debt as `WholeBodyQP.solve()`'s 40
    parameters, which `CLEANUP_CARRYOVER` §A1 already tracks. One record
    instead, built once at the boundary.

    Membership is **measured, not chosen**: a field is here iff the tail reads
    it before writing it. That test matters — `L_dot_est` and `R_err` look like
    inputs (the head assigns both) but the tail *recomputes* them from `rs_f`
    at :3176 and :3187, so they are locals of the logger, not fields here.

    Names come from where each value is logged, not from the head's
    abbreviations: `lr` is the NMPC contact-wrench reference (`log.lambda_ref`)
    so it is `lambda_ref`; `vp` is the planned CoM velocity (`log.v_com_ref`)
    so it is `v_com_ref`; `cref_r` is `r_com_ref`.

    Not fields, deliberately:
      ``cfg``  — it is only ever ``self.cfg``; the method reads it from self.
      ``log``  — the destination, not tick state. Passed separately.
    """

    # ── step context: what tick this is, and in which phase ──
    t: float                          # [s] simulation time at tick start
    phase: str                        # 'SS' | 'DS'
    step_idx: int                     # index into the contact sequence
    ss_end: float                     # [s] end of the current single-support
    settle_mode: bool                 # DS settle sub-phase active
    swing_arm: str                    # 'a' | 'b' — the arm in flight
    stance_arm: str                   # 'a' | 'b' — the anchored arm
    stance_a: object                  # SE3 anchor pose, arm A
    stance_b: object                  # SE3 anchor pose, arm B
    target_anchor: object             # SE3 the swing arm is reaching for
    hw: np.ndarray                    # (3,) [N·m·s] wheel momentum this tick
    L_com_prev: np.ndarray            # (3,) [N·m·s] centroidal L at tick start,
    #                                   differenced against L_com to get L̇

    # ── stage 1 outcome: the centroidal NMPC ──
    nmpc_ok: bool                     # solve succeeded
    nmpc_status_code: int             # solver status, numeric
    nmpc_cost_val: float              # optimal cost (inf when not solved)
    t_nmpc_ms: float                  # [ms] wall-clock in the NMPC
    nmpc_info: object                 # solver info object, or None
    lambda_ref: np.ndarray            # (12,) contact-wrench reference, u_plan[:,0]
    v_com_ref: np.ndarray             # (3,) [m/s] planned CoM velocity, knot 1
    r_com_ref: np.ndarray             # (3,) [m] CoM position reference fed to
    #                                   the NMPC (pre-planner when active)

    # ── stage 2 outcome: the whole-body QP, last sub-step of the tick ──
    qp_ok: bool                       # every QP sub-step succeeded
    t_qp_ms: float                    # [ms] wall-clock in the QP sub-loop
    lambda_qp: np.ndarray             # (12,) realized contact wrench
    tau_joints: np.ndarray            # (nq,) [N·m] joint torques
    tau_wheels: np.ndarray            # (3,) [N·m] wheel torques
    transport_term_mag: float         # [N·m] magnitude of the transport term

    # ── torso reference actually tracked ──
    # The output of the mapping layer at the LAST sub-step, which is what the
    # QP tracked — NOT the geometric TorsoPlanner quintic. None when the QP
    # sub-loop did not run, in which case the logger falls back to the planner.
    p_torso_ref_used: Optional[np.ndarray] = None


class TickLoggingMixin:
    """The per-tick recorders, mixed into `SimulationLoop`."""

    def _log_ds_tick(self, log, t_abs, step_idx, just_landed_arm,
                     anchor_a_idx, anchor_b_idx, tau, lambda_qp_sol,
                     tau_w_applied=None):
        """Append one per-tick DS log row (logging-only; no control change).

        Called from ``_run_ds_passivity_loop`` AFTER ``mj_step``. The row
        has the SAME field set as the SS row written by ``_step`` so the
        SimLog stays schema-consistent (every list grows by 1 per tick,
        all lengths equal). NaN sentinels are used for NMPC-side fields
        (NMPC is bypassed in this loop); zeros for the AOCS-side wheel
        torque (wheels are commanded to zero at line ~754).

        Phase-B guarantee: this method only reads MuJoCo / Pinocchio
        state and appends to the log. It does not call qp.solve, does
        not change ``self.mj_data.ctrl``, does not advance MuJoCo time,
        does not mutate ``self.gmo`` / ``self.contact_sm`` / planner
        state. Side-effect-free w.r.t. control.
        """
        cfg = self.cfg
        nq = self.robot.n_joints

        # Recompute Pinocchio state from current MuJoCo state.
        # This duplicates the next iteration's top-of-loop computation
        # by one tick; harmless (the next iteration recomputes anyway).
        pq, pv = mujoco_to_pinocchio(self.mj_data.qpos, self.mj_data.qvel)
        rs = self.robot.update(pq, pv)

        # Torso planner reference at the absolute tick time, clamped
        # into the phase window: during inter-step DS (t past the
        # step's t_end) this holds the quintic's TERMINAL pose p_t1, so
        # the exported p_torso_ref is continuous across SS->DS instead
        # of jumping back to the set_hold pose p_t0 (a full step stride
        # behind). Logging-only — the DS QP does not consume this value
        # (settle_mode gates all torso tasks off).
        tref = self.torso_planner.reference_at_clamped(t_abs)
        p_torso_ref = tref.p.copy()
        R_torso_ref = tref.R.copy()
        q_torso_ref = pin.Quaternion(R_torso_ref)
        # Torso tracking error (geodesic angle).
        R_err = R_torso_ref.T @ rs.oMf_torso.rotation
        angle_err = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))

        # End-effector data for the JUST-LANDED arm (the one whose
        # contact tenure we are measuring). If '' (initial DS), default
        # to arm A so the field is well-defined; the post-processor
        # knows to drop initial-DS rows.
        ee_arm = just_landed_arm if just_landed_arm in ('a', 'b') else 'a'
        J_ee, _, oMf_ee = self._get_ee_data(rs, ee_arm)
        # d_grip_swing = distance to the welded contact anchor (just-landed
        # arm). For initial DS (anchor < 0), report NaN.
        try:
            if ee_arm == 'a' and anchor_a_idx >= 0:
                d_swing = self._gripper_distance('a', anchor_a_idx)
            elif ee_arm == 'b' and anchor_b_idx >= 0:
                d_swing = self._gripper_distance('b', anchor_b_idx)
            else:
                d_swing = float('nan')
        except Exception:
            d_swing = float('nan')
        # d_grip_stance: the OTHER arm (continuing stance).
        try:
            if ee_arm == 'a' and anchor_b_idx >= 0:
                d_stance = self._gripper_distance('b', anchor_b_idx)
            elif ee_arm == 'b' and anchor_a_idx >= 0:
                d_stance = self._gripper_distance('a', anchor_a_idx)
            else:
                d_stance = float('nan')
        except Exception:
            d_stance = float('nan')

        # Wheel momentum from MJ wheel state.
        if self.has_rwa:
            rw_vel = self.mj_data.qvel[6:9].copy()
            hw = (cfg.rwa_I_w * rw_vel).copy()
        else:
            rw_vel = np.zeros(3)
            hw = np.zeros(3)

        # SS convention: tau_max_joint is the SCALAR max|tau| at this tick.

        # ── Schema-identical append ───────────────────────────────────
        log.t.append(float(t_abs))
        log.phase.append('DS')
        log.step_idx.append(int(step_idx))

        # Torso tracking
        log.p_torso.append(rs.oMf_torso.translation.copy())
        log.p_torso_ref.append(p_torso_ref)
        log.e_torso_pos.append(float(
            np.linalg.norm(rs.oMf_torso.translation - p_torso_ref)))
        log.e_torso_ori.append(float(np.degrees(angle_err)))
        q_torso_actual = pin.Quaternion(rs.oMf_torso.rotation)
        log.q_torso.append(np.array([q_torso_actual.w, q_torso_actual.x,
                                      q_torso_actual.y, q_torso_actual.z]))
        log.q_torso_ref.append(np.array([q_torso_ref.w, q_torso_ref.x,
                                          q_torso_ref.y, q_torso_ref.z]))

        # End-effector
        log.d_grip_swing.append(float(d_swing))
        log.d_grip_stance.append(float(d_stance))
        log.swing_arm.append(just_landed_arm)
        log.p_ee.append(oMf_ee.translation.copy())
        # No swing planner reference in DS — anchor IS the target.
        log.p_ee_ref.append(oMf_ee.translation.copy())
        q_ee_a = pin.Quaternion(oMf_ee.rotation)
        log.q_ee.append(np.array([q_ee_a.w, q_ee_a.x, q_ee_a.y, q_ee_a.z]))
        log.q_ee_ref.append(np.array([q_ee_a.w, q_ee_a.x,
                                       q_ee_a.y, q_ee_a.z]))

        # Joint / EE / torso velocities
        v_a = rs.J_tool_a @ rs.v
        v_b = rs.J_tool_b @ rs.v
        v_t = rs.J_torso @ rs.v
        log.qvel_joints_a.append(rs.dq_joints[:nq // 2].copy())
        log.qvel_joints_b.append(rs.dq_joints[nq // 2:].copy())
        log.v_ee_a.append(v_a[:3].copy())
        log.omega_ee_a.append(v_a[3:].copy())
        log.v_ee_b.append(v_b[:3].copy())
        log.omega_ee_b.append(v_b[3:].copy())
        log.v_torso.append(v_t[:3].copy())
        log.omega_torso.append(v_t[3:].copy())

        # CoM (use measured as ref → e_com = 0)
        log.r_com.append(rs.r_com.copy())
        log.r_com_ref.append(rs.r_com.copy())
        log.e_com.append(0.0)
        log.v_com.append(rs.v_com.copy())
        log.v_com_ref.append(np.zeros(3))

        # Momentum (L_dot left as zeros — would require prior tick state)
        log.L_com.append(rs.L_com.copy())
        log.L_com_norm.append(float(np.linalg.norm(rs.L_com)))
        log.L_com_ref.append(np.zeros(3))
        log.L_dot.append(np.zeros(3))
        log.L_dot_norm.append(0.0)
        log.hw.append(hw.copy())

        # RWA physical. With the inter-step AOCS re-activated (flag on),
        # tau_w_applied is the ACTUAL wheel torque commanded this tick;
        # when the AOCS is off (flag off / no RWA) it is zeros — the
        # legacy commanded value. None defaults to zeros for any caller
        # that does not pass it (schema-safe).
        log.hw_physical.append(hw.copy())
        log.tau_w.append((np.zeros(3) if tau_w_applied is None
                          else np.asarray(tau_w_applied, dtype=float)).copy())
        log.rw_speed.append(rw_vel.copy())

        # EE tracking error (vs same-frame ref → 0).
        log.e_ee_pos.append(0.0)
        log.e_ee_ori.append(0.0)

        # GMO / contact estimator: not updated in DS loop; report best-
        # effort current state (the GMO object is stateful; reading it
        # does NOT mutate). Fall back to zeros on any attribute error.
        try:
            log.gmo_residual_norm.append(float(np.linalg.norm(self.gmo.residual)))
        except Exception:
            log.gmo_residual_norm.append(0.0)
        # gmo_swing_residual requires a v-slice; not tracked in DS.
        log.gmo_swing_residual.append(0.0)
        try:
            log.gmo_contact_state.append(self.contact_sm.state.value)
        except Exception:
            log.gmo_contact_state.append(0)

        # H_rO / H estimator diagnostics — H_estimator not advanced here.
        try:
            log.H_rO.append(self.H_estimator.H_rO.copy())
            log.H_dot_est.append(self.H_estimator.H_dot.copy())
        except Exception:
            log.H_rO.append(np.zeros(3))
            log.H_dot_est.append(np.zeros(3))
        log.omega_struct.append(self.mj_data.qvel[3:6].copy())
        log.qfrc_constraint_torque.append(np.zeros(3))

        # Joint torques (clipped value commanded to MJ).
        log.tau.append(tau.copy())
        log.tau_max_joint.append(float(np.max(np.abs(tau))))

        # Structure state.
        log.struct_pos.append(self.mj_data.qpos[0:3].copy())
        log.struct_quat.append(self.mj_data.qpos[3:7].copy())
        # Euler (zyx) from quat — small-angle is fine for diag.
        qw_s, qx_s, qy_s, qz_s = self.mj_data.qpos[3:7]
        log.struct_euler_deg.append(np.degrees(np.array([
            np.arctan2(2*(qw_s*qx_s + qy_s*qz_s), 1 - 2*(qx_s*qx_s + qy_s*qy_s)),
            np.arcsin(np.clip(2*(qw_s*qy_s - qz_s*qx_s), -1, 1)),
            np.arctan2(2*(qw_s*qz_s + qx_s*qy_s), 1 - 2*(qy_s*qy_s + qz_s*qz_s)),
        ])))
        log.omega_s.append(self.mj_data.qvel[3:6].copy())

        # ── NMPC-bypassed sentinels ───────────────────────────────────
        # lambda_ref: NaN sentinel — there is NO plan in this regime.
        log.lambda_ref.append(np.full(12, np.nan))
        log.lambda_ref_norm.append(float('nan'))
        # QP-side wrenches: real values from this loop's QP solve.
        log.lambda_qp.append(np.asarray(lambda_qp_sol, dtype=float).copy())
        log.lambda_qp_norm.append(float(np.linalg.norm(lambda_qp_sol)))
        log.nmpc_ok.append(False)  # not run; not a failure
        log.qp_ok.append(True)
        log.nmpc_time_ms.append(0.0)
        log.qp_time_ms.append(0.0)
        log.nmpc_status.append(3)  # 3 = BYPASSED sentinel (post-proc reads)
        log.nmpc_cost.append(float('nan'))
        log.nmpc_status_str.append('NMPC_BYPASSED')
        log.nmpc_iterations.append(0)
        log.transport_term_mag.append(0.0)

        # Energy / passivity
        log.T_kinetic.append(0.5 * float(rs.v @ rs.H @ rs.v))

    def _log_ss_tick(self, log, ts: 'TickState'):
        """Record one controller tick. Returns ``(hw, L_com)`` for the caller.

        The single-support counterpart of `_log_ds_tick`. Reads nothing but
        `ts`, `log`, `self` and the MuJoCo/Pinocchio state; writes nothing but
        `log`. Two calls here do mutate — `mj_forward` refreshes derived MuJoCo
        fields and `robot.update` refreshes the Pinocchio cache — so the block
        is kept verbatim and in order; both recompute derived quantities from
        unchanged qpos/qvel.
        """
        cfg = self.cfg
        t, phase, step_idx = ts.t, ts.phase, ts.step_idx
        swing_arm, ss_end, settle_mode = ts.swing_arm, ts.ss_end, ts.settle_mode
        hw = ts.hw

        mujoco.mj_forward(self.mj_model, self.mj_data)
        rs_f = self.robot.update(
            *mujoco_to_pinocchio(self.mj_data.qpos, self.mj_data.qvel))
        # Recompute torso reference at the actual logged time (after QP steps)
        t_log = t + cfg.nmpc_period
        tref_log = self.torso_planner.reference_at(t_log)
        # For the torso-position error metric use the reference the QP
        # actually tracked at the LAST sub-step — i.e. the output of
        # the M5 mapping layer (`ts.p_torso_ref_used`), not the geometric
        # TorsoPlanner quintic (`tref_log.p`). These diverge whenever
        # the mapping is wired, and the old metric was measuring the
        # mapping-vs-planner discrepancy, not the controller's tracking
        # quality. Orientation still comes from the TorsoPlanner SLERP
        # since the mapping only maps position.
        if ts.p_torso_ref_used is not None:
            p_torso_ref_log = ts.p_torso_ref_used.copy()
        else:
            # Sub-loop didn't run (shouldn't happen in production).
            p_torso_ref_log = tref_log.p.copy()
        # DS settle ticks with NO planner phase covering t_log (trailing
        # settle past the last quintic, initial DS before the first): the
        # torso-position task is settle-gated off and ts.p_torso_ref_used
        # carries the live CoM-mapping output, which jumps at the SS->DS
        # switch and ramps during settle. Log the clamped planner pose
        # instead (terminal quintic pose p_t1 / initial hold) so the
        # exported reference is continuous across the whole traversal.
        # When a phase DOES cover t_log, ts.p_torso_ref_used is kept — the QP
        # genuinely tracks that moving centroidal reference. NB the guard
        # cannot key on ds_centroidal_active: the locked config runs
        # centroidal DS in the trailing settle too (dca sets
        # ds_centroidal_mode).
        if (phase == 'DS' and settle_mode
                and not self.torso_planner.has_phase_at(t_log)):
            p_torso_ref_log = self.torso_planner.reference_at_clamped(
                t_log).p.copy()
        d_swing = self._gripper_distance(swing_arm, ts.target_anchor)
        d_stance = self._gripper_distance(
            ts.stance_arm, ts.stance_a if ts.stance_arm == 'a' else ts.stance_b)
        L_dot_est = (rs_f.L_com - ts.L_com_prev) / cfg.nmpc_period
        sq = self.mj_data.qpos[3:7].copy()
        euler = quat_wxyz_to_euler_deg(sq[0], sq[1], sq[2], sq[3])

        log.t.append(t)
        log.phase.append(phase)
        log.step_idx.append(step_idx)
        log.p_torso.append(rs_f.oMf_torso.translation.copy())
        log.p_torso_ref.append(p_torso_ref_log.copy())
        log.e_torso_pos.append(float(np.linalg.norm(
            rs_f.oMf_torso.translation - p_torso_ref_log)))
        R_err = tref_log.R.T @ rs_f.oMf_torso.rotation
        angle_err = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        log.e_torso_ori.append(float(np.degrees(angle_err)))
        log.d_grip_swing.append(d_swing)
        log.d_grip_stance.append(d_stance)
        log.swing_arm.append(swing_arm)

        # EE tracking error (vs planned trajectory reference, not just target).
        # Plan-time (offset-corrected) for the swing planner lookup.
        import pinocchio as pin
        sr_log = self.swing_planner.reference_at(
            self._swing_query_time(t, phase, ss_end))
        _, _, oMf_ee_log = self._get_ee_data(rs_f, swing_arm)
        log.e_ee_pos.append(float(np.linalg.norm(oMf_ee_log.translation - sr_log.p_ee)))
        e_ori_ee = pin.log3(oMf_ee_log.rotation.T @ sr_log.R_ee)
        log.e_ee_ori.append(float(np.degrees(np.linalg.norm(e_ori_ee))))

        # GMO diagnostics
        sw_slice = (self.robot.arm_b_v_slice if swing_arm == 'b'
                    else self.robot.arm_a_v_slice)
        log.gmo_residual_norm.append(float(np.linalg.norm(self.gmo.residual)))
        log.gmo_swing_residual.append(self.gmo.swing_residual_norm(sw_slice))
        log.gmo_contact_state.append(self.contact_sm.state.value)
        log.r_com.append(rs_f.r_com.copy())
        # Log the reference actually fed to the NMPC (pre-planner if
        # active, else TorsoPlanner-derived cref) so diagnostics compare
        # against what the controller was actually tracking.
        log.r_com_ref.append(np.asarray(ts.r_com_ref, dtype=float).copy())
        log.e_com.append(float(np.linalg.norm(rs_f.r_com - np.asarray(ts.r_com_ref))))
        log.L_com.append(rs_f.L_com.copy())
        log.L_com_norm.append(float(np.linalg.norm(rs_f.L_com)))
        log.L_dot.append(L_dot_est.copy())
        log.L_dot_norm.append(float(np.linalg.norm(L_dot_est)))
        log.hw.append(hw.copy())
        if self.has_rwa:
            rw_vel_f = self.mj_data.qvel[6:9].copy()
            log.hw_physical.append((cfg.rwa_I_w * rw_vel_f).copy())
            log.tau_w.append(ts.tau_wheels.copy())
            log.rw_speed.append(rw_vel_f.copy())
            log.transport_term_mag.append(ts.transport_term_mag)
        else:
            log.hw_physical.append(hw.copy())
            log.tau_w.append(np.zeros(3))
            log.rw_speed.append(np.zeros(3))
            log.transport_term_mag.append(0.0)

        # H_{r/O} estimator diagnostics — the estimator branch was removed in
        # CLEANUP-13 (aocs_use_H_estimator is False on the canonical, so only
        # the zero-fill path ever ran). The channels stay so the log schema is
        # unchanged.
        log.H_rO.append(np.zeros(3))
        log.H_dot_est.append(np.zeros(3))
        log.omega_struct.append(np.zeros(3))
        log.qfrc_constraint_torque.append(np.zeros(3))
        log.tau.append(ts.tau_joints.copy())
        log.tau_max_joint.append(float(np.max(np.abs(ts.tau_joints))))
        log.struct_pos.append(self.mj_data.qpos[0:3].copy())
        log.struct_quat.append(sq)
        log.struct_euler_deg.append(euler)
        log.nmpc_ok.append(ts.nmpc_ok)
        log.qp_ok.append(ts.qp_ok)
        log.lambda_ref_norm.append(float(np.linalg.norm(ts.lambda_ref)))
        log.nmpc_time_ms.append(ts.t_nmpc_ms)
        log.qp_time_ms.append(ts.t_qp_ms)

        # ── M0.2 enrichment ────────────────────────────────────────────
        # Torso orientation (actual vs ref) as quaternion wxyz
        R_torso = rs_f.oMf_torso.rotation
        q_t_actual = pin.Quaternion(R_torso)  # xyzw
        log.q_torso.append(np.array([q_t_actual.w, q_t_actual.x,
                                      q_t_actual.y, q_t_actual.z]))
        R_ref = tref_log.R
        q_t_ref = pin.Quaternion(R_ref)
        log.q_torso_ref.append(np.array([q_t_ref.w, q_t_ref.x,
                                          q_t_ref.y, q_t_ref.z]))

        # EE position/orientation (actual and ref)
        _, _, oMf_ee_f = self._get_ee_data(rs_f, swing_arm)
        log.p_ee.append(oMf_ee_f.translation.copy())
        q_ee_actual = pin.Quaternion(oMf_ee_f.rotation)
        log.q_ee.append(np.array([q_ee_actual.w, q_ee_actual.x,
                                   q_ee_actual.y, q_ee_actual.z]))
        sr_f = self.swing_planner.reference_at(
            self._swing_query_time(t_log, phase, ss_end))
        log.p_ee_ref.append(sr_f.p_ee.copy())
        q_ee_r = pin.Quaternion(sr_f.R_ee)
        log.q_ee_ref.append(np.array([q_ee_r.w, q_ee_r.x,
                                       q_ee_r.y, q_ee_r.z]))

        # Joint, EE, and torso velocities (T15-post-2 instrumentation).
        # Joint rates: Pinocchio-ordered v[arm_{a,b}_v_slice]; same convention
        # as q_joints slicing. EE/torso velocities: J @ v in LOCAL_WORLD_ALIGNED
        # (matches the frame of oMf_*.translation = p_ee / p_torso).
        log.qvel_joints_a.append(
            rs_f.v[self.robot.arm_a_v_slice].copy())
        log.qvel_joints_b.append(
            rs_f.v[self.robot.arm_b_v_slice].copy())
        V_tool_a = rs_f.J_tool_a @ rs_f.v
        V_tool_b = rs_f.J_tool_b @ rs_f.v
        V_torso_body = rs_f.J_torso @ rs_f.v
        log.v_ee_a.append(V_tool_a[:3].copy())
        log.omega_ee_a.append(V_tool_a[3:].copy())
        log.v_ee_b.append(V_tool_b[:3].copy())
        log.omega_ee_b.append(V_tool_b[3:].copy())
        log.v_torso.append(V_torso_body[:3].copy())
        log.omega_torso.append(V_torso_body[3:].copy())

        # CoM velocity (actual from Pinocchio, ref from NMPC)
        log.v_com.append(rs_f.v_com.copy())
        log.v_com_ref.append(ts.v_com_ref.copy())

        # L_com reference (NMPC-planned)
        log.L_com_ref.append(rs_f.L_com.copy())  # best available: actual (no L plan)

        # Platform angular velocity
        log.omega_s.append(self.mj_data.qvel[3:6].copy())

        # NMPC solver diagnostics
        log.nmpc_status.append(ts.nmpc_status_code)
        log.nmpc_cost.append(ts.nmpc_cost_val)
        log.nmpc_status_str.append(
            ts.nmpc_info.status if ts.nmpc_info is not None else "no_info")
        log.nmpc_iterations.append(
            int(ts.nmpc_info.solver_stats.get('iter_count', -1))
            if ts.nmpc_info is not None and ts.nmpc_info.solver_stats else -1)

        # Contact wrenches
        log.lambda_ref.append(ts.lambda_ref.copy())
        _lqp = (ts.lambda_qp.copy() if hasattr(ts.lambda_qp, 'copy')
                else np.zeros(12))
        log.lambda_qp.append(_lqp)
        log.lambda_qp_norm.append(float(np.linalg.norm(_lqp)))

        # Kinetic energy: 0.5 * v^T H v (relative to structure)
        v_rel = rs_f.v[:rs_f.H.shape[0]]
        T_kin = 0.5 * float(v_rel @ rs_f.H @ v_rel)
        log.T_kinetic.append(T_kin)

        return hw, rs_f.L_com.copy()
