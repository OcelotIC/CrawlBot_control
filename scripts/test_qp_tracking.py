#!/usr/bin/env python3
"""M7 standalone QP tracking validation.

Purpose
-------
Bypass NMPC / pre-planner / state machine / AOCS entirely and drive the
WholeBodyQP at 100 Hz with two synthetic references:

  1. Torso 6D : constant (hold initial pose).
  2. Swing EE : smooth jerk-limited septic profile — 200 mm translation
                 + 15° rotation over 8 s, terminal conditions zero in
                 pos/vel/accel/jerk.

The stance arm stays welded (SS-A); the swing arm ('b') is released at
t=0 and its EE follows the QP-commanded trajectory. No NMPC, no
TorsoPlanner, no pre-planner, no CoM mapping. CoM ref = current CoM,
wrench ref = 0, hw current = 0.

Pass criteria (throughout the 8 s trajectory window)
----------------------------------------------------
  torso  pos err  < 5 mm
  torso  ori err  < 2°
  swing  EE  pos err < 10 mm
  swing  EE  ori err <  5°

If the QP cannot meet these with a hand-crafted, hand-synchronised
reference while solving only two tasks, the QP itself is the problem.
If it does meet them, the closed-loop failure is in the
NMPC / planner / reference-generation cascade.

Outputs
-------
  results/qp_tracking_test/
    sim_log.json         — full SimLog (JSON)
    fig9_ee_6d_tracking.png
    fig10_torso_6d_tracking.png
    summary.txt          — pass/fail table
"""
from __future__ import annotations

import os
import sys
import json

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import numpy as np
import mujoco
import pinocchio as pin

from crawlbot.simulation.sim_loop import SimulationLoop
from crawlbot.simulation.config import SimConfig
from crawlbot.simulation.logging import SimLog
from crawlbot.core.state_conversions import mujoco_to_pinocchio
from crawlbot.solvers.contact_phase import ContactConfig, ContactPhase
from crawlbot.diagnostics.plots import (
    _fig9_ee_6d_tracking, _fig10_torso_6d_tracking)


URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')
MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')


# ──────────────────────────────────────────────────────────────────────
#  Septic scalar profile with zero pos/vel/accel/jerk at both endpoints
#  s(τ) = 35τ⁴ − 84τ⁵ + 70τ⁶ − 20τ⁷   for τ ∈ [0,1], s(0)=0, s(1)=1
# ──────────────────────────────────────────────────────────────────────
def septic(tau: float) -> tuple[float, float, float]:
    """Return (s, s_dot_tau, s_ddot_tau): scalar, d/dτ, d²/dτ²."""
    if tau <= 0.0:
        return 0.0, 0.0, 0.0
    if tau >= 1.0:
        return 1.0, 0.0, 0.0
    s = 35*tau**4 - 84*tau**5 + 70*tau**6 - 20*tau**7
    sd = 140*tau**3 - 420*tau**4 + 420*tau**5 - 140*tau**6
    sdd = 420*tau**2 - 1680*tau**3 + 2100*tau**4 - 840*tau**5
    return s, sd, sdd


def ee_reference(t: float, T: float,
                 p_ee_0: np.ndarray, R_ee_0: np.ndarray,
                 dp: np.ndarray, dtheta: float, axis: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct EE 6D reference at time t.

    R_ref(t) = exp3(s(τ)·dθ·n̂) · R_ee_0,  axis n̂ in world/structure frame.
    ω_ref_world = ṡ·dθ/T · n̂
    α_ff_world  = s̈·dθ/T² · n̂
    Returns (p_ref, R_ref, v_ref_6d, a_ff_6d).
    """
    tau = np.clip(t / T, 0.0, 1.0)
    s, sd_tau, sdd_tau = septic(tau)

    # Position
    p_ref = p_ee_0 + dp * s
    v_lin = dp * (sd_tau / T)
    a_lin = dp * (sdd_tau / (T * T))

    # Orientation: rotation about fixed axis in world frame
    theta = dtheta * s
    theta_dot = dtheta * (sd_tau / T)
    theta_ddot = dtheta * (sdd_tau / (T * T))
    rv = axis * theta
    R_ref = pin.exp3(rv) @ R_ee_0
    omega = axis * theta_dot
    alpha = axis * theta_ddot

    v_ref_6d = np.concatenate([v_lin, omega])
    a_ff_6d = np.concatenate([a_lin, alpha])
    return p_ref, R_ref, v_ref_6d, a_ff_6d


def _quat_wxyz(R: np.ndarray) -> np.ndarray:
    q = pin.Quaternion(R)
    return np.array([q.w, q.x, q.y, q.z])


def run():
    cfg = SimConfig(
        use_m2_stack=True,
        alpha_com_soft=5.0,
        alpha_passivity=1.0,
        enforce_hw_conservation=True,
        h_max_tight=np.full(3, 5.0),
        w_L_nmpc=1.0,
        kappa_terminal=1.0,
        aocs_mode='legacy_corrected',
        aocs_use_legacy_corrected=True,
        aocs_use_H_estimator=False,
        preplanner_M=15,
        preplanner_kappa=0.7,
        preplanner_f_max=25.0,
        preplanner_tau_max=8.0,
        preplanner_w_L=1.0,
        preplanner_w_u=1e-2,
    )

    # ── Instantiate SimulationLoop just to inherit setup() ────────────
    #   One step, start_a=start_b=2 (same as M7 runs). The plan itself is
    #   discarded — we only need setup() to activate welds, settle, and
    #   give us a solvable starting state + a ready qp_ss.
    sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=cfg)
    sim.setup(n_steps=1, start_a=2, start_b=2)

    # ── QP weight override (QP-isolation diagnosis) ───────────────────
    # α_wrench=100 fights the dynamics: with λ_ref=0 it penalises every
    # contact force, but contact forces are the only way to produce
    # torso acceleration through the stance weld. Drop it to pure
    # regularisation. α_com_soft redundant with torso position.
    _aw_old = sim.qp_ss.config.alpha_wrench
    _acs_old = sim.qp_ss.config.alpha_com_soft
    sim.qp_ss.config.alpha_wrench = 0.01
    sim.qp_ss.config.alpha_com_soft = 0.0
    print(f"[QP override] alpha_wrench: {_aw_old} → {sim.qp_ss.config.alpha_wrench}")
    print(f"[QP override] alpha_com_soft: {_acs_old} → {sim.qp_ss.config.alpha_com_soft}")

    # Robot references
    mj_model = sim.mj_model
    mj_data = sim.mj_data
    robot = sim.robot
    qp = sim.qp_ss
    n_j = robot.n_joints
    has_rwa = sim.has_rwa

    # ── Swing arm 'b', stance arm 'a' ─────────────────────────────────
    swing_arm = 'b'
    stance_arm = 'a'
    swing_anchor_idx = 2        # start_b
    stance_anchor_idx = 2       # start_a

    # Release swing weld (permanently for this test).
    sim._deactivate_weld(swing_arm, swing_anchor_idx)
    mujoco.mj_forward(mj_model, mj_data)

    # Build stance-only contact config (SS-A) — only stance anchor is active.
    r_contact_a = sim.sched.anchors_a[stance_anchor_idx].copy()
    r_contact_b = sim.sched.anchors_b[swing_anchor_idx].copy()   # unused in SS-A
    cc_ss = ContactConfig.from_phase(
        ContactPhase.SINGLE_A, r_contact_a, r_contact_b)

    # ── Capture torso reference (constant) at t=0 ─────────────────────
    pq0, pv0 = mujoco_to_pinocchio(mj_data.qpos, mj_data.qvel)
    rs0 = robot.update(pq0, pv0)
    p_torso_ref = rs0.oMf_torso.translation.copy()
    R_torso_ref = rs0.oMf_torso.rotation.copy()
    v_torso_ref = np.zeros(6)
    a_torso_ff = np.zeros(6)

    # ── Capture EE start pose and define trajectory ───────────────────
    J_ee0, _, oMf_ee0 = sim._get_ee_data(rs0, swing_arm)
    p_ee_0 = oMf_ee0.translation.copy()
    R_ee_0 = oMf_ee0.rotation.copy()
    # 800 mm translation along structure +x; 45° rotation about structure +z.
    # Matches the M7 1% single-step geometry: swing arm traverses
    # ~0.8 m between adjacent anchor pairs with ~45 deg orientation
    # change between anchor frames.
    dp = np.array([0.80, 0.0, 0.0])
    dtheta = np.deg2rad(45.0)
    axis = np.array([0.0, 0.0, 1.0])

    T_traj = 7.3                     # trajectory duration [s] — matches pre-planner T_step
    T_total = T_traj + 2.0            # run 2 s extra to observe post-end hold
    dt = cfg.dt_qp
    n_ticks = int(round(T_total / dt))

    print(f"[test_qp_tracking] starting run: T_traj={T_traj}s, T_total={T_total}s, "
          f"dt={dt*1000:.1f}ms, n_ticks={n_ticks}")
    print(f"  swing arm: {swing_arm}, stance arm: {stance_arm} (anchor {stance_anchor_idx})")
    print(f"  dp = {dp} [m]  (‖dp‖ = {np.linalg.norm(dp)*1000:.1f} mm)")
    print(f"  dtheta = {np.degrees(dtheta):.1f}°  axis = {axis}")
    print(f"  p_ee_0 = {p_ee_0}")
    print(f"  p_torso_ref = {p_torso_ref}")

    log = SimLog()
    hw = np.zeros(3)  # start with zero wheel momentum

    # Cache stance-arm slice for reaction null-space task
    sw_slice = robot.arm_b_v_slice if swing_arm == 'b' else robot.arm_a_v_slice

    # ── Main QP loop ─────────────────────────────────────────────────
    fail_count = 0
    torso_pre_post_log = []   # list of (t, ||a_des_pre||, ||J_t@qdd_post||) per tick
    for k in range(n_ticks):
        t = k * dt

        pq, pv = mujoco_to_pinocchio(mj_data.qpos, mj_data.qvel)
        rs = robot.update(pq, pv)

        # Swing EE reference
        p_ee_ref, R_ee_ref, v_ee_ref, a_ee_ff = ee_reference(
            t, T_traj, p_ee_0, R_ee_0, dp, dtheta, axis)

        # Current swing EE data
        J_ee, Jdq_ee, oMf_ee = sim._get_ee_data(rs, swing_arm)

        # Contact Jacobians for the active stance contact only
        Jc, Jdc = robot.get_contact_jacobians(
            cc_ss.active_contacts[0], cc_ss.active_contacts[1])

        tkw = dict(
            J_torso=rs.J_torso, Jdot_dq_torso=rs.Jdot_dq_torso,
            p_torso=rs.oMf_torso.translation,
            R_torso=rs.oMf_torso.rotation,
            p_torso_ref=p_torso_ref, R_torso_ref=R_torso_ref,
            v_torso_ref=v_torso_ref, a_torso_ff=a_torso_ff)
        ek = dict(
            J_ee=J_ee, Jdot_dq_ee=Jdq_ee,
            p_ee=oMf_ee.translation, R_ee=oMf_ee.rotation,
            p_ee_ref=p_ee_ref, R_ee_ref=R_ee_ref,
            v_ee_ref=v_ee_ref, a_ee_ff=a_ee_ff)

        # Reaction null-space coupling block
        H_bs = rs.H[:6, sw_slice]

        try:
            qdd_t, qdd, lam_sol, tau, _info = qp.solve(
                q_t=rs.q_torso, dq_t=rs.dq_torso,
                q=rs.q_joints, dq=rs.dq_joints,
                r_com_ref=rs.r_com, v_com_ref=np.zeros(3),
                lambda_ref=np.zeros(12), a_com_ff=np.zeros(3),
                H_robot=rs.H, C_robot=rs.C,
                J_com=rs.J_com, Jdot_dq_com=rs.Jdot_dq_com,
                contact_config=cc_ss, J_contacts=Jc, Jdot_dq_contacts=Jdc,
                hw_current=hw, hw_min=cfg.hw_min, hw_max=cfg.hw_max,
                r_com=rs.r_com, L_com_current=rs.L_com,
                H_base_swing=H_bs, swing_v_slice=sw_slice,
                settle_mode=False, passivity_active=False,
                **tkw, **ek)
            qp_ok = True
        except Exception as e:
            print(f"  [!! QP exception at t={t:.3f}s: {e}]")
            tau = np.zeros(n_j)
            qp_ok = False
            fail_count += 1

        # Capture pre- vs post-solve torso accel (added by QP instrumentation)
        tdbg = getattr(qp, 'last_torso_debug', None)
        if tdbg is not None:
            pre = np.asarray(tdbg['a_torso_des_pre'])
            # Reconstruct J_torso @ qdd_opt using rs.J_torso at solve time
            qdd_full = np.concatenate([qdd_t, qdd])
            post = rs.J_torso @ qdd_full
            torso_pre_post_log.append((
                float(t),
                float(np.linalg.norm(pre[:3])),   float(np.linalg.norm(post[:3])),
                float(np.linalg.norm(pre[3:])),   float(np.linalg.norm(post[3:])),
            ))

        # Apply torques; disable RWA ctrl
        tau = np.clip(tau, -cfg.tau_max, cfg.tau_max)
        mj_data.ctrl[:n_j] = tau
        if has_rwa:
            mj_data.ctrl[n_j:n_j + 3] = 0.0
        mujoco.mj_step(mj_model, mj_data)

        # ── Logging ──────────────────────────────────────────────────
        # Recompute errors from current state for consistency
        e_pos_torso = float(np.linalg.norm(
            rs.oMf_torso.translation - p_torso_ref))
        R_err_t = rs.oMf_torso.rotation.T @ R_torso_ref
        ang_torso = np.arccos(np.clip((np.trace(R_err_t) - 1) / 2, -1, 1))
        e_ori_torso_deg = float(np.degrees(ang_torso))

        e_pos_ee_vec = oMf_ee.translation - p_ee_ref
        e_pos_ee = float(np.linalg.norm(e_pos_ee_vec))
        e_ori_ee_vec = pin.log3(oMf_ee.rotation.T @ R_ee_ref)
        e_ori_ee_deg = float(np.degrees(np.linalg.norm(e_ori_ee_vec)))

        log.t.append(float(t))
        log.phase.append('SS')   # constant — one phase for plotting
        log.step_idx.append(0)

        log.p_torso.append(rs.oMf_torso.translation.copy())
        log.p_torso_ref.append(p_torso_ref.copy())
        log.q_torso.append(_quat_wxyz(rs.oMf_torso.rotation))
        log.q_torso_ref.append(_quat_wxyz(R_torso_ref))
        log.e_torso_pos.append(e_pos_torso)
        log.e_torso_ori.append(e_ori_torso_deg)

        log.p_ee.append(oMf_ee.translation.copy())
        log.p_ee_ref.append(p_ee_ref.copy())
        log.q_ee.append(_quat_wxyz(oMf_ee.rotation))
        log.q_ee_ref.append(_quat_wxyz(R_ee_ref))
        log.e_ee_pos.append(e_pos_ee)
        log.e_ee_ori.append(e_ori_ee_deg)

        # Fill the few fields referenced by the plot helpers
        log.swing_arm.append(swing_arm)
        log.d_grip_swing.append(0.0)
        log.d_grip_stance.append(0.0)
        log.tau.append(np.asarray(tau, dtype=float).copy())
        log.qp_ok.append(bool(qp_ok))

        if (k % 100) == 0:
            print(f"  t={t:6.3f}s  torso_pos={e_pos_torso*1000:6.2f}mm  "
                  f"torso_ori={e_ori_torso_deg:6.3f}°  "
                  f"ee_pos={e_pos_ee*1000:7.2f}mm  ee_ori={e_ori_ee_deg:6.3f}°  "
                  f"|tau|_inf={np.max(np.abs(tau)):.3f}Nm")

    # ── Summary / pass-fail over the trajectory window t ∈ [0, T_traj] ─
    t_arr = np.array(log.t)
    in_traj = t_arr <= T_traj
    e_tp = np.array(log.e_torso_pos)[in_traj] * 1000
    e_to = np.array(log.e_torso_ori)[in_traj]
    e_ep = np.array(log.e_ee_pos)[in_traj] * 1000
    e_eo = np.array(log.e_ee_ori)[in_traj]

    thr = {
        'torso_pos_mm': 5.0,
        'torso_ori_deg': 2.0,
        'ee_pos_mm': 10.0,
        'ee_ori_deg': 5.0,
    }

    checks = [
        ('torso_pos_err_peak_mm', float(e_tp.max()), thr['torso_pos_mm']),
        ('torso_ori_err_peak_deg', float(e_to.max()), thr['torso_ori_deg']),
        ('ee_pos_err_peak_mm', float(e_ep.max()), thr['ee_pos_mm']),
        ('ee_ori_err_peak_deg', float(e_eo.max()), thr['ee_ori_deg']),
    ]

    out_dir = os.path.join('results', 'qp_tracking_test')
    os.makedirs(out_dir, exist_ok=True)

    # ── Plots (reuse stock fig9 / fig10) ─────────────────────────────
    _fig10_torso_6d_tracking(
        t_arr.tolist(), log.phase, [], log, out_dir, dpi=120, cfg=cfg)
    _fig9_ee_6d_tracking(
        t_arr.tolist(), log.phase, [], log, out_dir, dpi=120, cfg=cfg)

    # ── Write summary + JSON log ─────────────────────────────────────
    summary_lines = [
        '======================================================================',
        '  QP tracking test — standalone validation',
        '======================================================================',
        f'  T_traj        : {T_traj:.2f} s',
        f'  T_total       : {T_total:.2f} s',
        f'  dt_qp         : {dt*1000:.1f} ms   ({n_ticks} ticks)',
        f'  swing arm     : {swing_arm}',
        f'  dp            : {dp.tolist()} m  (‖dp‖={np.linalg.norm(dp)*1000:.1f} mm)',
        f'  dtheta        : {np.degrees(dtheta):.2f}°  axis={axis.tolist()}',
        f'  QP exceptions : {fail_count}',
        '',
        '  Peak errors over trajectory window (t ≤ T_traj):',
        f"{'metric':<26}{'peak':>12}{'thr':>10}  status",
        '  ----------------------------------------------------------',
    ]
    all_pass = True
    for name, val, thv in checks:
        status = 'PASS' if val < thv else '**FAIL'
        if val >= thv:
            all_pass = False
        summary_lines.append(f"  {name:<24}{val:>12.4f}{thv:>10.3f}  {status}")

    # Pre- vs post-solve torso accel ratios (over trajectory window)
    if torso_pre_post_log:
        arr = np.array(torso_pre_post_log)   # (N, 5): t, pre_lin, post_lin, pre_ang, post_ang
        in_win = arr[:, 0] <= T_traj
        a = arr[in_win]
        # Use medians + peaks to summarise
        ratio_lin = a[:, 2] / np.maximum(a[:, 1], 1e-9)
        ratio_ang = a[:, 4] / np.maximum(a[:, 3], 1e-9)
        summary_lines.append('')
        summary_lines.append(
            '  Pre- vs post-solve torso accel (across 0..T_traj):')
        summary_lines.append(
            f"    ||a_pre_lin||  peak={a[:,1].max():.4f}  mean={a[:,1].mean():.4f} m/s²")
        summary_lines.append(
            f"    ||a_post_lin|| peak={a[:,2].max():.4f}  mean={a[:,2].mean():.4f} m/s²")
        summary_lines.append(
            f"    linear ratio   median={np.median(ratio_lin):.4f}  p90={np.percentile(ratio_lin,90):.4f}")
        summary_lines.append(
            f"    ||a_pre_ang||  peak={a[:,3].max():.4f}  mean={a[:,3].mean():.4f} rad/s²")
        summary_lines.append(
            f"    ||a_post_ang|| peak={a[:,4].max():.4f}  mean={a[:,4].mean():.4f} rad/s²")
        summary_lines.append(
            f"    angular ratio  median={np.median(ratio_ang):.4f}  p90={np.percentile(ratio_ang,90):.4f}")

    # Peak joint torque over the whole run
    taus = np.stack([np.asarray(t_) for t_ in log.tau])
    taus_inf = np.max(np.abs(taus), axis=1)
    summary_lines.append('')
    summary_lines.append(
        f'  |tau|_inf over full run:  peak={taus_inf.max():.3f} Nm  '
        f'mean={taus_inf.mean():.3f} Nm  (budget {cfg.tau_max:.1f} Nm)')

    summary_lines.append('')
    summary_lines.append(
        '  VERDICT: ' + ('QP-LEVEL TRACKING OK — problem is upstream'
                         if all_pass else
                         'QP CANNOT TRACK — problem is in the QP itself'))
    summary = '\n'.join(summary_lines)
    print()
    print(summary)
    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        f.write(summary + '\n')

    # Save log as JSON (pin'd arrays → lists)
    with open(os.path.join(out_dir, 'sim_log.json'), 'w') as f:
        d = {}
        for k_, v in log.__dict__.items():
            if isinstance(v, list) and v and isinstance(v[0], np.ndarray):
                d[k_] = [a.tolist() for a in v]
            else:
                d[k_] = v
        json.dump(d, f)

    print(f"\n  Outputs: {out_dir}/")


if __name__ == '__main__':
    run()
