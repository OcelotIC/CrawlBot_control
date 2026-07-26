#!/usr/bin/env python3
"""M7 standalone QP tracking — v19 configuration.

Replicates test_qp_tracking.py but drives the torso task with a
MOVING reference computed exactly like v19 does in sim_loop:

  1. r_com_ref        = initial r_com (no net drift; synthetic "NMPC").
  2. v_com_ref, a_ff  = 0.
  3. q_planned(t)     = quintic interp (arm_b only) from q0 to q_end,
                        where q_end is IK for the final EE pose.
  4. dq_planned(t)    = d/dt of that interp.
  5. mapping.compute(r_com_ref, v_com_ref, 0, q_planned, dq_planned)
                      -> (r_b_ref, v_b_ref_lin, a_b_ff_lin, delta)
  6. Torso ori/ang references held at initial.

The EE task-consistent feedforward (v17) is already inside
wholebody_qp.py and auto-applies whenever the torso task is active.

Same 800 mm / 45° / 7.3 s swing. Pass/fail thresholds unchanged.
Output: Misc/runs/qp_tracking_test_v19/.
"""
from __future__ import annotations

import os
import sys
import json

_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import numpy as np
import mujoco
import pinocchio as pin

from crawlbot.simulation.sim_loop import SimulationLoop
from crawlbot.simulation.config import SimConfig
from crawlbot.simulation.logging import SimLog
from crawlbot.core.state_conversions import mujoco_to_pinocchio
from crawlbot.core.ik import solve_ik
from crawlbot.core.com_to_torso_mapping import CoMToTorsoMapping
from crawlbot.solvers.contact_phase import ContactConfig, ContactPhase
from crawlbot.diagnostics.plots import (
    _fig9_ee_6d_tracking, _fig10_torso_6d_tracking)


URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')
MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')


def septic(tau):
    if tau <= 0.0: return 0.0, 0.0, 0.0
    if tau >= 1.0: return 1.0, 0.0, 0.0
    s = 35*tau**4 - 84*tau**5 + 70*tau**6 - 20*tau**7
    sd = 140*tau**3 - 420*tau**4 + 420*tau**5 - 140*tau**6
    sdd = 420*tau**2 - 1680*tau**3 + 2100*tau**4 - 840*tau**5
    return s, sd, sdd


def ee_reference(t, T, p_ee_0, R_ee_0, dp, dtheta, axis):
    tau = np.clip(t / T, 0.0, 1.0)
    s, sd_tau, sdd_tau = septic(tau)
    p_ref = p_ee_0 + dp * s
    v_lin = dp * (sd_tau / T)
    a_lin = dp * (sdd_tau / (T * T))
    theta_dot = dtheta * (sd_tau / T)
    theta_ddot = dtheta * (sdd_tau / (T * T))
    rv = axis * (dtheta * s)
    R_ref = pin.exp3(rv) @ R_ee_0
    omega = axis * theta_dot
    alpha = axis * theta_ddot
    v_ref_6d = np.concatenate([v_lin, omega])
    a_ff_6d = np.concatenate([a_lin, alpha])
    return p_ref, R_ref, v_ref_6d, a_ff_6d


def _quat_wxyz(R):
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

    sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=cfg)
    sim.setup(n_steps=1, start_a=2, start_b=2)

    _aw_old = sim.qp_ss.config.alpha_wrench
    _acs_old = sim.qp_ss.config.alpha_com_soft
    sim.qp_ss.config.alpha_wrench = 0.01
    sim.qp_ss.config.alpha_com_soft = 0.0
    print(f"[QP override] alpha_wrench: {_aw_old} -> {sim.qp_ss.config.alpha_wrench}")
    print(f"[QP override] alpha_com_soft: {_acs_old} -> {sim.qp_ss.config.alpha_com_soft}")

    mj_model = sim.mj_model
    mj_data = sim.mj_data
    robot = sim.robot
    qp = sim.qp_ss
    n_j = robot.n_joints
    has_rwa = sim.has_rwa

    swing_arm = 'b'
    stance_arm = 'a'
    swing_anchor_idx = 2
    stance_anchor_idx = 2

    sim._deactivate_weld(swing_arm, swing_anchor_idx)
    mujoco.mj_forward(mj_model, mj_data)

    r_contact_a = sim.sched.anchors_a[stance_anchor_idx].copy()
    r_contact_b = sim.sched.anchors_b[swing_anchor_idx].copy()
    cc_ss = ContactConfig.from_phase(
        ContactPhase.SINGLE_A, r_contact_a, r_contact_b)

    pq0, pv0 = mujoco_to_pinocchio(mj_data.qpos, mj_data.qvel)
    rs0 = robot.update(pq0, pv0)
    p_torso_0 = rs0.oMf_torso.translation.copy()
    R_torso_0 = rs0.oMf_torso.rotation.copy()
    r_com_0 = rs0.r_com.copy()

    _, _, oMf_ee0 = sim._get_ee_data(rs0, swing_arm)
    p_ee_0 = oMf_ee0.translation.copy()
    R_ee_0 = oMf_ee0.rotation.copy()

    dp = np.array([0.80, 0.0, 0.0])
    dtheta = np.deg2rad(45.0)
    axis = np.array([0.0, 0.0, 1.0])
    T_traj = 7.3
    T_total = T_traj + 2.0
    dt = cfg.dt_qp
    n_ticks = int(round(T_total / dt))

    # ── v19: compute q_end via IK on swing-arm tool target ────────────
    p_ee_target = p_ee_0 + dp
    R_ee_target = pin.exp3(axis * dtheta) @ R_ee_0
    fid_b = robot.frame_tool_b
    targets = {fid_b: pin.SE3(R_ee_target, p_ee_target)}
    q_end, ik_err = solve_ik(
        robot.model, pq0.copy(), targets, max_iter=3000, base_gain=0.0)
    print(f"[IK] q_end error = {ik_err:.3e}  "
          f"(arm_b only, base/stance held)")

    # Sanity: the IK should have modified only arm_b joints.
    dq_q = q_end - pq0
    sl_q_b = robot.arm_b_q_slice
    sl_v_b = robot.arm_b_v_slice
    print(f"  ||dq_arm_b||={np.linalg.norm(dq_q[sl_q_b]):.3f} rad  "
          f"||dq_base||={np.linalg.norm(dq_q[:7]):.3e}  "
          f"||dq_arm_a||={np.linalg.norm(dq_q[robot.arm_a_q_slice]):.3e}")

    # ── v19: mapping, fed q_planned ──────────────────────────────────
    mapping = CoMToTorsoMapping(robot)
    dq_arm_b = q_end[sl_q_b] - pq0[sl_q_b]

    def planned_q_dq(t):
        tau = float(np.clip(t / T_traj, 0.0, 1.0))
        s = 10.0*tau**3 - 15.0*tau**4 + 6.0*tau**5
        sd = (30.0*tau**2 - 60.0*tau**3 + 30.0*tau**4) / T_traj
        q_plan = pq0.copy()
        q_plan[sl_q_b] = pq0[sl_q_b] + s * dq_arm_b
        dq_plan = np.zeros(robot.model.nv)
        dq_plan[sl_v_b] = sd * dq_arm_b
        return q_plan, dq_plan

    print(f"[test_qp_tracking_v19] T_traj={T_traj}s  T_total={T_total}s  "
          f"dt={dt*1000:.1f}ms  n_ticks={n_ticks}")
    print(f"  dp = {dp} m  (||dp||={np.linalg.norm(dp)*1000:.1f} mm)  "
          f"dtheta = {np.degrees(dtheta):.1f} deg")

    log = SimLog()
    hw = np.zeros(3)
    sw_slice = robot.arm_b_v_slice if swing_arm == 'b' else robot.arm_a_v_slice

    fail_count = 0
    for k in range(n_ticks):
        t = k * dt
        pq, pv = mujoco_to_pinocchio(mj_data.qpos, mj_data.qvel)
        rs = robot.update(pq, pv)

        # EE reference
        p_ee_ref, R_ee_ref, v_ee_ref, a_ee_ff = ee_reference(
            t, T_traj, p_ee_0, R_ee_0, dp, dtheta, axis)

        # Moving torso reference via mapping on q_planned (v19 exact)
        q_plan, dq_plan = planned_q_dq(t)
        r_b_ref, v_b_ref_lin, a_b_ff_lin, _ = mapping.compute(
            r_com_ref=r_com_0, v_com_ref=np.zeros(3),
            a_com_ff=np.zeros(3),
            q_current=q_plan, dq_current=dq_plan)

        p_torso_ref = r_b_ref
        R_torso_ref = R_torso_0
        v_torso_ref = np.concatenate([v_b_ref_lin, np.zeros(3)])
        a_torso_ff = np.concatenate([a_b_ff_lin, np.zeros(3)])

        J_ee, Jdq_ee, oMf_ee = sim._get_ee_data(rs, swing_arm)
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
        H_bs = rs.H[:6, sw_slice]

        try:
            qdd_t, qdd, lam_sol, tau, _info = qp.solve(
                q_t=rs.q_torso, dq_t=rs.dq_torso,
                q=rs.q_joints, dq=rs.dq_joints,
                r_com_ref=rs.r_com, v_com_ref=np.zeros(3),
                lambda_ref=np.zeros(12), a_com_ff=np.zeros(3),
                H_robot=rs.H, C_robot=rs.C,
                J_com=rs.J_com, Jdot_dq_com=rs.Jdot_dq_com,
                contact_config=cc_ss, J_contacts=Jc,
                Jdot_dq_contacts=Jdc,
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

        tau = np.clip(tau, -cfg.tau_max, cfg.tau_max)
        mj_data.ctrl[:n_j] = tau
        if has_rwa:
            mj_data.ctrl[n_j:n_j + 3] = 0.0
        mujoco.mj_step(mj_model, mj_data)

        e_pos_torso = float(np.linalg.norm(
            rs.oMf_torso.translation - p_torso_ref))
        R_err_t = rs.oMf_torso.rotation.T @ R_torso_ref
        ang_torso = np.arccos(np.clip((np.trace(R_err_t) - 1) / 2, -1, 1))
        e_ori_torso_deg = float(np.degrees(ang_torso))
        e_pos_ee = float(np.linalg.norm(oMf_ee.translation - p_ee_ref))
        e_ori_ee_vec = pin.log3(oMf_ee.rotation.T @ R_ee_ref)
        e_ori_ee_deg = float(np.degrees(np.linalg.norm(e_ori_ee_vec)))

        log.t.append(float(t))
        log.phase.append('SS')
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
        log.swing_arm.append(swing_arm)
        log.d_grip_swing.append(0.0)
        log.d_grip_stance.append(0.0)
        log.tau.append(np.asarray(tau, dtype=float).copy())
        log.qp_ok.append(bool(qp_ok))

        if (k % 100) == 0:
            print(f"  t={t:6.3f}s  torso_pos={e_pos_torso*1000:6.2f}mm  "
                  f"torso_ori={e_ori_torso_deg:6.3f}deg  "
                  f"ee_pos={e_pos_ee*1000:7.2f}mm  "
                  f"ee_ori={e_ori_ee_deg:6.3f}deg  "
                  f"|tau|_inf={np.max(np.abs(tau)):.3f}Nm")

    t_arr = np.array(log.t)
    in_traj = t_arr <= T_traj
    e_tp = np.array(log.e_torso_pos)[in_traj] * 1000
    e_to = np.array(log.e_torso_ori)[in_traj]
    e_ep = np.array(log.e_ee_pos)[in_traj] * 1000
    e_eo = np.array(log.e_ee_ori)[in_traj]

    thr = {'torso_pos_mm': 5.0, 'torso_ori_deg': 2.0,
           'ee_pos_mm': 10.0, 'ee_ori_deg': 5.0}
    checks = [
        ('torso_pos_err_peak_mm',  float(e_tp.max()), thr['torso_pos_mm']),
        ('torso_ori_err_peak_deg', float(e_to.max()), thr['torso_ori_deg']),
        ('ee_pos_err_peak_mm',     float(e_ep.max()), thr['ee_pos_mm']),
        ('ee_ori_err_peak_deg',    float(e_eo.max()), thr['ee_ori_deg']),
    ]

    out_dir = os.path.join('Misc', 'runs', 'qp_tracking_test_v19')
    os.makedirs(out_dir, exist_ok=True)
    _fig10_torso_6d_tracking(
        t_arr.tolist(), log.phase, [], log, out_dir, dpi=120, cfg=cfg)
    _fig9_ee_6d_tracking(
        t_arr.tolist(), log.phase, [], log, out_dir, dpi=120, cfg=cfg)

    lines = [
        '======================================================================',
        '  QP tracking test v19 — standalone (v19 torso ref + EE FF)',
        '======================================================================',
        f'  T_traj: {T_traj:.2f} s   T_total: {T_total:.2f} s   '
        f'dt_qp: {dt*1000:.1f} ms',
        f'  swing: {swing_arm}   dp: {dp.tolist()} m   '
        f'dtheta: {np.degrees(dtheta):.2f} deg',
        f'  QP exceptions: {fail_count}',
        '',
        '  Peak errors over trajectory window (t <= T_traj):',
        f"  {'metric':<26}{'peak':>12}{'thr':>10}  status",
        '  ----------------------------------------------------------',
    ]
    all_pass = True
    for name, val, thv in checks:
        status = 'PASS' if val < thv else '**FAIL'
        if val >= thv:
            all_pass = False
        lines.append(f"  {name:<24}{val:>12.4f}{thv:>10.3f}  {status}")
    taus = np.stack([np.asarray(t_) for t_ in log.tau])
    taus_inf = np.max(np.abs(taus), axis=1)
    lines.append('')
    lines.append(f'  |tau|_inf full-run peak={taus_inf.max():.3f} Nm  '
                 f'mean={taus_inf.mean():.3f} Nm  (budget {cfg.tau_max:.1f})')
    lines.append('')
    lines.append('  VERDICT: ' + (
        'QP+v19 TORSO REF + EE FF OK' if all_pass else
        'QP+v19 CANNOT TRACK (see peaks)'))
    summary = '\n'.join(lines)
    print('\n' + summary)
    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        f.write(summary + '\n')
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
