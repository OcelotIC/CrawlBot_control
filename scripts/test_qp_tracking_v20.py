#!/usr/bin/env python3
"""M7 v20 standalone QP tracking — trapezoidal + planned-δ.

Same geometry as test_qp_tracking_v19.py:
  - 800 mm / 45° / 7.3 s EE swing (septic endpoints).
  - Mapping-based moving torso reference using planned q (base + stance held,
    arm_b interpolated from q0 to IK-solved q_end).

v20 vs v19 difference: the arm interp time-scaling is now the same 20/60/20
half-cosine trapezoid used by TorsoPlanner._trapezoidal_params — ramp=0.20.
This shapes the mapped torso reference to have a cruise phase (τ∈[0.2, 0.8])
where a_torso_ff ≈ 0 and v_torso_ref ≈ constant. The QP should see its peak
joint-torque demand concentrated in the ramp windows, leaving the cruise
budget free for EE tracking.

The EE task-consistent FF (v17) is already live inside wholebody_qp.py and
auto-applies when the torso task is active.
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


def trap20(t: float, T: float):
    """20/60/20 half-cosine-ramp trapezoid. Mirror of TorsoPlanner
    _trapezoidal_params with ramp = 0.2.
    Returns (s, ds, dds) with ds = d/dt, dds = d2/dt2.
    """
    ramp = 0.20
    tau = float(np.clip(t / T, 0.0, 1.0))
    v_c = 1.0 / (1.0 - ramp)
    pi = np.pi
    if tau < ramp:
        phi = pi * tau / ramp
        s = v_c * (tau / 2.0 - ramp / (2.0 * pi) * np.sin(phi))
        ds = v_c * 0.5 * (1.0 - np.cos(phi)) / T
        dds = v_c * pi / (2.0 * ramp) * np.sin(phi) / (T * T)
    elif tau < 1.0 - ramp:
        s_ramp = v_c * ramp / 2.0
        s = s_ramp + v_c * (tau - ramp)
        ds = v_c / T
        dds = 0.0
    else:
        tau_d = 1.0 - tau
        phi = pi * tau_d / ramp
        s_tail = v_c * (tau_d / 2.0 - ramp / (2.0 * pi) * np.sin(phi))
        s = 1.0 - s_tail
        ds = v_c * 0.5 * (1.0 - np.cos(phi)) / T
        dds = -v_c * pi / (2.0 * ramp) * np.sin(phi) / (T * T)
    return s, ds, dds


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
    return (p_ref, R_ref,
            np.concatenate([v_lin, omega]),
            np.concatenate([a_lin, alpha]))


def _quat_wxyz(R):
    q = pin.Quaternion(R)
    return np.array([q.w, q.x, q.y, q.z])


def run():
    cfg = SimConfig(
        use_m2_stack=True, alpha_com_soft=5.0, alpha_passivity=1.0,
        enforce_hw_conservation=True, h_max_tight=np.full(3, 5.0),
        w_L_nmpc=1.0, kappa_terminal=1.0,
        aocs_mode='legacy_corrected',
        aocs_use_legacy_corrected=True, aocs_use_H_estimator=False,
        preplanner_M=15, preplanner_kappa=0.7,
        preplanner_f_max=25.0, preplanner_tau_max=8.0,
        preplanner_w_L=1.0, preplanner_w_u=1e-2)

    sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=cfg)
    sim.setup(n_steps=1, start_a=2, start_b=2)
    sim.qp_ss.config.alpha_wrench = 0.01
    sim.qp_ss.config.alpha_com_soft = 0.0

    mj_model = sim.mj_model
    mj_data = sim.mj_data
    robot = sim.robot
    qp = sim.qp_ss
    n_j = robot.n_joints
    has_rwa = sim.has_rwa

    swing_arm = 'b'
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

    p_ee_target = p_ee_0 + dp
    R_ee_target = pin.exp3(axis * dtheta) @ R_ee_0
    fid_b = robot.frame_tool_b
    targets = {fid_b: pin.SE3(R_ee_target, p_ee_target)}
    q_end, ik_err = solve_ik(
        robot.model, pq0.copy(), targets, max_iter=3000, base_gain=0.0)
    print(f"[IK] q_end err={ik_err:.3e}")

    mapping = CoMToTorsoMapping(robot)
    sl_q_b = robot.arm_b_q_slice
    sl_v_b = robot.arm_b_v_slice
    dq_arm_b = q_end[sl_q_b] - pq0[sl_q_b]

    def planned_q_dq_trap(t):
        """v20: arm interp uses trapezoidal 20/60/20 scaling so the
        mapped torso reference inherits a cruise phase."""
        s, sd, _ = trap20(t, T_traj)
        q_plan = pq0.copy()
        q_plan[sl_q_b] = pq0[sl_q_b] + s * dq_arm_b
        dq_plan = np.zeros(robot.model.nv)
        dq_plan[sl_v_b] = sd * dq_arm_b
        return q_plan, dq_plan

    print(f"[v20 test] trap ramp=0.20 (20/60/20 split). T_traj={T_traj}s")

    log = SimLog()
    hw = np.zeros(3)
    sw_slice = robot.arm_b_v_slice
    fail_count = 0
    tau_per_tick = []  # (t, |tau|_inf, phase_label)

    for k in range(n_ticks):
        t = k * dt
        pq, pv = mujoco_to_pinocchio(mj_data.qpos, mj_data.qvel)
        rs = robot.update(pq, pv)

        p_ee_ref, R_ee_ref, v_ee_ref, a_ee_ff = ee_reference(
            t, T_traj, p_ee_0, R_ee_0, dp, dtheta, axis)

        q_plan, dq_plan = planned_q_dq_trap(t)
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
            p_torso=rs.oMf_torso.translation, R_torso=rs.oMf_torso.rotation,
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

        # phase label for torque breakdown
        tau_val = float(np.max(np.abs(tau)))
        if t <= 0.20 * T_traj:
            ph_lbl = 'ramp_up'
        elif t <= 0.80 * T_traj:
            ph_lbl = 'cruise'
        elif t <= T_traj:
            ph_lbl = 'ramp_dn'
        else:
            ph_lbl = 'post'
        tau_per_tick.append((t, tau_val, ph_lbl))

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
            print(f"  t={t:6.3f}s ({ph_lbl:7s}) "
                  f"torso={e_pos_torso*1000:6.2f}mm "
                  f"ee={e_pos_ee*1000:7.2f}mm  "
                  f"|tau|={tau_val:.3f}Nm")

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

    out_dir = os.path.join('results', 'qp_tracking_test_v20')
    os.makedirs(out_dir, exist_ok=True)
    _fig10_torso_6d_tracking(
        t_arr.tolist(), log.phase, [], log, out_dir, dpi=120, cfg=cfg)
    _fig9_ee_6d_tracking(
        t_arr.tolist(), log.phase, [], log, out_dir, dpi=120, cfg=cfg)

    # torque breakdown by sub-phase
    arr = np.array(
        [(t_, tv, lbl) for (t_, tv, lbl) in tau_per_tick],
        dtype=[('t', float), ('tau', float), ('lbl', 'U10')])
    def stats(mask):
        if mask.sum() == 0:
            return (0.0, 0.0, 0.0)
        vals = arr['tau'][mask]
        return (float(vals.max()), float(vals.mean()), int(mask.sum()))
    m_ru = (arr['lbl'] == 'ramp_up')
    m_cr = (arr['lbl'] == 'cruise')
    m_rd = (arr['lbl'] == 'ramp_dn')
    ru = stats(m_ru); cr = stats(m_cr); rd = stats(m_rd)

    lines = [
        '======================================================================',
        '  QP tracking test v20 — trapezoidal 20/60/20 + planned-delta',
        '======================================================================',
        f'  T_traj: {T_traj:.2f} s   dt_qp: {dt*1000:.1f} ms   '
        f'QP exceptions: {fail_count}',
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
    lines.append('')
    lines.append('  |tau|_inf per sub-phase:')
    lines.append(f"    ramp_up   peak={ru[0]:6.3f} Nm  mean={ru[1]:6.3f} Nm  n={ru[2]}")
    lines.append(f"    cruise    peak={cr[0]:6.3f} Nm  mean={cr[1]:6.3f} Nm  n={cr[2]}")
    lines.append(f"    ramp_dn   peak={rd[0]:6.3f} Nm  mean={rd[1]:6.3f} Nm  n={rd[2]}")
    lines.append('')
    lines.append('  VERDICT: ' + (
        'OK' if all_pass else 'peaks above threshold — see breakdown'))
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
