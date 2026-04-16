#!/usr/bin/env python3
"""Bisection: standalone QP -> +NMPC+mapping -> +AOCS -> full sim_loop.

Setup matches scripts/test_qp_tracking.py exactly:
  - SimulationLoop.setup() with same SimConfig
  - swing arm 'b' weld released
  - septic 800 mm / 45° / 7.3 s EE reference
  - QP weight overrides: alpha_wrench=0.01, alpha_com_soft=0
  - SS-A contact config (only stance arm contributes to contact wrench)

Cases (selected by --case):
  A : standalone QP — torso ref CONSTANT at start pose (baseline).
  B : QP + NMPC + CoM->torso mapping — torso ref from mapping(NMPC out).
      NMPC regulates r_com to its INITIAL value; runs at 10 Hz with
      linear interpolation of r_com_plan / v_com_plan across QP sub-steps
      (matches sim_loop). a_com_ff from NMPC's planned wrench.
  C : adds AOCS — wheel torques via compute_aocs_command_legacy_corrected
      every QP tick.

EE peak position error is measured over t in [0, T_traj].

Run all cases:
  MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/bisect_qp_cascade.py --case A
  MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/bisect_qp_cascade.py --case B
  MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/bisect_qp_cascade.py --case C
"""
from __future__ import annotations

import os
import sys
import argparse

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import numpy as np
import mujoco
import pinocchio as pin

from crawlbot.simulation.sim_loop import SimulationLoop
from crawlbot.simulation.config import SimConfig
from crawlbot.core.state_conversions import mujoco_to_pinocchio
from crawlbot.core.ik import solve_ik
from crawlbot.solvers.contact_phase import ContactConfig, ContactPhase
from crawlbot.aocs.force_estimator import compute_aocs_command_legacy_corrected


URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')
MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')


def septic(tau: float):
    if tau <= 0.0:
        return 0.0, 0.0, 0.0
    if tau >= 1.0:
        return 1.0, 0.0, 0.0
    s = 35*tau**4 - 84*tau**5 + 70*tau**6 - 20*tau**7
    sd = 140*tau**3 - 420*tau**4 + 420*tau**5 - 140*tau**6
    sdd = 420*tau**2 - 1680*tau**3 + 2100*tau**4 - 840*tau**5
    return s, sd, sdd


def ee_reference(t, T, p_ee_0, R_ee_0, dp, dtheta, axis):
    tau = float(np.clip(t / T, 0.0, 1.0))
    s, sd_tau, sdd_tau = septic(tau)
    p_ref = p_ee_0 + dp * s
    v_lin = dp * (sd_tau / T)
    a_lin = dp * (sdd_tau / (T * T))
    theta = dtheta * s
    theta_dot = dtheta * (sd_tau / T)
    theta_ddot = dtheta * (sdd_tau / (T * T))
    rv = axis * theta
    R_ref = pin.exp3(rv) @ R_ee_0
    omega = axis * theta_dot
    alpha = axis * theta_ddot
    v6 = np.concatenate([v_lin, omega])
    a6 = np.concatenate([a_lin, alpha])
    return p_ref, R_ref, v6, a6


def run_case(case: str):
    """Run one bisection case ('A', 'B', or 'C')."""
    cfg = SimConfig(
        use_m2_stack=True,
        alpha_com_soft=5.0,             # SimConfig default for instantiation
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

    # Match standalone QP overrides used in test_qp_tracking.py.
    sim.qp_ss.config.alpha_wrench = 0.01
    sim.qp_ss.config.alpha_com_soft = 0.0

    mj_model = sim.mj_model
    mj_data = sim.mj_data
    robot = sim.robot
    qp = sim.qp_ss
    n_j = robot.n_joints
    has_rwa = sim.has_rwa
    use_nmpc = case in ('B', 'C', 'B_aware', 'B_aware_free', 'B_measured',
                        'B_const_torso', 'B_compensated')
    use_aocs = case in ('C',)
    arm_aware_ref = case in ('B_aware', 'B_aware_free')
    aware_base_gain = 1.0 if case == 'B_aware_free' else 0.0
    measured_ref = case in ('B_measured',)
    bypass_mapping = case in ('B_const_torso',)
    compensated_ref = case in ('B_compensated',)

    swing_arm = 'b'
    stance_arm = 'a'
    swing_anchor_idx = 2
    stance_anchor_idx = 2

    # Release swing weld.
    sim._deactivate_weld(swing_arm, swing_anchor_idx)
    mujoco.mj_forward(mj_model, mj_data)

    r_contact_a = sim.sched.anchors_a[stance_anchor_idx].copy()
    r_contact_b = sim.sched.anchors_b[swing_anchor_idx].copy()
    cc_ss = ContactConfig.from_phase(
        ContactPhase.SINGLE_A, r_contact_a, r_contact_b)

    # Initial state.
    pq0, pv0 = mujoco_to_pinocchio(mj_data.qpos, mj_data.qvel)
    rs0 = robot.update(pq0, pv0)
    p_torso_ref0 = rs0.oMf_torso.translation.copy()
    R_torso_ref0 = rs0.oMf_torso.rotation.copy()
    r_com_target = rs0.r_com.copy()           # NMPC regulates to this

    J_ee0, _, oMf_ee0 = sim._get_ee_data(rs0, swing_arm)
    p_ee_0 = oMf_ee0.translation.copy()
    R_ee_0 = oMf_ee0.rotation.copy()

    dp = np.array([0.80, 0.0, 0.0])
    dtheta = np.deg2rad(45.0)
    axis = np.array([0.0, 0.0, 1.0])

    T_traj = 7.3
    T_total = T_traj + 2.0
    dt = cfg.dt_qp
    n_qp_per_nmpc = int(round(cfg.dt_nmpc / cfg.dt_qp))
    n_ticks = int(round(T_total / dt))

    print(f"[case {case}] use_nmpc={use_nmpc}  use_aocs={use_aocs}  has_rwa={has_rwa}")
    print(f"           T_traj={T_traj}s  n_ticks={n_ticks}  n_qp_per_nmpc={n_qp_per_nmpc}")
    print(f"           p_ee_0={p_ee_0}  p_torso_ref0={p_torso_ref0}")
    print(f"           r_com_target={r_com_target}")

    # NMPC plan caches (linear interp between knots 0 and 1).
    rp_k0 = r_com_target.copy()
    rp_k1 = r_com_target.copy()
    vp_k0 = np.zeros(3)
    vp_k1 = np.zeros(3)
    a_com_ff_curr = np.zeros(3)
    nmpc_fail_count = 0

    # AOCS state.
    L_com_prev = rs0.L_com.copy()
    v_com_prev = rs0.v_com.copy()

    # Logged peak errors over [0, T_traj].
    e_ee_pos_peak = 0.0
    e_torso_pos_peak = 0.0
    e_ee_ori_peak = 0.0
    e_torso_ori_peak = 0.0
    qp_fail_count = 0

    # Mass for AOCS.
    robot_mass = robot._total_mass

    for k in range(n_ticks):
        t = k * dt

        pq, pv = mujoco_to_pinocchio(mj_data.qpos, mj_data.qvel)
        omega_s = mj_data.qvel[3:6].copy()
        rs = robot.update(pq, pv, omega_struct=omega_s)

        # ── NMPC every n_qp_per_nmpc ticks ────────────────────────────
        if use_nmpc and (k % n_qp_per_nmpc == 0):
            hw_for_nmpc = (cfg.rwa_I_w * mj_data.qvel[6:9]).copy() if has_rwa \
                          else np.zeros(3)

            # Arm-aware r_com_ref: predict CoM via FK at the swing-arm
            # config that places the EE on the swing reference at horizon
            # midpoint. Base frozen (base_gain=0); arm IK only.
            if arm_aware_ref:
                fid_swing = (robot.frame_tool_b if swing_arm == 'b'
                             else robot.frame_tool_a)
                t_mid = t + 0.5 * cfg.nmpc_N * cfg.nmpc_dt
                p_ee_mid, R_ee_mid, _, _ = ee_reference(
                    t_mid, T_traj, p_ee_0, R_ee_0, dp, dtheta, axis)
                tgt_mid = pin.SE3(R_ee_mid, p_ee_mid)
                q_pred, ik_err = solve_ik(
                    robot.model, rs.q.copy(), {fid_swing: tgt_mid},
                    max_iter=80, tol=1e-6, base_gain=aware_base_gain)
                d_pred = robot.model.createData()
                pin.centerOfMass(robot.model, d_pred, q_pred)
                r_com_ref_nmpc = d_pred.com[0].copy()
                # v_com_ref: finite-diff using a second IK at t_mid + dt_nmpc
                p_ee_n, R_ee_n, _, _ = ee_reference(
                    t_mid + cfg.dt_nmpc, T_traj, p_ee_0, R_ee_0,
                    dp, dtheta, axis)
                tgt_n = pin.SE3(R_ee_n, p_ee_n)
                q_pred_n, _ = solve_ik(
                    robot.model, q_pred.copy(), {fid_swing: tgt_n},
                    max_iter=40, tol=1e-6, base_gain=aware_base_gain)
                d_pred_n = robot.model.createData()
                pin.centerOfMass(robot.model, d_pred_n, q_pred_n)
                v_com_ref_nmpc = (d_pred_n.com[0] - r_com_ref_nmpc) / cfg.dt_nmpc
            elif measured_ref:
                # Track current measured CoM; do not regulate against drift.
                r_com_ref_nmpc = rs.r_com.copy()
                v_com_ref_nmpc = rs.v_com.copy()
            elif compensated_ref:
                # Cancel the mapping's delta(q) term. Want r_b_ref(t) =
                # p_torso_ref0 for all q, so set
                #   r_com_ref = (m_b * p_torso_ref0 + delta(q_current)) / m_total
                # which is the CoM that would result if torso stayed put
                # while the arm did its thing. The mapping then yields
                # r_b_ref = p_torso_ref0 exactly.
                delta_now = sim.mapping.compute_delta(rs.q)
                m_b = sim.mapping.m_b
                m_total = sim.mapping.m_total
                r_com_ref_nmpc = (m_b * p_torso_ref0 + delta_now) / m_total
                v_com_ref_nmpc = np.zeros(3)  # let NMPC infer drift
            else:
                r_com_ref_nmpc = r_com_target
                v_com_ref_nmpc = np.zeros(3)

            try:
                rp, vp, _, lr, info_n = sim.nmpc.solve(
                    r_com=rs.r_com, v_com=rs.v_com, L_com=rs.L_com,
                    r_com_ref=r_com_ref_nmpc, v_com_ref=v_com_ref_nmpc,
                    contact_config=cc_ss, warm_start=True,
                    hw_current=hw_for_nmpc, L_com_ref=np.zeros(3))
                a_com_ff_curr = sim.nmpc.compute_feedforward_acceleration(lr)
                if not info_n.success:
                    nmpc_fail_count += 1
            except Exception as e:
                nmpc_fail_count += 1
                rp = r_com_target.copy(); vp = np.zeros(3)
                a_com_ff_curr = np.zeros(3)

            # Cache knots for sub-step interpolation (matches sim_loop).
            x_plan, _u_plan, _ok = sim.nmpc.get_last_trajectory()
            if x_plan is not None:
                rp_k0 = x_plan[0:3, 0].copy(); rp_k1 = x_plan[0:3, 1].copy()
                vp_k0 = x_plan[3:6, 0].copy(); vp_k1 = x_plan[3:6, 1].copy()
            else:
                rp_k0 = rp.copy(); rp_k1 = rp.copy()
                vp_k0 = vp.copy(); vp_k1 = vp.copy()

        # ── Build torso reference for this QP sub-step ─────────────────
        if use_nmpc:
            qs = k % n_qp_per_nmpc
            alpha_interp = qs / n_qp_per_nmpc
            rp_interp = (1 - alpha_interp) * rp_k0 + alpha_interp * rp_k1
            vp_interp = (1 - alpha_interp) * vp_k0 + alpha_interp * vp_k1
            if bypass_mapping:
                # Constant torso ref (matches case A); NMPC outputs lambda_ref
                # and a_com_ff still feed the QP. Isolates NMPC plan vs mapping.
                p_torso_ref = p_torso_ref0
                v_torso_ref = np.zeros(6)
                a_torso_ff = np.zeros(6)
            else:
                r_b_ref, v_b_ref, a_b_ff, _ = sim.mapping.compute(
                    r_com_ref=rp_interp, v_com_ref=vp_interp,
                    a_com_ff=a_com_ff_curr, q_current=rs.q, dq_current=rs.v)
                p_torso_ref = r_b_ref
                v_torso_ref = np.concatenate([v_b_ref, np.zeros(3)])
                a_torso_ff = np.concatenate([a_b_ff, np.zeros(3)])
            r_com_for_qp = rp_interp
            v_com_for_qp = vp_interp
            lambda_ref_qp = lr if 'lr' in dir() else np.zeros(12)
            a_com_ff_qp = a_com_ff_curr
        else:
            p_torso_ref = p_torso_ref0
            v_torso_ref = np.zeros(6)
            a_torso_ff = np.zeros(6)
            r_com_for_qp = rs.r_com
            v_com_for_qp = np.zeros(3)
            lambda_ref_qp = np.zeros(12)
            a_com_ff_qp = np.zeros(3)

        R_torso_ref = R_torso_ref0  # orientation always held

        # ── EE reference ────────────────────────────────────────────────
        p_ee_ref, R_ee_ref, v_ee_ref, a_ee_ff = ee_reference(
            t, T_traj, p_ee_0, R_ee_0, dp, dtheta, axis)

        J_ee, Jdq_ee, oMf_ee = sim._get_ee_data(rs, swing_arm)
        Jc, Jdc = robot.get_contact_jacobians(
            cc_ss.active_contacts[0], cc_ss.active_contacts[1])

        sw_slice = (robot.arm_b_v_slice if swing_arm == 'b'
                    else robot.arm_a_v_slice)
        H_bs = rs.H[:6, sw_slice]

        hw_phys = (cfg.rwa_I_w * mj_data.qvel[6:9]).copy() if has_rwa \
                  else np.zeros(3)

        try:
            qdd_t, qdd, lam_sol, tau, _info = qp.solve(
                q_t=rs.q_torso, dq_t=rs.dq_torso,
                q=rs.q_joints, dq=rs.dq_joints,
                r_com_ref=r_com_for_qp, v_com_ref=v_com_for_qp,
                lambda_ref=lambda_ref_qp, a_com_ff=a_com_ff_qp,
                H_robot=rs.H, C_robot=rs.C,
                J_com=rs.J_com, Jdot_dq_com=rs.Jdot_dq_com,
                contact_config=cc_ss, J_contacts=Jc, Jdot_dq_contacts=Jdc,
                hw_current=hw_phys,
                hw_min=-cfg.hw_qp_tight, hw_max=cfg.hw_qp_tight,
                r_com=rs.r_com, L_com_current=rs.L_com,
                H_base_swing=H_bs, swing_v_slice=sw_slice,
                settle_mode=False, passivity_active=False,
                J_torso=rs.J_torso, Jdot_dq_torso=rs.Jdot_dq_torso,
                p_torso=rs.oMf_torso.translation,
                R_torso=rs.oMf_torso.rotation,
                p_torso_ref=p_torso_ref, R_torso_ref=R_torso_ref,
                v_torso_ref=v_torso_ref, a_torso_ff=a_torso_ff,
                J_ee=J_ee, Jdot_dq_ee=Jdq_ee,
                p_ee=oMf_ee.translation, R_ee=oMf_ee.rotation,
                p_ee_ref=p_ee_ref, R_ee_ref=R_ee_ref,
                v_ee_ref=v_ee_ref, a_ee_ff=a_ee_ff)
        except Exception as e:
            print(f"  [QP exception @ t={t:.3f}: {e}]")
            tau = np.zeros(n_j)
            qp_fail_count += 1

        tau = np.clip(tau, -cfg.tau_max, cfg.tau_max)
        mj_data.ctrl[:n_j] = tau

        # ── AOCS ────────────────────────────────────────────────────────
        if has_rwa:
            if use_aocs:
                tau_w = compute_aocs_command_legacy_corrected(
                    L_com=rs.L_com, L_com_prev=L_com_prev,
                    r_com=rs.r_com, v_com=rs.v_com, v_com_prev=v_com_prev,
                    hw_current=hw_phys, dt=cfg.dt_qp,
                    robot_mass=robot_mass,
                    K_hw=cfg.aocs_K_hw,
                    hw_min=cfg.hw_min, hw_max=cfg.hw_max,
                    tau_w_max=cfg.aocs_tau_w_max)
                mj_data.ctrl[n_j:n_j + 3] = tau_w
            else:
                mj_data.ctrl[n_j:n_j + 3] = 0.0

        L_com_prev = rs.L_com.copy()
        v_com_prev = rs.v_com.copy()

        mujoco.mj_step(mj_model, mj_data)

        # ── Errors ─────────────────────────────────────────────────────
        if t <= T_traj:
            e_ep = float(np.linalg.norm(oMf_ee.translation - p_ee_ref))
            e_eo = float(np.degrees(
                np.linalg.norm(pin.log3(oMf_ee.rotation.T @ R_ee_ref))))
            e_tp = float(np.linalg.norm(
                rs.oMf_torso.translation - p_torso_ref0))
            R_err_t = rs.oMf_torso.rotation.T @ R_torso_ref0
            e_to = float(np.degrees(np.arccos(
                np.clip((np.trace(R_err_t) - 1) / 2, -1, 1))))
            if e_ep > e_ee_pos_peak: e_ee_pos_peak = e_ep
            if e_eo > e_ee_ori_peak: e_ee_ori_peak = e_eo
            if e_tp > e_torso_pos_peak: e_torso_pos_peak = e_tp
            if e_to > e_torso_ori_peak: e_torso_ori_peak = e_to

        if (k % 100) == 0:
            print(f"  t={t:6.3f}s  ee_pos_peak={e_ee_pos_peak*1000:7.2f}mm  "
                  f"torso_pos_peak={e_torso_pos_peak*1000:6.2f}mm  "
                  f"ee_ori_peak={e_ee_ori_peak:5.2f}°  "
                  f"|tau|_inf={np.max(np.abs(tau)):.2f}Nm")

    print()
    print(f"[case {case}] RESULT over t in [0, T_traj]:")
    print(f"   EE pos peak    = {e_ee_pos_peak*1000:8.2f} mm")
    print(f"   EE ori peak    = {e_ee_ori_peak:8.3f} °")
    print(f"   Torso pos peak = {e_torso_pos_peak*1000:8.2f} mm")
    print(f"   Torso ori peak = {e_torso_ori_peak:8.3f} °")
    print(f"   QP failures    = {qp_fail_count}")
    print(f"   NMPC failures  = {nmpc_fail_count}")
    return {
        'case': case,
        'ee_pos_peak_mm': e_ee_pos_peak * 1000,
        'ee_ori_peak_deg': e_ee_ori_peak,
        'torso_pos_peak_mm': e_torso_pos_peak * 1000,
        'torso_ori_peak_deg': e_torso_ori_peak,
        'qp_fail': qp_fail_count,
        'nmpc_fail': nmpc_fail_count,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--case',
        choices=['A', 'B', 'C', 'B_aware', 'B_aware_free', 'B_measured',
                 'B_const_torso', 'B_compensated'],
        required=True)
    args = parser.parse_args()
    run_case(args.case)
