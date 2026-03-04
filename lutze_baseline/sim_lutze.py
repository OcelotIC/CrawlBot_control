"""
sim_lutze.py — Single-step simulation with Lutze baseline controller.

Replaces the NMPC + whole-body QP hierarchy with Lutze's single-step
contact wrench optimizer.  Uses the same MuJoCo simulation, robot model,
gait scheduler, and planners as sim_torso6d.py for a fair comparison.

Architecture:
    - Stance arm A: Lutze QP computes optimal contact wrench Fc_a
    - Swing arm B: Cartesian impedance to reach next anchor
    - No NMPC, no whole-body QP — just wrench mapping + impedance
    - Structure rotates freely (no active attitude control)

Usage:
    python -m lutze_baseline.sim_lutze --urdf models/VISPA_crawling_fixed.urdf \\
                                        --mjcf models/VISPA_crawling.xml [--plot]
"""

import sys
import os
import argparse
import numpy as np
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import pinocchio as pin
from robot_interface import RobotInterface
from contact_scheduler import ContactScheduler, read_anchors_from_mujoco
from locomotion_planner import LocomotionPlanner
from swing_planner import SwingPlanner
from ik import dock_configuration
from simulation_loop import pinocchio_to_mujoco, mujoco_to_pinocchio
from solvers.contact_phase import ContactPhase
from torso_planner import TorsoPlanner

from lutze_baseline.centroidal_model import compute_centroidal_state
from lutze_baseline.contact_adjoint import compute_dual_contact_adjoints
from lutze_baseline.momentum_map import compute_momentum_map
from lutze_baseline.lutze_feedforward import compute_feedforward, LutzeFeedforwardConfig
from lutze_baseline.lutze_qp import LutzeQP, LutzeQPConfig
from lutze_baseline.lutze_joint_torques import compute_joint_torques
from lutze_baseline.lutze_swing_controller import (
    compute_swing_torques, SwingImpedanceConfig,
)


# ═══════════════════════════════════════════════════════════════════
#  Configuration (same as sim_torso6d.py for fair comparison)
# ═══════════════════════════════════════════════════════════════════

TORSO_MASS = 40.0       # kg (corrected from URDF)
TAU_MAX = 10.0           # Nm per joint
WELD_R = 0.005           # 5 mm docking threshold
T_SWING = 6.0            # s single-support duration
T_DS = 0.5               # s double-support duration
TORSO_FRAC = 0.70        # fraction of IK torso displacement
TORSO_DELAY = 0.20       # fraction of swing before torso starts
T_MAX = 12.0             # s max simulation time
DT_CTRL = 0.01           # s control rate (= MuJoCo timestep)
DT_MJ = 0.01             # s MuJoCo timestep
CLEARANCE = 0.03         # m swing clearance


def run_simulation(urdf_path, mjcf_path, save_log=True, verbose=True):
    """Run single-step simulation with Lutze baseline controller.

    Returns
    -------
    log : dict
        Logged data arrays.
    docked : bool
        Whether real docking was achieved.
    """
    # ── MuJoCo setup ──
    mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = DT_MJ

    # Correct torso mass
    tid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
    rat = TORSO_MASS / mj_model.body_mass[tid]
    mj_model.body_mass[tid] = TORSO_MASS
    mj_model.body_inertia[tid] *= rat
    mujoco.mj_forward(mj_model, mj_data)

    mj_a, mj_b = read_anchors_from_mujoco(mj_model, mj_data)

    # ── Pinocchio setup ──
    robot = RobotInterface(urdf_path, gravity='zero', torso_mass=TORSO_MASS)
    model = robot.model

    # ── Contact scheduler ──
    sched = ContactScheduler(
        anchors_a=mj_a, anchors_b=mj_b,
        dt_ds=T_DS, dt_ss=T_SWING)
    plan = sched.plan_traversal(start_a=2, start_b=2, n_steps=1)

    # ── IK: dock configurations ──
    q_dock = dock_configuration(
        model, sched.anchor_se3('a', 2), sched.anchor_se3('b', 2))
    pin.forwardKinematics(model, robot.data, q_dock)
    pin.updateFramePlacements(model, robot.data)
    p_t0 = robot.data.oMi[1].translation.copy()
    R_t0 = robot.data.oMi[1].rotation.copy()

    p_4b = mj_b[3][:3]
    q_final = dock_configuration(
        model, sched.anchor_se3('a', 2),
        pin.SE3(sched.anchor_se3('b', 2).rotation, p_4b))
    pin.forwardKinematics(model, robot.data, q_final)
    pin.updateFramePlacements(model, robot.data)
    p_t1_full = robot.data.oMi[1].translation.copy()
    R_t1_full = robot.data.oMi[1].rotation.copy()

    # Torso target: TORSO_FRAC of full displacement
    dp = p_t1_full - p_t0
    dR = R_t0.T @ R_t1_full
    omega = pin.log3(dR)
    p_t1 = p_t0 + TORSO_FRAC * dp
    R_t1 = R_t0 @ pin.exp3(TORSO_FRAC * omega)

    # ── Torso planner (used for CoM reference) ──
    t_ss0 = plan.t_start[1]
    t_ss1 = plan.t_end[1]
    t_torso_start = t_ss0 + TORSO_DELAY * T_SWING

    torso_pl = TorsoPlanner()
    torso_pl.set_hold(p_t0, R_t0)
    torso_pl.add_phase(t_torso_start, t_ss1, p_t0, R_t0, p_t1, R_t1)

    if verbose:
        print(f"Torso: {TORSO_FRAC:.0%} of Δ={np.linalg.norm(dp)*100:.1f}cm "
              f"= {np.linalg.norm(p_t1-p_t0)*100:.1f}cm")
        print(f"Swing: [{t_ss0:.1f}, {t_ss1:.1f}]s, "
              f"Torso: [{t_torso_start:.1f}, {t_ss1:.1f}]s")

    # ── Initialize MuJoCo state ──
    sp = mj_data.qpos[0:3].copy()
    mqp, _ = pinocchio_to_mujoco(
        q_dock, np.zeros(18),
        struct_pos=sp, struct_quat=mj_data.qpos[3:7].copy())
    mj_data.qpos[:] = mqp
    mj_data.qvel[:] = 0.0

    for i in range(mj_model.neq):
        nm = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, i)
        mj_data.eq_active[i] = 1 if nm in ('grip_a_to_3a', 'grip_b_to_3b') else 0
    mujoco.mj_forward(mj_model, mj_data)
    for _ in range(200):
        mujoco.mj_step(mj_model, mj_data)

    # ── Planners ──
    rs = robot.update(*mujoco_to_pinocchio(mj_data.qpos, mj_data.qvel))
    am = sum(model.inertias[i].mass for i in range(2, 8))
    plnr = LocomotionPlanner(sched, arm_mass=am, total_mass=rs.total_mass)
    plnr.calibrate_from_config(rs.r_com)
    swp = SwingPlanner(sched, clearance=CLEARANCE)

    # ── Site IDs ──
    sg_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, 'gripper_b')
    sa_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, 'anchor_4b')
    sg_a_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, 'gripper_a')
    sa_a_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, 'anchor_3a')

    # ── Weld map ──
    wm = {}
    for i in range(mj_model.neq):
        nm = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, i)
        if nm and nm.startswith('grip_'):
            pts = nm.split('_to_')
            arm = pts[0].split('_')[1]
            wm[(arm, int(pts[1][0]) - 1)] = i

    # Start: A docked, B docked (DS)
    for eq in range(mj_model.neq):
        mj_data.eq_active[eq] = 0
    mj_data.eq_active[wm[('a', 2)]] = 1
    mj_data.eq_active[wm[('b', 2)]] = 1

    # ── Lutze controller setup ──
    cfg_ff = LutzeFeedforwardConfig(
        Kr=np.diag([50.0, 50.0, 50.0]),
        Dr=np.diag([10.0, 10.0, 10.0]),
        F_max=25.0,
    )
    cfg_qp = LutzeQPConfig(
        Qr=np.eye(6) * 1.0,
        Qb=np.eye(3) * 10.0,
        f_max=25.0,
        tau_max=8.0,
        tau_w_max=2.0,  # momentum rate constraint (matching MPC_crawling)
    )
    qp = LutzeQP(cfg_qp)
    cfg_swing = SwingImpedanceConfig(
        Kp=np.diag([80.0, 80.0, 80.0]),
        Kd=np.diag([15.0, 15.0, 15.0]),
        F_max=15.0,
    )

    # ── Logging ──
    cc_ss = sched.contact_config_at(t_ss0 + 0.1)
    hw = np.array([2., -1., 0.5])  # initial wheel momentum

    log = dict(
        t=[], p_torso=[], p_torso_ref=[], p_grip_b=[], p_grip_b_ref=[],
        d_grip_b=[], d_grip_a=[], tau=[], tau_max_joint=[],
        e_torso_pos=[], struct_pos=[], phase=[],
        L_com=[], struct_quat=[], struct_omega=[])

    t0_wall = time.time()
    t = 0.0
    docked = False
    b_released = False

    # ── Simulation loop ──
    while t < T_MAX and not docked:
        in_ds = t < t_ss0
        in_ss = t_ss0 <= t < t_ss1
        in_ext = t >= t_ss1
        phase_str = 'DS' if in_ds else ('SS' if in_ss else 'EXT')

        # Release B weld at SS start
        if in_ss and not b_released:
            mj_data.eq_active[wm[('b', 2)]] = 0
            b_released = True

        # Contact configuration
        active_a = True  # A always docked (stance)
        active_b = in_ds  # B only during double support

        # State
        pq, pv = mujoco_to_pinocchio(mj_data.qpos, mj_data.qvel)
        rs = robot.update(pq, pv)
        cs = compute_centroidal_state(robot, rs)

        # Structure state
        struct_pos = mj_data.qpos[0:3].copy()
        struct_quat_wxyz = mj_data.qpos[3:7].copy()  # MuJoCo: [w,x,y,z]
        struct_omega = mj_data.qvel[3:6].copy()

        # Reference: use torso planner for CoM reference
        tref = torso_pl.reference_at(t)
        r_ref = tref.p.copy()  # torso position as CoM proxy
        v_ref = tref.v[:3].copy()  # linear velocity

        # ── Lutze feedforward ──
        F_d_r, F_d_b = compute_feedforward(
            cs.r_com, cs.v_com, r_ref, v_ref,
            struct_quat_wxyz=struct_quat_wxyz,
            struct_omega=struct_omega,
            cfg=cfg_ff,
        )

        # ── Contact adjoints and momentum map ──
        Ad_a, Ad_b = compute_dual_contact_adjoints(
            rs, active_a=active_a, active_b=active_b)
        r_ca = rs.oMf_tool_a.translation if active_a else None
        r_cb = rs.oMf_tool_b.translation if active_b else None
        M_lam = compute_momentum_map(cs.r_com, r_ca, r_cb)

        # ── Lutze QP ──
        Fc_a, Fc_b, info = qp.solve(Ad_a, Ad_b, M_lam, F_d_r, F_d_b)

        # ── Joint torques from contact wrenches ──
        tau = compute_joint_torques(
            Fc_a, Fc_b, rs.J_tool_a, rs.J_tool_b,
            active_a=active_a, active_b=active_b,
            tau_max=TAU_MAX,
        )

        # ── Swing arm control (during SS and EXT) ──
        if in_ss or in_ext:
            if in_ss:
                sr = swp.reference_at(min(t, t_ss1 - 0.01))
                p_ee_ref = sr.p_ee
                v_ee_ref = sr.v_ee
            else:  # EXT
                mujoco.mj_forward(mj_model, mj_data)
                p_ee_ref = mj_data.site_xpos[sa_id].copy()
                v_ee_ref = np.zeros(3)

            p_ee = rs.oMf_tool_b.translation
            v_ee = (rs.J_tool_b @ pv)[:3]  # angular part — but we want linear
            # Actually: J_tool_b is (6, 18), J@v gives [ang(3); lin(3)]
            v_ee_cur = (rs.J_tool_b @ pv)[3:]  # linear velocity

            tau_swing = compute_swing_torques(
                p_ee, v_ee_cur, p_ee_ref, v_ee_ref,
                J_ee=rs.J_tool_b,
                cfg=cfg_swing,
                tau_max=TAU_MAX,
            )
            tau = tau + tau_swing

        # Clip total torques
        tau = np.clip(tau, -TAU_MAX, TAU_MAX)

        # ── Apply to MuJoCo ──
        mj_data.ctrl[:12] = tau
        mujoco.mj_step(mj_model, mj_data)

        # ── AOCS momentum update ──
        rs2 = robot.update(
            *mujoco_to_pinocchio(mj_data.qpos, mj_data.qvel))
        hw -= (rs2.L_com - rs.L_com) / DT_MJ * DT_MJ
        hw = np.clip(hw, -50., 50.)

        mujoco.mj_forward(mj_model, mj_data)

        # Real docking check
        d_b = np.linalg.norm(
            mj_data.site_xpos[sg_id] - mj_data.site_xpos[sa_id])
        d_a = np.linalg.norm(
            mj_data.site_xpos[sg_a_id] - mj_data.site_xpos[sa_a_id])
        if d_b < WELD_R and t > t_ss0 + 1.0:
            if verbose:
                print(f"*** REAL DOCK at t={t:.2f}s, d={d_b*1000:.1f}mm ***")
            docked = True

        # Log
        log['t'].append(t)
        log['p_torso'].append(mj_data.qpos[7:10].copy())
        log['p_torso_ref'].append(tref.p.copy())
        log['p_grip_b'].append(mj_data.site_xpos[sg_id].copy())
        sr_l = swp.reference_at(min(t, t_ss1 - 0.01))
        log['p_grip_b_ref'].append(
            sr_l.p_ee.copy() if sr_l.is_swinging
            else mj_data.site_xpos[sa_id].copy())
        log['d_grip_b'].append(d_b)
        log['d_grip_a'].append(d_a)
        log['tau'].append(tau.copy())
        log['tau_max_joint'].append(np.max(np.abs(tau)))
        log['e_torso_pos'].append(
            np.linalg.norm(rs.oMf_torso.translation - tref.p))
        log['struct_pos'].append(struct_pos.copy())
        log['phase'].append(phase_str)
        log['L_com'].append(cs.L_com.copy())
        log['struct_quat'].append(struct_quat_wxyz.copy())
        log['struct_omega'].append(struct_omega.copy())

        t += DT_CTRL

    wall = time.time() - t0_wall

    # Convert for JSON
    for k in log:
        log[k] = [x.tolist() if hasattr(x, 'tolist') else x for x in log[k]]

    if verbose:
        d_arr = np.array(log['d_grip_b'])
        t_arr = np.array(log['t'])
        pt = np.array(log['p_torso'])
        sp = np.array(log['struct_pos'])
        sd = np.linalg.norm(
            np.array(sp[-1]) - np.array(sp[0])) * 100
        L_arr = np.array(log['L_com'])
        L_max = np.max(np.linalg.norm(L_arr, axis=1)) if len(L_arr) > 0 else 0

        print(f"\nSimulation: {wall:.1f}s wall, {len(t_arr)} steps")
        print(f"min d_grip_b = {d_arr.min()*1000:.1f}mm "
              f"(t={t_arr[d_arr.argmin()]:.2f}s)")
        print(f"max |τ| = {max(log['tau_max_joint']):.2f} Nm")
        print(f"max |L_com| = {L_max:.3f} Nms")
        print(f"struct drift = {sd:.1f}cm")
        print(f"torso Δx = {(pt[-1,0]-pt[0,0])*100:+.1f}cm")
        print(f"docked = {docked}")

    if save_log:
        with open('sim_lutze_log.json', 'w') as f:
            json.dump(log, f)

    return log, docked


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='VISPA single-step with Lutze baseline controller')
    parser.add_argument('--urdf', default='models/VISPA_crawling_fixed.urdf')
    parser.add_argument('--mjcf', default='models/VISPA_crawling.xml')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    log, docked = run_simulation(args.urdf, args.mjcf)

    if args.plot:
        from lutze_baseline.plot_comparison import plot_results
        plot_results(log, save_path='sim_lutze_results.png')
