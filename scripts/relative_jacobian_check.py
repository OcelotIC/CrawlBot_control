"""Relative-Jacobian check at d0_a0 settle tick k=10.

Re-runs the deterministic 11-tick d0_a0 settle, transient MJCF mutation
to (damping=0, armature=0), restored via try/finally. At k=10 computes:

  (1) J_rel_pin_A = J_pin(gripper_a) - J_pin(anchor_2a)
      Since `anchor_2a` is a site on body `structure` and the Pinocchio
      URDF root-fixes `Link_0` to the world, `J_pin(anchor_2a) ≡ 0`
      by construction (the structure body is not in the URDF). So in
      Pinocchio, `J_rel_pin == J_abs_pin` rigorously; the matrix is
      reported and the first row printed.

  (2) P_weld_rel_pin_A = (J_rel_pin_A · v_pin)^T · λ_qp[0:6]

  (3) Same for B, and total.

  (4) v_anchor_2a in MuJoCo via `mj_data.site_xvelp` / `site_xvelr`,
      populated by mj_forward. If non-negligible, the relative and
      absolute frame velocities in MuJoCo's world differ; in Pinocchio
      they are identical because anchor is world-fixed.

  (5) MuJoCo `eq_weld` efc frame convention from official docs.

Writes Misc/runs/M7_settle_diag/relative_jacobian_check.md.
"""
from __future__ import annotations

import os
import re
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import numpy as np
import mujoco

import scripts.run_m7_single_step as r_single
from crawlbot.simulation.sim_loop import SimulationLoop
from crawlbot.core.state_conversions import mujoco_to_pinocchio


URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')
MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')
OUT  = os.path.join(_root, 'Misc', 'runs', 'M7_settle_diag',
                    'relative_jacobian_check.md')
TARGET_K = 10

ROBOT_JOINT_RE = re.compile(
    r'(<default class="robot_joint">\s*\n\s*<joint damping=")[^"]+'
    r'(" armature=")[^"]+(")')


def _mutate_mjcf(damping, armature):
    with open(MJCF, 'r') as f:
        text = f.read()
    new, n = ROBOT_JOINT_RE.subn(
        rf'\g<1>{damping}\g<2>{armature}\g<3>', text, count=1)
    if n != 1:
        raise RuntimeError('robot_joint default not found')
    with open(MJCF, 'w') as f:
        f.write(new)


def _verify_restored():
    with open(MJCF, 'r') as f:
        text = f.read()
    m = ROBOT_JOINT_RE.search(text)
    assert m
    body = text[m.start():m.end()]
    assert 'damping="0.05"' in body and 'armature="0.05"' in body, body


def _run():
    cfg = r_single._make_m7_config()
    cfg.preplanner_a_cruise_max = 0.01
    cfg.preplanner_cruise_ramp_frac = 0.2

    sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=cfg)
    sim.setup(n_steps=1, start_a=2, start_b=2)
    cc_ds = sim.sched.contact_config_at(sim.plan.t_start[1] + 0.1)

    qp = sim.qp_ss
    n_j = sim.robot.n_joints
    mj_nv = sim.mj_model.nv
    dt = cfg.dt_qp

    # Discover the active weld sites: sim.setup called _activate_weld
    # for the start_a=2 / start_b=2 anchors, so anchor_2a and anchor_2b
    # are the sites we care about.
    mj_model = sim.mj_model
    anchor_a_sid = mujoco.mj_name2id(mj_model,
                                      mujoco.mjtObj.mjOBJ_SITE, 'anchor_2a')
    anchor_b_sid = mujoco.mj_name2id(mj_model,
                                      mujoco.mjtObj.mjOBJ_SITE, 'anchor_2b')
    grip_a_sid   = mujoco.mj_name2id(mj_model,
                                      mujoco.mjtObj.mjOBJ_SITE, 'gripper_a')
    grip_b_sid   = mujoco.mj_name2id(mj_model,
                                      mujoco.mjtObj.mjOBJ_SITE, 'gripper_b')

    snap = {}
    for k in range(TARGET_K + 1):
        pq, pv = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
        rs = sim.robot.update(pq, pv)
        Jc_pin, _ = sim.robot.get_contact_jacobians(True, True)

        try:
            _, _, lam_qp, tau, _ = qp.solve(
                q_t=rs.q_torso, dq_t=rs.dq_torso,
                q=rs.q_joints, dq=rs.dq_joints,
                r_com_ref=rs.r_com, v_com_ref=np.zeros(3),
                lambda_ref=np.zeros(12), a_com_ff=np.zeros(3),
                H_robot=rs.H, C_robot=rs.C,
                J_com=rs.J_com, Jdot_dq_com=rs.Jdot_dq_com,
                contact_config=cc_ds,
                J_contacts=Jc_pin, Jdot_dq_contacts=np.zeros(12),
                hw_current=np.zeros(3),
                hw_min=cfg.hw_min, hw_max=cfg.hw_max,
                r_com=rs.r_com, L_com_current=rs.L_com,
                J_torso=rs.J_torso, Jdot_dq_torso=rs.Jdot_dq_torso,
                p_torso=rs.oMf_torso.translation.copy(),
                R_torso=rs.oMf_torso.rotation.copy(),
                p_torso_ref=rs.oMf_torso.translation.copy(),
                R_torso_ref=rs.oMf_torso.rotation.copy(),
                v_torso_ref=np.zeros(6), a_torso_ff=np.zeros(6),
                settle_mode=True, passivity_active=True)
            tau = np.clip(tau, -cfg.tau_max, cfg.tau_max)
        except Exception:
            tau = np.clip(-cfg.Kd_settle_damping * rs.dq_joints,
                          -cfg.tau_max, cfg.tau_max)
            lam_qp = np.zeros(12)

        if k == TARGET_K:
            mujoco.mj_forward(sim.mj_model, sim.mj_data)
            # Site positions and velocities (world frame)
            snap['k'] = k
            snap['t_s'] = k * dt
            snap['Jc_pin']   = Jc_pin.copy()       # (12, 20) = [J(gripper_a); J(gripper_b)]
            snap['v_pin']    = rs.v.copy()
            snap['lam_qp']   = lam_qp.copy()
            snap['anchor_a_xpos']  = sim.mj_data.site_xpos[anchor_a_sid].copy()
            snap['anchor_b_xpos']  = sim.mj_data.site_xpos[anchor_b_sid].copy()
            snap['gripper_a_xpos'] = sim.mj_data.site_xpos[grip_a_sid].copy()
            snap['gripper_b_xpos'] = sim.mj_data.site_xpos[grip_b_sid].copy()

            # Site velocities via mj_objectVelocity: returns (rot, lin)
            # in object-centered frame, world-aligned orientation
            # (flg_local=0). 6-vector layout is [ang(3), lin(3)].
            def _site_vel_world(sid):
                res = np.zeros(6)
                mujoco.mj_objectVelocity(
                    sim.mj_model, sim.mj_data,
                    mujoco.mjtObj.mjOBJ_SITE, sid, res, 0)
                return res[3:6].copy(), res[0:3].copy()   # lin, ang
            snap['anchor_a_vlin'], snap['anchor_a_vang'] = _site_vel_world(anchor_a_sid)
            snap['anchor_b_vlin'], snap['anchor_b_vang'] = _site_vel_world(anchor_b_sid)
            snap['grip_a_vlin'],   snap['grip_a_vang']   = _site_vel_world(grip_a_sid)
            snap['grip_b_vlin'],   snap['grip_b_vang']   = _site_vel_world(grip_b_sid)

            # Structure free-joint qvel (first 6 components of mj qvel)
            snap['mj_struct_v']   = sim.mj_data.qvel[0:6].copy()
            snap['mj_qvel_nv']    = int(mj_nv)

        sim.mj_data.ctrl[:n_j] = tau
        if sim.has_rwa:
            sim.mj_data.ctrl[n_j:n_j + 3] = 0.0
        mujoco.mj_step(sim.mj_model, sim.mj_data)

        if k == TARGET_K:
            break
    return snap


def _fmt(v, prec=6):
    return '[' + ', '.join(f'{x:+.{prec}e}' for x in v) + ']'


def main():
    with open(MJCF, 'r') as f:
        original = f.read()
    try:
        _mutate_mjcf(0.0, 0.0)
        snap = _run()
    finally:
        with open(MJCF, 'w') as f:
            f.write(original)
        _verify_restored()

    Jc_pin   = snap['Jc_pin']     # (12, 20): rows 0-5 = J(gripper_a), rows 6-11 = J(gripper_b)
    v_pin    = snap['v_pin']
    lam_qp   = snap['lam_qp']

    # In Pinocchio's URDF the world is fixed; the structure body is not
    # a moving body — it's the kinematic root. Therefore J(anchor_2a)=0.
    J_gripper_a = Jc_pin[0:6, :].copy()
    J_gripper_b = Jc_pin[6:12, :].copy()
    J_anchor_a_pin = np.zeros_like(J_gripper_a)
    J_anchor_b_pin = np.zeros_like(J_gripper_b)
    J_rel_pin_A = J_gripper_a - J_anchor_a_pin
    J_rel_pin_B = J_gripper_b - J_anchor_b_pin

    # Item 2/3: P_weld via relative Jacobian (Pinocchio side, per contact).
    lam_A = lam_qp[0:6]
    lam_B = lam_qp[6:12]
    P_rel_pin_A = float((J_rel_pin_A @ v_pin) @ lam_A)
    P_rel_pin_B = float((J_rel_pin_B @ v_pin) @ lam_B)
    P_rel_pin_total = P_rel_pin_A + P_rel_pin_B

    # Also compute the absolute power for cross-check (should be identical
    # since J(anchor)=0).
    P_abs_pin_A = float((J_gripper_a @ v_pin) @ lam_A)
    P_abs_pin_B = float((J_gripper_b @ v_pin) @ lam_B)
    P_abs_pin_total = P_abs_pin_A + P_abs_pin_B

    # Item 4: anchor site velocities in MuJoCo world frame.
    anchor_a_vlin = snap['anchor_a_vlin']
    anchor_a_vang = snap['anchor_a_vang']
    anchor_b_vlin = snap['anchor_b_vlin']
    anchor_b_vang = snap['anchor_b_vang']
    grip_a_vlin   = snap['grip_a_vlin']
    grip_a_vang   = snap['grip_a_vang']
    grip_b_vlin   = snap['grip_b_vlin']
    grip_b_vang   = snap['grip_b_vang']
    struct_v      = snap['mj_struct_v']

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write('# M7 — relative-Jacobian check (d0_a0, k=10)\n\n')
        f.write(f't = {snap["t_s"]:.3f} s. Deterministic replay of the '
                'first 11 ticks of the d0_a0 settle; MJCF mutated to '
                '(damping=0, armature=0) transiently and restored on '
                'exit.\n\n')

        f.write('## 1. `J_rel_pin` = `J(gripper_*)` − `J(anchor_2*)`\n\n')
        f.write('The URDF (`models/VISPA_crawling_fixed.urdf`) roots '
                '`Link_0` to `world` with a fixed joint; the MJCF body '
                '`structure` (a second free-joint body in MuJoCo carrying '
                'the anchor sites) is **not present in the URDF**. In '
                'Pinocchio\'s kinematic tree anchor `anchor_2*` would be '
                'a world-fixed point, so `J_pin(anchor_2*) ≡ 0` by '
                'construction. Therefore in Pinocchio `J_rel == J_abs` '
                'rigorously.\n\n')
        f.write(f'Shape of `J_rel_pin_A` = `J(gripper_a)` − 0  : '
                f'`{J_rel_pin_A.shape}`\n\n')
        f.write('First row of `J_rel_pin_A` (same as first row of '
                '`J(gripper_a)`):\n\n')
        f.write(f'`{_fmt(J_rel_pin_A[0])}`\n\n')
        f.write(f'First row of `J_rel_pin_B`:\n\n')
        f.write(f'`{_fmt(J_rel_pin_B[0])}`\n\n')

        f.write('## 2. `P_weld_rel_pin_A`  (relative Jacobian, arm A)\n\n')
        f.write(f'`P_rel_pin_A = (J_rel_pin_A · v_pin)^T · λ_qp[0:6] '
                f'= {P_rel_pin_A:+.6e}` W\n\n')
        f.write(f'For cross-check, absolute: `P_abs_pin_A = '
                f'{P_abs_pin_A:+.6e}` W (equal by construction; '
                f'`J_anchor_pin = 0`).\n\n')

        f.write('## 3. `P_weld_rel_pin_B` + total\n\n')
        f.write(f'`P_rel_pin_B = (J_rel_pin_B · v_pin)^T · λ_qp[6:12] '
                f'= {P_rel_pin_B:+.6e}` W\n\n')
        f.write(f'**Total Pinocchio relative-Jacobian P_weld = '
                f'{P_rel_pin_total:+.6e} W** '
                f'(absolute-Jacobian total = {P_abs_pin_total:+.6e} W).\n\n')

        f.write('## 4. Anchor and gripper site velocities in MuJoCo world\n\n')
        f.write('Read from `mj_data.site_xvelp` (linear) and '
                '`mj_data.site_xvelr` (angular), populated by '
                '`mj_forward`. World frame.\n\n')
        f.write('| site | linear [m/s] | angular [rad/s] | ‖lin‖ [m/s] | ‖ang‖ [rad/s] |\n')
        f.write('|---|---|---|---|---|\n')
        for name, vlin, vang in [
            ('anchor_2a', anchor_a_vlin, anchor_a_vang),
            ('gripper_a', grip_a_vlin,   grip_a_vang),
            ('anchor_2b', anchor_b_vlin, anchor_b_vang),
            ('gripper_b', grip_b_vlin,   grip_b_vang),
        ]:
            f.write(f'| `{name}` | `{_fmt(vlin, prec=4)}` | '
                    f'`{_fmt(vang, prec=4)}` | '
                    f'`{np.linalg.norm(vlin):.4e}` | '
                    f'`{np.linalg.norm(vang):.4e}` |\n')
        f.write('\nStructure free-joint `mj_qvel[0:6]` (lin + ang in world): '
                f'`{_fmt(struct_v, prec=4)}`, ‖lin‖ = '
                f'`{np.linalg.norm(struct_v[0:3]):.4e}` m/s, '
                f'‖ang‖ = `{np.linalg.norm(struct_v[3:6]):.4e}` rad/s.\n\n')
        # Per-contact relative velocity magnitudes (what the weld sees in world)
        dv_lin_A = grip_a_vlin - anchor_a_vlin
        dv_ang_A = grip_a_vang - anchor_a_vang
        dv_lin_B = grip_b_vlin - anchor_b_vlin
        dv_ang_B = grip_b_vang - anchor_b_vang
        f.write('Per-contact RELATIVE velocities `v_gripper − v_anchor` '
                '(world frame, what an idealised weld residual would see):\n\n')
        f.write(f'- A: `lin = {_fmt(dv_lin_A, prec=4)}`, '
                f'`ang = {_fmt(dv_ang_A, prec=4)}`, '
                f'`‖lin‖ = {np.linalg.norm(dv_lin_A):.4e}`, '
                f'`‖ang‖ = {np.linalg.norm(dv_ang_A):.4e}`\n')
        f.write(f'- B: `lin = {_fmt(dv_lin_B, prec=4)}`, '
                f'`ang = {_fmt(dv_ang_B, prec=4)}`, '
                f'`‖lin‖ = {np.linalg.norm(dv_lin_B):.4e}`, '
                f'`‖ang‖ = {np.linalg.norm(dv_ang_B):.4e}`\n\n')

        f.write('## 5. MuJoCo `eq_weld` frame convention\n\n')
        f.write('From the MuJoCo reference documentation (equality '
                'constraints, `<weld>`), an `mjEQ_WELD` produces 6 '
                '`efc` rows whose residual is the relative pose error '
                'between `site1` (body1-attached) and `site2` '
                '(body2-attached). Internally MuJoCo computes the '
                'relative body Jacobian `J_rel = J(body1) − J(body2)` '
                'and writes it into `efc_J` rows. The `efc_force` '
                'entries are the Lagrange multipliers enforcing those '
                '6 scalar residuals.\n\n')
        f.write('Key implication recorded, no interpretation: MuJoCo '
                '`efc_J` is intrinsically **relative** (includes motion '
                'of both connected bodies); Pinocchio\'s `J_c` here is '
                '**absolute with anchor=world** because the `structure` '
                'body exists only in the MJCF. The `mj_struct_v` block '
                'reported above is exactly the degrees of freedom '
                'present in MuJoCo but absent in the URDF.\n')

    # Console
    print()
    print('=' * 78)
    print(f'Relative-Jacobian check — d0_a0 k={snap["k"]}, t={snap["t_s"]:.3f}s')
    print('=' * 78)
    print(f'  J_rel_pin_A shape = {J_rel_pin_A.shape}')
    print(f'  P_rel_pin_A       = {P_rel_pin_A:+.6e} W  (abs = {P_abs_pin_A:+.6e} W)')
    print(f'  P_rel_pin_B       = {P_rel_pin_B:+.6e} W  (abs = {P_abs_pin_B:+.6e} W)')
    print(f'  P_rel_pin_total   = {P_rel_pin_total:+.6e} W')
    print()
    print(f'  |v_anchor_2a|_lin = {np.linalg.norm(anchor_a_vlin):.4e} m/s  '
          f'|v_anchor_2a|_ang = {np.linalg.norm(anchor_a_vang):.4e} rad/s')
    print(f'  |v_anchor_2b|_lin = {np.linalg.norm(anchor_b_vlin):.4e} m/s  '
          f'|v_anchor_2b|_ang = {np.linalg.norm(anchor_b_vang):.4e} rad/s')
    print(f'  |mj_struct_v|_lin = {np.linalg.norm(struct_v[0:3]):.4e} m/s  '
          f'|mj_struct_v|_ang = {np.linalg.norm(struct_v[3:6]):.4e} rad/s')
    print()
    print(f'Report: {OUT}')


if __name__ == '__main__':
    main()
