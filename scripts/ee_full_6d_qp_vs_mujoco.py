"""Offline 6D EE-task decomposition + QP-vs-MuJoCo comparison.

Loads results/M7_ee_ori_diag/A_swing_t3p6.npz (dumped by
scripts/bisect_qp_cascade.py:run_v21_case) and writes
results/M7_ee_ori_diag/ee_full_6d_qp_vs_mujoco.md. No interpretation.
"""
from __future__ import annotations

import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

import numpy as np
import pinocchio as pin


NPZ_PATH = os.path.join(_root, 'results', 'M7_ee_ori_diag',
                        'A_swing_t3p6.npz')
OUT_PATH = os.path.join(_root, 'results', 'M7_ee_ori_diag',
                        'ee_full_6d_qp_vs_mujoco.md')


def _v6(v, prec=6):
    return '[' + ', '.join(f'{x:+.{prec}f}' for x in v) + ']'


def main():
    d = np.load(NPZ_PATH)
    t_rel = float(d['t_rel'])
    dt    = float(d['dt_qp'])
    tau_max = float(d['tau_max'])

    # Matrices and state at the dump tick (pre-QP).
    J_ee     = d['J_ee']                # (6, 20)
    J_torso  = d['J_torso']             # (6, 20)
    Jdot_dq_ee     = d['Jdot_dq_ee']    # (6,)
    Jdot_dq_torso  = d['Jdot_dq_torso'] # (6,)
    v        = d['v']                   # (20,)  Pinocchio velocity at t_dump
    v_post   = d['v_post']              # (20,)  Pinocchio velocity at t_dump + dt

    R_ee     = d['R_ee']
    R_ee_ref = d['R_ee_ref']
    p_ee     = d['p_ee']
    p_ee_ref = d['p_ee_ref']
    v_ee_ref = d['v_ee_ref']
    a_ee_ff_planner = d['a_ee_ff']

    R_torso     = d['R_torso']
    R_torso_ref = d['R_torso_ref']
    p_torso     = d['p_torso']
    p_torso_ref = d['p_torso_ref']
    v_torso_ref = d['v_torso_ref']
    a_torso_ff  = d['a_torso_ff']

    Kp_torso  = d['Kp_torso']
    Kd_torso  = d['Kd_torso']
    Kp_ee     = d['Kp_ee']
    Kd_ee     = d['Kd_ee']
    Kp_ee_ang = d['Kp_ee_ang']
    Kd_ee_ang = d['Kd_ee_ang']

    qdd_t_qp = d['qdd_t_qp']            # (6,)
    qdd_qp   = d['qdd_qp']              # (14,)
    qdd_full = np.concatenate([qdd_t_qp, qdd_qp])     # (20,)
    tau_qp   = d['tau_qp']              # (14,)

    # ── (1) Decomposition of a_ee_des, full 6D ────────────────────
    # Torso PD output (constant ref → a_ff_t = 0 in A_swing).
    Kp_t_mat = np.diag(Kp_torso)
    Kd_t_mat = np.diag(Kd_torso)
    e_pos_t  = p_torso_ref - p_torso
    e_ori_t  = pin.log3(R_torso.T @ R_torso_ref)
    e_6d_t   = np.concatenate([e_pos_t, e_ori_t])
    v_torso_actual = J_torso @ v
    a_torso_des = (a_torso_ff
                   + Kp_t_mat @ e_6d_t
                   + Kd_t_mat @ (v_torso_ref - v_torso_actual))

    # v17 task-consistent FF.
    J_torso_pinv = np.linalg.pinv(J_torso, rcond=1e-8)
    a_ff_v17 = J_ee @ J_torso_pinv @ a_torso_des            # (6,)

    # SwingPlanner FF (linear+angular).
    a_ff_planner = a_ee_ff_planner.copy()                   # (6,)
    a_ff_total   = a_ff_planner + a_ff_v17                  # (6,)

    # PD term.
    Kp_ee_full = np.diag(np.concatenate([Kp_ee, Kp_ee_ang]))
    Kd_ee_full = np.diag(np.concatenate([Kd_ee, Kd_ee_ang]))
    e_pos_ee = p_ee_ref - p_ee
    e_ori_ee = pin.log3(R_ee.T @ R_ee_ref)
    e_6d_ee  = np.concatenate([e_pos_ee, e_ori_ee])
    v_ee_actual = J_ee @ v                                  # (6,)
    pd_term  = Kp_ee_full @ e_6d_ee + Kd_ee_full @ (v_ee_ref - v_ee_actual)

    a_ee_des = a_ff_total + pd_term                         # (6,)
    b_ee     = a_ee_des - Jdot_dq_ee                        # (6,)

    # ── (4) QP realization J_ee · qdd_qp, vs a_ee_des
    a_ee_qp_realized_no_jdq = J_ee @ qdd_full               # (6,)
    a_ee_qp_realized        = a_ee_qp_realized_no_jdq + Jdot_dq_ee
    qp_residual_6d          = a_ee_qp_realized - a_ee_des   # (6,)

    # ── (6) min-norm task-only solve qdd_task_6d = J_ee^+ @ b_ee
    J_ee_pinv = np.linalg.pinv(J_ee, rcond=1e-8)
    qdd_task_6d = J_ee_pinv @ b_ee                          # (20,)
    a_ee_task_alone = J_ee @ qdd_task_6d + Jdot_dq_ee       # (6,)

    # ── (7) qdd_mujoco_actual via finite-difference of v
    qdd_mujoco_actual = (v_post - v) / dt                   # (20,)

    # ── (8) Row-wise diff: J_ee · qdd_qp  vs  J_ee · qdd_mujoco
    a_ee_qp_only   = J_ee @ qdd_full                        # (6,)
    a_ee_mujoco    = J_ee @ qdd_mujoco_actual               # (6,)
    diff_qp_vs_mj  = a_ee_qp_only - a_ee_mujoco             # (6,)

    # qdd diff per joint (informative)
    qdd_qp_diff    = qdd_full - qdd_mujoco_actual

    # ── (9) tau_q vs saturation
    abs_tau     = np.abs(tau_qp)
    sat_frac    = abs_tau / tau_max
    sat_mask    = abs_tau >= 0.99 * tau_max
    n_sat       = int(np.sum(sat_mask))

    # Console summary
    print()
    print('=' * 72)
    print(f'EE 6D QP-vs-MuJoCo audit — A_swing, t = {t_rel:.4f} s, '
          f'dt = {dt:.3f} s, tau_max = {tau_max:.1f} Nm')
    print('=' * 72)
    print(f'(1) a_ee_des            = {_v6(a_ee_des)}')
    print(f'(2) Jdot_dq_ee          = {_v6(Jdot_dq_ee)}')
    print(f'(3) b_ee                = {_v6(b_ee)}')
    print(f'(4) J_ee@qdd_qp + Jdq   = {_v6(a_ee_qp_realized)}')
    print(f'    qp residual         = {_v6(qp_residual_6d)}')
    print(f'(5) e_pos_ee[mm]        = {_v6(e_pos_ee*1000)}, '
          f'e_ori_ee[deg] = {_v6(np.degrees(e_ori_ee))}')
    print(f'    omega_ee_actual     = {_v6(v_ee_actual[3:6])}, '
          f'v_ee_actual = {_v6(v_ee_actual[0:3])}')
    print(f'(6) qdd_task_6d realises{_v6(a_ee_task_alone)}')
    print(f'(7) qdd_mujoco (FD)     = {_v6(qdd_mujoco_actual)}')
    print(f'(8) J_ee·qdd_qp         = {_v6(a_ee_qp_only)}')
    print(f'    J_ee·qdd_mujoco     = {_v6(a_ee_mujoco)}')
    print(f'    diff (QP − MuJoCo)  = {_v6(diff_qp_vs_mj)}')
    print(f'(9) max|tau_q|/tau_max  = {sat_frac.max():.4f}, '
          f'n joints ≥ 0.99·tau_max = {n_sat}')

    # Markdown report
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        f.write('# M7 — EE 6D task: QP command vs MuJoCo execution '
                f'(A_swing, t = {t_rel:.3f} s)\n\n')
        f.write(f'Source: `{NPZ_PATH}`. Dump fired at `t_rel = '
                f'{t_rel:.6f} s`. Tick `dt_qp = {dt:.3f} s`. Joint torque '
                f'limit `tau_max = {tau_max:.1f} Nm`.\n')
        f.write('Reconstructions follow `crawlbot/solvers/wholebody_qp.py:540-621`.\n\n')

        f.write('## (1) Decomposition of `a_ee_des` (full 6D)\n\n')
        f.write('| component | linear (rows 0:3) | angular (rows 3:6) |\n')
        f.write('|---|---|---|\n')
        f.write(f'| `a_ff_ee[0:6]` (SwingPlanner) | `{_v6(a_ff_planner[:3])}` | `{_v6(a_ff_planner[3:])}` |\n')
        f.write(f'| `J_ee · J_torso^+ · a_torso_des` (v17 FF) | `{_v6(a_ff_v17[:3])}` | `{_v6(a_ff_v17[3:])}` |\n')
        f.write(f'| `Kp · e_ee + Kd · (v_ref − v_actual)` (PD) | `{_v6(pd_term[:3])}` | `{_v6(pd_term[3:])}` |\n')
        f.write(f'| **`a_ee_des` (sum)** | **`{_v6(a_ee_des[:3])}`** | **`{_v6(a_ee_des[3:])}`** |\n\n')
        f.write(f'For traceability: `a_torso_des = {_v6(a_torso_des)}`.\n\n')

        f.write('## (2) `Jdot_dq_ee` (full 6D)\n\n')
        f.write(f'`linear  = {_v6(Jdot_dq_ee[:3])}` [m/s²]\n\n')
        f.write(f'`angular = {_v6(Jdot_dq_ee[3:])}` [rad/s²]\n\n')

        f.write('## (3) `b_ee = a_ee_des − Jdot_dq_ee` (RHS of EE task)\n\n')
        f.write(f'`linear  = {_v6(b_ee[:3])}`\n\n')
        f.write(f'`angular = {_v6(b_ee[3:])}`\n\n')

        f.write('## (4) QP realisation\n\n')
        f.write(f'`J_ee · qdd_qp + Jdot_dq_ee = {_v6(a_ee_qp_realized)}`\n\n')
        f.write('Compared row-by-row to `a_ee_des` '
                f'(= `b_ee + Jdot_dq_ee`):\n\n')
        f.write('| row | a_ee_des | J_ee·qdd_qp + Jdq | residual |\n')
        f.write('|---|---|---|---|\n')
        labels = ['lin x', 'lin y', 'lin z', 'ang x', 'ang y', 'ang z']
        for i, lbl in enumerate(labels):
            f.write(f'| {lbl} | {a_ee_des[i]:+.6f} | '
                    f'{a_ee_qp_realized[i]:+.6f} | '
                    f'{qp_residual_6d[i]:+.6f} |\n')
        f.write(f'\n`‖residual_6d‖ = {np.linalg.norm(qp_residual_6d):.6e}`. '
                f'`‖residual_lin‖ = {np.linalg.norm(qp_residual_6d[:3]):.6e}`. '
                f'`‖residual_ang‖ = {np.linalg.norm(qp_residual_6d[3:]):.6e}`.\n\n')

        f.write('## (5) Reference errors and EE velocity (this tick)\n\n')
        f.write(f'- `e_ee_pos[0:3]` (mm) = `{_v6(e_pos_ee*1000)}`\n')
        f.write(f'- `e_ori[3:6]` (deg)   = `{_v6(np.degrees(e_ori_ee))}`\n')
        f.write(f'- `v_ee_actual[0:3]` (m/s)     = `{_v6(v_ee_actual[:3])}`\n')
        f.write(f'- `omega_ee_actual[3:6]` (rad/s)= `{_v6(v_ee_actual[3:])}`\n')
        f.write(f'- `v_ee_ref[0:3]` (m/s)         = `{_v6(v_ee_ref[:3])}`\n')
        f.write(f'- `omega_ee_ref[3:6]` (rad/s)   = `{_v6(v_ee_ref[3:])}`\n\n')

        f.write('## (6) Min-norm task-only solve `qdd_task_6d = J_ee^+ · (a_ee_des − Jdot_dq_ee)`\n\n')
        f.write('`rcond = 1e-8`. Reference only — the QP does not compute this.\n\n')
        f.write(f'`qdd_task_6d` (length 20):\n`{_v6(qdd_task_6d)}`\n\n')
        f.write(f'`J_ee · qdd_task_6d + Jdot_dq_ee` realises:\n')
        f.write(f'`linear  = {_v6(a_ee_task_alone[:3])}`,  '
                f'`angular = {_v6(a_ee_task_alone[3:])}`\n\n')

        f.write('## (7) `qdd_mujoco_actual` via finite-difference\n\n')
        f.write(f'Computed as `(v_post − v) / dt_qp = (v(t+dt) − v(t)) / {dt:.3f}`. '
                'v from `mujoco_to_pinocchio(qpos, qvel)` at the start of the '
                'next tick.\n\n')
        f.write(f'`qdd_mujoco_actual` (length 20):\n`{_v6(qdd_mujoco_actual)}`\n\n')

        f.write('## (8) Row-wise diff: `J_ee · qdd_qp` vs `J_ee · qdd_mujoco`\n\n')
        f.write('| row | J_ee·qdd_qp | J_ee·qdd_mujoco | diff (QP − MuJoCo) |\n')
        f.write('|---|---|---|---|\n')
        for i, lbl in enumerate(labels):
            f.write(f'| {lbl} | {a_ee_qp_only[i]:+.6f} | '
                    f'{a_ee_mujoco[i]:+.6f} | '
                    f'{diff_qp_vs_mj[i]:+.6f} |\n')
        f.write(f'\n`‖diff_6d‖ = {np.linalg.norm(diff_qp_vs_mj):.6e}`.  '
                f'`‖diff_lin‖ = {np.linalg.norm(diff_qp_vs_mj[:3]):.6e}`.  '
                f'`‖diff_ang‖ = {np.linalg.norm(diff_qp_vs_mj[3:]):.6e}`.\n\n')
        f.write('Per-DOF qdd diff `qdd_qp − qdd_mujoco_actual`:\n\n')
        f.write(f'base (6): `{_v6(qdd_qp_diff[:6])}`\n\n')
        f.write(f'joints (14): `{_v6(qdd_qp_diff[6:])}`\n\n')

        f.write('## (9) Joint torques vs saturation\n\n')
        f.write(f'`tau_max` = {tau_max:.1f} Nm (uniform across all 14 joints).\n')
        f.write(f'`max(|tau_q|) / tau_max` = `{sat_frac.max():.4f}`. '
                f'Joints at `≥ 0.99·tau_max`: `{n_sat}` of 14.\n\n')
        f.write('| joint idx | |tau_q| [Nm] | / tau_max |\n')
        f.write('|---|---|---|\n')
        for i in range(len(tau_qp)):
            mark = '  **(SAT)**' if sat_mask[i] else ''
            f.write(f'| {i} | {abs_tau[i]:.6f} | {sat_frac[i]:.6f} |{mark}\n')

    print(f'\nReport: {OUT_PATH}')


if __name__ == '__main__':
    main()
