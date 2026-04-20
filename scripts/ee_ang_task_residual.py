"""Offline analysis: EE angular task residual at A_swing t = 3.6 s.

Loads results/M7_ee_ori_diag/A_swing_t3p6.npz and computes the
quantities listed in the task spec. Writes
results/M7_ee_ori_diag/ee_ang_task_residual.md. No interpretation.
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
                        'ee_ang_task_residual.md')


def _vec_str(v, prec=6):
    return '[' + ', '.join(f'{x:+.{prec}f}' for x in v) + ']'


def main():
    d = np.load(NPZ_PATH)
    t_rel = float(d['t_rel'])

    # Matrices and state ─────────────────────────────────────────────
    J_ee     = d['J_ee']                # (6, 20)
    J_torso  = d['J_torso']             # (6, 20)
    Jdot_dq_ee     = d['Jdot_dq_ee']    # (6,)
    Jdot_dq_torso  = d['Jdot_dq_torso'] # (6,)
    v        = d['v']                   # (20,) Pinocchio velocity

    # EE actuals + reference at this tick
    R_ee     = d['R_ee']
    R_ee_ref = d['R_ee_ref']
    p_ee     = d['p_ee']
    p_ee_ref = d['p_ee_ref']
    v_ee_ref = d['v_ee_ref']            # (6,)
    a_ee_ff_planner = d['a_ee_ff']      # (6,) — SwingPlanner FF, no v17 term yet

    # Torso actuals + (constant) reference
    R_torso     = d['R_torso']
    R_torso_ref = d['R_torso_ref']
    p_torso     = d['p_torso']
    p_torso_ref = d['p_torso_ref']
    v_torso_ref = d['v_torso_ref']      # zeros for A_swing
    a_torso_ff  = d['a_torso_ff']       # zeros for A_swing

    # QP gains (vectors)
    Kp_torso = d['Kp_torso']            # (6,)
    Kd_torso = d['Kd_torso']            # (6,)
    Kp_ee     = d['Kp_ee']              # (3,)
    Kd_ee     = d['Kd_ee']              # (3,)
    Kp_ee_ang = d['Kp_ee_ang']          # (3,)
    Kd_ee_ang = d['Kd_ee_ang']          # (3,)

    # Actual QP solution at this tick
    qdd_t_qp  = d['qdd_t_qp']           # (6,)
    qdd_qp    = d['qdd_qp']             # (14,)
    qdd_full  = np.concatenate([qdd_t_qp, qdd_qp])   # (20,)

    # ── (1) J_ee_ang_dot_dq = (J̇_ee · q̇)[3:6] ────────────────────
    J_ee_ang_dot_dq = Jdot_dq_ee[3:6].copy()        # (3,)

    # ── (2) omega_ee_actual = (J_ee · v)[3:6] ─────────────────────
    v_ee_actual = J_ee @ v                          # (6,)
    omega_ee_actual = v_ee_actual[3:6].copy()       # (3,)

    # ── (3) Reconstruct a_ee_des[3:6] ──────────────────────────────
    # First reconstruct a_torso_des exactly as wholebody_qp.py:540-555.
    Kp_t_mat = np.diag(Kp_torso)
    Kd_t_mat = np.diag(Kd_torso)
    e_pos_t  = p_torso_ref - p_torso
    e_ori_t  = pin.log3(R_torso.T @ R_torso_ref)
    e_6d_t   = np.concatenate([e_pos_t, e_ori_t])
    v_torso_actual = J_torso @ v
    a_torso_des = (a_torso_ff
                   + Kp_t_mat @ e_6d_t
                   + Kd_t_mat @ (v_torso_ref - v_torso_actual))

    # v17 task-consistent FF term: J_ee · J_torso^+ · a_torso_des
    # (wholebody_qp.py:619-621, rcond=1e-8).
    J_torso_pinv = np.linalg.pinv(J_torso, rcond=1e-8)
    a_ff_v17 = J_ee @ J_torso_pinv @ a_torso_des     # (6,)
    a_ff_ee_full = a_ee_ff_planner + a_ff_v17        # (6,) — what QP sees

    # EE PD: see wholebody_qp.py:596-610.
    Kp_ee_full = np.diag(np.concatenate([Kp_ee, Kp_ee_ang]))
    Kd_ee_full = np.diag(np.concatenate([Kd_ee, Kd_ee_ang]))
    e_pos_ee = p_ee_ref - p_ee
    e_ori_ee = pin.log3(R_ee.T @ R_ee_ref)
    e_6d_ee  = np.concatenate([e_pos_ee, e_ori_ee])

    a_ee_des_6 = (a_ff_ee_full
                  + Kp_ee_full @ e_6d_ee
                  + Kd_ee_full @ (v_ee_ref - v_ee_actual))     # (6,)
    a_ee_des_ang = a_ee_des_6[3:6].copy()                       # (3,)

    # ── (4) Standalone LSQ: J_ee_ang · qdd ≈ a_ee_des_ang − Jdq_ee_ang
    A = J_ee[3:6, :]                                # (3, 20)
    b = a_ee_des_ang - J_ee_ang_dot_dq              # (3,)
    qdd_lsq, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    residual_lsq = A @ qdd_lsq - b                  # (3,)
    residual_lsq_norm = float(np.linalg.norm(residual_lsq))

    # ── (5) Plug actual QP qdd into the same expression ──────────────
    a_ee_ang_qp = J_ee[3:6, :] @ qdd_full + J_ee_ang_dot_dq   # (3,)
    diff_qp = a_ee_ang_qp - a_ee_des_ang                       # (3,)
    diff_qp_norm = float(np.linalg.norm(diff_qp))

    # Console summary
    print()
    print('=' * 70)
    print(f'EE angular task residual — A_swing, t = {t_rel:.4f} s')
    print('=' * 70)
    print(f'(1) J_ee_ang_dot_dq          = {_vec_str(J_ee_ang_dot_dq)}  rad/s²')
    print(f'(2) omega_ee_actual          = {_vec_str(omega_ee_actual)}  rad/s')
    print(f'(3) a_ee_des[3:6]            = {_vec_str(a_ee_des_ang)}  rad/s²')
    print(f'    (a_ff_v17[3:6])          = {_vec_str(a_ff_v17[3:6])}  rad/s²')
    print(f'    (a_ee_ff_planner[3:6])   = {_vec_str(a_ee_ff_planner[3:6])}  rad/s²')
    print(f'(4) qdd_lsq                  = {_vec_str(qdd_lsq)}  rad/s²')
    print(f'    residual_lsq             = {_vec_str(residual_lsq)}')
    print(f'    ‖residual_lsq‖           = {residual_lsq_norm:.6e}')
    print(f'(5) J_ee[3:6]·qdd_qp + Jdq_ee[3:6] = {_vec_str(a_ee_ang_qp)}')
    print(f'    diff vs a_ee_des[3:6]    = {_vec_str(diff_qp)}')
    print(f'    ‖diff‖                   = {diff_qp_norm:.6e}')

    # ── Markdown report ────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        f.write('# M7 — EE angular task residual (A_swing, t = 3.6 s)\n\n')
        f.write(f'Source: `{NPZ_PATH}`. All values at the closest QP tick to '
                f't = 3.6 s, observed at `t_rel = {t_rel:.6f} s`.\n')
        f.write('Reconstructions follow `crawlbot/solvers/wholebody_qp.py`:\n'
                '`a_torso_des` from lines 540-555, `a_ee_des` from 596-610,\n'
                '`v17` task-consistent FF from 619-621 (`rcond = 1e-8`).\n\n')

        f.write('## (1) `J_ee_ang_dot_dq = (J̇_ee · q̇)[3:6]`\n\n')
        f.write(f'`{_vec_str(J_ee_ang_dot_dq)}`  [rad/s²]\n\n')

        f.write('## (2) `omega_ee_actual = (J_ee · q̇)[3:6]`\n\n')
        f.write(f'`{_vec_str(omega_ee_actual)}`  [rad/s]\n\n')
        f.write(f'For reference, `omega_ee_ref = v_ee_ref[3:6] = '
                f'{_vec_str(v_ee_ref[3:6])}` rad/s.\n\n')

        f.write('## (3) `a_ee_des[3:6]` reconstruction\n\n')
        f.write('Components fed into the QP at this tick:\n\n')
        f.write(f'- SwingPlanner `a_ee_ff[3:6]` (alpha):  `{_vec_str(a_ee_ff_planner[3:6])}`\n')
        f.write(f'- v17 task-consistent FF `J_ee · J_torso^+ · a_torso_des` (angular rows):\n'
                f'  `{_vec_str(a_ff_v17[3:6])}`\n')
        f.write(f'- `a_torso_des = Kp_t·e_6d_t − Kd_t·(J_torso·q̇)` '
                f'(constant ref, A_swing):\n'
                f'  `{_vec_str(a_torso_des)}`\n')
        f.write(f'- Orientation error `e_ori_ee = log3(R_ee^T R_ee_ref)` '
                f'[rad]: `{_vec_str(e_ori_ee)}`\n')
        f.write(f'- `omega_ref − omega_actual` [rad/s]: '
                f'`{_vec_str(v_ee_ref[3:6] - omega_ee_actual)}`\n')
        f.write(f'- `Kp_ee_ang = {_vec_str(Kp_ee_ang)}`,  '
                f'`Kd_ee_ang = {_vec_str(Kd_ee_ang)}`\n\n')
        f.write(f'**`a_ee_des[3:6] = {_vec_str(a_ee_des_ang)}`** [rad/s²].\n\n')
        f.write('Full 6D for completeness: '
                f'`a_ee_des = {_vec_str(a_ee_des_6)}`.\n\n')

        f.write('## (4) Standalone least-squares — angular EE task alone\n\n')
        f.write('Solve `min_x ‖J_ee[3:6,:] · x + J_ee_ang_dot_dq − a_ee_des[3:6]‖²` '
                'over `x ∈ R²⁰`, no other constraints.\n\n')
        f.write('Per `np.linalg.lstsq(A, b, rcond=None)` with `A = J_ee[3:6,:]`, '
                '`b = a_ee_des[3:6] − J_ee_ang_dot_dq`:\n\n')
        f.write(f'`qdd_lsq` (length 20):\n`{_vec_str(qdd_lsq)}`\n\n')
        f.write(f'Residual `A·qdd_lsq − b`: `{_vec_str(residual_lsq)}`\n\n')
        f.write(f'**‖residual‖ = {residual_lsq_norm:.6e}**\n\n')

        f.write('## (5) Actual QP solution plugged back in\n\n')
        f.write('Reconstruction of the angular EE task acceleration the QP '
                'realises with its own `qdd`:\n\n')
        f.write('`J_ee[3:6, :] · qdd_qp + J_ee_ang_dot_dq` = '
                f'`{_vec_str(a_ee_ang_qp)}` rad/s²\n\n')
        f.write(f'Difference from the desired `a_ee_des[3:6]`:\n\n')
        f.write(f'`diff = {_vec_str(diff_qp)}`\n\n')
        f.write(f'**`‖diff‖ = {diff_qp_norm:.6e}` rad/s²**\n\n')

        f.write('## Auxiliary numerics (for traceability)\n\n')
        f.write(f'`qdd_t_qp` (base): `{_vec_str(qdd_t_qp)}`\n\n')
        f.write(f'`qdd_qp` (joints, 14): `{_vec_str(qdd_qp)}`\n\n')
        f.write(f'`Kp_torso` = `{_vec_str(Kp_torso)}`,  '
                f'`Kd_torso` = `{_vec_str(Kd_torso)}`\n\n')
        f.write(f'`v_ee_ref` = `{_vec_str(v_ee_ref)}` (linear; angular)\n')

    print(f'\nReport: {OUT_PATH}')


if __name__ == '__main__':
    main()
