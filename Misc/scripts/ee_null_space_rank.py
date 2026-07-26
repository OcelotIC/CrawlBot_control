"""Offline linear-algebra audit for the EE angular task vs torso null
space at t=3.6 s of A_swing.

Loads Misc/runs/M7_ee_ori_diag/A_swing_t3p6.npz (dumped by
Misc/scripts/bisect_qp_cascade.py:run_v21_case) and writes
Misc/runs/M7_ee_ori_diag/ee_null_space_rank.md. No interpretation.
"""
from __future__ import annotations

import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)

import numpy as np


NPZ_PATH = os.path.join(_root, 'Misc', 'runs', 'M7_ee_ori_diag',
                        'A_swing_t3p6.npz')
OUT_PATH = os.path.join(_root, 'Misc', 'runs', 'M7_ee_ori_diag',
                        'ee_null_space_rank.md')

# Numerical-rank tolerance — np.linalg.matrix_rank's default if tol=None
# is max(M.shape) * eps(svmax). We override with a single explicit value
# applied uniformly across all rank computations so the report is
# self-contained and reproducible.
RANK_TOL = 1e-8


def _rank(M, tol=RANK_TOL):
    return int(np.linalg.matrix_rank(M, tol=tol))


def _svals(M):
    return np.linalg.svd(M, compute_uv=False)


def _mat_str(M, indent='  ', prec=6):
    return '\n'.join(
        indent + '  '.join(f'{M[i, j]:+.{prec}f}' for j in range(M.shape[1]))
        for i in range(M.shape[0]))


def _vec_str(v, prec=6):
    return '[' + ', '.join(f'{x:+.{prec}f}' for x in v) + ']'


def main():
    d = np.load(NPZ_PATH)
    t_rel = float(d['t_rel'])
    J_ee = d['J_ee']             # (6, 20)
    J_torso = d['J_torso']       # (6, 20)
    n = J_ee.shape[1]

    # ── (1) rank(J_ee[3:6, :]) — angular EE Jacobian ──────────────────
    J_ee_ang = J_ee[3:6, :]              # (3, 20)
    rank_J_ee_ang = _rank(J_ee_ang)
    sv_J_ee_ang = _svals(J_ee_ang)

    # ── Build N_torso in qdd space: I − J_torso^+ J_torso ─────────────
    # Damped pseudo-inverse with rcond=1e-8 to match wholebody_qp.py
    # (line 582, 620). N_torso here is (20, 20).
    J_torso_pinv = np.linalg.pinv(J_torso, rcond=1e-8)
    N_torso = np.eye(n) - J_torso_pinv @ J_torso
    rank_J_torso = _rank(J_torso)
    rank_N_torso = _rank(N_torso)

    # ── (2) rank(J_ee[3:6, :] @ N_torso) ──────────────────────────────
    M_proj = J_ee_ang @ N_torso          # (3, 20)
    rank_M_proj = _rank(M_proj)

    # ── (3) Singular values of M_proj — exactly three (3-row matrix) ──
    sv_M_proj = _svals(M_proj)

    # ── (4) Row-space intersection between J_ee_ang and J_torso ──────
    # Two equivalent measures:
    #   (i)  rank deficiency: dim(intersect) = rank(J_ee_ang) - rank(M_proj)
    #   (ii) stacked rank: dim(intersect) = rank(J_ee_ang) + rank(J_torso)
    #                                       - rank([J_ee_ang; J_torso])
    stacked = np.vstack([J_ee_ang, J_torso])
    rank_stacked = _rank(stacked)
    intersect_dim_i  = rank_J_ee_ang - rank_M_proj
    intersect_dim_ii = rank_J_ee_ang + rank_J_torso - rank_stacked

    # Per-row test: how much of each angular EE row lies in row(J_torso)?
    # Project each row onto row(J_torso) using J_torso^+ J_torso, and
    # report the fraction of row energy that survives projection.
    P_torso_row = J_torso_pinv @ J_torso       # (20, 20) projector onto row(J_torso)
    row_energy = []
    for i in range(3):
        r = J_ee_ang[i, :]
        n_r = float(np.linalg.norm(r))
        if n_r == 0:
            row_energy.append((0.0, 0.0, 0.0))
            continue
        r_in_torso  = r @ P_torso_row.T          # row in row(J_torso)
        r_perp      = r - r_in_torso             # row in null(J_torso)
        row_energy.append((
            n_r,
            float(np.linalg.norm(r_in_torso)) / n_r,
            float(np.linalg.norm(r_perp))    / n_r,
        ))

    # ── Write report ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        f.write('# M7 — EE angular task vs torso null space (A_swing, t = 3.6 s)\n\n')
        f.write(f'Source matrices: `{NPZ_PATH}` (dumped by '
                '`Misc/scripts/bisect_qp_cascade.py --case A_swing`).\n')
        f.write(f'Numerical-rank tolerance: `tol = {RANK_TOL:.0e}` '
                '(applied uniformly).\n')
        f.write(f'Damped pseudo-inverse: `rcond = 1e-8` '
                '(matches `wholebody_qp.py:582,620`).\n\n')

        f.write(f'**Tick:** `t_rel = {t_rel:.6f} s` of A_swing SS '
                f'(closest QP tick at or after the 3.6 s target).\n\n')
        f.write(f'**Shapes:** `J_ee = {J_ee.shape}`, '
                f'`J_torso = {J_torso.shape}`, '
                f'`J_ee_ang = J_ee[3:6, :] = {J_ee_ang.shape}`, '
                f'`N_torso = {N_torso.shape}`.\n\n')

        f.write('## (1) `rank(J_ee[3:6, :])`\n\n')
        f.write(f'`rank = {rank_J_ee_ang}` (out of max 3).\n\n')
        f.write('Singular values of `J_ee[3:6, :]`:\n')
        f.write(f'`{_vec_str(sv_J_ee_ang)}`\n\n')

        f.write('## (2) `rank(J_ee[3:6, :] @ N_torso)`\n\n')
        f.write(f'`rank = {rank_M_proj}` (out of max 3).\n\n')
        f.write('For reference: `rank(J_torso) = '
                f'{rank_J_torso}` (out of max 6); '
                f'`rank(N_torso) = {rank_N_torso}` (= n − rank(J_torso) = '
                f'{n} − {rank_J_torso} = {n - rank_J_torso}).\n\n')

        f.write('## (3) Singular values of `J_ee[3:6, :] @ N_torso`\n\n')
        f.write(f'`{_vec_str(sv_M_proj)}`\n\n')

        f.write('## (4) Row-space intersection between `J_ee[3:6, :]` and `J_torso`\n\n')
        f.write('Stacked-rank test:\n\n')
        f.write(f'- `rank(J_ee_ang)        = {rank_J_ee_ang}`\n')
        f.write(f'- `rank(J_torso)         = {rank_J_torso}`\n')
        f.write(f'- `rank([J_ee_ang; J_torso]) = {rank_stacked}`\n')
        f.write(f'- `dim(intersection)     = rank(J_ee_ang) + rank(J_torso) '
                f'− rank(stack) = {intersect_dim_ii}`\n\n')
        f.write('Null-space deficiency check '
                '(must agree with the stacked-rank test):\n\n')
        f.write(f'- `rank(J_ee_ang)            = {rank_J_ee_ang}`\n')
        f.write(f'- `rank(J_ee_ang @ N_torso)  = {rank_M_proj}`\n')
        f.write(f'- `dim(intersection)         = rank(J_ee_ang) − '
                f'rank(J_ee_ang @ N_torso) = {intersect_dim_i}`\n\n')
        f.write('Per-row decomposition into the row spaces of `J_torso` and `N_torso`:\n\n')
        f.write('| row | ‖row‖ | ‖proj onto row(J_torso)‖ / ‖row‖ | '
                '‖proj onto null(J_torso)‖ / ‖row‖ |\n')
        f.write('|---|---|---|---|\n')
        for i, (n_r, frac_in, frac_perp) in enumerate(row_energy):
            f.write(f'| J_ee[{3+i}, :] | {n_r:.6f} | '
                    f'{frac_in:.6f} | {frac_perp:.6f} |\n')

    # Console summary.
    print()
    print('=' * 70)
    print(f'EE angular vs torso-null-space audit (A_swing, t={t_rel:.4f} s)')
    print('=' * 70)
    print(f'(1) rank(J_ee[3:6, :])             = {rank_J_ee_ang}')
    print(f'(2) rank(J_ee[3:6, :] @ N_torso)   = {rank_M_proj}')
    print(f'(3) sv(J_ee[3:6, :] @ N_torso)     = {_vec_str(sv_M_proj)}')
    print(f'(4) dim(row(J_ee_ang) ∩ row(J_torso)) = {intersect_dim_ii}  '
          f'(= {intersect_dim_i} via null-space deficiency)')
    print(f'\nReport: {OUT_PATH}')


if __name__ == '__main__':
    main()
