#!/usr/bin/env python3
"""TASK 2 (ablation Phase-1b): diagnose the storage-off pre-planner failure.
ZERO crawlbot/ change — this script monkeypatches CoarsePrePlanner.solve (class
attribute) to, on the FIRST (failing) storage-off solve, re-solve the SAME
problem with ipopt print_level=5 + max_iter=1000, then dump the IPOPT status and
the last-iterate h_w trajectory so we can classify: (a) unbounded (h_w/objective
blowing up) vs (b) local-solver/restoration failure (iterates bounded but stuck).
It reuses the diag's full canonical config via runpy + --envelope-constraint
storage-off (DIAG B: enforce_hw_conservation=False + h_max_tight=1e6, rate cap kept).

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_preplanner_storage.py
"""
import runpy
import sys
import numpy as np
from crawlbot.planning.coarse_preplanner import (
    CoarsePrePlanner, CoarsePrePlannerConfig)

_done = {}
_orig_solve = CoarsePrePlanner.solve


def _patched_solve(self, **kw):
    result = _orig_solve(self, **kw)          # the storage-off solve (expected: fail)
    if not _done:
        _done['x'] = True
        m = float(self.config.robot_mass)
        print('\n' + '=' * 78)
        print('TASK 2 — storage-off pre-planner FIRST solve (as run by the sim):')
        print(f'  success={result.success}  status={result.status!r}')
        print(f'  cost={result.cost}  iter_count={result.iter_count}  h_max(in)='
              f'{float(np.ravel(kw["h_max"])[0]):g}')
        # last-iterate h_w from opti.debug values already in `result`
        hw = np.array([kw['c_const'] - L - np.cross(r, m * v)
                       for r, v, L in zip(result.r_com, result.v_com, result.L_com)])
        print('  last-iterate trajectory magnitudes:')
        print(f'    |h_w|inf per-knot max = {np.max(np.abs(hw)):.4e} Nms  '
              f'(box was ±{float(np.ravel(kw["h_max"])[0]):g})')
        print(f'    |r_com|max={np.max(np.linalg.norm(result.r_com, axis=1)):.3e}  '
              f'|v_com|max={np.max(np.linalg.norm(result.v_com, axis=1)):.3e}  '
              f'|L_com|max={np.max(np.linalg.norm(result.L_com, axis=1)):.3e}')
        print(f'    h_w inf-norm per knot: '
              f'{np.round(np.max(np.abs(hw), axis=1), 3).tolist()}')

        # Re-solve the SAME problem with a verbose IPOPT for the return_status.
        print('\n' + '-' * 78)
        print('Re-solve with ipopt print_level=5, max_iter=1000 (verbose EXIT status):')
        d = dict(self.config.__dict__)
        d['ipopt_print_level'] = 5
        d['ipopt_max_iter'] = 1000
        p2 = CoarsePrePlanner(CoarsePrePlannerConfig(**d))
        p2.build()
        r2 = p2.solve(**kw)
        hw2 = np.array([kw['c_const'] - L - np.cross(r, m * v)
                        for r, v, L in zip(r2.r_com, r2.v_com, r2.L_com)])
        print('-' * 78)
        print(f'  RE-SOLVE: success={r2.success}  status={r2.status!r}  '
              f'cost={r2.cost}  iter={r2.iter_count}')
        print(f'  RE-SOLVE last-iterate |h_w|inf max = {np.max(np.abs(hw2)):.4e}  '
              f'|L_com|max={np.max(np.linalg.norm(r2.L_com, axis=1)):.3e}')
        print('=' * 78)
        sys.exit(0)
    return result


CoarsePrePlanner.solve = _patched_solve

sys.argv = [
    'diag_cooperative_arms.py',
    '--ss-two-task', '--ss-alpha-mom', '5000', '--alpha-torso-pose', '24000',
    '--ss-kp-torso', '3', '--ss-kd-torso', '2.5',
    '--aocs_mode', 'legacy_pid_numerical', '--K_omega', '50', '--qp-envelope-exact',
    '--interstep-settle-alpha-wrench', '3', '--interstep-settle-epsilon-v', '5e-3',
    '--n-steps', '1', '--envelope-constraint', 'storage-off', '--out-dir', '_task2_diag',
]
runpy.run_module('scripts.diag_cooperative_arms', run_name='__main__')
