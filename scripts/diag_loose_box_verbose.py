#!/usr/bin/env python3
"""Phase-1c FOLLOW-UP — why the loose-active box (h_max=10) fails while C (h_max=5)
converges. ZERO crawlbot/ change (monkeypatch only).

*** SUPERSEDED / UNFAITHFUL — kept only for the methodology record. ***
This re-solve rebuilds a FRESH CoarsePrePlanner and solves the captured step-0 `kw`
WITHOUT the sim's warm-start / initial guess. Its POSITIVE CONTROL (h_max=5.0) FAILS
here (lands at cost ~193, "local infeasibility") even though the REAL sim converges at
h_max=5 (cost 11.205, canon5_control.log). So this harness does NOT faithfully
reproduce the sim's solve and CANNOT characterize the box-value sensitivity. The
faithful test is the REAL-path pair: diag_nominal_shw.py (h_max=5 -> succeeds, docks
match C) vs diag_loose_box.py (h_max=10 -> step-0 fails), with the verbatim IPOPT EXIT
from diag_loose_box_realverbose.py. See PHASE1C_LOOSE_BOX_DIAGNOSIS.md.

The loose-active-box run (diag_loose_box.py, h_max_tight=10, enforce=True, else
canonical) FAILED at the step-0 pre-planner with the same Opti::solve error as the
storage-off case — even though canonical C (h_max_tight=5) docks all 6 steps. Since a
LARGER box cannot shrink the feasible set, a box=5-converges / box=10-fails split can
only be a local-solver / numerical sensitivity of the pre-planner NLP to the box VALUE.

This script isolates that: it captures the step-0 pre-planner solve's boundary
conditions (kw: c_const, r/v/L bc, T guess — independent of h_max), then re-solves the
SAME problem at a sweep of h_max values, everything else identical. h_max=5.0 is the
POSITIVE CONTROL — it must succeed (== canonical C), proving (a) the harness is valid
and (b) the h_max=10 failure is real and specifically about the box value. Then a verbose
(print_level=5) re-solve at h_max=10 prints the verbatim IPOPT EXIT status, and the
last-iterate h_w classifies unbounded vs local-solver (as in Task 2).

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_loose_box_verbose.py
"""
import json
import runpy
import sys

import numpy as np

import scripts.run_m7_single_step as r_single
from crawlbot.planning.coarse_preplanner import (
    CoarsePrePlanner, CoarsePrePlannerConfig)

OUT = 'results/j2_ablation_envelope/loose10_sweep.json'
SWEEP = [5.0, 6.0, 7.0, 8.0, 10.0, 20.0, 100.0, 1.0e6]   # 5.0 = positive control (== C)

# make the FIRST (captured) solve the loose h_max=10 case (so kw is the step-0 problem)
_orig_make = r_single._make_m7_config


def _mk(*a, **k):
    c = _orig_make(*a, **k)
    c.h_max_tight = np.full(3, 10.0)
    return c


r_single._make_m7_config = _mk

_done = {}
_orig_solve = CoarsePrePlanner.solve


def _hw_of(result, kw, m):
    """last-iterate wheel momentum h_w = c_const - L - r x (m v), per knot."""
    return np.array([kw['c_const'] - L - np.cross(r, m * v)
                     for r, v, L in zip(result.r_com, result.v_com, result.L_com)])


def _patched_solve(self, **kw):
    if _done:
        return _orig_solve(self, **kw)
    _done['x'] = True
    m = float(self.config.robot_mass)
    base = dict(self.config.__dict__)
    print('\n' + '=' * 78)
    print('PHASE-1C SWEEP — step-0 pre-planner vs storage-box value (all else identical):')
    print(f'  robot_mass={m:g}  captured h_max(in)={float(np.ravel(kw["h_max"])[0]):g}')
    print(f'  {"h_max":>10} | {"success":>7} | {"|h_w|inf":>9} | {"|L|inf":>7} | {"cost":>10} | status')
    rows = []
    for H in SWEEP:
        d = dict(base); d['h_max'] = np.full(3, H)
        d['ipopt_print_level'] = 0; d['ipopt_max_iter'] = 1000
        p = CoarsePrePlanner(CoarsePrePlannerConfig(**d)); p.build()
        kw2 = dict(kw); kw2['h_max'] = np.full(3, H)
        r = p.solve(**kw2)
        hw = _hw_of(r, kw2, m)
        row = dict(h_max=H, success=bool(r.success),
                   hw_inf=float(np.max(np.abs(hw))),
                   L_inf=float(np.max(np.linalg.norm(r.L_com, axis=1))),
                   cost=(float(r.cost) if r.cost is not None else None),
                   status=str(r.status))
        rows.append(row)
        st = row['status'] if len(row['status']) < 46 else row['status'][:43] + '...'
        cost = f"{row['cost']:.3f}" if row['cost'] is not None else 'None'
        print(f'  {H:>10g} | {str(r.success):>7} | {row["hw_inf"]:>9.3f} | '
              f'{row["L_inf"]:>7.3f} | {cost:>10} | {st}')

    # verbatim IPOPT EXIT for the failing h_max=10, and the passing h_max=5
    print('\n' + '-' * 78)
    for H in (10.0, 5.0):
        print(f'--- verbose IPOPT (print_level=5) at h_max={H:g} '
              f'({"FAILING" if H == 10.0 else "POSITIVE CONTROL"}): ---')
        d = dict(base); d['h_max'] = np.full(3, H)
        d['ipopt_print_level'] = 5; d['ipopt_max_iter'] = 1000
        p = CoarsePrePlanner(CoarsePrePlannerConfig(**d)); p.build()
        kw2 = dict(kw); kw2['h_max'] = np.full(3, H)
        r = p.solve(**kw2)
        hw = _hw_of(r, kw2, m)
        print(f'  => h_max={H:g}: success={r.success}  |h_w|inf={np.max(np.abs(hw)):.4e}  '
              f'|L|inf={np.max(np.linalg.norm(r.L_com, axis=1)):.3e}  cost={r.cost}')
        print('-' * 78)

    json.dump(rows, open(OUT, 'w'), indent=2)
    print(f'wrote {OUT}')
    print('=' * 78)
    sys.exit(0)


CoarsePrePlanner.solve = _patched_solve

sys.argv = [
    'diag_cooperative_arms.py',
    '--ss-two-task', '--ss-alpha-mom', '5000', '--alpha-torso-pose', '24000',
    '--ss-kp-torso', '3', '--ss-kd-torso', '2.5',
    '--aocs_mode', 'legacy_pid_numerical', '--K_omega', '50', '--qp-envelope-exact',
    '--interstep-settle-alpha-wrench', '3', '--interstep-settle-epsilon-v', '5e-3',
    '--n-steps', '1', '--envelope-constraint', 'on', '--out-dir', '_loose_verbose',
]
runpy.run_module('scripts.diag_cooperative_arms', run_name='__main__')
