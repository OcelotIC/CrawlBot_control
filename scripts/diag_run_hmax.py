#!/usr/bin/env python3
"""Phase-1e realized validation — full closed-loop traversal at a chosen storage budget.

Runs the canonical scenario with cfg.h_max_tight = HMAX_TIGHT (env var), everything else
canonical (tau_w_max=5, --envelope-constraint on). For h_max in {6,7} the pre-planner
converges at all 6 steps (Level-1 frontier), so the traversal completes and yields realized
Hdot_s / h_w / docks / theta_s at the higher budget/speed. ZERO crawlbot/ change (monkeypatch
of _make_m7_config only). Export the tidy CSV with export_figure_data.py afterwards.

Run: MUJOCO_GL=disabled PYTHONPATH=. HMAX_TIGHT=6 HMAX_OUTDIR=figC_hmax6 \
       python3 scripts/diag_run_hmax.py
"""
import os
import runpy
import sys

import numpy as np

import scripts.run_m7_single_step as r_single

H = float(os.environ['HMAX_TIGHT'])
OUT = os.environ.get('HMAX_OUTDIR', f'figC_hmax{int(H)}')

_orig_make = r_single._make_m7_config


def _mk(*a, **k):
    c = _orig_make(*a, **k)
    c.h_max_tight = np.full(3, H)          # only the storage budget changes
    return c


r_single._make_m7_config = _mk

sys.argv = [
    'diag_cooperative_arms.py',
    '--ss-two-task', '--ss-alpha-mom', '5000', '--alpha-torso-pose', '24000',
    '--ss-kp-torso', '3', '--ss-kd-torso', '2.5',
    '--aocs_mode', 'legacy_pid_numerical', '--K_omega', '50', '--qp-envelope-exact',
    '--interstep-settle-alpha-wrench', '3', '--interstep-settle-epsilon-v', '5e-3',
    '--n-steps', '6', '--envelope-constraint', 'on', '--out-dir', OUT,
]
print(f'[run-hmax] h_max_tight={H}  out-dir=results/{OUT}  (else canonical, tau_w_max=5)')
runpy.run_module('scripts.diag_cooperative_arms', run_name='__main__')
