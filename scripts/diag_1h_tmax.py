#!/usr/bin/env python3
"""Phase 1h addendum — does an EXTENDED execution window (t_hold_max=T_MAX) let the h_max=6
closed loop close the 5 mm dock gate, or does the EE plateau/drift above 5 mm forever?
ZERO crawlbot/ change (monkeypatch of _make_m7_config: h_max=6 + t_hold_max=T_MAX).

Run: MUJOCO_GL=disabled PYTHONPATH=. T_HOLD_MAX=200 python3 scripts/diag_1h_tmax.py
"""
import os
import runpy
import sys

import numpy as np

import scripts.run_m7_single_step as r_single

THOLD = float(os.environ.get('T_HOLD_MAX', '200'))

_orig_make = r_single._make_m7_config


def _mk(*a, **k):
    c = _orig_make(*a, **k)
    c.h_max_tight = np.full(3, 6.0)
    c.t_hold_max = THOLD          # extend the convergence-hold window (execution-side lever)
    return c


r_single._make_m7_config = _mk

sys.argv = [
    'diag_cooperative_arms.py',
    '--ss-two-task', '--ss-alpha-mom', '5000', '--alpha-torso-pose', '24000',
    '--ss-kp-torso', '3', '--ss-kd-torso', '2.5',
    '--aocs_mode', 'legacy_pid_numerical', '--K_omega', '50', '--qp-envelope-exact',
    '--interstep-settle-alpha-wrench', '3', '--interstep-settle-epsilon-v', '5e-3',
    '--n-steps', '1', '--envelope-constraint', 'on', '--out-dir', 'figC_hmax6_tmax',
]
print(f'[1h-tmax] h_max=6, t_hold_max={THOLD}s (extended execution window); step 0')
runpy.run_module('scripts.diag_cooperative_arms', run_name='__main__')
