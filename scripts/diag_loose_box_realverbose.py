#!/usr/bin/env python3
"""Phase-1c — verbatim IPOPT EXIT for the REAL-path loose-box (h_max=10) step-0 failure.

The loose-active box (h_max_tight=10, enforce=True, else canonical) fails at the step-0
pre-planner in the real sim, while canonical h_max=5 succeeds there (canon5_control.log).
This captures the REAL solve's IPOPT EXIT status (real warm-start, NOT the warm-start-free
re-solve sweep, which was unfaithful). ZERO crawlbot/ change: patches _make_m7_config
(h_max=10) and wraps CoarsePrePlanner.build to force ipopt_print_level=5 / max_iter=1000.

The run crashes downstream (diag instrumentation records q_start after the skipped step),
but the verbose IPOPT EXIT is printed DURING the solve, before that — captured in the log.

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_loose_box_realverbose.py
"""
import runpy
import sys

import numpy as np

import scripts.run_m7_single_step as r_single
from crawlbot.planning.coarse_preplanner import CoarsePrePlanner

_orig_make = r_single._make_m7_config


def _mk(*a, **k):
    c = _orig_make(*a, **k)
    c.h_max_tight = np.full(3, 10.0)
    return c


r_single._make_m7_config = _mk

_orig_build = CoarsePrePlanner.build


def _build(self, *a, **k):
    self.config.ipopt_print_level = 5
    self.config.ipopt_max_iter = 1000
    return _orig_build(self, *a, **k)


CoarsePrePlanner.build = _build

sys.argv = [
    'diag_cooperative_arms.py',
    '--ss-two-task', '--ss-alpha-mom', '5000', '--alpha-torso-pose', '24000',
    '--ss-kp-torso', '3', '--ss-kd-torso', '2.5',
    '--aocs_mode', 'legacy_pid_numerical', '--K_omega', '50', '--qp-envelope-exact',
    '--interstep-settle-alpha-wrench', '3', '--interstep-settle-epsilon-v', '5e-3',
    '--n-steps', '1', '--envelope-constraint', 'on', '--out-dir', '_loose_realverbose',
]
print('[loose realverbose] h_max_tight=10, enforce=True, ipopt_print_level=5 (REAL path)')
try:
    runpy.run_module('scripts.diag_cooperative_arms', run_name='__main__')
except SystemExit:
    pass
except Exception as e:  # noqa: BLE001 — downstream diag crash AFTER the verbose solve prints
    print(f'[loose realverbose] downstream crash after solve (expected): {type(e).__name__}: {e}')
