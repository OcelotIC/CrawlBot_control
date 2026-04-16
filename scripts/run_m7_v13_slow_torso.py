#!/usr/bin/env python3
"""M7 v13: closed-loop run with r_com_displacement_scale = 0.5.

Hypothesis (per cascade bisection, 2026-04-16): the residual closed-loop
EE error after the delta_dot fix (v12: 137 mm SS peak) is dominated by
the torso (driven by the pre-planner CoM trajectory through the mapping)
moving too fast for the swing arm to track. Scaling the planned r_com
DISPLACEMENT (not duration) by 0.5 makes the torso traverse only half
the planned ground over T_step, giving the arm more headroom.

Output: results/M7_1pct_1step_v13/.
"""
from __future__ import annotations

import os
import sys

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

# Patch _make_m7_config to set r_com_displacement_scale=0.5.
import scripts.run_m7_single_step as r
_orig = r._make_m7_config


def _make_m7_v13_config():
    cfg = _orig()
    cfg.r_com_displacement_scale = 0.5
    return cfg


r._make_m7_config = _make_m7_v13_config


if __name__ == '__main__':
    r.run_case('1pct 1-step v13 (r_com_disp_scale=0.5)',
               'results/M7_1pct_1step_v13', n_steps=1)
