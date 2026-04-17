#!/usr/bin/env python3
"""M7 v15: torso_early_finish_fraction=0.6 + mapping bypass during hold.

Same knob as v14, but with the v15 sim_loop change: during the hold
window (t_local > 0.6*T_step), the mapping is bypassed and the torso
reference is the TorsoPlanner's terminal-pose hold (p_end, R_end, v=0,
a=0). The swing arm sees a truly stationary base for its precision
approach.

For t_local <= 0.6*T_step the active flow is unchanged (mapping on,
linear from r_com_ref via the mass-weighted identity).

Output: results/M7_1pct_1step_v15/.
"""
from __future__ import annotations

import os
import sys

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import scripts.run_m7_single_step as r
_orig = r._make_m7_config


def _make_m7_v15_config():
    cfg = _orig()
    cfg.torso_early_finish_fraction = 0.6
    return cfg


r._make_m7_config = _make_m7_v15_config


if __name__ == '__main__':
    r.run_case('1pct 1-step v15 (ff=0.6 + mapping bypass during hold)',
               'results/M7_1pct_1step_v15', n_steps=1)
