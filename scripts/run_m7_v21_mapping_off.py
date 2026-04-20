#!/usr/bin/env python3
"""M7 v21 single-step with mapping_bypass_in_ss=True.

Mirrors scripts/run_m7_v21_preplanner_cruise.py exactly except for the
new SimConfig flag mapping_bypass_in_ss=True. With the flag on, sim_loop
freezes the linear torso reference at its SS-entry value during SS only;
angular reference still comes from TorsoPlanner. DS phase is unchanged.

Output: results/M7_1pct_1step_v21_mapping_off/.
"""
from __future__ import annotations

import os
import sys

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import scripts.run_m7_single_step as r


if __name__ == '__main__':
    cfg = r._make_m7_config()
    cfg.preplanner_a_cruise_max = 0.01
    cfg.preplanner_cruise_ramp_frac = 0.2
    cfg.mapping_bypass_in_ss = True
    r.run_case(
        '1pct 1-step v21 (mapping_bypass_in_ss=True)',
        'results/M7_1pct_1step_v21_mapping_off',
        n_steps=1, config=cfg)
