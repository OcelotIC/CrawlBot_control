#!/usr/bin/env python3
"""M7 v14: closed-loop run with torso_early_finish_fraction = 0.6.

The torso quintic completes at t = 0.6 * T_step, then HOLDS the terminal
torso pose (zero velocity, zero acceleration) through the remaining 40%
of T_step. The swing planner is unchanged: it runs over the full
[0, T_step]. This gives the swing arm a precision-approach window
against a stationary base — the regime that B_const_torso (cascade
bisection 2026-04-16) showed tracks to 13.5 mm peak EE error.

The same knob (cfg.torso_early_finish_fraction) is read both by
TorsoPlanner.add_phase (where the quintic clamps σ to 1 after
0.6*T_step via effective_duration = 0.6*duration) and by sim_loop's
coarse-plan CoM-trajectory time warp (so the NMPC's r_com_ref also
finishes at 0.6*T_step and holds — keeping mapping consistent with
the torso reference).

Both trajectories remain synchronized over the same T_step. This is
not gain scheduling — only the time profile of the torso/CoM ref is
compressed; the gains and task weights are unchanged.

Output: Misc/runs/M7_1pct_1step_v14/.
"""
from __future__ import annotations

import os
import sys

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import scripts.run_m7_single_step as r
_orig = r._make_m7_config


def _make_m7_v14_config():
    cfg = _orig()
    cfg.torso_early_finish_fraction = 0.6
    return cfg


r._make_m7_config = _make_m7_v14_config


if __name__ == '__main__':
    r.run_case('1pct 1-step v14 (torso_early_finish_fraction=0.6)',
               'Misc/runs/M7_1pct_1step_v14', n_steps=1)
