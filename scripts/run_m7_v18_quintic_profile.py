#!/usr/bin/env python3
"""M7 v18: quintic torso profile (was trapezoidal in v11-v17).

v17 held constant v_ref with a_ff=0 for ~30% of T_step (trapezoidal
cruise), so the QP's torso task degenerated to pure position PD on a
moving reference for 2+ seconds. The arm-reaction torque was absorbed
as position lag, peaking at 120 mm. Switching to the quintic profile
(already implemented in TorsoPlanner._quintic_params) gives a smooth
a_ff everywhere except the instantaneous points tau in {0, 0.5, 1}.

Keeps v16/v17 behaviour: TorsoPlanner drives SS torso ref, mapping
drives DS, EE task-consistent feedforward on.

Output: results/M7_1pct_1step_v18/.
"""
from __future__ import annotations

import os
import sys

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import scripts.run_m7_single_step as r


if __name__ == '__main__':
    r.run_case('1pct 1-step v18 (quintic torso profile)',
               'results/M7_1pct_1step_v18', n_steps=1)
