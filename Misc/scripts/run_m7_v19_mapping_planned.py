#!/usr/bin/env python3
"""M7 v19: mapping-based torso ref in SS, fed q_planned not q_current.

Configuration:
  - TorsoPlanner: quintic profile (v18 default).
  - SwingPlanner: quintic horizontal + asymmetric bump (unchanged).
  - Mapping active in SS and DS (v12-style), but in SS the mapping is
    fed q_planned (quintic interp between step-start and step-end arm
    configurations) and dq_planned = d/dt of that interp. This makes
    the delta(q)/m_b and delta_dot/m_b corrections FEEDFORWARD
    anticipations of the planned arm motion, not reactive feedback on
    arm tracking error.
  - DS: mapping still uses q_current (actual settle state).
  - EE task-consistent feedforward (v17) still on.

Hypothesis: v12 with mapping + delta_dot + q_current peaked at 22 mm
torso pos. v16..v18 dropped mapping in SS and peaked at 120..140 mm
because the kinematic torso reference didn't compensate for arm-
induced base drift. v19 restores the mapping in SS while making it
purely feedforward (not stealing from arm PD residual).

Output: Misc/runs/M7_1pct_1step_v19/.
"""
from __future__ import annotations

import os
import sys

_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import scripts.run_m7_single_step as r


if __name__ == '__main__':
    r.run_case('1pct 1-step v19 (SS mapping + q_planned + quintic torso)',
               'Misc/runs/M7_1pct_1step_v19', n_steps=1)
