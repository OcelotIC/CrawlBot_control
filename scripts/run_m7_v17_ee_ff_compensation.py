#!/usr/bin/env python3
"""M7 v17: task-consistent EE feedforward + v16 SS torso reference.

v16 config is unchanged (TorsoPlanner drives SS torso ref; mapping still
drives DS). The new piece is in wholebody_qp: the EE task now augments
its feedforward acceleration with J_ee · J_torso^+ · a_torso_des, where
a_torso_des is the torso task's full PD+FF command. This cancels the
first-order EE/torso coupling through the shared floating base so the
EE PD does not have to correct it reactively.

Output: Misc/runs/M7_1pct_1step_v17/.
"""
from __future__ import annotations

import os
import sys

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import scripts.run_m7_single_step as r


if __name__ == '__main__':
    r.run_case('1pct 1-step v17 (v16 + EE task-consistent feedforward)',
               'Misc/runs/M7_1pct_1step_v17', n_steps=1)
