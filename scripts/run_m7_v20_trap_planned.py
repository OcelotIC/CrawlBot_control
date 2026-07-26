#!/usr/bin/env python3
"""M7 v20: trapezoidal TorsoPlanner + v19 planned-delta mapping.

Configuration
-------------
- TorsoPlanner: trapezoidal velocity profile with ramp = 0.20
  (20/60/20 split). Ramp-up tau in [0, 0.2], cruise tau in [0.2, 0.8]
  with a_ff == 0 and v_ref constant, ramp-down tau in [0.8, 1.0].
- Mapping: active in SS and DS; in SS fed q_planned (v19, quintic
  interp of arm joints from start to IK end configuration). The
  mapping supplies v_b_ref = -delta_dot(q_planned, dq_planned)/m_b
  as feedforward compensation against arm-induced base drift, even
  during the cruise window where a_torso_ff == 0.
- EE task-consistent feedforward (v17) still active in wholebody_qp.py.
- All other settings from v19 unchanged: alpha_wrench=0.01 (QP
  config), alpha_com_soft=0.0, weight_ratio=1.0, 7-DOF arms,
  manipulability IK init, quintic SwingPlanner.

Rationale (see docs/architecture/M7_TECHNICAL_LOG.md section 9)
- v18 quintic alone: smooth a_ff but peak joint torque saturated
  the actuator budget at 20 Nm -> EE tracking degraded.
- v19 quintic + planned-delta: torso 42 mm, but EE still 165 mm
  because the stance arm burnt the torque budget pushing the torso.
- v20 trapezoidal + planned-delta: cruise frees 15+ Nm for the EE
  task while the mapping continues to give velocity feedforward
  through delta_dot.

Output: Misc/runs/M7_1pct_1step_v20/.
"""
from __future__ import annotations

import os
import sys

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import scripts.run_m7_single_step as r


if __name__ == '__main__':
    r.run_case('1pct 1-step v20 (trap 20/60/20 + planned-delta + EE FF)',
               'Misc/runs/M7_1pct_1step_v20', n_steps=1)
