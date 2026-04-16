#!/usr/bin/env python3
"""M7 v16: SS torso reference from TorsoPlanner (no mapping during SS).

Config is default M7: torso_early_finish_fraction = 1.0, no
displacement scaling. The change is entirely in sim_loop:

  SS  -> TorsoPlanner.reference_at(t) drives p, v, a
  DS  -> M1 mapping drives p, v, a (unchanged)

The NMPC still runs during SS; its lambda_ref and a_com_ff still enter
the QP's wrench task and CoM-feedforward term. What changes is that
torso DISPLACEMENT no longer comes from (r_com_ref, delta(q)) via the
mass-weighted identity — it comes from the planner's quintic over
[0, T_step]. The arm sees a torso reference whose acceleration is
decoupled from arm CoM motion.

Output: results/M7_1pct_1step_v16/.
"""
from __future__ import annotations

import os
import sys

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import scripts.run_m7_single_step as r


if __name__ == '__main__':
    r.run_case('1pct 1-step v16 (SS torso ref from TorsoPlanner)',
               'results/M7_1pct_1step_v16', n_steps=1)
