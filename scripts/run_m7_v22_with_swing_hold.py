"""M7 v22 — swing early-finish + hold window, single-step closed loop.

Configuration:
  - Pinocchio armature = 0.05 on arm joints (from commit 63a072f, on branch).
  - MJCF arm joints: damping = 0, armature = 0.05 (transient mutation,
    byte-exactly restored on exit via try/finally).
  - SimConfig: v21 baseline + mapping_bypass_in_ss = True +
    swing_early_finish_fraction = 0.80.

Output:
    results/M7_1pct_1step_v22_with_swing_hold/
        sim_log.json, metrics.csv, physics_trace.pkl, 10 diagnostic plots.
"""
from __future__ import annotations

import os
import re
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import scripts.run_m7_single_step as r_single


MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')
OUT  = os.path.join(_root, 'results', 'M7_1pct_1step_v22_with_swing_hold')

ROBOT_JOINT_RE = re.compile(
    r'(<default class="robot_joint">\s*\n\s*<joint damping=")[^"]+'
    r'(" armature=")[^"]+(")')


def _mutate_mjcf(damping: float, armature: float):
    with open(MJCF, 'r') as f:
        text = f.read()
    new, n = ROBOT_JOINT_RE.subn(
        rf'\g<1>{damping}\g<2>{armature}\g<3>', text, count=1)
    if n != 1:
        raise RuntimeError('robot_joint default not found')
    with open(MJCF, 'w') as f:
        f.write(new)


def main():
    with open(MJCF, 'r') as f:
        original = f.read()
    try:
        _mutate_mjcf(damping=0.0, armature=0.05)

        cfg = r_single._make_m7_config()
        cfg.preplanner_a_cruise_max = 0.01
        cfg.preplanner_cruise_ramp_frac = 0.2
        cfg.mapping_bypass_in_ss = True
        cfg.swing_early_finish_fraction = 0.80

        r_single.run_case(
            '1pct 1-step v22 (swing_early_finish_fraction=0.80, '
            'mapping_bypass_in_ss=True, MJCF damping=0 armature=0.05)',
            OUT, n_steps=1, config=cfg)
    finally:
        with open(MJCF, 'w') as f:
            f.write(original)
        with open(MJCF, 'r') as f:
            assert f.read() == original, 'MJCF restoration failed'
        print('[mjcf restored byte-exactly] True')


if __name__ == '__main__':
    main()
