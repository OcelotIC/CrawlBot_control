"""T15 — M7 v22 three-step traversal at mass_ratio = 0.01.

First multi-step validation under the T12-closure controller
configuration. Mirrors scripts/run_m7_v22_with_swing_hold.py (T11
single-step 1% reproducer) with a single change: n_steps = 3.

Controller configuration (HEAD of main, T12 closure bundle):
  - aocs_off_in_ds = True (explicit, defensive; retained T12 closure)
  - ds_ramp_duration_s = 2.0 (SimConfig default)
  - IPOPT max_iter = 200 (nmpc_solver.py:566)
  - All T11 fixes active:
        preplanner_a_cruise_max = 0.01
        preplanner_cruise_ramp_frac = 0.2
        mapping_bypass_in_ss = True
        swing_early_finish_fraction = 0.80
  - MJCF transient mutation (restored byte-exactly on exit):
        robot_joint damping = 0.0, armature = 0.05
  - No mass-ratio mutation (MJCF structure block unchanged: 1% by design).

Run:
  MUJOCO_GL=osmesa PYTHONPATH=. python3 scripts/run_m7_v22_1pct_3step_t15.py

Output:
    Misc/runs/M7_1pct_3step_v22_t15/
        sim_log.json, metrics.csv, physics_trace.pkl, diagnostic plots.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import scripts.run_m7_single_step as r_single


MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')
OUT  = os.path.join(_root, 'Misc', 'runs', 'M7_1pct_3step_v22_t15_bug1fix_vel')

ROBOT_JOINT_RE = re.compile(
    r'(<default class="robot_joint">\s*\n\s*<joint damping=")[^"]+'
    r'(" armature=")[^"]+(")')


def _mjcf_md5(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


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
    pre_hash = _mjcf_md5(MJCF)
    print(f'[T15] MJCF md5 (pre-run, pre-mutation):  {pre_hash}')
    try:
        _mutate_mjcf(damping=0.0, armature=0.05)
        mid_hash = _mjcf_md5(MJCF)
        print(f'[T15] MJCF md5 (during run, mutated): {mid_hash}')

        cfg = r_single._make_m7_config()
        # T11 fixes
        cfg.preplanner_a_cruise_max = 0.01
        cfg.preplanner_cruise_ramp_frac = 0.2
        cfg.mapping_bypass_in_ss = True
        cfg.swing_early_finish_fraction = 0.80
        # T12 closure (explicit, defensive)
        cfg.aocs_off_in_ds = True

        # Echo active config for the report §1
        print(f'[T15] aocs_off_in_ds           = {cfg.aocs_off_in_ds}')
        print(f'[T15] ds_ramp_duration_s       = {cfg.ds_ramp_duration_s}')
        print(f'[T15] mapping_bypass_in_ss     = {cfg.mapping_bypass_in_ss}')
        print(f'[T15] swing_early_finish_fraction = '
              f'{cfg.swing_early_finish_fraction}')
        print(f'[T15] preplanner_a_cruise_max  = '
              f'{cfg.preplanner_a_cruise_max}')
        print(f'[T15] preplanner_cruise_ramp_frac = '
              f'{cfg.preplanner_cruise_ramp_frac}')

        r_single.run_case(
            '1pct 3-step v22 T15 (aocs_off_in_ds=True, '
            'swing_early_finish_fraction=0.80, mapping_bypass_in_ss=True, '
            'MJCF damping=0 armature=0.05)',
            OUT, n_steps=3, config=cfg)
    finally:
        with open(MJCF, 'w') as f:
            f.write(original)
        with open(MJCF, 'r') as f:
            assert f.read() == original, 'MJCF restoration failed'
        post_hash = _mjcf_md5(MJCF)
        print(f'[T15] MJCF md5 (post-run, restored):   {post_hash}')
        print(f'[mjcf restored byte-exactly] '
              f'{post_hash == pre_hash}')


if __name__ == '__main__':
    main()
