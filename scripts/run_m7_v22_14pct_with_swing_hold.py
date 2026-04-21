"""T12 — M7 v22 single-step closed loop at mass_ratio = 0.14.

Identical to scripts/run_m7_v22_with_swing_hold.py (T11), with a single
parametric change: structure mass 7110 kg → 7110/14 = 507.857... kg, with
the principal inertia tensor scaled by the same factor (preserves shape).
The scaling factor 1/14 takes mass_ratio 0.01 → 0.14 exactly.

Transient MJCF mutation with byte-exact restoration (mirrors T11 pattern
for armature/damping). No commits, no new MJCF files.

Output:
    results/M7_14pct_1step_v22_with_swing_hold/
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
OUT  = os.path.join(_root, 'results', 'M7_14pct_1step_v22_with_swing_hold')

ROBOT_JOINT_RE = re.compile(
    r'(<default class="robot_joint">\s*\n\s*<joint damping=")[^"]+'
    r'(" armature=")[^"]+(")')

STRUCTURE_INERTIAL_RE = re.compile(
    r'(<body name="structure"[^>]*>\s*\n\s*<freejoint name="structure_free"/>'
    r'\s*\n\s*<inertial pos="0 0 0" mass=")7110("\s*\n\s*'
    r'fullinertia=")597 1493 1777 0 0 0(")')

MASS_OLD = 7110.0
INERTIA_OLD = (597.0, 1493.0, 1777.0)
SCALE = 1.0 / 14.0   # mass_ratio 0.01 → 0.14 ⇒ structure mass × 1/14
MASS_NEW = MASS_OLD * SCALE                              # 507.8571428571428
INERTIA_NEW = tuple(v * SCALE for v in INERTIA_OLD)      # (42.642..., 106.642..., 126.928...)


def _mutate_mjcf(damping: float, armature: float,
                 mass: float, inertia: tuple):
    with open(MJCF, 'r') as f:
        text = f.read()
    new, n1 = ROBOT_JOINT_RE.subn(
        rf'\g<1>{damping}\g<2>{armature}\g<3>', text, count=1)
    if n1 != 1:
        raise RuntimeError('robot_joint default not found')
    ixx, iyy, izz = inertia
    ms = f'{mass:.10f}'
    fs = f'{ixx:.10f} {iyy:.10f} {izz:.10f} 0 0 0'
    new2, n2 = STRUCTURE_INERTIAL_RE.subn(
        rf'\g<1>{ms}\g<2>{fs}\g<3>', new, count=1)
    if n2 != 1:
        raise RuntimeError('structure inertial block not found')
    with open(MJCF, 'w') as f:
        f.write(new2)


def main():
    with open(MJCF, 'r') as f:
        original = f.read()
    try:
        _mutate_mjcf(damping=0.0, armature=0.05,
                     mass=MASS_NEW, inertia=INERTIA_NEW)

        cfg = r_single._make_m7_config()
        cfg.preplanner_a_cruise_max = 0.01
        cfg.preplanner_cruise_ramp_frac = 0.2
        cfg.mapping_bypass_in_ss = True
        cfg.swing_early_finish_fraction = 0.80

        print(f'[T12] mass_ratio target = 0.14 '
              f'(scale = 1/14 applied to structure body)')
        print(f'[T12] structure body mass: {MASS_OLD} -> {MASS_NEW:.6f} kg')
        print(f'[T12] structure fullinertia: {INERTIA_OLD} -> '
              f'({INERTIA_NEW[0]:.6f}, {INERTIA_NEW[1]:.6f}, {INERTIA_NEW[2]:.6f})')

        r_single.run_case(
            '14pct 1-step v22 (mass_ratio=0.14, swing_early_finish_fraction=0.80, '
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
