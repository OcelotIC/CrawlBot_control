"""One-off: MJCF damping=0 / armature=0.05 on arm joints, v21 mapping-off.

Mutates models/VISPA_crawling_rwa3.xml transiently (try/finally restore,
verify byte-exact on exit). The MJCF change per this run:
  - armature = 0.05 on arm joints (unchanged from committed MJCF)
  - damping  = 0    on arm joints (reduced from 0.05 per Part 2 findings)

Pinocchio `model.armature = 0.05` on arm joints is applied by
`crawlbot/core/robot_interface.py` as installed in commit 63a072f
(Part 1 fix); this is verified once before the run.

Calls the same `scripts.run_m7_single_step.run_case` helper as
`scripts/run_m7_v21_mapping_off.py`, with `mapping_bypass_in_ss=True`
and output redirected to
`Misc/runs/M7_1pct_1step_v21_armature_fix_damping_off_mapping_off/`.

Diagnostic only. Do not commit.
"""
from __future__ import annotations

import os
import re
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import numpy as np
import pinocchio as pin

import scripts.run_m7_single_step as r_single
from crawlbot.core.robot_interface import RobotInterface


URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')
MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')
OUT  = os.path.join(_root, 'Misc',
                    'runs',
                    'M7_1pct_1step_v21_armature_fix_damping_off_mapping_off')

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


def _verify_restored(original_text):
    with open(MJCF, 'r') as f:
        return f.read() == original_text


def _verify_pinocchio_armature():
    """Confirm the committed Part-1 armature-install block is in effect."""
    r = RobotInterface(URDF, gravity='zero')
    arm = np.asarray(r.model.armature, dtype=float)
    nv = r.model.nv
    ok_shape = arm.shape == (nv,)
    ok_base  = bool(np.all(arm[0:6] == 0.0))
    ok_arms  = bool(np.all(np.abs(arm[r.joints_v_slice] - 0.05) < 1e-12))
    return {
        'nv': nv, 'shape_ok': ok_shape,
        'base_zero': ok_base, 'arms_all_0p05': ok_arms,
        'base_slice': arm[0:6].tolist(),
        'arm_slice':  arm[r.joints_v_slice].tolist(),
    }


def main():
    with open(MJCF, 'r') as f:
        original = f.read()
    try:
        # Quick-check Pinocchio armature before mutating the MJCF.
        check = _verify_pinocchio_armature()
        print('[pin armature check]', check)
        assert check['shape_ok'] and check['base_zero'] \
            and check['arms_all_0p05'], 'Pinocchio armature sanity check failed'

        _mutate_mjcf(damping=0.0, armature=0.05)

        # Build config identical to scripts/run_m7_v21_mapping_off.py.
        cfg = r_single._make_m7_config()
        cfg.preplanner_a_cruise_max = 0.01
        cfg.preplanner_cruise_ramp_frac = 0.2
        cfg.mapping_bypass_in_ss = True

        r_single.run_case(
            '1pct 1-step v21 (armature=0.05, damping=0, mapping_bypass_in_ss=True)',
            OUT, n_steps=1, config=cfg)
    finally:
        with open(MJCF, 'w') as f:
            f.write(original)
        ok = _verify_restored(original)
        print(f'[mjcf restored byte-exactly] {ok}')


if __name__ == '__main__':
    main()
