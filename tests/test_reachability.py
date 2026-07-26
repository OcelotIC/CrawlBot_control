#!/usr/bin/env python3
"""Reachability analysis for VISPA crawling robot.

Tests whether each target anchor is reachable from the docked
configuration at each step of the locomotion plan.

Usage:
    PYTHONPATH=. MUJOCO_GL=disabled python3 tests/test_reachability.py
"""
import sys
import os
import numpy as np

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import pinocchio as pin
from crawlbot.core.robot_interface import RobotInterface
# NB: do NOT `from ... import FRAME_TOOL_A/B`. Those are module-level 6-DOF
# defaults (18/32 = Link_6_a / Link_5_b) rebound by `global` only at
# construction; a by-value import freezes the stale ids and this script then
# does IK onto a mid-arm link. Use the instance attributes instead.
from crawlbot.planning.contact_scheduler import ContactScheduler
from crawlbot.core.ik import dock_configuration, solve_ik

PASS = '\033[92m PASS \033[0m'
FAIL = '\033[91m FAIL \033[0m'
WARN = '\033[93m WARN \033[0m'

urdf = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')
robot = RobotInterface(urdf, gravity='zero')
model = robot.model

# Anchor positions (from MJCF, structure body frame at pos=[0,0,-1.8])
anchors_a_local = [
    [-2.0,  0.3, 0.025], [-1.2,  0.3, 0.025], [-0.4,  0.3, 0.025],
    [ 0.4,  0.3, 0.025], [ 1.2,  0.3, 0.025], [ 2.0,  0.3, 0.025],
]
anchors_b_local = [
    [-2.0, -0.3, 0.025], [-1.2, -0.3, 0.025], [-0.4, -0.3, 0.025],
    [ 0.4, -0.3, 0.025], [ 1.2, -0.3, 0.025], [ 2.0, -0.3, 0.025],
]

struct_pos = np.array([0.0, 0.0, -1.8])

def anchor_se3(arm, idx):
    """Get anchor SE3 in world frame (structure at nominal position)."""
    local = anchors_a_local[idx] if arm == 'a' else anchors_b_local[idx]
    pos = struct_pos + np.array(local)
    return pin.SE3(np.eye(3), pos)


def check_reachability(stance_a_idx, stance_b_idx, swing_arm, target_idx,
                       q_start=None):
    """Check if target anchor is reachable from current docked config.

    Returns (reachable, ik_error, q_dock, details)
    """
    se3_a_stance = anchor_se3('a', stance_a_idx)
    se3_b_stance = anchor_se3('b', stance_b_idx)

    # Start config: both arms docked at stance anchors
    try:
        if q_start is None:
            q_start = dock_configuration(model, se3_a_stance, se3_b_stance)
    except RuntimeError as e:
        return False, np.inf, None, f"Start IK failed: {e}"

    # Target config: swing arm at target, stance arm unchanged
    if swing_arm == 'a':
        se3_a_target = anchor_se3('a', target_idx)
        target_anchors = (se3_a_target, se3_b_stance)
    else:
        se3_b_target = anchor_se3('b', target_idx)
        target_anchors = (se3_a_stance, se3_b_target)

    try:
        q_dock = dock_configuration(model, target_anchors[0], target_anchors[1],
                                    torso_pos=q_start[:3])
        # Verify actual tool positions
        data = model.createData()
        pin.forwardKinematics(model, data, q_dock)
        pin.updateFramePlacements(model, data)

        tool_fid = robot.frame_tool_a if swing_arm == 'a' else robot.frame_tool_b
        target_se3 = target_anchors[0] if swing_arm == 'a' else target_anchors[1]
        tool_pos = data.oMf[tool_fid].translation
        err_pos = np.linalg.norm(tool_pos - target_se3.translation)

        return err_pos < 0.001, err_pos, q_dock, f"pos_err={err_pos*1000:.2f}mm"

    except RuntimeError as e:
        return False, np.inf, None, f"Target IK failed: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# Test: reachability for the 3-step locomotion plan (start_a=2, start_b=2)
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('  Reachability analysis — 3-step locomotion plan')
print('  Start: arm_a@anchor_3a (idx=2), arm_b@anchor_3b (idx=2)')
print('=' * 70)

# The plan: alternating swing_b, swing_a, swing_b
# Step 0: swing_b from 2 to 3, stance: a@2, b@2 → target b@3
# Step 1: swing_a from 2 to 3, stance: a@2, b@3 → target a@3
# Step 2: swing_b from 3 to 4, stance: a@3, b@3 → target b@4

steps = [
    {'stance_a': 2, 'stance_b': 2, 'swing': 'b', 'target': 3, 'desc': 'b: 3b→4b'},
    {'stance_a': 2, 'stance_b': 3, 'swing': 'a', 'target': 3, 'desc': 'a: 3a→4a'},
    {'stance_a': 3, 'stance_b': 3, 'swing': 'b', 'target': 4, 'desc': 'b: 4b→5b'},
]

n_pass = n_fail = 0
q_current = None

for i, step in enumerate(steps):
    reachable, err, q_dock, detail = check_reachability(
        step['stance_a'], step['stance_b'],
        step['swing'], step['target'],
        q_start=q_current)

    if reachable:
        n_pass += 1
        print(f'  [{PASS}] Step {i} ({step["desc"]}): {detail}')
        q_current = q_dock
    else:
        n_fail += 1
        print(f'  [{FAIL}] Step {i} ({step["desc"]}): {detail}')

# Also test with wider reach: what's the max reachable anchor?
print('\n' + '=' * 70)
print('  Extended reachability — max anchor distance from stance')
print('=' * 70)

for swing_arm in ['a', 'b']:
    for stance_idx in range(5):
        other_arm = 'b' if swing_arm == 'a' else 'a'
        for target_idx in range(6):
            if target_idx == stance_idx:
                continue
            r, err, _, detail = check_reachability(
                stance_idx if other_arm == 'b' else stance_idx,
                stance_idx if other_arm == 'a' else stance_idx,
                swing_arm, target_idx)
            status = PASS if r else FAIL
            delta = target_idx - stance_idx
            print(f'  [{status}] swing_{swing_arm} stance@{stance_idx} '
                  f'target@{target_idx} (Δ={delta:+d}): {detail}')

print(f'\n  RESULTS: {n_pass}/{n_pass+n_fail} steps reachable')
