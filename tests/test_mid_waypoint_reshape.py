"""Regression tests for the manipulability mid-waypoint reshape (Option B).

Covers:
  test_step2_mid_waypoint_resolves_singularity  — mid-waypoint IK on
      the T15 step-2 diagnostic fixture: success + dense piecewise
      sampling stays >= 1e-3.
  test_step0_no_regression                     — path-feasibility check
      on a well-conditioned anchor pair reports all-feasible (no mid-
      waypoint insertion needed).
  test_torso_planner_piecewise_continuous      — TorsoPlanner piecewise
      quintic: C0 continuous at t_mid; v=0, a=0 at all three waypoints.
  test_safety_threshold_triggers_fallback      — mid-waypoint IK
      returns success=False when w_min_threshold is set above the
      converged w_worst.

Per T15_step2_path_geometry.md §7.3 Option B and Phase-6 spec.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pinocchio as pin
import pytest

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

from crawlbot.core.ik import (  # noqa: E402
    manipulability_config_mid_waypoint,
    manipulability_config_trajectory,
    check_path_feasibility,
    _interpolate_q_quintic,
    _get_tool_frames,
    _arm_v_slice,
    _sigma_min_pair,
)
from crawlbot.core.robot_interface import RobotInterface  # noqa: E402
from crawlbot.planning.torso_planner import TorsoPlanner  # noqa: E402


URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')
FIXTURE = os.path.join(_root, 'diagnostic', 'step2_ss_entry_fixture.npz')


@pytest.fixture(scope='module')
def robot():
    return RobotInterface(URDF, tau_max=20.0, gravity='zero')


@pytest.fixture(scope='module')
def step2_fixture():
    fx = np.load(FIXTURE, allow_pickle=True)
    return {
        'q_start': np.asarray(fx['q_start'], dtype=float),
        'a_t': np.asarray(fx['se3_a_target_t'], dtype=float),
        'b_t': np.asarray(fx['se3_b_target_t'], dtype=float),
    }


def test_step2_mid_waypoint_resolves_singularity(robot, step2_fixture):
    """Mid-waypoint IK on T15 step-2 produces a deterministic, well-
    conditioned q_mid; the piecewise quintic stays above 1e-3 across
    21 dense samples per sub-segment.
    """
    model = robot.model
    data = model.createData()
    fid_a, fid_b = _get_tool_frames(model)
    sl_a = _arm_v_slice(model, fid_a)
    sl_b = _arm_v_slice(model, fid_b)

    # First obtain q_end from the existing trajectory IK (the fix from
    # the manipulability-IK-fix branch).
    q_start = step2_fixture['q_start']
    a_t = step2_fixture['a_t']
    b_t = step2_fixture['b_t']
    q_end, _, w_end = manipulability_config_trajectory(
        model, a_t, b_t, q_start.copy(), n_samples=5)
    assert w_end >= 1e-2, (
        f"Sanity precondition failed: trajectory IK should already "
        f"produce a well-conditioned endpoint, got w_end={w_end:.2e}")

    # Run mid-waypoint IK.
    se3_a = pin.SE3(np.eye(3), a_t)
    se3_b = pin.SE3(np.eye(3), b_t)
    q_mid, w_worst, success = manipulability_config_mid_waypoint(
        model, se3_a, se3_b, q_start, q_end,
        swing_arm='b', n_interior_samples=5,
        w_min_threshold=1e-3,
    )
    assert success, (
        f"manipulability_config_mid_waypoint reported failure on the "
        f"step-2 fixture (w_worst={w_worst:.2e} < threshold 1e-3). "
        f"Pre-fix IK_FORMULATION-fixed trajectory IK already gave "
        f"w_end >= 1e-2; the mid-waypoint search shouldn't regress.")
    assert q_mid is not None
    assert w_worst >= 1e-3

    # Piecewise sweep: 21 dense τ samples per sub-segment; every
    # sample's σ_min product must stay >= 1e-3.
    taus = np.linspace(0.05, 0.95, 19)
    for label, qa, qb in [('q_start->q_mid', q_start, q_mid),
                          ('q_mid->q_end',   q_mid,   q_end)]:
        for tau in taus:
            q_k = _interpolate_q_quintic(model, qa, qb, float(tau))
            sa, sb = _sigma_min_pair(model, data, q_k,
                                     fid_a, fid_b, sl_a, sl_b)
            w = sa * sb
            assert w >= 1e-3, (
                f"Piecewise sample failed: segment {label} τ={tau:.2f} "
                f"w={w:.2e} < 1e-3. Mid-waypoint did not resolve "
                f"the singular interior.")


def test_step0_no_regression(robot, step2_fixture):
    """On a well-conditioned anchor pair, check_path_feasibility
    reports all-feasible (no mid-waypoint insertion needed).

    Constructs a "step-0-like" easy case by using anchor positions
    very close to the swing arm's current FK position from
    q_start — i.e. minimal displacement, so the reference path is
    trivially feasible at every τ.
    """
    model = robot.model
    data = model.createData()
    fid_a, fid_b = _get_tool_frames(model)
    q_start = step2_fixture['q_start']

    # FK at q_start to read current EE poses (Pinocchio frame).
    pin.forwardKinematics(model, data, q_start)
    pin.updateFramePlacements(model, data)
    p_a_curr = data.oMf[fid_a].translation.copy()
    p_b_curr = data.oMf[fid_b].translation.copy()
    # Easy targets: arm A stays at its current pose, arm B moves
    # 50 mm in +x (a small step).
    a_t = p_a_curr.copy()
    b_t = p_b_curr + np.array([0.05, 0.0, 0.0])

    se3_a = pin.SE3(np.eye(3), a_t)
    se3_b = pin.SE3(np.eye(3), b_t)
    q_end, _, w_end = manipulability_config_trajectory(
        model, a_t, b_t, q_start.copy(), n_samples=5)
    assert w_end >= 1e-2

    result = check_path_feasibility(
        model, q_start, q_end, se3_a, se3_b,
        swing_arm='b', fid_torso=robot.frame_torso,
        n_samples=11, w_min_threshold=1e-3,
    )
    assert result['all_samples_feasible'], (
        f"Easy anchor pair flagged infeasible — false positive. "
        f"w_min={result['w_min']:.2e}@τ={result['w_min_tau']:.2f}, "
        f"n_below={result['n_infeasible_w']}, "
        f"n_ik_fail={result['n_ik_failures']}")
    assert result['w_min'] >= 1e-3


def test_torso_planner_piecewise_continuous():
    """TorsoPlanner piecewise quintic is C0 at t_mid and v=0, a=0 at
    all three waypoints (within numerical tolerance)."""
    tp = TorsoPlanner()
    p_start, R_start = np.array([0.0, 0.0, 0.0]), np.eye(3)
    p_mid_, R_mid_ = (np.array([0.6, 0.2, -0.1]),
                      pin.exp3(np.array([0.0, 0.0, 0.25])))
    p_end, R_end = (np.array([1.2, 0.0, 0.0]),
                    pin.exp3(np.array([0.0, 0.0, 0.5])))
    t_start, t_mid, t_end = 0.0, 0.4, 1.0
    tp.add_phase(t_start, t_end, p_start, R_start, p_end, R_end,
                 p_mid=p_mid_, R_mid=R_mid_, t_mid=t_mid)
    assert len(tp._phases) == 2

    # Exact values at the three waypoints.
    ref_start = tp.reference_at(t_start)
    ref_mid_left = tp.reference_at(t_mid - 1e-9)
    ref_mid_right = tp.reference_at(t_mid + 1e-9)
    ref_end = tp.reference_at(t_end)

    np.testing.assert_allclose(ref_start.p, p_start, atol=1e-10)
    np.testing.assert_allclose(ref_start.R, R_start, atol=1e-10)
    np.testing.assert_allclose(ref_end.p, p_end, atol=1e-10)
    np.testing.assert_allclose(ref_end.R, R_end, atol=1e-10)

    # C0 at t_mid: position & rotation continuous.
    np.testing.assert_allclose(ref_mid_left.p, p_mid_, atol=1e-9)
    np.testing.assert_allclose(ref_mid_right.p, p_mid_, atol=1e-9)
    np.testing.assert_allclose(ref_mid_left.R, R_mid_, atol=1e-9)
    np.testing.assert_allclose(ref_mid_right.R, R_mid_, atol=1e-9)

    # v=0, a=0 at all three waypoints (quintic time-scaling property).
    for label, ref in [('t_start', ref_start),
                       ('t_mid (left)', ref_mid_left),
                       ('t_mid (right)', ref_mid_right),
                       ('t_end', ref_end)]:
        v_norm = float(np.linalg.norm(ref.v))
        a_norm = float(np.linalg.norm(ref.a))
        assert v_norm < 1e-6, (
            f"v != 0 at {label}: ||v||={v_norm:.2e}")
        assert a_norm < 1e-6, (
            f"a != 0 at {label}: ||a||={a_norm:.2e}")

    # Also sample interior of each segment at τ=0.5 — should NOT be
    # zero velocity (otherwise the segment isn't actually moving).
    ref_seg1_mid = tp.reference_at(0.5 * (t_start + t_mid))
    ref_seg2_mid = tp.reference_at(0.5 * (t_mid + t_end))
    assert float(np.linalg.norm(ref_seg1_mid.v)) > 1e-3
    assert float(np.linalg.norm(ref_seg2_mid.v)) > 1e-3


def test_safety_threshold_triggers_fallback(robot, step2_fixture):
    """When the converged w_worst falls below the threshold, the
    mid-waypoint IK returns success=False (caller falls back).
    Verified by setting the threshold artificially high (well above
    any feasible w_worst on this fixture).
    """
    model = robot.model
    q_start = step2_fixture['q_start']
    a_t = step2_fixture['a_t']
    b_t = step2_fixture['b_t']
    q_end, _, _ = manipulability_config_trajectory(
        model, a_t, b_t, q_start.copy(), n_samples=5)

    se3_a = pin.SE3(np.eye(3), a_t)
    se3_b = pin.SE3(np.eye(3), b_t)
    # A threshold of 1.0 is two orders above any plausible σ_min
    # product on this hardware (typical ~5e-2). Forces failure.
    q_mid, w_worst, success = manipulability_config_mid_waypoint(
        model, se3_a, se3_b, q_start, q_end,
        swing_arm='b', n_interior_samples=5,
        w_min_threshold=1.0,
    )
    assert not success, (
        f"Expected failure with extreme threshold, got success "
        f"(w_worst={w_worst:.2e})")
    assert w_worst < 1.0
