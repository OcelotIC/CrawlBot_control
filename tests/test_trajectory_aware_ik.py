"""Unit tests for trajectory-aware manipulability IK (Manipulability-IK-1).

Design: docs/architecture/T15_MANIPULABILITY_IK_DESIGN.md §9.

This file is the Phase-2 Commit-1 skeleton. The four tests below are
all marked `pytest.skip` at module level so the baseline suite stays
green. Commit 2 (ik.py implementation) removes the skip and adds the
real assertions.
"""
import numpy as np
import pytest
import pinocchio as pin

# All four tests skipped until Commit 2 lands the ik.py helpers.
pytestmark = pytest.mark.skip(reason="implementation pending (Commit 2)")


def test_quintic_interpolation_endpoints(pinocchio_model):
    """_interpolate_q_quintic(q0, q1, 0)≈q0; (q0, q1, 1)≈q1; zero slope at endpoints."""
    from crawlbot.core.ik import _interpolate_q_quintic
    model = pinocchio_model
    q0 = pin.neutral(model)
    q1 = pin.randomConfiguration(model)
    assert np.allclose(_interpolate_q_quintic(model, q0, q1, 0.0), q0)
    assert np.allclose(_interpolate_q_quintic(model, q0, q1, 1.0), q1)
    # Zero-slope quintic: difference near endpoints must be O(τ^3).
    dtau = 1e-3
    q_near0 = _interpolate_q_quintic(model, q0, q1, dtau)
    q_near1 = _interpolate_q_quintic(model, q0, q1, 1.0 - dtau)
    assert np.linalg.norm(q_near0 - q0) < 1e-6
    assert np.linalg.norm(q_near1 - q1) < 1e-6


def test_trajectory_ik_matches_endpoint_for_k1(pinocchio_model, contact_scheduler):
    """With n_samples=1, trajectory IK reduces to endpoint IK."""
    from crawlbot.core.ik import manipulability_config, manipulability_config_trajectory
    model = pinocchio_model
    # Pick a representative anchor pair from the scheduler.
    anchors_a = contact_scheduler.anchors_a
    anchors_b = contact_scheduler.anchors_b
    se3_a = pin.SE3(np.eye(3), anchors_a[0].copy())
    se3_b = pin.SE3(np.eye(3), anchors_b[0].copy())
    q_end_ep, w_ep = manipulability_config(model, se3_a, se3_b)
    q_start = pin.neutral(model)
    q_end_tr, w_worst, w_end = manipulability_config_trajectory(
        model, anchors_a[0], anchors_b[0], q_start, n_samples=1,
    )
    # Endpoint values should agree to IK tolerance.
    assert abs(w_end - w_ep) < 1e-3
    assert abs(w_worst - w_end) < 1e-6  # single sample at τ=1


def test_trajectory_ik_improves_worst_case(pinocchio_model, contact_scheduler):
    """Trajectory IK delivers ≥ endpoint w_worst (never worse, usually better)."""
    from crawlbot.core.ik import manipulability_config, manipulability_config_trajectory
    model = pinocchio_model
    anchors_a = contact_scheduler.anchors_a
    anchors_b = contact_scheduler.anchors_b
    se3_a = pin.SE3(np.eye(3), anchors_a[1].copy())
    se3_b = pin.SE3(np.eye(3), anchors_b[1].copy())
    q_end_ep, _ = manipulability_config(model, se3_a, se3_b)
    q_start = pin.neutral(model)
    _, w_worst_tr, _ = manipulability_config_trajectory(
        model, anchors_a[1], anchors_b[1], q_start, n_samples=5,
    )
    # Replay the interior samples of the endpoint solution to get its w_worst.
    from crawlbot.core.ik import _interpolate_q_quintic, _arm_v_slice, _get_tool_frames
    fid_a, fid_b = _get_tool_frames(model)
    data = model.createData()
    w_worst_ep = np.inf
    for k in range(1, 6):
        tau = k / 5
        q_k = _interpolate_q_quintic(model, q_start, q_end_ep, tau)
        pin.forwardKinematics(model, data, q_k)
        pin.updateFramePlacements(model, data)
        pin.computeJointJacobians(model, data, q_k)
        Ja = pin.getFrameJacobian(model, data, fid_a, pin.LOCAL)[:, _arm_v_slice(model, fid_a)]
        Jb = pin.getFrameJacobian(model, data, fid_b, pin.LOCAL)[:, _arm_v_slice(model, fid_b)]
        w_k = float(np.linalg.svd(Ja, compute_uv=False)[-1]
                    * np.linalg.svd(Jb, compute_uv=False)[-1])
        w_worst_ep = min(w_worst_ep, w_k)
    assert w_worst_tr >= w_worst_ep - 1e-6


def test_chain_consistency(pinocchio_model, contact_scheduler):
    """Chained precompute: map entry n's q_start_assumed == map entry n-1's q_end."""
    from crawlbot.core.ik import precompute_torso_map
    model = pinocchio_model
    plan = contact_scheduler.plan
    anchor_pair_sequence = [(ph.anchor_a_idx, ph.anchor_b_idx) for ph in plan.phases]
    anchors_a = contact_scheduler.anchors_a
    anchors_b = contact_scheduler.anchors_b
    # Build endpoint map first, seed chain from its first pair.
    endpoint_map = precompute_torso_map(model, anchors_a, anchors_b)
    q_initial = endpoint_map[anchor_pair_sequence[0]]
    traj_map = precompute_torso_map(
        model, anchors_a, anchors_b,
        anchor_pair_sequence=anchor_pair_sequence,
        q_initial=q_initial,
        n_samples=5,
        use_trajectory_aware=True,
    )
    # Walk the sequence; each entry's q_start_assumed must equal the previous q_end.
    prev_q_end = q_initial
    for pair in anchor_pair_sequence:
        entry = traj_map[pair]
        assert isinstance(entry, dict)
        assert np.allclose(entry['q_start_assumed'], prev_q_end)
        prev_q_end = entry['q_end']
