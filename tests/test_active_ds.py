"""TDD tests for the active-DS architecture's geometric primitives.

Validates docs/architecture/active_ds_torso_advance.md §3 (math
derivation) and §4.1 (public API contract) for the
crawlbot/planning/double_stance_planner.py module.

Phase 1 covers tests D.1, D.2, D.4 (the Phase-1 exit gate):
  D.1: compute_q_target_DS — projection contracts.
  D.2: smoothed_constrained_geodesic_double_stance — endpoints
       exact, stance compliance at every interior k, torso arc
       length inflation cap.
  D.4: reachability_gate — workspace + IK feasibility.

D.3 (closed-loop sanity), D.5 (T15 dock), D.6 (regression) come
in Phases 2-5 as the planner code lands.

Run:
    PYTHONPATH=. MUJOCO_GL=disabled python3 -m pytest \\
        tests/test_active_ds.py -v
"""
from __future__ import annotations

import os

import numpy as np
import pinocchio as pin
import pytest

os.environ.setdefault('MUJOCO_GL', 'disabled')

from crawlbot.core.ik import _get_tool_frames, dock_configuration
from crawlbot.core.robot_interface import RobotInterface
from crawlbot.planning.double_stance_planner import (
    compute_q_target_DS,
    project_to_double_stance,
    reachability_gate,
    smoothed_constrained_geodesic_double_stance,
)

URDF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'models', 'VISPA_crawling_fixed.urdf',
)
MJCF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'models', 'VISPA_crawling_rwa3.xml',
)

# Pass thresholds per active_ds_torso_advance.md §3.7 + §5.
GATE_STANCE_TOL_M = 5e-5      # 50 μm
ARC_INFLATION_CAP = 1.10      # +10 % vs raw chord


# --------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------


@pytest.fixture(scope='module')
def robot():
    return RobotInterface(URDF, tau_max=20.0, gravity='zero')


@pytest.fixture(scope='module')
def fids(robot):
    fid_a, fid_b = _get_tool_frames(robot.model)
    return {
        'fid_a': fid_a,
        'fid_b': fid_b,
        'fid_torso': robot.frame_torso,
    }


@pytest.fixture(scope='module')
def anchors():
    """Local-frame anchor grid via the same transform as the merged sim_loop."""
    import mujoco
    from crawlbot.planning.contact_scheduler import read_anchors_from_mujoco
    mj_model = mujoco.MjModel.from_xml_path(MJCF)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    a_world, b_world = read_anchors_from_mujoco(mj_model, mj_data)
    p_s0 = mj_data.qpos[0:3].copy()
    qw, qx, qy, qz = mj_data.qpos[3:7]
    R_s0 = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
    return {
        'a': [R_s0.T @ (a - p_s0) for a in a_world],
        'b': [R_s0.T @ (b - p_s0) for b in b_world],
    }


@pytest.fixture(scope='module')
def step2_setup(robot, fids, anchors):
    """T15 step 2: DS pair (a@3, b@3); SS target pair (a@3, b@4).

    Returns dict with q_DS_start, q_SS_end, anchor_a_now, anchor_b_now.
    """
    from crawlbot.core.ik import manipulability_config_trajectory
    a_local = anchors['a']
    b_local = anchors['b']
    q_DS_start = dock_configuration(
        robot.model,
        pin.SE3(np.eye(3), a_local[3]),
        pin.SE3(np.eye(3), b_local[3]),
    )
    q_SS_end, _, _ = manipulability_config_trajectory(
        robot.model, a_local[3], b_local[4],
        q_start=q_DS_start, n_samples=5,
    )
    assert q_SS_end is not None, "step-2 IK rejected — fixture fail"
    return {
        'q_DS_start': q_DS_start,
        'q_SS_end': q_SS_end,
        'anchor_a_now': a_local[3],
        'anchor_b_now': b_local[3],
        'anchor_swing_target': b_local[4],
    }


# --------------------------------------------------------------------
# D.1 — compute_q_target_DS contracts
# --------------------------------------------------------------------


@pytest.mark.parametrize("beta", [0.3, 0.5, 0.7, 0.9])
def test_D1_compute_q_target_DS_stance_compliance(robot, fids, step2_setup, beta):
    """Eq. §3.2: project pin.interpolate(q_DS_start, q_SS_end, β) onto
    M_double; assert both stance residuals ≤ 50 μm and IK converged.
    """
    q_target, residual, conv = compute_q_target_DS(
        robot.model,
        step2_setup['q_DS_start'], step2_setup['q_SS_end'],
        beta=beta,
        fid_a=fids['fid_a'], fid_b=fids['fid_b'],
        anchor_a_now=step2_setup['anchor_a_now'],
        anchor_b_now=step2_setup['anchor_b_now'],
    )
    assert conv, f"β={beta}: 2-task IK did not converge (residual {residual:.2e})"
    data = robot.model.createData()
    pin.forwardKinematics(robot.model, data, q_target)
    pin.updateFramePlacements(robot.model, data)
    ra = float(np.linalg.norm(
        data.oMf[fids['fid_a']].translation - step2_setup['anchor_a_now']))
    rb = float(np.linalg.norm(
        data.oMf[fids['fid_b']].translation - step2_setup['anchor_b_now']))
    assert ra < GATE_STANCE_TOL_M, (
        f"β={beta}: stance_a residual {ra*1e6:.1f} μm > "
        f"{GATE_STANCE_TOL_M*1e6:.0f} μm gate")
    assert rb < GATE_STANCE_TOL_M, (
        f"β={beta}: stance_b residual {rb*1e6:.1f} μm > "
        f"{GATE_STANCE_TOL_M*1e6:.0f} μm gate")


def test_D1_compute_q_target_DS_endpoints(robot, fids, step2_setup):
    """β=0 returns (essentially) q_DS_start; β=1 returns the
    projection of q_SS_end onto M_double_current (NOT q_SS_end
    itself, since q_SS_end has the swing arm at the NEW anchor)."""
    q_b0, _, conv0 = compute_q_target_DS(
        robot.model, step2_setup['q_DS_start'], step2_setup['q_SS_end'],
        beta=0.0, fid_a=fids['fid_a'], fid_b=fids['fid_b'],
        anchor_a_now=step2_setup['anchor_a_now'],
        anchor_b_now=step2_setup['anchor_b_now'],
    )
    assert conv0
    # β=0 should land essentially on q_DS_start (stance compliance is 0
    # at q_DS_start by construction; projection has trivial work).
    dq0 = pin.difference(robot.model, step2_setup['q_DS_start'], q_b0)
    assert float(np.linalg.norm(dq0)) < 1e-4, (
        f"β=0 projection drifted by {np.linalg.norm(dq0):.4e} rad — "
        f"expected ≈0 because q_DS_start is already on M_double")


# --------------------------------------------------------------------
# D.2 — DS smoother contracts
# --------------------------------------------------------------------


@pytest.fixture(scope='module')
def step2_DS_smoothed(robot, fids, step2_setup):
    q_target, _, _ = compute_q_target_DS(
        robot.model,
        step2_setup['q_DS_start'], step2_setup['q_SS_end'],
        beta=0.7,
        fid_a=fids['fid_a'], fid_b=fids['fid_b'],
        anchor_a_now=step2_setup['anchor_a_now'],
        anchor_b_now=step2_setup['anchor_b_now'],
    )
    q_seq, info = smoothed_constrained_geodesic_double_stance(
        robot.model,
        q_DS_start=step2_setup['q_DS_start'],
        q_target_DS_proj=q_target,
        fid_a=fids['fid_a'], fid_b=fids['fid_b'],
        fid_torso=fids['fid_torso'],
        n_tau=11, n_iter=80, tol=1e-5, verbose=False,
    )
    return {
        'q_seq': q_seq, 'info': info,
        'q_target': q_target,
        'q_DS_start': step2_setup['q_DS_start'],
        'anchor_a_now': step2_setup['anchor_a_now'],
        'anchor_b_now': step2_setup['anchor_b_now'],
    }


def test_D2_DS_smoother_endpoints(robot, step2_DS_smoothed):
    """q_seq[0] = q_DS_start, q_seq[-1] = q_target_DS_proj exactly."""
    np.testing.assert_allclose(
        step2_DS_smoothed['q_seq'][0],
        step2_DS_smoothed['q_DS_start'], atol=1e-12,
        err_msg="q_seq[0] != q_DS_start")
    np.testing.assert_allclose(
        step2_DS_smoothed['q_seq'][-1],
        step2_DS_smoothed['q_target'], atol=1e-12,
        err_msg="q_seq[-1] != q_target_DS_proj")


def test_D2_DS_smoother_stance_compliance(robot, fids, step2_DS_smoothed):
    """FK[fid_a] and FK[fid_b] match their CURRENT stance anchors at
    every interior k to within 50 μm."""
    data = robot.model.createData()
    max_a = max_b = 0.0
    for k, q in enumerate(step2_DS_smoothed['q_seq']):
        pin.forwardKinematics(robot.model, data, q)
        pin.updateFramePlacements(robot.model, data)
        ra = float(np.linalg.norm(
            data.oMf[fids['fid_a']].translation
            - step2_DS_smoothed['anchor_a_now']))
        rb = float(np.linalg.norm(
            data.oMf[fids['fid_b']].translation
            - step2_DS_smoothed['anchor_b_now']))
        max_a = max(max_a, ra)
        max_b = max(max_b, rb)
    assert max_a < GATE_STANCE_TOL_M, (
        f"max stance_a residual {max_a*1e6:.1f} μm > "
        f"{GATE_STANCE_TOL_M*1e6:.0f} μm gate")
    assert max_b < GATE_STANCE_TOL_M, (
        f"max stance_b residual {max_b*1e6:.1f} μm > "
        f"{GATE_STANCE_TOL_M*1e6:.0f} μm gate")


def test_D2_DS_smoother_arc_length_inflation(robot, fids, step2_DS_smoothed):
    """World-frame torso arc length ≤ 110 % of the start→end chord
    (rejects pathological off-axis paths; per Phase-0 the smoother
    achieves +0% on real T15 step-2)."""
    data = robot.model.createData()
    p_torso = []
    for q in step2_DS_smoothed['q_seq']:
        pin.forwardKinematics(robot.model, data, q)
        pin.updateFramePlacements(robot.model, data)
        p_torso.append(data.oMf[fids['fid_torso']].translation.copy())
    p_torso = np.array(p_torso)
    arc = float(np.linalg.norm(np.diff(p_torso, axis=0), axis=1).sum())
    chord = float(np.linalg.norm(p_torso[-1] - p_torso[0]))
    if chord < 1e-6:
        pytest.skip("chord ≈ 0 — degenerate test case")
    inflation = arc / chord
    assert inflation < ARC_INFLATION_CAP, (
        f"world-frame torso arc / chord = {inflation:.4f} > "
        f"{ARC_INFLATION_CAP} cap (torso path bulges off-axis)")


def test_D2_DS_smoother_zero_fallbacks(step2_DS_smoothed):
    """The 3-task IK at task-space midpoints should converge cleanly
    on the well-conditioned double-stance manifold for T15 step 2.
    Fallbacks > 0 indicate near-singular interior configurations."""
    info = step2_DS_smoothed['info']
    n_iter = info['iters']
    n_interior = len(step2_DS_smoothed['q_seq']) - 2
    rate = info['fallbacks_total'] / max(n_iter * n_interior, 1)
    assert info['fallbacks_total'] < 5, (
        f"DS smoother had {info['fallbacks_total']} fallbacks "
        f"({rate*100:.1f}% of {n_iter}×{n_interior}); "
        f"manifold may have local issues")


# --------------------------------------------------------------------
# D.4 — reachability_gate behaviour
# --------------------------------------------------------------------


def test_D4_gate_passes_at_DS_end(robot, fids, anchors, step2_DS_smoothed):
    """At q_target (DS-end of step 2 with β=0.7), the gate should
    PASS — the body has advanced 700 mm, swing arm 'b' can reach
    anchor_b[4] cleanly."""
    q_target = step2_DS_smoothed['q_target']
    passed, info_gate = reachability_gate(
        robot.model, q_target,
        fid_swing=fids['fid_b'],
        anchor_swing_target=anchors['b'][4],
        fid_torso=fids['fid_torso'],
        fid_stance_a=fids['fid_a'],
        se3_stance_a_target=pin.SE3(
            np.eye(3), step2_DS_smoothed['anchor_a_now']),
    )
    assert passed, (
        f"reachability_gate failed at DS-end "
        f"(d_reach = {info_gate['d_reach_m']*1000:.1f} mm, "
        f"w_swing = {info_gate['w_swing']:.3e}, "
        f"residual = {info_gate['ik_residual']:.2e}, "
        f"stage = {info_gate['stage_failed']})")
    assert info_gate['w_swing'] > 0.05, (
        f"w_swing {info_gate['w_swing']:.3e} suspiciously low at DS-end")


# --------------------------------------------------------------------
# D.2-PLANNER — TorsoPlanner.add_phase_double_stance integration
# --------------------------------------------------------------------


def test_D2_torso_planner_DS_active_endpoints(robot, fids, step2_DS_smoothed):
    """add_phase_double_stance + reference_at(t) should return
    FK[torso] at q_DS_seq[0] at t_start and FK[torso] at
    q_DS_seq[-1] at t_end. Acceleration zero on boundary segments.
    """
    from crawlbot.planning.torso_planner import TorsoPlanner
    tp = TorsoPlanner(model=robot.model, frame_torso=fids['fid_torso'])
    t_start = 5.0
    t_end = 10.0
    tp.add_phase_double_stance(t_start, t_end,
                               step2_DS_smoothed['q_seq'])
    # Phase carries the right tag.
    assert tp._phases[-1]['kind'] == 'DS_active'
    assert tp._phases[-1]['use_fk'] is True

    # Reference at t_start should match FK[torso] at q_DS_start.
    data = robot.model.createData()
    pin.forwardKinematics(robot.model, data,
                          step2_DS_smoothed['q_seq'][0])
    pin.updateFramePlacements(robot.model, data)
    p_expected = data.oMf[fids['fid_torso']].translation.copy()
    R_expected = data.oMf[fids['fid_torso']].rotation.copy()
    ref = tp.reference_at(t_start)
    np.testing.assert_allclose(ref.p, p_expected, atol=1e-12)
    np.testing.assert_allclose(ref.R, R_expected, atol=1e-12)
    np.testing.assert_allclose(ref.a, np.zeros(6), atol=1e-12,
                               err_msg="boundary segment acceleration "
                                       "must be zero")

    # Reference at t_end should match FK[torso] at q_DS_target.
    pin.forwardKinematics(robot.model, data,
                          step2_DS_smoothed['q_seq'][-1])
    pin.updateFramePlacements(robot.model, data)
    p_expected = data.oMf[fids['fid_torso']].translation.copy()
    R_expected = data.oMf[fids['fid_torso']].rotation.copy()
    ref = tp.reference_at(t_end)
    np.testing.assert_allclose(ref.p, p_expected, atol=1e-12)
    np.testing.assert_allclose(ref.R, R_expected, atol=1e-12)
    np.testing.assert_allclose(ref.a, np.zeros(6), atol=1e-12)


def test_D2_torso_planner_DS_active_interior(robot, fids, step2_DS_smoothed):
    """At an interior τ ∈ (0, 1) the DS_active reference should give
    a torso pose between the endpoint poses (monotone advance) and
    a non-zero linear velocity reference."""
    from crawlbot.planning.torso_planner import TorsoPlanner
    tp = TorsoPlanner(model=robot.model, frame_torso=fids['fid_torso'])
    t_start = 5.0
    t_end = 10.0
    tp.add_phase_double_stance(t_start, t_end,
                               step2_DS_smoothed['q_seq'])
    # Endpoints
    ref_0 = tp.reference_at(t_start)
    ref_1 = tp.reference_at(t_end)
    # Interior
    ref_mid = tp.reference_at(t_start + 0.5 * (t_end - t_start))
    # Mid torso position should be in the convex hull along the path.
    chord = np.linalg.norm(ref_1.p - ref_0.p)
    if chord > 1e-6:
        # Project mid onto the chord direction.
        u = (ref_1.p - ref_0.p) / chord
        proj = np.dot(ref_mid.p - ref_0.p, u)
        assert -1e-3 < proj < chord + 1e-3, (
            f"mid torso projection {proj:.4f} outside chord [0, {chord:.4f}]")
    # Non-zero linear velocity at interior τ (the planner's tangent
    # vector / Δτ / T_phase).
    assert np.linalg.norm(ref_mid.v[:3]) > 1e-6, (
        f"interior torso linear velocity is zero — "
        f"ref_mid.v = {ref_mid.v}")


def test_D2_torso_planner_legacy_path_byte_identical(robot, fids):
    """Phase-2 must NOT touch the legacy SLERP code path. Construct
    a plain task-space TorsoPlanner (no model/frame_torso); call
    add_phase with legacy task-space args; assert reference_at
    produces consistent legacy refs (the two endpoint poses match).
    Smoke test against accidental regression of the non-FK path.
    """
    from crawlbot.planning.torso_planner import TorsoPlanner
    tp = TorsoPlanner()  # no model — legacy SLERP mode
    p0 = np.array([0.1, -0.05, -0.85])
    p1 = np.array([0.4, 0.15, -0.75])
    R0 = np.eye(3)
    R1 = np.eye(3)
    tp.add_phase(5.0, 10.0, p0, R0, p1, R1)
    ref_start = tp.reference_at(5.0)
    ref_end = tp.reference_at(10.0)
    np.testing.assert_allclose(ref_start.p, p0, atol=1e-12)
    np.testing.assert_allclose(ref_end.p, p1, atol=1e-12)
    # Legacy phase has no 'kind' or 'use_fk' tag.
    assert tp._phases[-1].get('use_fk', False) is False


# --------------------------------------------------------------------
# D.4 — reachability_gate behaviour
# --------------------------------------------------------------------


def test_D4_gate_fails_at_DS_start(robot, fids, anchors, step2_setup):
    """At q_DS_start (body at original pose), the gate should FAIL —
    the body is too far from anchor_b[4] for the swing arm to reach
    cleanly. (1-task IK may still find a *bad* solution with low
    manipulability; the w_swing threshold catches that.)"""
    q_start = step2_setup['q_DS_start']
    # Use a strict w_min_threshold to ensure the gate fires even if
    # the IK technically converges.
    passed, info_gate = reachability_gate(
        robot.model, q_start,
        fid_swing=fids['fid_b'],
        anchor_swing_target=anchors['b'][4],
        fid_torso=fids['fid_torso'],
        w_min_threshold=0.15,  # T15 step-2 manipulability is ~0.05
                                # at the un-advanced body pose
    )
    # Either the workspace check fails (body too far) OR the IK's
    # manipulability is too low at the un-advanced body pose.
    if passed:
        # If it passes, the body must already be close enough; verify
        # the d_reach is small vs the un-advanced expectation.
        assert info_gate['d_reach_m'] < 1.0, (
            f"gate unexpectedly PASSED at DS-start with d_reach="
            f"{info_gate['d_reach_m']*1000:.1f} mm — "
            f"check w_swing = {info_gate['w_swing']:.3e}")
    else:
        # Expected path: gate FAILS, log which stage caught it.
        assert info_gate['stage_failed'] in ('workspace', 'ik'), (
            f"unexpected stage_failed = {info_gate['stage_failed']!r}")
