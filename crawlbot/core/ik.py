"""Inverse kinematics for VISPA docking configurations.

Provides:
    solve_ik              — iterative 6D IK for multiple frame targets
    dock_configuration    — convenience: both tools at anchor poses
    manipulability_config — optimal-manipulability configuration for a dock pair
    manipulability_config_trajectory — trajectory-aware variant (M7 / Candidate 1)
    precompute_torso_map  — offline map of optimal configs for all anchor pairs
    solve_ik_waypoints    — chain of IK solutions along a swing arc
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import pinocchio as pin
from scipy.optimize import minimize as scipy_minimize

from crawlbot.core.robot_interface import FRAME_TOOL_A, FRAME_TOOL_B, _detect_arm_slices


def _get_tool_frames(model: pin.Model):
    """Get tool frame IDs from model by name lookup."""
    return model.getFrameId("tool_a"), model.getFrameId("tool_b")

# Cache arm slices per model (by id) to avoid recomputing
_arm_slice_cache: Dict[int, dict] = {}

def _get_arm_slices(model: pin.Model) -> dict:
    """Get arm velocity-space slices for a model, cached by (id, nv)."""
    key = (id(model), model.nv)
    if key not in _arm_slice_cache:
        _arm_slice_cache[key] = _detect_arm_slices(model)
    return _arm_slice_cache[key]


def _arm_v_slice(model: pin.Model, frame_id: int) -> slice:
    """Return the velocity-space slice for the arm owning frame_id."""
    slices = _get_arm_slices(model)
    # Determine which arm owns this frame by checking if the frame's
    # parent joint falls within arm A or arm B's joint range
    pj = model.frames[frame_id].parentJoint
    a_start = slices['arm_a_v'].start
    a_stop = slices['arm_a_v'].stop
    b_start = slices['arm_b_v'].start
    b_stop = slices['arm_b_v'].stop
    # Parent joint's velocity index
    jv = model.idx_vs[pj]
    if a_start <= jv < a_stop:
        return slices['arm_a_v']
    elif b_start <= jv < b_stop:
        return slices['arm_b_v']
    else:
        raise ValueError(f"Frame {frame_id} parent joint {pj} not in arm A or B")


def solve_ik(
    model: pin.Model,
    q0: np.ndarray,
    targets: Dict[int, pin.SE3],
    max_iter: int = 500,
    tol: float = 1e-8,
    base_gain: float = 0.3,
) -> Tuple[np.ndarray, float]:
    """
    Iterative IK placing tool frames at target SE3 poses.

    Uses damped least-squares on the full Jacobian (base + arm joints)
    for each target. When multiple targets are specified, contributions
    are summed — no priority ordering.

    Parameters
    ----------
    model : pin.Model
        Pinocchio model (free-flyer).
    q0 : (nq,) initial configuration guess.
    targets : {frame_id: SE3}
        e.g. {FRAME_TOOL_A: anchor_3a, FRAME_TOOL_B: anchor_3b}
    max_iter : int
    tol : float
        Convergence on sum of ||log6(err)||.

    Returns
    -------
    q : (nq,) converged configuration
    err : float, final error norm
    """
    q = q0.copy()
    data = model.createData()
    nv = model.nv

    for it in range(max_iter):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        pin.computeJointJacobians(model, data, q)
        dq = np.zeros(nv)
        err_tot = 0.0

        for fid, tgt in targets.items():
            err = pin.log6(data.oMf[fid].actInv(tgt)).vector
            J = pin.getFrameJacobian(model, data, fid, pin.LOCAL)

            # Arm-specific joint slice (DOF-generic)
            idx = _arm_v_slice(model, fid)
            n_arm = idx.stop - idx.start

            # Arm joints: primary contribution (low regularization)
            Ja = J[:, idx]
            dq[idx] += np.linalg.solve(
                Ja.T @ Ja + 1e-4 * np.eye(n_arm), Ja.T @ err)

            # Base (free-flyer): scaled by base_gain to control how much
            # the base moves vs the arm. Default 0.3 is conservative (good
            # from neutral). Use 1.0 for chain-seeded IK where the base
            # needs to advance naturally.
            Jb = J[:, :6]
            dq[:6] += np.linalg.solve(
                Jb.T @ Jb + 1e-3 * np.eye(6), Jb.T @ err) * base_gain

            err_tot += np.linalg.norm(err)

        alpha = min(1.0, 0.5 / max(np.max(np.abs(dq)), 1e-10))
        q = pin.integrate(model, q, alpha * dq)

        if err_tot < tol:
            break

    return q, err_tot


def dock_configuration(
    model: pin.Model,
    anchor_a: pin.SE3,
    anchor_b: pin.SE3,
    torso_pos: np.ndarray = None,
    q_init: np.ndarray = None,
) -> np.ndarray:
    """
    Convenience: compute a valid configuration with both tools at anchors.

    Parameters
    ----------
    model : pin.Model
    anchor_a, anchor_b : SE3 target poses for tool_a, tool_b
    torso_pos : (3,) initial torso position guess (default: midpoint of anchors)
    q_init : (nq,) full configuration to use as seed. If provided, used
             instead of neutral + torso_pos. This ensures the IK converges
             to the same branch as the current robot configuration.

    Returns
    -------
    q : (nq,) valid docking configuration
    """
    if q_init is not None:
        q0 = q_init.copy()
    else:
        q0 = pin.neutral(model)
    if torso_pos is None and q_init is None:
        torso_pos = 0.5 * (anchor_a.translation + anchor_b.translation)
    if torso_pos is not None:
        q0[:3] = torso_pos

    fid_a, fid_b = _get_tool_frames(model)
    targets = {fid_a: anchor_a, fid_b: anchor_b}
    q, err = solve_ik(model, q0, targets, max_iter=2000)
    if err > 1e-4:
        raise RuntimeError(f"IK failed to converge: err={err:.2e}")
    return q


def dock_configuration_fixed_rotation(
    model: pin.Model,
    anchor_a: pin.SE3,
    anchor_b: pin.SE3,
    R_torso_fixed: np.ndarray,
    torso_pos: np.ndarray = None,
    q_init: np.ndarray = None,
    max_iter: int = 2000,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, float, float]:
    """IK with torso rotation held at R_torso_fixed (M7 / change A).

    Solves for (torso_position, arm joints) such that both tools reach
    their anchor poses while the torso orientation stays at
    `R_torso_fixed`. Per the M7 crawling philosophy: the robot
    translates between steps and only reorients when the geometry
    strictly demands it.

    The torso has 3 free DOFs (position) plus 2*7=14 arm DOFs = 17
    DOFs to satisfy 12 constraints (2 × SE3), leaving a 5-dim null
    space — enough slack for a good solution when one exists.

    Parameters
    ----------
    R_torso_fixed : (3,3)
        Desired torso rotation matrix (structure frame).
    torso_pos : (3,), optional
        Initial guess for torso position. Defaults to the midpoint of
        the two anchors.
    q_init : (nq,), optional
        Full configuration seed. Its torso rotation is overwritten by
        `R_torso_fixed`.

    Returns
    -------
    q : (nq,) converged configuration (torso rotation = R_torso_fixed)
    err : float — final residual norm
    w_product : float — manipulability product w_a * w_b at q
    """
    # Seed q0 with the fixed torso rotation. Pinocchio free-flyer uses
    # quaternion (x, y, z, w) at q[3:7].
    q0 = q_init.copy() if q_init is not None else pin.neutral(model)
    if torso_pos is None:
        torso_pos = 0.5 * (anchor_a.translation + anchor_b.translation)
    q0[:3] = torso_pos
    quat = pin.Quaternion(R_torso_fixed)
    q0[3] = quat.x
    q0[4] = quat.y
    q0[5] = quat.z
    q0[6] = quat.w

    fid_a, fid_b = _get_tool_frames(model)
    targets = {fid_a: anchor_a, fid_b: anchor_b}

    q = q0.copy()
    data = model.createData()
    nv = model.nv

    # Custom IK loop: same as solve_ik but zero out base angular dq
    # at every iteration to hold the rotation fixed.
    for it in range(max_iter):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        pin.computeJointJacobians(model, data, q)
        dq = np.zeros(nv)
        err_tot = 0.0

        for fid, tgt in targets.items():
            err = pin.log6(data.oMf[fid].actInv(tgt)).vector
            J = pin.getFrameJacobian(model, data, fid, pin.LOCAL)

            # Arm joints: primary contribution
            idx = _arm_v_slice(model, fid)
            n_arm = idx.stop - idx.start
            Ja = J[:, idx]
            dq[idx] += np.linalg.solve(
                Ja.T @ Ja + 1e-4 * np.eye(n_arm), Ja.T @ err)

            # Base: only the *linear* 3 DOFs are free (drop angular cols).
            Jb_lin = J[:, :3]
            dq[:3] += np.linalg.solve(
                Jb_lin.T @ Jb_lin + 1e-3 * np.eye(3),
                Jb_lin.T @ err) * 0.3

            err_tot += np.linalg.norm(err)

        # Integrate; base angular dq stays zero so rotation is held.
        alpha = min(1.0, 0.5 / max(np.max(np.abs(dq)), 1e-10))
        q = pin.integrate(model, q, alpha * dq)

        if err_tot < tol:
            break

    # Manipulability product (Yoshikawa, combined) at the solution.
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pin.computeJointJacobians(model, data, q)
    sl_a = _arm_v_slice(model, fid_a)
    sl_b = _arm_v_slice(model, fid_b)
    Ja = pin.getFrameJacobian(model, data, fid_a, pin.LOCAL)[:, sl_a]
    Jb = pin.getFrameJacobian(model, data, fid_b, pin.LOCAL)[:, sl_b]
    w_a = float(np.sqrt(max(np.linalg.det(Ja @ Ja.T), 0.0)))
    w_b = float(np.sqrt(max(np.linalg.det(Jb @ Jb.T), 0.0)))
    return q, float(err_tot), w_a * w_b


def manipulability_config(
    model: pin.Model,
    anchor_a: pin.SE3,
    anchor_b: pin.SE3,
) -> Tuple[np.ndarray, float]:
    """Find the configuration maximizing combined arm manipulability.

    Optimizes the torso (base) position to maximize the product of
    Yoshikawa manipulability indices for both arms, subject to both
    tools reaching their respective anchors via IK.

    Parameters
    ----------
    model : pin.Model
    anchor_a, anchor_b : SE3
        Target poses for tool_a and tool_b.

    Returns
    -------
    q_opt : ndarray (nq,)
        Optimal configuration.
    w_product : float
        Manipulability product w_a * w_b at the optimum.
    """
    data = model.createData()
    fid_a, fid_b = _get_tool_frames(model)
    targets = {fid_a: anchor_a, fid_b: anchor_b}
    midpoint = 0.5 * (anchor_a.translation + anchor_b.translation)

    # Cache for IK solutions at each torso position.
    # Reuse previous solution as seed to avoid singularity traps.
    _cache = {'q_prev': None}

    def cost(torso_xyz):
        q0 = pin.neutral(model)
        q0[:3] = torso_xyz
        # Also try seeding from previous successful solve
        candidates = [q0]
        if _cache['q_prev'] is not None:
            q_seed = _cache['q_prev'].copy()
            q_seed[:3] = torso_xyz
            candidates.insert(0, q_seed)

        best_score = -1.0
        best_q = None
        for q_try in candidates:
            q, err = solve_ik(model, q_try, targets, max_iter=500)
            if err > 1e-3:
                continue
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            pin.computeJointJacobians(model, data, q)

            sl_a = _arm_v_slice(model, fid_a)
            sl_b = _arm_v_slice(model, fid_b)
            Ja = pin.getFrameJacobian(
                model, data, fid_a, pin.LOCAL)[:, sl_a]
            Jb = pin.getFrameJacobian(
                model, data, fid_b, pin.LOCAL)[:, sl_b]
            # Minimum singular value: distance from singularity
            sigma_a = np.linalg.svd(Ja, compute_uv=False)[-1]
            sigma_b = np.linalg.svd(Jb, compute_uv=False)[-1]
            score = sigma_a * sigma_b
            if score > best_score:
                best_score = score
                best_q = q

        if best_q is not None:
            _cache['q_prev'] = best_q
        return -best_score if best_score > 0 else 1e6

    # Run Nelder-Mead from multiple starting points to escape singularities
    best_result = None
    best_cost = 1e6
    for dz in [0.0, -0.3, -0.6]:
        x0 = midpoint.copy()
        x0[2] += dz
        result = scipy_minimize(cost, x0, method='Nelder-Mead',
                                options={'xatol': 1e-3, 'fatol': 1e-8,
                                         'maxiter': 200, 'adaptive': True})
        if result.fun < best_cost:
            best_cost = result.fun
            best_result = result
        _cache['q_prev'] = None  # reset cache between starts

    if best_result is None:
        raise RuntimeError(
            "manipulability_config: all multi-starts produced IK failure "
            "(infeasible anchor pair)."
        )

    # Recover the optimal configuration
    q0 = pin.neutral(model)
    q0[:3] = best_result.x
    q_opt, err = solve_ik(model, q0, targets, max_iter=2000)
    if err > 1e-4:
        # Fallback: use midpoint
        q_opt = dock_configuration(model, anchor_a, anchor_b)

    # Compute final manipulability
    pin.forwardKinematics(model, data, q_opt)
    pin.updateFramePlacements(model, data)
    pin.computeJointJacobians(model, data, q_opt)
    sl_a = _arm_v_slice(model, fid_a)
    sl_b = _arm_v_slice(model, fid_b)
    Ja = pin.getFrameJacobian(model, data, fid_a, pin.LOCAL)[:, sl_a]
    Jb = pin.getFrameJacobian(model, data, fid_b, pin.LOCAL)[:, sl_b]
    w_a = np.sqrt(max(np.linalg.det(Ja @ Ja.T), 0.0))
    w_b = np.sqrt(max(np.linalg.det(Jb @ Jb.T), 0.0))

    return q_opt, w_a * w_b


def precompute_torso_map(
    model: pin.Model,
    anchors_a: np.ndarray,
    anchors_b: np.ndarray,
    *,
    anchor_pair_sequence: Optional[List[Tuple[int, int]]] = None,
    q_initial: Optional[np.ndarray] = None,
    n_samples: int = 5,
    use_trajectory_aware: bool = False,
) -> Dict[Tuple[int, int], object]:
    """Precompute optimal configurations for all anchor pairs.

    Two modes:

    * use_trajectory_aware=False (default, backwards-compatible):
        returns {(ai, bi): q_opt: ndarray} for the full grid product.
        This is the endpoint-only behaviour and matches the legacy signature.

    * use_trajectory_aware=True:
        walks ``anchor_pair_sequence`` in order, chaining q_start from
        the previous entry's q_end. Seeds from ``q_initial`` (required).
        Returns {(ai, bi): {'q_end', 'q_start_assumed', 'w_worst', 'w_end'}}.
        For repeated pairs, the last occurrence overwrites earlier ones
        — earlier occurrences will hit the drift-tolerance fallback at
        runtime (which is the intended graceful-degradation behaviour).

    Parameters
    ----------
    model : pin.Model
    anchors_a : (N, 3) array of anchor positions for tool A.
    anchors_b : (M, 3) array of anchor positions for tool B.
    anchor_pair_sequence : ordered list of (ai, bi) pairs (trajectory mode only).
    q_initial : (nq,) seed for the first chained pair (trajectory mode only).
        Typically the endpoint-IK solution for the first pair.
    n_samples : K — number of τ∈(0,1] samples (trajectory mode only).
    use_trajectory_aware : enable chained trajectory-aware IK.

    Returns
    -------
    torso_map : dict
        Endpoint mode: {(a_idx, b_idx): q_opt}.
        Trajectory mode: {(a_idx, b_idx): {q_end, q_start_assumed, w_worst, w_end}}.
    """
    if not use_trajectory_aware:
        torso_map: Dict[Tuple[int, int], object] = {}
        for ai in range(len(anchors_a)):
            for bi in range(len(anchors_b)):
                se3_a = pin.SE3(np.eye(3), anchors_a[ai].copy())
                se3_b = pin.SE3(np.eye(3), anchors_b[bi].copy())
                try:
                    q_opt, w = manipulability_config(model, se3_a, se3_b)
                    torso_map[(ai, bi)] = q_opt
                except RuntimeError:
                    pass  # infeasible pair — skip
        return torso_map

    # Trajectory-aware chained mode.
    if anchor_pair_sequence is None or q_initial is None:
        raise ValueError(
            "precompute_torso_map(use_trajectory_aware=True) requires "
            "both `anchor_pair_sequence` and `q_initial`."
        )

    traj_map: Dict[Tuple[int, int], object] = {}
    q_start = np.asarray(q_initial, dtype=float).copy()
    for (ai, bi) in anchor_pair_sequence:
        a_pos = anchors_a[ai]
        b_pos = anchors_b[bi]
        try:
            q_end, w_worst, w_end = manipulability_config_trajectory(
                model, a_pos, b_pos, q_start, n_samples=n_samples,
            )
        except RuntimeError:
            # Infeasible: skip this entry so runtime falls back to endpoint map.
            continue
        traj_map[(ai, bi)] = {
            'q_end': q_end,
            'q_start_assumed': q_start.copy(),
            'w_worst': float(w_worst),
            'w_end': float(w_end),
        }
        q_start = q_end  # chain
    return traj_map


def _interpolate_q_quintic(
    model: pin.Model,
    q_start: np.ndarray,
    q_end: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Quintic time-scaling on the configuration manifold.

    s(τ) = 10τ³ − 15τ⁴ + 6τ⁵  (C² with s(0)=0, s(1)=1, ṡ(0)=ṡ(1)=s̈(0)=s̈(1)=0)

    The time-scaled parameter is then passed to ``pin.interpolate``, which
    uses the Lie-group structure of the free-flyer base (SE(3) geodesic)
    plus linear interpolation for revolute joints.

    Matches the joint-space quintic used in
    ``sim_loop._planned_arm_config`` for SS swing timing.
    """
    s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    return pin.interpolate(model, q_start, q_end, s)


def _trajectory_worst_w(
    model: pin.Model,
    data,
    q_start: np.ndarray,
    q_end: np.ndarray,
    n_samples: int,
    fid_a: int,
    fid_b: int,
    sl_a: slice,
    sl_b: slice,
) -> Tuple[float, float]:
    """Return (w_worst, w_end): worst-case and endpoint σ_min products.

    Samples τ ∈ {1/K, 2/K, …, 1.0} (τ=0 is q_start, which is fixed input
    and common to all candidates, so excluded from the optimised set).
    """
    w_worst = np.inf
    w_end = 0.0
    for k in range(1, n_samples + 1):
        tau = k / n_samples
        q_k = _interpolate_q_quintic(model, q_start, q_end, tau)
        pin.forwardKinematics(model, data, q_k)
        pin.updateFramePlacements(model, data)
        pin.computeJointJacobians(model, data, q_k)
        Ja = pin.getFrameJacobian(model, data, fid_a, pin.LOCAL)[:, sl_a]
        Jb = pin.getFrameJacobian(model, data, fid_b, pin.LOCAL)[:, sl_b]
        sigma_a = float(np.linalg.svd(Ja, compute_uv=False)[-1])
        sigma_b = float(np.linalg.svd(Jb, compute_uv=False)[-1])
        w_k = sigma_a * sigma_b
        if w_k < w_worst:
            w_worst = w_k
        if k == n_samples:
            w_end = w_k
    return float(w_worst), float(w_end)


def manipulability_config_trajectory(
    model: pin.Model,
    anchor_a: np.ndarray,
    anchor_b: np.ndarray,
    q_start: np.ndarray,
    n_samples: int = 5,
    q_guess: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float]:
    """Trajectory-aware manipulability IK (M7 / Manipulability-IK-1).

    Optimise torso xyz to maximise the **worst-case** σ_min(J_a)·σ_min(J_b)
    across K interior samples of the planned quintic interpolation from
    q_start to the docked end configuration. Trades a small amount of
    endpoint optimality for guarantees on the interior trajectory —
    addresses the near-singular interior configurations flagged in the
    T15 post-5 audit §8.4 (Candidate 1).

    Decision variables and multi-start strategy match ``manipulability_config``
    (torso xyz, dz ∈ {0.0, −0.3, −0.6}, Nelder-Mead adaptive). The arm
    joints are solved inside the cost via ``solve_ik``.

    Parameters
    ----------
    model : pin.Model
    anchor_a, anchor_b : (3,) world positions for tool A / tool B anchors.
    q_start : (nq,) fixed τ=0 configuration (chained from the previous step).
    n_samples : K — samples at τ ∈ {1/K, 2/K, …, 1.0}.
    q_guess : optional (nq,) extra seed for the internal IK.

    Returns
    -------
    q_end : (nq,) optimal end configuration.
    w_worst : float — worst-case σ_min product across the K samples (maximised).
    w_end : float — endpoint-only σ_min product at q_end (diagnostic).
    """
    se3_a = pin.SE3(np.eye(3), np.asarray(anchor_a, dtype=float).copy())
    se3_b = pin.SE3(np.eye(3), np.asarray(anchor_b, dtype=float).copy())
    data = model.createData()
    fid_a, fid_b = _get_tool_frames(model)
    targets = {fid_a: se3_a, fid_b: se3_b}
    midpoint = 0.5 * (se3_a.translation + se3_b.translation)
    q_start = np.asarray(q_start, dtype=float)
    sl_a = _arm_v_slice(model, fid_a)
    sl_b = _arm_v_slice(model, fid_b)

    # IK 3 inner-solve seed: q_start with torso xyz overwritten.
    # The seed is deterministic in the decision variable (torso xyz)
    # and identical across all cost evaluations within a single IK
    # invocation. This makes cost(p_t) a deterministic function of
    # p_t and eliminates pathology (C) from IK_ANOMALY_REPORT §3.1,
    # §5.3 (warm-start `_cache['q_prev']` allowed cost(xyz) to vary
    # by 7+ orders of magnitude depending on the prior Nelder-Mead
    # path). Spec: docs/architecture/IK_FORMULATION.md §9.1.
    def cost(torso_xyz):
        q_seed = q_start.copy()
        q_seed[:3] = torso_xyz
        q_end, err = solve_ik(model, q_seed, targets, max_iter=500)
        if err > 1e-3:
            return 1e6
        w_worst, _ = _trajectory_worst_w(
            model, data, q_start, q_end, n_samples, fid_a, fid_b, sl_a, sl_b,
        )
        return -w_worst if w_worst > 0 else 1e6

    best_result = None
    best_cost = 1e6
    for dz in [0.0, -0.3, -0.6]:
        x0 = midpoint.copy()
        x0[2] += dz
        result = scipy_minimize(
            cost, x0, method='Nelder-Mead',
            options={'xatol': 1e-3, 'fatol': 1e-8,
                     'maxiter': 200, 'adaptive': True},
        )
        if result.fun < best_cost:
            best_cost = result.fun
            best_result = result

    if best_result is None or best_cost >= 1e6:
        raise RuntimeError(
            "manipulability_config_trajectory: all multi-starts produced "
            "IK failure (infeasible anchor pair)."
        )

    # Recover the optimal q_end with the same deterministic seed used
    # in the cost (q_start with torso xyz overwritten); no fallback
    # to dock_configuration which would change the kinematic branch.
    q_seed = q_start.copy()
    q_seed[:3] = best_result.x
    q_end, err = solve_ik(model, q_seed, targets, max_iter=2000)
    if err > 1e-4:
        q_end = dock_configuration(model, se3_a, se3_b)

    w_worst, w_end = _trajectory_worst_w(
        model, data, q_start, q_end, n_samples, fid_a, fid_b, sl_a, sl_b,
    )
    return q_end, float(w_worst), float(w_end)


def solve_ik_waypoints(
    model: pin.Model,
    q_start: np.ndarray,
    stance_frame: int,
    stance_target: pin.SE3,
    swing_frame: int,
    swing_start: np.ndarray,
    swing_end: np.ndarray,
    n_waypoints: int = 10,
    clearance: float = 0.08,
    away_normal: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """Compute a chain of IK solutions along a swing arc.

    Produces n_waypoints+1 configurations from swing_start to swing_end.
    Each waypoint is seeded from the previous one (chain seeding), ensuring
    smooth configuration transitions with no branch jumps.

    The swing arm follows a straight line with a clearance bump:
        p(s) = (1-s)*start + s*end + clearance * sin²(π*s) * away_normal

    Parameters
    ----------
    model : pin.Model
        Pinocchio model (free-flyer).
    q_start : ndarray (nq,)
        Starting configuration (current robot state).
    stance_frame : int
        Frame ID of the stance (docked) tool (FRAME_TOOL_A or FRAME_TOOL_B).
    stance_target : pin.SE3
        SE3 pose of the stance anchor (stays fixed throughout).
    swing_frame : int
        Frame ID of the swing (free) tool.
    swing_start : ndarray (3,)
        Starting position of the swing tool (departure anchor).
    swing_end : ndarray (3,)
        Target position of the swing tool (arrival anchor).
    n_waypoints : int
        Number of intermediate waypoints (total configs = n_waypoints + 1).
    clearance : float
        Maximum clearance height of the swing arc [m].
    away_normal : ndarray (3,), optional
        Direction of the clearance bump. Default: [0, 0, -1] (away from structure).

    Returns
    -------
    q_chain : list of ndarray (nq,)
        Chain of n_waypoints+1 configurations.
        q_chain[0] ≈ q_start (re-converged at swing_start).
        q_chain[-1] has swing tool at swing_end.
    """
    if away_normal is None:
        away_normal = np.array([0.0, 0.0, -1.0])
    away_normal = np.asarray(away_normal, dtype=float)

    q_chain = []
    q_prev = q_start.copy()
    data = model.createData()

    # Determine which arm slice corresponds to stance/swing (DOF-generic)
    stance_idx = _arm_v_slice(model, stance_frame)
    swing_idx = _arm_v_slice(model, swing_frame)
    n_stance = stance_idx.stop - stance_idx.start
    n_swing = swing_idx.stop - swing_idx.start

    for i in range(n_waypoints + 1):
        s = i / n_waypoints

        # Swing position with clearance bump
        p_swing = ((1.0 - s) * swing_start + s * swing_end
                   + clearance * np.sin(np.pi * s) ** 2 * away_normal)

        # Iterative IK: stance arm 6D (full SE3), swing arm 3D (position only).
        # This is a custom loop rather than calling solve_ik, because we need
        # asymmetric targets (6D stance + 3D swing).
        q = q_prev.copy()
        for it in range(300):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            pin.computeJointJacobians(model, data, q)
            dq = np.zeros(model.nv)
            err_tot = 0.0

            # Stance arm: full 6D constraint (keep at anchor)
            err_s = pin.log6(data.oMf[stance_frame].actInv(stance_target)).vector
            J_s = pin.getFrameJacobian(model, data, stance_frame, pin.LOCAL)
            Ja_s = J_s[:, stance_idx]
            dq[stance_idx] = np.linalg.solve(
                Ja_s.T @ Ja_s + 1e-4 * np.eye(n_stance), Ja_s.T @ err_s)
            Jb_s = J_s[:, :6]
            dq[:6] += np.linalg.solve(
                Jb_s.T @ Jb_s + 1e-3 * np.eye(6), Jb_s.T @ err_s) * 0.3
            err_tot += np.linalg.norm(err_s)

            # Swing arm: 3D position constraint only
            p_cur = data.oMf[swing_frame].translation
            err_p = p_swing - p_cur
            J_sw = pin.getFrameJacobian(
                model, data, swing_frame, pin.LOCAL_WORLD_ALIGNED)
            J_sw_pos = J_sw[:3, :]  # position rows only
            Ja_p = J_sw_pos[:, swing_idx]
            dq[swing_idx] += np.linalg.solve(
                Ja_p.T @ Ja_p + 1e-4 * np.eye(n_swing), Ja_p.T @ err_p)
            Jb_p = J_sw_pos[:, :6]
            dq[:6] += np.linalg.solve(
                Jb_p.T @ Jb_p + 1e-3 * np.eye(6), Jb_p.T @ err_p) * 0.5
            err_tot += np.linalg.norm(err_p)

            alpha = min(1.0, 0.5 / max(np.max(np.abs(dq)), 1e-10))
            q = pin.integrate(model, q, alpha * dq)

            if err_tot < 1e-4:
                break

        q_chain.append(q)
        q_prev = q

    return q_chain
