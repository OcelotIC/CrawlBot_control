"""Inverse kinematics for VISPA docking configurations.

Provides:
    solve_ik              — iterative 6D IK for multiple frame targets
    dock_configuration    — convenience: both tools at anchor poses
    manipulability_config — optimal-manipulability configuration for a dock pair
    precompute_torso_map  — offline map of optimal configs for all anchor pairs
    solve_ik_waypoints    — chain of IK solutions along a swing arc
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import pinocchio as pin
from scipy.optimize import minimize as scipy_minimize

from crawlbot.core.robot_interface import FRAME_TOOL_A, FRAME_TOOL_B, _detect_arm_slices

# Cache arm slices per model (by id) to avoid recomputing
_arm_slice_cache: Dict[int, dict] = {}

def _get_arm_slices(model: pin.Model) -> dict:
    """Get arm velocity-space slices for a model, cached."""
    mid = id(model)
    if mid not in _arm_slice_cache:
        _arm_slice_cache[mid] = _detect_arm_slices(model)
    return _arm_slice_cache[mid]


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

    targets = {FRAME_TOOL_A: anchor_a, FRAME_TOOL_B: anchor_b}
    q, err = solve_ik(model, q0, targets, max_iter=1000)
    if err > 1e-4:
        raise RuntimeError(f"IK failed to converge: err={err:.2e}")
    return q


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
    targets = {FRAME_TOOL_A: anchor_a, FRAME_TOOL_B: anchor_b}
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

            sl_a = _arm_v_slice(model, FRAME_TOOL_A)
            sl_b = _arm_v_slice(model, FRAME_TOOL_B)
            Ja = pin.getFrameJacobian(
                model, data, FRAME_TOOL_A, pin.LOCAL)[:, sl_a]
            Jb = pin.getFrameJacobian(
                model, data, FRAME_TOOL_B, pin.LOCAL)[:, sl_b]
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

    # Recover the optimal configuration
    q0 = pin.neutral(model)
    q0[:3] = best_result.x
    q_opt, err = solve_ik(model, q0, targets, max_iter=1000)
    if err > 1e-4:
        # Fallback: use midpoint
        q_opt = dock_configuration(model, anchor_a, anchor_b)

    # Compute final manipulability
    pin.forwardKinematics(model, data, q_opt)
    pin.updateFramePlacements(model, data)
    pin.computeJointJacobians(model, data, q_opt)
    sl_a = _arm_v_slice(model, FRAME_TOOL_A)
    sl_b = _arm_v_slice(model, FRAME_TOOL_B)
    Ja = pin.getFrameJacobian(model, data, FRAME_TOOL_A, pin.LOCAL)[:, sl_a]
    Jb = pin.getFrameJacobian(model, data, FRAME_TOOL_B, pin.LOCAL)[:, sl_b]
    w_a = np.sqrt(max(np.linalg.det(Ja @ Ja.T), 0.0))
    w_b = np.sqrt(max(np.linalg.det(Jb @ Jb.T), 0.0))

    return q_opt, w_a * w_b


def precompute_torso_map(
    model: pin.Model,
    anchors_a: np.ndarray,
    anchors_b: np.ndarray,
) -> Dict[Tuple[int, int], np.ndarray]:
    """Precompute optimal configurations for all anchor pairs.

    Returns a lookup table mapping (anchor_a_idx, anchor_b_idx) to the
    configuration that maximizes combined arm manipulability with both
    tools at their respective anchors.

    Parameters
    ----------
    model : pin.Model
    anchors_a : (N, 3) array of anchor positions for tool A.
    anchors_b : (M, 3) array of anchor positions for tool B.

    Returns
    -------
    torso_map : dict
        {(a_idx, b_idx): q_opt} for each feasible pair.
    """
    torso_map = {}
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
