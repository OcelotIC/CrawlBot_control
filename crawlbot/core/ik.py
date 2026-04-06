"""Inverse kinematics for VISPA docking configurations.

Provides:
    solve_ik           — iterative 6D IK for multiple frame targets
    dock_configuration — convenience: both tools at anchor poses
    solve_ik_waypoints — chain of IK solutions along a swing arc
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import pinocchio as pin

from crawlbot.core.robot_interface import FRAME_TOOL_A, FRAME_TOOL_B


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

            # Arm-specific joint slice
            if fid == FRAME_TOOL_A:
                idx = slice(6, 12)
            else:
                idx = slice(12, 18)

            # Arm joints: primary contribution (low regularization)
            Ja = J[:, idx]
            dq[idx] += np.linalg.solve(
                Ja.T @ Ja + 1e-4 * np.eye(6), Ja.T @ err)

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

    # Determine which arm slice corresponds to stance/swing
    if stance_frame == FRAME_TOOL_A:
        stance_idx = slice(6, 12)
        swing_idx = slice(12, 18)
    else:
        stance_idx = slice(12, 18)
        swing_idx = slice(6, 12)

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
                Ja_s.T @ Ja_s + 1e-4 * np.eye(6), Ja_s.T @ err_s)
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
                Ja_p.T @ Ja_p + 1e-4 * np.eye(6), Ja_p.T @ err_p)
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
