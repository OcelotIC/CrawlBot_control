"""SE(3) adjoint wrench mapping for contact frames.

Computes the 6x6 adjoint matrices that map a wrench expressed at each
contact (gripper) frame to the equivalent wrench at the satellite/structure
center of mass.  This is the Ad_{g_cb}^T from Lutze et al. (2023) eq. 18,
extended to dual contacts.
"""

import numpy as np
import pinocchio as pin


def skew(v: np.ndarray) -> np.ndarray:
    """3-vector → 3x3 skew-symmetric matrix."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def compute_contact_adjoint(oMf: pin.SE3) -> np.ndarray:
    """Compute 6x6 adjoint matrix for wrench mapping from contact to world.

    Given the placement oMf of a contact frame in the world, this returns
    the matrix A such that:

        F_world = A @ F_contact

    where F = [torque(3); force(3)] is a spatial wrench.

    This corresponds to Ad_{g}^{-T} (inverse-transpose of the adjoint of
    the transform from world to contact), which is the co-adjoint for
    wrenches.

    Parameters
    ----------
    oMf : pin.SE3
        Placement of the contact frame in world coordinates.

    Returns
    -------
    A : (6, 6) ndarray
        Wrench mapping matrix: F_world = A @ F_contact.
    """
    R = oMf.rotation       # (3,3) contact → world rotation
    p = oMf.translation    # (3,) position of contact in world frame

    # Wrench transformation (co-adjoint):
    # [τ_w]   [R,   [p]× R] [τ_c]
    # [f_w] = [0,      R   ] [f_c]
    A = np.zeros((6, 6))
    A[:3, :3] = R
    A[:3, 3:] = skew(p) @ R
    A[3:, 3:] = R
    return A


def compute_dual_contact_adjoints(rs, active_a: bool, active_b: bool):
    """Compute adjoint matrices for both contacts.

    Parameters
    ----------
    rs : RobotState
        Current robot state from Pinocchio.
    active_a : bool
        Whether gripper A is in contact.
    active_b : bool
        Whether gripper B is in contact.

    Returns
    -------
    Ad_a : (6, 6) or None
        Wrench mapping for contact A (world frame).
    Ad_b : (6, 6) or None
        Wrench mapping for contact B (world frame).
    """
    Ad_a = compute_contact_adjoint(rs.oMf_tool_a) if active_a else None
    Ad_b = compute_contact_adjoint(rs.oMf_tool_b) if active_b else None
    return Ad_a, Ad_b
