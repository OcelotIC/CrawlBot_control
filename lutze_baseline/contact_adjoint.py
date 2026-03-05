"""SE(3) adjoint wrench mapping for contact frames.

Computes the 6x6 adjoint matrices that map a wrench expressed at each
contact (gripper) frame to the equivalent wrench at the satellite/structure
center of mass.
"""

import numpy as np
import pinocchio as pin


def skew(v):
    """3-vector -> 3x3 skew-symmetric matrix."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def compute_contact_adjoint(oMf):
    """Compute 6x6 adjoint matrix for wrench mapping from contact to world.

    Given the placement oMf of a contact frame in the world, this returns
    the matrix A such that:

        F_world = A @ F_contact

    where F = [torque(3); force(3)] in spatial wrench convention.

    Parameters
    ----------
    oMf : pin.SE3
        Contact frame placement in world.

    Returns
    -------
    A : (6, 6) wrench co-adjoint matrix.
    """
    R = oMf.rotation
    p = oMf.translation
    A = np.zeros((6, 6))
    A[:3, :3] = R              # torque -> torque
    A[:3, 3:] = skew(p) @ R   # force -> torque (lever arm)
    A[3:, 3:] = R              # force -> force
    return A


def compute_dual_contact_adjoints(model, data, q, active_a=True, active_b=True):
    """Compute adjoint matrices for both contacts.

    Parameters
    ----------
    model : pin.Model
    data : pin.Data
    q : (nq,) joint configuration.
    active_a : bool
        Whether gripper A is in contact.
    active_b : bool
        Whether gripper B is in contact.

    Returns
    -------
    Ad_a : (6, 6) or None
    Ad_b : (6, 6) or None
    """
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    Ad_a = None
    Ad_b = None

    if active_a:
        fid_a = model.getFrameId('tool_a')
        Ad_a = compute_contact_adjoint(data.oMf[fid_a])

    if active_b:
        fid_b = model.getFrameId('tool_b')
        Ad_b = compute_contact_adjoint(data.oMf[fid_b])

    return Ad_a, Ad_b
