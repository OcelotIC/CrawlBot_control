"""Contact Jacobian computation.

Translated from compute_contact_jacobian.m.
"""

from ..spart.jacobian import geometric_jacobian


def compute_contact_jacobian(r_contact, r0, rL, P0, pm, robot):
    """Compute the 6D Jacobian at the contact point.

    Uses SPART's geometric Jacobian for the last link.

    Parameters
    ----------
    r_contact : (3,) – Contact point position (inertial).
    r0 : (3,) – Base-link position.
    rL : (3, n) – Link positions.
    P0, pm – Twist-propagation quantities.
    robot : dict – Robot model.

    Returns
    -------
    Jc_base : (6, 6) – Base-link Jacobian at contact.
    Jc_joints : (6, n_q) – Manipulator Jacobian at contact.
    """
    n = robot['n_links_joints']
    Jc_base, Jc_joints = geometric_jacobian(r_contact, r0, rL, P0, pm, n, robot)
    return Jc_base, Jc_joints
