"""Momentum map: contact wrenches -> angular momentum rate.

Computes M_lambda, the (3 x 6*nc) matrix that maps stacked contact
wrenches [Fc_a; Fc_b] to the rate of change of angular momentum about the
system center of mass.

Wrench convention: [f(3), tau(3)] per contact (force first, matching MPC).
"""

import numpy as np


def _skew(v):
    """3-vector -> 3x3 skew-symmetric matrix."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def compute_momentum_map(r_com, r_contact_a=None, r_contact_b=None):
    """Build the (3 x 12) or (3 x 6) momentum map matrix.

    Each contact wrench Fc_j = [f_j(3), tau_j(3)] contributes to angular
    momentum rate:

        L_dot_j = tau_j + (r_Cj - r_com) x f_j

    So the 3x6 block for contact j is: [skew(r_Cj - r_com), I_3]
    (force columns first, torque columns second).

    Parameters
    ----------
    r_com : (3,) CoM position.
    r_contact_a : (3,) or None. Contact A position.
    r_contact_b : (3,) or None. Contact B position.

    Returns
    -------
    M_lambda : (3, 12), (3, 6), or (3, 0) depending on active contacts.
    """
    blocks = []

    if r_contact_a is not None:
        lever_a = r_contact_a - r_com
        M_a = np.hstack([_skew(lever_a), np.eye(3)])  # (3, 6)
        blocks.append(M_a)

    if r_contact_b is not None:
        lever_b = r_contact_b - r_com
        M_b = np.hstack([_skew(lever_b), np.eye(3)])  # (3, 6)
        blocks.append(M_b)

    if len(blocks) == 0:
        return np.zeros((3, 0))

    return np.hstack(blocks)  # (3, 6) or (3, 12)
