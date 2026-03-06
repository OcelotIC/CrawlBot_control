"""Momentum map: contact wrenches → angular momentum rate.

Computes M_lambda, the (3 x 6*nc) matrix that maps stacked contact
wrenches [Fc_a; Fc_b] to the rate of change of angular momentum about the
system center of mass (structure + robot).

    L_dot = sum_j [ (r_Cj - r_com) x f_j + tau_j ]

This is the centroidal dynamics that Lutze's QP implicitly minimizes via
the satellite stabilization term.  Making it explicit allows direct
minimization of |L_dot|.
"""

import numpy as np
from .contact_adjoint import skew


def compute_momentum_map(
    r_com: np.ndarray,
    r_contact_a: np.ndarray = None,
    r_contact_b: np.ndarray = None,
) -> np.ndarray:
    """Build the (3 x 12) or (3 x 6) momentum map matrix.

    Each contact wrench Fc_j = [tau_j(3); f_j(3)] contributes to angular
    momentum rate:

        L_dot_j = tau_j + (r_Cj - r_com) x f_j

    So the j-th block (3 x 6) of M_lambda is:

        M_j = [ I_3,  skew(r_Cj - r_com) ]

    Parameters
    ----------
    r_com : (3,) ndarray
        Center of mass position of the full system (robot + structure)
        in world frame.
    r_contact_a : (3,) ndarray or None
        Position of contact A in world frame.
    r_contact_b : (3,) ndarray or None
        Position of contact B in world frame.

    Returns
    -------
    M_lambda : (3, n) ndarray
        Momentum map.  n = 6 * (number of active contacts).
        Column order: [Fc_a; Fc_b] matching the QP decision variable.
    """
    blocks = []
    for r_c in (r_contact_a, r_contact_b):
        if r_c is not None:
            lever = r_c - r_com
            # L_dot = tau + lever x f  →  M_j = [I_3,  skew(lever)]
            M_j = np.zeros((3, 6))
            M_j[:, :3] = np.eye(3)
            M_j[:, 3:] = skew(lever)
            blocks.append(M_j)

    if not blocks:
        return np.zeros((3, 0))

    return np.hstack(blocks)
