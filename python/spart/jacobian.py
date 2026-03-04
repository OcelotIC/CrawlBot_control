"""Geometric Jacobian computation.

Translated from Jacob.m.
"""

import numpy as np
from ..utils.spatial import skew_sym


def geometric_jacobian(rp, r0, rL, P0, pm, i, robot):
    """Compute the geometric Jacobian of a point p on link i.

    The twist at point p is: tp = J0 @ u0 + Jm @ um

    Parameters
    ----------
    rp : (3,) array – Position of the point of interest (inertial).
    r0 : (3,) array – Base-link position (inertial).
    rL : (3, n) array – Link positions (inertial).
    P0 : (6, 6) array – Base-link twist-propagation.
    pm : (6, n) array – Manipulator twist-propagation vectors.
    i  : int – Link index (1-based, SPART convention).
    robot : dict – Robot model.

    Returns
    -------
    J0 : (6, 6) array – Base-link Jacobian.
    Jm : (6, n_q) array – Manipulator Jacobian.
    """
    rp = np.asarray(rp).ravel()
    r0 = np.asarray(r0).ravel()
    branch = robot['con']['branch']

    # Base-link Jacobian
    J0 = np.block([
        [np.eye(3),                     np.zeros((3, 3))],
        [skew_sym(r0 - rp),             np.eye(3)],
    ]) @ P0

    # Manipulator Jacobian
    Jm = np.zeros((6, robot['n_q']))

    # Iterate through all joints up to link i (1-based)
    for j in range(i):
        if robot['joints'][j]['type'] != 0:
            q_id = robot['joints'][j]['q_id'] - 1
            # Check if link i-1 (0-based) and link j are on same branch
            if branch[i - 1, j] == 1:
                Jm[:, q_id] = np.block([
                    [np.eye(3),                      np.zeros((3, 3))],
                    [skew_sym(rL[:, j] - rp),        np.eye(3)],
                ]) @ pm[:, j]

    return J0, Jm
