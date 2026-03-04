"""Operational-space velocity computation.

Translated from Velocities.m / Velocities_casadi.m.
"""

import numpy as np


def velocities(Bij, Bi0, P0, pm, u0, um, robot):
    """Compute operational-space velocities (twists) of the multibody system.

    Parameters
    ----------
    Bij : list[list[(6,6)]] – Twist-propagation matrices.
    Bi0 : list[(6,6)] – Base twist-propagation matrices.
    P0  : (6, 6) array – Base-link twist-propagation.
    pm  : (6, n) array – Manipulator twist-propagation vectors.
    u0  : (6,) array – Base-link velocities [omega; rdot].
    um  : (n_q,) array – Joint velocities.
    robot : dict – Robot model.

    Returns
    -------
    t0 : (6,) array – Base-link twist [omega; rdot] (inertial).
    tL : (6, n) array – Link twists (inertial).
    """
    n = robot['n_links_joints']

    tL = np.zeros((6, n))
    t0 = P0 @ u0

    for i in range(n):
        if robot['joints'][i]['parent_link'] == 0:
            tL[:, i] = Bi0[i] @ t0
        else:
            parent_id = robot['joints'][i]['parent_link'] - 1
            tL[:, i] = Bij[i][parent_id] @ tL[:, parent_id]

        if robot['joints'][i]['type'] != 0:
            q_id = robot['joints'][i]['q_id'] - 1
            tL[:, i] += pm[:, i] * um[q_id]

    return t0, tL
