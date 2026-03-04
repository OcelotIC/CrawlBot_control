"""Operational-space accelerations (twist-rate) computation.

Translated from Accelerations.m.
"""

import numpy as np
from ..utils.spatial import skew_sym


def accelerations(t0, tL, P0, pm, Bi0, Bij, u0, um, u0dot, umdot, robot):
    """Compute operational-space accelerations (twist-rates).

    Parameters
    ----------
    t0 : (6,) – Base-link twist.
    tL : (6, n) – Manipulator twists.
    P0 : (6, 6) – Base twist-propagation.
    pm : (6, n) – Manipulator twist-propagation vectors.
    Bi0 : list[(6,6)] – Base twist-propagation matrices.
    Bij : list[list[(6,6)]] – Twist-propagation matrices.
    u0 : (6,) – Base-link velocities.
    um : (n_q,) – Joint velocities.
    u0dot : (6,) – Base-link accelerations.
    umdot : (n_q,) – Joint accelerations.
    robot : dict – Robot model.

    Returns
    -------
    t0dot : (6,) – Base-link twist-rate (inertial).
    tLdot : (6, n) – Manipulator twist-rates (inertial).
    """
    n = robot['n_links_joints']

    # Base-link Omega
    Omega0 = np.zeros((6, 6))
    Omega0[:3, :3] = skew_sym(t0[:3])

    # Manipulator Omega
    Omegam = [np.zeros((6, 6)) for _ in range(n)]
    for i in range(n):
        Omegam[i][:3, :3] = skew_sym(tL[:3, i])
        Omegam[i][3:, 3:] = skew_sym(tL[:3, i])

    # Base twist-rate
    t0dot = Omega0 @ P0 @ u0 + P0 @ u0dot

    # Manipulator twist-rates
    tLdot = np.zeros((6, n))

    for i in range(n):
        if robot['joints'][i]['parent_link'] == 0:
            Bdot_contrib = np.zeros((6, 6))
            Bdot_contrib[3:, :3] = skew_sym(t0[3:] - tL[3:, i])
            tLdot[:, i] = Bi0[i] @ t0dot + Bdot_contrib @ t0
        else:
            p_link = robot['joints'][i]['parent_link'] - 1
            Bdot_contrib = np.zeros((6, 6))
            Bdot_contrib[3:, :3] = skew_sym(tL[3:, p_link] - tL[3:, i])
            tLdot[:, i] = Bij[i][p_link] @ tLdot[:, p_link] + Bdot_contrib @ tL[:, p_link]

        if robot['joints'][i]['type'] != 0:
            q_id = robot['joints'][i]['q_id'] - 1
            tLdot[:, i] += Omegam[i] @ pm[:, i] * um[q_id] + pm[:, i] * umdot[q_id]

    return t0dot, tLdot
