"""Generalized Inertia Matrix (GIM) computation.

Translated from GIM.m / GIM_casadi.m.
"""

import numpy as np


def generalized_inertia_matrix(M0_tilde, Mm_tilde, Bij, Bi0, P0, pm, robot):
    """Compute the Generalized Inertia Matrix H.

    The full GIM is: H = [[H0, H0m], [H0m.T, Hm]]

    Parameters
    ----------
    M0_tilde : (6, 6) – Base-link MCB matrix.
    Mm_tilde : list[(6,6)] – Manipulator MCB matrices.
    Bij, Bi0, P0, pm : kinematic quantities from diff_kinematics.
    robot : dict – Robot model.

    Returns
    -------
    H0  : (6, 6) – Base-link inertia matrix.
    H0m : (6, n_q) – Coupling inertia matrix.
    Hm  : (n_q, n_q) – Manipulator inertia matrix.
    """
    n_q = robot['n_q']
    n = robot['n_links_joints']

    # Base-link inertia
    H0 = P0.T @ M0_tilde @ P0

    # Manipulator inertia
    Hm = np.zeros((n_q, n_q))
    for j in range(n):
        for i in range(j, n):
            if robot['joints'][i]['type'] != 0 and robot['joints'][j]['type'] != 0:
                qi = robot['joints'][i]['q_id'] - 1
                qj = robot['joints'][j]['q_id'] - 1
                val = pm[:, i] @ Mm_tilde[i] @ Bij[i][j] @ pm[:, j]
                Hm[qi, qj] = val
                Hm[qj, qi] = val

    # Coupling inertia
    H0m = np.zeros((6, n_q))
    for i in range(n):
        if robot['joints'][i]['type'] != 0:
            qi = robot['joints'][i]['q_id'] - 1
            H0m[:, qi] = (pm[:, i] @ Mm_tilde[i] @ Bi0[i] @ P0).T

    return H0, H0m, Hm
