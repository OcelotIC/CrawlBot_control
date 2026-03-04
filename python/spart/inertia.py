"""Inertia projection and Mass Composite Body computation.

Translated from I_I.m / I_I_casadi.m and MCB.m / MCB_casadi.m.
"""

import numpy as np


def inertia_projection(R0, RL, robot):
    """Project link inertias into the inertial coordinate system.

    Parameters
    ----------
    R0 : (3, 3) array – Base-link rotation matrix.
    RL : list of (3, 3) arrays – Link rotation matrices.
    robot : dict – Robot model.

    Returns
    -------
    I0 : (3, 3) array – Base-link inertia in inertial frame.
    Im : list of (3, 3) arrays – Link inertias in inertial frame.
    """
    I0 = R0 @ robot['base_link']['inertia'] @ R0.T

    n = robot['n_links_joints']
    Im = [None] * n
    for i in range(n):
        Ri = RL[i]
        Im[i] = Ri @ robot['links'][i]['inertia'] @ Ri.T

    return I0, Im


def mass_composite_body(I0, Im, Bij, Bi0, robot):
    """Compute the Mass Composite Body (MCB) matrices.

    Parameters
    ----------
    I0 : (3, 3) array – Base-link inertia (inertial).
    Im : list of (3, 3) arrays – Link inertias (inertial).
    Bij : list[list[(6,6)]] – Twist-propagation matrices.
    Bi0 : list[(6,6)] – Base twist-propagation matrices.
    robot : dict – Robot model.

    Returns
    -------
    M0_tilde : (6, 6) array – Base-link mass composite body matrix.
    Mm_tilde : list of (6, 6) arrays – Manipulator MCB matrices.
    """
    n = robot['n_links_joints']
    child = robot['con']['child']
    child_base = robot['con']['child_base']

    # Initialize Mm_tilde
    Mm_tilde = [np.zeros((6, 6)) for _ in range(n)]

    # Backwards recursion
    for i in range(n - 1, -1, -1):
        Mm_tilde[i] = np.block([
            [Im[i],          np.zeros((3, 3))],
            [np.zeros((3, 3)), robot['links'][i]['mass'] * np.eye(3)],
        ])
        # Add children contributions
        children = np.where(child[:, i] == 1)[0]
        for c in children:
            Mm_tilde[i] += Bij[c][i].T @ Mm_tilde[c] @ Bij[c][i]

    # Base-link M tilde
    M0_tilde = np.block([
        [I0,              np.zeros((3, 3))],
        [np.zeros((3, 3)), robot['base_link']['mass'] * np.eye(3)],
    ])
    children = np.where(child_base == 1)[0]
    for c in children:
        M0_tilde += Bi0[c].T @ Mm_tilde[c] @ Bi0[c]

    return M0_tilde, Mm_tilde
