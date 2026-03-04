"""Differential kinematics: twist-propagation matrices.

Translated from DiffKinematics.m / DiffKinematics_casadi.m.
"""

import numpy as np
from ..utils.spatial import skew_sym


def diff_kinematics(R0, r0, rL, e, g, robot):
    """Compute the twist-propagation matrices.

    Parameters
    ----------
    R0 : (3, 3) array – Base-link rotation matrix.
    r0 : (3,) array – Base-link CoM position.
    rL : (3, n) array – Link positions.
    e  : (3, n) array – Joint axes (inertial).
    g  : (3, n) array – Joint-to-link vectors (inertial).
    robot : dict – Robot model.

    Returns
    -------
    Bij : list of list of (6, 6) arrays – Twist-propagation Bij[i][j].
    Bi0 : list of (6, 6) arrays – Twist-propagation from base Bi0[i].
    P0  : (6, 6) array – Base-link twist-propagation.
    pm  : (6, n) array – Manipulator twist-propagation vectors.
    """
    n = robot['n_links_joints']
    r0 = np.asarray(r0).ravel()
    branch = robot['con']['branch']

    # Bij[i][j] – nested list
    Bij = [[np.zeros((6, 6)) for _ in range(n)] for _ in range(n)]
    for j in range(n):
        for i in range(n):
            if branch[i, j] == 1:
                Bij[i][j] = np.block([
                    [np.eye(3),                              np.zeros((3, 3))],
                    [skew_sym(rL[:, j] - rL[:, i]),          np.eye(3)],
                ])

    # Bi0[i]
    Bi0 = [np.zeros((6, 6)) for _ in range(n)]
    for i in range(n):
        Bi0[i] = np.block([
            [np.eye(3),                        np.zeros((3, 3))],
            [skew_sym(r0 - rL[:, i]),           np.eye(3)],
        ])

    # P0
    P0 = np.block([
        [R0,              np.zeros((3, 3))],
        [np.zeros((3, 3)), np.eye(3)],
    ])

    # pm
    pm = np.zeros((6, n))
    for i in range(n):
        jtype = robot['joints'][i]['type']
        if jtype == 1:  # Revolute
            pm[:, i] = np.concatenate([e[:, i], np.cross(e[:, i], g[:, i])])
        elif jtype == 2:  # Prismatic
            pm[:, i] = np.concatenate([np.zeros(3), e[:, i]])
        # Fixed: stays zero

    return Bij, Bi0, P0, pm
