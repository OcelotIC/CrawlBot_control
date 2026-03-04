"""Forward kinematics for the multibody system (SPART algorithm).

Translated from Kinematics.m and Kinematics_casadi.m.
"""

import numpy as np
from .attitude import euler_dcm


def kinematics(R0, r0, qm, robot):
    """Compute positions and orientations of all joints and links.

    Parameters
    ----------
    R0 : (3, 3) array – Base-link rotation matrix (body -> inertial).
    r0 : (3,) array – Base-link CoM position in inertial frame.
    qm : (n_q,) array – Joint displacements.
    robot : dict – Robot model from urdf2robot.

    Returns
    -------
    RJ : list of (3, 3) arrays – Joint rotation matrices (inertial frame).
    RL : list of (3, 3) arrays – Link rotation matrices (inertial frame).
    rJ : (3, n) array – Joint positions (inertial frame).
    rL : (3, n) array – Link positions (inertial frame).
    e  : (3, n) array – Joint axes (inertial frame).
    g  : (3, n) array – Vectors from joint origin to link CoM (inertial frame).
    """
    n = robot['n_links_joints']
    r0 = np.asarray(r0).reshape(3, 1)

    # Homogeneous transforms
    TJ = [np.zeros((4, 4)) for _ in range(n)]
    TL = [np.zeros((4, 4)) for _ in range(n)]

    # Base-link transform
    T0 = np.eye(4)
    T0[:3, :3] = R0
    T0[:3, 3:4] = r0

    # Forward kinematics recursion
    for i in range(n):
        cjoint = robot['joints'][i]
        jid = cjoint['id'] - 1  # 0-indexed

        # Joint kinematics
        if cjoint['parent_link'] == 0:
            TJ[jid] = T0 @ cjoint['T']
        else:
            parent_lid = cjoint['parent_link'] - 1  # 0-indexed
            TJ[jid] = TL[parent_lid] @ cjoint['T']

        # Transformation due to joint variable
        if cjoint['type'] == 1:  # Revolute
            R_qm = euler_dcm(cjoint['axis'], qm[cjoint['q_id'] - 1]).T
            T_qm = np.eye(4)
            T_qm[:3, :3] = R_qm
        elif cjoint['type'] == 2:  # Prismatic
            T_qm = np.eye(4)
            T_qm[:3, 3] = cjoint['axis'] * qm[cjoint['q_id'] - 1]
        else:  # Fixed
            T_qm = np.eye(4)

        # Link kinematics
        clink = robot['links'][cjoint['child_link'] - 1]
        link_id = clink['id'] - 1  # 0-indexed
        parent_joint_id = clink['parent_joint'] - 1  # 0-indexed
        TL[link_id] = TJ[parent_joint_id] @ T_qm @ clink['T']

    # Extract rotation matrices, positions, axes, geometry
    RJ = [TJ[i][:3, :3].copy() for i in range(n)]
    RL = [TL[i][:3, :3].copy() for i in range(n)]

    rJ = np.zeros((3, n))
    rL = np.zeros((3, n))
    e = np.zeros((3, n))
    g = np.zeros((3, n))

    for i in range(n):
        rJ[:, i] = TJ[i][:3, 3]
        e[:, i] = RJ[i] @ robot['joints'][i]['axis']

    for i in range(n):
        rL[:, i] = TL[i][:3, 3]
        parent_joint_idx = robot['links'][i]['parent_joint'] - 1
        g[:, i] = rL[:, i] - rJ[:, parent_joint_idx]

    return RJ, RL, rJ, rL, e, g
