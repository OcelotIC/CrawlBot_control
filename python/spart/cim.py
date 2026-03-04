"""Convective Inertia Matrix (CIM) computation.

Translated from CIM.m / CIM_casadi.m.
"""

import numpy as np
from ..utils.spatial import skew_sym


def convective_inertia_matrix(t0, tL, I0, Im, M0_tilde, Mm_tilde,
                               Bij, Bi0, P0, pm, robot):
    """Compute the Generalized Convective Inertia Matrix C.

    The full CIM is: C = [[C0, C0m], [Cm0, Cm]]

    Returns
    -------
    C0  : (6, 6) – Base-link convective inertia.
    C0m : (6, n_q) – Base-manipulator coupling.
    Cm0 : (n_q, 6) – Manipulator-base coupling.
    Cm  : (n_q, n_q) – Manipulator convective inertia.
    """
    n_q = robot['n_q']
    n = robot['n_links_joints']
    branch = robot['con']['branch']
    child = robot['con']['child']
    child_base = robot['con']['child_base']

    # --- Omega matrices ---
    Omega0 = np.zeros((6, 6))
    Omega0[:3, :3] = skew_sym(t0[:3])

    Omega = [np.zeros((6, 6)) for _ in range(n)]
    for i in range(n):
        S = skew_sym(tL[:3, i])
        Omega[i][:3, :3] = S
        Omega[i][3:, 3:] = S

    # --- Mdot ---
    Mdot0 = np.zeros((6, 6))
    Mdot0[:3, :3] = Omega0[:3, :3] @ I0

    Mdot = [np.zeros((6, 6)) for _ in range(n)]
    for i in range(n):
        Mdot[i][:3, :3] = Omega[i][:3, :3] @ Im[i]

    # --- Mdot_tilde (backwards recursion) ---
    Mdot_tilde = [m.copy() for m in Mdot]
    for i in range(n - 1, -1, -1):
        children = np.where(child[:, i] == 1)[0]
        for c in children:
            Mdot_tilde[i] = Mdot_tilde[i] + Mdot_tilde[c]

    Mdot0_tilde = Mdot0.copy()
    children = np.where(child_base == 1)[0]
    for c in children:
        Mdot0_tilde = Mdot0_tilde + Mdot_tilde[c]

    # --- Bdot ---
    Bdotij = [[np.zeros((6, 6)) for _ in range(n)] for _ in range(n)]
    for j in range(n):
        for i in range(n):
            if branch[i, j] == 1:
                Bdotij[i][j][3:, :3] = skew_sym(tL[3:, j] - tL[3:, i])

    # --- Hij_tilde ---
    Hij_tilde = [[np.zeros((6, 6)) for _ in range(n)] for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            Hij_tilde[i][j] = Mm_tilde[i] @ Bdotij[i][j]
            children = np.where(child[:, i] == 1)[0]
            for k in children:
                Hij_tilde[i][j] = Hij_tilde[i][j] + Bij[k][i].T @ Hij_tilde[k][i]

    # --- Hi0_tilde ---
    Hi0_tilde = [np.zeros((6, 6)) for _ in range(n)]
    for i in range(n - 1, -1, -1):
        Bdot = np.zeros((6, 6))
        Bdot[3:, :3] = skew_sym(t0[3:] - tL[3:, i])
        Hi0_tilde[i] = Mm_tilde[i] @ Bdot
        children = np.where(child[:, i] == 1)[0]
        for k in children:
            Hi0_tilde[i] = Hi0_tilde[i] + Bij[k][i].T @ Hij_tilde[k][i]

    # --- C matrices ---
    Cm = np.zeros((n_q, n_q))
    C0m = np.zeros((6, n_q))
    Cm0 = np.zeros((n_q, 6))

    # Cm
    for j in range(n):
        for i in range(n):
            ji = robot['joints'][i]
            jj = robot['joints'][j]
            if (ji['type'] != 0 and jj['type'] != 0 and
                    (branch[i, j] == 1 or branch[j, i] == 1)):
                qi = ji['q_id'] - 1
                qj = jj['q_id'] - 1
                if i <= j:
                    child_con = np.zeros((6, 6))
                    children = np.where(child[:, j] == 1)[0]
                    for k in children:
                        child_con += Bij[k][i].T @ Hij_tilde[k][j]
                    Cm[qi, qj] = pm[:, i] @ (
                        Bij[j][i].T @ Mm_tilde[j] @ Omega[j]
                        + child_con + Mdot_tilde[j]
                    ) @ pm[:, j]
                else:
                    Cm[qi, qj] = pm[:, i] @ (
                        Mm_tilde[i] @ Bij[i][j] @ Omega[j]
                        + Hij_tilde[i][j] + Mdot_tilde[i]
                    ) @ pm[:, j]

    # C0
    child_con = np.zeros((6, 6))
    children = np.where(child_base == 1)[0]
    for k in children:
        child_con += Bi0[k].T @ Hi0_tilde[k]
    C0 = P0.T @ (M0_tilde @ Omega0 + child_con + Mdot0_tilde) @ P0

    # C0m
    for j in range(n):
        if robot['joints'][j]['type'] != 0:
            qj = robot['joints'][j]['q_id'] - 1
            if j == n - 1:
                C0m[:, qj] = P0.T @ (
                    Bi0[j].T @ Mm_tilde[j] @ Omega[j] + Mdot_tilde[j]
                ) @ pm[:, j]
            else:
                child_con = np.zeros((6, 6))
                children = np.where(child[:, j] == 1)[0]
                for k in children:
                    child_con += Bi0[k].T @ Hij_tilde[k][j]
                C0m[:, qj] = P0.T @ (
                    Bi0[j].T @ Mm_tilde[j] @ Omega[j]
                    + child_con + Mdot_tilde[j]
                ) @ pm[:, j]

    # Cm0
    for i in range(n):
        if robot['joints'][i]['type'] != 0:
            qi = robot['joints'][i]['q_id'] - 1
            Cm0[qi, :] = pm[:, i] @ (
                Mm_tilde[i] @ Bi0[i] @ Omega0
                + Hi0_tilde[i] + Mdot_tilde[i]
            ) @ P0

    return C0, C0m, Cm0, Cm
