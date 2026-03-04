"""Forward and Inverse Dynamics.

Translated from FD.m and ID.m.
"""

import numpy as np
from ..utils.spatial import skew_sym
from .accelerations import accelerations


def inverse_dynamics(wF0, wFm, t0, tL, t0dot, tLdot, P0, pm, I0, Im,
                     Bij, Bi0, robot):
    """Inverse dynamics: compute generalized forces from accelerations.

    Parameters
    ----------
    wF0 : (6,) – External wrench on base [n, f] (inertial).
    wFm : (6, n) – External wrenches on links (inertial).
    t0, tL – Twists.
    t0dot, tLdot – Twist rates.
    P0, pm, I0, Im, Bij, Bi0 – Kinematic/dynamic quantities.
    robot : dict – Robot model.

    Returns
    -------
    tau0 : (6,) – Base-link forces [n, f].
    taum : (n_q,) – Joint forces/torques.
    """
    n = robot['n_links_joints']
    child = robot['con']['child']
    child_base = robot['con']['child_base']

    # --- Mdot ---
    Mdot0 = np.zeros((6, 6))
    Mdot0[:3, :3] = skew_sym(t0[:3]) @ I0

    Mdot = [np.zeros((6, 6)) for _ in range(n)]
    for i in range(n):
        Mdot[i][:3, :3] = skew_sym(tL[:3, i]) @ Im[i]

    # --- Forces ---
    # Base-link
    M0 = np.block([
        [I0,               np.zeros((3, 3))],
        [np.zeros((3, 3)), robot['base_link']['mass'] * np.eye(3)],
    ])
    wq0 = M0 @ t0dot + Mdot0 @ t0 - wF0

    # Manipulator
    wq = np.zeros((6, n))
    for i in range(n):
        Mi = np.block([
            [Im[i],            np.zeros((3, 3))],
            [np.zeros((3, 3)), robot['links'][i]['mass'] * np.eye(3)],
        ])
        wq[:, i] = Mi @ tLdot[:, i] + Mdot[i] @ tL[:, i] - wFm[:, i]

    # --- wq_tilde (backwards recursion) ---
    wq_tilde = wq.copy()
    for i in range(n - 1, -1, -1):
        children = np.where(child[:, i] == 1)[0]
        for c in children:
            wq_tilde[:, i] += Bij[c][i].T @ wq_tilde[:, c]

    # Base-link
    wq_tilde0 = wq0.copy()
    children = np.where(child_base == 1)[0]
    for c in children:
        wq_tilde0 += Bi0[c].T @ wq_tilde[:, c]

    # --- Joint forces ---
    tau0 = P0.T @ wq_tilde0

    taum = np.zeros(robot['n_q'])
    for i in range(n):
        if robot['joints'][i]['type'] != 0:
            q_id = robot['joints'][i]['q_id'] - 1
            taum[q_id] = pm[:, i] @ wq_tilde[:, i]

    return tau0, taum


def forward_dynamics(tau0, taum, wF0, wFm, t0, tL, P0, pm, I0, Im,
                     Bij, Bi0, u0, um, robot):
    """Forward dynamics: compute accelerations from forces.

    Parameters
    ----------
    tau0 : (6,) – Base-link forces [n, f].
    taum : (n_q,) – Joint forces/torques.
    wF0 : (6,) – External wrench on base (inertial).
    wFm : (6, n) – External wrenches on links (inertial).
    t0, tL – Twists.
    P0, pm, I0, Im, Bij, Bi0 – Kinematic/dynamic quantities.
    u0, um – Velocities.
    robot : dict – Robot model.

    Returns
    -------
    u0dot : (6,) – Base-link accelerations.
    umdot : (n_q,) – Joint accelerations.
    """
    n = robot['n_links_joints']
    n_q = robot['n_q']
    child = robot['con']['child']
    child_base = robot['con']['child_base']

    # --- Inverse dynamics with zero accelerations ---
    t0dot_z, tLdot_z = accelerations(t0, tL, P0, pm, Bi0, Bij,
                                      u0, um, np.zeros(6), np.zeros(n_q), robot)
    tau0_0, taum_0 = inverse_dynamics(wF0, wFm, t0, tL, t0dot_z, tLdot_z,
                                       P0, pm, I0, Im, Bij, Bi0, robot)

    # Initialize solution
    phi0 = tau0 - tau0_0
    phi = taum - taum_0

    # --- M_hat, psi_hat, psi (backwards recursion) ---
    M_hat = [np.zeros((6, 6)) for _ in range(n)]
    psi_hat = np.zeros((6, n))
    psi = np.zeros((6, n))

    for i in range(n - 1, -1, -1):
        M_hat[i] = np.block([
            [Im[i],            np.zeros((3, 3))],
            [np.zeros((3, 3)), robot['links'][i]['mass'] * np.eye(3)],
        ])
        children = np.where(child[:, i] == 1)[0]
        for c in children:
            M_hat_cc = M_hat[c] - np.outer(psi_hat[:, c], psi[:, c])
            M_hat[i] += Bij[c][i].T @ M_hat_cc @ Bij[c][i]

        if robot['joints'][i]['type'] == 0:
            psi_hat[:, i] = np.zeros(6)
            psi[:, i] = np.zeros(6)
        else:
            psi_hat[:, i] = M_hat[i] @ pm[:, i]
            denom = pm[:, i] @ psi_hat[:, i]
            psi[:, i] = psi_hat[:, i] / denom

    # Base-link
    M_hat0 = np.block([
        [I0,               np.zeros((3, 3))],
        [np.zeros((3, 3)), robot['base_link']['mass'] * np.eye(3)],
    ])
    children = np.where(child_base == 1)[0]
    for c in children:
        M_hat0_cc = M_hat[c] - np.outer(psi_hat[:, c], psi[:, c])
        M_hat0 += Bi0[c].T @ M_hat0_cc @ Bi0[c]

    psi_hat0 = M_hat0 @ P0

    # --- eta (backwards recursion) ---
    eta = np.zeros((6, n))
    phi_hat = np.zeros(n)
    phi_tilde = np.zeros(n_q)

    for i in range(n - 1, -1, -1):
        children = np.where(child[:, i] == 1)[0]
        for c in children:
            eta[:, i] += Bij[c][i].T @ (psi[:, c] * phi_hat[c] + eta[:, c])

        phi_hat[i] = -pm[:, i] @ eta[:, i]
        if robot['joints'][i]['type'] != 0:
            q_id = robot['joints'][i]['q_id'] - 1
            phi_hat[i] += phi[q_id]
            phi_tilde[q_id] = phi_hat[i] / (pm[:, i] @ psi_hat[:, i])

    # Base-link
    eta0 = np.zeros(6)
    children = np.where(child_base == 1)[0]
    for c in children:
        eta0 += Bi0[c].T @ (psi[:, c] * phi_hat[c] + eta[:, c])

    phi_hat0 = phi0 - P0.T @ eta0
    u0dot = np.linalg.solve(P0.T @ psi_hat0, phi_hat0)

    # --- Manipulator acceleration (forward recursion) ---
    mu = np.zeros((6, n))
    umdot = np.zeros(n_q)

    for i in range(n):
        if robot['joints'][i]['parent_link'] == 0:
            mu[:, i] = Bi0[i] @ (P0 @ u0dot)
        else:
            p_link = robot['joints'][i]['parent_link'] - 1
            p_joint = robot['links'][p_link]['parent_joint'] - 1
            if robot['joints'][p_joint]['type'] != 0:
                p_q_id = robot['joints'][p_joint]['q_id'] - 1
                mu_aux = pm[:, p_joint] * umdot[p_q_id] + mu[:, p_joint]
            else:
                mu_aux = mu[:, p_joint]
            mu[:, i] = Bij[i][p_link] @ mu_aux

        if robot['joints'][i]['type'] != 0:
            q_id = robot['joints'][i]['q_id'] - 1
            umdot[q_id] = phi_tilde[q_id] - psi[:, i] @ mu[:, i]

    return u0dot, umdot
