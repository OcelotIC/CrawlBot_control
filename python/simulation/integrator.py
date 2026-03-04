"""Dynamics integration with Baumgarte stabilization.

Translated from integrate_dynamics.m.

The satellite is modeled as the SPART base: its inertia is augmented into H0.
The satellite dynamics emerge naturally from the coupled equations of motion
via conservation of angular momentum (no reaction wheels — matching the paper).
"""

import numpy as np
from ..utils.quaternion import quat_integrate


def integrate_dynamics(state, tau_joints, Fc, sys, robot, dt, env):
    """Integrate dynamics forward by one time step.

    Uses Euler integration with Baumgarte constraint stabilization.
    The satellite inertia is added to the SPART base inertia (H0) so that
    u0[:3] represents the satellite angular velocity.  No reaction wheels
    are used — the satellite rotates freely per angular momentum conservation,
    and the QP controller minimizes disturbance through contact force
    optimization (matching Lutze et al. 2023).

    Parameters
    ----------
    state : dict – Current system state.
    tau_joints : (n_q,) – Joint torques from controller.
    Fc : (6,) – Contact wrench.
    sys : dict – System state (dynamics matrices, Jacobians).
    robot : dict – Robot model.
    dt : float – Time step.
    env : dict – Environment configuration.

    Returns
    -------
    state_next : dict – Updated state.
    """
    # 1. Extract dynamics matrices
    H0 = sys['H0'].copy()
    H0m = sys['H0m']
    Hm = sys['Hm']
    C0 = sys['C0']
    C0m = sys['C0m']
    Cm0 = sys['Cm0']
    Cm = sys['Cm']

    n_q = robot['n_q']

    # --- Augment base inertia with satellite inertia ---
    # In the paper (eq. 11), the SPART base IS the satellite.  We add the
    # satellite's mass and rotational inertia to H0 so the coupled dynamics
    # naturally produce the correct angular accelerations via momentum
    # conservation.  No reaction wheels are modeled (matching the paper).
    I_sat = env['satellite']['inertia']     # (3,3) rotational inertia
    m_sat = env['satellite']['mass']        # scalar
    H0_sat = np.zeros((6, 6))
    H0_sat[:3, :3] = I_sat
    H0_sat[3, 3] = m_sat
    H0_sat[4, 4] = m_sat
    H0_sat[5, 5] = m_sat
    H0_aug = H0 + H0_sat

    H_full = np.block([
        [H0_aug,  H0m],
        [H0m.T,   Hm],
    ])
    C_full = np.block([
        [C0,  C0m],
        [Cm0, Cm],
    ])
    u_full = np.concatenate([sys['u0'], state['um']])

    # 2. Generalized efforts (no base actuation — satellite is free-floating)
    tau_full = np.concatenate([np.zeros(6), tau_joints])

    # 3. External forces via Jacobian
    J0_T = sys['Jc_base'].T
    Jm_T = sys['Jc_joints'].T
    F_ext = np.concatenate([J0_T @ Fc, Jm_T @ Fc])

    # 4. Baumgarte stabilization
    alpha_stab = 5.0
    beta_stab = 10.0
    Phi_error = np.zeros(6)

    J_full = np.hstack([sys['Jc_base'], sys['Jc_joints']])
    gamma_stab = -2 * alpha_stab * J_full @ u_full - beta_stab**2 * Phi_error
    stabilization_term = np.concatenate([J0_T, Jm_T], axis=0) @ gamma_stab

    # 5. Equation of motion
    rhs = tau_full + F_ext - C_full @ u_full + stabilization_term

    # 6. Solve (robust)
    cond_H = np.linalg.cond(H_full)
    if cond_H > 1e10:
        H_full = H_full + 1e-6 * np.eye(H_full.shape[0])

    try:
        udot_full = np.linalg.solve(H_full, rhs)
    except np.linalg.LinAlgError:
        udot_full = np.linalg.pinv(H_full, rcond=1e-6) @ rhs

    # Validation and saturation
    if np.any(np.isnan(udot_full)) or np.any(np.isinf(udot_full)):
        udot_full = np.zeros_like(udot_full)

    accel_max = 50.0
    if np.linalg.norm(udot_full) > accel_max:
        udot_full = udot_full / np.linalg.norm(udot_full) * accel_max

    u0dot = udot_full[:6]
    umdot = udot_full[6:]

    # 7. Euler integration
    state_next = dict(state)
    state_next['u_base'] = state['u_base'] + u0dot[3:6] * dt
    state_next['um'] = state['um'] + umdot * dt
    state_next['q_base'] = state['q_base'] + state_next['u_base'] * dt
    state_next['qm'] = state['qm'] + state_next['um'] * dt

    # Base/satellite angular velocity (they are coupled via augmented inertia)
    omega_base_new = state['omega_base'] + u0dot[:3] * dt
    state_next['omega_base'] = omega_base_new
    if np.linalg.norm(omega_base_new) > 1e-8:
        state_next['quat_base'] = quat_integrate(
            state['quat_base'], omega_base_new, dt
        )
    else:
        state_next['quat_base'] = state['quat_base'].copy()

    # 8. Satellite state = base state (satellite IS the SPART base)
    state_next['omega_satellite'] = omega_base_new
    state_next['h_RW_stored'] = state['h_RW_stored']  # unchanged (no RW)

    if np.linalg.norm(omega_base_new) > 1e-12:
        state_next['quat_satellite'] = quat_integrate(
            state['quat_satellite'], omega_base_new, dt
        )
    else:
        state_next['quat_satellite'] = state['quat_satellite'].copy()

    return state_next
