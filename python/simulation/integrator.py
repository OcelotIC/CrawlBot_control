"""Dynamics integration with Baumgarte stabilization.

Translated from integrate_dynamics.m.
"""

import numpy as np
from ..utils.quaternion import quat_integrate


def integrate_dynamics(state, tau_joints, Fc, sys, robot, dt, env):
    """Integrate dynamics forward by one time step.

    Uses Euler integration with Baumgarte constraint stabilization.

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
    H0 = sys['H0']
    H0m = sys['H0m']
    Hm = sys['Hm']
    C0 = sys['C0']
    C0m = sys['C0m']
    Cm0 = sys['Cm0']
    Cm = sys['Cm']

    n_q = robot['n_q']

    H_full = np.block([
        [H0,     H0m],
        [H0m.T,  Hm],
    ])
    C_full = np.block([
        [C0,  C0m],
        [Cm0, Cm],
    ])
    u_full = np.concatenate([sys['u0'], state['um']])

    # 2. Generalized efforts
    tau_full = np.concatenate([np.zeros(6), tau_joints])

    # 3. External forces via Jacobian
    J0_T = sys['Jc_base'].T
    Jm_T = sys['Jc_joints'].T
    F_ext = np.concatenate([J0_T @ Fc, Jm_T @ Fc])

    # 4. Baumgarte stabilization
    alpha_stab = 5.0
    beta_stab = 10.0

    Phi_error = np.zeros(6)  # Simplified: assume constraint satisfied

    J_full = np.hstack([sys['Jc_base'], sys['Jc_joints']])
    gamma_stab = -2 * alpha_stab * J_full @ u_full - beta_stab**2 * Phi_error

    stabilization_term = np.concatenate([J0_T, Jm_T], axis=0) @ gamma_stab

    # 5. Equation of motion with stabilization
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

    # Quaternion integration for robot base
    omega_base_robot = u0dot[:3]
    if np.linalg.norm(omega_base_robot) > 1e-8:
        state_next['quat_base'] = quat_integrate(
            state['quat_base'], omega_base_robot, dt
        )
    else:
        state_next['quat_base'] = state['quat_base'].copy()

    # 8. Satellite dynamics (reaction torque from contact)
    # The satellite absorbs the reaction from the robot's contact forces.
    # Torque on satellite from contact = -moment part of Fc (Newton's 3rd law)
    tau_on_satellite = -Fc[:3]  # Reaction torque

    # Reaction wheel absorbs part of the disturbance
    I_sat = env['satellite']['inertia']
    omega_sat = state['omega_satellite']

    # Simple model: RW absorbs torque, satellite rotates from remainder
    h_RW_dot = tau_on_satellite  # RW tries to absorb all torque
    h_RW_new = state['h_RW_stored'] + h_RW_dot * dt

    # Saturate RW
    h_max = env['RW']['h_total_max']
    h_norm = np.linalg.norm(h_RW_new)
    if h_norm > h_max:
        h_RW_new = h_RW_new / h_norm * h_max
        # Excess goes to satellite
        excess = h_RW_dot * dt - (h_RW_new - state['h_RW_stored'])
        omega_sat = omega_sat + np.linalg.solve(I_sat, excess)

    state_next['h_RW_stored'] = h_RW_new
    state_next['omega_satellite'] = omega_sat

    # Satellite quaternion integration
    if np.linalg.norm(omega_sat) > 1e-12:
        state_next['quat_satellite'] = quat_integrate(
            state['quat_satellite'], omega_sat, dt
        )
    else:
        state_next['quat_satellite'] = state['quat_satellite'].copy()

    state_next['omega_base'] = state['omega_base'] + u0dot[:3] * dt

    return state_next
