"""System state computation wrapper.

Translated from compute_system_state.m.
Orchestrates all SPART kinematics/dynamics computations.
"""

import numpy as np
from ..spart.attitude import quat_dcm
from ..spart.kinematics import kinematics
from ..spart.diff_kinematics import diff_kinematics
from ..spart.velocities import velocities
from ..spart.inertia import inertia_projection, mass_composite_body
from ..spart.gim import generalized_inertia_matrix
from ..spart.cim import convective_inertia_matrix
from .contact_jacobian import compute_contact_jacobian
from .angular_momentum import compute_angular_momentum


def compute_system_state(state, robot, env):
    """Compute all kinematic and dynamic quantities for the current state.

    Parameters
    ----------
    state : dict – Current state with keys:
        q_base, quat_base, qm, u_base, um, omega_base,
        omega_satellite, h_RW_stored.
    robot : dict – Robot model.
    env : dict – Environment configuration.

    Returns
    -------
    sys : dict – Complete system state with all kinematic/dynamic quantities.
    """
    # 1. Prepare SPART state
    R0 = quat_dcm(state['quat_base'])
    r0 = state['q_base']
    u0 = np.concatenate([state['omega_base'], state['u_base']])
    qm = state['qm']
    um = state['um']

    # 2. Kinematics
    RJ, RL, rJ, rL, e, g = kinematics(R0, r0, qm, robot)

    # 3. Differential kinematics
    Bij, Bi0, P0, pm = diff_kinematics(R0, r0, rL, e, g, robot)

    # 4. Velocities
    t0, tm = velocities(Bij, Bi0, P0, pm, u0, um, robot)

    # 5. Inertia projection
    I0, Im = inertia_projection(R0, RL, robot)

    # 6. Mass composite body
    M0_tilde, Mm_tilde = mass_composite_body(I0, Im, Bij, Bi0, robot)

    # 7. Generalized inertia matrix
    H0, H0m, Hm = generalized_inertia_matrix(M0_tilde, Mm_tilde, Bij, Bi0, P0, pm, robot)

    # 8. Convective inertia matrix
    C0, C0m, Cm0, Cm = convective_inertia_matrix(
        t0, tm, I0, Im, M0_tilde, Mm_tilde, Bij, Bi0, P0, pm, robot
    )

    # 9. Contact Jacobian (end-effector of last link)
    r_contact = rL[:, -1]
    Jc_base, Jc_joints = compute_contact_jacobian(r_contact, r0, rL, P0, pm, robot)

    # 10. Angular momentum
    state_with_dynamics = dict(state)
    state_with_dynamics.update({
        'H0': H0, 'H0m': H0m, 'Hm': Hm,
        'I0': I0, 'Im': Im, 'u0': u0,
    })
    h_total, h_satellite, h_robot, h_RW = compute_angular_momentum(
        state_with_dynamics, robot, env
    )

    # Package output
    sys = {
        # Kinematics
        'RJ': RJ, 'RL': RL, 'rJ': rJ, 'rL': rL, 'e': e, 'g': g,
        # Differential kinematics
        'Bij': Bij, 'Bi0': Bi0, 'P0': P0, 'pm': pm,
        # Velocities
        't0': t0, 'tm': tm,
        # Inertias
        'I0': I0, 'Im': Im,
        # Composite body
        'M0_tilde': M0_tilde, 'Mm_tilde': Mm_tilde,
        # Generalized inertia
        'H0': H0, 'H0m': H0m, 'Hm': Hm,
        # Coriolis
        'C0': C0, 'C0m': C0m, 'Cm0': Cm0, 'Cm': Cm,
        # Contact
        'Jc_base': Jc_base, 'Jc_joints': Jc_joints, 'r_contact': r_contact,
        # Angular momentum
        'h_total': h_total, 'h_satellite': h_satellite,
        'h_robot': h_robot, 'h_RW': h_RW,
        # State references
        'r0': np.asarray(r0).ravel(),
        'quat_base': state['quat_base'],
        'quat_satellite': state.get('quat_satellite', np.array([0., 0., 0., 1.])),
        'u0': u0,
        'omega_satellite': state['omega_satellite'],
    }
    return sys
