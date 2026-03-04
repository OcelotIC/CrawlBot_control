"""Angular momentum computation.

Translated from compute_angular_momentum.m.
"""

import numpy as np


def compute_angular_momentum(state, robot, env):
    """Compute angular momentum components.

    h_total = h_satellite + h_robot + h_RW (constant in orbit).

    Parameters
    ----------
    state : dict – System state (must include H0, H0m, Hm, u0, etc.).
    robot : dict – Robot model.
    env : dict – Environment configuration.

    Returns
    -------
    h_total, h_satellite, h_robot, h_RW : (3,) arrays.
    """
    # 1. Satellite angular momentum
    omega_satellite = state['omega_satellite']
    I_satellite = env['satellite']['inertia']
    h_satellite = I_satellite @ omega_satellite

    # 2. Robot angular momentum (base + manipulator)
    u0 = state['u0']
    H0 = state['H0']
    h_robot_base = H0[:3, :] @ u0

    um = state.get('um', np.zeros(robot['n_q']))
    if 'H0m' in state and 'Hm' in state:
        H0m = state['H0m']
        Hm = state['Hm']
        h_robot_manip = H0m[:3, :] @ um + Hm[:3, :] @ um if Hm.shape[0] >= 3 else np.zeros(3)
    else:
        h_robot_manip = np.zeros(3)

    h_robot = h_robot_base + h_robot_manip

    # 3. Reaction wheels
    h_RW = state.get('h_RW_stored', np.zeros(3))

    # 4. Total
    h_total = h_satellite + h_robot + h_RW

    return h_total, h_satellite, h_robot, h_RW
