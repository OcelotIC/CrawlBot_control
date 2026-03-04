"""Main simulation loop.

Translated from simulate_experiment.m.
"""

import numpy as np
from ..spart.attitude import quat_angles321
from ..dynamics.system_state import compute_system_state
from .integrator import integrate_dynamics


def simulate_experiment(env, robot, traj, sim_params, controller, use_optimization):
    """Run a complete simulation experiment.

    Parameters
    ----------
    env : dict – Environment configuration.
    robot : dict – Robot model.
    traj : dict – Reference trajectory.
    sim_params : dict – Simulation parameters.
    controller : LutzeQPController – Controller instance.
    use_optimization : bool – If True, use QP optimization.

    Returns
    -------
    results : dict – Simulation results.
    """
    print("  Initializing...")

    # Initial state
    state = {
        'q_base': traj['pos'][:, 0].copy(),
        'quat_base': np.array([0.0, 0.0, 0.0, 1.0]),  # SPART: scalar last
        'qm': np.deg2rad([
            15, -30, 60, 0, -30, 0,    # arm 1
            -15, -30, 60, 0, -30, 0,   # arm 2
        ]),
        'omega_base': np.zeros(3),
        'u_base': np.zeros(3),
        'um': np.zeros(robot['n_q']),
        'omega_satellite': np.zeros(3),
        'quat_satellite': np.array([0.0, 0.0, 0.0, 1.0]),
        'h_RW_stored': np.zeros(3),
    }

    # Pre-allocate results
    N = len(traj['t'])
    results = {
        't': traj['t'],
        'alpha': np.zeros(N),
        'beta': np.zeros(N),
        'gamma': np.zeros(N),
        'Fc': np.zeros((6, N)),
        'h_RW': np.zeros((3, N)),
        'h_total': np.zeros((3, N)),
        'pos_robot': np.zeros((3, N)),
        'pos_error': np.zeros((3, N)),
    }

    # Time loop
    for k in range(N - 1):
        # Compute system state
        sys = compute_system_state(state, robot, env)

        # Solve controller
        Fc_opt, tau_joints = controller.solve(sys, traj, k, use_optimization)

        # Integrate dynamics
        state = integrate_dynamics(state, tau_joints, Fc_opt, sys, robot,
                                   sim_params['dt'], env)

        # Store results
        results['Fc'][:, k] = Fc_opt

        # Satellite Euler angles (321 / ZYX sequence)
        angles_sat = quat_angles321(state['quat_satellite'])
        results['alpha'][k] = angles_sat[0]
        results['beta'][k] = angles_sat[1]
        results['gamma'][k] = angles_sat[2]

        results['h_RW'][:, k] = state['h_RW_stored']
        results['h_total'][:, k] = sys['h_total']
        results['pos_robot'][:, k] = state['q_base']
        results['pos_error'][:, k] = traj['pos'][:, k] - state['q_base']

        if (k + 1) % 1000 == 0:
            print(f"    {100 * (k + 1) / N:.1f}%")

    # Last point
    results['Fc'][:, -1] = results['Fc'][:, -2]
    results['alpha'][-1] = results['alpha'][-2]
    results['beta'][-1] = results['beta'][-2]
    results['gamma'][-1] = results['gamma'][-2]
    results['h_RW'][:, -1] = results['h_RW'][:, -2]
    results['h_total'][:, -1] = results['h_total'][:, -2]
    results['pos_robot'][:, -1] = state['q_base']
    results['pos_error'][:, -1] = traj['pos'][:, -1] - state['q_base']

    print("  Complete!")
    return results
