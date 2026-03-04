"""Trajectory generation for the three benchmark experiments.

Translated from setup_trajectories.m.
5th-order polynomial trajectories with zero velocity and acceleration at
boundaries.
"""

import numpy as np


def setup_trajectories(experiment_id):
    """Generate smooth reference trajectories.

    Parameters
    ----------
    experiment_id : int – 1, 2, or 3.

    Returns
    -------
    traj : dict – Trajectory data (pos, vel, acc, quat, omega, alpha, t).
    sim_params : dict – Simulation parameters (dt, T_motion, T_total, time).
    """
    # Temporal parameters
    sim_params = {
        'dt': 0.01,             # 100 Hz
        'T_motion': 30.0,       # seconds
        'T_stabilization': 40.0,
    }
    sim_params['T_total'] = sim_params['T_motion'] + sim_params['T_stabilization']
    sim_params['time'] = np.arange(0, sim_params['T_total'] + sim_params['dt'] / 2,
                                    sim_params['dt'])

    N = len(sim_params['time'])

    # Select experiment
    if experiment_id == 1:
        description = 'Straight line crossing satellite CoM'
        pos_start = np.array([-7.5, 0.0, 1.0])
        pos_goal = np.array([7.5, 0.0, 1.0])
        is_circular = False
    elif experiment_id == 2:
        description = 'Straight line with CoM offset'
        pos_start = np.array([1.0, 1.0, 1.0])
        pos_goal = np.array([4.0, 2.0, 1.0])
        is_circular = False
    elif experiment_id == 3:
        description = 'Circular arc (20 deg on edge)'
        r_structure = 7.5
        theta_start = 0.0
        theta_end = np.deg2rad(20)
        pos_start = np.array([
            r_structure * np.cos(theta_start),
            r_structure * np.sin(theta_start),
            1.0,
        ])
        pos_goal = np.array([
            r_structure * np.cos(theta_end),
            r_structure * np.sin(theta_end),
            1.0,
        ])
        is_circular = True
    else:
        raise ValueError("Experiment ID must be 1, 2, or 3")

    # Generate smooth trajectory (5th order polynomial)
    T_motion = sim_params['T_motion']
    t = sim_params['time']
    idx_motion = t <= T_motion
    t_motion = t[idx_motion]
    N_motion = len(t_motion)

    # Normalized parameter tau in [0, 1]
    tau = t_motion / T_motion

    # 5th order polynomial: s(tau)
    s = 10*tau**3 - 15*tau**4 + 6*tau**5
    s_dot = (30*tau**2 - 60*tau**3 + 30*tau**4) / T_motion
    s_ddot = (60*tau - 180*tau**2 + 120*tau**3) / (T_motion**2)

    # Initialize
    pos = np.zeros((3, N))
    vel = np.zeros((3, N))
    acc = np.zeros((3, N))

    if not is_circular:
        direction = pos_goal - pos_start
        for d in range(3):
            pos[d, :N_motion] = pos_start[d] + direction[d] * s
            vel[d, :N_motion] = direction[d] * s_dot
            acc[d, :N_motion] = direction[d] * s_ddot
    else:
        theta = theta_start + s * (theta_end - theta_start)
        theta_dot = s_dot * (theta_end - theta_start)
        theta_ddot = s_ddot * (theta_end - theta_start)

        pos[0, :N_motion] = r_structure * np.cos(theta)
        pos[1, :N_motion] = r_structure * np.sin(theta)
        pos[2, :N_motion] = 1.0

        vel[0, :N_motion] = -r_structure * np.sin(theta) * theta_dot
        vel[1, :N_motion] = r_structure * np.cos(theta) * theta_dot
        vel[2, :N_motion] = 0.0

        acc[0, :N_motion] = (-r_structure * np.sin(theta) * theta_ddot
                             - r_structure * np.cos(theta) * theta_dot**2)
        acc[1, :N_motion] = (r_structure * np.cos(theta) * theta_ddot
                             - r_structure * np.sin(theta) * theta_dot**2)
        acc[2, :N_motion] = 0.0

    # Stabilization phase: hold final position
    idx_stab = t > T_motion
    N_stab = np.sum(idx_stab)
    for d in range(3):
        pos[d, idx_stab] = pos[d, N_motion - 1]
    vel[:, idx_stab] = 0.0
    acc[:, idx_stab] = 0.0

    # Orientation (constant identity)
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (N, 1)).T  # Note: not SPART convention for traj
    omega = np.zeros((3, N))
    alpha = np.zeros((3, N))

    traj = {
        'description': description,
        't': sim_params['time'],
        'pos': pos,
        'vel': vel,
        'acc': acc,
        'quat': quat,
        'omega': omega,
        'alpha': alpha,
        'pos_start': pos_start,
        'pos_goal': pos_goal,
        'experiment_id': experiment_id,
    }

    # Validation
    v_init = np.linalg.norm(vel[:, 0])
    v_final = np.linalg.norm(vel[:, N_motion - 1])
    if v_init > 1e-8 or v_final > 1e-8:
        import warnings
        warnings.warn("Initial/final velocities non-zero!")

    print(f"  v_max = {np.max(np.linalg.norm(vel, axis=0)):.3f} m/s")
    print(f"  Distance = {np.linalg.norm(pos_goal - pos_start):.2f} m")

    return traj, sim_params
