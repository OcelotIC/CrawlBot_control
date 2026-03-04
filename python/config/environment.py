"""Environment configuration for MIRROR system.

Translated from setup_environment.m.
"""

import numpy as np


def setup_environment():
    """Set up satellite, actuator, and control parameters.

    Returns
    -------
    env : dict – Environment configuration.
    """
    env = {}

    # --- Satellite ---
    r = 7.5    # radius [m]
    h = 0.3    # height [m]
    m = 2040   # mass [kg]

    env['satellite'] = {
        'mass': m,
        'radius': r,
        'height': h,
        'inertia': np.diag([
            (1/12) * m * (3*r**2 + h**2),
            (1/12) * m * (3*r**2 + h**2),
            (1/2) * m * r**2,
        ]),
    }

    # --- Standard Interconnect ---
    env['SI'] = {
        'F_max': 3000.0,
        'tau_max': 300.0,
    }

    # --- Reaction Wheels ---
    n_wheels = 4
    h_max = 50.0
    theta = 54.74 * np.pi / 180

    env['RW'] = {
        'n_wheels': n_wheels,
        'h_max': h_max,
        'h_total_max': n_wheels * h_max,
        'tau_max': 10.0,
        'installation_matrix': np.array([
            [np.sin(theta)*np.cos(0),       np.sin(theta)*np.cos(np.pi/2),
             np.sin(theta)*np.cos(np.pi),   np.sin(theta)*np.cos(3*np.pi/2)],
            [np.sin(theta)*np.sin(0),       np.sin(theta)*np.sin(np.pi/2),
             np.sin(theta)*np.sin(np.pi),   np.sin(theta)*np.sin(3*np.pi/2)],
            [np.cos(theta),                 np.cos(theta),
             np.cos(theta),                 np.cos(theta)],
        ]),
    }

    # --- Control gains ---
    # QP weights: Qr (tracking), Qb (satellite stabilization), Qc (wrench reg.)
    # The paper uses a rigid-body robot where Fc fully controls the satellite
    # disturbance.  For our multi-body model, joint dynamics create additional
    # coupling, so we disable satellite state feedback (Kb=Db=0) to avoid
    # feedback oscillations and let the QP minimize ||Ad'Fc||² directly.
    env['control'] = {
        'Qr': np.eye(6) * 1.0,       # Robot tracking
        'Qb': np.eye(6) * 10.0,      # Satellite wrench minimization
        'Qc': np.eye(6) * 0.01,      # Wrench regularization
        'Kr': np.diag([50, 50, 50, 25, 25, 25]),
        'Dr': np.diag([10, 10, 10, 5, 5, 5]),
        'Kb': np.diag([0, 0, 0]),     # No satellite attitude feedback
        'Db': np.diag([0, 0, 0]),     # No satellite rate feedback
    }

    return env
