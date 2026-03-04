"""Shared test fixtures and utilities."""

import sys
import os
import numpy as np
import pytest

# Ensure python package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from python.spart.robot_model import urdf2robot
from python.config.environment import setup_environment
from python.config.trajectories import setup_trajectories
from python.control.controller import LutzeQPController


@pytest.fixture(scope="session")
def project_root():
    return os.path.join(os.path.dirname(__file__), '..', '..')


@pytest.fixture(scope="session")
def urdf_path(project_root):
    return os.path.join(project_root, 'URDF_models', 'MAR_DualArm_6DoF.urdf')


@pytest.fixture(scope="session")
def robot_and_keys(urdf_path):
    return urdf2robot(urdf_path)


@pytest.fixture(scope="session")
def robot(robot_and_keys):
    return robot_and_keys[0]


@pytest.fixture(scope="session")
def env():
    return setup_environment()


@pytest.fixture(scope="session")
def controller(env):
    return LutzeQPController(env)


@pytest.fixture
def default_qm():
    """Default joint configuration for the dual 6-DOF arms."""
    return np.deg2rad([15, -30, 60, 0, -30, 0, -15, -30, 60, 0, -30, 0])


@pytest.fixture
def identity_state(robot, default_qm):
    """Initial state at origin with identity orientation."""
    return {
        'q_base': np.array([0.0, 0.0, 0.0]),
        'quat_base': np.array([0.0, 0.0, 0.0, 1.0]),
        'qm': default_qm,
        'omega_base': np.zeros(3),
        'u_base': np.zeros(3),
        'um': np.zeros(robot['n_q']),
        'omega_satellite': np.zeros(3),
        'quat_satellite': np.array([0.0, 0.0, 0.0, 1.0]),
        'h_RW_stored': np.zeros(3),
    }
