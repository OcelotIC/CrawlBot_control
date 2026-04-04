"""Shared pytest fixtures for CrawlBot_control benchmark suite.

Session-scoped where possible to amortise expensive setup (model loading,
solver building) across all benchmarks within a session.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import pinocchio as pin

PROJECT_ROOT = Path(__file__).resolve().parent.parent
URDF_PATH = str(PROJECT_ROOT / "models" / "VISPA_crawling_fixed.urdf")
MJCF_PATH = str(PROJECT_ROOT / "models" / "VISPA_crawling_rwa3_8pct.xml")


# ---------------------------------------------------------------------------
# Robot interface (Pinocchio)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def robot_interface():
    """Session-scoped RobotInterface loaded from URDF."""
    from crawlbot.core.robot_interface import RobotInterface
    return RobotInterface(URDF_PATH, tau_max=20.0, gravity='zero')


# ---------------------------------------------------------------------------
# Nominal state
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def nominal_state(robot_interface):
    """Neutral q and zero v in Pinocchio convention."""
    q = pin.neutral(robot_interface.model)
    v = np.zeros(robot_interface.model.nv)
    return q, v


@pytest.fixture(scope="session")
def nominal_robot_state(robot_interface, nominal_state):
    """RobotState computed at the nominal configuration."""
    q, v = nominal_state
    return robot_interface.update(q.copy(), v.copy())


# ---------------------------------------------------------------------------
# MuJoCo model + data
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def mujoco_model():
    """Session-scoped MuJoCo model (RWA-3, 8 pct mass ratio)."""
    import mujoco
    return mujoco.MjModel.from_xml_path(MJCF_PATH)


@pytest.fixture
def mujoco_data(mujoco_model):
    """Function-scoped MuJoCo data (fresh each benchmark)."""
    import mujoco
    data = mujoco.MjData(mujoco_model)
    mujoco.mj_forward(mujoco_model, data)
    return data


# ---------------------------------------------------------------------------
# NMPC solver (session-scoped, expensive to build)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def centroidal_nmpc():
    """Built CentroidalNMPC with benchmark config (suppressed output)."""
    import io
    import contextlib
    from crawlbot.solvers.centroidal_nmpc import CentroidalNMPC, CentroidalNMPCConfig

    cfg = CentroidalNMPCConfig(
        N=8, dt=0.1,
        robot_mass=71.0,
        f_max=25.0, tau_max=8.0,
        hw_min=-5.0 * np.ones(3),
        hw_max=5.0 * np.ones(3),
        solver_opts={'ipopt.print_level': 0, 'print_time': 0,
                     'ipopt.max_iter': 200},
    )
    nmpc = CentroidalNMPC(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        nmpc.build()
    return nmpc


# ---------------------------------------------------------------------------
# Contact configs
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def single_contact_config():
    """ContactConfig in single-support-A phase."""
    from crawlbot.solvers.contact_phase import ContactConfig, ContactPhase
    return ContactConfig.from_phase(
        ContactPhase.SINGLE_A,
        r_contact_A=np.array([0.0, 0.3, 0.0]),
        r_contact_B=np.array([0.0, -0.3, 0.0]),
    )


@pytest.fixture(scope="session")
def double_contact_config():
    """ContactConfig in double-support phase."""
    from crawlbot.solvers.contact_phase import ContactConfig, ContactPhase
    return ContactConfig.from_phase(
        ContactPhase.DOUBLE,
        r_contact_A=np.array([0.0, 0.3, 0.0]),
        r_contact_B=np.array([0.0, -0.3, 0.0]),
    )


# ---------------------------------------------------------------------------
# NMPC nominal inputs (reusable dict for solver benchmarks)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def nmpc_nominal_inputs():
    """Dict with nominal NMPC inputs for benchmarking."""
    return dict(
        r_com=np.array([0.35, 0.0, 0.0]),
        v_com=np.zeros(3),
        L_com=np.zeros(3),
        hw=np.zeros(3),
        r_ref=np.array([0.35, 0.0, 0.0]),
        v_ref=np.zeros(3),
    )


# ---------------------------------------------------------------------------
# Force estimator (function-scoped for isolation)
# ---------------------------------------------------------------------------
@pytest.fixture
def force_estimator():
    """Fresh MomentumDisturbanceEstimator for each benchmark."""
    from crawlbot.aocs.force_estimator import MomentumDisturbanceEstimator, EstimatorConfig
    cfg = EstimatorConfig(robot_mass=71.0, dt=0.01, filter_tau=0.016)
    return MomentumDisturbanceEstimator(config=cfg)
