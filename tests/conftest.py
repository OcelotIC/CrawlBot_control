"""Shared pytest fixtures for CrawlBot_control test suite.

Provides session-scoped fixtures for expensive resources (Pinocchio model,
MuJoCo model) and function-scoped fixtures for mutable state.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
URDF_PATH = str(PROJECT_ROOT / "models" / "VISPA_crawling_fixed.urdf")
MJCF_RWA3_8PCT = str(PROJECT_ROOT / "models" / "VISPA_crawling_rwa3_8pct.xml")
MJCF_RWA3 = str(PROJECT_ROOT / "models" / "VISPA_crawling_rwa3.xml")


# ---------------------------------------------------------------------------
# Robot interface (Pinocchio)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def robot_interface():
    """Session-scoped RobotInterface loaded from URDF."""
    from crawlbot.core.robot_interface import RobotInterface
    return RobotInterface(URDF_PATH, tau_max=20.0, gravity='zero')


@pytest.fixture(scope="session")
def pinocchio_model(robot_interface):
    """Pinocchio model from the robot interface."""
    return robot_interface.model


# ---------------------------------------------------------------------------
# MuJoCo model + data
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def mujoco_model():
    """Session-scoped MuJoCo model (RWA-3, 8% mass ratio)."""
    import mujoco
    return mujoco.MjModel.from_xml_path(MJCF_RWA3_8PCT)


@pytest.fixture
def mujoco_data(mujoco_model):
    """Function-scoped MuJoCo data (fresh each test)."""
    import mujoco
    data = mujoco.MjData(mujoco_model)
    mujoco.mj_forward(mujoco_model, data)
    return data


# ---------------------------------------------------------------------------
# Nominal state (valid q, v for benchmarks / tests)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def nominal_state(robot_interface):
    """A known-good (q, v) in Pinocchio convention.

    Uses the neutral configuration (identity base pose, zero joints)
    with zero velocity.
    """
    import pinocchio as pin
    q = pin.neutral(robot_interface.model)
    v = np.zeros(robot_interface.model.nv)
    return q, v


@pytest.fixture(scope="session")
def nominal_robot_state(robot_interface, nominal_state):
    """RobotState computed at the nominal configuration."""
    q, v = nominal_state
    return robot_interface.update(q.copy(), v.copy())


# ---------------------------------------------------------------------------
# Simulation config
# ---------------------------------------------------------------------------
@pytest.fixture
def sim_config():
    """Default SimConfig."""
    from crawlbot.simulation.config import SimConfig
    return SimConfig()


# ---------------------------------------------------------------------------
# Contact scheduler with synthetic anchors
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def contact_scheduler():
    """ContactScheduler with default synthetic anchor grid, 4-step plan."""
    from crawlbot.planning.contact_scheduler import ContactScheduler
    sched = ContactScheduler(dt_ss=6.0, dt_ds=0.5)
    sched.plan_traversal(start_a=0, start_b=0, n_steps=4)
    return sched


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def centroidal_nmpc():
    """Built CentroidalNMPC with default config."""
    from crawlbot.solvers.centroidal_nmpc import CentroidalNMPC, CentroidalNMPCConfig
    cfg = CentroidalNMPCConfig(
        N=8, dt=0.1,
        robot_mass=71.0,
        f_max=25.0, tau_max=8.0,
        L_max=10.0,
        tau_w_max=5.0,
        p_max=50.0,  # linear momentum bound [kg·m/s]
        solver_opts={'ipopt.print_level': 0, 'print_time': 0,
                     'ipopt.max_iter': 200},
    )
    nmpc = CentroidalNMPC(cfg)
    nmpc.build()
    return nmpc


@pytest.fixture(scope="session")
def wholebody_qp(robot_interface):
    """Built WholeBodyQP with default config."""
    from crawlbot.solvers.wholebody_qp import WholeBodyQP, WholeBodyQPConfig
    cfg = WholeBodyQPConfig(nq=12, nc_max=2)
    qp = WholeBodyQP(cfg)
    return qp


# ---------------------------------------------------------------------------
# Force estimator
# ---------------------------------------------------------------------------
@pytest.fixture
def force_estimator():
    """MomentumDisturbanceEstimator with default config."""
    from crawlbot.aocs.force_estimator import MomentumDisturbanceEstimator, EstimatorConfig
    cfg = EstimatorConfig(robot_mass=71.0, dt=0.01, filter_tau=0.016)
    return MomentumDisturbanceEstimator(config=cfg)


# ---------------------------------------------------------------------------
# Helper: single-contact config for solver tests
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
