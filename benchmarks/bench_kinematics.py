"""Benchmarks for kinematics operations: Pinocchio update, Jacobians,
state conversions, and inverse kinematics.
"""
import numpy as np
import pytest
import pinocchio as pin

from crawlbot.core.state_conversions import mujoco_to_pinocchio, pinocchio_to_mujoco
from crawlbot.core.ik import dock_configuration


# ---------------------------------------------------------------------------
# bench_robot_update
# ---------------------------------------------------------------------------
def test_bench_robot_update(benchmark, robot_interface, nominal_state):
    """Benchmark RobotInterface.update(q, v) at neutral config."""
    q, v = nominal_state

    def do_update():
        return robot_interface.update(q.copy(), v.copy())

    result = benchmark(do_update)
    assert result is not None
    assert result.r_com.shape == (3,)


# ---------------------------------------------------------------------------
# bench_contact_jacobians
# ---------------------------------------------------------------------------
def test_bench_contact_jacobians(benchmark, robot_interface, nominal_state):
    """Benchmark contact Jacobian computation (both contacts active)."""
    q, v = nominal_state
    robot_interface.update(q.copy(), v.copy())

    result = benchmark(robot_interface.get_contact_jacobians, True, True)
    J_contacts, Jdot_dq = result
    assert J_contacts is not None
    assert J_contacts.shape[0] == 12  # 6 * 2 contacts


# ---------------------------------------------------------------------------
# bench_state_conversion_mj_to_pin
# ---------------------------------------------------------------------------
def test_bench_state_conversion_mj_to_pin(benchmark):
    """Benchmark mujoco_to_pinocchio state conversion (RWA model)."""
    np.random.seed(42)
    # RWA model: qpos length 29, qvel length 27
    mj_qpos = np.zeros(29)
    mj_qvel = np.zeros(27)
    # Set valid quaternions (wxyz convention in MuJoCo)
    mj_qpos[3] = 1.0   # struct quaternion w
    mj_qpos[13] = 1.0  # torso quaternion w (offset by 3 for RWA angles)
    # Small perturbation on positions
    mj_qpos[0:3] = np.array([0.0, 0.0, 0.0])
    mj_qpos[10:13] = np.array([0.35, 0.0, 0.0])

    result = benchmark(mujoco_to_pinocchio, mj_qpos, mj_qvel)
    pin_q, pin_v = result
    assert pin_q.shape == (19,)
    assert pin_v.shape == (18,)


# ---------------------------------------------------------------------------
# bench_state_conversion_pin_to_mj
# ---------------------------------------------------------------------------
def test_bench_state_conversion_pin_to_mj(benchmark, robot_interface):
    """Benchmark pinocchio_to_mujoco state conversion."""
    pin_q = pin.neutral(robot_interface.model)
    pin_v = np.zeros(robot_interface.model.nv)

    result = benchmark(pinocchio_to_mujoco, pin_q, pin_v, True)
    mj_qpos, mj_qvel = result
    assert mj_qpos.shape[0] >= 26
    assert mj_qvel.shape[0] >= 24


# ---------------------------------------------------------------------------
# bench_ik_solve
# ---------------------------------------------------------------------------
def test_bench_ik_solve(benchmark, robot_interface):
    """Benchmark dock_configuration IK solve."""
    model = robot_interface.model
    anchor_a = pin.SE3(np.eye(3), np.array([0.4, 0.3, 0.0]))
    anchor_b = pin.SE3(np.eye(3), np.array([0.4, -0.3, 0.0]))

    result = benchmark.pedantic(
        dock_configuration,
        args=(model, anchor_a, anchor_b),
        rounds=20,
        warmup_rounds=2,
    )
    assert result is not None
    assert result.shape == (model.nq,)
