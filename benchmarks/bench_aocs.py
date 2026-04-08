"""Benchmarks for AOCS subsystem: momentum disturbance estimator and
wheel torque command computation.
"""
import numpy as np
import pytest

from crawlbot.aocs.force_estimator import (
    MomentumDisturbanceEstimator,
    EstimatorConfig,
    compute_aocs_command,
)


# ---------------------------------------------------------------------------
# bench_estimator_update
# ---------------------------------------------------------------------------
def test_bench_estimator_update(benchmark, force_estimator):
    """Benchmark MomentumDisturbanceEstimator.update (finite-difference)."""
    np.random.seed(42)
    r_com = np.array([0.35, 0.0, 0.0])
    v_com = np.array([0.01, 0.0, 0.0])
    L_com = np.zeros(3)
    omega_s = np.array([0.0, 0.0, 0.001])

    # Prime the estimator with one call so internal state is initialised
    force_estimator.update(r_com, v_com, L_com, omega_s)

    result = benchmark(force_estimator.update, r_com, v_com, L_com, omega_s)
    assert result is not None
    assert result.shape == (3,)


# ---------------------------------------------------------------------------
# bench_aocs_command
# ---------------------------------------------------------------------------
def test_bench_aocs_command(benchmark):
    """Benchmark compute_aocs_command (closed-form wheel torque)."""
    np.random.seed(42)
    H_dot = np.array([0.01, -0.005, 0.002])
    omega_s = np.array([0.0, 0.0, 0.001])
    hw = np.zeros(3)
    hw_target = np.zeros(3)
    K_omega = 50.0
    K_h = 0.5
    tau_w_max = 5.0

    result = benchmark(
        compute_aocs_command,
        H_dot, omega_s, hw, hw_target, K_omega, K_h, tau_w_max,
    )
    assert result is not None
    assert result.shape == (3,)
    # Torque should be bounded
    assert np.all(np.abs(result) <= tau_w_max + 1e-12)


# ---------------------------------------------------------------------------
# bench_estimator_analytical
# ---------------------------------------------------------------------------
def test_bench_estimator_analytical(benchmark, force_estimator):
    """Benchmark MomentumDisturbanceEstimator.update_analytical (Variant B)."""
    np.random.seed(42)
    r_com = np.array([0.35, 0.0, 0.0])
    v_com = np.array([0.01, 0.0, 0.0])
    L_com = np.zeros(3)
    L_com_prev = np.zeros(3)
    a_com = np.array([0.001, 0.0, 0.0])
    omega_s = np.array([0.0, 0.0, 0.001])

    # Prime with one call
    force_estimator.update_analytical(
        r_com, v_com, L_com, L_com_prev, a_com, omega_s,
    )

    result = benchmark(
        force_estimator.update_analytical,
        r_com, v_com, L_com, L_com_prev, a_com, omega_s,
    )
    assert result is not None
    assert result.shape == (3,)
