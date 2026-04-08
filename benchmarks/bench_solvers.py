"""Benchmarks for CentroidalNMPC solver performance.

Covers cold-start build+solve, warm-start re-solve, full trajectory
extraction, and feedforward acceleration computation.

NMPC state: x = [r_com(3), v_com(3), L_com(3)]  (NX=9)
"""
import io
import contextlib

import numpy as np
import pytest

from crawlbot.solvers.centroidal_nmpc import CentroidalNMPC, CentroidalNMPCConfig
from crawlbot.solvers.contact_phase import ContactConfig, ContactPhase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_nmpc():
    """Build a fresh CentroidalNMPC (cold start)."""
    cfg = CentroidalNMPCConfig(
        N=8, dt=0.1,
        robot_mass=71.0,
        f_max=25.0, tau_max=8.0,
        L_max=10.0, tau_w_max=5.0, p_max=50.0,
        solver_opts={'ipopt.print_level': 0, 'print_time': 0,
                     'ipopt.max_iter': 200},
    )
    nmpc = CentroidalNMPC(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        nmpc.build()
    return nmpc


def _make_double_config():
    return ContactConfig.from_phase(
        ContactPhase.DOUBLE,
        r_contact_A=np.array([0.0, 0.3, 0.0]),
        r_contact_B=np.array([0.0, -0.3, 0.0]),
    )


def _nominal_inputs():
    return dict(
        r_com=np.array([0.35, 0.0, 0.0]),
        v_com=np.zeros(3),
        L_com=np.zeros(3),
        r_com_ref=np.array([0.35, 0.0, 0.0]),
        v_com_ref=np.zeros(3),
    )


# ---------------------------------------------------------------------------
# bench_nmpc_cold_start
# ---------------------------------------------------------------------------
def test_bench_nmpc_cold_start(benchmark):
    """Benchmark cold-start: build NMPC + first solve."""
    contact_cfg = _make_double_config()
    inputs = _nominal_inputs()

    def cold_start():
        nmpc = _make_nmpc()
        result = nmpc.solve(
            **inputs,
            contact_config=contact_cfg,
            warm_start=False,
        )
        return result

    result = benchmark.pedantic(cold_start, rounds=5, warmup_rounds=1)
    assert result is not None
    assert len(result) >= 5


# ---------------------------------------------------------------------------
# bench_nmpc_warm_start
# ---------------------------------------------------------------------------
def test_bench_nmpc_warm_start(benchmark, centroidal_nmpc,
                                double_contact_config, nmpc_nominal_inputs):
    """Benchmark warm-start re-solve on session-scoped NMPC."""
    inp = nmpc_nominal_inputs

    # Warmup solve (outside benchmark)
    centroidal_nmpc.solve(
        r_com=inp['r_com'], v_com=inp['v_com'], L_com=inp['L_com'],
        r_com_ref=inp['r_ref'], v_com_ref=inp['v_ref'],
        contact_config=double_contact_config,
        warm_start=False,
    )

    def warm_solve():
        return centroidal_nmpc.solve(
            r_com=inp['r_com'], v_com=inp['v_com'], L_com=inp['L_com'],
            r_com_ref=inp['r_ref'], v_com_ref=inp['v_ref'],
            contact_config=double_contact_config,
            warm_start=True,
        )

    result = benchmark(warm_solve)
    assert result is not None
    assert len(result) >= 5


# ---------------------------------------------------------------------------
# bench_nmpc_full_trajectory
# ---------------------------------------------------------------------------
def test_bench_nmpc_full_trajectory(benchmark, centroidal_nmpc,
                                     double_contact_config,
                                     nmpc_nominal_inputs):
    """Benchmark get_full_trajectory (solve + state extraction)."""
    inp = nmpc_nominal_inputs

    # Ensure solver is warmed up
    centroidal_nmpc.solve(
        r_com=inp['r_com'], v_com=inp['v_com'], L_com=inp['L_com'],
        r_com_ref=inp['r_ref'], v_com_ref=inp['v_ref'],
        contact_config=double_contact_config,
        warm_start=False,
    )

    def full_traj():
        return centroidal_nmpc.get_full_trajectory(
            r_com=inp['r_com'], v_com=inp['v_com'], L_com=inp['L_com'],
            r_com_ref=inp['r_ref'], v_com_ref=inp['v_ref'],
            contact_config=double_contact_config,
        )

    result = benchmark(full_traj)
    assert result is not None
    x_opt = result[0]
    assert x_opt.shape[0] == 9  # NX = 9


# ---------------------------------------------------------------------------
# bench_feedforward_acceleration
# ---------------------------------------------------------------------------
def test_bench_feedforward_acceleration(benchmark, centroidal_nmpc):
    """Benchmark compute_feedforward_acceleration (trivial arithmetic)."""
    np.random.seed(42)
    lambda_ref = np.random.randn(12) * 0.1

    result = benchmark(centroidal_nmpc.compute_feedforward_acceleration,
                       lambda_ref)
    assert result is not None
    assert result.shape == (3,)
