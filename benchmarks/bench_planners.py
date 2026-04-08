"""Benchmarks for planning subsystem: torso planner, contact scheduler,
and anchor grid generation.
"""
import numpy as np
import pytest

from crawlbot.planning.torso_planner import TorsoPlanner
from crawlbot.planning.contact_scheduler import ContactScheduler, make_anchor_grid


# ---------------------------------------------------------------------------
# bench_torso_reference
# ---------------------------------------------------------------------------
def test_bench_torso_reference(benchmark):
    """Benchmark TorsoPlanner.reference_at(t) during a hold phase."""
    planner = TorsoPlanner()
    planner.set_hold(
        np.array([0.4, 0.0, 0.0]),
        np.eye(3),
        np.array([0.35, 0.0, 0.0]),
    )

    result = benchmark(planner.reference_at, 0.0)
    assert result is not None
    assert result.p.shape == (3,)
    assert result.R.shape == (3, 3)


# ---------------------------------------------------------------------------
# bench_contact_config_at
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def scheduler_with_plan():
    """ContactScheduler with a 4-step traversal plan."""
    sched = ContactScheduler(dt_ss=6.0, dt_ds=0.5)
    sched.plan_traversal(start_a=0, start_b=0, n_steps=4)
    return sched


def test_bench_contact_config_at(benchmark, scheduler_with_plan):
    """Benchmark ContactScheduler.contact_config_at(t)."""
    result = benchmark(scheduler_with_plan.contact_config_at, 3.0)
    assert result is not None


# ---------------------------------------------------------------------------
# bench_contact_sequence_horizon
# ---------------------------------------------------------------------------
def test_bench_contact_sequence_horizon(benchmark, scheduler_with_plan):
    """Benchmark contact_sequence_over_horizon (N=20 steps)."""
    result = benchmark(
        scheduler_with_plan.contact_sequence_over_horizon,
        t=1.0, dt=0.1, N=20,
    )
    assert result is not None
    assert len(result) == 20


# ---------------------------------------------------------------------------
# bench_make_anchor_grid
# ---------------------------------------------------------------------------
def test_bench_make_anchor_grid(benchmark):
    """Benchmark make_anchor_grid (default 6 anchors)."""
    result = benchmark(make_anchor_grid)
    anchors_a, anchors_b = result
    assert len(anchors_a) == 6
    assert len(anchors_b) == 6
    assert anchors_a[0].shape == (3,)
