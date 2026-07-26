"""Regression tests for IK_ANOMALY_REPORT pathologies (B) and (C).

Pre-fix observation (Phase 4 sim_loop run, T15 step 2 anchor pair (3,4)):
    ``manipulability_config_trajectory`` converged to
    ``w_end = 4.09e-08`` — pathologically singular endpoint. Diagnostic
    showed the same call from the same fixture could land anywhere
    from 4e-2 to 4e-8 depending on internal Nelder-Mead path through
    a warm-start cache.

Post-fix targets (per docs/architecture/IK_FORMULATION.md §9 and
Misc/runs/q1_q2/IK_ANOMALY_REPORT.md §6.2):
    - Output is deterministic in the inputs (cost is a function only
      of torso xyz, not Nelder-Mead history).
    - ``w_end > 1e-3`` always (well above the singular band).
    - Multi-start has 7 seeds (or 6 on fallback).

These tests are the regression gate for those properties.
"""
import os
import numpy as np
import pinocchio as pin
import pytest

from crawlbot.core.ik import (
    manipulability_config_trajectory,
    dock_configuration_fixed_rotation,
)


pytestmark = pytest.mark.slow   # every test here takes 5.7-11.8 s (CLEANUP-29 durations)

_FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'fixtures', 'step2_ss_entry_fixture.npz',
)


@pytest.fixture(scope='module')
def step2_fixture():
    """Load the T15 step-2 SS-entry diagnostic fixture."""
    if not os.path.exists(_FIXTURE):
        pytest.skip(f"diagnostic fixture not present at {_FIXTURE}")
    fx = np.load(_FIXTURE, allow_pickle=True)
    return {
        'q_start': np.asarray(fx['q_start'], dtype=float),
        'a_t': np.asarray(fx['se3_a_target_t'], dtype=float),
        'b_t': np.asarray(fx['se3_b_target_t'], dtype=float),
        'anchor_pair': (int(fx['anchor_pair'][0]), int(fx['anchor_pair'][1])),
    }


def test_step2_anomaly_resolved(pinocchio_model, step2_fixture):
    """The Phase 4 step-2 anomaly (w_end=4e-8) must not recur.

    Per IK_ANOMALY_REPORT §3.1 the pre-fix output ranged from 4e-2
    to 4e-8 depending on internal Nelder-Mead path. Post-fix the
    output should be deterministic and well-conditioned (w_end >= 1e-3,
    the safety threshold from IK_FORMULATION §9.3).
    """
    fx = step2_fixture
    q_end, w_worst, w_end = manipulability_config_trajectory(
        model=pinocchio_model,
        anchor_a=fx['a_t'],
        anchor_b=fx['b_t'],
        q_start=fx['q_start'].copy(),
        n_samples=5,
    )

    assert q_end is not None, (
        "Trajectory-aware IK returned None — check that the fixture "
        "is a feasible (3,4) configuration"
    )
    assert w_end > 1e-3, (
        f"w_end={w_end:.2e} below safety threshold (pre-fix this "
        f"could be 4e-8); fix has regressed"
    )
    assert w_worst > 1e-3, (
        f"w_worst={w_worst:.2e} below safety threshold; the K=5 "
        f"interior path passes through a near-singular configuration"
    )


def test_step2_determinism(pinocchio_model, step2_fixture):
    """Two consecutive calls with identical inputs must produce
    bit-identical outputs.

    Pre-fix (warm-start cache) the same fixture could yield outputs
    differing by ~7 orders of magnitude in w_end across runs. Post-fix
    the cost function is a deterministic function of the decision
    variable.
    """
    fx = step2_fixture
    q_a, ww_a, we_a = manipulability_config_trajectory(
        model=pinocchio_model,
        anchor_a=fx['a_t'], anchor_b=fx['b_t'],
        q_start=fx['q_start'].copy(), n_samples=5,
    )
    q_b, ww_b, we_b = manipulability_config_trajectory(
        model=pinocchio_model,
        anchor_a=fx['a_t'], anchor_b=fx['b_t'],
        q_start=fx['q_start'].copy(), n_samples=5,
    )

    np.testing.assert_allclose(
        q_a, q_b, atol=1e-9,
        err_msg="Trajectory-aware IK output is not deterministic in q_end")
    np.testing.assert_allclose(
        we_a, we_b, atol=1e-9,
        err_msg="Trajectory-aware IK is not deterministic in w_end")
    np.testing.assert_allclose(
        ww_a, ww_b, atol=1e-9,
        err_msg="Trajectory-aware IK is not deterministic in w_worst")


def test_seed_set_size(pinocchio_model, step2_fixture):
    """The multi-start should evaluate at least 6 distinct seeds.

    Post-fix the seed set is 7: q_start, midpoint, midpoint+/-x,
    midpoint+/-y, p_fixed (fallback to 6 if fixed-rotation seed
    fails). Test by monkey-patching scipy_minimize to count calls.
    """
    from crawlbot.core import ik as ik_module

    call_log = []
    real_minimize = ik_module.scipy_minimize

    def counting_minimize(*args, **kwargs):
        call_log.append((args[1].copy() if len(args) > 1 else None,))
        return real_minimize(*args, **kwargs)

    ik_module.scipy_minimize = counting_minimize
    try:
        fx = step2_fixture
        q_end, _, _ = manipulability_config_trajectory(
            model=pinocchio_model,
            anchor_a=fx['a_t'], anchor_b=fx['b_t'],
            q_start=fx['q_start'].copy(), n_samples=5,
        )
        assert q_end is not None
        assert len(call_log) >= 6, (
            f"multi-start used only {len(call_log)} seeds; expected >= 6 "
            f"per IK_FORMULATION §9.2"
        )
        # Ensure seeds are distinct (no duplicate x0)
        seeds_arr = np.array([c[0] for c in call_log])
        for i in range(len(seeds_arr)):
            for j in range(i + 1, len(seeds_arr)):
                assert np.linalg.norm(seeds_arr[i] - seeds_arr[j]) > 1e-6, (
                    f"seeds {i} and {j} are identical: {seeds_arr[i]}"
                )
    finally:
        ik_module.scipy_minimize = real_minimize


def test_safety_threshold_triggers_fallback(pinocchio_model, step2_fixture):
    """When w_min_threshold is set above the converged w_end, the IK
    must return None for q_end so the caller falls back to fixed-rotation.

    Construct an artificial scenario by passing a threshold the
    legitimate optimum (w_end ~ 4.4e-2) cannot satisfy.
    """
    fx = step2_fixture
    # First check: legitimate threshold (1e-3) must NOT trigger fallback
    q_end, _, w_end = manipulability_config_trajectory(
        model=pinocchio_model,
        anchor_a=fx['a_t'], anchor_b=fx['b_t'],
        q_start=fx['q_start'].copy(), n_samples=5,
        w_min_threshold=1e-3,
    )
    assert q_end is not None, (
        f"Legitimate threshold 1e-3 spuriously triggered fallback "
        f"(w_end={w_end:.2e})"
    )

    # Second check: oversized threshold (1.0) MUST trigger fallback
    q_end, _, w_end = manipulability_config_trajectory(
        model=pinocchio_model,
        anchor_a=fx['a_t'], anchor_b=fx['b_t'],
        q_start=fx['q_start'].copy(), n_samples=5,
        w_min_threshold=1.0,
    )
    assert q_end is None, (
        f"Oversized threshold 1.0 did NOT trigger fallback "
        f"(w_end={w_end:.2e}); safety check is broken"
    )
