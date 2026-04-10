"""M5 — TorsoPlanner + SwingPlanner 6D tests.

Covers:
  - TorsoPlanner.l_com_reference_at(t) — returns I_torso·ω_ref, zero
    outside phases.
  - SwingPlanner delayed-cosine orientation timing:
      σ(0) = σ(τ_d) = 0,  σ(1) = 1
      σ̇ continuous at τ_d and τ=1
  - SwingPlanner SLERP endpoint exactness: R_ee(t_start) = R_start,
    R_ee(t_end) = R_dock.
  - Full 6D output format.
"""

import numpy as np
import pinocchio as pin
import pytest

from crawlbot.planning.swing_planner import SwingPlanner
from crawlbot.planning.torso_planner import TorsoPlanner
from crawlbot.planning.contact_scheduler import ContactScheduler


# ---------------------------------------------------------------------------
# SwingPlanner fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def swing_planner():
    sched = ContactScheduler(dt_ss=6.0, dt_ds=0.5)
    sched.plan_traversal(start_a=2, start_b=2, n_steps=1)
    sp = SwingPlanner(sched, clearance=0.03, rotation_delay_ratio=0.2)
    return sp


# ---------------------------------------------------------------------------
# Delayed cosine timing function
# ---------------------------------------------------------------------------

class TestDelayedCosine:
    def test_zero_before_delay(self):
        tau_d = 0.2
        for tau in [0.0, 0.05, 0.15, 0.2]:
            assert SwingPlanner._delayed_cosine(tau, tau_d) == 0.0

    def test_one_at_end(self):
        assert SwingPlanner._delayed_cosine(1.0, 0.2) == pytest.approx(1.0)
        # Past-end clamp
        assert SwingPlanner._delayed_cosine(1.5, 0.2) == 1.0

    def test_monotonic_after_delay(self):
        tau_d = 0.2
        prev = 0.0
        for tau in np.linspace(0.2, 1.0, 50):
            val = SwingPlanner._delayed_cosine(tau, tau_d)
            assert val >= prev - 1e-12
            prev = val

    def test_derivative_zero_at_boundaries(self):
        """σ̇(τ_d) and σ̇(1) must both be zero (smooth start and end)."""
        tau_d = 0.2
        assert abs(SwingPlanner._delayed_cosine_dot(tau_d, tau_d)) < 1e-12
        assert abs(SwingPlanner._delayed_cosine_dot(1.0, tau_d)) < 1e-12

    def test_derivative_positive_in_middle(self):
        tau_d = 0.2
        assert SwingPlanner._delayed_cosine_dot(0.5, tau_d) > 0

    def test_derivative_continuity_at_delay(self):
        """σ̇ must be continuous at τ_d (both sides = 0)."""
        tau_d = 0.2
        eps = 1e-6
        left = SwingPlanner._delayed_cosine_dot(tau_d - eps, tau_d)
        right = SwingPlanner._delayed_cosine_dot(tau_d + eps, tau_d)
        # Left side is 0 by definition; right side should be ~0
        # because the cosine derivative is 0 at u=0.
        assert abs(left - right) < 1e-4

    def test_concentrates_rotation_in_second_half(self):
        """At τ=0.5, σ should still be small since rotation is delayed."""
        tau_d = 0.2
        half = SwingPlanner._delayed_cosine(0.5, tau_d)
        # With tau_d=0.2, τ=0.5 is u=0.375 → σ=0.5*(1-cos(0.375π))≈0.309
        assert 0.2 < half < 0.4

        # Most rotation happens after τ=0.5
        assert SwingPlanner._delayed_cosine(0.75, tau_d) > 0.6


# ---------------------------------------------------------------------------
# SwingPlanner 6D — SLERP endpoint exactness
# ---------------------------------------------------------------------------

class TestSwingSlerp:
    def test_R_start_at_tau_zero(self, swing_planner):
        R_start = pin.exp3(np.array([0.5, -0.3, 0.2]))
        swing_planner.set_swing_orientation(R_start)
        sp_plan = swing_planner.plan
        # Find the single-support phase
        ss_idx = next(i for i, gp in enumerate(sp_plan.phases)
                      if gp.phase.value != 'double')
        t_start = sp_plan.t_start[ss_idx]
        ref = swing_planner.reference_at(t_start + 1e-9)
        assert ref.is_swinging
        assert np.allclose(ref.R_ee, R_start, atol=1e-9)

    def test_R_end_at_tau_one(self, swing_planner):
        R_start = pin.exp3(np.array([0.5, -0.3, 0.2]))
        swing_planner.set_swing_orientation(R_start)
        sp_plan = swing_planner.plan
        ss_idx = next(i for i, gp in enumerate(sp_plan.phases)
                      if gp.phase.value != 'double')
        t_end = sp_plan.t_end[ss_idx]
        ref = swing_planner.reference_at(t_end - 1e-9)
        # R_end is identity (tool-aligned with structure)
        assert np.allclose(ref.R_ee, np.eye(3), atol=1e-6)

    def test_orientation_held_during_delay(self, swing_planner):
        """During the delay window (τ < 0.2), R_ee == R_start exactly."""
        R_start = pin.exp3(np.array([0.5, -0.3, 0.2]))
        swing_planner.set_swing_orientation(R_start)
        sp_plan = swing_planner.plan
        ss_idx = next(i for i, gp in enumerate(sp_plan.phases)
                      if gp.phase.value != 'double')
        t_start = sp_plan.t_start[ss_idx]
        T = sp_plan.phases[ss_idx].duration
        # τ = 0.1 is within the delay window (τ_d = 0.2)
        t_early = t_start + 0.1 * T
        ref = swing_planner.reference_at(t_early)
        assert np.allclose(ref.R_ee, R_start, atol=1e-9)
        # ω_ee is zero during the delay window
        assert np.allclose(ref.omega_ee, 0.0, atol=1e-12)

    def test_omega_smooth_at_delay_boundary(self, swing_planner):
        """ω_ee must be continuous at τ=τ_d (no discontinuity)."""
        R_start = pin.exp3(np.array([0.5, -0.3, 0.2]))
        swing_planner.set_swing_orientation(R_start)
        sp_plan = swing_planner.plan
        ss_idx = next(i for i, gp in enumerate(sp_plan.phases)
                      if gp.phase.value != 'double')
        t_start = sp_plan.t_start[ss_idx]
        T = sp_plan.phases[ss_idx].duration
        tau_d = swing_planner.rotation_delay_ratio
        t_d = t_start + tau_d * T
        eps = 1e-4 * T
        ref_left = swing_planner.reference_at(t_d - eps)
        ref_right = swing_planner.reference_at(t_d + eps)
        # Both should have near-zero ω (left is in delay, right is at
        # the smooth cosine start where σ̇ = 0).
        assert np.linalg.norm(ref_left.omega_ee) < 1e-4
        assert np.linalg.norm(ref_right.omega_ee) < 1e-4

    def test_full_6d_fields_present(self, swing_planner):
        R_start = pin.exp3(np.array([0.3, 0.0, 0.0]))
        swing_planner.set_swing_orientation(R_start)
        sp_plan = swing_planner.plan
        ss_idx = next(i for i, gp in enumerate(sp_plan.phases)
                      if gp.phase.value != 'double')
        t_mid = 0.5 * (sp_plan.t_start[ss_idx] + sp_plan.t_end[ss_idx])
        ref = swing_planner.reference_at(t_mid)
        # All 6D fields populated
        assert ref.p_ee.shape == (3,)
        assert ref.R_ee.shape == (3, 3)
        assert ref.v_ee.shape == (3,)
        assert ref.omega_ee.shape == (3,)
        assert ref.a_ee.shape == (3,)
        assert ref.alpha_ee.shape == (3,)


# ---------------------------------------------------------------------------
# TorsoPlanner L_com_ref
# ---------------------------------------------------------------------------

class TestTorsoLComRef:
    def test_zero_when_no_inertia_set(self):
        tp = TorsoPlanner()
        L = tp.l_com_reference_at(0.0)
        assert np.allclose(L, 0.0)

    def test_zero_outside_phases(self):
        tp = TorsoPlanner()
        tp.set_torso_inertia(np.diag([1.0, 2.0, 3.0]))
        # No phases defined → hold → omega_ref = 0 → L_com_ref = 0
        L = tp.l_com_reference_at(0.0)
        assert np.allclose(L, 0.0)

    def test_matches_I_times_omega_at_peak(self):
        """At the peak of the SLERP trajectory, L = I · ω should match
        the exact relation to within 5% (spec pass criterion).
        """
        tp = TorsoPlanner()
        I_body = np.diag([0.5, 0.8, 0.3])
        tp.set_torso_inertia(I_body)

        p0 = np.zeros(3)
        R0 = np.eye(3)
        p1 = np.array([0.3, 0.0, 0.0])
        # ~30° rotation about +z
        theta = np.radians(30.0)
        R1 = pin.exp3(np.array([0.0, 0.0, theta]))
        dc0 = np.array([0.1, 0.0, -0.2])   # arbitrary CoM offset
        dc1 = np.array([0.1, 0.0, -0.2])
        tp.add_phase(
            t_start=0.0, t_end=6.0,
            p_start=p0, R_start=R0, p_end=p1, R_end=R1,
            delta_com_start=dc0, delta_com_end=dc1)

        # Sample in the middle of the phase
        t = 3.0
        tref = tp.reference_at(t)
        omega = tref.v[3:6]
        # Manually compute expected L
        R_t = tref.R
        I_world = R_t @ I_body @ R_t.T
        L_expected = I_world @ omega

        L_actual = tp.l_com_reference_at(t)
        assert np.allclose(L_actual, L_expected, atol=1e-12)

    def test_nonzero_during_rotation(self):
        tp = TorsoPlanner()
        tp.set_torso_inertia(np.diag([0.5, 0.8, 0.3]))
        tp.add_phase(
            t_start=0.0, t_end=6.0,
            p_start=np.zeros(3), R_start=np.eye(3),
            p_end=np.array([0.3, 0.0, 0.0]),
            R_end=pin.exp3(np.array([0.0, 0.0, np.radians(45.0)])),
            delta_com_start=np.zeros(3),
            delta_com_end=np.zeros(3))
        # At t=3.0 (middle), omega should be at its peak magnitude
        L = tp.l_com_reference_at(3.0)
        assert np.linalg.norm(L) > 1e-3
