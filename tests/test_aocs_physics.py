"""Tests for the AOCS estimator and command computation.

Validates MomentumDisturbanceEstimator behavior (H_{r/O} computation,
finite-difference derivative, EMA filtering, transport term) and the
compute_aocs_command function (feedforward, damping, desaturation, clipping).
"""

import numpy as np
import pytest

from crawlbot.aocs.force_estimator import (
    MomentumDisturbanceEstimator,
    EstimatorConfig,
    compute_aocs_command,
)


# ---------------------------------------------------------------------------
# Estimator: H_{r/O} computation
# ---------------------------------------------------------------------------

class TestEstimatorHrO:

    def test_estimator_H_rO_computation(self, force_estimator):
        """Verify H_{r/O} = L_com + cross(r_com, m * v_com)."""
        r_com = np.array([1.0, 0.0, 0.0])
        v_com = np.array([0.0, 1.0, 0.0])
        L_com = np.array([0.0, 0.0, 0.5])
        omega_s = np.zeros(3)

        force_estimator.update(r_com, v_com, L_com, omega_s)

        m = force_estimator._m
        expected = L_com + np.cross(r_com, m * v_com)
        np.testing.assert_allclose(
            force_estimator.H_rO, expected, atol=1e-12,
            err_msg="H_{r/O} does not match L_com + r_com x (m * v_com)",
        )


# ---------------------------------------------------------------------------
# Estimator: derivative behavior
# ---------------------------------------------------------------------------

class TestEstimatorDerivative:

    def test_estimator_zero_derivative_constant_input(self):
        """Constant inputs should produce H_dot -> 0 after warmup."""
        cfg = EstimatorConfig(robot_mass=71.0, dt=0.01, filter_tau=0.016)
        est = MomentumDisturbanceEstimator(config=cfg)

        r_com = np.array([0.1, 0.0, 0.0])
        v_com = np.array([0.0, 0.05, 0.0])
        L_com = np.array([0.0, 0.0, 0.1])
        omega_s = np.zeros(3)

        for _ in range(20):
            H_dot = est.update(r_com, v_com, L_com, omega_s)

        np.testing.assert_allclose(
            H_dot, np.zeros(3), atol=1e-6,
            err_msg="H_dot should converge to zero with constant inputs",
        )

    def test_estimator_step_response(self):
        """A step change in L_com should produce transient H_dot that decays."""
        cfg = EstimatorConfig(
            robot_mass=71.0, dt=0.01, filter_tau=0.016,
            include_transport=False,
        )
        est = MomentumDisturbanceEstimator(config=cfg)

        r_com = np.zeros(3)
        v_com = np.zeros(3)
        omega_s = np.zeros(3)
        L_com_before = np.array([0.0, 0.0, 0.0])
        L_com_after = np.array([1.0, 0.0, 0.0])

        # Warmup with initial L_com
        for _ in range(20):
            est.update(r_com, v_com, L_com_before, omega_s)

        # Step change
        H_dot_step = est.update(r_com, v_com, L_com_after, omega_s)
        assert np.linalg.norm(H_dot_step) > 0.1, (
            f"H_dot should be non-zero right after step, got {H_dot_step}"
        )

        # Let EMA converge
        for _ in range(200):
            H_dot = est.update(r_com, v_com, L_com_after, omega_s)

        np.testing.assert_allclose(
            H_dot, np.zeros(3), atol=1e-4,
            err_msg="H_dot should decay back to zero after step settles",
        )


# ---------------------------------------------------------------------------
# Estimator: transport term
# ---------------------------------------------------------------------------

class TestEstimatorTransport:

    def test_estimator_transport_term(self):
        """With constant H and non-zero omega_s, H_dot = cross(omega_s, H_rO)."""
        cfg = EstimatorConfig(
            robot_mass=71.0, dt=0.01, filter_tau=0.016,
            include_transport=True,
        )
        est = MomentumDisturbanceEstimator(config=cfg)

        r_com = np.array([0.5, 0.0, 0.0])
        v_com = np.array([0.0, 0.1, 0.0])
        L_com = np.array([0.0, 0.0, 1.0])
        omega_s = np.array([0.0, 0.0, 0.1])

        # Warmup so the finite-difference term converges to zero
        for _ in range(200):
            H_dot = est.update(r_com, v_com, L_com, omega_s)

        # Expected: cross(omega_s, H_rO) since dH/dt_struct -> 0
        H_rO = L_com + np.cross(r_com, 71.0 * v_com)
        expected_transport = np.cross(omega_s, H_rO)

        np.testing.assert_allclose(
            H_dot, expected_transport, atol=1e-3,
            err_msg="After warmup, H_dot should equal cross(omega_s, H_rO)",
        )


# ---------------------------------------------------------------------------
# Estimator: reset
# ---------------------------------------------------------------------------

class TestEstimatorReset:

    def test_estimator_reset(self):
        """After reset(), estimator behaves like a fresh instance."""
        cfg = EstimatorConfig(robot_mass=71.0, dt=0.01, filter_tau=0.016)
        est = MomentumDisturbanceEstimator(config=cfg)

        r_com = np.array([0.1, 0.2, 0.0])
        v_com = np.array([0.0, 0.1, 0.0])
        L_com = np.array([0.0, 0.0, 0.5])
        omega_s = np.zeros(3)

        # Run several updates to build up state
        for _ in range(10):
            est.update(r_com, v_com, L_com, omega_s)
        assert est.initialized

        # Reset
        est.reset()
        assert not est.initialized
        np.testing.assert_allclose(est.H_dot, np.zeros(3), atol=1e-15)
        np.testing.assert_allclose(est.H_rO, np.zeros(3), atol=1e-15)

        # First update after reset: H_dot should be zero (no previous sample)
        H_dot = est.update(r_com, v_com, L_com, omega_s)
        np.testing.assert_allclose(
            H_dot, np.zeros(3), atol=1e-15,
            err_msg="First update after reset should yield zero H_dot",
        )


# ---------------------------------------------------------------------------
# Estimator: analytical variant
# ---------------------------------------------------------------------------

class TestEstimatorAnalytical:

    def test_estimator_analytical_variant(self):
        """update_analytical with known values matches expected H_dot."""
        cfg = EstimatorConfig(
            robot_mass=71.0, dt=0.01, filter_tau=0.0,
            include_transport=False,
        )
        est = MomentumDisturbanceEstimator(config=cfg)

        r_com = np.array([1.0, 0.0, 0.0])
        v_com = np.array([0.0, 0.0, 0.0])
        a_com = np.array([0.0, 1.0, 0.0])
        L_com = np.array([0.0, 0.0, 1.0])
        L_com_prev = np.array([0.0, 0.0, 0.5])
        omega_s = np.zeros(3)

        H_dot = est.update_analytical(
            r_com, v_com, L_com, L_com_prev, a_com, omega_s,
        )

        # L_dot = (L_com - L_com_prev) / dt = [0, 0, 0.5] / 0.01 = [0, 0, 50]
        L_dot_expected = (L_com - L_com_prev) / cfg.dt
        # cross(r_com, m * a_com) = cross([1,0,0], 71*[0,1,0]) = [0,0,71]
        orbital_dot = np.cross(r_com, cfg.robot_mass * a_com)
        expected = L_dot_expected + orbital_dot

        np.testing.assert_allclose(
            H_dot, expected, atol=1e-10,
            err_msg="Analytical H_dot does not match expected value",
        )


# ---------------------------------------------------------------------------
# compute_aocs_command tests
# ---------------------------------------------------------------------------

class TestAOCSCommand:

    def test_aocs_command_basic(self):
        """Basic feedforward: tau_w = -H_dot_est when other terms are zero."""
        H_dot_est = np.array([1.0, 0.0, 0.0])
        omega_s = np.zeros(3)
        hw = np.zeros(3)

        tau_w = compute_aocs_command(
            H_dot_est, omega_s, hw, K_omega=0.0, K_h=0.0,
        )

        np.testing.assert_allclose(
            tau_w, np.array([-1.0, 0.0, 0.0]), atol=1e-15,
            err_msg="tau_w should be -H_dot_est for pure feedforward",
        )

    def test_aocs_command_clipping(self):
        """Verify tau_w_max clipping."""
        H_dot_est = np.array([100.0, 0.0, 0.0])
        omega_s = np.zeros(3)
        hw = np.zeros(3)

        tau_w = compute_aocs_command(
            H_dot_est, omega_s, hw, K_omega=0.0, K_h=0.0, tau_w_max=5.0,
        )

        np.testing.assert_allclose(
            tau_w[0], -5.0, atol=1e-15,
            err_msg="tau_w[0] should be clipped to -5.0",
        )
        # All components within bounds
        assert np.all(np.abs(tau_w) <= 5.0 + 1e-15)

    def test_aocs_command_damping(self):
        """Non-zero omega_s adds +K_omega * omega_s damping.

        Sign convention (Newton-Euler about structure CoM):
            I_s · ω̇_s = -Ḣ_s - τ_w
        For ω_s>0 to brake (ω̇_s<0), τ_w must EXCEED -Ḣ_s — the
        damping contribution adds positive. Prior versions of this
        formula used -K_ω·ω_s (wrong sign) — never exposed because
        all earlier tests used ω_s=0.
        """
        H_dot_est = np.zeros(3)
        omega_s = np.array([0.1, 0.0, 0.0])
        hw = np.zeros(3)

        tau_w = compute_aocs_command(
            H_dot_est, omega_s, hw, K_omega=50.0, K_h=0.0, tau_w_max=100.0,
        )

        # Expected: +K_omega * omega_s = +50 * 0.1 = +5.0 in x
        np.testing.assert_allclose(
            tau_w[0], +5.0, atol=1e-15,
            err_msg="Damping must be +K_omega * omega_s (positive sign)",
        )

    def test_aocs_command_damping_physical_consistency(self):
        """Damping term must produce a torque that REDUCES ω_s.

        Setup: ω_s positive about z, no disturbance, no desat.
        Reaction on structure from wheels is -τ_w (Newton's 3rd).
        For braking: I_s · ω̇_s = -τ_w < 0 → τ_w > 0.
        """
        H_dot_est = np.zeros(3)
        hw = np.zeros(3)
        # ω_s positive in all three axes
        for axis in range(3):
            omega_s = np.zeros(3)
            omega_s[axis] = 0.05
            tau_w = compute_aocs_command(
                H_dot_est, omega_s, hw,
                K_omega=50.0, K_h=0.0, tau_w_max=100.0,
            )
            assert tau_w[axis] > 0, (
                f"axis {axis}: τ_w={tau_w[axis]} should be POSITIVE to brake "
                f"positive ω_s (Newton-Euler about struct CoM)")
            # And opposite sign reverses it.
            omega_s[axis] = -0.05
            tau_w = compute_aocs_command(
                H_dot_est, omega_s, hw,
                K_omega=50.0, K_h=0.0, tau_w_max=100.0,
            )
            assert tau_w[axis] < 0, (
                f"axis {axis}: τ_w={tau_w[axis]} should be NEGATIVE to brake "
                f"negative ω_s")

    def test_aocs_command_desaturation(self):
        """Non-zero hw with K_h > 0 adds desaturation term."""
        H_dot_est = np.zeros(3)
        omega_s = np.zeros(3)
        hw_current = np.array([2.0, 0.0, 0.0])
        hw_target = np.array([0.0, 0.0, 0.0])

        tau_w = compute_aocs_command(
            H_dot_est, omega_s, hw_current,
            hw_target=hw_target, K_omega=0.0, K_h=0.5, tau_w_max=100.0,
        )

        # Expected: -K_h * (hw_current - hw_target) = -0.5 * 2 = -1.0
        np.testing.assert_allclose(
            tau_w[0], -1.0, atol=1e-15,
            err_msg="Desaturation contribution should be -K_h * (hw - hw*)",
        )

    def test_aocs_command_all_terms(self):
        """All terms combined: feedforward + damping + desaturation."""
        H_dot_est = np.array([1.0, 0.0, 0.0])
        omega_s = np.array([0.1, 0.0, 0.0])
        hw_current = np.array([2.0, 0.0, 0.0])
        hw_target = np.zeros(3)

        tau_w = compute_aocs_command(
            H_dot_est, omega_s, hw_current,
            hw_target=hw_target, K_omega=50.0, K_h=0.5, tau_w_max=100.0,
        )

        # Expected x: -Ḣ + K_ω·ω - K_h·(hw - hw*)
        #           = -1.0 + 50·0.1 - 0.5·2.0 = -1.0 + 5.0 - 1.0 = +3.0
        expected_x = -1.0 + 50.0 * 0.1 - 0.5 * 2.0
        np.testing.assert_allclose(
            tau_w[0], expected_x, atol=1e-15,
            err_msg="Combined command: τ_w = -Ḣ + K_ω·ω - K_h·(hw-hw*)",
        )
