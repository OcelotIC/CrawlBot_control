"""Physics invariant tests for the VISPA robot model.

Validates fundamental properties of the rigid-body dynamics computed by
Pinocchio via RobotInterface: mass matrix symmetry/positive-definiteness,
Jacobian consistency, centroidal momentum relations, and contact utilities.
"""

import numpy as np
import pinocchio as pin
import pytest

from crawlbot.solvers.contact_phase import skew, compute_momentum_map


# ---------------------------------------------------------------------------
# Mass matrix tests
# ---------------------------------------------------------------------------

class TestMassMatrix:

    def test_mass_matrix_symmetry(self, nominal_robot_state):
        """H should be symmetric at the nominal configuration."""
        H = nominal_robot_state.H
        np.testing.assert_allclose(H, H.T, atol=1e-12)

    def test_mass_matrix_positive_definite(self, nominal_robot_state):
        """All eigenvalues of H should be strictly positive."""
        eigvals = np.linalg.eigvalsh(nominal_robot_state.H)
        assert np.all(eigvals > 0), (
            f"Non-positive eigenvalues found: {eigvals[eigvals <= 0]}"
        )

    def test_mass_matrix_symmetry_random_config(self, robot_interface):
        """H should be symmetric at 5 random configurations."""
        np.random.seed(42)
        for _ in range(5):
            q = pin.randomConfiguration(robot_interface.model)
            v = np.random.randn(robot_interface.model.nv) * 0.1
            state = robot_interface.update(q, v)
            np.testing.assert_allclose(state.H, state.H.T, atol=1e-12)


# ---------------------------------------------------------------------------
# CoM / Jacobian consistency
# ---------------------------------------------------------------------------

class TestCoMConsistency:

    def test_com_velocity_jacobian_consistency(self, robot_interface):
        """v_com should equal J_com @ v at nominal and random configs."""
        np.random.seed(42)

        # Nominal config
        q0 = pin.neutral(robot_interface.model)
        v0 = np.zeros(robot_interface.model.nv)
        state0 = robot_interface.update(q0, v0)
        np.testing.assert_allclose(
            state0.v_com, state0.J_com @ state0.v, atol=1e-12,
        )

        # Random configs with nonzero velocity
        for _ in range(3):
            q = pin.randomConfiguration(robot_interface.model)
            v = np.random.randn(robot_interface.model.nv) * 0.1
            state = robot_interface.update(q, v)
            np.testing.assert_allclose(
                state.v_com, state.J_com @ state.v, atol=1e-10,
            )


# ---------------------------------------------------------------------------
# Centroidal momentum
# ---------------------------------------------------------------------------

class TestCentroidalMomentum:

    def test_centroidal_momentum_linear(self, robot_interface):
        """h_centroidal[0:3] should equal total_mass * v_com."""
        np.random.seed(42)
        q = pin.neutral(robot_interface.model)
        v = np.random.randn(robot_interface.model.nv) * 0.1
        state = robot_interface.update(q, v)
        np.testing.assert_allclose(
            state.h_centroidal[0:3],
            state.total_mass * state.v_com,
            atol=1e-10,
        )

    def test_centroidal_momentum_angular_zero_velocity(self, robot_interface):
        """At zero velocity, angular momentum L_com should be zero."""
        q = pin.neutral(robot_interface.model)
        v = np.zeros(robot_interface.model.nv)
        state = robot_interface.update(q, v)
        np.testing.assert_allclose(state.L_com, np.zeros(3), atol=1e-14)

    def test_centroidal_momentum_angular_consistency(self, robot_interface):
        """L_com should match h_centroidal[3:6] at nonzero velocity."""
        np.random.seed(42)
        q = pin.randomConfiguration(robot_interface.model)
        v = np.random.randn(robot_interface.model.nv) * 0.1
        state = robot_interface.update(q, v)
        np.testing.assert_allclose(
            state.L_com, state.h_centroidal[3:6], atol=1e-14,
        )


# ---------------------------------------------------------------------------
# Bias vector
# ---------------------------------------------------------------------------

class TestBiasVector:

    def test_bias_zero_velocity(self, robot_interface):
        """With v=0 and zero gravity, bias C should be zero."""
        q = pin.neutral(robot_interface.model)
        v = np.zeros(robot_interface.model.nv)
        state = robot_interface.update(q, v)
        np.testing.assert_allclose(state.C, np.zeros(robot_interface.model.nv), atol=1e-14)

    def test_dynamics_consistency_zero_acceleration(self, robot_interface):
        """At zero velocity / zero gravity, C must be zero (=> H@0 + C = 0)."""
        q = pin.neutral(robot_interface.model)
        v = np.zeros(robot_interface.model.nv)
        state = robot_interface.update(q, v)
        np.testing.assert_allclose(state.C, np.zeros(robot_interface.model.nv), atol=1e-14)


# ---------------------------------------------------------------------------
# Skew-symmetric matrix utilities
# ---------------------------------------------------------------------------

class TestSkew:

    def test_skew_antisymmetry(self):
        """skew(v) should be antisymmetric: S == -S.T."""
        np.random.seed(42)
        for _ in range(5):
            v = np.random.randn(3)
            S = skew(v)
            np.testing.assert_allclose(S, -S.T, atol=1e-15)

    def test_skew_cross_product(self):
        """skew(v) @ w should equal np.cross(v, w)."""
        np.random.seed(42)
        for _ in range(5):
            v = np.random.randn(3)
            w = np.random.randn(3)
            np.testing.assert_allclose(
                skew(v) @ w, np.cross(v, w), atol=1e-14,
            )


# ---------------------------------------------------------------------------
# Momentum map
# ---------------------------------------------------------------------------

class TestMomentumMap:

    def test_momentum_map_dimensions(
        self, single_contact_config, double_contact_config,
    ):
        """compute_momentum_map should return (3, 12) for any contact mode."""
        r_com = np.zeros(3)
        M_single = compute_momentum_map(r_com, single_contact_config)
        M_double = compute_momentum_map(r_com, double_contact_config)
        assert M_single.shape == (3, 12)
        assert M_double.shape == (3, 12)

    def test_momentum_map_zero_for_inactive(self, single_contact_config):
        """In single-support-A, columns 6:12 (contact B) should be zero."""
        r_com = np.array([0.1, 0.0, 0.0])
        M = compute_momentum_map(r_com, single_contact_config)
        np.testing.assert_allclose(M[:, 6:12], np.zeros((3, 6)), atol=1e-15)


# ---------------------------------------------------------------------------
# Contact Jacobians
# ---------------------------------------------------------------------------

class TestContactJacobians:

    def test_contact_jacobian_both_active(self, robot_interface):
        """Both contacts active: J is (12, nv), Jdot_dq is (12,)."""
        q = pin.neutral(robot_interface.model)
        v = np.zeros(robot_interface.model.nv)
        robot_interface.update(q, v)
        J, Jdot_dq = robot_interface.get_contact_jacobians(True, True)
        assert J.shape == (12, robot_interface.model.nv)
        assert Jdot_dq.shape == (12,)

    def test_contact_jacobian_only_A(self, robot_interface):
        """Only A active: J is (6, nv), Jdot_dq is (6,)."""
        q = pin.neutral(robot_interface.model)
        v = np.zeros(robot_interface.model.nv)
        robot_interface.update(q, v)
        J, Jdot_dq = robot_interface.get_contact_jacobians(True, False)
        assert J.shape == (6, robot_interface.model.nv)
        assert Jdot_dq.shape == (6,)

    def test_contact_jacobian_neither_active(self, robot_interface):
        """No active contacts: returns (None, None)."""
        q = pin.neutral(robot_interface.model)
        v = np.zeros(robot_interface.model.nv)
        robot_interface.update(q, v)
        J, Jdot_dq = robot_interface.get_contact_jacobians(False, False)
        assert J is None
        assert Jdot_dq is None
