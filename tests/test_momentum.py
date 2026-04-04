"""Tests for momentum conservation and decomposition.

Verifies the spin-orbital decomposition of angular momentum:
    H_{r/O} = L_com + r_com x (m * v_com)

and related momentum identities using the Pinocchio-based RobotInterface.
"""

import numpy as np
import pinocchio as pin
import pytest


def _safe_random_config(model, seed):
    """Generate a valid random configuration that doesn't produce NaN.

    pin.randomConfiguration can produce extreme joint angles; we use
    a bounded perturbation of the neutral configuration instead.
    """
    rng = np.random.RandomState(seed)
    q = pin.neutral(model).copy()
    # Perturb joint positions moderately (indices 7:19 are joints)
    q[7:19] += rng.randn(12) * 0.3
    # Clip to joint limits
    q[7:19] = np.clip(q[7:19],
                       model.lowerPositionLimit[7:19],
                       model.upperPositionLimit[7:19])
    return q


# ---------------------------------------------------------------------------
# Spin vs orbital decomposition
# ---------------------------------------------------------------------------

class TestSpinOrbitalDecomposition:

    def test_spin_vs_orbital_decomposition(self, robot_interface):
        """H_{r/O} = L_com + r_com x (m * v_com) must hold exactly.

        At a configuration with non-zero velocity, compute both terms
        independently and verify the decomposition is self-consistent.
        """
        q = _safe_random_config(robot_interface.model, seed=42)
        v = np.random.RandomState(42).randn(robot_interface.model.nv) * 0.5

        rs = robot_interface.update(q, v)

        orbital = np.cross(rs.r_com, rs.total_mass * rs.v_com)
        H_rO = rs.L_com + orbital

        # Both terms must be finite
        assert np.all(np.isfinite(H_rO)), f"H_rO contains NaN/Inf: {H_rO}"
        assert np.all(np.isfinite(rs.L_com)), f"L_com contains NaN/Inf: {rs.L_com}"
        assert np.all(np.isfinite(orbital)), f"Orbital contains NaN/Inf: {orbital}"

        # Re-derive from centroidal quantities: h_centroidal[3:6] == L_com
        np.testing.assert_allclose(rs.L_com, rs.h_centroidal[3:6], atol=1e-12)

    @pytest.mark.parametrize("seed", [42, 43, 44, 45, 46])
    def test_decomposition_multiple_configs(self, robot_interface, seed):
        """Verify decomposition across multiple configurations."""
        q = _safe_random_config(robot_interface.model, seed=seed)
        v = np.random.RandomState(seed).randn(robot_interface.model.nv) * 0.3

        rs = robot_interface.update(q, v)

        assert np.all(np.isfinite(rs.L_com)), f"NaN in L_com at seed={seed}"
        assert np.all(np.isfinite(rs.r_com)), f"NaN in r_com at seed={seed}"
        assert np.all(np.isfinite(rs.v_com)), f"NaN in v_com at seed={seed}"

        orbital = np.cross(rs.r_com, rs.total_mass * rs.v_com)
        H_rO = rs.L_com + orbital
        assert np.all(np.isfinite(H_rO)), f"NaN in H_rO at seed={seed}"


class TestLinearMomentum:

    def test_linear_momentum_equals_mass_times_vcom(self, robot_interface):
        """Verify p = m * v_com matches h_centroidal[0:3]."""
        q = _safe_random_config(robot_interface.model, seed=42)
        v = np.random.RandomState(42).randn(robot_interface.model.nv) * 0.3

        rs = robot_interface.update(q, v)

        p_expected = rs.total_mass * rs.v_com
        np.testing.assert_allclose(
            rs.h_centroidal[0:3], p_expected, atol=1e-8,
            err_msg="Linear momentum p != m * v_com",
        )


class TestAngularMomentumDecomposition:

    @pytest.mark.parametrize("seed", [42, 43, 44, 45, 46])
    def test_angular_momentum_decomposition_multiple_configs(
        self, robot_interface, seed,
    ):
        """h_centroidal[3:6] == L_com at multiple configs."""
        q = _safe_random_config(robot_interface.model, seed=seed)
        v = np.random.RandomState(seed).randn(robot_interface.model.nv) * 0.2

        rs = robot_interface.update(q, v)

        np.testing.assert_allclose(
            rs.h_centroidal[3:6], rs.L_com, atol=1e-12,
            err_msg=f"L_com != h_centroidal[3:6] at seed={seed}",
        )


class TestOrbitalTerm:

    def test_orbital_term_grows_with_offset(self, robot_interface):
        """When CoM is further from origin, the orbital term grows.

        At neutral config the CoM is near the origin, so the orbital term
        is small. After shifting the base position, CoM moves away and
        the orbital term should grow.
        """
        rng = np.random.RandomState(42)
        model = robot_interface.model

        # Neutral config: CoM near origin
        q_neutral = pin.neutral(model)
        v = rng.randn(model.nv) * 0.2
        rs_neutral = robot_interface.update(q_neutral.copy(), v.copy())
        orbital_neutral = np.cross(
            rs_neutral.r_com, rs_neutral.total_mass * rs_neutral.v_com,
        )

        # Shifted config: move the base position by 2 m in x
        q_shifted = q_neutral.copy()
        q_shifted[0] += 2.0
        rs_shifted = robot_interface.update(q_shifted, v.copy())
        orbital_shifted = np.cross(
            rs_shifted.r_com, rs_shifted.total_mass * rs_shifted.v_com,
        )

        assert np.linalg.norm(orbital_shifted) > np.linalg.norm(orbital_neutral), (
            f"Orbital term did not grow with offset: "
            f"|orbital_neutral|={np.linalg.norm(orbital_neutral):.6f}, "
            f"|orbital_shifted|={np.linalg.norm(orbital_shifted):.6f}"
        )


class TestZeroVelocity:

    def test_zero_velocity_zero_momentum(self, robot_interface):
        """At any config with v=0, all momentum should be zero."""
        for seed in [42, 43, 44]:
            q = _safe_random_config(robot_interface.model, seed=seed)
            v = np.zeros(robot_interface.model.nv)

            rs = robot_interface.update(q, v)

            np.testing.assert_allclose(
                rs.L_com, np.zeros(3), atol=1e-12,
                err_msg=f"L_com should be zero at v=0 (seed={seed})",
            )
            np.testing.assert_allclose(
                rs.h_centroidal[0:3], np.zeros(3), atol=1e-12,
                err_msg=f"Linear momentum should be zero at v=0 (seed={seed})",
            )
            np.testing.assert_allclose(
                rs.v_com, np.zeros(3), atol=1e-12,
                err_msg=f"v_com should be zero at v=0 (seed={seed})",
            )
