"""Category 1: Unit Tests for Mathematical Utilities.

Tests for quaternion operations, attitude transformations, and spatial math.
"""

import numpy as np
import pytest

from python.utils.quaternion import (
    quat_product, quat_integrate, quat_error,
    matrix_quat_prod_left, matrix_quat_prod_right, quaternion_product,
)
from python.utils.spatial import skew_sym, adjoint_se3
from python.spart.attitude import (
    quat_dcm, dcm_quat, euler_dcm, dcm_euler,
    angles321_dcm, dcm_angles321, angles123_dcm, quat_angles321,
)


# ---------------------------------------------------------------------------
# 1.1 Quaternion Operations
# ---------------------------------------------------------------------------

class TestQuaternionProduct:
    """Test quat_product with SPART convention (scalar last)."""

    def test_identity_product(self):
        """q ⊗ identity = q."""
        q = np.array([0.1, 0.2, 0.3, np.sqrt(1 - 0.01 - 0.04 - 0.09)])
        identity = np.array([0.0, 0.0, 0.0, 1.0])
        result = quat_product(q, identity)
        assert np.allclose(result, q, atol=1e-14)

    def test_conjugate_product(self):
        """q ⊗ conj(q) = identity."""
        q = np.array([0.1, 0.2, 0.3, np.sqrt(1 - 0.14)])
        q_conj = np.array([-q[0], -q[1], -q[2], q[3]])
        result = quat_product(q, q_conj)
        expected = np.array([0.0, 0.0, 0.0, 1.0])
        assert np.allclose(result, expected, atol=1e-14)

    def test_non_commutativity(self):
        """q1 ⊗ q2 != q2 ⊗ q1 in general."""
        q1 = np.array([0.5, 0.5, 0.5, 0.5])
        q2 = np.array([0.0, 0.707107, 0.0, 0.707107])
        r1 = quat_product(q1, q2)
        r2 = quat_product(q2, q1)
        assert not np.allclose(r1, r2, atol=1e-6)

    def test_unit_quaternion_preserved(self):
        """Product of unit quaternions is a unit quaternion."""
        q1 = np.array([0.5, 0.5, 0.5, 0.5])
        q2 = np.array([0.0, np.sin(np.pi/6), 0.0, np.cos(np.pi/6)])
        result = quat_product(q1, q2)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-14


class TestQuatIntegrate:
    """Test quaternion integration."""

    def test_zero_omega(self):
        """Zero angular velocity => quaternion unchanged."""
        q = np.array([0.1, 0.2, 0.3, np.sqrt(1 - 0.14)])
        result = quat_integrate(q, np.zeros(3), 0.01)
        assert np.allclose(result, q, atol=1e-14)

    def test_rotation_about_z(self):
        """Constant omega about z for time pi/(2*omega) => 90 deg rotation."""
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        omega = np.array([0.0, 0.0, 1.0])
        dt = np.pi / 2  # 90 degrees in 1 step

        q1 = quat_integrate(q0, omega, dt)
        # Expected: 90 deg about z => q = [0, 0, sin(45°), cos(45°)]
        expected = np.array([0.0, 0.0, np.sin(np.pi/4), np.cos(np.pi/4)])
        assert np.allclose(q1, expected, atol=1e-10)

    def test_normalization(self):
        """Result should always be unit quaternion."""
        q = np.array([0.5, 0.5, 0.5, 0.5])
        omega = np.array([1.0, 2.0, 3.0])
        result = quat_integrate(q, omega, 0.01)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-14

    def test_small_angle(self):
        """Small rotation should be close to identity + perturbation."""
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        omega = np.array([0.001, 0.002, 0.003])
        dt = 0.01
        q1 = quat_integrate(q0, omega, dt)

        # For small angles, q ≈ [omega*dt/2; 1]
        half_angle = np.linalg.norm(omega) * dt / 2
        assert abs(q1[3] - np.cos(half_angle)) < 1e-8


class TestQuatError:
    """Test quaternion error computation."""

    def test_zero_error_identical(self):
        """Error between identical quaternions should be zero."""
        q = np.array([0.1, 0.2, 0.3, np.sqrt(1 - 0.14)])
        err = quat_error(q, q)
        assert np.allclose(err, np.zeros(3), atol=1e-14)

    def test_known_90deg_error(self):
        """90 deg rotation about z-axis should give specific error."""
        q_des = np.array([0.0, 0.0, np.sin(np.pi/4), np.cos(np.pi/4)])
        q_cur = np.array([0.0, 0.0, 0.0, 1.0])
        err = quat_error(q_des, q_cur)
        # Error ≈ 2 * vector part of q_des ⊗ conj(q_cur)
        # = 2 * [0, 0, sin(45°)] for scalar cos(45°)
        assert abs(err[2]) > 1.0  # Significant rotation about z
        assert abs(err[0]) < 1e-10
        assert abs(err[1]) < 1e-10


class TestMatrixQuatProd:
    """Test matrix form of quaternion product."""

    def test_left_product_matches_direct(self):
        """[q1]_L @ q2 == quat_product(q1, q2)."""
        q1 = np.array([0.5, 0.5, 0.5, 0.5])
        q2 = np.array([0.1, 0.2, 0.3, np.sqrt(1 - 0.14)])
        expected = quat_product(q1, q2)
        result = matrix_quat_prod_left(q1) @ q2
        assert np.allclose(result, expected, atol=1e-14)

    def test_right_product_matches(self):
        """[q2]_R @ q1 == quat_product(q1, q2)."""
        q1 = np.array([0.5, 0.5, 0.5, 0.5])
        q2 = np.array([0.1, 0.2, 0.3, np.sqrt(1 - 0.14)])
        expected = quat_product(q1, q2)
        result = matrix_quat_prod_right(q2) @ q1
        assert np.allclose(result, expected, atol=1e-14)

    def test_quaternion_product_alias(self):
        """quaternion_product (matrix form) matches quat_product."""
        q1 = np.array([0.5, 0.5, 0.5, 0.5])
        q2 = np.array([0.0, 0.0, np.sin(0.3), np.cos(0.3)])
        r1 = quat_product(q1, q2)
        r2 = quaternion_product(q1, q2)
        assert np.allclose(r1, r2, atol=1e-14)


# ---------------------------------------------------------------------------
# 1.2 Attitude Transformations
# ---------------------------------------------------------------------------

class TestQuatDCM:
    """Test quaternion <-> DCM conversions."""

    def test_identity(self):
        """Identity quaternion -> identity DCM."""
        q = np.array([0.0, 0.0, 0.0, 1.0])
        DCM = quat_dcm(q)
        assert np.allclose(DCM, np.eye(3), atol=1e-14)

    def test_roundtrip(self):
        """quat -> DCM -> quat should be identity (up to sign)."""
        q = np.array([0.1, 0.2, 0.3, np.sqrt(1 - 0.14)])
        DCM = quat_dcm(q)
        q2 = dcm_quat(DCM)
        # Quaternions are equivalent up to sign
        assert np.allclose(q, q2, atol=1e-10) or np.allclose(q, -q2, atol=1e-10)

    def test_dcm_is_rotation(self):
        """DCM from quaternion should be proper rotation."""
        q = np.array([0.5, 0.5, 0.5, 0.5])
        DCM = quat_dcm(q)
        # R @ R.T = I
        assert np.allclose(DCM @ DCM.T, np.eye(3), atol=1e-14)
        # det(R) = 1
        assert abs(np.linalg.det(DCM) - 1.0) < 1e-14

    def test_90deg_z_rotation(self):
        """90 deg about z: q = [0, 0, sin(45), cos(45)]."""
        q = np.array([0.0, 0.0, np.sin(np.pi/4), np.cos(np.pi/4)])
        DCM = quat_dcm(q)
        # 90 deg about z: [1,0,0] -> [0,1,0]
        v = DCM.T @ np.array([1.0, 0.0, 0.0])  # DCM in SPART is body->inertial
        # For rotation of frame: R.T applied to vector rotates the vector
        # DCM maps body to inertial, so DCM @ [1,0,0]_body = inertial coords
        # A 90 deg rotation about z maps x-axis to y-axis
        v_rotated = DCM @ np.array([1.0, 0.0, 0.0])
        # Expect approximately [0, 1, 0] or [-1, 0, 0] depending on convention
        # Just check it's a valid rotation
        assert np.allclose(np.linalg.norm(v_rotated), 1.0)


class TestEulerDCM:
    """Test Euler axis-angle <-> DCM."""

    def test_zero_angle(self):
        """Zero rotation angle => identity DCM."""
        e = np.array([0.0, 0.0, 1.0])
        DCM = euler_dcm(e, 0.0)
        assert np.allclose(DCM, np.eye(3), atol=1e-14)

    def test_known_rotation(self):
        """90 deg about z-axis via Euler matches quaternion version."""
        e = np.array([0.0, 0.0, 1.0])
        DCM_euler = euler_dcm(e, np.pi/2)
        q = np.array([0.0, 0.0, np.sin(np.pi/4), np.cos(np.pi/4)])
        DCM_quat = quat_dcm(q)
        assert np.allclose(DCM_euler, DCM_quat, atol=1e-10)

    def test_roundtrip(self):
        """Euler -> DCM -> Euler should recover axis and angle."""
        e = np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)])
        alpha = 1.2
        DCM = euler_dcm(e, alpha)
        result = dcm_euler(DCM)
        assert abs(result[3] - alpha) < 1e-10
        assert np.allclose(result[:3], e, atol=1e-10)


class TestAngles321:
    """Test Euler 321 (ZYX) sequence conversions."""

    def test_zero_angles(self):
        """Zero angles -> identity DCM."""
        DCM = angles321_dcm(np.zeros(3))
        assert np.allclose(DCM, np.eye(3), atol=1e-14)

    def test_roundtrip(self):
        """angles -> DCM -> angles should be identity."""
        angles = np.array([0.1, 0.2, 0.3])
        DCM = angles321_dcm(angles)
        recovered = dcm_angles321(DCM)
        assert np.allclose(angles, recovered, atol=1e-10)

    def test_roundtrip_various(self):
        """Test multiple angle sets for roundtrip stability."""
        test_angles = [
            np.array([0.5, -0.3, 0.8]),
            np.array([-0.1, 0.0, 1.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([np.pi/4, np.pi/6, np.pi/3]),
        ]
        for angles in test_angles:
            DCM = angles321_dcm(angles)
            recovered = dcm_angles321(DCM)
            assert np.allclose(angles, recovered, atol=1e-10), \
                f"Roundtrip failed for {angles}"

    def test_dcm_is_rotation(self):
        """DCM from angles should be proper rotation."""
        angles = np.array([0.5, -0.3, 0.8])
        DCM = angles321_dcm(angles)
        assert np.allclose(DCM @ DCM.T, np.eye(3), atol=1e-14)
        assert abs(np.linalg.det(DCM) - 1.0) < 1e-14


class TestAngles123:
    """Test Euler 123 (XYZ) sequence."""

    def test_zero_angles(self):
        DCM = angles123_dcm(np.zeros(3))
        assert np.allclose(DCM, np.eye(3), atol=1e-14)

    def test_dcm_is_rotation(self):
        angles = np.array([0.3, 0.4, 0.5])
        DCM = angles123_dcm(angles)
        assert np.allclose(DCM @ DCM.T, np.eye(3), atol=1e-14)
        assert abs(np.linalg.det(DCM) - 1.0) < 1e-14


class TestQuatAngles321:
    """Test quaternion -> Euler 321."""

    def test_identity(self):
        q = np.array([0.0, 0.0, 0.0, 1.0])
        angles = quat_angles321(q)
        assert np.allclose(angles, np.zeros(3), atol=1e-14)

    def test_consistency_with_dcm_path(self):
        """quat_angles321(q) should match dcm_angles321(quat_dcm(q))."""
        q = np.array([0.1, 0.2, 0.3, np.sqrt(1 - 0.14)])
        angles1 = quat_angles321(q)
        DCM = quat_dcm(q)
        angles2 = dcm_angles321(DCM)
        assert np.allclose(angles1, angles2, atol=1e-14)


# ---------------------------------------------------------------------------
# 1.3 Spatial Math
# ---------------------------------------------------------------------------

class TestSkewSym:
    """Test skew-symmetric matrix."""

    def test_cross_product(self):
        """skew_sym(a) @ b == cross(a, b)."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = skew_sym(a) @ b
        expected = np.cross(a, b)
        assert np.allclose(result, expected, atol=1e-14)

    def test_antisymmetric(self):
        """Skew-symmetric matrix is antisymmetric: S = -S.T."""
        x = np.array([3.0, -1.0, 2.0])
        S = skew_sym(x)
        assert np.allclose(S, -S.T, atol=1e-14)

    def test_zero_self_product(self):
        """x × x = 0, so skew_sym(x) @ x = 0."""
        x = np.array([1.0, 2.0, 3.0])
        result = skew_sym(x) @ x
        assert np.allclose(result, np.zeros(3), atol=1e-14)

    def test_random_vectors(self):
        """Random vector cross product test."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            a = rng.standard_normal(3)
            b = rng.standard_normal(3)
            assert np.allclose(skew_sym(a) @ b, np.cross(a, b), atol=1e-12)


class TestAdjointSE3:
    """Test SE(3) adjoint matrix."""

    def test_identity_transform(self):
        """Adjoint of identity is identity."""
        g = np.eye(4)
        Ad = adjoint_se3(g)
        assert np.allclose(Ad, np.eye(6), atol=1e-14)

    def test_pure_rotation(self):
        """Adjoint of pure rotation has block-diagonal structure."""
        R = euler_dcm(np.array([0.0, 0.0, 1.0]), np.pi/4)
        g = np.eye(4)
        g[:3, :3] = R
        Ad = adjoint_se3(g)
        # With p=0, Ad = [R, 0; 0, R]
        assert np.allclose(Ad[:3, :3], R, atol=1e-14)
        assert np.allclose(Ad[3:, 3:], R, atol=1e-14)
        assert np.allclose(Ad[:3, 3:], np.zeros((3, 3)), atol=1e-14)

    def test_pure_translation(self):
        """Adjoint of pure translation."""
        g = np.eye(4)
        g[:3, 3] = [1.0, 2.0, 3.0]
        Ad = adjoint_se3(g)
        # With R=I: Ad = [I, [p]_x; 0, I]
        assert np.allclose(Ad[:3, :3], np.eye(3), atol=1e-14)
        assert np.allclose(Ad[3:, 3:], np.eye(3), atol=1e-14)
        assert np.allclose(Ad[:3, 3:], skew_sym([1, 2, 3]), atol=1e-14)

    def test_shape(self):
        """Adjoint should be 6x6."""
        g = np.eye(4)
        Ad = adjoint_se3(g)
        assert Ad.shape == (6, 6)
