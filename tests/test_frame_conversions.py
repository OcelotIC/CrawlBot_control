"""Tests for MuJoCo <-> Pinocchio frame conversion robustness.

Verifies round-trip consistency, velocity Coriolis terms, quaternion
convention handling, Euler angle extraction, RWA detection, and joint
passthrough for the state_conversions module.

Run with:
    PYTHONPATH=. MUJOCO_GL=disabled pytest tests/test_frame_conversions.py -v
"""

import numpy as np
import pinocchio as pin
import pytest

from crawlbot.core.state_conversions import (
    mujoco_to_pinocchio,
    pinocchio_to_mujoco,
    quat_wxyz_to_euler_deg,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _Rz(theta):
    """Rotation matrix about z-axis, angle in radians."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])


def _quat_wxyz_from_angle_z(theta):
    """MuJoCo wxyz quaternion for rotation of *theta* rad about z."""
    return np.array([np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)])


def _make_rwa_state(struct_pos, struct_quat_wxyz, torso_pos, torso_quat_wxyz,
                    joints, struct_v=None, struct_omega=None,
                    torso_v=None, torso_omega=None, joint_vel=None):
    """Build RWA-3 MuJoCo state arrays (nq=29, nv=27)."""
    qpos = np.zeros(29)
    qvel = np.zeros(27)

    qpos[0:3] = struct_pos
    qpos[3:7] = struct_quat_wxyz
    qpos[7:10] = 0.0                    # rw_angles
    qpos[10:13] = torso_pos
    qpos[13:17] = torso_quat_wxyz
    qpos[17:29] = joints

    if struct_v is not None:
        qvel[0:3] = struct_v
    if struct_omega is not None:
        qvel[3:6] = struct_omega
    # qvel[6:9] = rw_vel (zero)
    if torso_v is not None:
        qvel[9:12] = torso_v
    if torso_omega is not None:
        qvel[12:15] = torso_omega
    if joint_vel is not None:
        qvel[15:27] = joint_vel

    return qpos, qvel


# ===================================================================
# Round-trip tests
# ===================================================================

class TestRoundTrip:
    """mujoco_to_pinocchio -> pinocchio_to_mujoco should recover original."""

    def test_round_trip_identity(self):
        """Identity structure, known torso pose — perfect round-trip."""
        torso_pos = np.array([1.0, 2.0, 3.0])
        torso_quat = _quat_wxyz_from_angle_z(np.radians(25.0))
        joints = np.linspace(-0.3, 0.3, 12)
        joint_vel = np.linspace(-0.1, 0.1, 12)
        torso_v = np.array([0.5, -0.3, 0.1])
        torso_omega = np.array([0.01, -0.02, 0.03])

        qpos, qvel = _make_rwa_state(
            struct_pos=np.zeros(3),
            struct_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            torso_pos=torso_pos,
            torso_quat_wxyz=torso_quat,
            joints=joints,
            torso_v=torso_v,
            torso_omega=torso_omega,
            joint_vel=joint_vel,
        )

        pin_q, pin_v = mujoco_to_pinocchio(qpos, qvel)
        rec_qpos, rec_qvel = pinocchio_to_mujoco(
            pin_q, pin_v,
            struct_pos=qpos[0:3],
            struct_quat=qpos[3:7],
            rwa=True,
        )

        np.testing.assert_allclose(rec_qpos[10:13], torso_pos, atol=1e-10)
        np.testing.assert_allclose(rec_qpos[13:17], torso_quat, atol=1e-10)
        np.testing.assert_allclose(rec_qpos[17:29], joints, atol=1e-10)
        np.testing.assert_allclose(rec_qvel[9:12], torso_v, atol=1e-10)
        np.testing.assert_allclose(rec_qvel[12:15], torso_omega, atol=1e-10)
        np.testing.assert_allclose(rec_qvel[15:27], joint_vel, atol=1e-10)

    @pytest.mark.parametrize("angle_deg", [10, 20, 30])
    def test_round_trip_with_structure_rotation(self, angle_deg):
        """Structure rotated about z — torso still recovers after round-trip."""
        theta = np.radians(angle_deg)
        struct_quat = _quat_wxyz_from_angle_z(theta)
        struct_pos = np.array([0.5, -0.2, 0.1])

        # Torso offset in world frame
        torso_pos = struct_pos + np.array([1.0, 0.5, 0.0])
        torso_quat = _quat_wxyz_from_angle_z(theta + np.radians(15.0))
        joints = np.ones(12) * 0.1
        torso_v = np.array([0.2, 0.3, -0.1])
        torso_omega = np.array([0.0, 0.0, 0.05])
        joint_vel = np.ones(12) * 0.02

        qpos, qvel = _make_rwa_state(
            struct_pos=struct_pos,
            struct_quat_wxyz=struct_quat,
            torso_pos=torso_pos,
            torso_quat_wxyz=torso_quat,
            joints=joints,
            torso_v=torso_v,
            torso_omega=torso_omega,
            joint_vel=joint_vel,
        )

        pin_q, pin_v = mujoco_to_pinocchio(qpos, qvel)
        rec_qpos, rec_qvel = pinocchio_to_mujoco(
            pin_q, pin_v,
            struct_pos=struct_pos,
            struct_quat=struct_quat,
            rwa=True,
        )

        np.testing.assert_allclose(rec_qpos[10:13], torso_pos, atol=1e-10)
        # Quaternion sign ambiguity: compare rotation matrices
        qw, qx, qy, qz = rec_qpos[13:17]
        R_rec = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
        qw0, qx0, qy0, qz0 = torso_quat
        R_orig = pin.Quaternion(qw0, qx0, qy0, qz0).toRotationMatrix()
        np.testing.assert_allclose(R_rec, R_orig, atol=1e-10)
        np.testing.assert_allclose(rec_qpos[17:29], joints, atol=1e-10)
        np.testing.assert_allclose(rec_qvel[9:12], torso_v, atol=1e-10)
        np.testing.assert_allclose(rec_qvel[12:15], torso_omega, atol=1e-10)
        np.testing.assert_allclose(rec_qvel[15:27], joint_vel, atol=1e-10)


# ===================================================================
# Velocity / Coriolis
# ===================================================================

class TestVelocityCoriolis:
    """Verify the Coriolis cross-product term in velocity transformation."""

    def test_velocity_coriolis_term(self):
        """Non-zero structure omega produces correct Coriolis contribution."""
        theta = np.radians(15.0)
        struct_pos = np.zeros(3)
        struct_quat = _quat_wxyz_from_angle_z(theta)
        R_s = _Rz(theta)

        omega_s = np.array([0.0, 0.0, 0.1])   # spinning about z
        torso_pos = np.array([2.0, 0.0, 0.0])  # offset along x
        dp = torso_pos - struct_pos

        v_s = np.array([0.0, 0.0, 0.0])
        v_t = np.array([0.3, -0.2, 0.1])
        omega_t = np.array([0.0, 0.0, 0.05])

        joints = np.zeros(12)
        joint_vel = np.zeros(12)

        qpos, qvel = _make_rwa_state(
            struct_pos=struct_pos,
            struct_quat_wxyz=struct_quat,
            torso_pos=torso_pos,
            torso_quat_wxyz=_quat_wxyz_from_angle_z(0.0),
            joints=joints,
            struct_v=v_s,
            struct_omega=omega_s,
            torso_v=v_t,
            torso_omega=omega_t,
            joint_vel=joint_vel,
        )

        pin_q, pin_v = mujoco_to_pinocchio(qpos, qvel)

        # Manual expected values
        expected_v_local = R_s.T @ (v_t - v_s - np.cross(omega_s, dp))
        expected_omega_local = R_s.T @ (omega_t - omega_s)

        np.testing.assert_allclose(pin_v[0:3], expected_v_local, atol=1e-12)
        np.testing.assert_allclose(pin_v[3:6], expected_omega_local, atol=1e-12)


# ===================================================================
# Quaternion conventions
# ===================================================================

class TestQuaternionConvention:
    """MuJoCo wxyz <-> Pinocchio xyzw."""

    def test_quaternion_convention_mujoco_pinocchio(self):
        """45-deg z-rotation: wxyz -> xyzw ordering is correct."""
        angle = np.radians(45.0)
        c = np.cos(angle / 2)
        s = np.sin(angle / 2)

        # MuJoCo wxyz
        mj_quat = np.array([c, 0.0, 0.0, s])

        qpos, qvel = _make_rwa_state(
            struct_pos=np.zeros(3),
            struct_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            torso_pos=np.array([1.0, 0.0, 0.0]),
            torso_quat_wxyz=mj_quat,
            joints=np.zeros(12),
        )

        pin_q, _ = mujoco_to_pinocchio(qpos, qvel)

        # Pinocchio xyzw
        expected_xyzw = np.array([0.0, 0.0, s, c])
        np.testing.assert_allclose(pin_q[3:7], expected_xyzw, atol=1e-14)


# ===================================================================
# Euler angle extraction
# ===================================================================

class TestQuatToEuler:
    """quat_wxyz_to_euler_deg edge cases."""

    def test_quat_wxyz_to_euler_deg_identity(self):
        """Identity quaternion -> [0, 0, 0] degrees."""
        result = quat_wxyz_to_euler_deg(1.0, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-14)

    def test_quat_wxyz_to_euler_deg_90_yaw(self):
        """90-deg yaw -> [0, 0, 90] degrees."""
        angle = np.radians(90.0)
        qw, qx, qy, qz = np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)
        result = quat_wxyz_to_euler_deg(qw, qx, qy, qz)
        np.testing.assert_allclose(result, [0.0, 0.0, 90.0], atol=1e-10)

    def test_quat_wxyz_to_euler_deg_gimbal_lock(self):
        """Pure 90-deg pitch (gimbal lock) — no NaN, pitch ~ 90 deg."""
        angle = np.radians(90.0)
        qw = np.cos(angle / 2)
        qy = np.sin(angle / 2)
        result = quat_wxyz_to_euler_deg(qw, 0.0, qy, 0.0)
        assert not np.any(np.isnan(result)), "NaN in gimbal-lock output"
        np.testing.assert_allclose(result[1], 90.0, atol=1e-10)


# ===================================================================
# RWA detection
# ===================================================================

class TestRWADetection:
    """mujoco_to_pinocchio should handle both RWA and non-RWA layouts."""

    def test_rwa_detection(self):
        """RWA (nq=29) vs non-RWA (nq=26) produce same Pinocchio output
        when structure and torso states are equivalent."""
        struct_pos = np.zeros(3)
        struct_quat = np.array([1.0, 0.0, 0.0, 0.0])
        torso_pos = np.array([1.0, 0.5, 0.0])
        torso_quat = _quat_wxyz_from_angle_z(np.radians(10.0))
        joints = np.linspace(0.0, 0.2, 12)
        joint_vel = np.linspace(-0.05, 0.05, 12)

        # --- RWA layout (nq=29, nv=27) ---
        rwa_qpos = np.zeros(29)
        rwa_qpos[0:3] = struct_pos
        rwa_qpos[3:7] = struct_quat
        rwa_qpos[7:10] = 0.0           # rw_angles
        rwa_qpos[10:13] = torso_pos
        rwa_qpos[13:17] = torso_quat
        rwa_qpos[17:29] = joints

        rwa_qvel = np.zeros(27)
        rwa_qvel[15:27] = joint_vel

        # --- non-RWA layout (nq=26, nv=24) ---
        norwa_qpos = np.zeros(26)
        norwa_qpos[0:3] = struct_pos
        norwa_qpos[3:7] = struct_quat
        norwa_qpos[7:10] = torso_pos
        norwa_qpos[10:14] = torso_quat
        norwa_qpos[14:26] = joints

        norwa_qvel = np.zeros(24)
        norwa_qvel[12:24] = joint_vel

        pin_q_rwa, pin_v_rwa = mujoco_to_pinocchio(rwa_qpos, rwa_qvel)
        pin_q_no, pin_v_no = mujoco_to_pinocchio(norwa_qpos, norwa_qvel)

        np.testing.assert_allclose(pin_q_rwa, pin_q_no, atol=1e-14)
        np.testing.assert_allclose(pin_v_rwa, pin_v_no, atol=1e-14)


# ===================================================================
# Joint passthrough
# ===================================================================

class TestJointPassthrough:
    """Joint positions / velocities should pass through unchanged."""

    def test_joint_passthrough(self):
        """Joints are identical in MuJoCo and Pinocchio representations."""
        joints = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.5,
                           0.4, -0.3, 0.2, -0.1, 0.05, -0.05])
        joint_vel = np.array([1.0, -1.0, 0.5, -0.5, 0.3, -0.3,
                              0.2, -0.2, 0.1, -0.1, 0.01, -0.01])

        qpos, qvel = _make_rwa_state(
            struct_pos=np.array([0.5, 0.5, 0.5]),
            struct_quat_wxyz=_quat_wxyz_from_angle_z(np.radians(42.0)),
            torso_pos=np.array([1.5, 1.0, 0.5]),
            torso_quat_wxyz=_quat_wxyz_from_angle_z(np.radians(17.0)),
            joints=joints,
            joint_vel=joint_vel,
        )

        pin_q, pin_v = mujoco_to_pinocchio(qpos, qvel)

        np.testing.assert_allclose(pin_q[7:19], joints, atol=1e-14)
        np.testing.assert_allclose(pin_v[6:18], joint_vel, atol=1e-14)
