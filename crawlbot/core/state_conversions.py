"""
State conversions between MuJoCo (world frame) and Pinocchio (structure frame).

All controller-internal quantities (r_com, v_com, L_com, Jacobians, etc.) are
expressed in the structure body frame. Conversion to/from MuJoCo world frame
happens only at these two functions.

Layout — RWA-3 model (nq=29, nv=27):
    qpos: [struct_pos(3) struct_quat(4) rw_angles(3)
           torso_pos(3)  torso_quat(4)  joints(12)]
    qvel: [struct_v(3) struct_omega(3) rw_vel(3)
           torso_v(3) torso_omega(3) joint_vel(12)]

Layout — original model (nq=26, nv=24):
    Same without the 3 rw_angles / rw_vel entries.
"""

import numpy as np
import pinocchio as pin


def mujoco_to_pinocchio(mj_qpos, mj_qvel):
    """Convert MuJoCo state to Pinocchio convention in structure frame.

    Returns
    -------
    pin_q : ndarray (19,)
        [torso_pos_struct(3), torso_quat_xyzw_struct(4), joints(12)]
    pin_v : ndarray (18,)
        [torso_vel_struct(3), torso_omega_struct(3), joint_vel(12)]
    """
    rwa = len(mj_qpos) >= 29
    off_q = 3 if rwa else 0
    off_v = 3 if rwa else 0

    # Structure pose / twist in world (qvel is world-frame for free joints)
    p_s = mj_qpos[0:3]
    qw_s, qx_s, qy_s, qz_s = mj_qpos[3:7]
    R_s = pin.Quaternion(qw_s, qx_s, qy_s, qz_s).toRotationMatrix()
    v_s = mj_qvel[0:3]
    omega_s = mj_qvel[3:6]

    # Torso pose / twist in world
    p_t = mj_qpos[7 + off_q : 10 + off_q]
    qw_t, qx_t, qy_t, qz_t = mj_qpos[10 + off_q : 14 + off_q]
    R_t = pin.Quaternion(qw_t, qx_t, qy_t, qz_t).toRotationMatrix()
    v_t = mj_qvel[6 + off_v : 9 + off_v]
    omega_t = mj_qvel[9 + off_v : 12 + off_v]

    # Transform torso to structure frame
    dp = p_t - p_s
    p_local = R_s.T @ dp
    R_local = R_s.T @ R_t
    v_local = R_s.T @ (v_t - v_s - np.cross(omega_s, dp))
    omega_local = R_s.T @ (omega_t - omega_s)

    # Pinocchio quaternion convention: xyzw
    q_local = pin.Quaternion(R_local)
    coeffs = q_local.coeffs()  # [x, y, z, w]

    pin_q = np.zeros(19)
    pin_v = np.zeros(18)
    pin_q[0:3] = p_local
    pin_q[3:7] = coeffs
    pin_q[7:19] = mj_qpos[14 + off_q : 26 + off_q]
    pin_v[0:3] = v_local
    pin_v[3:6] = omega_local
    pin_v[6:18] = mj_qvel[12 + off_v : 24 + off_v]
    return pin_q, pin_v


def pinocchio_to_mujoco(pin_q, pin_v, struct_pos=None, struct_quat=None,
                        rwa=False):
    """Convert Pinocchio state (structure frame) to MuJoCo convention (world frame).

    If rwa=True, produces the RWA-3 layout (nq=29, nv=27) with zero wheel
    angles/vels.
    """
    off_q = 3 if rwa else 0
    off_v = 3 if rwa else 0
    nq = 29 if rwa else 26
    nv = 27 if rwa else 24

    s_pos = np.asarray(struct_pos, dtype=float) if struct_pos is not None else np.zeros(3)
    s_quat = np.asarray(struct_quat, dtype=float) if struct_quat is not None else np.array([1., 0., 0., 0.])
    qw_s, qx_s, qy_s, qz_s = s_quat
    R_s = pin.Quaternion(qw_s, qx_s, qy_s, qz_s).toRotationMatrix()

    # Torso in structure frame → world frame
    p_local = pin_q[0:3]
    x, y, z, w_ = pin_q[3:7]                       # Pinocchio xyzw
    R_local = pin.Quaternion(w_, x, y, z).toRotationMatrix()

    p_world = s_pos + R_s @ p_local
    R_world = R_s @ R_local
    q_world = pin.Quaternion(R_world)
    cw = q_world.coeffs()                           # [x, y, z, w]

    mj_qpos = np.zeros(nq)
    mj_qvel = np.zeros(nv)
    mj_qpos[0:3] = s_pos
    mj_qpos[3:7] = s_quat
    mj_qpos[7 + off_q : 10 + off_q] = p_world
    mj_qpos[10 + off_q : 14 + off_q] = [cw[3], cw[0], cw[1], cw[2]]  # wxyz
    mj_qpos[14 + off_q : 26 + off_q] = pin_q[7:19]

    # Velocities: struct → world (assumes v_struct ≈ 0 at setup)
    mj_qvel[6 + off_v : 9 + off_v] = R_s @ pin_v[0:3]
    mj_qvel[9 + off_v : 12 + off_v] = R_s @ pin_v[3:6]
    mj_qvel[12 + off_v : 24 + off_v] = pin_v[6:18]
    return mj_qpos, mj_qvel


def quat_wxyz_to_euler_deg(qw, qx, qy, qz):
    """Quaternion (w,x,y,z) → Euler (roll, pitch, yaw) in degrees."""
    sinr = 2 * (qw * qx + qy * qz)
    cosr = 1 - 2 * (qx**2 + qy**2)
    roll = np.arctan2(sinr, cosr)
    sinp = np.clip(2 * (qw * qy - qz * qx), -1, 1)
    pitch = np.arcsin(sinp)
    siny = 2 * (qw * qz + qx * qy)
    cosy = 1 - 2 * (qy**2 + qz**2)
    yaw = np.arctan2(siny, cosy)
    return np.degrees(np.array([roll, pitch, yaw]))
