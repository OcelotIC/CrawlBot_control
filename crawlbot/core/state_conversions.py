"""
State conversions between MuJoCo (world frame) and Pinocchio (structure frame).

All controller-internal quantities (r_com, v_com, L_com, Jacobians, etc.) are
expressed in the structure body frame. Conversion to/from MuJoCo world frame
happens only at these two functions.

DOF-generic: the number of arm joints (N) is inferred from the state
vector length. Supports 6-DOF (N=12) and 7-DOF (N=14) arm configurations.

Layout — RWA-3 model (nq=14+N+15, nv=12+N+15):
    qpos: [struct_pos(3) struct_quat(4) rw_angles(3)
           torso_pos(3)  torso_quat(4)  joints(N)]
    qvel: [struct_v(3) struct_omega(3) rw_vel(3)
           torso_v(3) torso_omega(3) joint_vel(N)]

Layout — original model (nq=14+N, nv=12+N):
    Same without the 3 rw_angles / rw_vel entries.
"""

import numpy as np
import pinocchio as pin

# ── Constants for the MuJoCo layout ─────────────────────────
# Structure (free joint): 7 qpos (pos+quat), 6 qvel
# RWA (optional): 3 qpos (angles), 3 qvel
# Torso (free joint): 7 qpos (pos+quat), 6 qvel
# Arm joints: N qpos, N qvel
_MJ_STRUCT_NQ = 7
_MJ_STRUCT_NV = 6
_MJ_RWA_NQ = 3
_MJ_RWA_NV = 3
_MJ_TORSO_NQ = 7
_MJ_TORSO_NV = 6


def _infer_n_joints(mj_qpos, rwa):
    """Infer number of arm joints from MuJoCo state vector length."""
    off_q = _MJ_RWA_NQ if rwa else 0
    return len(mj_qpos) - _MJ_STRUCT_NQ - off_q - _MJ_TORSO_NQ


def mujoco_to_pinocchio(mj_qpos, mj_qvel):
    """Convert MuJoCo state to Pinocchio convention in structure frame.

    Infers the number of arm joints from the state vector length.

    Returns
    -------
    pin_q : ndarray (7+N,)
        [torso_pos_struct(3), torso_quat_xyzw_struct(4), joints(N)]
    pin_v : ndarray (6+N,)
        [torso_vel_struct(3), torso_omega_struct(3), joint_vel(N)]
    """
    rwa = len(mj_qpos) >= 29  # at least 29 for 6-DOF arms with RWA
    off_q = _MJ_RWA_NQ if rwa else 0
    off_v = _MJ_RWA_NV if rwa else 0
    n_joints = _infer_n_joints(mj_qpos, rwa)

    # Structure pose / twist in world (qvel is world-frame for free joints)
    p_s = mj_qpos[0:3]
    qw_s, qx_s, qy_s, qz_s = mj_qpos[3:7]
    R_s = pin.Quaternion(qw_s, qx_s, qy_s, qz_s).toRotationMatrix()
    v_s = mj_qvel[0:3]
    omega_s = mj_qvel[3:6]

    # Torso pose / twist in world
    torso_q_start = _MJ_STRUCT_NQ + off_q
    p_t = mj_qpos[torso_q_start: torso_q_start + 3]
    qw_t, qx_t, qy_t, qz_t = mj_qpos[torso_q_start + 3: torso_q_start + 7]
    R_t = pin.Quaternion(qw_t, qx_t, qy_t, qz_t).toRotationMatrix()
    torso_v_start = _MJ_STRUCT_NV + off_v
    v_t = mj_qvel[torso_v_start: torso_v_start + 3]
    omega_t = mj_qvel[torso_v_start + 3: torso_v_start + 6]

    # Joint states in MuJoCo
    joints_q_start = torso_q_start + _MJ_TORSO_NQ
    joints_v_start = torso_v_start + _MJ_TORSO_NV
    mj_joints_q = mj_qpos[joints_q_start: joints_q_start + n_joints]
    mj_joints_v = mj_qvel[joints_v_start: joints_v_start + n_joints]

    # Transform torso to structure frame
    dp = p_t - p_s
    p_local = R_s.T @ dp
    R_local = R_s.T @ R_t
    v_local = R_s.T @ (v_t - v_s - np.cross(omega_s, dp))
    omega_local = R_s.T @ (omega_t - omega_s)

    # Pinocchio quaternion convention: xyzw
    q_local = pin.Quaternion(R_local)
    coeffs = q_local.coeffs()  # [x, y, z, w]

    pin_q = np.zeros(7 + n_joints)
    pin_v = np.zeros(6 + n_joints)
    pin_q[0:3] = p_local
    pin_q[3:7] = coeffs
    pin_q[7:7 + n_joints] = mj_joints_q
    pin_v[0:3] = v_local
    pin_v[3:6] = omega_local
    pin_v[6:6 + n_joints] = mj_joints_v
    return pin_q, pin_v


def pinocchio_to_mujoco(pin_q, pin_v, struct_pos=None, struct_quat=None,
                        rwa=False):
    """Convert Pinocchio state (structure frame) to MuJoCo convention (world frame).

    DOF-generic: infers joint count from pin_q length (7 + N_joints).
    If rwa=True, adds 3 zero entries for RWA angles/velocities.
    """
    n_joints = len(pin_q) - 7  # pin_q = [torso(7), joints(N)]
    off_q = _MJ_RWA_NQ if rwa else 0
    off_v = _MJ_RWA_NV if rwa else 0
    nq = _MJ_STRUCT_NQ + off_q + _MJ_TORSO_NQ + n_joints
    nv = _MJ_STRUCT_NV + off_v + _MJ_TORSO_NV + n_joints

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
    torso_q = _MJ_STRUCT_NQ + off_q
    mj_qpos[torso_q: torso_q + 3] = p_world
    mj_qpos[torso_q + 3: torso_q + 7] = [cw[3], cw[0], cw[1], cw[2]]  # wxyz
    joints_q = torso_q + _MJ_TORSO_NQ
    mj_qpos[joints_q: joints_q + n_joints] = pin_q[7:7 + n_joints]

    # Velocities: struct → world (assumes v_struct ≈ 0 at setup)
    torso_v = _MJ_STRUCT_NV + off_v
    mj_qvel[torso_v: torso_v + 3] = R_s @ pin_v[0:3]
    mj_qvel[torso_v + 3: torso_v + 6] = R_s @ pin_v[3:6]
    joints_v = torso_v + _MJ_TORSO_NV
    mj_qvel[joints_v: joints_v + n_joints] = pin_v[6:6 + n_joints]
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
