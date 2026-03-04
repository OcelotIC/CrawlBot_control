"""Feedforward wrench computation (Lutze et al. 2023, eq. 12-16).

Computes desired wrenches for robot tracking and structure stabilization.
These feed into the QP as reference targets.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class LutzeFeedforwardConfig:
    """Gains for the feedforward wrench computation."""
    # Robot tracking (Cartesian PD on CoM)
    Kr: np.ndarray = None   # (3,) or (3,3) position stiffness [N/m]
    Dr: np.ndarray = None   # (3,) or (3,3) velocity damping [N·s/m]
    # Structure attitude stabilization
    Kb: np.ndarray = None   # (3,) or (3,3) angular stiffness [Nm/rad]
    Db: np.ndarray = None   # (3,) or (3,3) angular damping [Nm·s/rad]
    # Saturation
    e_pos_max: float = 0.5  # [m] position error clamp
    e_vel_max: float = 0.5  # [m/s] velocity error clamp
    F_max: float = 25.0     # [N] force magnitude clamp

    def __post_init__(self):
        if self.Kr is None:
            self.Kr = np.diag([50.0, 50.0, 50.0])
        if self.Dr is None:
            self.Dr = np.diag([10.0, 10.0, 10.0])
        if self.Kb is None:
            self.Kb = np.diag([0.0, 0.0, 0.0])  # off by default (see roadmap)
        if self.Db is None:
            self.Db = np.diag([0.0, 0.0, 0.0])


def compute_feedforward(
    r_com: np.ndarray,
    v_com: np.ndarray,
    r_com_ref: np.ndarray,
    v_com_ref: np.ndarray,
    struct_quat_wxyz: np.ndarray = None,
    struct_omega: np.ndarray = None,
    cfg: LutzeFeedforwardConfig = None,
):
    """Compute desired wrenches for the Lutze QP.

    Parameters
    ----------
    r_com : (3,) – Current robot CoM position.
    v_com : (3,) – Current robot CoM velocity.
    r_com_ref : (3,) – Desired CoM position.
    v_com_ref : (3,) – Desired CoM velocity.
    struct_quat_wxyz : (4,) – Structure quaternion [w,x,y,z] (MuJoCo convention).
        None → identity (no attitude error).
    struct_omega : (3,) – Structure angular velocity [rad/s].
        None → zeros.
    cfg : LutzeFeedforwardConfig – Gains.

    Returns
    -------
    F_d_r : (6,) – Desired robot wrench [torque(3); force(3)] in world frame.
    F_d_b : (6,) – Desired structure stabilization wrench [torque(3); force(3)].
    """
    if cfg is None:
        cfg = LutzeFeedforwardConfig()

    # --- Robot tracking wrench (eq. 12-14) ---
    e_pos = r_com_ref - r_com
    e_vel = v_com_ref - v_com

    # Clamp errors
    e_pos_norm = np.linalg.norm(e_pos)
    if e_pos_norm > cfg.e_pos_max:
        e_pos = e_pos / e_pos_norm * cfg.e_pos_max
    e_vel_norm = np.linalg.norm(e_vel)
    if e_vel_norm > cfg.e_vel_max:
        e_vel = e_vel / e_vel_norm * cfg.e_vel_max

    f_track = cfg.Kr @ e_pos + cfg.Dr @ e_vel

    # Clamp force
    f_norm = np.linalg.norm(f_track)
    if f_norm > cfg.F_max:
        f_track = f_track / f_norm * cfg.F_max

    # Wrench: [torque(3); force(3)] — no angular tracking for CoM
    F_d_r = np.zeros(6)
    F_d_r[3:] = f_track

    # --- Structure stabilization wrench (eq. 16) ---
    F_d_b = np.zeros(6)
    if struct_quat_wxyz is not None and struct_omega is not None:
        # Small-angle attitude error from quaternion vector part
        # quat = [w, x, y, z]; error ≈ 2 * [x, y, z] for small angles
        e_att = 2.0 * struct_quat_wxyz[1:4]  # (3,)
        tau_stab = cfg.Kb @ e_att + cfg.Db @ (-struct_omega)
        F_d_b[:3] = tau_stab

    return F_d_r, F_d_b
