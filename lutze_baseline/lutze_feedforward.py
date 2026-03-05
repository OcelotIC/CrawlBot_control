"""Feedforward wrench computation (Lutze et al. 2023, eq. 12-16).

Computes desired wrenches for robot tracking and structure stabilization.
These feed into the QP as reference targets.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class LutzeFeedforwardConfig:
    """Gains for the feedforward wrench computation."""
    Kr_trans: np.ndarray = field(default_factory=lambda: np.diag([25.0, 25.0, 25.0]))
    Dr_trans: np.ndarray = field(default_factory=lambda: np.diag([5.0, 5.0, 5.0]))
    Kb: np.ndarray = field(default_factory=lambda: np.diag([40.0, 40.0, 40.0]))
    Db: np.ndarray = field(default_factory=lambda: np.diag([8.0, 8.0, 8.0]))
    pos_err_max: float = 10.0
    vel_err_max: float = 5.0
    quat_err_max: float = 1.0
    F_d_r_max: float = 1000.0
    F_d_b_max: float = 500.0

    def __post_init__(self):
        for attr in ['Kr_trans', 'Dr_trans', 'Kb', 'Db']:
            val = getattr(self, attr)
            if not isinstance(val, np.ndarray):
                setattr(self, attr, np.asarray(val))


def compute_feedforward(r_com, v_com, r_com_ref, v_com_ref,
                        struct_quat_wxyz=None, struct_omega=None,
                        cfg=None):
    """Compute desired wrenches for the Lutze QP.

    Parameters
    ----------
    r_com : (3,) - Current robot CoM position.
    v_com : (3,) - Current robot CoM velocity.
    r_com_ref : (3,) - Desired CoM position.
    v_com_ref : (3,) - Desired CoM velocity.
    struct_quat_wxyz : (4,) or None - Structure quaternion [w, x, y, z].
    struct_omega : (3,) or None - Structure angular velocity.
    cfg : LutzeFeedforwardConfig or None.

    Returns
    -------
    F_d_r : (6,) desired robot tracking wrench [torque(3); force(3)].
    F_d_b : (6,) desired structure stabilization wrench [torque(3); force(3)].
    """
    if cfg is None:
        cfg = LutzeFeedforwardConfig()

    # --- Robot tracking (eq. 12-14) ---
    e_pos = r_com_ref - r_com
    e_vel = v_com_ref - v_com

    # Saturate
    e_pos_norm = np.linalg.norm(e_pos)
    if e_pos_norm > cfg.pos_err_max:
        e_pos = e_pos / e_pos_norm * cfg.pos_err_max
    e_vel_norm = np.linalg.norm(e_vel)
    if e_vel_norm > cfg.vel_err_max:
        e_vel = e_vel / e_vel_norm * cfg.vel_err_max

    f_trans = cfg.Kr_trans @ e_pos + cfg.Dr_trans @ e_vel
    F_d_r = np.concatenate([np.zeros(3), f_trans])

    # Saturate total
    F_d_r_norm = np.linalg.norm(F_d_r)
    if F_d_r_norm > cfg.F_d_r_max:
        F_d_r = F_d_r / F_d_r_norm * cfg.F_d_r_max

    # --- Structure stabilization (eq. 16) ---
    F_d_b = np.zeros(6)
    if struct_quat_wxyz is not None and struct_omega is not None:
        # Quaternion error: 2 * vector part (small-angle approx)
        e_quat = 2.0 * struct_quat_wxyz[1:4]
        e_quat_norm = np.linalg.norm(e_quat)
        if e_quat_norm > cfg.quat_err_max:
            e_quat = e_quat / e_quat_norm * cfg.quat_err_max

        tau_b = cfg.Kb @ e_quat + cfg.Db @ (-struct_omega)
        F_d_b[:3] = tau_b

        F_d_b_norm = np.linalg.norm(F_d_b)
        if F_d_b_norm > cfg.F_d_b_max:
            F_d_b = F_d_b / F_d_b_norm * cfg.F_d_b_max

    return F_d_r, F_d_b
