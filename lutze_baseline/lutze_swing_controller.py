"""Cartesian impedance controller for the swing arm.

During single-support phases, the free (swing) arm must track a trajectory
to the next anchor.  This module computes the additional joint torques via
Cartesian impedance:

    F_swing = Kp * (p_ref - p_ee) + Kd * (v_ref - v_ee)
    tau_swing = J_swing^T @ F_swing

Only the translational (3D) part is controlled; orientation is left free.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class SwingImpedanceConfig:
    """Gains for swing arm Cartesian impedance."""
    Kp: np.ndarray = None   # (3,) or (3,3) position stiffness [N/m]
    Kd: np.ndarray = None   # (3,) or (3,3) velocity damping [N·s/m]
    F_max: float = 15.0     # [N] max swing force magnitude

    def __post_init__(self):
        if self.Kp is None:
            self.Kp = np.diag([80.0, 80.0, 80.0])
        if self.Kd is None:
            self.Kd = np.diag([15.0, 15.0, 15.0])


def compute_swing_torques(
    p_ee: np.ndarray,
    v_ee: np.ndarray,
    p_ee_ref: np.ndarray,
    v_ee_ref: np.ndarray,
    J_ee: np.ndarray,
    cfg: SwingImpedanceConfig = None,
    tau_max: float = 10.0,
) -> np.ndarray:
    """Compute swing arm joint torques via Cartesian impedance.

    Parameters
    ----------
    p_ee : (3,) – Current end-effector position.
    v_ee : (3,) – Current end-effector velocity (J @ v).
    p_ee_ref : (3,) – Desired end-effector position.
    v_ee_ref : (3,) – Desired end-effector velocity.
    J_ee : (6, 18) – End-effector Jacobian (LOCAL_WORLD_ALIGNED).
    cfg : SwingImpedanceConfig – Gains.
    tau_max : float – Per-joint torque limit [Nm].

    Returns
    -------
    tau_joints : (12,) – Joint torques from swing control (actuated DOFs only).
    """
    if cfg is None:
        cfg = SwingImpedanceConfig()

    # Translational impedance
    e_pos = p_ee_ref - p_ee
    e_vel = v_ee_ref - v_ee

    F_swing = cfg.Kp @ e_pos + cfg.Kd @ e_vel

    # Clamp force
    f_norm = np.linalg.norm(F_swing)
    if f_norm > cfg.F_max:
        F_swing = F_swing / f_norm * cfg.F_max

    # Build 6D wrench (only translation, no angular control)
    W_swing = np.zeros(6)
    W_swing[3:] = F_swing  # force part (pinocchio convention: [ang; lin])

    # Map to generalized torques
    nv = J_ee.shape[1]  # 18
    tau_full = J_ee.T @ W_swing

    # Extract actuated joints
    tau_joints = tau_full[6:]
    tau_joints = np.clip(tau_joints, -tau_max, tau_max)

    return tau_joints
