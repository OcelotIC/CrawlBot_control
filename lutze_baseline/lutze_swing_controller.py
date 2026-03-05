"""Cartesian impedance controller for the swing arm.

During single-support phases, the free (swing) arm must track a trajectory
to the next anchor.  This module computes the additional joint torques via
Cartesian impedance control: tau = J^T @ [0; Kp*e + Kd*e_dot].
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SwingImpedanceConfig:
    """Gains for swing arm Cartesian impedance."""
    Kp: np.ndarray = field(default_factory=lambda: np.diag([200.0, 200.0, 200.0]))
    Kd: np.ndarray = field(default_factory=lambda: np.diag([20.0, 20.0, 20.0]))

    def __post_init__(self):
        for attr in ['Kp', 'Kd']:
            val = getattr(self, attr)
            if not isinstance(val, np.ndarray):
                setattr(self, attr, np.asarray(val, dtype=float))


def compute_swing_torques(p_ee, v_ee, p_ee_ref, v_ee_ref, J_ee, cfg=None,
                          tau_max=50.0):
    """Compute swing arm joint torques via Cartesian impedance.

    Parameters
    ----------
    p_ee : (3,) - Current end-effector position.
    v_ee : (3,) - Current end-effector velocity (J @ v).
    p_ee_ref : (3,) - Desired end-effector position.
    v_ee_ref : (3,) - Desired end-effector velocity.
    J_ee : (6, nv) - End-effector Jacobian (world frame).
    cfg : SwingImpedanceConfig or None.
    tau_max : float - Joint torque saturation (Nm).

    Returns
    -------
    tau : (nv,) joint torques for the swing arm.
    """
    if cfg is None:
        cfg = SwingImpedanceConfig()

    e_pos = p_ee_ref - p_ee
    e_vel = v_ee_ref - v_ee

    # Cartesian force (translation only)
    f_cart = cfg.Kp @ e_pos + cfg.Kd @ e_vel

    # Spatial wrench [torque(3); force(3)] — zero torque, position force only
    F_cart = np.concatenate([np.zeros(3), f_cart])

    # Map to joint torques
    tau = J_ee.T @ F_cart

    # Saturate
    tau = np.clip(tau, -tau_max, tau_max)

    return tau
