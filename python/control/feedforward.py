"""Feedforward wrench computation.

Translated from compute_feedforward_wrench.m.
Computes desired wrenches for robot and base (eq. 12-16).
"""

import numpy as np
import warnings


def compute_feedforward_wrench(sys, traj_des, t_idx, gains):
    """Compute desired wrenches for robot tracking and base stabilization.

    Parameters
    ----------
    sys : dict – System state.
    traj_des : dict – Desired trajectory.
    t_idx : int – Current time index (0-based).
    gains : dict – Control gains (Kr, Dr, Kb, Db).

    Returns
    -------
    F_d_r : (6,) – Desired robot wrench.
    F_d_b : (6,) – Desired base wrench.
    """
    # Validate inputs
    if np.any(np.isnan(sys['r0'])) or np.any(np.isinf(sys['r0'])):
        warnings.warn("[Feedforward] sys.r0 invalid")
        return np.zeros(6), np.zeros(6)

    # --- Robot wrench (eq. 12-14) ---
    r_des = traj_des['pos'][:, t_idx]
    v_des = traj_des['vel'][:, t_idx]

    r_cur = sys['r0']
    v_cur = sys['t0'][3:6]  # Linear velocity from twist

    # Errors
    e_pos = r_des - r_cur
    e_vel = v_des - v_cur

    # Saturation
    e_pos_max = 10.0
    e_vel_max = 5.0

    e_pos_norm = np.linalg.norm(e_pos)
    if e_pos_norm > e_pos_max:
        warnings.warn(f"[Feedforward] Position error excessive: {e_pos_norm:.2f} m")
        e_pos = e_pos / e_pos_norm * e_pos_max

    e_vel_norm = np.linalg.norm(e_vel)
    if e_vel_norm > e_vel_max:
        e_vel = e_vel / e_vel_norm * e_vel_max

    # Feedback (translation only, using lower-right 3x3 of Kr, Dr)
    Kr_trans = gains['Kr'][3:6, 3:6]
    Dr_trans = gains['Dr'][3:6, 3:6]
    F_fb_trans = Kr_trans @ e_pos + Dr_trans @ e_vel
    F_fb_rot = np.zeros(3)
    F_d_r = np.concatenate([F_fb_rot, F_fb_trans])

    # Saturation
    F_d_r_max = 1000.0
    if np.linalg.norm(F_d_r) > F_d_r_max:
        F_d_r = F_d_r / np.linalg.norm(F_d_r) * F_d_r_max

    # --- Base/satellite stabilization wrench (eq. 16) ---
    # Use satellite state (not robot base) for stabilization feedback
    if 'quat_satellite' in sys:
        omega_base = sys.get('omega_satellite', np.zeros(3))
        e_quat = 2.0 * sys['quat_satellite'][:3]
    else:
        omega_base = sys['t0'][:3]
        e_quat = 2.0 * sys.get('quat_base', np.array([0, 0, 0, 1.0]))[:3]

    # Saturation
    e_quat_norm = np.linalg.norm(e_quat)
    if e_quat_norm > 1.0:
        e_quat = e_quat / e_quat_norm

    F_d_b_rot = gains['Kb'] @ e_quat + gains['Db'] @ (-omega_base)
    F_d_b = np.concatenate([F_d_b_rot, np.zeros(3)])

    # Saturation
    F_d_b_max = 500.0
    if np.linalg.norm(F_d_b) > F_d_b_max:
        F_d_b = F_d_b / np.linalg.norm(F_d_b) * F_d_b_max

    return F_d_r, F_d_b
