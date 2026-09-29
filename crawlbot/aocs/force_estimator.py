"""
The AOCS law: reaction-wheel torque for the free-floating platform.

One law is kept — the one the paper uses, ``legacy_pid_numerical``:

    τ_w = ff + K_hw·(clip(h_w) − h_w) + K_θ·θ_s + K_ω·ω_s + K_d·ω̇_s,num
    ω̇_s,num = (ω_s − ω_s,prev)/dt
    ff = −L̇_com − r_com × m·v̇_com          (FD feedforward, SS)
    ff = tau_struct_ff = −Σ_i (r_Ci×f_i + τ_i) from λ_qp  (wrench FF, DS)

clipped to ±τ_w,max. Its variants (legacy, legacy_corrected, legacy_pd_*,
legacy_pid_model), the H_{r/O} estimator law (H_est) and the
MomentumDisturbanceEstimator were never run by the canonical nor by the paper's
Table 2 configurations and were retired (refactor/sim-loop-split R1; history
in git, memo results/j2_adjconv/PHASE_R1_AOCS_MODES_RETIRED.md).
"""

import numpy as np


def compute_aocs_command_legacy_pid_numerical(
    L_com: np.ndarray,
    L_com_prev: np.ndarray,
    r_com: np.ndarray,
    v_com: np.ndarray,
    v_com_prev: np.ndarray,
    omega_s: np.ndarray,
    omega_s_prev: np.ndarray,
    theta_s: np.ndarray,
    hw_current: np.ndarray,
    dt: float,
    robot_mass: float,
    K_hw: float = 2.0,
    K_omega: float = 50.0,
    K_d: float = 25.0,
    K_theta: float = 1.0,
    hw_min: np.ndarray = None,
    hw_max: np.ndarray = None,
    tau_w_max: float = 2.5,
    tau_struct_ff: np.ndarray = None,
) -> np.ndarray:
    """Legacy-corrected AOCS + PID on attitude (numerical ω̇_s).

    Extends ``compute_aocs_command_legacy_pd_numerical`` with an
    attitude-tracking P term that drives the structure back to its
    reference orientation (θ_s = 0), recovering the irreversible net
    per-traversal rotation the PD-only mode cannot undo.

        τ_w = -Ḣ_s_est + K_hw·hw_error
              + K_θ·θ_s + K_ω·ω_s + K_d·ω̇_s_num

    Sign on K_θ is positive (same derivation as K_ω, K_d): Newton-Euler
    about structure CoM with τ_w on wheels giving -τ_w reaction on the
    structure. For θ_s > 0 to decrease, need negative angular
    acceleration ⇒ τ_w > -Ḣ_s ⇒ K_θ contribution adds positive.

    Parameters
    ----------
    theta_s : (3,) structure attitude error vector [rad], in body frame.
        Computed in sim_loop as log3(R_init.T @ R_now). Small-angle
        approximation: θ_s ≈ axis × angle of the rotation from reference
        to current.
    K_theta : float, attitude-error gain [Nm/rad]. Default 1.0 sized
        for slow rotate-back (~60 s time constant) with the existing
        K_ω = 50: ζ ≈ K_ω / (2·sqrt(K_θ·I_s)) for the SISO PD.
    Other parameters: see ``compute_aocs_command_legacy_pd_numerical``.

    Notes
    -----
    The attitude term is bounded by wheel **momentum** capacity, not
    torque capacity: rotating the structure back by Δθ requires the
    wheels to transiently carry |h_w| = I_s·ω_max ≤ h_w_max, capping
    the rotate-back speed at ω_max = h_w_max/I_s. For h_w_max = 5 Nms
    and I_s ≈ 1500 kg·m², that's ~3.3 mrad/s — slow but feasible for
    typical per-traversal rotation (~2° = 35 mrad → ~10 s minimum).
    """
    if hw_min is None:
        hw_min = -np.full(3, np.inf)
    if hw_max is None:
        hw_max = np.full(3, np.inf)

    hw_error = np.clip(hw_current, hw_min, hw_max) - hw_current
    omega_dot_est = (omega_s - omega_s_prev) / dt
    pid_term = K_theta * theta_s + K_omega * omega_s + K_d * omega_dot_est

    if tau_struct_ff is None:
        # Kinematic feedforward via FD on robot centroidal momentum.
        # Complete only when the robot is kinematically free at the
        # contact (SS). In DS the welded loop carries internal stress
        # that contributes a couple (r_CA−r_CB)×f on the structure
        # invisible to L_com — see compute_aocs_command_wrench_ff below.
        L_dot_est = (L_com - L_com_prev) / dt
        dv_com_est = (v_com - v_com_prev) / dt
        orbital = np.cross(r_com, robot_mass * dv_com_est)
        ff_term = -L_dot_est - orbital
    else:
        # Direct wrench feedforward: τ_w = −Σ_i (r_Ci × f_i + τ_i),
        # already signed and summed by the caller from λ_qp.
        ff_term = tau_struct_ff

    tau_w = ff_term + K_hw * hw_error + pid_term
    return np.clip(tau_w, -tau_w_max, tau_w_max)
