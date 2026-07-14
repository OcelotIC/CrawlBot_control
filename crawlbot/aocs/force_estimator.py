"""
MomentumDisturbanceEstimator — Estimate the disturbance torque applied by
the robot to the spacecraft structure, for AOCS feedforward.

Theory (see docs/force_estimator_note.md):

    The total angular momentum of the robot about O (structure CoM) is:

        H_{r/O} = L_com + r_com × (m_r · v_com)

    Its inertial time derivative gives the disturbance torque on the structure:

        τ_dist = -dH/dt|_inertial
               = -(dH/dt|_struct + ω_s × H_{r/O})

    The AOCS must reject this disturbance via wheel torques.

Two estimator variants are provided:

    Variant A — Finite-difference on H_{r/O} (recommended):
        Ḣ_fd = (H_k - H_{k-1}) / dt
        Ḣ_inertial = Ḣ_fd + ω_s × H_k

    Variant B — Analytical using a_com (requires acceleration):
        Ḣ_analytical = L̇_com + r_com × (m_r · a_com)

Both variants support pre-derivative EMA filtering to reduce noise.

Usage:
    estimator = MomentumDisturbanceEstimator(robot_mass=71.0, dt=0.01)
    ...
    # In QP inner loop, after Pinocchio update:
    H_dot = estimator.update(r_com, v_com, L_com, omega_s)
    tau_w_ff = -H_dot  # feedforward for wheels
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class EstimatorConfig:
    """Configuration for MomentumDisturbanceEstimator.

    Parameters
    ----------
    robot_mass : float
        Total mass of the robot (not including structure or wheels) [kg].
    dt : float
        Timestep of the estimator (QP inner loop rate) [s].
    filter_tau : float
        EMA filter time constant [s]. Applied to H_{r/O} before
        differentiation. Set to 0.0 to disable filtering.
        Default: 0.016 s (~10 Hz cutoff at 100 Hz sampling).
    include_transport : bool
        If True, include the ω_s × H term (inertial derivative).
        Should be True for production, can be False for debugging.
    """
    robot_mass: float = 71.0
    dt: float = 0.01
    filter_tau: float = 0.016
    include_transport: bool = True


class MomentumDisturbanceEstimator:
    """Estimate dH_{r/O}/dt for AOCS feedforward.

    All inputs and outputs are in the structure body frame R_s.
    The origin of R_s must coincide with O (structure CoM).

    Parameters
    ----------
    config : EstimatorConfig, optional
        Estimator configuration.
    robot_mass : float, optional
        Shortcut: if provided (and config is None), creates a default
        config with this mass.
    dt : float, optional
        Shortcut: if provided (and config is None), creates a default
        config with this timestep.
    """

    def __init__(self, config: Optional[EstimatorConfig] = None,
                 robot_mass: float = None, dt: float = None):
        if config is None:
            config = EstimatorConfig()
        if robot_mass is not None:
            config.robot_mass = robot_mass
        if dt is not None:
            config.dt = dt

        self.cfg = config
        self._m = config.robot_mass
        self._dt = config.dt

        # EMA filter coefficient
        if config.filter_tau > 0:
            self._alpha = config.dt / (config.filter_tau + config.dt)
        else:
            self._alpha = 1.0  # no filtering

        # State
        self._H_prev: Optional[np.ndarray] = None      # H_{r/O} at k-1
        self._H_filtered: Optional[np.ndarray] = None   # EMA-filtered H
        self._H_dot_est = np.zeros(3)                    # latest estimate
        self._H_rO = np.zeros(3)                         # latest H_{r/O}
        self._initialized = False

    def reset(self) -> None:
        """Reset estimator state. Call on phase transitions or re-initialization."""
        self._H_prev = None
        self._H_filtered = None
        self._H_dot_est = np.zeros(3)
        self._H_rO = np.zeros(3)
        self._initialized = False

    def update(
        self,
        r_com: np.ndarray,
        v_com: np.ndarray,
        L_com: np.ndarray,
        omega_s: np.ndarray,
    ) -> np.ndarray:
        """Update estimator with current state and return Ḣ_{r/O} estimate.

        Parameters
        ----------
        r_com : ndarray (3,)
            Robot CoM position in R_s (relative to O = structure CoM).
        v_com : ndarray (3,)
            Robot CoM velocity in R_s.
        L_com : ndarray (3,)
            Robot centroidal angular momentum in R_s.
        omega_s : ndarray (3,)
            Structure angular velocity in world frame (from MuJoCo qvel[3:6]).

        Returns
        -------
        H_dot_est : ndarray (3,)
            Estimated inertial derivative of H_{r/O} [Nm].
            The AOCS feedforward should apply τ_w_ff = -H_dot_est.
        """
        # Compute H_{r/O} = L_com + r_com × (m · v_com)
        H_rO = L_com + np.cross(r_com, self._m * v_com)
        self._H_rO = H_rO.copy()

        # Apply EMA filter to H_{r/O} before differentiation
        if self._H_filtered is None:
            self._H_filtered = H_rO.copy()
        else:
            self._H_filtered = (self._alpha * H_rO +
                                (1.0 - self._alpha) * self._H_filtered)

        # Finite-difference derivative (in structure frame)
        if self._H_prev is None:
            H_dot_struct = np.zeros(3)
        else:
            H_dot_struct = (self._H_filtered - self._H_prev) / self._dt

        # Store filtered value for next step
        self._H_prev = self._H_filtered.copy()

        # Inertial derivative: add transport term
        if self.cfg.include_transport:
            H_dot_inertial = H_dot_struct + np.cross(omega_s, H_rO)
        else:
            H_dot_inertial = H_dot_struct

        self._H_dot_est = H_dot_inertial.copy()
        self._initialized = True
        return self._H_dot_est

    def update_analytical(
        self,
        r_com: np.ndarray,
        v_com: np.ndarray,
        L_com: np.ndarray,
        L_com_prev: np.ndarray,
        a_com: np.ndarray,
        omega_s: np.ndarray,
    ) -> np.ndarray:
        """Analytical variant using a_com directly (Variant B).

        Parameters
        ----------
        r_com : ndarray (3,)
            Robot CoM position in R_s.
        v_com : ndarray (3,)
            Robot CoM velocity in R_s.
        L_com : ndarray (3,)
            Current centroidal angular momentum.
        L_com_prev : ndarray (3,)
            Previous centroidal angular momentum.
        a_com : ndarray (3,)
            Robot CoM acceleration in R_s (e.g., from J_com @ qdd + Jdot_dq_com).
        omega_s : ndarray (3,)
            Structure angular velocity.

        Returns
        -------
        H_dot_est : ndarray (3,)
            Estimated inertial derivative of H_{r/O}.
        """
        H_rO = L_com + np.cross(r_com, self._m * v_com)
        self._H_rO = H_rO.copy()

        L_dot = (L_com - L_com_prev) / self._dt
        H_dot_struct = L_dot + np.cross(r_com, self._m * a_com)

        if self.cfg.include_transport:
            H_dot_inertial = H_dot_struct + np.cross(omega_s, H_rO)
        else:
            H_dot_inertial = H_dot_struct

        self._H_dot_est = H_dot_inertial.copy()
        self._initialized = True
        return self._H_dot_est

    # ── Properties for logging / diagnostics ──────────────────────

    @property
    def H_rO(self) -> np.ndarray:
        """Current H_{r/O} (unfiltered)."""
        return self._H_rO.copy()

    @property
    def H_dot(self) -> np.ndarray:
        """Latest Ḣ_{r/O} estimate."""
        return self._H_dot_est.copy()

    @property
    def initialized(self) -> bool:
        return self._initialized

    def __repr__(self) -> str:
        return (f"MomentumDisturbanceEstimator(m={self._m:.1f}kg, "
                f"dt={self._dt*1000:.0f}ms, "
                f"τ_f={self.cfg.filter_tau*1000:.0f}ms, "
                f"transport={self.cfg.include_transport})")


def compute_aocs_command(
    H_dot_est: np.ndarray,
    omega_s: np.ndarray,
    hw_current: np.ndarray,
    hw_target: np.ndarray = None,
    K_omega: float = 50.0,
    K_h: float = 0.5,
    tau_w_max: float = 2.5,
) -> np.ndarray:
    """Compute AOCS wheel torque command.

    τ_w = -Ḣ_est + K_ω · ω_s - K_h · (h_w - h_w*)

    Damping sign derivation: Newton-Euler about the structure CoM gives
    I_s · ω̇_s = -Ḣ_s - τ_w (wheel reaction). For ω_s>0 to brake
    (ω̇_s<0), we need τ_w > -Ḣ_s, i.e. the damping contribution
    must ADD positive — so +K_ω·ω_s, NOT the negated form.

    Parameters
    ----------
    H_dot_est : ndarray (3,)
        Estimated inertial derivative of H_{r/O}.
    omega_s : ndarray (3,)
        Structure angular velocity.
    hw_current : ndarray (3,)
        Current wheel angular momentum.
    hw_target : ndarray (3,), optional
        Target wheel momentum for desaturation. Default: zeros.
    K_omega : float
        Attitude damping gain [Nm·s/rad].
    K_h : float
        Desaturation gain [1/s].
    tau_w_max : float
        Maximum wheel torque magnitude [Nm].

    Returns
    -------
    tau_w : ndarray (3,)
        Wheel torque command (to be sent to MuJoCo ctrl[12:15]).
    """
    if hw_target is None:
        hw_target = np.zeros(3)

    tau_w = -H_dot_est + K_omega * omega_s - K_h * (hw_current - hw_target)
    tau_w = np.clip(tau_w, -tau_w_max, tau_w_max)
    return tau_w


def compute_aocs_command_legacy_corrected(
    L_com: np.ndarray,
    L_com_prev: np.ndarray,
    r_com: np.ndarray,
    v_com: np.ndarray,
    v_com_prev: np.ndarray,
    hw_current: np.ndarray,
    dt: float,
    robot_mass: float,
    K_hw: float = 2.0,
    hw_min: np.ndarray = None,
    hw_max: np.ndarray = None,
    tau_w_max: float = 2.5,
) -> np.ndarray:
    """Corrected legacy AOCS command with the orbital term (§5.8 / M4).

    Sign derivation (MuJoCo convention, verified empirically):
        ctrl[rw_i] is the motor torque on the wheel joint DOF.
        Ḣ_wheels = +ctrl[rw_i]  (wheel accelerates positively)
        Ḣ_struct = -ctrl[rw_i]  (Newton's third law reaction)

    Feedforward (cancel the robot's angular-momentum disturbance):
        conservation ⇒ Ḣ_struct = -ctrl[rw] - Ḣ_robot
        Ḣ_robot = L̇_com + r_com × m·v̇_com ≡ L_dot_est + orbital
        For Ḣ_struct = 0 : ctrl[rw] = -L_dot_est - orbital        ✓

    Desaturation (pull h_w back into the box):
        Ḣ_wheels = +ctrl[rw]; we want ḣ_w → hw_target, i.e.,
            ḣ_w = K_hw · (hw_target − h_w)
        With  hw_error := clip(h_w, box) − h_w = hw_target − h_w:
            ctrl[rw] += +K_hw · hw_error
        The "+" is correct — the code used to have "−K_hw · hw_error"
        which actively accelerated a saturated wheel AWAY from the
        box (verified 04/2026: with h_w=+7, K_hw=2, the old formula
        drove h_w to +8.96 Nms over 1 s; the corrected form drives
        it to +4.97 Nms).

    Full command:
        τ_w = -L̇_com_est - r_com × m·v̇_com_est + K_hw · hw_error

    The orbital term r_com × m·v̇_com_est was missing from the
    original legacy formula and is the M4 correction.

    All quantities are in the structure frame. Finite differences are
    one-step (suitable at dt_qp = 0.01 s). The `v_com_prev` and
    `L_com_prev` arguments are the state **at the previous QP sub-step**
    (before the current mj_step was applied).

    Parameters
    ----------
    L_com, L_com_prev : (3,) current and previous centroidal angular
        momentum (struct frame).
    r_com : (3,) current robot CoM position.
    v_com, v_com_prev : (3,) current and previous CoM velocity.
    hw_current : (3,) current wheel angular momentum (I_w · ω_wheels).
    dt : float, QP sub-step time step [s].
    robot_mass : float, total robot mass [kg].
    K_hw : float, desaturation gain [1/s].
    hw_min, hw_max : (3,) wheel momentum bounds (clip target).
    tau_w_max : float, wheel torque magnitude limit [Nm].

    Returns
    -------
    tau_w : (3,) clipped wheel torque command.
    """
    if hw_min is None:
        hw_min = -np.full(3, np.inf)
    if hw_max is None:
        hw_max = np.full(3, np.inf)

    # Centroidal (spin) rate estimate.
    L_dot_est = (L_com - L_com_prev) / dt

    # CoM acceleration estimate (new — this enables the orbital term).
    dv_com_est = (v_com - v_com_prev) / dt

    # Orbital rate term: d/dt (r_com × m·v_com) under the product rule
    # reduces to r_com × m·v̇_com (the v × m·v term vanishes identically).
    orbital = np.cross(r_com, robot_mass * dv_com_est)

    # Desaturation: drive h_w back into the feasible box.
    hw_error = np.clip(hw_current, hw_min, hw_max) - hw_current

    # Sign: +K_hw·hw_error, not −K_hw·hw_error (see docstring).
    tau_w = -L_dot_est - orbital + K_hw * hw_error
    return np.clip(tau_w, -tau_w_max, tau_w_max)


def compute_aocs_command_legacy_pd_numerical(
    L_com: np.ndarray,
    L_com_prev: np.ndarray,
    r_com: np.ndarray,
    v_com: np.ndarray,
    v_com_prev: np.ndarray,
    omega_s: np.ndarray,
    omega_s_prev: np.ndarray,
    hw_current: np.ndarray,
    dt: float,
    robot_mass: float,
    K_hw: float = 2.0,
    K_omega: float = 50.0,
    K_d: float = 25.0,
    hw_min: np.ndarray = None,
    hw_max: np.ndarray = None,
    tau_w_max: float = 2.5,
) -> np.ndarray:
    """Legacy-corrected AOCS + PD on ω_s with NUMERICAL ω̇_s estimate.

    Same feedforward + desaturation as ``compute_aocs_command_legacy_corrected``,
    plus a PD regulator that drives ω_s → 0:

        τ_w = -Ḣ_s_est + K_hw·hw_error - K_ω·ω_s - K_d·ω̇_s

    where ω̇_s is estimated by one-step finite difference:

        ω̇_s ≈ (ω_s − ω_s_prev) / dt

    The numerical-derivative variant is simple but noise-amplifying at
    dt_qp = 0.01 s — ω_s comes from MuJoCo's qvel with quantization noise
    that this derivative re-injects. Lower K_d reduces saturation risk
    from differentiation noise.

    Parameters
    ----------
    omega_s, omega_s_prev : (3,) current and previous structure angular
        velocity [rad/s], in structure body frame.
    K_omega : float, attitude-rate damping gain [Nm·s/rad].
    K_d : float, attitude-acceleration damping gain [Nm·s²/rad].

    Other parameters: see ``compute_aocs_command_legacy_corrected``.
    """
    if hw_min is None:
        hw_min = -np.full(3, np.inf)
    if hw_max is None:
        hw_max = np.full(3, np.inf)

    L_dot_est = (L_com - L_com_prev) / dt
    dv_com_est = (v_com - v_com_prev) / dt
    orbital = np.cross(r_com, robot_mass * dv_com_est)
    hw_error = np.clip(hw_current, hw_min, hw_max) - hw_current

    omega_dot_est = (omega_s - omega_s_prev) / dt
    # Damping sign derivation (Newton-Euler about structure CoM):
    #   I_s · ω̇_s = -Ḣ_s - τ_w  (wheel reaction = -τ_w by Newton's 3rd)
    # For ω_s>0 to decrease, want ω̇_s<0 → τ_w > -Ḣ_s, i.e. ADD positive
    # contribution. So the PD term is +K_ω·ω_s + K_d·ω̇_s, NOT negated.
    # (H_est uses -K_ω·ω_s — that sign was untested under non-zero ω_s.)
    pd_term = K_omega * omega_s + K_d * omega_dot_est

    tau_w = -L_dot_est - orbital + K_hw * hw_error + pd_term
    return np.clip(tau_w, -tau_w_max, tau_w_max)


def compute_aocs_command_legacy_pd_model(
    L_com: np.ndarray,
    L_com_prev: np.ndarray,
    r_com: np.ndarray,
    v_com: np.ndarray,
    v_com_prev: np.ndarray,
    omega_s: np.ndarray,
    tau_w_prev: np.ndarray,
    I_struct: np.ndarray,
    hw_current: np.ndarray,
    dt: float,
    robot_mass: float,
    K_hw: float = 2.0,
    K_omega: float = 50.0,
    K_d: float = 25.0,
    hw_min: np.ndarray = None,
    hw_max: np.ndarray = None,
    tau_w_max: float = 2.5,
) -> np.ndarray:
    """Legacy-corrected AOCS + PD on ω_s with MODEL-BASED ω̇_s estimate.

    Structure-body Newton-Euler gives:

        I_s · ω̇_s = -Ḣ_s_real - τ_w

    Using the AOCS's own disturbance estimate Ḣ_s_est and the previously
    applied wheel torque τ_w_prev (one-step lag avoids self-reference):

        ω̇_s_est = (-Ḣ_s_est - τ_w_prev) / I_s

    Interpretation: ω̇_s_est captures "the residual disturbance the last
    tick's wheel torque did NOT absorb, per unit structure inertia." The
    D term -K_d·ω̇_s_est then increases the current command in the
    direction that would have absorbed it. Cleaner than numerical
    differentiation (no high-frequency noise re-injection) but tied to
    the model's accuracy (depends on a credible Ḣ_s_est and a correct
    structure-inertia value).

    Parameters
    ----------
    omega_s : (3,) current structure angular velocity [rad/s].
    tau_w_prev : (3,) wheel torque commanded the previous QP sub-tick
        [Nm]. Use the actually-clipped command (post-saturation).
    I_struct : (3,) structure-body principal moments of inertia [kg·m²].
        Read from ``mj_model.body_inertia[struct_bid]``.
    K_omega : float, attitude-rate damping gain [Nm·s/rad].
    K_d : float, attitude-acceleration damping gain [Nm·s²/rad].

    Other parameters: see ``compute_aocs_command_legacy_corrected``.
    """
    if hw_min is None:
        hw_min = -np.full(3, np.inf)
    if hw_max is None:
        hw_max = np.full(3, np.inf)

    L_dot_est = (L_com - L_com_prev) / dt
    dv_com_est = (v_com - v_com_prev) / dt
    orbital = np.cross(r_com, robot_mass * dv_com_est)
    H_dot_est = L_dot_est + orbital  # ≡ Ḣ_s
    hw_error = np.clip(hw_current, hw_min, hw_max) - hw_current

    # Model-based ω̇_s from last tick's residual.
    omega_dot_est = (-H_dot_est - tau_w_prev) / np.maximum(I_struct, 1e-9)
    # Damping sign: +K_ω·ω_s + K_d·ω̇_s (see numerical mode for derivation).
    pd_term = K_omega * omega_s + K_d * omega_dot_est

    tau_w = -H_dot_est + K_hw * hw_error + pd_term
    return np.clip(tau_w, -tau_w_max, tau_w_max)


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


def compute_aocs_command_legacy_pid_model(
    L_com: np.ndarray,
    L_com_prev: np.ndarray,
    r_com: np.ndarray,
    v_com: np.ndarray,
    v_com_prev: np.ndarray,
    omega_s: np.ndarray,
    theta_s: np.ndarray,
    tau_w_prev: np.ndarray,
    I_struct: np.ndarray,
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
    """Legacy-corrected AOCS + PID on attitude (model-based ω̇_s).

    Extends ``compute_aocs_command_legacy_pd_model`` with the
    attitude-tracking P term (see legacy_pid_numerical for the
    derivation and wheel-capacity ceiling).

        τ_w = -Ḣ_s_est + K_hw·hw_error
              + K_θ·θ_s + K_ω·ω_s + K_d·ω̇_s_model

    Parameters: see ``compute_aocs_command_legacy_pd_model`` and
    ``compute_aocs_command_legacy_pid_numerical``.
    """
    if hw_min is None:
        hw_min = -np.full(3, np.inf)
    if hw_max is None:
        hw_max = np.full(3, np.inf)

    hw_error = np.clip(hw_current, hw_min, hw_max) - hw_current

    if tau_struct_ff is None:
        # FD-based Ḣ feedforward (kinematic; complete in SS only).
        L_dot_est = (L_com - L_com_prev) / dt
        dv_com_est = (v_com - v_com_prev) / dt
        orbital = np.cross(r_com, robot_mass * dv_com_est)
        H_dot_est = L_dot_est + orbital
    else:
        # Direct wrench feedforward (DS-correct): tau_struct_ff = -Ḣ_robot
        # computed by the caller from λ_qp. The model-based ω̇ estimate
        # uses H_dot_est = -tau_struct_ff (matches the SS conservation
        # identity Ḣ_robot = -tau_struct_ff used in the FD branch).
        H_dot_est = -tau_struct_ff

    omega_dot_est = (-H_dot_est - tau_w_prev) / np.maximum(I_struct, 1e-9)
    pid_term = K_theta * theta_s + K_omega * omega_s + K_d * omega_dot_est

    tau_w = -H_dot_est + K_hw * hw_error + pid_term
    return np.clip(tau_w, -tau_w_max, tau_w_max)
