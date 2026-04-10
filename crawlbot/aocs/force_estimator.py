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
    tau_w_max: float = 5.0,
) -> np.ndarray:
    """Compute AOCS wheel torque command.

    τ_w = -Ḣ_est - K_ω · ω_s - K_h · (h_w - h_w*)

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

    tau_w = -H_dot_est - K_omega * omega_s - K_h * (hw_current - hw_target)
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
    tau_w_max: float = 5.0,
) -> np.ndarray:
    """Corrected legacy AOCS command with the orbital term (§5.8 / M4).

    Legacy formula (currently in sim_loop.py) was:

        τ_w = -L̇_com_est - K_hw · (h_w - clip(h_w, bounds))

    The M4 fix adds the missing orbital rate term so that the wheels
    reject the full disturbance about O_s, not just the centroidal
    (spin) component:

        τ_w = -L̇_com_est
              - r_com × m · v̇_com_est          <-- NEW (orbital rate)
              - K_hw · (h_w - clip(h_w, bounds))

    This matches the decomposition from §4.5–4.6:
        Ḣ_{r/O} = L̇_com + r_com × m · v̇_com   (struct-frame derivative)
    The legacy formula had only the first term, which is why the
    platform rotated ~24° at 14 % mass ratio in the M0 baseline.

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

    tau_w = -L_dot_est - orbital - K_hw * hw_error
    return np.clip(tau_w, -tau_w_max, tau_w_max)
