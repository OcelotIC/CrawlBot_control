"""
Generalized Momentum Observer (GMO) for sensorless contact detection.

Implements the De Luca (2006) observer on the full 18D generalized momentum
of a free-flying dual-arm robot. The observer residual converges to the
generalized external force tau_ext = J_c^T f_ext without requiring
acceleration measurements or force/torque sensors.

Theory
------
Equations of motion (zero gravity):

    M(q) v_dot + C(q,v) v = S^T tau + J_c^T f_ext

Generalized momentum p = M(q) v satisfies:

    p_dot = S^T tau + C^T v + J_c^T f_ext

Observer (De Luca 2006, integral form):

    beta_dot = S^T tau + C^T v + r      (integrator with feedback)
    r = K_O (p - beta)                  (residual signal)

Convergence: r_dot = -K_O r + K_O tau_ext  =>  r -> tau_ext = J_c^T f_ext.

See Misc/reports/contact_estimator_derivation.md for the full derivation.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional


@dataclass
class ContactObserverConfig:
    """Configuration for the GMO and contact state machine."""
    K_O: float = 80.0             # Observer gain [1/s]
    dt: float = 0.01              # QP timestep [s]
    nv: int = 18                  # Generalized velocity dimension
    F_threshold: float = 5.0     # Swing residual norm for CONTACT [N]
    d_proximity: float = 0.020   # NO_CONTACT -> PROXIMITY [m]
    d_contact: float = 0.010     # PROXIMITY -> CONTACT [m]
    d_reset: float = 0.030       # Any -> NO_CONTACT [m]
    debounce_count: int = 5      # CONTACT -> CONFIRMED [cycles]


class ContactState(Enum):
    """Contact detection state machine states."""
    NO_CONTACT = 0
    PROXIMITY = 1
    CONTACT = 2
    CONFIRMED = 3


class GeneralizedMomentumObserver:
    """De Luca (2006) generalized momentum observer.

    Runs at the QP servo rate (100 Hz). Integrates the known torques
    and Coriolis terms, comparing the integral against the measured
    generalized momentum p = M(q) v. The residual r converges to
    J_c^T f_ext — the generalized external force.

    Parameters
    ----------
    config : ContactObserverConfig
        Observer gain K_O, timestep dt, velocity dimension nv.
    """

    def __init__(self, config: ContactObserverConfig):
        self.cfg = config
        self._beta = np.zeros(config.nv)     # integrator state
        self._r = np.zeros(config.nv)        # residual
        self._initialized = False

    def reset(self, M: np.ndarray, v: np.ndarray) -> None:
        """Initialize integrator: beta_0 = p_0 = M v, r_0 = 0.

        Call at startup and on phase transitions (weld activate/deactivate)
        to avoid transients from discontinuous contact forces.

        Parameters
        ----------
        M : ndarray (nv, nv)
            Mass matrix at current configuration.
        v : ndarray (nv,)
            Generalized velocity.
        """
        self._beta = M @ v
        self._r = np.zeros(self.cfg.nv)
        self._initialized = True

    def update(
        self,
        M: np.ndarray,
        v: np.ndarray,
        C_matrix: np.ndarray,
        tau_applied: np.ndarray,
    ) -> np.ndarray:
        """One observer step. Returns the residual r_k (nv,).

        Parameters
        ----------
        M : ndarray (nv, nv)
            Mass matrix M(q_k).
        v : ndarray (nv,)
            Generalized velocity v_k.
        C_matrix : ndarray (nv, nv)
            Coriolis matrix C(q_k, v_k).
        tau_applied : ndarray (nv,)
            Applied generalized force: [0_6; tau_joints].

        Returns
        -------
        r : ndarray (nv,)
            Observer residual. Converges to J_c^T f_ext.
        """
        if not self._initialized:
            self.reset(M, v)

        K = self.cfg.K_O
        dt = self.cfg.dt

        # Forward Euler integration of beta
        # beta_dot = tau_applied + C^T v + r
        CT_v = C_matrix.T @ v
        self._beta += dt * (tau_applied + CT_v + self._r)

        # Generalized momentum
        p = M @ v

        # Residual
        self._r = K * (p - self._beta)
        return self._r.copy()

    @property
    def residual(self) -> np.ndarray:
        """Current residual r (nv,). Converges to J_c^T f_ext."""
        return self._r.copy()

    @property
    def initialized(self) -> bool:
        return self._initialized

    def swing_residual_norm(self, swing_v_slice: slice) -> float:
        """Norm of the residual on the swing arm joints.

        Parameters
        ----------
        swing_v_slice : slice
            Velocity indices of the swing arm (e.g. slice(6,12) for arm_a).

        Returns
        -------
        float
            ||r[swing_joints]|| — indicator of external force on swing arm.
        """
        return float(np.linalg.norm(self._r[swing_v_slice]))


class ContactStateMachine:
    """Multi-modal contact detection fusing GMO residual + FK distance.

    State transitions::

        NO_CONTACT --[d < d_proximity]--> PROXIMITY
        PROXIMITY  --[contact_condition AND d < d_contact]--> CONTACT
        CONTACT    --[sustained debounce_count cycles]--> CONFIRMED
        Any        --[d > d_reset]--> NO_CONTACT  (hysteresis)

    Contact condition (mode-dependent):
    - Hardware (force_mode=True): ||r_swing|| > F_threshold (force-based)
    - Simulation (force_mode=False): d < d_contact (kinematic-only, since
      MuJoCo has no contact force before weld activation)

    Parameters
    ----------
    config : ContactObserverConfig
        Thresholds and debounce parameters.
    """

    def __init__(self, config: ContactObserverConfig):
        self.cfg = config
        self._state = ContactState.NO_CONTACT
        self._debounce_counter = 0

    def update(
        self,
        r_swing_norm: float,
        d_FK: float,
        force_mode: bool = False,
    ) -> ContactState:
        """Advance the state machine with new measurements.

        Parameters
        ----------
        r_swing_norm : float
            ||r[swing_joints]|| from the GMO.
        d_FK : float
            Euclidean distance between gripper and target anchor (FK).
        force_mode : bool
            If True, require force residual for CONTACT (hardware mode).
            If False, use kinematic proximity only (simulation mode).

        Returns
        -------
        ContactState
            Current state after transition.
        """
        cfg = self.cfg

        # Global reset: if distance exceeds hysteresis threshold
        if d_FK > cfg.d_reset:
            self._state = ContactState.NO_CONTACT
            self._debounce_counter = 0
            return self._state

        # Contact condition depends on mode
        if force_mode:
            contact_condition = (r_swing_norm > cfg.F_threshold
                                 and d_FK < cfg.d_contact)
        else:
            # Simulation: no contact force before weld, use kinematic only
            contact_condition = d_FK < cfg.d_contact

        if self._state == ContactState.NO_CONTACT:
            if d_FK < cfg.d_proximity:
                self._state = ContactState.PROXIMITY

        elif self._state == ContactState.PROXIMITY:
            if contact_condition:
                self._state = ContactState.CONTACT
                self._debounce_counter = 0

        elif self._state == ContactState.CONTACT:
            if contact_condition:
                self._debounce_counter += 1
                if self._debounce_counter >= cfg.debounce_count:
                    self._state = ContactState.CONFIRMED
            else:
                self._state = ContactState.PROXIMITY
                self._debounce_counter = 0

        # CONFIRMED is absorbing (only reset by d > d_reset or explicit reset())

        return self._state

    def reset(self) -> None:
        """Reset to NO_CONTACT. Call on phase transitions."""
        self._state = ContactState.NO_CONTACT
        self._debounce_counter = 0

    @property
    def state(self) -> ContactState:
        return self._state

    @property
    def is_docked(self) -> bool:
        """True when contact is fully confirmed (CONFIRMED state)."""
        return self._state == ContactState.CONFIRMED
