"""Swing arm trajectory planner for crawling locomotion.

Generates smooth Cartesian position/velocity references for the free
end-effector during single-support phases.

All coordinates are in the **structure body frame**.  Since the scheduler
anchors are now constant positions in this frame, no live-anchor machinery
or nominal→live transforms are needed.

Trajectory design
-----------------
Given a swing from anchor_start → anchor_end over duration T:

    p(τ) = p_start + Δp · s(τ)  +  clearance · n̂ · bump(τ)

where τ = (t − t_phase_start) / T ∈ [0, 1],

    s(τ) = 10τ³ − 15τ⁴ + 6τ⁵     (quintic, rest-to-rest)
    bump(τ) = sin²(πτ)              (clearance bell, C¹ boundaries)
    n̂ = unit vector away from structure surface (in structure frame)

This ensures:
    - p(0) = p_start,  p(1) = p_end
    - v(0) = v(1) = 0  (smooth detach/attach)
    - a(0) = a(1) = 0  (no jerk at transitions)
    - Maximum clearance at mid-swing (τ = 0.5)
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin
from dataclasses import dataclass
from typing import Optional, Tuple

from .contact_scheduler import ContactScheduler, GaitPhase, GaitPlan
from ..solvers.contact_phase import ContactPhase


# Default clearance [m] — matches SimConfig.swing_clearance.
DEFAULT_CLEARANCE = 0.03

# Normal vector pointing away from the structure surface (structure frame).
# Structure surface is at z ≈ +0.025 in structure frame; robot hangs below → away = −z.
DEFAULT_AWAY_NORMAL = np.array([0.0, 0.0, -1.0])


@dataclass
class SwingReference:
    """Cartesian reference for the swing arm at a given instant (structure frame)."""
    p_ee: np.ndarray       # (3,) position [m]
    v_ee: np.ndarray       # (3,) velocity [m/s]
    a_ee: np.ndarray       # (3,) acceleration [m/s²]
    R_ee: np.ndarray       # (3,3) rotation matrix
    omega_ee: np.ndarray   # (3,) angular velocity [rad/s]
    alpha_ee: np.ndarray   # (3,) angular acceleration [rad/s²]
    swing_arm: str         # 'a' or 'b'
    is_swinging: bool      # True during single-support swing
    phase_progress: float  # τ ∈ [0, 1]


class SwingPlanner:
    """Plan Cartesian trajectories for the free arm during crawling.

    All coordinates are in the structure body frame.  The scheduler anchors
    are constant positions in this frame, so no live-reading or frame
    transform is needed.

    Parameters
    ----------
    scheduler : ContactScheduler
        Must have a traversal plan already built.  Anchors must be in
        structure-local frame.
    clearance : float
        Peak clearance distance normal to the structure surface [m].
    away_normal : ndarray (3,)
        Unit vector pointing away from the structure surface (struct frame).
    """

    def __init__(
        self,
        scheduler: ContactScheduler,
        clearance: float = DEFAULT_CLEARANCE,
        away_normal: np.ndarray = DEFAULT_AWAY_NORMAL,
        rotation_delay_ratio: float = 0.2,
        bump_peak_tau: float = 0.5,
        early_finish_fraction: float = 1.0,
    ):
        self.scheduler = scheduler
        self.clearance = clearance
        self.away_normal = away_normal / np.linalg.norm(away_normal)
        # Orientation at swing release (set by sim_loop before SS phase)
        self._R_start: np.ndarray = np.eye(3)
        # Target orientation: identity (tool aligned with structure frame)
        self._R_end: np.ndarray = np.eye(3)
        # M5: delayed cosine timing — orientation stays at R_start until
        # τ = rotation_delay_ratio, then smoothly rotates to R_end.
        # Concentrates the rotation in the approach phase.
        self.rotation_delay_ratio = float(rotation_delay_ratio)
        # M7: clearance-bump peak location.
        # 0.5 = legacy symmetric sin²(πτ) — peak at mid-swing.
        # 0.25 = shift peak early so the arm is back near the port
        # plane during the (N_torso·J_ee)-singular window at τ≈0.5.
        # The bump is a C¹-smooth asymmetric sin² built from two
        # half-period sinusoids joined at τ=bump_peak_tau.
        if not (0.0 < bump_peak_tau < 1.0):
            raise ValueError(
                f"bump_peak_tau must be in (0,1), got {bump_peak_tau}")
        self.bump_peak_tau = float(bump_peak_tau)
        # M7 v22: effective-duration fraction. Analogue of
        # TorsoPlanner's early_finish_fraction. When < 1.0, the swing
        # trajectory completes at τ=1 on the compressed timebase
        # `t_eff = (t - t_start) / (ef · T)`, then HOLDS at the target
        # with v=0, a=0 for the remainder of the planned phase. The
        # hold is automatic because the quintic, bump, and
        # delayed-cosine profiles all have zero first derivative at
        # τ=1 by construction, and `np.clip(..., 0, 1)` freezes τ.
        if not (0.0 < float(early_finish_fraction) <= 1.0):
            raise ValueError(
                f"early_finish_fraction must be in (0, 1], "
                f"got {early_finish_fraction}")
        self.early_finish_fraction = float(early_finish_fraction)

    def set_swing_orientation(self, R_start: np.ndarray) -> None:
        """Set the tool rotation at swing release for SLERP interpolation."""
        self._R_start = R_start.copy()

    @property
    def plan(self) -> GaitPlan:
        return self.scheduler.plan

    # ── Primitive profiles ───────────────────────────────────────

    @staticmethod
    def _quintic(tau: float) -> float:
        t2 = tau * tau
        t3 = t2 * tau
        return 10.0 * t3 - 15.0 * t2 * t2 + 6.0 * t2 * t3

    @staticmethod
    def _quintic_dot(tau: float) -> float:
        t2 = tau * tau
        return 30.0 * t2 - 60.0 * t2 * tau + 30.0 * t2 * t2

    @staticmethod
    def _quintic_ddot(tau: float) -> float:
        return 60.0 * tau - 180.0 * tau * tau + 120.0 * tau * tau * tau

    # M7: asymmetric clearance bump. Peak at tau = bump_peak_tau
    # (default 0.5 = legacy symmetric sin²(πτ)). Built from two
    # half-period sinusoids joined C¹-smoothly at the peak:
    #   rise    (τ ≤ τ_p)  : sin²( π·τ       / (2·τ_p)     )
    #   descent (τ ≥ τ_p)  : sin²( π·(1-τ)   / (2·(1-τ_p)) )
    # Both reach 1 at τ=τ_p, 0 at τ=0 and τ=1, with zero first
    # derivative at 0, τ_p, and 1. Reduces to sin²(πτ) when τ_p=0.5.
    def _bump(self, tau: float) -> float:
        tp = self.bump_peak_tau
        if tau <= tp:
            s = np.sin(np.pi * tau / (2.0 * tp))
        else:
            s = np.sin(np.pi * (1.0 - tau) / (2.0 * (1.0 - tp)))
        return s * s

    def _bump_dot(self, tau: float) -> float:
        tp = self.bump_peak_tau
        if tau <= tp:
            # d/dτ sin²(πτ/(2τ_p)) = (π/(2τ_p))·sin(πτ/τ_p)
            return (np.pi / (2.0 * tp)) * np.sin(np.pi * tau / tp)
        else:
            # d/dτ sin²(π(1-τ)/(2(1-τ_p))) =
            #   -(π/(2(1-τ_p)))·sin(π(1-τ)/(1-τ_p))
            return -(np.pi / (2.0 * (1.0 - tp))) * \
                np.sin(np.pi * (1.0 - tau) / (1.0 - tp))

    def _bump_ddot(self, tau: float) -> float:
        tp = self.bump_peak_tau
        if tau <= tp:
            # d²/dτ² = (π²/(2·τ_p²))·cos(πτ/τ_p)
            return (np.pi ** 2 / (2.0 * tp * tp)) * \
                np.cos(np.pi * tau / tp)
        else:
            # d²/dτ² = (π²/(2·(1-τ_p)²))·cos(π(1-τ)/(1-τ_p))
            return (np.pi ** 2 / (2.0 * (1.0 - tp) ** 2)) * \
                np.cos(np.pi * (1.0 - tau) / (1.0 - tp))

    # ── M5: Delayed cosine timing for orientation SLERP ─────────
    # Concentrates rotation in the second half of the swing so the
    # EE first arcs over (clearance bump peaks at tau=0.5) and THEN
    # rotates into the dock orientation during the approach.
    #
    #   σ(τ) = 0                                 if τ < τ_d
    #          0.5 * (1 - cos(π·(τ - τ_d)/(1 - τ_d)))   else
    # where τ_d = delay_ratio ∈ [0, 1).  σ(τ_d)=0, σ(1)=1, σ̇(τ_d)=0,
    # σ̇(1)=0 (smooth at both endpoints and at the delay boundary).
    @staticmethod
    def _delayed_cosine(tau: float, tau_d: float) -> float:
        if tau <= tau_d:
            return 0.0
        if tau >= 1.0:
            return 1.0
        u = (tau - tau_d) / (1.0 - tau_d)
        return 0.5 * (1.0 - np.cos(np.pi * u))

    @staticmethod
    def _delayed_cosine_dot(tau: float, tau_d: float) -> float:
        if tau <= tau_d or tau >= 1.0:
            return 0.0
        denom = 1.0 - tau_d
        u = (tau - tau_d) / denom
        return 0.5 * np.pi * np.sin(np.pi * u) / denom

    @staticmethod
    def _delayed_cosine_ddot(tau: float, tau_d: float) -> float:
        if tau <= tau_d or tau >= 1.0:
            return 0.0
        denom = 1.0 - tau_d
        u = (tau - tau_d) / denom
        return 0.5 * (np.pi ** 2) * np.cos(np.pi * u) / (denom ** 2)

    # ── Main query ───────────────────────────────────────────────

    def reference_at(self, t: float) -> SwingReference:
        """Get the swing arm reference at time t (structure frame).

        Parameters
        ----------
        t : float
            Simulation time [s].

        Returns
        -------
        ref : SwingReference
        """
        plan = self.plan
        gp, idx = plan.phase_at(t)

        # ── Double support: no swing ─────────────────────────────
        if gp.phase == ContactPhase.DOUBLE:
            arm, p_ee = self._last_swing_position(idx)
            return SwingReference(
                p_ee=p_ee, v_ee=np.zeros(3), a_ee=np.zeros(3),
                R_ee=self._R_end.copy(), omega_ee=np.zeros(3), alpha_ee=np.zeros(3),
                swing_arm=arm, is_swinging=False, phase_progress=1.0)

        # ── Single support: compute swing trajectory ─────────────
        t_start = plan.t_start[idx]
        # M7 v22: effective duration = early_finish_fraction · T_step.
        # tau reaches 1 at t = t_start + T_eff, then stays clipped at 1
        # for the remainder of the planned phase → position at target,
        # velocity and acceleration are zero (all three profiles
        # — quintic s, bump, delayed-cosine σ_r — have ṗ(τ=1)=0 by
        # construction, and np.clip freezes τ).
        T = gp.duration
        T_eff = T * self.early_finish_fraction
        tau = np.clip((t - t_start) / T_eff, 0.0, 1.0)

        # Anchor positions from scheduler (constant structure-frame coordinates)
        if gp.swing_arm == 'b':
            p_start = self.scheduler.anchors_b[gp.swing_from_idx].copy()
            p_end = self.scheduler.anchors_b[gp.swing_to_idx].copy()
        elif gp.swing_arm == 'a':
            p_start = self.scheduler.anchors_a[gp.swing_from_idx].copy()
            p_end = self.scheduler.anchors_a[gp.swing_to_idx].copy()
        else:
            raise ValueError(f"SS phase without swing_arm set at idx={idx}")

        dp = p_end - p_start
        n = self.away_normal

        # Position
        s = self._quintic(tau)
        bump = self._bump(tau)
        p_ee = p_start + dp * s + self.clearance * n * bump

        # Velocity (chain rule: dτ/dt = 1/T_eff)
        s_dot = self._quintic_dot(tau) / T_eff
        bump_dot = self._bump_dot(tau) / T_eff
        v_ee = dp * s_dot + self.clearance * n * bump_dot

        # Acceleration (d²τ/dt² scales as 1/T_eff²)
        s_ddot = self._quintic_ddot(tau) / (T_eff * T_eff)
        bump_ddot = self._bump_ddot(tau) / (T_eff * T_eff)
        a_ee = dp * s_ddot + self.clearance * n * bump_ddot

        # M5: Orientation via SLERP with delayed-cosine timing.
        # Position uses quintic `s` (above); orientation uses a
        # separate σ_r that is zero until τ=rotation_delay_ratio and
        # then smoothly rotates. This concentrates the orientation
        # change in the approach phase while keeping the EE nearly
        # flat during the clearance bump.
        tau_d = self.rotation_delay_ratio
        sigma_r = self._delayed_cosine(tau, tau_d)
        sigma_r_dot = self._delayed_cosine_dot(tau, tau_d) / T_eff
        sigma_r_ddot = self._delayed_cosine_ddot(tau, tau_d) / (T_eff * T_eff)

        dR = self._R_start.T @ self._R_end
        omega_total = pin.log3(dR)          # (3,) body-frame axis·angle
        R_ee = self._R_start @ pin.exp3(sigma_r * omega_total)
        # Body-frame angular velocity = σ̇·Δθ; transport to world via R.
        omega_ee = R_ee @ (sigma_r_dot * omega_total)
        alpha_ee = R_ee @ (sigma_r_ddot * omega_total)

        return SwingReference(
            p_ee=p_ee, v_ee=v_ee, a_ee=a_ee,
            R_ee=R_ee, omega_ee=omega_ee, alpha_ee=alpha_ee,
            swing_arm=gp.swing_arm,
            is_swinging=True,
            phase_progress=tau)

    # ── Adaptive re-planning ────────────────────────────────────

    def adaptive_reference_at(
        self,
        t: float,
        p_ee_current: np.ndarray,
        T_remaining_min: float = 0.5,
    ) -> SwingReference:
        """Closed-loop swing reference: re-plan quintic from current EE position.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        p_ee_current : ndarray (3,)
            Measured EE position in structure frame.
        T_remaining_min : float
            Minimum horizon for the re-planned quintic [s].

        Returns
        -------
        ref : SwingReference
        """
        plan = self.plan
        gp, idx = plan.phase_at(min(t, plan.t_end[-1] - 1e-6))

        if gp.phase == ContactPhase.DOUBLE and t < plan.t_end[-1]:
            return self.reference_at(t)

        # Find the active or most recent SS phase
        ss_idx = idx
        if gp.phase == ContactPhase.DOUBLE:
            for i in range(idx - 1, -1, -1):
                if plan.phases[i].swing_arm:
                    ss_idx = i
                    break
        gp_ss = plan.phases[ss_idx]
        if not gp_ss.swing_arm:
            return self.reference_at(min(t, plan.t_end[-1] - 1e-6))

        # Target position (constant in structure frame)
        if gp_ss.swing_arm == 'b':
            p_end = self.scheduler.anchors_b[gp_ss.swing_to_idx].copy()
        else:
            p_end = self.scheduler.anchors_a[gp_ss.swing_to_idx].copy()

        dp = p_end - p_ee_current
        d_remaining = np.linalg.norm(dp)

        if d_remaining < 1e-4:
            return SwingReference(
                p_ee=p_end, v_ee=np.zeros(3), a_ee=np.zeros(3),
                swing_arm=gp_ss.swing_arm, is_swinging=True,
                phase_progress=1.0)

        t_ss_end = plan.t_end[ss_idx]
        T_rem = max(t_ss_end - t, T_remaining_min)

        dt_eval = 0.01
        tau_eval = np.clip(dt_eval / T_rem, 0.0, 0.5)

        s = self._quintic(tau_eval)
        s_dot = self._quintic_dot(tau_eval) / T_rem
        s_ddot = self._quintic_ddot(tau_eval) / (T_rem * T_rem)

        t_ss_start = plan.t_start[ss_idx]
        T_original = gp_ss.duration
        tau_original = np.clip((t - t_ss_start) / T_original, 0.0, 1.0)

        if tau_original < 0.8:
            bump = self._bump(tau_original)
            bump_dot = self._bump_dot(tau_original) / T_original
            bump_ddot = self._bump_ddot(tau_original) / (T_original ** 2)
            n = self.away_normal
            p_ee = p_ee_current + dp * s + self.clearance * n * bump
            v_ee = dp * s_dot + self.clearance * n * bump_dot
            a_ee = dp * s_ddot + self.clearance * n * bump_ddot
        else:
            p_ee = p_ee_current + dp * s
            v_ee = dp * s_dot
            a_ee = dp * s_ddot

        return SwingReference(
            p_ee=p_ee, v_ee=v_ee, a_ee=a_ee,
            swing_arm=gp_ss.swing_arm, is_swinging=True,
            phase_progress=tau_original)

    # ── Helpers ──────────────────────────────────────────────────

    def _last_swing_position(self, current_idx: int) -> Tuple[str, np.ndarray]:
        """Find the anchor where the last swing arm landed (structure frame)."""
        plan = self.plan
        for i in range(current_idx - 1, -1, -1):
            gp_prev = plan.phases[i]
            if gp_prev.swing_arm:
                if gp_prev.swing_arm == 'b':
                    p = self.scheduler.anchors_b[gp_prev.swing_to_idx].copy()
                else:
                    p = self.scheduler.anchors_a[gp_prev.swing_to_idx].copy()
                return gp_prev.swing_arm, p

        gp0 = plan.phases[0]
        return 'b', self.scheduler.anchors_b[gp0.anchor_b_idx].copy()

    def swing_trajectory(
        self,
        phase_idx: int,
        dt: float = 0.001,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample the full swing trajectory for a single-support phase.

        Returns
        -------
        t_vec : ndarray (K,)
        p_vec : ndarray (K, 3)
        v_vec : ndarray (K, 3)
        """
        plan = self.plan
        gp = plan.phases[phase_idx]
        assert gp.swing_arm, f"Phase {phase_idx} is not a swing phase"

        t0 = plan.t_start[phase_idx]
        T = gp.duration
        K = int(round(T / dt)) + 1
        t_vec = np.linspace(t0, t0 + T, K)
        p_vec = np.zeros((K, 3))
        v_vec = np.zeros((K, 3))

        for k, tk in enumerate(t_vec):
            ref = self.reference_at(tk)
            p_vec[k] = ref.p_ee
            v_vec[k] = ref.v_ee

        return t_vec, p_vec, v_vec
