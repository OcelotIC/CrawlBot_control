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
from dataclasses import dataclass
from typing import Optional, Tuple

from contact_scheduler import ContactScheduler, GaitPhase, GaitPlan
from solvers.contact_phase import ContactPhase


# Default clearance: 8 cm away from the structure surface.
DEFAULT_CLEARANCE = 0.08

# Normal vector pointing away from the structure surface (structure frame).
# Structure surface is at z ≈ +0.025 in structure frame; robot hangs below → away = −z.
DEFAULT_AWAY_NORMAL = np.array([0.0, 0.0, -1.0])


@dataclass
class SwingReference:
    """Cartesian reference for the swing arm at a given instant (structure frame)."""
    p_ee: np.ndarray       # (3,) position [m]
    v_ee: np.ndarray       # (3,) velocity [m/s]
    a_ee: np.ndarray       # (3,) acceleration [m/s²]
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
    ):
        self.scheduler = scheduler
        self.clearance = clearance
        self.away_normal = away_normal / np.linalg.norm(away_normal)

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

    @staticmethod
    def _bump(tau: float) -> float:
        s = np.sin(np.pi * tau)
        return s * s

    @staticmethod
    def _bump_dot(tau: float) -> float:
        return np.pi * np.sin(2.0 * np.pi * tau)

    @staticmethod
    def _bump_ddot(tau: float) -> float:
        return 2.0 * np.pi * np.pi * np.cos(2.0 * np.pi * tau)

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
                swing_arm=arm, is_swinging=False, phase_progress=1.0)

        # ── Single support: compute swing trajectory ─────────────
        t_start = plan.t_start[idx]
        T = gp.duration
        tau = np.clip((t - t_start) / T, 0.0, 1.0)

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

        # Velocity (chain rule: dp/dt = dp/dτ · 1/T)
        s_dot = self._quintic_dot(tau) / T
        bump_dot = self._bump_dot(tau) / T
        v_ee = dp * s_dot + self.clearance * n * bump_dot

        # Acceleration (d²p/dt² = d²p/dτ² · 1/T²)
        s_ddot = self._quintic_ddot(tau) / (T * T)
        bump_ddot = self._bump_ddot(tau) / (T * T)
        a_ee = dp * s_ddot + self.clearance * n * bump_ddot

        return SwingReference(
            p_ee=p_ee, v_ee=v_ee, a_ee=a_ee,
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
