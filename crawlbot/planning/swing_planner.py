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
        model: Optional[pin.Model] = None,
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
        # Piecewise-quintic override store for mid-waypoint reshape
        # (T15_step2_path_geometry.md §7.3 Option B). When non-empty,
        # ``reference_at(t)`` consults this list first; if t falls in
        # an override window, the piecewise reference is used instead
        # of the scheduler-driven anchor interpolation. Empty list
        # preserves the legacy single-quintic-from-anchors behaviour.
        self._phase_overrides: list = []
        # FK-mode wiring (used by reference_source='joint_space_fk').
        # Each FK phase carries q_seq + dq_seg + frame_swing + n_away
        # in its override dict.
        self._model = model
        self._data = pin.Data(model) if model is not None else None
        # One-shot DeprecationWarning gate for set_swing_orientation
        # under FK mode.
        self._warned_set_swing_orientation_fk = False

    def set_swing_orientation(self, R_start: np.ndarray) -> None:
        """Set the tool rotation at swing release for SLERP interpolation.

        Under FK mode (any phase override has ``use_fk=True``), this is
        a no-op: rotation comes from FK on the smoothed q-sequence by
        construction (plan §2.6 eq. 8d). A one-shot
        ``DeprecationWarning`` is emitted on the first call after
        entering FK mode.
        """
        if any(ov.get('use_fk', False) for ov in self._phase_overrides):
            if not self._warned_set_swing_orientation_fk:
                import warnings
                warnings.warn(
                    "set_swing_orientation is a no-op under FK mode "
                    "(rotation comes from FK on the smoothed q-sequence).",
                    DeprecationWarning, stacklevel=2,
                )
                self._warned_set_swing_orientation_fk = True
            return
        self._R_start = R_start.copy()

    def add_phase(
        self,
        t_start: float,
        t_end: float,
        p_ee_start: np.ndarray,
        R_ee_start: np.ndarray,
        p_ee_end: np.ndarray,
        R_ee_end: np.ndarray,
        swing_arm: str,
        p_ee_mid: Optional[np.ndarray] = None,
        R_ee_mid: Optional[np.ndarray] = None,
        t_mid: Optional[float] = None,
        q_seq: Optional[list] = None,
        frame_swing: Optional[int] = None,
        n_away: Optional[np.ndarray] = None,
    ) -> None:
        """Register an explicit EE-reference override for one phase.

        The override is consulted by ``reference_at`` before the
        scheduler-driven anchor interpolation. Used by sim_loop to
        install a mid-waypoint between the swing start and end EE
        poses (T15_step2_path_geometry.md §7.3 Option B).

        Single-quintic mode (``p_ee_mid``/``R_ee_mid``/``t_mid`` all
        omitted): position quintic + clearance bump + delayed-cosine
        SLERP, identical shape to the scheduler-driven path. The
        bump and rotation profiles use the same `early_finish_fraction`
        and `bump_peak_tau` settings as the SwingPlanner instance.

        Piecewise mode (all three mid-waypoint args provided): two
        position-quintic segments through (start, mid, end) with
        v=0 at each waypoint; two SLERP segments through the same
        three rotations with ω=0 at each waypoint; one continuous
        clearance bump that spans the full ``[t_start, t_end]``
        window unchanged (the bump's shape is independent of the
        manipulability-driven mid-waypoint).

        Parameters
        ----------
        t_start, t_end : float — phase timing.
        p_ee_start, p_ee_end : (3,) — EE positions at the endpoints.
        R_ee_start, R_ee_end : (3,3) — EE orientations at the endpoints.
        swing_arm : 'a' or 'b' — which arm the override applies to.
        p_ee_mid, R_ee_mid : (3,)/(3,3), optional — mid-waypoint EE pose.
        t_mid : float, optional — time at the mid-waypoint; must
            satisfy ``t_start < t_mid < t_end``.
        q_seq : list[(nq,) ndarray], optional
            Smoothed q-sequence from
            ``crawlbot.planning.constrained_geodesic.smoothed_constrained_geodesic``.
            When provided (alongside ``frame_swing``), enables FK mode:
            the swing-EE reference is derived by FK on the interpolated
            q(τ) plus the additive clearance bump (plan §2.6, §2.7,
            §2.9). Requires ``model`` at SwingPlanner construction.
        frame_swing : int, optional
            Pinocchio frame ID of the swing EE for the FK lookup.
            Required when ``q_seq`` is given.
        n_away : (3,) ndarray, optional
            Override for the away-normal unit vector used by the
            additive clearance bump. Defaults to the planner-level
            ``self.away_normal``.
        """
        # FK mode: short-circuit to a dedicated override entry.
        if q_seq is not None:
            if self._model is None:
                raise RuntimeError(
                    "SwingPlanner: q_seq passed to add_phase but the planner "
                    "was not constructed with model — pass it to "
                    "SwingPlanner.__init__ to enable FK mode.")
            if frame_swing is None:
                raise ValueError(
                    "frame_swing is required when q_seq is provided")
            from crawlbot.planning.constrained_geodesic import (
                precompute_segment_tangents,
            )
            q_seq_copy = [np.asarray(q, dtype=float).copy() for q in q_seq]
            dq_seg = precompute_segment_tangents(self._model, q_seq_copy)
            n_away_use = (np.asarray(n_away, dtype=float).copy()
                          if n_away is not None else self.away_normal.copy())
            n_away_use = n_away_use / np.linalg.norm(n_away_use)
            self._phase_overrides.append({
                't_start': float(t_start),
                't_end': float(t_end),
                'p_ee_start': np.asarray(p_ee_start, dtype=float).copy(),
                'R_ee_start': np.asarray(R_ee_start, dtype=float).copy(),
                'p_ee_end': np.asarray(p_ee_end, dtype=float).copy(),
                'R_ee_end': np.asarray(R_ee_end, dtype=float).copy(),
                'swing_arm': swing_arm,
                'piecewise': False,
                'p_ee_mid': None, 'R_ee_mid': None, 't_mid': None,
                'use_fk': True,
                'q_seq': q_seq_copy,
                'dq_seg': dq_seg,
                'frame_swing': int(frame_swing),
                'n_away': n_away_use,
                'T': float(t_end - t_start),
            })
            return

        piecewise = (p_ee_mid is not None) and (R_ee_mid is not None)
        if piecewise:
            if t_mid is None:
                raise ValueError(
                    "t_mid is required when p_ee_mid/R_ee_mid given")
            if not (t_start < t_mid < t_end):
                raise ValueError(
                    f"t_mid={t_mid} must satisfy {t_start} < t_mid < {t_end}")
        elif (p_ee_mid is not None) ^ (R_ee_mid is not None):
            raise ValueError(
                "p_ee_mid and R_ee_mid must be provided together "
                "(piecewise) or both omitted (single-quintic)")

        self._phase_overrides.append({
            't_start': float(t_start),
            't_end': float(t_end),
            'p_ee_start': np.asarray(p_ee_start, dtype=float).copy(),
            'R_ee_start': np.asarray(R_ee_start, dtype=float).copy(),
            'p_ee_end': np.asarray(p_ee_end, dtype=float).copy(),
            'R_ee_end': np.asarray(R_ee_end, dtype=float).copy(),
            'swing_arm': swing_arm,
            'p_ee_mid': (np.asarray(p_ee_mid, dtype=float).copy()
                         if piecewise else None),
            'R_ee_mid': (np.asarray(R_ee_mid, dtype=float).copy()
                         if piecewise else None),
            't_mid': float(t_mid) if piecewise else None,
            'piecewise': piecewise,
        })

    def clear_phase_overrides(self) -> None:
        """Clear any registered phase overrides (return to scheduler-driven)."""
        self._phase_overrides = []

    def _override_reference_at(self, t: float, ov: dict) -> 'SwingReference':
        """Compute the SwingReference from an override (single, piecewise,
        or FK)."""
        if ov.get('use_fk', False):
            return self._override_reference_at_fk(t, ov)
        T = ov['t_end'] - ov['t_start']
        T_eff = T * self.early_finish_fraction
        # Single-phase τ for the bump (continuous across full phase).
        tau_phase = float(np.clip((t - ov['t_start']) / T_eff, 0.0, 1.0))
        bump = self._bump(tau_phase)
        bump_dot = self._bump_dot(tau_phase) / T_eff
        bump_ddot = self._bump_ddot(tau_phase) / (T_eff * T_eff)
        n = self.away_normal

        if not ov['piecewise']:
            dp = ov['p_ee_end'] - ov['p_ee_start']
            s = self._quintic(tau_phase)
            s_dot = self._quintic_dot(tau_phase) / T_eff
            s_ddot = self._quintic_ddot(tau_phase) / (T_eff * T_eff)
            p_ee = ov['p_ee_start'] + dp * s + self.clearance * n * bump
            v_ee = dp * s_dot + self.clearance * n * bump_dot
            a_ee = dp * s_ddot + self.clearance * n * bump_ddot
            # SLERP in orientation
            tau_d = self.rotation_delay_ratio
            sigma_r = self._delayed_cosine(tau_phase, tau_d)
            sigma_r_dot = self._delayed_cosine_dot(tau_phase, tau_d) / T_eff
            sigma_r_ddot = self._delayed_cosine_ddot(tau_phase, tau_d) / (T_eff * T_eff)
            dR = ov['R_ee_start'].T @ ov['R_ee_end']
            omega_total = pin.log3(dR)
            R_ee = ov['R_ee_start'] @ pin.exp3(sigma_r * omega_total)
            omega_ee = R_ee @ (sigma_r_dot * omega_total)
            alpha_ee = R_ee @ (sigma_r_ddot * omega_total)
        else:
            # Piecewise quintic in position; piecewise SLERP in rotation;
            # bump uses single-phase tau (continuous).
            t_mid = ov['t_mid']
            if t <= t_mid:
                seg_T = t_mid - ov['t_start']
                seg_T_eff = seg_T  # ff applies only to seg-2 tail
                seg_tau = float(np.clip((t - ov['t_start']) / seg_T, 0.0, 1.0))
                p0 = ov['p_ee_start']; p1 = ov['p_ee_mid']
                R0 = ov['R_ee_start']; R1 = ov['R_ee_mid']
            else:
                seg_T = ov['t_end'] - t_mid
                seg_T_eff = seg_T * self.early_finish_fraction
                seg_tau = float(np.clip((t - t_mid) / seg_T_eff, 0.0, 1.0))
                p0 = ov['p_ee_mid']; p1 = ov['p_ee_end']
                R0 = ov['R_ee_mid']; R1 = ov['R_ee_end']
            dp = p1 - p0
            s = self._quintic(seg_tau)
            s_dot = self._quintic_dot(seg_tau) / seg_T_eff
            s_ddot = self._quintic_ddot(seg_tau) / (seg_T_eff * seg_T_eff)
            p_ee_pos = p0 + dp * s
            v_ee_lin = dp * s_dot
            a_ee_lin = dp * s_ddot
            # Apply continuous bump on full-phase τ
            p_ee = p_ee_pos + self.clearance * n * bump
            v_ee = v_ee_lin + self.clearance * n * bump_dot
            a_ee = a_ee_lin + self.clearance * n * bump_ddot
            # Per-segment SLERP with v=0 at endpoints (quintic timing).
            dR_seg = R0.T @ R1
            omega_total_seg = pin.log3(dR_seg)
            sigma_r = self._quintic(seg_tau)
            sigma_r_dot = self._quintic_dot(seg_tau) / seg_T_eff
            sigma_r_ddot = self._quintic_ddot(seg_tau) / (seg_T_eff * seg_T_eff)
            R_ee = R0 @ pin.exp3(sigma_r * omega_total_seg)
            omega_ee = R_ee @ (sigma_r_dot * omega_total_seg)
            alpha_ee = R_ee @ (sigma_r_ddot * omega_total_seg)

        return SwingReference(
            p_ee=p_ee, v_ee=v_ee, a_ee=a_ee,
            R_ee=R_ee, omega_ee=omega_ee, alpha_ee=alpha_ee,
            swing_arm=ov['swing_arm'],
            is_swinging=True,
            phase_progress=tau_phase)

    def _override_reference_at_fk(self, t: float, ov: dict) -> 'SwingReference':
        """FK-mode swing reference (plan §2.6, §2.7, §2.9).

        Position / rotation / linear+angular twist / linear+angular
        acceleration of the swing EE at parameter τ along the smoothed
        q-sequence, plus the additive clearance bump on the linear
        component only (rotation untouched).
        """
        from crawlbot.planning.constrained_geodesic import (
            frame_reference_at_tau,
        )
        T = ov['T']
        tau = float(np.clip((t - ov['t_start']) / T, 0.0, 1.0))
        # FK pose + LWA twist + LWA accel from the smoothed q-seq.
        p_fk, R, v6_fk, a6_fk = frame_reference_at_tau(
            self._model, self._data,
            ov['q_seq'], ov['dq_seg'],
            ov['frame_swing'],
            tau=tau, T_phase=T,
        )
        # Additive bump on linear position (eq. 12).
        n = ov['n_away']
        c = self.clearance
        b = self._bump(tau)
        bd = self._bump_dot(tau)
        bdd = self._bump_ddot(tau)
        p_ee = p_fk + c * n * b
        v_ee = v6_fk[0:3] + (c * n * bd) / T
        a_ee = a6_fk[0:3] + (c * n * bdd) / (T * T)
        omega_ee = v6_fk[3:6]
        alpha_ee = a6_fk[3:6]
        return SwingReference(
            p_ee=p_ee, v_ee=v_ee, a_ee=a_ee,
            R_ee=R, omega_ee=omega_ee, alpha_ee=alpha_ee,
            swing_arm=ov['swing_arm'],
            is_swinging=True,
            phase_progress=tau,
        )

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
        # Phase-override path (mid-waypoint reshape, T15_step2_path
        # _geometry.md §7.3 Option B). Consulted before the
        # scheduler-driven path.
        for ov in self._phase_overrides:
            if ov['t_start'] - 1e-6 <= t <= ov['t_end'] + 1e-6:
                return self._override_reference_at(t, ov)

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
