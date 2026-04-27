"""
TorsoPlanner — Generates 6D torso + CoM reference trajectories.

Instead of tracking a static CoM, the torso advances during swing using
the stance arm as an inverted manipulator. This eliminates the CoM vs EE
conflict that prevents docking under tight torque limits.

The planner also derives a CoM reference from the torso trajectory:

    r_com(t) = p_torso(t) + R_torso(t) · δ_com(s(t))

where δ_com is the CoM offset in the torso body frame, interpolated
between start and end configurations to account for arm reconfiguration.
This CoM reference feeds the NMPC centroidal planner so it generates
momentum-feasible contact wrenches consistent with the actual motion.

Structure-frame operation
-------------------------
All trajectories are stored and returned in the **structure body frame**.
Since mujoco_to_pinocchio() now expresses the torso state relative to
the structure, the planner works entirely in this frame without any
world-frame reconstruction.  This eliminates Fix 3 and all the
p_struct/R_struct/v_struct/omega_struct arguments that were previously
needed.

Usage:
    planner = TorsoPlanner()
    planner.add_phase(t0, tf, p0, R0, pf, Rf,
                      delta_com_start, delta_com_end)
    ref = planner.reference_at(t)
"""

import numpy as np
import pinocchio as pin
from dataclasses import dataclass
from typing import Optional


@dataclass
class TorsoReference:
    """Torso 6D reference at a given time instant (structure frame)."""
    p: np.ndarray          # (3,) position
    R: np.ndarray          # (3,3) rotation matrix
    v: np.ndarray          # (6,) twist [linear(3), angular(3)]
    a: np.ndarray          # (6,) acceleration [linear(3), angular(3)]


@dataclass
class ComReference:
    """CoM reference derived from torso trajectory (structure frame)."""
    r_com: np.ndarray      # (3,) position
    v_com: np.ndarray      # (3,) velocity


class TorsoPlanner:
    """Plan 6D torso + CoM trajectories synchronized with locomotion.

    All quantities are in the structure body frame.
    """

    def __init__(
        self,
        model: Optional[pin.Model] = None,
        frame_torso: Optional[int] = None,
    ):
        """Construct a TorsoPlanner.

        Parameters
        ----------
        model : pin.Model, optional
            Pinocchio model. Required for ``reference_source =
            'joint_space_fk'`` mode (FK-on-smoothed-q-sequence path).
            Legacy task-space SLERP mode does not use it.
        frame_torso : int, optional
            Pinocchio frame ID of the torso body. Required alongside
            ``model`` for FK mode.
        """
        self._phases = []
        self._hold_p = None       # (3,) torso position in structure frame
        self._hold_R = None       # (3,3) torso rotation in structure frame
        self._hold_com = None     # (3,) CoM position in structure frame
        # M5: torso inertia about its CoM in torso body frame. Used by
        # l_com_reference_at(t) to produce the NMPC angular momentum
        # reference L_com_ref(t) = R(t) * I_body * R(t)^T * omega_ref(t).
        # None → L_com_ref returns zero (backwards compatible).
        self._I_torso_body = None  # (3,3)
        # FK-mode wiring (used by reference_source='joint_space_fk').
        self._model = model
        self._frame_torso = frame_torso
        self._data = pin.Data(model) if model is not None else None
        # One-shot DeprecationWarning gate for set_torso_inertia
        # under FK mode.
        self._warned_set_torso_inertia_fk = False

    # ── Public API ────────────────────────────────────────────────────────

    def set_torso_inertia(self, I_body: np.ndarray) -> None:
        """Set the torso inertia tensor about its CoM, in the torso
        body frame. Usually extracted from Pinocchio:

            I_body = robot.model.inertias[1].inertia   # (3,3)

        This enables l_com_reference_at(t) to produce a meaningful
        momentum feedforward for the NMPC cost `w_L * ||L - L_ref||²`.

        Under FK mode (any phase has ``use_fk=True``), this is a
        no-op: ``l_com_reference_at`` uses the full-body centroidal
        momentum from ``pin.computeCentroidalMomentum`` instead of
        the torso-only formula. A one-shot ``DeprecationWarning`` is
        emitted on the first call after entering FK mode.
        """
        if any(p.get('use_fk', False) for p in self._phases):
            if not self._warned_set_torso_inertia_fk:
                import warnings
                warnings.warn(
                    "set_torso_inertia is a no-op under FK mode "
                    "(l_com_reference_at uses pin.computeCentroidalMomentum "
                    "instead of the torso-only inertia formula).",
                    DeprecationWarning, stacklevel=2,
                )
                self._warned_set_torso_inertia_fk = True
            return
        self._I_torso_body = np.asarray(I_body, dtype=float).reshape(3, 3)

    def set_hold(self, p: np.ndarray, R: np.ndarray,
                 r_com: Optional[np.ndarray] = None):
        """Set a static hold reference (DS phase or before swing starts).

        Parameters
        ----------
        p, R : (3,), (3,3)
            Torso position and rotation in structure frame.
        r_com : (3,), optional
            CoM position in structure frame.
        """
        self._hold_p = p.copy()
        self._hold_R = R.copy()
        self._hold_com = r_com.copy() if r_com is not None else None

    def set_from_waypoints(
        self,
        t_start: float,
        t_end: float,
        torso_wps: list,
        com_wps: list,
    ):
        """Build trajectory from IK-derived waypoint sequence.

        Creates piecewise quintic phases between consecutive waypoints.
        The last waypoint becomes the hold reference for times beyond t_end.

        Parameters
        ----------
        t_start, t_end : float
            Start and end times of the full trajectory.
        torso_wps : list of (p, R) tuples
            Torso position (3,) and rotation (3,3) at each waypoint.
        com_wps : list of ndarray (3,)
            CoM positions at each waypoint.
        """
        self.clear_phases()
        n = len(torso_wps)
        if n < 2:
            if n == 1:
                p, R = torso_wps[0]
                self.set_hold(p, R, r_com=com_wps[0])
            return

        dt = (t_end - t_start) / (n - 1)

        for i in range(n - 1):
            p0, R0 = torso_wps[i]
            p1, R1 = torso_wps[i + 1]
            delta0 = R0.T @ (com_wps[i] - p0)
            delta1 = R1.T @ (com_wps[i + 1] - p1)
            self.add_phase(
                t_start + i * dt, t_start + (i + 1) * dt,
                p0, R0, p1, R1,
                delta_com_start=delta0, delta_com_end=delta1)

        # Hold at the FIRST waypoint for times before the trajectory,
        # and at the LAST waypoint for times after.
        # set_hold is the fallback for t outside all phases.
        # Before the trajectory starts (DS phase), hold at the start.
        p0, R0 = torso_wps[0]
        self._hold_p = p0.copy()
        self._hold_R = R0.copy()
        self._hold_com = com_wps[0].copy()
        # After the trajectory, the last phase's p_end becomes the implicit hold
        # (reference_at falls through to _hold_reference which uses _hold_p).
        # We'll update _hold to the end after the trajectory is done.
        # For now, the piecewise phases handle the interpolation, and
        # _hold_reference handles times outside phases (= before t_start).

    def add_phase(self, t_start: float, t_end: float,
                  p_start: np.ndarray, R_start: np.ndarray,
                  p_end: np.ndarray, R_end: np.ndarray,
                  delta_com_start: Optional[np.ndarray] = None,
                  delta_com_end: Optional[np.ndarray] = None,
                  early_finish_fraction: float = 1.0,
                  p_mid: Optional[np.ndarray] = None,
                  R_mid: Optional[np.ndarray] = None,
                  t_mid: Optional[float] = None,
                  delta_com_mid: Optional[np.ndarray] = None,
                  q_seq: Optional[list] = None):
        """Add a trajectory phase (all coordinates in structure frame).

        Parameters
        ----------
        t_start, t_end : float
            Phase timing.
        p_start, p_end : ndarray (3,)
            Torso positions in structure frame.
        R_start, R_end : ndarray (3,3)
            Torso orientations in structure frame.
        delta_com_start : ndarray (3,), optional
            CoM offset in torso body frame at start config.
        delta_com_end : ndarray (3,), optional
            CoM offset in torso body frame at end config.
        early_finish_fraction : float in (0, 1], default 1.0
            M7 change (B): fraction of the phase window over which the
            torso trajectory is interpolated. 1.0 → the profile fills
            the full [t_start, t_end] window (legacy behaviour).
            Values < 1 (e.g. 0.7) compress the profile so the torso
            arrives at p_end, R_end at t_start + ff·(t_end - t_start)
            and then HOLDS static for the remaining (1-ff) fraction.
            The swing-arm's planner is unaffected, so this decouples
            the torso's completion time from the swing's — giving the
            swing a stable base during its precision-approach tail.
        p_mid, R_mid : ndarray, optional
            Mid-waypoint torso pose for piecewise quintic. If both are
            provided (along with t_mid), generates two consecutive
            quintic segments (t_start → t_mid, t_mid → t_end), each
            with v=0 and a=0 at its endpoints. Per
            T15_step2_path_geometry.md §7.3 Option B.
        t_mid : float, optional
            Time at the mid-waypoint. Must satisfy
            t_start < t_mid < t_end. Required when p_mid/R_mid given.
        delta_com_mid : ndarray (3,), optional
            CoM offset at mid-waypoint (body frame). If None and
            piecewise mode is active, ``delta_com_mid`` is linearly
            interpolated from (delta_com_start, delta_com_end) at the
            time fraction (t_mid - t_start) / (t_end - t_start).
        q_seq : list[(nq,) ndarray], optional
            Smoothed q-sequence from
            ``crawlbot.planning.constrained_geodesic.smoothed_constrained_geodesic``.
            When provided, the phase becomes an FK-from-smoothed-q
            phase: torso (and CoM) references are derived by FK on
            the interpolated q(τ), with v/a from per-segment
            finite-differences (plan §2.3, §2.4). The legacy
            (p_start, R_start, p_end, R_end, *_mid) arguments are
            recorded for diagnostic / hold-fallback use but not
            consulted in this path. Requires ``model`` and
            ``frame_torso`` to have been passed at construction.
        """
        if q_seq is not None:
            return self._add_phase_fk(
                t_start, t_end, p_start, R_start, p_end, R_end, q_seq)

        ff = float(early_finish_fraction)
        if not (0.0 < ff <= 1.0):
            raise ValueError(
                f"early_finish_fraction must be in (0, 1], got {ff}")
        # Decide single- vs piecewise-quintic mode.
        piecewise = (p_mid is not None) and (R_mid is not None)
        if piecewise:
            if t_mid is None:
                raise ValueError(
                    "t_mid is required when p_mid/R_mid are provided")
            if not (t_start < t_mid < t_end):
                raise ValueError(
                    f"t_mid={t_mid} must satisfy {t_start} < t_mid < {t_end}")
        elif (p_mid is not None) ^ (R_mid is not None):
            raise ValueError(
                "p_mid and R_mid must be provided together (piecewise) or "
                "both omitted (single-quintic)")

        duration = t_end - t_start
        effective_duration = duration * ff

        if not piecewise:
            # Legacy single-quintic phase — byte-identical behavior.
            self._phases.append({
                't_start': t_start, 't_end': t_end,
                'p_start': p_start.copy(), 'R_start': R_start.copy(),
                'p_end':   p_end.copy(),   'R_end':   R_end.copy(),
                'duration': duration,
                'effective_duration': effective_duration,
                'early_finish_fraction': ff,
                'delta_com_start': delta_com_start.copy() if delta_com_start is not None else None,
                'delta_com_end':   delta_com_end.copy()   if delta_com_end   is not None else None,
            })
            return

        # Piecewise quintic: emit two phases through (start, mid, end).
        # Each segment uses v=0, a=0 at its endpoints by construction
        # of the quintic time scaling. Continuity at t_mid is C0 in (p, R)
        # but velocity is zero on both sides (continuous, derivative
        # discontinuous in the next derivative — same as the single-quintic
        # at its endpoints).
        # ``early_finish_fraction`` is applied to the FULL phase window
        # only at the second segment: the second segment compresses by
        # ff and then holds. The first segment uses ff=1.0 (fills its
        # window completely). This matches the original semantics of
        # "torso arrives early at p_end" while keeping the mid-waypoint
        # exact at t_mid.

        # Interpolate delta_com at the mid-waypoint if not provided.
        if delta_com_mid is None and (
                delta_com_start is not None and delta_com_end is not None):
            f = (t_mid - t_start) / (t_end - t_start)
            delta_com_mid_eff = (1 - f) * delta_com_start + f * delta_com_end
        else:
            delta_com_mid_eff = (
                delta_com_mid.copy() if delta_com_mid is not None else None)

        # Segment 1: t_start → t_mid, ff=1 (full segment quintic).
        seg1_dur = t_mid - t_start
        self._phases.append({
            't_start': t_start, 't_end': t_mid,
            'p_start': p_start.copy(), 'R_start': R_start.copy(),
            'p_end':   p_mid.copy(),   'R_end':   R_mid.copy(),
            'duration': seg1_dur,
            'effective_duration': seg1_dur,
            'early_finish_fraction': 1.0,
            'delta_com_start': delta_com_start.copy() if delta_com_start is not None else None,
            'delta_com_end':   delta_com_mid_eff.copy() if delta_com_mid_eff is not None else None,
        })
        # Segment 2: t_mid → t_end, ff applies (compress + hold tail).
        seg2_dur = t_end - t_mid
        self._phases.append({
            't_start': t_mid, 't_end': t_end,
            'p_start': p_mid.copy(), 'R_start': R_mid.copy(),
            'p_end':   p_end.copy(), 'R_end':   R_end.copy(),
            'duration': seg2_dur,
            'effective_duration': seg2_dur * ff,
            'early_finish_fraction': ff,
            'delta_com_start': delta_com_mid_eff.copy() if delta_com_mid_eff is not None else None,
            'delta_com_end':   delta_com_end.copy() if delta_com_end is not None else None,
        })

    def clear_phases(self):
        self._phases = []

    # ── FK-mode helpers ───────────────────────────────────────────────────

    def _add_phase_fk(self, t_start, t_end, p_start, R_start,
                      p_end, R_end, q_seq):
        """FK-mode add_phase: cache the smoothed q-sequence and per-
        segment tangents. The legacy task-space args are kept for
        the hold-fallback in case t falls outside any phase, but
        are otherwise unused.
        """
        if self._model is None or self._frame_torso is None:
            raise RuntimeError(
                "TorsoPlanner: q_seq passed to add_phase but the planner "
                "was not constructed with model/frame_torso — pass these "
                "to TorsoPlanner.__init__ to enable FK mode.")
        from crawlbot.planning.constrained_geodesic import (
            precompute_segment_tangents,
        )
        q_seq_copy = [np.asarray(q, dtype=float).copy() for q in q_seq]
        dq_seg = precompute_segment_tangents(self._model, q_seq_copy)
        self._phases.append({
            't_start': t_start, 't_end': t_end,
            'p_start': p_start.copy(), 'R_start': R_start.copy(),
            'p_end': p_end.copy(), 'R_end': R_end.copy(),
            'use_fk': True,
            'q_seq': q_seq_copy,
            'dq_seg': dq_seg,
            'n_tau': len(q_seq_copy),
            'T': float(t_end - t_start),
            # Legacy diagnostic keys read by the run-script's
            # _print_phase_sync_report — preserve schema parity.
            'duration': float(t_end - t_start),
            'effective_duration': float(t_end - t_start),
            'early_finish_fraction': 1.0,
        })

    def _reference_at_fk(self, phase, t: float) -> TorsoReference:
        """FK-mode reference: extract pose, twist, accel via FK on
        the smoothed q-sequence at τ = (t - t_start) / T_phase.
        """
        from crawlbot.planning.constrained_geodesic import (
            frame_reference_at_tau,
        )
        T = phase['T']
        tau = float(np.clip((t - phase['t_start']) / T, 0.0, 1.0))
        p, R, v6, a6 = frame_reference_at_tau(
            self._model, self._data,
            phase['q_seq'], phase['dq_seg'],
            self._frame_torso,
            tau=tau, T_phase=T,
        )
        return TorsoReference(p=p, R=R, v=v6, a=a6)

    def _com_reference_at_fk(self, phase, t: float) -> ComReference:
        """FK-mode CoM reference: full-body CoM from q(τ).

        Computes r_com via ``pin.centerOfMass`` and v_com via
        ``pin.computeCentroidalMomentum`` (data.vcom[0] after the
        call) — both consistent with the same q(τ), v_real(τ) used
        by ``_reference_at_fk`` and ``_l_com_reference_at_fk``.
        """
        from crawlbot.planning.constrained_geodesic import q_v_real_at_tau
        T = phase['T']
        tau = float(np.clip((t - phase['t_start']) / T, 0.0, 1.0))
        q_tau, v_real = q_v_real_at_tau(
            self._model, phase['q_seq'], phase['dq_seg'],
            tau=tau, T_phase=T,
        )
        pin.centerOfMass(self._model, self._data, q_tau, v_real)
        r_com = self._data.com[0].copy()
        v_com = self._data.vcom[0].copy()
        return ComReference(r_com=r_com, v_com=v_com)

    def _l_com_reference_at_fk(self, phase, t: float) -> np.ndarray:
        """FK-mode L_com reference: full-body centroidal angular
        momentum from q(τ), v_real(τ).

        Replaces the legacy torso-only ``L = R·I_torso·R^T·ω``
        formula (which had a documented ~20% limb-contribution
        error) with ``pin.computeCentroidalMomentum(q, v).vector[3:6]``,
        the exact full-body angular momentum about the CoM in the
        Pinocchio world frame (= structure-local frame).
        """
        from crawlbot.planning.constrained_geodesic import q_v_real_at_tau
        T = phase['T']
        tau = float(np.clip((t - phase['t_start']) / T, 0.0, 1.0))
        q_tau, v_real = q_v_real_at_tau(
            self._model, phase['q_seq'], phase['dq_seg'],
            tau=tau, T_phase=T,
        )
        pin.computeCentroidalMomentum(self._model, self._data, q_tau, v_real)
        return self._data.hg.vector[3:6].copy()

    # ── Reference query API ──────────────────────────────────────────────

    def reference_at(self, t: float) -> TorsoReference:
        """Compute 6D torso reference at time t in structure frame."""
        for phase in self._phases:
            if phase['t_start'] - 1e-6 <= t <= phase['t_end'] + 1e-6:
                if phase.get('use_fk', False):
                    return self._reference_at_fk(phase, t)
                return self._interpolate_phase(t, phase)

        # Outside all phases: hold
        return self._hold_reference()

    def com_reference_at(self, t: float) -> ComReference:
        """Compute CoM reference derived from torso trajectory (structure frame).

        Legacy mode (delta_com_start/end on the phase):
            r_com(t) = p_torso(t) + R_torso(t) · δ_com(s(t))
            v_com = v_torso_lin + ω_torso × (R·δ) + R·δ̇

        FK mode (q_seq on the phase):
            r_com(t) = pin.centerOfMass(model, data, q(τ))
            v_com   = data.vcom[0]  (after computeCentroidalMomentum)
        """
        for phase in self._phases:
            if phase['t_start'] - 1e-6 <= t <= phase['t_end'] + 1e-6:
                if phase.get('use_fk', False):
                    return self._com_reference_at_fk(phase, t)
                return self._interpolate_com(t, phase)

        # Outside phases: hold
        if self._hold_com is not None:
            return ComReference(r_com=self._hold_com.copy(), v_com=np.zeros(3))

        # Fallback: use torso position (no CoM offset data)
        tref = self.reference_at(t)
        return ComReference(r_com=tref.p.copy(), v_com=tref.v[:3].copy())

    def l_com_reference_at(self, t: float) -> np.ndarray:
        """M5 / M7: angular-momentum reference for the NMPC cost.

        FK mode (q_seq on the active phase): full-body centroidal
        angular momentum from ``pin.computeCentroidalMomentum(q, v)``,
        exact for the smoothed-geodesic reference (no limb-contribution
        approximation).

        Legacy mode (no q_seq): torso-only approximation per spec §5.3:

            L_com_ref(t) = I_torso^com(t) · omega_ref(t)

        where I_torso^com is the torso inertia about the torso CoM and
        omega_ref comes from the SLERP derivative. Transported to the
        structure frame via R(t):

            I_world(t) = R(t) · I_body · R(t)^T
            L_com_ref(t) = I_world(t) · omega_ref(t)

        Limitations of the legacy mode:
          - Ignores the limb contribution to centroidal angular momentum
            (~20 % error) — absorbed by the NMPC's feedback term
            `w_L ||L_com - L_com_ref||²`.
          - Returns zero if no torso inertia has been set
            (backwards-compatible with the M3 stub).

        Parameters
        ----------
        t : float
            Query time.

        Returns
        -------
        L_com_ref : (3,) ndarray
            Angular momentum reference in the structure frame.
        """
        # FK mode: full-body centroidal momentum if the active phase
        # carries a smoothed q-sequence.
        for phase in self._phases:
            if phase['t_start'] - 1e-6 <= t <= phase['t_end'] + 1e-6:
                if phase.get('use_fk', False):
                    return self._l_com_reference_at_fk(phase, t)
                break
        # Legacy mode (no q_seq): torso-only formula.
        if self._I_torso_body is None:
            return np.zeros(3)
        tref = self.reference_at(t)
        # tref.v = [v_lin(3), omega(3)]; tref.R is the current rotation.
        omega = tref.v[3:6]
        if np.linalg.norm(omega) < 1e-12:
            return np.zeros(3)
        R = tref.R
        I_world = R @ self._I_torso_body @ R.T
        return I_world @ omega

    # ── Internal helpers ──────────────────────────────────────────────────

    def _hold_reference(self) -> TorsoReference:
        """Return hold reference in structure frame."""
        if self._hold_p is None:
            if self._phases:
                last = self._phases[-1]
                return TorsoReference(
                    p=last['p_end'].copy(), R=last['R_end'].copy(),
                    v=np.zeros(6), a=np.zeros(6))
            return TorsoReference(p=np.zeros(3), R=np.eye(3),
                                  v=np.zeros(6), a=np.zeros(6))

        return TorsoReference(
            p=self._hold_p.copy(), R=self._hold_R.copy(),
            v=np.zeros(6), a=np.zeros(6))

    def _trapezoidal_params(self, t: float, phase: dict):
        """Compute smooth trapezoidal velocity profile parameters.

        Uses half-cosine ramps for C¹-continuous acceleration (no
        discontinuities at ramp/cruise transitions). The velocity
        profile is:
            ramp-up:   v(tau) = v_cruise * 0.5 * (1 - cos(pi * tau / ramp))
            cruise:    v(tau) = v_cruise
            ramp-down: v(tau) = v_cruise * 0.5 * (1 + cos(pi * (tau - 1 + ramp) / ramp))

        Returns (tau, s, ds, dds) matching the quintic interface:
            s   : position fraction [0, 1]
            ds  : ds/dt [1/s]
            dds : d²s/dt² [1/s²]

        M7 change (B): time is scaled by the phase's `effective_duration`
        rather than its `duration`. When effective_duration < duration,
        the profile finishes early (at t = t_start + effective_duration)
        and tau is clipped to 1 thereafter — producing a static HOLD on
        the torso reference through the rest of the phase window.
        """
        T = phase.get('effective_duration', phase['duration'])
        tau = np.clip((t - phase['t_start']) / T, 0.0, 1.0)
        # M7 v20: ramp = 0.2 gives 20/60/20 split — ramp-up τ∈[0, 0.2],
        # cruise τ∈[0.2, 0.8], ramp-down τ∈[0.8, 1.0]. During cruise
        # a_torso_ff ≡ 0, freeing the actuator budget for EE tracking;
        # planned-δ mapping (v19) continues to supply feedforward
        # compensation against arm-induced base drift through v_b_ref.
        ramp = 0.20  # fraction of total time for each ramp

        # Cruise velocity such that total displacement = 1:
        # Area = ramp*v_c/2 + (1-2*ramp)*v_c + ramp*v_c/2 = (1-ramp)*v_c = 1
        v_c = 1.0 / (1.0 - ramp)
        pi = np.pi

        if tau < ramp:
            # Ramp up: half-cosine velocity profile
            phi = pi * tau / ramp
            s = v_c * (tau / 2.0 - ramp / (2.0 * pi) * np.sin(phi))
            ds = v_c * 0.5 * (1.0 - np.cos(phi)) / T
            dds = v_c * pi / (2.0 * ramp) * np.sin(phi) / (T**2)
        elif tau < 1.0 - ramp:
            # Cruise: constant velocity, zero acceleration
            s_ramp = v_c * ramp / 2.0  # area under ramp-up
            s = s_ramp + v_c * (tau - ramp)
            ds = v_c / T
            dds = 0.0
        else:
            # Ramp down: mirror of ramp-up
            tau_d = 1.0 - tau
            phi = pi * tau_d / ramp
            s_tail = v_c * (tau_d / 2.0 - ramp / (2.0 * pi) * np.sin(phi))
            s = 1.0 - s_tail
            ds = v_c * 0.5 * (1.0 - np.cos(phi)) / T
            dds = -v_c * pi / (2.0 * ramp) * np.sin(phi) / (T**2)

        return tau, s, ds, dds

    def _profile_params(self, t: float, phase: dict):
        """Compute time-scaling parameters (quintic, restored in v21).

        v18: switched to quintic from trapezoidal. v20: briefly back to
        trap (no closed-loop effect since SS linear is via mapping).
        v21: quintic restored — SS angular reference needs continuous
        a_ff; the cruise-phase shaping is now handled at the preplanner
        level (CoM acceleration constraint) which flows through the
        mapping into the torso linear reference.
        """
        return self._quintic_params(t, phase)

    def _quintic_params(self, t: float, phase: dict):
        """Compute quintic time scaling parameters.

        M7 change (B): uses `effective_duration` so the quintic finishes
        at t_start + effective_duration (<= t_end) and then HOLDS.
        """
        T = phase.get('effective_duration', phase['duration'])
        tau = np.clip((t - phase['t_start']) / T, 0.0, 1.0)
        s   = 10*tau**3  - 15*tau**4  + 6*tau**5
        ds  = (30*tau**2 - 60*tau**3  + 30*tau**4) / T
        dds = (60*tau    - 180*tau**2 + 120*tau**3) / (T**2)
        return tau, s, ds, dds

    def _interpolate_phase(self, t: float, phase: dict) -> TorsoReference:
        """Interpolate trajectory phase in structure frame."""
        _, s, ds, dds = self._profile_params(t, phase)

        dp    = phase['p_end'] - phase['p_start']
        p     = phase['p_start'] + s * dp
        v_lin = ds  * dp
        a_lin = dds * dp

        R0          = phase['R_start']
        dR          = R0.T @ phase['R_end']
        omega_total = pin.log3(dR)

        R       = R0 @ pin.exp3(s * omega_total)
        omega_f = R @ (ds  * omega_total)
        alpha_f = R @ (dds * omega_total)

        return TorsoReference(
            p=p, R=R,
            v=np.concatenate([v_lin, omega_f]),
            a=np.concatenate([a_lin, alpha_f]))

    def _interpolate_com(self, t: float, phase: dict) -> ComReference:
        """Derive CoM reference from torso trajectory + interpolated δ_com."""
        _, s, ds, _ = self._profile_params(t, phase)

        d0 = phase['delta_com_start']
        d1 = phase['delta_com_end']

        ref = self._interpolate_phase(t, phase)

        if d0 is None or d1 is None:
            return ComReference(r_com=ref.p.copy(), v_com=ref.v[:3].copy())

        delta     = (1 - s) * d0 + s * d1
        delta_dot = ds * (d1 - d0)

        R     = ref.R
        omega = ref.v[3:6]

        r_com = ref.p + R @ delta
        Rd    = R @ delta
        v_com = ref.v[:3] + np.cross(omega, Rd) + R @ delta_dot

        return ComReference(r_com=r_com, v_com=v_com)
