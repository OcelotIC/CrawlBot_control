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

    def __init__(self):
        self._phases = []
        self._hold_p = None       # (3,) torso position in structure frame
        self._hold_R = None       # (3,3) torso rotation in structure frame
        self._hold_com = None     # (3,) CoM position in structure frame

    # ── Public API ────────────────────────────────────────────────────────

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

    def add_phase(self, t_start: float, t_end: float,
                  p_start: np.ndarray, R_start: np.ndarray,
                  p_end: np.ndarray, R_end: np.ndarray,
                  delta_com_start: Optional[np.ndarray] = None,
                  delta_com_end: Optional[np.ndarray] = None):
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
        """
        self._phases.append({
            't_start': t_start, 't_end': t_end,
            'p_start': p_start.copy(), 'R_start': R_start.copy(),
            'p_end':   p_end.copy(),   'R_end':   R_end.copy(),
            'duration': t_end - t_start,
            'delta_com_start': delta_com_start.copy() if delta_com_start is not None else None,
            'delta_com_end':   delta_com_end.copy()   if delta_com_end   is not None else None,
        })

    def clear_phases(self):
        self._phases = []

    def reference_at(self, t: float) -> TorsoReference:
        """Compute 6D torso reference at time t in structure frame."""
        for phase in self._phases:
            if phase['t_start'] - 1e-6 <= t <= phase['t_end'] + 1e-6:
                return self._interpolate_phase(t, phase)

        # Outside all phases: hold
        return self._hold_reference()

    def com_reference_at(self, t: float) -> ComReference:
        """Compute CoM reference derived from torso trajectory (structure frame).

        r_com(t) = p_torso(t) + R_torso(t) · δ_com(s(t))
        v_com = v_torso_lin + ω_torso × (R·δ) + R·δ̇
        """
        for phase in self._phases:
            if phase['t_start'] - 1e-6 <= t <= phase['t_end'] + 1e-6:
                return self._interpolate_com(t, phase)

        # Outside phases: hold
        if self._hold_com is not None:
            return ComReference(r_com=self._hold_com.copy(), v_com=np.zeros(3))

        # Fallback: use torso position (no CoM offset data)
        tref = self.reference_at(t)
        return ComReference(r_com=tref.p.copy(), v_com=tref.v[:3].copy())

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

    def _quintic_params(self, t: float, phase: dict):
        """Compute quintic time scaling parameters."""
        T = phase['duration']
        tau = np.clip((t - phase['t_start']) / T, 0.0, 1.0)
        s   = 10*tau**3  - 15*tau**4  + 6*tau**5
        ds  = (30*tau**2 - 60*tau**3  + 30*tau**4) / T
        dds = (60*tau    - 180*tau**2 + 120*tau**3) / (T**2)
        return tau, s, ds, dds

    def _interpolate_phase(self, t: float, phase: dict) -> TorsoReference:
        """Interpolate quintic in structure frame."""
        _, s, ds, dds = self._quintic_params(t, phase)

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
        _, s, ds, _ = self._quintic_params(t, phase)

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
