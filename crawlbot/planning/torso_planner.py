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
        """
        T = phase['duration']
        tau = np.clip((t - phase['t_start']) / T, 0.0, 1.0)
        ramp = 0.35  # fraction of total time for each ramp

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
        """Compute time-scaling parameters using trapezoidal profile."""
        return self._trapezoidal_params(t, phase)

    def _quintic_params(self, t: float, phase: dict):
        """Compute quintic time scaling parameters."""
        T = phase['duration']
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
