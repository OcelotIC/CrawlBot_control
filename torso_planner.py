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

Structure-frame support (Fix 3)
--------------------------------
When the floating structure drifts in the inertial world frame, a trajectory
planned in world coordinates becomes inconsistent with the robot's actual
reachable workspace (all anchors move rigidly with the structure). To handle
this, the planner can store trajectories in the **structure body frame** and
reconstruct world-frame references at query time using the live structure pose.

Planning (world → structure frame, called once per step):
    p_s   = R_s0ᵀ · (p_world − p_s0)
    R_s   = R_s0ᵀ · R_world

Query (structure → world frame, called at 100 Hz):
    p_world(t) = p_s(t) + R_s(t) · p_s(τ)
    R_world(t) = R_s(t) · R_s(τ)
    v_lin(t)   = v_s(t) + ω_s(t) × (R_s · p_s_frame) + R_s · v_s_frame(τ)
    ω_world(t) = ω_s(t) + R_s(t) · ω_s_frame(τ)

This is fully backward-compatible: when no structure pose is supplied the
planner behaves as before (world-frame trajectories, structure = identity).

Usage:
    planner = TorsoPlanner()

    # With structure-frame support (recommended):
    planner.add_phase(t0, tf, p0, R0, pf, Rf,
                      delta_com_start, delta_com_end,
                      p_struct=p_s0, R_struct=R_s0)
    ref = planner.reference_at(t,
                               p_struct=p_s_cur, R_struct=R_s_cur,
                               v_struct=v_s_cur, omega_struct=omega_s_cur)

    # Backward-compatible (world frame):
    planner.add_phase(t0, tf, p0, R0, pf, Rf)
    ref = planner.reference_at(t)
"""

import numpy as np
import pinocchio as pin
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class TorsoReference:
    """Torso 6D reference at a given time instant."""
    p: np.ndarray          # (3,) position
    R: np.ndarray          # (3,3) rotation matrix
    v: np.ndarray          # (6,) twist [linear(3), angular(3)]
    a: np.ndarray          # (6,) acceleration [linear(3), angular(3)]


@dataclass
class ComReference:
    """CoM reference derived from torso trajectory."""
    r_com: np.ndarray      # (3,) position
    v_com: np.ndarray      # (3,) velocity


class TorsoPlanner:
    """Plan 6D torso + CoM trajectories synchronized with locomotion.

    The CoM reference is derived from the torso pose plus an offset
    δ_com(s) that is interpolated in the torso body frame between
    the start and end dock configurations.

    Trajectories can optionally be stored in the structure body frame so that
    they automatically follow structure drift (see module docstring).
    """

    def __init__(self):
        self._phases = []
        # Hold state (structure frame when _hold_in_struct=True)
        self._hold_p_s = None       # (3,) torso position in structure frame
        self._hold_R_s = None       # (3,3) torso rotation in structure frame
        self._hold_com_s = None     # (3,) CoM position in structure frame
        self._hold_in_struct = False

    # ── Public API ────────────────────────────────────────────────────────

    def set_hold(self, p: np.ndarray, R: np.ndarray,
                 r_com: Optional[np.ndarray] = None,
                 p_struct: Optional[np.ndarray] = None,
                 R_struct: Optional[np.ndarray] = None):
        """Set a static hold reference (DS phase or before swing starts).

        Parameters
        ----------
        p, R : (3,), (3,3)
            Torso position and rotation in world frame.
        r_com : (3,), optional
            CoM position in world frame.
        p_struct, R_struct : (3,), (3,3), optional
            Structure pose in world frame at planning time.  When provided,
            the hold is stored in the structure body frame so that it tracks
            structure drift at query time.
        """
        if p_struct is not None and R_struct is not None:
            self._hold_p_s = R_struct.T @ (p - p_struct)
            self._hold_R_s = R_struct.T @ R
            if r_com is not None:
                self._hold_com_s = R_struct.T @ (r_com - p_struct)
            else:
                self._hold_com_s = None
            self._hold_in_struct = True
        else:
            self._hold_p_s = p.copy()
            self._hold_R_s = R.copy()
            self._hold_com_s = r_com.copy() if r_com is not None else None
            self._hold_in_struct = False

    def add_phase(self, t_start: float, t_end: float,
                  p_start: np.ndarray, R_start: np.ndarray,
                  p_end: np.ndarray, R_end: np.ndarray,
                  delta_com_start: Optional[np.ndarray] = None,
                  delta_com_end: Optional[np.ndarray] = None,
                  p_struct: Optional[np.ndarray] = None,
                  R_struct: Optional[np.ndarray] = None):
        """Add a trajectory phase.

        Parameters
        ----------
        t_start, t_end : float
            Phase timing.
        p_start, p_end : ndarray (3,)
            Torso positions in world frame.
        R_start, R_end : ndarray (3,3)
            Torso orientations in world frame.
        delta_com_start : ndarray (3,), optional
            CoM offset in torso body frame at start config.
            Computed as R_start.T @ (r_com_start - p_start).
        delta_com_end : ndarray (3,), optional
            CoM offset in torso body frame at end config.
            Computed as R_end.T @ (r_com_end - p_end).
        p_struct, R_struct : ndarray (3,), (3,3), optional
            Structure pose in world frame at planning time.  When provided,
            the trajectory is stored internally in the structure body frame.
        """
        if p_struct is not None and R_struct is not None:
            # Convert to structure frame
            p_s = R_struct.T @ (p_start - p_struct)
            R_s = R_struct.T @ R_start
            p_e = R_struct.T @ (p_end   - p_struct)
            R_e = R_struct.T @ R_end
            in_struct = True
        else:
            p_s, R_s = p_start.copy(), R_start.copy()
            p_e, R_e = p_end.copy(),   R_end.copy()
            in_struct = False

        self._phases.append({
            't_start': t_start, 't_end': t_end,
            'p_start': p_s, 'R_start': R_s,
            'p_end':   p_e, 'R_end':   R_e,
            'duration': t_end - t_start,
            'delta_com_start': delta_com_start.copy() if delta_com_start is not None else None,
            'delta_com_end':   delta_com_end.copy()   if delta_com_end   is not None else None,
            'in_struct': in_struct,
        })

    def clear_phases(self):
        self._phases = []

    def reference_at(self, t: float,
                     p_struct: Optional[np.ndarray] = None,
                     R_struct: Optional[np.ndarray] = None,
                     v_struct: Optional[np.ndarray] = None,
                     omega_struct: Optional[np.ndarray] = None) -> TorsoReference:
        """Compute 6D torso reference at time t in world frame.

        Parameters
        ----------
        t : float
            Query time.
        p_struct, R_struct : (3,), (3,3), optional
            Current structure pose in world frame.  Required when the phase
            was stored in structure frame (p_struct passed to add_phase).
        v_struct : (3,), optional
            Current structure linear velocity in world frame [m/s].
        omega_struct : (3,), optional
            Current structure angular velocity in world frame [rad/s].
        """
        for phase in self._phases:
            if phase['t_start'] - 1e-6 <= t <= phase['t_end'] + 1e-6:
                ref_s = self._interpolate_phase(t, phase)
                if phase['in_struct'] and p_struct is not None and R_struct is not None:
                    return self._struct_to_world(
                        ref_s, p_struct, R_struct,
                        v_struct  if v_struct     is not None else np.zeros(3),
                        omega_struct if omega_struct is not None else np.zeros(3))
                return ref_s

        # Outside all phases: hold
        return self._hold_reference(p_struct, R_struct, v_struct, omega_struct)

    def com_reference_at(self, t: float,
                         p_struct: Optional[np.ndarray] = None,
                         R_struct: Optional[np.ndarray] = None,
                         v_struct: Optional[np.ndarray] = None,
                         omega_struct: Optional[np.ndarray] = None) -> ComReference:
        """Compute CoM reference derived from torso trajectory.

        r_com(t) = p_torso(t) + R_torso(t) · δ_com(s(t))

        The velocity is obtained by chain rule:
        v_com = v_torso_lin + ω_torso × (R·δ) + R·δ̇

        Structure-frame arguments mirror those of reference_at.
        """
        for phase in self._phases:
            if phase['t_start'] - 1e-6 <= t <= phase['t_end'] + 1e-6:
                return self._interpolate_com(
                    t, phase, p_struct, R_struct, v_struct, omega_struct)

        # Outside phases: hold
        if self._hold_com_s is not None:
            if self._hold_in_struct and p_struct is not None and R_struct is not None:
                r_com_w = p_struct + R_struct @ self._hold_com_s
                v_s = v_struct if v_struct is not None else np.zeros(3)
                om  = omega_struct if omega_struct is not None else np.zeros(3)
                v_com_w = v_s + np.cross(om, R_struct @ self._hold_com_s)
                return ComReference(r_com=r_com_w, v_com=v_com_w)
            else:
                return ComReference(r_com=self._hold_com_s.copy(), v_com=np.zeros(3))

        # Fallback: use torso position (no CoM offset data)
        tref = self.reference_at(t, p_struct, R_struct, v_struct, omega_struct)
        return ComReference(r_com=tref.p.copy(), v_com=tref.v[:3].copy())

    # ── Internal helpers ──────────────────────────────────────────────────

    def _hold_reference(self, p_struct, R_struct, v_struct, omega_struct) -> TorsoReference:
        """Return hold reference, applying structure transform if needed."""
        if self._hold_p_s is None:
            if self._phases:
                last = self._phases[-1]
                ref_s = TorsoReference(
                    p=last['p_end'].copy(), R=last['R_end'].copy(),
                    v=np.zeros(6), a=np.zeros(6))
                if last['in_struct'] and p_struct is not None and R_struct is not None:
                    return self._struct_to_world(
                        ref_s, p_struct, R_struct,
                        v_struct      if v_struct      is not None else np.zeros(3),
                        omega_struct  if omega_struct  is not None else np.zeros(3))
                return ref_s
            return TorsoReference(p=np.zeros(3), R=np.eye(3), v=np.zeros(6), a=np.zeros(6))

        ref_s = TorsoReference(
            p=self._hold_p_s.copy(), R=self._hold_R_s.copy(),
            v=np.zeros(6), a=np.zeros(6))

        if self._hold_in_struct and p_struct is not None and R_struct is not None:
            return self._struct_to_world(
                ref_s, p_struct, R_struct,
                v_struct      if v_struct      is not None else np.zeros(3),
                omega_struct  if omega_struct  is not None else np.zeros(3))
        return ref_s

    @staticmethod
    def _struct_to_world(ref_s: TorsoReference,
                         p_struct: np.ndarray, R_struct: np.ndarray,
                         v_struct: np.ndarray, omega_struct: np.ndarray) -> TorsoReference:
        """Convert a structure-frame TorsoReference to world frame.

        World-frame quantities:
            p_w   = p_s + R_s · p_sf
            R_w   = R_s · R_sf
            v_lin = v_s + ω_s × (R_s · p_sf) + R_s · v_sf_lin
            ω_w   = ω_s + R_s · ω_sf
            a_lin = a_s + α_s × (R_s·p_sf) + ω_s×(ω_s×(R_s·p_sf))
                        + 2 ω_s × (R_s·v_sf_lin) + R_s·a_sf_lin
            α_w   = α_s + R_s · α_sf  (α_s ≈ 0, ignored)

        Note: structure acceleration (a_s, α_s) is assumed small and
        ignored — valid for the slow drift regime of a floating platform.
        """
        p_sf  = ref_s.p             # torso pos in structure frame
        R_sf  = ref_s.R             # torso rot in structure frame
        v_sf  = ref_s.v             # [v_lin(3), omega(3)] in structure frame
        a_sf  = ref_s.a             # [a_lin(3), alpha(3)] in structure frame

        # Position / orientation
        Rsp   = R_struct @ p_sf
        p_w   = p_struct + Rsp
        R_w   = R_struct @ R_sf

        # Linear velocity: v_s + ω_s × (R_s·p_sf) + R_s·v_sf_lin
        v_lin_w = v_struct + np.cross(omega_struct, Rsp) + R_struct @ v_sf[:3]

        # Angular velocity: ω_s + R_s·ω_sf
        omega_w = omega_struct + R_struct @ v_sf[3:]

        # Linear acceleration (structure accel neglected):
        # a_sf_lin carried through + centripetal + Coriolis
        a_lin_w = (R_struct @ a_sf[:3]
                   + np.cross(omega_struct, np.cross(omega_struct, Rsp))
                   + 2.0 * np.cross(omega_struct, R_struct @ v_sf[:3]))

        # Angular acceleration: R_s·α_sf (structure α neglected)
        alpha_w = R_struct @ a_sf[3:]

        return TorsoReference(
            p=p_w,
            R=R_w,
            v=np.concatenate([v_lin_w, omega_w]),
            a=np.concatenate([a_lin_w, alpha_w]))

    def _quintic_params(self, t: float, phase: dict):
        """Compute quintic time scaling parameters."""
        T = phase['duration']
        tau = np.clip((t - phase['t_start']) / T, 0.0, 1.0)
        s   = 10*tau**3  - 15*tau**4  + 6*tau**5
        ds  = (30*tau**2 - 60*tau**3  + 30*tau**4) / T
        dds = (60*tau    - 180*tau**2 + 120*tau**3) / (T**2)
        return tau, s, ds, dds

    def _interpolate_phase(self, t: float, phase: dict) -> TorsoReference:
        """Interpolate quintic in the phase's native frame (struct or world)."""
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

    def _interpolate_com(self, t: float, phase: dict,
                         p_struct, R_struct, v_struct, omega_struct) -> ComReference:
        """Derive CoM reference from torso trajectory + interpolated δ_com."""
        _, s, ds, _ = self._quintic_params(t, phase)

        d0 = phase['delta_com_start']
        d1 = phase['delta_com_end']

        # Get torso reference (in native frame of the phase)
        ref_s = self._interpolate_phase(t, phase)

        if d0 is None or d1 is None:
            # No CoM offset data: fall back to torso position
            if phase['in_struct'] and p_struct is not None and R_struct is not None:
                ref_w = self._struct_to_world(
                    ref_s, p_struct, R_struct,
                    v_struct      if v_struct      is not None else np.zeros(3),
                    omega_struct  if omega_struct  is not None else np.zeros(3))
                return ComReference(r_com=ref_w.p.copy(), v_com=ref_w.v[:3].copy())
            return ComReference(r_com=ref_s.p.copy(), v_com=ref_s.v[:3].copy())

        # δ_com is always stored in torso body frame — independent of struct frame
        delta     = (1 - s) * d0 + s * d1      # δ(s) in torso frame
        delta_dot = ds * (d1 - d0)              # dδ/dt

        # Convert torso reference to world frame first
        if phase['in_struct'] and p_struct is not None and R_struct is not None:
            ref_w = self._struct_to_world(
                ref_s, p_struct, R_struct,
                v_struct      if v_struct      is not None else np.zeros(3),
                omega_struct  if omega_struct  is not None else np.zeros(3))
        else:
            ref_w = ref_s

        R     = ref_w.R
        omega = ref_w.v[3:6]          # world-frame angular velocity of torso

        # r_com = p_torso + R_torso · δ
        r_com = ref_w.p + R @ delta

        # v_com = v_torso_lin + ω_torso × (R·δ) + R·δ̇
        Rd    = R @ delta
        v_com = ref_w.v[:3] + np.cross(omega, Rd) + R @ delta_dot

        return ComReference(r_com=r_com, v_com=v_com)
