"""
TorsoReferenceShaper — the torso reference the whole-body QP tracks, built from
the planner's torso reference ``tr`` and, where it applies, the CoM→torso
δ-mapping of the NMPC CoM plan.

It GENERATES a reference; it does not command. In the ROS 2 layout it belongs on
the reference side, next to the planners — hence a module of its own, out of
``WholeBodyController.track`` (refactor/sim-loop-split, C2). Moved as is: same
expressions, same order.

Paths, in order of precedence:

    mapping bypass (SS, cfg.mapping_bypass_in_ss)  linear ref frozen at the
                                                   SS-entry torso position
    δ-mapping (SS or DS, use_m2_stack, NOT two-task SS)
        r_b = (m/m_b)·r_com_ref − δ(q)/m_b,  δ, δ̇ cached once per NMPC tick
        (F-RATE); F-SAT caps the per-tick increment in SS; the post-dock DS
        blend (Option A) eases the linear ref from the SS-entry pose
    raw planner reference (two-task SS)           tr.p / tr.v / tr.a

then the ``freeze_ref`` and ``pure_pd`` diagnostic overrides. See
docs/crawlbot/control/torso_reference.md for which paths the canonical takes.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TorsoRef:
    """6-D torso reference for the QP: position ``p`` (3), rotation ``R``
    (3×3), twist ``v`` (6: linear, angular), feedforward acceleration ``a``
    (6). Structure frame."""
    p: np.ndarray
    R: np.ndarray
    v: np.ndarray
    a: np.ndarray


class TorsoReferenceShaper:
    """Mapping-layer state + the torso-reference shaping of one QP sub-step."""

    def __init__(self, cfg, mapping, diag):
        self._cfg = cfg
        self._mapping = mapping
        self._diag = diag
        # M7 EE-bisection follow-up: torso linear position at SS entry
        # (set in _setup_torso_for_step). Read by shape() when
        # cfg.mapping_bypass_in_ss is True; otherwise unused.
        self._ss_entry_p_torso: Optional[np.ndarray] = None
        # Option A (T12 fix, 2026-04-22): post-dock blend state for
        # the DS torso position reference. Populated at weld
        # activation; cleared on SS entry. See
        # cfg.ds_ramp_duration_s and Misc/reports/architecture/M7_T12_MEMO.md §5.
        self._ds_ramp_t_start: Optional[float] = None
        self._ds_ramp_p_start: Optional[np.ndarray] = None
        # _diag_freeze_ref: first-sample r_b_ref / R_b_ref held by the hook.
        self._diag_frozen_r_b_ref: Optional[np.ndarray] = None
        self._diag_frozen_R_b_ref: Optional[np.ndarray] = None
        # Most recent CoMToTorsoMapping δ(q) output; populated by track()
        # only on cycles where the live mapping was invoked (i.e. NOT
        # under mapping_bypass_in_ss). None otherwise.
        self._last_mapping_delta: Optional[np.ndarray] = None
        # Auxiliary: δ(q_current) computed alongside the live mapping
        # (which uses q_planned during SS) for the planned-vs-current
        # mass-distribution diagnostic. Not used in control.
        self._last_mapping_delta_current: Optional[np.ndarray] = None
        # F-RATE: cache for mapping outputs at NMPC rate (10 Hz). The
        # mapping (δ(q), δ̇(q,q̇)) is recomputed once per NMPC tick and
        # reused across the WBC sub-steps. Avoids the q_current → δ → r_b
        # → q feedback loop running at 100 Hz.
        # F-SAT: previous-cycle r_b_ref kept so per-WBC-tick increment
        # can be saturated against the physical body-acceleration bound.
        self._mapping_cache_delta: Optional[np.ndarray] = None
        self._mapping_cache_delta_dot: Optional[np.ndarray] = None
        self._last_r_b_ref_out: Optional[np.ndarray] = None
        # F-SAT telemetry: counts of cycles where the increment was
        # clipped, max clip magnitude, per-step bookkeeping.
        self._sat_total_calls: int = 0
        self._sat_clipped_calls: int = 0
        self._sat_max_clip_mm: float = 0.0

    def shape(self, tr, rs, qs, tq, phase, rp_interp, vp_interp, af):
        """The torso reference for QP sub-step ``qs`` at time ``tq``.

        ``tr`` the planner's torso reference at ``tq``; ``rs`` the robot
        state (δ(q) is computed from ``rs.q``); ``rp_interp`` / ``vp_interp``
        the NMPC CoM reference interpolated to ``tq``; ``af`` the NMPC CoM
        feedforward acceleration.
        """
        cfg = self._cfg
        if (phase == 'SS' and cfg.mapping_bypass_in_ss
                and self._ss_entry_p_torso is not None):
            # Diagnostic bypass: freeze the linear torso reference at
            # its SS-entry value; angular reference still from
            # TorsoPlanner. Mapping is not called this tick.
            p_torso_ref_used = self._ss_entry_p_torso.copy()
            v_torso_ref_used = np.concatenate([np.zeros(3), tr.v[3:6]])
            a_torso_ff_used = np.concatenate([np.zeros(3), tr.a[3:6]])
        elif (phase in ('SS', 'DS') and self._mapping is not None
                and cfg.use_m2_stack
                and not (cfg.ss_two_task_mode and phase == 'SS')):
            # Phase-2.1 two-task: in SS the torso-pose task is fed the RAW
            # TorsoPlanner quintic+SLERP (the `else` branch below, tr.p/v/a)
            # — NO CoMToTorsoMapping δ. (DS still uses the mapping.)
            af_for_mapping = np.zeros(3) if self._diag.pure_pd else af
            # Planned-vs-current diag (commit 64479ab) confirmed
            # q_current is the right q-source for the δ term. But
            # q_current at the WBC rate (100 Hz) closes a mapping
            # feedback loop that oscillates r_b_ref by up to
            # 237 mm/tick on large swings (commit 1b5b841).
            # F-RATE: recompute δ(q) and δ̇(q,q̇) ONCE per NMPC tick
            # (10 Hz). The (m_total/m_b)·rp_interp term still varies
            # smoothly at WBC rate via the existing interpolation.
            ratio = self._mapping.ratio
            m_b = self._mapping.m_b
            # CLEANUP-14: use_local_delta_mapping was False on the canonical, so
            # only the cached-delta path below ever ran. The loop-free D_local
            # variant and its flag were removed.
            if qs == 0 or self._mapping_cache_delta is None:
                self._mapping_cache_delta = np.asarray(
                    self._mapping.compute_delta(rs.q),
                    dtype=float).copy()
                self._mapping_cache_delta_dot = np.asarray(
                    self._mapping.compute_delta_dot(rs.q, rs.v),
                    dtype=float).copy()
            _delta_q = self._mapping_cache_delta
            _delta_dot = self._mapping_cache_delta_dot
            r_b_ref_m = ratio * rp_interp - _delta_q / m_b
            v_b_ref_m = ratio * vp_interp - _delta_dot / m_b
            a_b_ff_m = ratio * af_for_mapping
            # F-SAT: cap the per-WBC-tick r_b_ref increment at the
            # *planned* torso-reference velocity (|v_b_ref_m|, already
            # feasibility-bounded by the pre-planner CoM trajectory)
            # plus a jitter slack. This clips the multi-m/s cross-tick
            # δ(q_current) jitter the limiter exists for, while letting
            # the reference advance at its commanded rate. The previous
            # cap, (f_max/m_b)·dt²·2 ≈ 0.125mm/tick, was the body's
            # 2-tick *startup* distance — it throttled sustained motion,
            # so the torso reference advanced only ~0.125mm × n_ticks and
            # never reached the dock on large steps (T15 step 2: needed
            # ~590mm, old cap allowed ~200mm), stranding the swing arm.
            if self._last_r_b_ref_out is not None and phase == 'SS':
                v_ref_ff = float(np.linalg.norm(v_b_ref_m))
                threshold = ((v_ref_ff + cfg.fsat_jitter_margin)
                             * cfg.dt_qp)
                delta_rb = r_b_ref_m - self._last_r_b_ref_out
                nrm = float(np.linalg.norm(delta_rb))
                self._sat_total_calls += 1
                if nrm > threshold and nrm > 0.0:
                    r_b_ref_m = (self._last_r_b_ref_out
                                 + threshold * delta_rb / nrm)
                    self._sat_clipped_calls += 1
                    self._sat_max_clip_mm = max(
                        self._sat_max_clip_mm,
                        (nrm - threshold) * 1000.0)
            self._last_r_b_ref_out = r_b_ref_m.copy()
            # Telemetry (still records δ used by the WBC; q_current
            # comparison still computed for the planned-vs-current
            # diagnostic, unchanged).
            self._last_mapping_delta = self._mapping_cache_delta.copy()
            try:
                self._last_mapping_delta_current = np.asarray(
                    self._mapping.compute_delta(rs.q),
                    dtype=float).copy()
            except Exception:
                self._last_mapping_delta_current = None
            # Option A: post-dock blend of the DS torso linear
            # position reference from the SS-exit pose to the live
            # mapping output over cfg.ds_ramp_duration_s. Quintic
            # shape s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5.
            # Orientation reference (tr.R below) is not blended.
            if phase == 'DS':
                T_ramp = cfg.ds_ramp_duration_s
                if (T_ramp > 0.0
                        and self._ds_ramp_t_start is not None
                        and self._ds_ramp_p_start is not None):
                    tau_blend = (tq - self._ds_ramp_t_start) / T_ramp
                    if tau_blend <= 0.0:
                        r_b_ref_m = self._ds_ramp_p_start.copy()
                    elif tau_blend < 1.0:
                        s_blend = (10.0 * tau_blend ** 3
                                   - 15.0 * tau_blend ** 4
                                   + 6.0 * tau_blend ** 5)
                        r_b_ref_m = ((1.0 - s_blend) * self._ds_ramp_p_start
                                     + s_blend * r_b_ref_m)
            p_torso_ref_used = r_b_ref_m
            v_torso_ref_used = np.concatenate([v_b_ref_m, tr.v[3:6]])
            a_torso_ff_used = np.concatenate([a_b_ff_m, tr.a[3:6]])
        else:
            p_torso_ref_used = tr.p
            v_torso_ref_used = tr.v
            a_torso_ff_used = tr.a

        if self._diag.freeze_ref:
            # Freeze r_b_ref / R_b_ref to the first sample taken.
            # Used to probe PD stability at a STATIC target.
            if self._diag_frozen_r_b_ref is None:
                self._diag_frozen_r_b_ref = p_torso_ref_used.copy()
                self._diag_frozen_R_b_ref = tr.R.copy()
            p_torso_ref_used = self._diag_frozen_r_b_ref.copy()
            v_torso_ref_used = np.zeros(6)
            a_torso_ff_used = np.zeros(6)

        if self._diag.pure_pd:
            # Strip the torso feedforward entering the QP (the controller
            # strips λ_ref and a_com_ff).
            a_torso_ff_used = np.zeros(6)
        R_torso_ref_used = (self._diag_frozen_R_b_ref
                            if self._diag.freeze_ref else tr.R)
        return TorsoRef(p=p_torso_ref_used, R=R_torso_ref_used,
                        v=v_torso_ref_used, a=a_torso_ff_used)

    # ── Events from the orchestrator ──────────────────────────────────────

    def on_ss_entry(self, p_torso_entry):
        """SS entry: record the torso position and reset the post-dock DS
        blend (Option A); the next dock re-arms it."""
        self._ss_entry_p_torso = p_torso_entry.copy()
        self._ds_ramp_t_start = None
        self._ds_ramp_p_start = None

    def on_dock(self, t):
        """Weld activation: start the post-dock DS blend from the SS-entry
        torso position (Option A; see Misc/reports/architecture/M7_T12_MEMO.md §5).
        The blend endpoint is not stored — the DS torso reference recomputes
        the live mapping output each tick and blends it against the start."""
        self._ds_ramp_t_start = float(t)
        if self._ss_entry_p_torso is not None:
            self._ds_ramp_p_start = (
                self._ss_entry_p_torso.copy())
