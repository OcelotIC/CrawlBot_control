"""
SimulationLoop — the orchestrator of the closed-loop MuJoCo simulation.

Since the refactor/sim-loop-split chantier the loop is four explicit parts:

    MujocoPlant   (plant.py)               the only writer of MuJoCo state
    SensorSuite   (sensors.py)             the only reader, on the command path
    WholeBodyController (control/controller.py, control/attitude.py)
                  NMPC 10 Hz + whole-body QP 100 Hz + AOCS; measurements in,
                  commands out; references queried BY TIME (PlannerReferences)
    SimulationLoop (this file)             orchestration: planners, the gait
                                           sequence, docking, telemetry

The single simulation loop is ``_drive``: every iteration advances the active
request by one control period dt_qp (measure -> control -> plant I/O ->
plant.step), or completes it. Two request kinds:

    _DSSettle   energy-based DS settle (spec §7.1.1), NMPC bypassed
    _NMPCTick   one NMPC period: controller.plan, then n_qp_per_nmpc QP
                sub-steps (controller.track -> plant -> controller.after_step),
                then the hand-off to telemetry

The gait sequencer ``_gait_program`` is a coroutine that yields those requests
and decides the next one (DS -> [DWELL] -> planning handoff -> SS -> [HOLD] ->
dock -> ... -> trailing DS). It never steps the plant.

Phase machine per step (M7, two-phase):
    DS (double support, energy-based exit) ->
    SS (single support, T_step-synchronized trajectories) ->
    dock (d<5mm AND ori<5deg)

Reading this file
-----------------
    "how is a step sequenced?"         _gait_program()   (the former run() body)
    "what advances time?"              _drive()          — the only loop
    "what happens in one NMPC period?" _nmpc_begin / _qp_substep / _nmpc_handoff
    "how does DS settle?"              _ds_begin / _ds_tick / _ds_end
    "how is a step prepared?"          _setup_torso_for_step(), _run_preplanner()
    "where is the control law?"        NOT HERE — crawlbot/control/
    "where is X logged?"               NOT HERE — tick_logging.py

``_step`` and ``_run_ds_passivity_loop`` remain as compatibility entry points
that run one request through ``_drive``.
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import mujoco
except ImportError:
    mujoco = None

try:
    import pinocchio as pin
except ImportError:
    pin = None

from crawlbot.core.robot_interface import RobotInterface
from crawlbot.core.state_conversions import (
    pinocchio_to_mujoco, quat_wxyz_to_euler_deg)
from crawlbot.core.com_to_torso_mapping import CoMToTorsoMapping
from crawlbot.core.ik import (
    dock_configuration, dock_configuration_fixed_rotation,
    manipulability_config)
from crawlbot.planning.contact_scheduler import ContactScheduler
# LocomotionPlanner removed — CoM reference comes from TorsoPlanner
from crawlbot.planning.swing_planner import SwingPlanner
from crawlbot.planning.torso_planner import TorsoPlanner
from crawlbot.planning.coarse_preplanner import (
    CoarsePrePlanner, CoarsePrePlannerConfig, CoarsePlanResult,
)
from crawlbot.solvers.centroidal_nmpc import CentroidalNMPC, CentroidalNMPCConfig
from crawlbot.solvers.wholebody_qp import WholeBodyQP, WholeBodyQPConfig
from crawlbot.solvers.contact_phase import ContactConfig, ContactPhase
from crawlbot.aocs.force_estimator import (
    MomentumDisturbanceEstimator, EstimatorConfig)

from .config import SimConfig
from .logging import SimLog, capture_environment
from .tick_logging import TickState, TickLoggingMixin
from .plotting import plot_simulation
from .plant import MujocoPlant
from .sensors import SensorSuite
from crawlbot.control.attitude import AttitudeController
from crawlbot.control.controller import (
    ControlIntent, DiagHooks, WholeBodyController, ee_data)
# ── Simulation loop ──────────────────────────────────────────────────────────
# TickState and the two per-tick recorders (_log_ds_tick / _log_ss_tick) live in
# tick_logging.py — telemetry, separated from control. See that module's
# docstring for why, and for the three loop-owned geometry queries it calls back.

def _kinetic_energy(v_full, H_robot):
    return 0.5 * float(v_full @ H_robot @ v_full)


def _once(req):
    """A one-request sequencer: yields ``req``, returns its result."""
    return (yield req)


@dataclass
class _DSSettle:
    """Request: an energy-based DS settle (spec §7.1.1), NMPC bypassed.

    The shared dissipation engine of BOTH the setup-phase stage-2 settling
    and the inter-step DS settling. Runs the M2 QP in settle_mode +
    passivity_active, one tick per ``_drive`` iteration, until:
        1. Target met: T_kin < T_settle = 0.5·epsilon_v²·lambda_min(H)
        2. Plateau: T(k) > plateau_ratio · T(k - plateau_window)
        3. max_steps iterations.
    ``min_steps`` forces at least that many iterations before exits 1/2 —
    the inter-step call uses it to absorb the post-dock impact.
    Assumes DS (both tools welded).

    ``t_log`` / ``T_log`` (optional) receive plot samples every 5 steps at
    t = (t_log_step_offset + k)·dt_qp. The ``log_*`` fields are logging
    only: with ``log_obj`` set, each tick emits a row schema-identical to
    the SS log (``_log_ds_tick``).

    Result (sent back to the sequencer): dict with ``n_steps``, ``T_start``,
    ``T_end``, ``T_settle``, ``lambda_min``, ``exit_reason``
    ('target_met' | 'plateau' | 'max_steps').
    """
    contact_config: Any
    max_steps: int
    epsilon_v: float
    plateau_window: int = 50
    plateau_ratio: float = 0.999
    min_steps: int = 0
    fallback_Kd: float = 20.0
    t_log: Optional[list] = None
    T_log: Optional[list] = None
    t_log_step_offset: int = 0
    log_obj: Any = None
    log_step_idx: int = -1
    log_just_landed_arm: str = ''
    log_anchor_a_idx: int = -1
    log_anchor_b_idx: int = -1
    log_t_abs: float = 0.0


@dataclass
class _DSRun:
    """Progress of an active ``_DSSettle`` (next iteration index ``k``)."""
    req: _DSSettle
    lambda_min: float
    T_settle: float
    T_start: float
    hw_current: Any
    T_history: list = field(default_factory=list)
    exit_reason: str = 'max_steps'
    k: int = 0


@dataclass
class _NMPCTick:
    """Request: one NMPC period — plan, then n_qp_per_nmpc QP sub-steps.

    All quantities in structure frame. ``cc_ss`` is the tick's contact
    configuration; ``hw`` / ``L_com_prev`` the loop's carries (returned
    updated, as ``(hw, L_com_prev)``). ``passivity_hold`` activates the QP
    passivity inequality in SS (convergence hold); ``passivity_override``
    (diagnostic H_DS3), when not None, overrides the phase-based passivity
    gate.
    """
    t: float
    phase: str
    step_idx: int
    swing_arm: str
    stance_arm: str
    cc_ss: Any
    target_anchor: int
    stance_a: int
    stance_b: int
    hw: Any
    L_com_prev: Any
    log: Any
    ss_end: Optional[float] = None
    settle_mode: bool = False
    passivity_hold: bool = False
    passivity_override: Optional[bool] = None
    ds_centroidal_active: bool = False


@dataclass
class _NMPCRun:
    """Progress of an active ``_NMPCTick`` (next QP sub-step ``qs``)."""
    t: float
    phase: str
    step_idx: int
    swing_arm: str
    stance_arm: str
    stance_a: int
    stance_b: int
    target_anchor: int
    log: Any
    ss_end: Any
    settle_mode: bool
    L_com_prev: Any
    intent: Any
    plan: Any
    carry: Any
    vp: Any
    cref_r: Any
    nmpc_ok: bool
    nmpc_status_code: int
    nmpc_cost_val: float
    info_n: Any
    t_nmpc_ms: float
    t_qp_start: float
    qs: int = 0


class PlannerReferences:
    """The ``ReferenceSource`` the controller queries, over today's planners.

    The controller asks for references BY TIME only — the shape of
    ``GaitTrajectory.at(t)`` in docs/architecture/unified_planner_architecture.md
    §3.2 — and never sees TorsoPlanner / the coarse pre-planner / SwingPlanner
    or the plan-time offset. When the unified planner lands it replaces this
    adapter; the controller does not change.

    Reads the planner state of ``sim`` live (the coarse plan and its t0 are
    re-installed every step), so it holds the loop, not a snapshot.
    """

    def __init__(self, sim):
        self._sim = sim

    def com_at(self, t, settle_mode):
        """NMPC CoM reference ``(r_com_ref, v_com_ref)`` for the tick at ``t``.

        Queried at the horizon end ``t + N·dt_nmpc``; the coarse pre-planner's
        momentum-feasible trajectory overrides the TorsoPlanner CoM outside
        settle mode.
        """
        sim = self._sim
        cfg = sim.cfg

        # Torso/CoM references (structure frame — no struct pose needed)
        tref = sim.torso_planner.reference_at(t)
        # Query CoM reference at horizon end, not current time.
        # The NMPC uses a constant reference across all N horizon steps,
        # so passing the current-time reference causes systematic lag.
        t_horizon = t + cfg.nmpc_N * cfg.nmpc_dt
        cref = sim.torso_planner.com_reference_at(t_horizon)

        # M6: override the NMPC CoM reference with the coarse pre-planner
        # trajectory when it is available. Replaces the geometric CoM
        # path with a momentum-feasible one, so the NMPC tracks something
        # it can actually realize within the hw box.
        #
        # M7 change (B): the *torso* CoM reference is compressed into
        # the first `torso_early_finish_fraction` of T_step and holds
        # thereafter. The pre-planner's CoM trajectory runs over the
        # FULL T_step (matching the swing), so to stagger we rescale
        # the query time here. The torso position reference (through
        # the M5 mapping below) sees a static r_com_goal during the
        # last (1 - ff)·T_step. The swing planner is queried
        # independently on the full T_step — unchanged.
        if (sim._coarse_plan is not None) and (not settle_mode):
            tau_rel = t_horizon - sim._coarse_plan_t0
            ff = float(getattr(cfg, 'torso_early_finish_fraction', 1.0))
            T_plan = float(sim._coarse_plan.T_step)
            if 0.0 < ff < 1.0:
                # Compressed time: profile covers [0, ff·T_plan] in
                # real-time, then holds. Positions accelerate; velocity
                # scales by 1/ff during the active window, goes to 0
                # afterwards (the pre-planner's v_com[-1] ≈ 0 anyway,
                # so clamping is safe).
                if tau_rel <= ff * T_plan:
                    tau_comp = tau_rel / ff
                    rp_coarse = sim._coarse_plan.r_com_at(tau_comp)
                    vp_coarse = sim._coarse_plan.v_com_at(tau_comp) / ff
                else:
                    rp_coarse = sim._coarse_plan.r_com_at(T_plan)
                    vp_coarse = np.zeros(3)
            else:
                rp_coarse = sim._coarse_plan.r_com_at(tau_rel)
                vp_coarse = sim._coarse_plan.v_com_at(tau_rel)
            cref_r = rp_coarse
            cref_v = vp_coarse
        else:
            cref_r = cref.r_com
            cref_v = cref.v_com

        return cref_r, cref_v

    def L_com_at(self, t_mid):
        """Centroidal angular-momentum reference (TorsoPlanner feedforward)."""
        return self._sim.torso_planner.l_com_reference_at(t_mid)

    def torso_at(self, tq, phase, ss_end):
        """6-D torso reference at QP time ``tq``.

        In SS the query is capped at ``ss_end - 1 ms`` so the quintic's
        terminal pose (p_t1, v=0, a=0) holds through the post-T_step
        margin/hold window, instead of falling through to
        ``_hold_reference()`` (which holds the initial p_t0) — see
        ``TorsoPlanner.reference_at``.
        """
        if phase == 'SS' and ss_end is not None:
            tq_planner = min(tq, ss_end - 1e-3)
        else:
            tq_planner = tq
        return self._sim.torso_planner.reference_at(tq_planner)

    def swing_at(self, tq, phase, ss_end):
        """6-D swing-EE reference at QP time ``tq`` (plan-time offset and the
        SS-hold clamp applied by ``SimulationLoop._swing_query_time``)."""
        sim = self._sim
        return sim.swing_planner.reference_at(
            sim._swing_query_time(tq, phase, ss_end))


class SimulationLoop(TickLoggingMixin):
    """Closed-loop MuJoCo simulation with hierarchical NMPC+QP controller."""

    def __init__(self, mjcf_path: str, urdf_path: str,
                 config: Optional[SimConfig] = None):
        assert mujoco is not None, "mujoco package required"
        assert pin is not None, "pinocchio package required"
        self.mjcf_path = mjcf_path
        self.urdf_path = urdf_path
        self.cfg = config or SimConfig()
        self.n_qp_per_nmpc = int(round(self.cfg.dt_nmpc / self.cfg.dt_qp))

        # The MuJoCo plant — sole writer of MuJoCo state (plant.py).
        self.plant: Optional[MujocoPlant] = None
        # Measurement channels over the plant — sole reader of MuJoCo state
        # on the command path (sensors.py).
        self.sensors: Optional[SensorSuite] = None
        self.robot = None
        self.sched = None
        self.swing_planner = None
        self.torso_planner = None
        # Global flat torso-orientation reference = R_torso(t=0), structure
        # frame. Captured ONCE at the start of run() and sourced at every
        # phase (SS dock-IK / hold / swing-end, DS holds) instead of
        # re-sampling live torso state — so the orientation reference is
        # continuous by construction (eliminates the CASE-B per-phase
        # re-anchoring; see ORI_CHAIN_CONTINUITY_DIAG.md).
        self._R_torso_flat = None
        self.nmpc = None
        self.qp_ss = None
        self.plan = None
        # M6/M7: coarse pre-planner — mandatory, built in setup().
        self.preplanner: Optional[CoarsePrePlanner] = None
        # Most recent pre-planner result for the active step; None in DS.
        self._coarse_plan: Optional[CoarsePlanResult] = None
        # M7: T_step from the pre-planner for the active step.
        self._current_T_step: float = 0.0
        # Per-step planned arm-joint trajectory endpoints. Their original
        # consumer, `_planned_arm_config`, was retired in CLEANUP-34 (zero
        # callers). These four SURVIVE because the canonical driver reads them:
        # `scripts/diag_cooperative_arms.py:510-511` logs q_start/q_end per step
        # into `_step_q_end_log`. Deleting them as "stranded" broke the canonical
        # replay immediately — an AST orphan-scan over sim_loop.py alone cannot
        # see a reader that lives in scripts/.
        self._step_q_start: Optional[np.ndarray] = None
        self._step_q_end: Optional[np.ndarray] = None
        self._step_t_ss_start: float = 0.0
        self._step_T_step: float = 0.0
        # M7 / FK-on-smoothed-q: cached smoothed q-sequence + per-segment
        # tangents for the active SS step. Populated by
        # _setup_torso_for_step under cfg.reference_source='joint_space_fk';
        # consumed by the torso-reference path so the mapping sees the same
        # q(τ) the planners are tracking.
        self._step_q_seq: Optional[list] = None
        self._step_dq_seg: Optional[list] = None
        # Simulation time at which the active coarse plan was anchored
        # (so r_com_at(t - t0) gives the right reference at current time).
        self._coarse_plan_t0: float = 0.0
        # Per-step telemetry (infeasibilities, solve times, etc.)
        self._preplanner_stats = []
        # ── Diagnostic hooks (runtime-only, not config fields) ────────
        # Shared with the controller: control/controller.py DiagHooks.
        # The _diag_* names below are properties onto it.
        self.diag = DiagHooks()
        # _diag_disable_aocs: if True, force tau_w_cmd = 0 every QP sub-
        #   step (used to measure the raw robot-disturbance-induced
        #   platform drift without AOCS compensation).
        # _diag_lock_arm_joints: if True, set qvel[arm joints] = 0 after
        #   every mj_step and clear arm joint actuation (used to
        #   measure the contact/weld/MJ baseline drift with the robot
        #   "frozen").
        # _diag_pure_pd: strips ALL feedforward terms entering the QP
        #   (a_com_ff → 0, a_torso_ff → 0, λ_ref → 0) and the NMPC's
        #   L_com_ref → 0, leaving only PD feedback on r_b_ref from
        #   the mapping layer. Used to localize feedforward-injected
        #   instabilities vs. PD-loop instabilities.
        # Per-step trace of the pure-PD diagnostic (filled in _step).
        self._diag_pure_pd_trace: list = []
        # _diag_freeze_ref: keep r_b_ref / v_b_ref held at the first-
        # sample value during the run. Used to probe the PD loop's
        # stability around a FIXED torso target (no reference motion).
        # Cumulative plan-time offset from inter-step settling. The sim
        # clock `t` advances with settle time, but the ContactScheduler
        # plan's t_start fields are frozen at the nominal plan times.
        # SwingPlanner queries the plan by time via plan.phase_at(t), so
        # it must be fed `t - _t_plan_offset` to stay in sync. The torso
        # planner and coarse pre-planner already receive offset-adjusted
        # times when they are set up per-step, so they use `t` directly.
        self._t_plan_offset: float = 0.0

        # Step 2 diagnostics B+C (QP realization + mapping layer). Set
        # _step2_diag_enabled=True externally before run() to populate.
        # Each entry: dict(t, qs, c_ref, r_b_ref, p_torso, a_torso_des,
        # a_torso_qp, delta_q). Captured only during step 2 SS to limit size.
        self._step2_diag_enabled: bool = False
        self._step2_diag_log: list = []

    # MuJoCo model/data stay readable as attributes: tick_logging.py and the
    # diagnostic drivers read them. Writes go through self.plant.
    @property
    def mj_model(self):
        return self.plant.model if self.plant is not None else None

    @property
    def mj_data(self):
        return self.plant.data if self.plant is not None else None

    # F-SAT telemetry lives in the controller's mapping layer; the canonical
    # driver reads these three after run().
    @property
    def _sat_total_calls(self):
        return self.controller.torso_shaper._sat_total_calls

    @property
    def _sat_clipped_calls(self):
        return self.controller.torso_shaper._sat_clipped_calls

    @property
    def _sat_max_clip_mm(self):
        return self.controller.torso_shaper._sat_max_clip_mm

    # ── Diagnostic hooks: properties onto the shared DiagHooks record ─────
    @property
    def _diag_pure_pd(self):
        return self.diag.pure_pd

    @_diag_pure_pd.setter
    def _diag_pure_pd(self, v):
        self.diag.pure_pd = v

    @property
    def _diag_freeze_ref(self):
        return self.diag.freeze_ref

    @_diag_freeze_ref.setter
    def _diag_freeze_ref(self, v):
        self.diag.freeze_ref = v

    @property
    def _diag_disable_aocs(self):
        return self.diag.disable_aocs

    @_diag_disable_aocs.setter
    def _diag_disable_aocs(self, v):
        self.diag.disable_aocs = v

    @property
    def _diag_lock_arm_joints(self):
        return self.diag.lock_arm_joints

    @_diag_lock_arm_joints.setter
    def _diag_lock_arm_joints(self, v):
        self.diag.lock_arm_joints = v

    # ── Setup ────────────────────────────────────────────────────────────

    def setup(self, n_steps: int = 3, start_a: int = 2, start_b: int = 2,
              sequence_path: str = None):
        """Initialize all components.

        If ``sequence_path`` is provided, the gait plan is built from
        that ``.seq`` file (see ``crawlbot.planning.sequence_loader``)
        and ``n_steps`` / ``start_a`` / ``start_b`` are ignored.
        """
        cfg = self.cfg

        # MuJoCo plant (loads the MJCF, sets dt, detects the RWA)
        self.plant = MujocoPlant(self.mjcf_path, cfg.dt_qp)
        self.sensors = SensorSuite(self.plant, cfg.rwa_I_w)

        # Verify torso mass
        tid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        assert abs(self.mj_model.body_mass[tid] - 40.0) < 1.0, \
            f"Torso mass mismatch: {self.mj_model.body_mass[tid]}"
        # Cache structure principal inertia for AOCS PD model variant.
        # body_inertia is (nbody, 3) — principal moments at body CoM.
        sid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY,
                                'structure')
        self._struct_I = self.mj_model.body_inertia[sid].copy() \
            if sid >= 0 else np.array([597.0, 1493.0, 1777.0])
        # Cache initial structure attitude (wxyz quaternion) for the
        # legacy_pid_* AOCS modes: θ_s = log3(R_init.T @ R_now) gives
        # the small-angle attitude error in body frame.
        self._struct_quat_init = self.sensors.struct_quat()
        self.plant.forward()

        # Read anchor sites in world frame and convert to structure-local frame
        mj_a_world, mj_b_world = self.sensors.anchor_sites_world()
        p_s0 = self.sensors.struct_pos()
        w, x, y, z = self.sensors.struct_quat()
        R_s0 = pin.Quaternion(w, x, y, z).toRotationMatrix()
        anchors_a_local = [R_s0.T @ (a - p_s0) for a in mj_a_world]
        anchors_b_local = [R_s0.T @ (b - p_s0) for b in mj_b_world]

        # Pinocchio
        self.robot = RobotInterface(
            self.urdf_path, gravity='zero')
        self.plant.n_joints = self.robot.n_joints

        # Scheduler (anchors in structure-local frame).
        # M7: SS phases get duration=0 (placeholder); the real T_step
        # is installed per-step via plan.set_step_duration() after the
        # coarse pre-planner runs. DS uses a nominal value for initial
        # timeline queries, but the actual DS exit is energy-based.
        self.sched = ContactScheduler(
            anchors_a=anchors_a_local, anchors_b=anchors_b_local,
            dt_ds=self.cfg.dt_ds, dt_ss=0.0)
        if sequence_path is not None:
            from crawlbot.planning.sequence_loader import (
                load_sequence, plan_from_sequence)
            seq = load_sequence(sequence_path,
                                n_anchors=len(anchors_a_local))
            self.plan = plan_from_sequence(self.sched, seq)
            start_a = seq.start_a
            start_b = seq.start_b
            n_steps = len(seq.swing_targets)
            self._sequence_path = sequence_path
        else:
            self.plan = self.sched.plan_traversal(
                start_a=start_a, start_b=start_b, n_steps=n_steps)
            self._sequence_path = None

        # Swing planner (anchors already in structure frame — no transforms needed)
        self.swing_planner = SwingPlanner(
            self.sched,
            clearance=cfg.swing_clearance,
            bump_peak_tau=cfg.swing_bump_peak_tau,
            early_finish_fraction=cfg.swing_early_finish_fraction)

        # Torso planner (reconfigured per step)
        self.torso_planner = TorsoPlanner()
        # M5: provide torso body inertia so l_com_reference_at() can
        # produce a meaningful feedforward for the NMPC cost.
        # Pinocchio joint index 1 == root_joint (torso); .inertia is
        # the 3x3 principal inertia tensor at the body CoM in body frame.
        self.torso_planner.set_torso_inertia(
            self.robot.model.inertias[1].inertia)

        # Precompute manipulability-optimal configs for anchor pairs
        # used in this gait. Only compute the pairs we actually need.
        needed_pairs = set()
        for gp in self.plan.phases:
            needed_pairs.add((gp.anchor_a_idx, gp.anchor_b_idx))
        # Startup-IK regularizers (default None/0 ⇒ legacy behaviour):
        # pitch/roll-level the torso to the structure (yaw free) and
        # bias arm joints toward a natural pose via whole-system
        # null-space posture. Scoped to ONLY the (start_a, start_b)
        # pair: the leveled+posture solve is ~11× slower (the outer
        # Nelder-Mead runs many inner IKs), and only (start_a,
        # start_b) becomes q_dock_init (the actual initial state +
        # QP nominal posture). The other torso_map entries are just
        # manipulability references / IK seeds / rare fallbacks, and a
        # leveled start propagates forward through the live R_t0 of
        # the fixed-rotation per-step IK — so they stay on the fast
        # legacy path. Bounds the added setup cost to one extra solve.
        _ik_kw = dict(
            level_axis=getattr(cfg, 'ik_level_axis', None),
            q_nominal=getattr(cfg, 'ik_q_nominal', None),
            w_posture=float(getattr(cfg, 'ik_w_posture', 0.0)),
        )
        self.torso_map = {}
        for (ai, bi) in needed_pairs:
            se3_a = self.sched.anchor_se3('a', ai)
            se3_b = self.sched.anchor_se3('b', bi)
            pair_kw = _ik_kw if (ai, bi) == (start_a, start_b) else {}
            try:
                q_opt, w = manipulability_config(
                    self.robot.model, se3_a, se3_b, **pair_kw)
                self.torso_map[(ai, bi)] = q_opt
            except RuntimeError:
                pass

        # Initial configuration — M7: use the manipulability-optimized
        # torso_map entry so the setup qpos AND the QP nominal posture
        # both start at an extended-arm pose, rather than whatever
        # branch neutral-seeded IK returns. The torso_map was built
        # above for every feasible anchor pair via
        # manipulability_config(); (start_a, start_b) is a DS pair so
        # it should always be present. Fall back to a plain
        # dock_configuration only if the entry is missing.
        q_manip = self.torso_map.get((start_a, start_b))
        if q_manip is not None:
            self.q_dock_init = q_manip.copy()
        else:
            self.q_dock_init = dock_configuration(
                self.robot.model,
                self.sched.anchor_se3('a', start_a),
                self.sched.anchor_se3('b', start_b),
                **_ik_kw)

        # Constant CoM-z standoff: re-solve the INITIAL config at the
        # standoff height so the robot starts at crawl height. Otherwise
        # the z-reference ramps from the unconstrained init (CoM-z ~-0.48)
        # to the standoff dock targets (-0.35), and that startup z
        # transient competes with EE tracking and breaks docking.
        if cfg.use_com_z_standoff:
            rs_init = self.robot.update(
                self.q_dock_init, np.zeros(self.robot.model.nv))
            R_t_init = rs_init.oMf_torso.rotation.copy()
            q_z, err_z, _, _ = dock_configuration_fixed_rotation(
                self.robot.model,
                self.sched.anchor_se3('a', start_a),
                self.sched.anchor_se3('b', start_b),
                R_torso_fixed=R_t_init,
                q_init=self.q_dock_init.copy(),
                com_z_target=cfg.com_z_standoff)
            if err_z < 1e-4:
                self.q_dock_init = q_z
            else:
                print(f"  [standoff] init IK residual {err_z:.2e} >= 1e-4; "
                      f"keeping unconstrained init")

        sp = self.sensors.struct_pos()
        sq = self.sensors.struct_quat()
        mj_qpos, _ = pinocchio_to_mujoco(
            self.q_dock_init, np.zeros(self.robot.model.nv), struct_pos=sp, struct_quat=sq,
            rwa=True)
        self.plant.set_state(mj_qpos, 0.0)

        # Welds
        self.plant.build_weld_map()
        self.plant.deactivate_all_welds()
        self.plant.activate_weld('a', start_a)
        self.plant.activate_weld('b', start_b)
        self.plant.forward()

        # Initial CoM calibration (no settling yet — state is hot from the
        # weld activation, but we only need total_mass + frame IDs for
        # building the NMPC/QP, which are invariant to velocity.)
        rs0 = self.robot.update(
            *self.sensors.joint_state())

        # Site IDs
        self.plant.cache_site_ids()

        # NMPC — plans robot motion only (hw managed by AOCS independently).
        # M3: when enforce_hw_conservation is on, the NMPC adds a
        # conservation-law box constraint at every knot using
        # c_simple = h_w_0 + L_com_0 + r_com_0 × m·v_com_0.
        self.nmpc = CentroidalNMPC(CentroidalNMPCConfig(
            robot_mass=rs0.total_mass,
            N=cfg.nmpc_N, dt=cfg.nmpc_dt,
            f_max=cfg.nmpc_f_max, tau_max=cfg.nmpc_tau_max,
            L_max=cfg.L_max, tau_w_max=cfg.tau_w_max,
            p_max=cfg.nmpc_p_max,
            Wv=cfg.nmpc_Wv * np.ones(3),
            Wr=cfg.nmpc_Wr * np.ones(3),
            Wu_f=cfg.nmpc_Wu_f, Wu_tau=cfg.nmpc_Wu_tau,
            Qf_r=cfg.nmpc_Qf_r * np.ones(3),
            Qf_v=cfg.nmpc_Qf_v * np.ones(3),
            Qf_L=cfg.nmpc_Qf_L,
            enforce_hw_conservation=cfg.enforce_hw_conservation,
            h_max_tight=cfg.h_max_tight,
            w_L=cfg.w_L_nmpc,
            kappa_terminal=cfg.kappa_terminal))
        self.nmpc.build()

        # M6/M7: coarse pre-planner — mandatory. Built once; solved per step.
        # Produces T_step (step duration from the momentum envelope) and a
        # momentum-feasible CoM trajectory. The T_step_default is only a
        # bootstrap for the NLP's time grid; each solve receives the real
        # T_step from the per-step heuristic guess (see _run_preplanner).
        pre_cfg = CoarsePrePlannerConfig(
            M=cfg.preplanner_M,
            robot_mass=rs0.total_mass,
            h_max=np.asarray(cfg.h_max_tight, dtype=float).reshape(3),
            kappa_terminal=cfg.preplanner_kappa,
            f_max=cfg.preplanner_f_max,
            tau_max=cfg.preplanner_tau_max,
            tau_w_max=cfg.tau_w_max,
            w_L=cfg.preplanner_w_L,
            w_u=cfg.preplanner_w_u,
            ipopt_max_iter=cfg.preplanner_max_iter,
            a_cruise_max=cfg.preplanner_a_cruise_max,
            cruise_ramp_frac=cfg.preplanner_cruise_ramp_frac,
        )
        self.preplanner = CoarsePrePlanner(pre_cfg)
        self.preplanner.build()

        # M1/M5: CoM-to-torso mapping layer. Converts NMPC centroidal
        # outputs (r_com, v_com, a_com_ff) into torso position references
        # via the mass-weighted identity
        #   r_b_ref = (m_total/m_b) * r_com_ref - delta(q)/m_b
        # The QP then tracks this mapped torso reference instead of the
        # TorsoPlanner's raw p_torso, ensuring the torso task is
        # consistent with the momentum-feasible NMPC plan.
        self.mapping = CoMToTorsoMapping(self.robot)

        # M7: single QP variant. EXT variants (qp_ext, qp_approach) with
        # their gain scheduling and close-approach latches are removed —
        # synchronized trajectories (both torso and swing over [0, T_step])
        # bring the EE to zero velocity at the target without gain ramps.
        self.qp_ss = self._build_qp(
            cfg.ss_alpha_ee,
            cfg.ss_alpha_posture, cfg.ss_alpha_wrench,
            cfg.ss_Kp_com, cfg.ss_Kd_com,
            cfg.ss_Kp_torso, cfg.ss_Kd_torso,
            cfg.ss_Kp_ee, cfg.ss_Kd_ee,
            cfg.ss_Kp_ee_ang, cfg.ss_Kd_ee_ang)

        # H_{r/O} momentum disturbance estimator for AOCS
        self.H_estimator = MomentumDisturbanceEstimator(
            robot_mass=rs0.total_mass,
            dt=cfg.dt_qp,
            config=EstimatorConfig(
                robot_mass=rs0.total_mass,
                dt=cfg.dt_qp,
                filter_tau=cfg.aocs_filter_tau,
                include_transport=True,
            ),
        )

        # AOCS (reaction wheels) — crawlbot/control/attitude.py
        self.aocs = AttitudeController(
            cfg, self.robot, self.sensors, self.H_estimator,
            self._struct_quat_init, self._struct_I, self.diag)

        # GMO contact estimator
        from crawlbot.estimation.contact_estimator import (
            GeneralizedMomentumObserver, ContactStateMachine,
            ContactObserverConfig)
        obs_cfg = ContactObserverConfig(
            K_O=cfg.gmo_K_O, dt=cfg.dt_qp, nv=self.robot.model.nv,
            F_threshold=cfg.gmo_F_threshold,
            d_proximity=cfg.gmo_d_proximity,
            d_contact=cfg.gmo_d_contact,
            d_reset=cfg.gmo_d_reset,
            debounce_count=cfg.gmo_debounce_count)
        self.gmo = GeneralizedMomentumObserver(obs_cfg)
        self.contact_sm = ContactStateMachine(obs_cfg)

        # The controller block (NMPC + whole-body QP + AOCS) and the reference
        # source it queries by time — control/controller.py.
        self.refs = PlannerReferences(self)
        self.controller = WholeBodyController(
            cfg, self.robot, self.sensors, self.nmpc, self.qp_ss, self.mapping,
            self.aocs, self.gmo, self.diag, self.n_qp_per_nmpc)

        # ── Two-stage setup settling ──────────────────────────────────
        # Stage 1 (open-loop damping) removed - see _settle_setup.
        #          steps to absorb the weld activation impulse.
        # Stage 2: M2 QP with settle_mode + passivity_active, exit when
        #          T_kin < 0.5·epsilon_v²·lambda_min(H).
        #
        # qvel is NEVER zeroed — the passivity-constrained QP dissipates
        # momentum via joint torques, respecting dynamics.
        self._settling_log = self._settle_setup(start_a, start_b)

        print(f"[SimulationLoop] Initialized:")
        print(f"  Robot mass:     {rs0.total_mass:.1f} kg")
        print("  RWA model:      YES (3 wheels)")
        print(f"  AOCS estimator: {'H_{r/O}' if cfg.aocs_use_H_estimator else 'L_dot (legacy)'}")
        print(f"  NMPC:           {1/cfg.dt_nmpc:.0f} Hz, N={cfg.nmpc_N}")
        print(f"  QP:             {1/cfg.dt_qp:.0f} Hz, {self.n_qp_per_nmpc} per NMPC")
        print(f"  Gait:           {n_steps} step(s), "
              f"T_step from pre-planner (margin={cfg.t_ss_margin}s)")
        print(f"  Constraints:    L_max={cfg.L_max} Nms, tau_w={cfg.tau_w_max} Nm, "
              f"tau_joint={cfg.tau_max} Nm")
        print(f"  hw bounds:      [{cfg.hw_min[0]:.1f}, {cfg.hw_max[0]:.1f}] Nms")
        print(f"  Dock threshold: {cfg.weld_radius*1000:.1f} mm")
        s = self._settling_log
        T_dt = cfg.dt_qp
        settle_time = (s['stage1_steps'] + s['stage2_steps']) * T_dt
        print(f"  Settling:       stage1={s['stage1_steps']}, "
              f"stage2={s['stage2_steps']} "
              f"({settle_time:.2f}s sim, exit={s['exit_reason']})")
        print(f"                  T: {s['T_start']:.3e} -> {s['T_end']:.3e} J  "
              f"(target T_settle={s['T_settle']:.3e})")
        print(f"                  lambda_min(H)={s['lambda_min']:.4e} kg·m²")
        print(f"                  initial |v_com|={np.linalg.norm(s['initial_vcom'])*1000:.4f} mm/s, "
              f"|L_com|={np.linalg.norm(s['initial_Lcom'])*1000:.4f} mNms")

    def _settle_setup(self, start_a, start_b):
        """Two-stage setup-phase settling.

        Stage 1 — weld-snap absorption (open-loop damping):
            No QP. Purpose: dissipate the large constraint-force impulse
            from the weld activation (~300 N) which the QP cannot handle
            gracefully because the initial constraint violation is large.

        Stage 2 — passivity-constrained QP settling:
            Use the M2 QP with settle_mode=True (skips torso/EE tasks,
            adds joint velocity damping task) AND passivity_active=True
            (adds dq_j^T·τ_q + 2α·T ≤ 0 inequality). Exit when
            T_kin < T_settle = 0.5 · epsilon_v² · lambda_min(H).

        Returns
        -------
        log : dict with keys
            'stage1_steps', 'stage2_steps', 'T_start', 'T_end',
            'T_settle', 'lambda_min', 't_log', 'T_log',
            'initial_vcom', 'initial_Lcom'
        """
        cfg = self.cfg
        dt = cfg.dt_qp

        log = {
            'stage1_steps': 0, 'stage2_steps': 0,
            'T_start': 0.0, 'T_end': 0.0, 'T_settle': 0.0,
            'lambda_min': 0.0, 'exit_reason': '',
            't_log': [], 'T_log': [],
            'initial_vcom': None, 'initial_Lcom': None,
        }

        def _kinetic_energy(v_full, H_robot):
            return 0.5 * float(v_full @ H_robot @ v_full)

        # Log the initial hot state after weld activation
        pq0, pv0 = self.sensors.joint_state()
        rs0 = self.robot.update(pq0, pv0)
        T_initial = _kinetic_energy(rs0.v, rs0.H)
        log['T_start'] = T_initial

        # Stage 1 (open-loop joint-velocity damping) was removed in
        # CLEANUP-13: n_settle_damping_steps is 0 on the canonical, so the
        # loop never executed. The manipulability-optimized init places the
        # arms near weld equilibrium (no impulse to absorb) and the stage-2
        # passivity QP holds posture.

        # ── Stage 2: passivity-constrained QP ─────────────────────────
        # Delegated to the shared _run_ds_passivity_loop() helper so the
        # inter-step settle (§7.1.1) reuses the exact same dissipation
        # machinery. Pass the initial DS contact config (both anchors
        # active at their start positions).
        cc_ds_setup = self.sched.contact_config_at(0.1)
        stage2_start_step = log['stage1_steps']
        stage2_result = self._run_ds_passivity_loop(
            contact_config=cc_ds_setup,
            max_steps=cfg.n_settle_max_steps,
            epsilon_v=cfg.settle_epsilon_v,
            plateau_window=50,
            plateau_ratio=cfg.settle_plateau_ratio,
            min_steps=0,
            fallback_Kd=cfg.Kd_settle_damping,
            t_log=log['t_log'],
            T_log=log['T_log'],
            t_log_step_offset=stage2_start_step,
        )
        log['stage2_steps'] = stage2_result['n_steps']
        log['lambda_min'] = stage2_result['lambda_min']
        log['T_settle'] = stage2_result['T_settle']
        log['exit_reason'] = stage2_result['exit_reason']

        # Record final state
        self.plant.zero_ctrl()
        self.plant.forward()
        pq_end, pv_end = self.sensors.joint_state()
        rs_end = self.robot.update(pq_end, pv_end)
        log['T_end'] = _kinetic_energy(rs_end.v, rs_end.H)
        log['initial_vcom'] = rs_end.v_com.copy()
        log['initial_Lcom'] = rs_end.L_com.copy()
        # Final sample for the plot
        t_final = (log['stage1_steps'] + log['stage2_steps']) * dt
        log['t_log'].append(t_final)
        log['T_log'].append(log['T_end'])
        return log

    def _run_ds_passivity_loop(self, **kw) -> dict:
        """One energy-based DS settle, run through the single loop.

        Compatibility entry point (``_settle_setup`` uses it): builds a
        ``_DSSettle`` request from the keyword arguments and drives it with
        ``_drive``. Returns the result dict — see ``_DSSettle``.
        """
        return self._drive(_once(_DSSettle(**kw)))

    def _ds_begin(self, r):
        """Entry of a DS settle: threshold from H, h_w snapshot, ω_s history."""
        # Threshold from H at entry (stable over the small displacement
        # we expect during settling).
        pq0, pv0 = self.sensors.joint_state()
        rs0 = self.robot.update(pq0, pv0)
        eig_H = np.linalg.eigvalsh(rs0.H)
        lambda_min = float(np.min(np.abs(eig_H)))
        T_settle = 0.5 * (r.epsilon_v ** 2) * lambda_min
        T_start = _kinetic_energy(rs0.v, rs0.H)

        # DS contact config (both anchors active). Caller must pass a
        # ContactConfig whose r_contact_A/B hold the CURRENT structure-
        # frame anchor positions — these are used by compute_momentum_map
        # to build the lever arms for the hw safety constraint.
        hw_current = self.sensors.wheel_momentum()

        # The AOCS seeds its own ω_s history from the entry ω_s (⇒ ω̇_s = 0
        # on the first tick); the DS wrench feedforward needs no L_com/v_com
        # history (AOCS-FF audit).
        self.controller.begin_settle()

        return _DSRun(req=r, lambda_min=lambda_min, T_settle=T_settle,
                      T_start=T_start, hw_current=hw_current)

    def _ds_tick(self, st):
        """One DS-settle iteration k: exit checks, then one control period.

        An iteration that fires an exit returns without applying control —
        and, as the original ``for k`` loop did, counts toward ``n_steps``
        (``n = min(k + 1, max_steps)``).
        """
        r = st.req
        cfg = self.cfg
        dt = cfg.dt_qp
        k = st.k
        if k >= r.max_steps:
            return True, self._ds_end(st, k - 1)
        pq, pv = self.sensors.joint_state()
        rs = self.robot.update(pq, pv)
        T = _kinetic_energy(rs.v, rs.H)
        st.T_history.append(T)

        if (r.t_log is not None) and (r.T_log is not None) and (k % 5 == 0):
            r.t_log.append((r.t_log_step_offset + k) * dt)
            r.T_log.append(T)

        # Exits (only after min_steps)
        if k >= r.min_steps:
            if T < st.T_settle:
                st.exit_reason = 'target_met'
                return True, self._ds_end(st, k)
            if k >= r.plateau_window:
                T_old = st.T_history[k - r.plateau_window]
                if T > r.plateau_ratio * T_old:
                    st.exit_reason = 'plateau'
                    return True, self._ds_end(st, k)

        # Settle QP + inter-step AOCS — WholeBodyController.settle.
        tau, lambda_qp_sol, tau_w_applied, wheel_cmd = (
            self.controller.settle(rs, r.contact_config, st.hw_current,
                                   r.fallback_Kd))
        self.plant.apply_joint_torques(tau)
        self.plant.apply_wheel_torques(wheel_cmd)
        # NB: no diagnostic arm lock here — this loop never applied it.
        self.plant.step()

        # ── Phase-B per-tick logging (logging-only; no control change) ──
        # Emit a row schema-identical to the SS log block. NaN sentinels
        # for NMPC-side quantities (NMPC is bypassed in this settle).
        if r.log_obj is not None:
            t_abs_tick = r.log_t_abs + (k + 1) * dt
            self._log_ds_tick(
                r.log_obj, t_abs_tick,
                r.log_step_idx, r.log_just_landed_arm,
                r.log_anchor_a_idx, r.log_anchor_b_idx,
                tau, lambda_qp_sol, tau_w_applied)
        st.k = k + 1
        return False, None

    def _ds_end(self, st, k_last):
        """Exit of a DS settle: final energy and the result dict."""
        r = st.req
        n_steps_run = min(k_last + 1, r.max_steps) if r.max_steps > 0 else 0
        # Record final energy (no ctrl reset — caller manages the handoff)
        pq_end, pv_end = self.sensors.joint_state()
        rs_end = self.robot.update(pq_end, pv_end)
        T_end = _kinetic_energy(rs_end.v, rs_end.H)

        return {
            'n_steps': n_steps_run,
            'T_start': st.T_start,
            'T_end': T_end,
            'T_settle': st.T_settle,
            'lambda_min': st.lambda_min,
            'exit_reason': st.exit_reason,
        }

    def _build_qp(self, ae, ap, aw,
                   kpc, kdc, kpt, kdt, kpe, kde,
                   kpe_ang=5.0, kde_ang=3.0):
        cfg = self.cfg
        # The QP runs the Phase-2.1 two-task SS stack plus the centroidal-DS
        # tasks. The legacy CoM / torso-6D-P1 / cooperative-split / Option-D
        # / soft-CoM channels were removed in CLEANUP-6, so their weights
        # (alpha_com, alpha_torso, alpha_com_soft, r_tube, w_tube_lin,
        # alpha_torso_ang/lin, ...) are no longer passed.
        c = WholeBodyQPConfig(
            nq=self.robot.n_joints, nc_max=2, dt_qp=cfg.dt_qp,
            tau_max=cfg.tau_max * np.ones(self.robot.n_joints),
            alpha_ee=ae,
            alpha_posture=ap, alpha_wrench=aw,
            # CANONICAL-2p5 / Add-5 freeze: torque-min must stay ≳5× the
            # accel-reg floor or SS redundancy resolution degrades to a
            # step-0 dock timeout (PHASE_COPRIORITY_1000 Addendum 5).
            alpha_torque=5e0, alpha_reg=1e0,
            alpha_lambda_int=cfg.ss_alpha_lambda_int,
            ds_centroidal_mode=cfg.ds_centroidal_mode,
            ds_alpha_com=cfg.ds_alpha_com,
            ds_alpha_torso_ori=cfg.ds_alpha_torso_ori,
            ds_alpha_posture=cfg.ds_alpha_posture,
            Kp_com=np.diag([kpc]*3), Kd_com=np.diag([kdc]*3),
            # M7: torso P1 task uses uniform PD gains across all 6
            # dimensions. The legacy 0.6x angular scaling was a
            # heuristic from the era when CoM was the primary task and
            # torso was secondary; in the M2 stack the torso IS the
            # primary task, so there's no reason for angular gains to
            # be softer than linear.
            Kp_torso=np.array([kpt]*6),
            Kd_torso=np.array([kdt]*6),
            Kp_ee=kpe * np.ones(3), Kd_ee=kde * np.ones(3),
            Kp_ee_ang=kpe_ang * np.ones(3), Kd_ee_ang=kde_ang * np.ones(3),
            Kp_posture=1.0, Kd_posture=1.5,
            L_max=cfg.L_max, tau_w_max=cfg.tau_w_max,
            # NB: QP-side contact wrench bound is intentionally NOT
            # piped from cfg.nmpc_f_max. Data shows the WBC needs
            # ~460 N transiently to track a 49 N NMPC plan (step 2,
            # commit-pending diag); capping the QP at 100 N regresses
            # step 2 from 164 mm → 545 mm. The NMPC f_max bounds what
            # the planner BUDGETS for; the QP delivers what the
            # CoM/torso/EE tracking needs. The WB QP default 3000 N
            # is effectively unbounded and stays as a sanity backstop.
            alpha_passivity=cfg.alpha_passivity,
            passivity_W_budget=cfg.passivity_W_budget,
            qp_envelope_exact=cfg.qp_envelope_exact,
            ss_alpha_mom=cfg.ss_alpha_mom,
            ss_two_task_mode=cfg.ss_two_task_mode,
            alpha_torso_pose=cfg.alpha_torso_pose,
            )
        qp = WholeBodyQP(c)
        qp.set_nominal_posture(self.q_dock_init[self.robot.joints_q_slice])
        return qp

    def _gripper_distance(self, arm, anchor_idx):
        # Kept as a method: tick_logging.py calls it on self.
        return self.sensors.gripper_distance(arm, anchor_idx)

    def _gripper_speed(self, arm):
        """Swing-EE linear speed relative to the structure [m/s].

        Anchors are static in the structure frame, so this is the EE
        approach speed onto the anchor. Used as a dock-gate criterion:
        a clean dock needs low relative speed, else the weld's inelastic
        impact projection injects a momentum impulse (recoil / force
        spike). Pinocchio relative twist -> J_ee @ v gives EE velocity
        relative to the structure (LOCAL_WORLD_ALIGNED linear part).
        """
        pq, pv = self.sensors.joint_state()
        rs = self.robot.update(pq, pv)
        J_ee, _, _ = self._get_ee_data(rs, arm)
        return float(np.linalg.norm((J_ee @ pv)[0:3]))

    def _gripper_ori_err_deg(self, arm, anchor_idx):
        """Angle between the gripper frame and its target anchor frame.

        Anchor frames are Identity in the structure frame, so this
        reduces to the angle between the gripper's structure-frame
        rotation matrix and I. Returns the angle in degrees.
        """
        pq, pv = self.sensors.joint_state()
        rs = self.robot.update(pq, pv)
        _, _, oMf = self._get_ee_data(rs, arm)
        R_ee = np.asarray(oMf.rotation)
        R_tgt = np.asarray(self.sched.anchor_se3(arm, anchor_idx).rotation)
        R_err = R_ee.T @ R_tgt
        return float(np.degrees(np.linalg.norm(pin.log3(R_err))))

    def _dock_gate(self, swing_arm, target_idx, *, log=None, t=0.0,
                   step_idx=-1):
        """Evaluate the dock gate. Returns (docked, d, ori_deg, twist_norm).

        Fix C (J2 #1): the velocity criterion is the 6-D weld-relative
        twist ‖Jc·v⁻‖ < cfg.dock_twist_max (via ``sensors.weld_relative_twist``),
        not the legacy LINEAR EE speed. Pose criteria (d < weld_radius,
        ori < dock_ori_threshold_deg) are unchanged. The legacy linear
        gate is kept behind ``cfg.dock_use_6d_twist=False`` for A/B.
        Caller must have run ``mj_forward``. When ``log`` is given, one
        ``dock_gate_trace`` row is appended per evaluation. ``twist_norm``
        is always computed (for logging) even on the legacy path.
        """
        cfg = self.cfg
        d = self._gripper_distance(swing_arm, target_idx)
        ori_err_deg = self._gripper_ori_err_deg(swing_arm, target_idx)
        twist_norm = float(np.linalg.norm(
            self.sensors.weld_relative_twist(swing_arm, target_idx)))
        pos_ok = d < cfg.weld_radius
        ori_ok = ori_err_deg < cfg.dock_ori_threshold_deg
        if cfg.dock_use_6d_twist:
            vel_ok = twist_norm < cfg.dock_twist_max
        else:
            vel_ok = self._gripper_speed(swing_arm) < cfg.dock_vel_max
        docked = pos_ok and ori_ok and vel_ok
        if log is not None:
            log.dock_gate_trace.append({
                't': round(float(t), 3), 'step': int(step_idx),
                'd_mm': round(d * 1000, 3),
                'ori_deg': round(ori_err_deg, 3),
                'twist': round(twist_norm, 6),
                'fired': bool(docked)})
        return docked, d, ori_err_deg, twist_norm

    # ── Per-step planning handoff (M7) ───────────────────────────────────


    def _setup_torso_for_step(self, t_ss_start, swing_arm,
                              stance_a, stance_b, target_arm, target_idx,
                              ss_phase_idx: int):
        """Plan torso + swing trajectories for one crawling step (M7).

        Per spec §6 / HANDOFF §M7 per-step planning sequence:
          1. Compute start/end kinematic configurations (IK + δ_com).
          2. Run the coarse pre-planner → returns T_step and momentum-
             feasible CoM trajectory.
          3. On success: install T_step in the scheduler's SS phase
             (cascades timing to subsequent phases), set up the torso
             planner over [t_ss_start, t_ss_start + T_step]. Both torso
             and swing trajectories now share the same horizon.
          4. On failure: log, return step_feasible=False; the main loop
             will hold position and skip this step. No heuristic fallback
             inside sim_loop — from_heuristic is a test fixture only.

        Returns
        -------
        (q_dock, T_step, step_feasible) : (np.ndarray | None, float, bool)
        """
        cfg = self.cfg
        model = self.robot.model

        # Current robot state (structure frame)
        pq_live, pv_live = self.sensors.joint_state()
        rs_s = self.robot.update(pq_live, pv_live)
        p_t0 = rs_s.oMf_torso.translation.copy()
        R_t0 = rs_s.oMf_torso.rotation.copy()
        r_com0 = rs_s.r_com.copy()
        delta0 = R_t0.T @ (r_com0 - p_t0)

        # End configuration — M7 change A: prefer to hold torso
        # rotation at R_start ("crawl forward, don't pirouette").
        # Fall back to manipulability-optimized q_end only when the
        # fixed-rotation solution is near-singular.
        if target_arm == 'b':
            end_a, end_b = stance_a, target_idx
        else:
            end_a, end_b = target_idx, stance_b
        se3_a = self.sched.anchor_se3('a', end_a)
        se3_b = self.sched.anchor_se3('b', end_b)

        q_end = None
        ik_mode = 'manipulability'  # default label for logging
        w_fixed = float('nan')          # Yoshikawa, fixed_rotation
        w_sigma_min_fixed = float('nan')  # σ_min product, fixed_rotation
        traj_drift = float('nan')

        # q_end is produced by the fixed-rotation IK below, falling back to the
        # endpoint-only manipulability IK (IK 1 / IK 2, IK_FORMULATION.md §5-§6).
        # A third path used to sit here — the trajectory-aware IK 3 — disabled on
        # the canonical by CLEANUP-15 and retired from ik.py entirely by
        # CLEANUP-30. Do not re-add a branch for it without reviving its subject.

        if q_end is None and cfg.ik_fixed_rotation:
            try:
                q_fixed, err_fixed, w_fixed, w_sigma_min_fixed = (
                    dock_configuration_fixed_rotation(
                        model, se3_a, se3_b,
                        R_torso_fixed=self._R_torso_flat,
                        torso_pos=0.5 * (se3_a.translation + se3_b.translation),
                        q_init=pq_live.copy(),
                        com_z_target=(cfg.com_z_standoff
                                      if cfg.use_com_z_standoff else None)))
                if err_fixed < 1e-4 and w_fixed >= cfg.ik_fixed_rotation_w_min:
                    q_end = q_fixed
                    ik_mode = 'fixed_rotation'
            except Exception:
                q_end = None

        if q_end is None:
            # Fallback: manipulability-optimized configuration.
            q_end = self.torso_map.get((end_a, end_b))
            if q_end is None:
                q_end = dock_configuration(model, se3_a, se3_b)
            ik_mode = 'manipulability'

        rs_e = self.robot.update(q_end, np.zeros(self.robot.model.nv))
        p_t1 = rs_e.oMf_torso.translation.copy()
        R_t1 = rs_e.oMf_torso.rotation.copy()
        r_com1 = rs_e.r_com.copy()
        delta1 = R_t1.T @ (r_com1 - p_t1)

        # Print R_start / R_goal / rotation-angle diagnostics.
        dR = R_t0.T @ R_t1
        theta_goal = float(np.linalg.norm(pin.log3(dR)))
        dp_torso = p_t1 - p_t0
        print(
            f"  [IK] mode={ik_mode}  "
            f"||log(R_start^T R_goal)|| = {np.degrees(theta_goal):.2f} deg  "
            f"|Δp_torso| = {np.linalg.norm(dp_torso)*1000:.1f} mm  "
            f"w_a*w_b(fixed) = {w_fixed:.2e}  threshold = {cfg.ik_fixed_rotation_w_min:.2e}"
        )
        # Capture for top-level scripts
        trace = getattr(self, '_debug_ik_trace', None)
        if trace is None:
            self._debug_ik_trace = []
        self._debug_ik_trace.append({
            'step_ab_end': (int(end_a), int(end_b)),
            'mode': ik_mode,
            'R_start': R_t0.copy(),
            'R_goal': R_t1.copy(),
            'theta_deg': float(np.degrees(theta_goal)),
            'dp_mm': float(np.linalg.norm(dp_torso) * 1000),
            'w_fixed': w_fixed,                    # Yoshikawa
            'w_sigma_min_fixed': w_sigma_min_fixed,  # σ_min product
            'traj_drift': traj_drift,
        })

        # CLEANUP-18: the SwingPlanner phase-override mechanism was removed
        # entirely; reference_at() always takes the scheduler-driven gait plan.

        # 2. Run the coarse pre-planner. T_step is produced by the
        #    pre-planner from the momentum envelope; on failure we do
        #    NOT silently fall back to a heuristic — that would hide
        #    infeasible steps from the diagnostic record.
        T_step, preplan_success = self._run_preplanner(
            t_plan_start=t_ss_start,
            stance_arm='a' if target_arm == 'b' else 'b',
            stance_a=stance_a, stance_b=stance_b,
            r_com_0=r_com0, r_com_goal=r_com1,
        )
        if not preplan_success:
            # Caller handles the hold/skip decision.
            return (None, 0.0, False)

        # 3. Install T_step in the scheduler's SS phase. This updates
        #    GaitPhase.duration and cascades t_start/t_end for all
        #    subsequent phases, so the SwingPlanner (which reads
        #    gp.duration in reference_at) plans over [0, T_step] —
        #    identical to the torso planner's horizon below.
        self.plan.set_step_duration(ss_phase_idx, T_step)
        self._current_T_step = T_step

        # Option Z: reset plan-time offset at SS entry so that
        # plan_query_t(t_ss_start) aligns with plan.t_start[ss_phase_idx].
        # This absorbs both (a) unused SS runway from prior-step early
        # dock and (b) any inter-step DS-settle slack vs nominal DS
        # duration. Idempotent per SS entry.
        self._t_plan_offset = t_ss_start - self.plan.t_start[ss_phase_idx]

        # 4. Torso planner over the SAME [t_ss_start, t_ss_start + T_step].
        #    No torso_delay, no EXT extension.
        #    M7 change (B): the torso trajectory completes in
        #    `torso_early_finish_fraction · T_step` (default 0.7) and
        #    holds for the remainder. This gives the QP a STATIC torso
        #    reference during the post-singularity recovery window of
        #    (N_torso · J_ee_stance), letting PD arrest the drift
        #    accumulated during the singular window without
        #    simultaneously tracking a moving target.
        self.torso_planner.clear_phases()
        # Orientation reference = global R_flat (NOT live R_t0); position/CoM
        # held-setpoint (p_t0, r_com0) unchanged.
        self.torso_planner.set_hold(p_t0, self._R_torso_flat, r_com=r_com0)
        # Mid-waypoint args are passed only when the reshape
        # succeeded — otherwise the call falls through to the legacy
        # single-quintic add_phase signature (byte-identical to the
        # pre-Phase-4 behaviour).
        torso_phase_kwargs = dict(
            delta_com_start=delta0, delta_com_end=delta1,
            early_finish_fraction=cfg.torso_early_finish_fraction)

        # M7 / FK-on-smoothed-q reference path
        # (cfg.reference_source='joint_space_fk').
        # Compute the task-space-smoothed constrained geodesic ONCE per
        # SS step. The same q_seq is shared by both planners, ensuring
        # the torso and swing references at every τ derive from a
        # single q satisfying the stance constraint by construction.
        # Phase-0 measurement: ~0.3 s wall-clock for n_tau=21,
        # n_iter=120; eliminates the kinematically-uncoupled-refs
        # failure mode at T15 step 2 (synthesis §6, plan §2.2).
        # CLEANUP-15: reference_source is task_space on the canonical, so the
        # joint-space-FK (smoothed constrained geodesic) reference path,
        # including its swing-phase override below, never ran.

        # Orientation: R_start = R_end = R_flat (R_t1 == R_flat via the
        # R_torso_fixed=R_flat dock-IK above), so the SS orientation reference
        # is the constant global R_flat — zero DS<->SS seam, and the QP closes
        # any residual torso drift back to R_flat during the swing. Position
        # (p_t0->p_t1) and delta_com still advance as before.
        self.torso_planner.add_phase(
            t_ss_start, t_ss_start + T_step,
            p_t0, self._R_torso_flat, p_t1, R_t1,
            **torso_phase_kwargs)
        # Install the swing-EE phase override when mid-waypoint
        # reshape succeeded OR FK mode is on. Endpoints come from FK
        # at pq_live (start) and FK at q_end (end). The legacy
        # SwingPlanner path is taken when no override is registered.

        # Capture the per-step planned joint endpoints. Read by the canonical
        # driver's per-step q-log (diag_cooperative_arms.py:510-511), not by
        # anything in crawlbot/ — see __init__ for the CLEANUP-34 note.
        self._step_q_start = pq_live.copy()
        self._step_q_end = q_end.copy()
        self._step_t_ss_start = float(t_ss_start)
        self._step_T_step = float(T_step)
        # M7 / FK-on-smoothed-q: cache the smoothed q-sequence + per-
        # segment tangents (so the mapping sees the same q(τ) the
        # planners are tracking).
        # Cleared (set to None) under legacy mode so the legacy quintic
        # path is taken.
        self._step_q_seq = None
        self._step_dq_seg = None
        # Snapshot the torso linear position at SS entry — read by the
        # mapping_bypass_in_ss diagnostic in _step() to freeze the SS
        # linear torso reference. p_t0 is the torso pose computed from
        # the live state above (line ~810), so it equals the actual
        # torso position at the moment SS begins.
        # Option A: this also resets the post-dock DS blend; the next weld
        # activation re-arms it (controller.on_dock).
        self.controller.on_ss_entry(p_t0)

        return (q_end, T_step, True)

    def _run_preplanner(
        self,
        t_plan_start: float,
        stance_arm: str,
        stance_a: int,
        stance_b: int,
        r_com_0: np.ndarray,
        r_com_goal: np.ndarray,
    ) -> 'tuple[float, bool]':
        """Solve the coarse pre-planner for the upcoming step (M7).

        The pre-planner produces T_step (step duration from the momentum
        envelope) and a momentum-feasible CoM trajectory. The initial
        T_step guess for the NLP is derived inline from the momentum
        envelope: v_max ≈ min(h_max)/(m·lever_arm), T_step ≈
        ‖Δr_com‖/v_max. `CoarsePlanResult.from_heuristic` exists for
        unit tests only and is NOT called here — a failed NLP solve
        skips the step (no silent heuristic fallback).

        On success, caches ``_coarse_plan`` and ``_coarse_plan_t0 =
        t_plan_start`` so ``_step()`` can evaluate the reference at
        current sim time via ``r_com_at(t - t0)``.

        Returns
        -------
        (T_step, success) : (float, bool)
        """
        cfg = self.cfg
        # Live state for (v0, L0)
        pq_live, pv_live = self.sensors.joint_state()
        rs_live = self.robot.update(pq_live, pv_live)
        v_com_0 = rs_live.v_com.copy()
        L_com_0 = rs_live.L_com.copy()
        # Conservation constant c = hw_0 + L_com_0 + r_com_0 × m·v_com_0
        hw_0 = self.sensors.wheel_momentum()
        m = float(rs_live.total_mass)
        c_const = hw_0 + L_com_0 + np.cross(r_com_0, m * v_com_0)

        # Stance contact point (constant anchor in structure frame).
        if stance_arm == 'a':
            r_C = self.sched.anchors_a[stance_a].copy()
        else:
            r_C = self.sched.anchors_b[stance_b].copy()

        # Initial T_step guess from the momentum envelope (inline —
        # CoarsePlanResult.from_heuristic is test-only and must not
        # appear on this code path per M7 design).
        h_max = np.asarray(cfg.h_max_tight, dtype=float).reshape(3)
        lever_arm = 1.0  # conservative nominal |r_com| for ASTROHUB-scale
        v_max = float(np.min(np.abs(h_max))) / max(m, 1e-6) / max(lever_arm, 1e-6)
        distance = float(np.linalg.norm(r_com_goal - r_com_0))
        T_step_guess = max(0.5, distance / v_max) if v_max > 0.0 else 1.0
        # Standoff-keyed dock-margin safety factor (default off). Only steps whose
        # standoff exceeds the knee get extra swing time — keying the knee ABOVE
        # the other steps' standoff isolates the change to the highest-standoff
        # docking step, leaving all others bit-identical (no cross-step coupling).
        if float(np.linalg.norm(r_com_0)) > cfg.preplanner_tstep_standoff_knee:
            T_step_guess *= (1.0 + cfg.preplanner_tstep_standoff_gain)
        # Per-step T_step scale (DIAGNOSTIC, default off). len(self._preplanner_stats)
        # is the 0-based index of THIS step (one pre-planner call per step, appended
        # after the solve below), so this isolates a single step's T_step.
        if len(self._preplanner_stats) == cfg.preplanner_tstep_scale_step:
            T_step_guess *= cfg.preplanner_tstep_scale_factor

        result = self.preplanner.solve(
            r_com_0=r_com_0,
            v_com_0=v_com_0,
            L_com_0=L_com_0,
            r_com_goal=r_com_goal,
            r_C_stance=r_C,
            c_const=c_const,
            T_step=T_step_guess,
            h_max=h_max,
        )
        self._preplanner_stats.append({
            'success': result.success,
            'solve_ms': result.solve_time_ms,
            'iter_count': result.iter_count,
            'cost': result.cost,
            'status': result.status,
            't_plan_start': t_plan_start,
            'T_step': float(result.T_step) if result.success else float(T_step_guess),
        })
        if result.success:
            self._coarse_plan = result
            self._coarse_plan_t0 = t_plan_start
            peak_v = float(max(np.linalg.norm(v) for v in result.v_com))
            peak_L = float(max(np.linalg.norm(L) for L in result.L_com))
            print(f"[CoarsePrePlanner] success in "
                  f"{result.solve_time_ms:.1f} ms "
                  f"({result.iter_count} iters, cost={result.cost:.3f}, "
                  f"T_step={result.T_step:.2f}s, peak |v|={peak_v:.3f} m/s, "
                  f"peak |L|={peak_L:.3f} Nms)")
            return (float(result.T_step), True)
        else:
            # Failure: clear the cached plan and return False. sim_loop
            # logs the failure, holds position, and skips the step.
            # No silent heuristic fallback.
            self._coarse_plan = None
            print(f"[CoarsePrePlanner] FAILED: {result.status} — "
                  f"step will be skipped")
            return (0.0, False)

    # ── Run ──────────────────────────────────────────────────────────────

    def _capture_snapshot(self, log, t, label):
        """Append a snapshot (t, qpos, qvel, label) for offline rendering."""
        qpos, qvel = self.sensors.raw_state()
        log.snapshots.append((round(t, 3), qpos, qvel, label))

    def run(self, verbose=True):
        """Run full multi-step locomotion simulation.

        The gait sequencer (``_gait_program``) driven by the single simulation
        loop (``_drive``). Returns the ``SimLog``.
        """
        return self._drive(self._gait_program(verbose))

    # ── The simulation loop ──────────────────────────────────────────────

    def _drive(self, program):
        """THE simulation loop — the only place the plant is stepped.

        ``program`` is a sequencer coroutine that yields work requests:
        ``_DSSettle`` (energy-based passivity settle, NMPC bypassed) or
        ``_NMPCTick`` (one NMPC period: plan, then n_qp_per_nmpc QP
        sub-steps). Each iteration advances the active request by ONE
        control period dt_qp — measure, control, plant I/O, ``plant.step()``
        — or completes it (a DS exit check, the NMPC hand-off to telemetry);
        the result is sent back into the sequencer, which picks the next
        request. That is the shape of a ROS 2 timer callback.

        Returns the sequencer's return value.
        """
        try:
            req = next(program)
        except StopIteration as stop:
            return stop.value
        tick = self._begin(req)
        while True:
            done, result = tick()
            if not done:
                continue
            try:
                req = program.send(result)
            except StopIteration as stop:
                return stop.value
            tick = self._begin(req)

    def _begin(self, req):
        """Start a request; return its tick function (-> (done, result))."""
        if isinstance(req, _DSSettle):
            st = self._ds_begin(req)
            return lambda: self._ds_tick(st)
        st = self._nmpc_begin(req)
        return lambda: self._nmpc_tick(st)

    def _gait_program(self, verbose=True):
        """The gait SEQUENCER, as a coroutine driven by ``_drive``.

        This is the former body of ``run()``, verbatim, except that it no
        longer advances the physics itself: where it used to call
        ``_run_ds_passivity_loop(...)`` it now yields a ``_DSSettle`` request,
        and where it called ``_step(...)`` it yields an ``_NMPCTick``. The
        request's result comes back as the value of the ``yield``. It decides
        WHAT runs next (DS settle, DWELL, planning handoff, SS, HOLD, dock,
        trailing DS); ``_drive`` decides nothing and just ticks.
        """
        cfg = self.cfg
        log = SimLog()
        # Fingerprint the execution environment once at simulation start.
        # Stored under log.environment and persisted in sim_log.json so
        # archived logs carry the toolchain state that produced them.
        log.environment = capture_environment()
        plan = self.plan

        # Copy the setup-phase settling trace into the log so it shows
        # up in Fig 4 (energy decay) of the diagnostic suite.
        s = self._settling_log
        log.settling_t = list(s['t_log'])
        log.settling_T = list(s['T_log'])
        log.settling_T_target = float(s['T_settle'])
        log.settling_stage1_steps = int(s['stage1_steps'])
        log.settling_stage2_steps = int(s['stage2_steps'])
        log.settling_exit_reason = str(s['exit_reason'])

        hw = cfg.hw_init.copy()
        t = 0.0
        L_com_prev = None

        # Capture R_torso(t=0) ONCE as the global flat orientation reference
        # (structure frame). Sourced at all phases below (NOT re-sampled from
        # live state), so the orientation reference never re-anchors on an FSM
        # boundary. R_flat is the real mounting orientation (rpy ~ 0,0,-5.16°),
        # NOT identity — holding it imposes no frame convention and snaps
        # nothing at t=0.
        _pq_flat, _pv_flat = self.sensors.joint_state()
        _rs_flat = self.robot.update(_pq_flat, _pv_flat)
        self._R_torso_flat = _rs_flat.oMf_torso.rotation.copy()
        # Seed the hold so the initial DS (before the first step's setup)
        # already references R_flat + the initial torso pose, instead of the
        # identity/(0,0,0) sentinel that _hold_reference returns when nothing
        # is set (which otherwise shows a one-time ~5.16° init jump at the
        # first DS->SS). The first _setup_torso_for_step overwrites this.
        self.torso_planner.set_hold(
            _rs_flat.oMf_torso.translation.copy(), self._R_torso_flat,
            r_com=_rs_flat.r_com.copy())

        # Parse phases: DS-SS pairs
        phases = plan.phases
        step_idx = 0
        i = 0
        t_offset = 0.0   # Cumulative time offset from inter-step settling
        self._t_plan_offset = 0.0  # mirror for _step's swing-planner query
        while i < len(phases):
            gp = phases[i]
            if gp.phase.value == 'double':
                # Look ahead for SS phase
                if i + 1 < len(phases) and phases[i+1].phase.value != 'double':
                    ss_gp = phases[i+1]
                    ss_phase_idx = i + 1

                    swing_arm = ss_gp.swing_arm
                    stance_arm = 'a' if swing_arm == 'b' else 'b'
                    stance_a = ss_gp.anchor_a_idx
                    stance_b = ss_gp.anchor_b_idx
                    target_idx = ss_gp.swing_to_idx

                    if verbose:
                        print(f"\n[Step {step_idx}] swing={swing_arm}, "
                              f"stance=({stance_a}a,{stance_b}b), "
                              f"target={target_idx}{swing_arm}")

                    # DS contact config lookup (still uses scheduler's
                    # phase skeleton for anchor pair).
                    cc_ds = self.sched.contact_config_at(plan.t_start[i] + 0.1)
                    if step_idx == 0 and len(log.snapshots) == 0:
                        self._capture_snapshot(log, t, 'initial')

                    # ── 1. DS — energy-based exit (spec §7.1.1) ──────────
                    # A _DSSettle request (ticked by _drive) drives T_kin <
                    # T_settle via the passivity-constrained QP. NMPC is
                    # bypassed during DS (no reference motion to track).
                    # n_ds_max_steps is the safety cap only — there is no
                    # time-based exit.
                    if verbose:
                        print(f"  DS: t={t:.2f}s (energy-based exit)")
                    t_ds_start_wall = t
                    min_steps_ds = max(
                        0, int(round(cfg.t_settle_inter_min / cfg.dt_qp)))
                    # ── Phase-B logging context for inter-step DS ───────
                    # just_landed_arm = swing arm of the PRIOR SS phase
                    # (phases[i-1]); '' for the initial DS (i == 0). The
                    # logged step_idx is the prior step that just finished
                    # (= current step_idx − 1; −1 for the initial DS).
                    _ds_log_just_landed = ''
                    _ds_log_swing_to = -1
                    if i > 0 and plan.phases[i - 1].swing_arm:
                        _ds_log_just_landed = plan.phases[i - 1].swing_arm
                        _ds_log_swing_to = plan.phases[i - 1].swing_to_idx
                    _ds_log_step_idx = step_idx - 1
                    # Anchor indices for the DS contact pair — the
                    # _post-step_ welded positions (cc_ds was built from
                    # the inter-step DOUBLE phase, line ~1555).
                    _ds_anchor_a = phases[i].anchor_a_idx
                    _ds_anchor_b = phases[i].anchor_b_idx
                    ds_result = yield _DSSettle(
                        contact_config=cc_ds,
                        max_steps=cfg.n_ds_max_steps,
                        # Settle-exit fix (POINT A): dock-tolerance-derived ε_v
                        # override when > 0 (else the byte-identical 1 mm/s).
                        epsilon_v=(cfg.interstep_settle_epsilon_v
                                   if cfg.interstep_settle_epsilon_v > 0
                                   else cfg.settle_inter_epsilon_v),
                        plateau_window=50,
                        plateau_ratio=cfg.settle_plateau_ratio,
                        min_steps=min_steps_ds,
                        fallback_Kd=cfg.Kd_settle_damping,
                        # Logging-only kwargs (no control effect):
                        log_obj=log,
                        log_step_idx=_ds_log_step_idx,
                        log_just_landed_arm=_ds_log_just_landed,
                        log_anchor_a_idx=_ds_anchor_a,
                        log_anchor_b_idx=_ds_anchor_b,
                        log_t_abs=t_ds_start_wall,
                    )
                    dt_ds_elapsed = ds_result['n_steps'] * cfg.dt_qp
                    t += dt_ds_elapsed
                    t_offset += dt_ds_elapsed
                    self._t_plan_offset = t_offset
                    log.inter_step_settles.append({
                        'step_idx': int(step_idx),
                        't_start': float(t_ds_start_wall),
                        't_end': float(t),
                        'n_steps': int(ds_result['n_steps']),
                        'T_start': float(ds_result['T_start']),
                        'T_end': float(ds_result['T_end']),
                        'T_settle': float(ds_result['T_settle']),
                        'lambda_min': float(ds_result['lambda_min']),
                        'exit_reason': ds_result['exit_reason'],
                    })
                    if verbose:
                        print(
                            f"  DS settle: {t_ds_start_wall:.2f}"
                            f" → +{dt_ds_elapsed:.3f}s "
                            f"({ds_result['n_steps']} steps, "
                            f"exit={ds_result['exit_reason']}, "
                            f"T: {ds_result['T_start']:.2e} → "
                            f"{ds_result['T_end']:.2e} J)")

                    # ── 1b. DWELL (optional) — extended inter-step DS ─────
                    # When the GaitPlan's DS phase has a duration much
                    # longer than the natural energy-decay window, model
                    # external AOCS desat by continuing to run the QP in
                    # centroidal-DS settle mode for the remainder of the
                    # planned duration. The Stage 3 captured reference is
                    # set so the WBC holds the welded equilibrium.
                    _dwell_target = gp.duration - dt_ds_elapsed
                    if _dwell_target > 1.0 and cfg.ds_centroidal_mode:
                        if verbose:
                            print(f"  DWELL: continuing DS for "
                                  f"{_dwell_target:.1f}s "
                                  f"(centroidal-DS active, passivity on)")
                        # Capture welded-state torso reference for the dwell.
                        pq_d, pv_d = self.sensors.joint_state()
                        rs_d = self.robot.update(pq_d, pv_d)
                        t_dwell_end = t + _dwell_target
                        # CLEANUP-14: ds_mobile_com_magnitude was 0.0 on the canonical;
                        # the CoM-mobile DS variant never ran. Hold the welded state.
                        self.torso_planner.set_hold(
                            rs_d.oMf_torso.translation.copy(),
                            self._R_torso_flat.copy(),
                            r_com=rs_d.r_com.copy())
                        # Use stance pair from the upcoming SS phase as the
                        # contact config (both arms welded → DOUBLE).
                        last_sa_d = stance_a
                        last_sb_d = stance_b
                        while t < t_dwell_end:
                            hw, L_com_prev = yield _NMPCTick(
                                t, 'DS', step_idx - 1,
                                swing_arm, stance_arm,
                                cc_ds, 0, last_sa_d, last_sb_d,
                                hw, L_com_prev, log, ss_end=t,
                                settle_mode=True,
                                passivity_override=True,
                                ds_centroidal_active=True)
                            t += cfg.dt_nmpc
                        if verbose:
                            print(f"  DWELL end: t={t:.2f}s, "
                                  f"‖h_w‖={np.linalg.norm(hw):.3f} Nms")
                        # α (J2 #2): drop the (possibly moving) dwell phase so
                        # the next SS plans fresh and no stale phase leaks into
                        # reference_at during the transition.
                        self.torso_planner.clear_phases()
                        # B-probe: reset NMPC warm-start after the long
                        # dwell. The NMPC ran ~thousands of ticks during
                        # the dwell with a static DS reference; carrying
                        # that warm-state into SS planning is stale.
                        self.nmpc.reset_warm_start()
                        L_com_prev = None  # force AOCS FD to re-seed too

                    # ── 2. Planning handoff: pre-planner → T_step ────────
                    # Runs the coarse pre-planner (mandatory) to get
                    # T_step and a momentum-feasible CoM trajectory, and
                    # sets up the torso planner over [t, t + T_step].
                    # Installs T_step in the scheduler's SS phase so the
                    # SwingPlanner uses the same horizon.
                    t_ss_start = t
                    q_dock, T_step, step_feasible = self._setup_torso_for_step(
                        t_ss_start, swing_arm,
                        stance_a, stance_b, swing_arm, target_idx,
                        ss_phase_idx=ss_phase_idx)
                    # Mirror Option Z reset into the outer loop's offset.
                    t_offset = self._t_plan_offset
                    if not step_feasible:
                        log.aborted_steps.append({
                            'step_idx': int(step_idx),
                            't': float(t),
                            'reason': 'preplanner_infeasible'})
                        if verbose:
                            print(f"  SKIP step {step_idx}: "
                                  f"pre-planner infeasible — holding position")
                        step_idx += 1
                        i += 2
                        if cfg.stop_on_failed_step:
                            if verbose:
                                print(f"  [stop_on_failed_step] aborting "
                                      f"traversal after step "
                                      f"{step_idx - 1} SKIP")
                            break
                        continue

                    self.qp_ss.set_nominal_posture(
                        q_dock[self.robot.joints_q_slice])
                    log.preplanner_T_steps.append(float(T_step))

                    # Contact config during SS (swing arm lifted).
                    # The scheduler's plan was just updated with T_step,
                    # so phase_at(t_ss_start + 0.1) returns the SS phase.
                    cc_ss = self.sched.contact_config_at(
                        plan.t_start[ss_phase_idx] + 0.1)

                    # ── 3. Release swing arm, reset NMPC/GMO ─────────────
                    self._capture_snapshot(log, t, f'release_step{step_idx}')
                    old_anchor = ss_gp.swing_from_idx
                    self.plant.deactivate_weld(swing_arm, old_anchor)
                    self.nmpc.reset_warm_start()
                    pq_r, pv_r = self.sensors.joint_state()
                    rs_r = self.robot.update(pq_r, pv_r)
                    self.gmo.reset(rs_r.H, rs_r.v)
                    self.contact_sm.reset()
                    _, _, oMf_release = self._get_ee_data(rs_r, swing_arm)
                    self.swing_planner.set_swing_orientation(
                        oMf_release.rotation)

                    # ── 4. SS — synchronized trajectory, dock detection ──
                    # Torso and swing planners share [0, T_step] and both
                    # arrive at their targets at T_step. The QP uses the
                    # single qp_ss variant throughout — no gain scheduling,
                    # no approach thresholds. Dock detection runs every
                    # NMPC tick after cfg.dock_check_delay.
                    t_ss_deadline = t_ss_start + T_step + cfg.t_ss_margin
                    t_hold_deadline = t_ss_deadline + cfg.t_hold_max
                    if verbose:
                        print(f"  SS: t={t_ss_start:.2f}s, "
                              f"T_step={T_step:.2f}s, "
                              f"deadline={t_ss_deadline:.2f}s, "
                              f"released {swing_arm}@{old_anchor}")

                    docked = False
                    # Periodic snapshot capture for offline rendering.
                    # When cfg.frames_per_step > 0, schedule
                    # frames_per_step evenly-spaced captures across
                    # [t_ss_start, t_ss_start + T_step].
                    if int(getattr(cfg, 'frames_per_step', 0)) > 0:
                        n_frames = int(cfg.frames_per_step)
                        if n_frames == 1:
                            self._frame_capture_times = [t_ss_start]
                        else:
                            self._frame_capture_times = [
                                t_ss_start + (k / (n_frames - 1)) * T_step
                                for k in range(n_frames)]
                        self._frame_capture_step_idx = int(step_idx)
                        self._frame_capture_kidx = 0
                    else:
                        self._frame_capture_times = []
                    while t < t_ss_deadline and not docked:
                        # Periodic frame capture (cfg.frames_per_step > 0).
                        # Triggered when t crosses the next scheduled time.
                        if self._frame_capture_times and t >= self._frame_capture_times[0]:
                            self.plant.forward()
                            label = (
                                f'frame_step{self._frame_capture_step_idx}_'
                                f'{self._frame_capture_kidx}')
                            self._capture_snapshot(log, t, label)
                            self._frame_capture_times.pop(0)
                            self._frame_capture_kidx += 1
                        hw, L_com_prev = yield _NMPCTick(
                            t, 'SS', step_idx, swing_arm, stance_arm,
                            cc_ss, target_idx, stance_a, stance_b,
                            hw, L_com_prev, log,
                            ss_end=t_ss_start + T_step)
                        t += cfg.dt_nmpc

                        # Dock gate prerequisites (all must hold):
                        #   (1) dock_check_delay satisfied (avoid release noise)
                        #   (2) M7 v22: t ≥ t_ss_start + ef·T_step so the
                        #       swing planner has completed its trajectory
                        #       (velocity and acceleration at target are zero)
                        #   (3) position + orientation thresholds
                        swing_done = ((t - t_ss_start)
                                      >= cfg.swing_early_finish_fraction * T_step)
                        if ((t - t_ss_start) > cfg.dock_check_delay
                                and swing_done):
                            self.plant.forward()
                            docked, d, ori_err_deg, twist_norm = (
                                self._dock_gate(
                                    swing_arm, target_idx,
                                    log=log, t=t, step_idx=step_idx))
                            if docked:
                                log.dock_events.append({
                                    't': round(t, 3), 'step': step_idx,
                                    'd_mm': round(d*1000, 2),
                                    'ori_deg': round(ori_err_deg, 2),
                                    'twist': round(twist_norm, 6),
                                    'arm': swing_arm, 'anchor': target_idx,
                                    'method': 'kinematic'})
                                self._capture_snapshot(
                                    log, t, f'dock_step{step_idx}')
                                if verbose:
                                    print(f"  *** DOCK step {step_idx}: "
                                          f"t={t:.2f}s d={d*1000:.1f}mm "
                                          f"ori={ori_err_deg:.2f}° ***")

                    # ── 5. Convergence hold (if not docked) ──────────────
                    # The torso planner holds at its last endpoint past
                    # T_step (reference_at falls through to _hold_reference);
                    # the swing planner clamps tau to 1.0 → p_dock with
                    # v=0, a=0. Because both terminal references are
                    # static with zero velocity, PD feedback is
                    # self-decelerating — no passivity constraint is
                    # needed. We ran a first pass with
                    # passivity_hold=True and it prevented the arm from
                    # doing the positive work required to close the last
                    # few mm of position error. Normal SS tracking
                    # (passivity_active=False) is the correct regime for
                    # the hold window.
                    if not docked:
                        if verbose:
                            print(f"  HOLD (tracking): {t:.2f} → "
                                  f"{t_hold_deadline:.2f}s")
                        while t < t_hold_deadline and not docked:
                            hw, L_com_prev = yield _NMPCTick(
                                t, 'SS', step_idx, swing_arm, stance_arm,
                                cc_ss, target_idx, stance_a, stance_b,
                                hw, L_com_prev, log,
                                ss_end=t_ss_start + T_step,
                                # Dock-floor audit: default False (the SS-hold
                                # escape — passivity OFF); the audit forces it
                                # ON to test whether passivity limits the close.
                                passivity_hold=cfg.dock_hold_passivity_on)
                            t += cfg.dt_nmpc

                            self.plant.forward()
                            docked, d, ori_err_deg, twist_norm = (
                                self._dock_gate(
                                    swing_arm, target_idx,
                                    log=log, t=t, step_idx=step_idx))
                            if docked:
                                log.dock_events.append({
                                    't': round(t, 3), 'step': step_idx,
                                    'd_mm': round(d*1000, 2),
                                    'ori_deg': round(ori_err_deg, 2),
                                    'twist': round(twist_norm, 6),
                                    'arm': swing_arm, 'anchor': target_idx,
                                    'method': 'kinematic'})
                                self._capture_snapshot(
                                    log, t, f'dock_step{step_idx}')
                                if verbose:
                                    print(f"  *** DOCK (hold) step "
                                          f"{step_idx}: t={t:.2f}s "
                                          f"d={d*1000:.1f}mm "
                                          f"ori={ori_err_deg:.2f}° ***")

                    if not docked:
                        recent = (log.d_grip_swing[-20:]
                                  if len(log.d_grip_swing) >= 20
                                  else log.d_grip_swing)
                        min_d = min(recent) * 1000 if recent else float('nan')
                        ori_at_timeout = self._gripper_ori_err_deg(
                            swing_arm, target_idx)
                        log.aborted_steps.append({
                            'step_idx': int(step_idx),
                            't': float(t),
                            'reason': 'dock_timeout',
                            'd_mm': float(min_d),
                            'ori_deg': float(ori_at_timeout)})
                        if verbose:
                            print(f"  TIMEOUT step {step_idx}: "
                                  f"min d={min_d:.1f}mm "
                                  f"ori_at_exit={ori_at_timeout:.1f}°")

                    # ── 6. Post-dock: activate weld + inelastic impact ───
                    if docked:
                        self.plant.activate_weld(swing_arm, target_idx)
                        self.plant.forward()
                        self.nmpc.reset_warm_start()
                        # Option A: capture the SS-exit torso position
                        # and weld time for the post-dock DS blend. The
                        # blend endpoint is not stored
                        # here — _step() recomputes the live mapping
                        # output each tick and blends it against
                        # _ds_ramp_p_start. See M7_T12_MEMO.md §5.
                        self.controller.on_dock(t)

                        # ── Inelastic impact: FULL-DOF momentum-consistent ──
                        # (Fix A, dock-leak Part 3) — see plant.py.
                        self.plant.apply_dock_impact(verbose)

                    # Post-dock energy-based settling is now handled by
                    # the *next* step's DS block (above). One shared
                    # implementation: the _DSSettle request.

                    step_idx += 1
                    i += 2  # skip SS phase (already processed)
                    if cfg.stop_on_failed_step and not docked:
                        if verbose:
                            print(f"  [stop_on_failed_step] aborting "
                                  f"traversal after step "
                                  f"{step_idx - 1} TIMEOUT")
                        break
                else:
                    # Trailing DS (end of gait): run settling phase
                    t_ds_start = plan.t_start[i] + t_offset
                    t_ds_settle = t + cfg.t_settle_final
                    cc_ds = self.sched.contact_config_at(plan.t_start[i] + 0.1)

                    # Diagnostic: did the preceding SS abort on dock_timeout?
                    # Used only to gate the three diag_*_on_abort flags. No
                    # effect on normal operation.
                    _abort_ds = bool(
                        log.aborted_steps
                        and log.aborted_steps[-1].get('reason') == 'dock_timeout'
                        and log.aborted_steps[-1].get('step_idx') == step_idx - 1
                    )

                    # H_DS1 diagnostic override — force SINGLE_A to match the
                    # physical single-weld state after dock_timeout.
                    if _abort_ds and cfg.diag_force_single_contact_on_abort:
                        cc_ds = ContactConfig.from_phase(
                            ContactPhase.SINGLE_A,
                            cc_ds.r_contact_A.copy(),
                            cc_ds.r_contact_B.copy())

                    # Use last swing step's info for logging
                    last_swing = 'b'; last_stance = 'a'
                    last_sa = plan.phases[i].anchor_a_idx if hasattr(plan.phases[i], 'anchor_a_idx') else 0
                    last_sb = plan.phases[i].anchor_b_idx if hasattr(plan.phases[i], 'anchor_b_idx') else 0
                    if i > 0 and plan.phases[i-1].swing_arm:
                        last_swing = plan.phases[i-1].swing_arm
                        last_stance = 'a' if last_swing == 'b' else 'b'
                        last_sa = plan.phases[i-1].anchor_a_idx
                        last_sb = plan.phases[i-1].anchor_b_idx

                    if verbose:
                        print(f"  DS settle: {t:.2f} → +{cfg.t_settle_final}s")

                    # Compute DS equilibrium via IK: both tools at anchors.
                    # This gives the true static configuration rather than
                    # the transient pose at dock time (which has residual
                    # velocity and doesn't match the welded equilibrium).
                    pq, pv = self.sensors.joint_state()
                    rs_hold = self.robot.update(pq, pv)
                    # Stage 3 (DS memo §6.3, §7.4): when on, hold at the
                    # actual welded state, not the dock-IK target. The
                    # dock-IK is the documented source of the persistent
                    # ~3.86° torso ori error (it solves both-tools-at-
                    # anchors, over-determined once welds are active).
                    _use_state = (cfg.ds_torso_ref_from_state
                                  or (_abort_ds
                                      and cfg.diag_freeze_torso_ref_on_abort))
                    if _use_state:
                        self.torso_planner.set_hold(
                            rs_hold.oMf_torso.translation.copy(),
                            self._R_torso_flat.copy(),
                            r_com=rs_hold.r_com.copy())
                    else:
                        try:
                            anchor_a_se3 = self.sched.anchor_se3('a', last_sa)
                            anchor_b_se3 = self.sched.anchor_se3('b', last_sb)
                            q_eq = dock_configuration(
                                self.robot.model, anchor_a_se3, anchor_b_se3,
                                q_init=pq)
                            rs_eq = self.robot.update(q_eq, np.zeros(self.robot.model.nv))
                            self.torso_planner.set_hold(
                                rs_eq.oMf_torso.translation.copy(),
                                self._R_torso_flat.copy(),
                                r_com=rs_eq.r_com.copy())
                        except RuntimeError:
                            # IK failed — fall back to current torso POSITION;
                            # orientation still the global R_flat.
                            self.torso_planner.set_hold(
                                rs_hold.oMf_torso.translation.copy(),
                                self._R_torso_flat.copy(),
                                r_com=rs_hold.r_com.copy())

                    # H_DS3 diagnostic override — disable the passivity
                    # inequality for trailing DS post-abort (_step reads this
                    # via the passivity_override kwarg).
                    # When ds_centroidal_mode is on, the trailing-DS
                    # settle uses the passivity inequality for energy
                    # dissipation (replacing the joint-vel-damping cost),
                    # so we force it ON regardless of the abort flag.
                    if cfg.ds_centroidal_mode:
                        _pass_override = True
                    else:
                        _pass_override = (
                            False if (_abort_ds and cfg.diag_disable_passivity_on_abort)
                            else None
                        )

                    while t < t_ds_settle:
                        hw, L_com_prev = yield _NMPCTick(
                            t, 'DS', step_idx - 1, last_swing, last_stance,
                            cc_ds, 0, last_sa, last_sb,
                            hw, L_com_prev, log, ss_end=t,
                            settle_mode=True,
                            passivity_override=_pass_override,
                            ds_centroidal_active=cfg.ds_centroidal_mode)
                        t += cfg.dt_nmpc

                    i += 1
            else:
                # Standalone SS phase (shouldn't happen in normal plan)
                i += 1

        self._capture_snapshot(log, t, 'final')
        if verbose:
            self._print_summary(log)
        return log

    # ── Single NMPC+QP step ──────────────────────────────────────────────

    def _swing_query_time(self, t_raw: float, phase: str, ss_end) -> float:
        """Plan-time fed to ``SwingPlanner.reference_at``.

        Clamped so an extended SS convergence-hold queries the dock
        target (quintic τ pinned at 1, v=a=0) rather than walking into
        the *next* scheduled phase's anchors — which belong to the
        other arm and the subsequent target, and which the controller
        never tracks. The control and logging paths MUST share this:
        when they computed it independently the logging path omitted
        the clamp and e_ee_pos reported an 800mm phantom error against
        a reference the QP was not following (T15 schedule/execution
        desync).
        """
        tq = t_raw - self._t_plan_offset
        if phase == 'SS' and ss_end is not None:
            tq = min(tq, (ss_end - self._t_plan_offset) - 0.01)
        return tq

    def _step(self, t, phase, step_idx, swing_arm, stance_arm,
              cc_ss, target_anchor, stance_a, stance_b,
              hw, L_com_prev, log, ss_end=None, settle_mode=False,
              passivity_hold: bool = False,
              passivity_override=None,
              ds_centroidal_active: bool = False):
        """One NMPC period (plan + QP sub-steps), run through the single loop.

        Compatibility entry point: builds an ``_NMPCTick`` request and drives
        it with ``_drive``. Returns ``(hw, L_com_prev)``. Parameters: see
        ``_NMPCTick``.
        """
        return self._drive(_once(_NMPCTick(
            t, phase, step_idx, swing_arm, stance_arm, cc_ss, target_anchor,
            stance_a, stance_b, hw, L_com_prev, log, ss_end=ss_end,
            settle_mode=settle_mode, passivity_hold=passivity_hold,
            passivity_override=passivity_override,
            ds_centroidal_active=ds_centroidal_active)))

    def _nmpc_begin(self, r):
        """Start of an NMPC period: intent, stage 1 (NMPC), QP carry."""
        (t, phase, step_idx, swing_arm, stance_arm, cc_ss, target_anchor,
         stance_a, stance_b, hw, L_com_prev, log, ss_end, settle_mode,
         passivity_hold, passivity_override, ds_centroidal_active) = (
            r.t, r.phase, r.step_idx, r.swing_arm, r.stance_arm, r.cc_ss,
            r.target_anchor, r.stance_a, r.stance_b, r.hw, r.L_com_prev,
            r.log, r.ss_end, r.settle_mode, r.passivity_hold,
            r.passivity_override, r.ds_centroidal_active)
        # ══════════════════════════════════════════════════════════════════
        # PHASE 0 — the tick's intent
        # Contact geometry and the mode flags, handed to the controller as one
        # ControlIntent. References are NOT read here any more: the controller
        # queries them by time from self.refs (PlannerReferences).
        # ══════════════════════════════════════════════════════════════════
        cfg = self.cfg

        cc_nmpc = ContactConfig.from_phase(
            cc_ss.phase,
            self.sched.anchors_a[stance_a].copy(),
            self.sched.anchors_b[stance_b].copy())

        if ss_end is None:
            ss_end = t + cfg.dt_nmpc  # fallback
        intent = ControlIntent(
            t=t, phase=phase, step_idx=step_idx,
            contact=cc_ss, contact_nmpc=cc_nmpc,
            stance_anchors=(self.sched.anchors_a[stance_a],
                            self.sched.anchors_b[stance_b]),
            swing_arm=swing_arm, ss_end=ss_end,
            settle_mode=settle_mode, passivity_hold=passivity_hold,
            passivity_override=passivity_override,
            ds_centroidal_active=ds_centroidal_active)

        # Stage 1 — WholeBodyController.plan: CoM reference query, state,
        # NMPC solve, shifted fallback, QP-rate knots (control/controller.py).
        plan = self.controller.plan(intent, self.refs)
        rs = plan.rs
        if L_com_prev is None:
            L_com_prev = plan.L_com_now
        cref_r = plan.cref_r
        vp = plan.vp
        nmpc_ok, nmpc_status_code = plan.ok, plan.status_code
        nmpc_cost_val, info_n, t_nmpc_ms = plan.cost, plan.info, plan.t_ms
        # M7 debug: capture L_com_ref trace for the first N SS calls so
        # we can verify the TorsoPlanner momentum feedforward is wired
        # (expected nonzero during the 45.7° reorientation). Opt-in via
        # setattr(sim, '_debug_l_com_ref_trace_limit', N) before run().
        if (phase == 'SS'
                and getattr(self, '_debug_l_com_ref_trace_limit', 0) > 0):
            trace = getattr(self, '_debug_l_com_ref_trace', None)
            if trace is None:
                self._debug_l_com_ref_trace = []
                trace = self._debug_l_com_ref_trace
            if len(trace) < self._debug_l_com_ref_trace_limit:
                trace.append({
                    't': float(t),
                    't_mid': float(plan.t_mid),
                    'L_com_ref': plan.L_com_ref.copy(),
                    'norm': float(np.linalg.norm(plan.L_com_ref)),
                })

        # M7: single QP variant throughout DS and SS. Synchronized
        # trajectories eliminate the need for gain scheduling.
        # ══════════════════════════════════════════════════════════════════
        # STAGE 2 — whole-body QP sub-loop  (dt_qp = 0.01 s, n_qp_per_nmpc ticks)
        # Per sub-step: controller.track (torso mapping + F-SAT, QP, clip, AOCS)
        # -> diagnostic traces -> plant I/O -> controller.after_step. The ~25
        # live locals that blocked this cut now travel in a QPCarry.
        # ══════════════════════════════════════════════════════════════════
        t_qp_start = time.perf_counter()
        carry = self.controller.begin_tracking(plan, hw)


        return _NMPCRun(**{k: v for k, v in locals().items()
                           if k in _NMPCRun.__dataclass_fields__})

    def _nmpc_tick(self, st):
        """One QP sub-step of the NMPC period; the last one hands off."""
        if st.qs < self.n_qp_per_nmpc:
            self._qp_substep(st)
            st.qs += 1
        if st.qs >= self.n_qp_per_nmpc:
            return True, self._nmpc_handoff(st)
        return False, None

    def _qp_substep(self, st):
        """Stage 2, one dt_qp: controller.track -> diagnostic traces ->
        plant I/O -> plant.step -> controller.after_step."""
        cfg = self.cfg
        qs = st.qs
        (t, phase, step_idx, swing_arm, stance_arm, target_anchor, log,
         intent, plan, carry) = (
            st.t, st.phase, st.step_idx, st.swing_arm, st.stance_arm,
            st.target_anchor, st.log, st.intent, st.plan, st.carry)
        tq = t + qs * cfg.dt_qp
        # Stage 2 — WholeBodyController.track: measure, torso / swing
        # references, whole-body QP, clip, AOCS (control/controller.py).
        out = self.controller.track(carry, qs, tq, intent, self.refs, plan)
        # Local names for the diagnostic traces below (pure reads; kept
        # BEFORE plant.step because rs may view Pinocchio data).
        qp = self.qp_ss
        rs, tau = out.rs, out.tau_raw
        passivity_active, qp_ok = out.passivity_active, out.qp_ok
        qdd_t_qp, lambda_qp_sol = out.qdd_t, out.lambda_qp
        rp_interp, p_torso_ref_used = out.rp_interp, out.p_torso_ref_used

        # α (J2 #2): CoM-mobile DS conflict trace (gated; DWELL ticks).
        # Disentangles which constraint binds during the moving-CoM DS:
        #   pass_resid = dqⱼᵀτ_q + 2α·T_kin  (≈0 ⇒ passivity binding),
        #   Hdot_inf   = ‖Σ r_Cj×f_j + τ_j‖∞ from the QP wrench
        #               (→ τ_w_max ⇒ envelope binding),
        #   com_err    = ‖r_com − cref_r‖  (tracking),
        #   qp_ok / nmpc_status  (feasibility).

        # Dock-floor audit: per-SS-tick joint mechanical power + dock
        # distance, to confirm whether the arm does positive work
        # (dqⱼᵀτ_q > 0) while closing, and under which passivity setting.
        if cfg.log_dock_work and phase == 'SS':
            log.dock_work_trace.append({
                't': round(float(t), 3), 'step': int(step_idx),
                'd_mm': round(self._gripper_distance(
                    swing_arm, target_anchor) * 1000, 3),
                'dq_tau': float(rs.dq_joints @ tau),
                'pass_active': bool(passivity_active)})

        # ── Diagnostic B + C: per-cycle log of (c_ref, r_b_ref,
        # p_torso_actual, a_torso_des, a_torso_qp, δ(q_planned),
        # δ(q_current)) during step 0 OR step 2 SS. Gated on a
        # runtime attribute. Reads qp.last_torso_debug populated by
        # WholeBodyQP.solve(); also captures both δ-variants for
        # the planned-vs-current mass-distribution diagnostic.
        if getattr(self, '_step2_diag_enabled', False) \
                and phase == 'SS':
            td = getattr(qp, 'last_torso_debug', None)
            entry = {
                't': float(tq), 'qs': int(qs),
                'c_ref': np.asarray(rp_interp, dtype=float).tolist(),
                'r_b_ref': np.asarray(p_torso_ref_used,
                                      dtype=float).tolist(),
                'p_torso': np.asarray(
                    rs.oMf_torso.translation, dtype=float).tolist(),
            }
            if td is not None:
                entry['a_torso_des'] = (
                    np.asarray(td['a_torso_des_pre'],
                               dtype=float).tolist())
                entry['a_torso_qp'] = (
                    np.asarray(td['x_dd_torso_post'],
                               dtype=float).tolist())
            else:
                entry['a_torso_des'] = None
                entry['a_torso_qp'] = None
            if self.controller.torso_shaper._last_mapping_delta is not None:
                entry['delta_q'] = self.controller.torso_shaper._last_mapping_delta.tolist()
            else:
                entry['delta_q'] = None
            if self.controller.torso_shaper._last_mapping_delta_current is not None:
                entry['delta_q_current'] = (
                    self.controller.torso_shaper._last_mapping_delta_current.tolist())
            else:
                entry['delta_q_current'] = None
            entry['step_idx'] = int(step_idx)
            self._step2_diag_log.append(entry)

        # M7 physics-trace capture (SS only, first-QP-substep, 1 Hz).
        # No control change — reads QP outputs + kinematic conditioning
        # to diagnose where the 67° torso disturbance is coming from.
        if (phase == 'SS' and qs == 0 and qp_ok
                and getattr(self, '_debug_physics_trace_limit', 0) > 0):
            self._debug_physics_count = getattr(
                self, '_debug_physics_count', 0) + 1
            sample_every = int(getattr(
                self, '_debug_physics_sample_every', 10))
            sample_idx = self._debug_physics_count - 1
            trace = getattr(self, '_debug_physics_trace', None)
            if trace is None:
                self._debug_physics_trace = []
                trace = self._debug_physics_trace
            if (sample_idx % sample_every == 0
                    and len(trace) < self._debug_physics_trace_limit):
                # Stance EE Jacobian and null-space projection.
                try:
                    J_ee_stance, _, _ = self._get_ee_data(rs, stance_arm)
                except Exception:
                    J_ee_stance = None
                J_t = rs.J_torso
                sig_t = np.linalg.svd(J_t, compute_uv=False)
                cond_t = (float(sig_t[0] / sig_t[-1])
                          if sig_t[-1] > 1e-12 else float('inf'))
                cond_NJe = float('nan')
                sig_NJe_min = float('nan')
                if J_ee_stance is not None:
                    # Damped pseudo-inverse of J_torso
                    lam = 1e-6
                    JJt = J_t @ J_t.T + lam * np.eye(J_t.shape[0])
                    J_t_pinv = J_t.T @ np.linalg.inv(JJt)
                    N_t = (np.eye(J_t.shape[1])
                           - J_t_pinv @ J_t)
                    NJe = J_ee_stance @ N_t
                    sig_n = np.linalg.svd(NJe, compute_uv=False)
                    sig_NJe_min = float(sig_n[-1])
                    cond_NJe = (float(sig_n[0] / sig_n[-1])
                                if sig_n[-1] > 1e-12 else float('inf'))
                # Split lambda into per-contact 6D (force, torque).
                lam_v = np.asarray(lambda_qp_sol, dtype=float).ravel()
                per_contact = []
                for ci in range(len(lam_v) // 6):
                    f = lam_v[6*ci:6*ci+3]
                    tq_c = lam_v[6*ci+3:6*ci+6]
                    per_contact.append(
                        (float(np.linalg.norm(f)),
                         float(np.linalg.norm(tq_c))))
                tau_arr = np.asarray(tau, dtype=float).ravel()
                sat_mask = np.abs(tau_arr) >= 0.99 * cfg.tau_max
                # Also capture q so we can reproduce κ offline.
                q_snap = np.asarray(rs.q, dtype=float).copy()
                # M7 torso PD diagnosis: capture pre-solve desired
                # torso accel vs post-solve achieved torso accel.
                torso_dbg = getattr(qp, 'last_torso_debug', None)
                entry = {
                    't': float(t),
                    'phase': str(phase),
                    'qdd_t': np.asarray(qdd_t_qp, dtype=float).copy(),
                    'tau_q': tau_arr.copy(),
                    'tau_abs_max': float(np.max(np.abs(tau_arr))),
                    'tau_l2':      float(np.linalg.norm(tau_arr)),
                    'tau_sat_idx': [int(i) for i, s in
                                    enumerate(sat_mask) if s],
                    'lambda':      lam_v.copy(),
                    'contact_fL':  per_contact,
                    'cond_J_t':    cond_t,
                    'sig_min_J_t': float(sig_t[-1]),
                    'cond_NJe':    cond_NJe,
                    'sig_min_NJe': sig_NJe_min,
                    'q':           q_snap,
                }
                if torso_dbg is not None:
                    entry['torso_debug'] = {
                        k: (v.copy() if hasattr(v, 'copy') else v)
                        for k, v in torso_dbg.items()}
                trace.append(entry)

        self.plant.apply_joint_torques(out.tau)
        self.plant.apply_wheel_torques(out.tau_w)
        self.plant.step(lock_arm_joints=self._diag_lock_arm_joints)
        self.controller.after_step(carry, out.tau)

        # Phase-2.1: optional 100 Hz (QP-rate) SS logging of reaction-wheel
        # torque + stored momentum. Gated (default OFF) ⇒ no behavioural
        # change; tau_w_last (init 2495, updated this tick at the AOCS block
        # every sub-step), hw, tq, phase are all in scope.
        if cfg.log_hifreq_ss and phase == 'SS':
            log.t_ss_hifreq.append(float(tq))
            log.tau_w_ss_hifreq.append(
                np.asarray(carry.tau_w_last, dtype=float).copy())
            log.hw_ss_hifreq.append(np.asarray(carry.hw, dtype=float).copy())


    def _nmpc_handoff(self, st):
        """End of the NMPC period: hand the tick to telemetry."""
        (t, phase, step_idx, swing_arm, stance_arm, stance_a, stance_b,
         target_anchor, log, ss_end, settle_mode, L_com_prev, carry, vp,
         cref_r, nmpc_ok, nmpc_status_code, nmpc_cost_val, info_n, t_nmpc_ms,
         t_qp_start) = (
            st.t, st.phase, st.step_idx, st.swing_arm, st.stance_arm,
            st.stance_a, st.stance_b, st.target_anchor, st.log, st.ss_end,
            st.settle_mode, st.L_com_prev, st.carry, st.vp, st.cref_r,
            st.nmpc_ok, st.nmpc_status_code, st.nmpc_cost_val, st.info_n,
            st.t_nmpc_ms, st.t_qp_start)
        hw, lr, qp_ok = carry.hw, carry.lr, carry.qp_ok
        lambda_qp_sol = carry.lambda_qp_sol
        tau_last, tau_w_last = carry.tau_last, carry.tau_w_last
        transport_mag_last = carry.transport_mag_last
        p_torso_ref_used = carry.p_torso_ref_used

        # ══════════════════════════════════════════════════════════════════
        # HAND OFF — telemetry
        # Nothing below decides anything. TickState carries the tick to
        # _log_ss_tick in tick_logging.py.
        # ══════════════════════════════════════════════════════════════════
        t_qp_ms = (time.perf_counter() - t_qp_start) * 1000

        # Everything below this point only RECORDS the tick — no control
        # decision is taken. It lives in `_log_ss_tick`, the single-support
        # counterpart of `_log_ds_tick`. See TickState for why the values
        # cross as one record rather than 29 arguments.
        #
        # `p_torso_ref_used` is None if the QP sub-loop did not run (the carry's
        # initial value) — formerly a caught NameError, same behaviour.
        _p_torso_ref_used = p_torso_ref_used

        return self._log_ss_tick(log, TickState(
            t=t, phase=phase, step_idx=step_idx, ss_end=ss_end,
            settle_mode=settle_mode, swing_arm=swing_arm,
            stance_arm=stance_arm, stance_a=stance_a, stance_b=stance_b,
            target_anchor=target_anchor, hw=hw, L_com_prev=L_com_prev,
            nmpc_ok=nmpc_ok, nmpc_status_code=nmpc_status_code,
            nmpc_cost_val=nmpc_cost_val, t_nmpc_ms=t_nmpc_ms,
            nmpc_info=info_n, lambda_ref=lr, v_com_ref=vp, r_com_ref=cref_r,
            qp_ok=qp_ok, t_qp_ms=t_qp_ms, lambda_qp=lambda_qp_sol,
            tau_joints=tau_last, tau_wheels=tau_w_last,
            transport_term_mag=transport_mag_last,
            p_torso_ref_used=_p_torso_ref_used,
        ))

    def _get_ee_data(self, rs, arm):
        """Return (J_ee, Jdq_ee, oMf_ee) for the given arm (tick_logging.py
        calls this on self; the controller uses ``ee_data`` directly)."""
        return ee_data(rs, arm)

    # ── Summary ──────────────────────────────────────────────────────────

    def _print_summary(self, log):
        t = np.array(log.t)
        Ln = np.array(log.L_com_norm)
        Ldn = np.array(log.L_dot_norm)
        euler = np.array(log.struct_euler_deg)
        sp = np.array(log.struct_pos)

        print(f"\n{'='*60}")
        print(f"SIMULATION SUMMARY")
        print(f"{'='*60}")
        print(f"Duration:        {t[-1]:.1f}s")
        print(f"Dock events:     {len(log.dock_events)}")
        for ev in log.dock_events:
            print(f"  Step {ev['step']}: t={ev['t']}s d={ev['d_mm']}mm arm={ev['arm']}")
        print(f"max |tau_joint|:  {max(log.tau_max_joint):.2f} Nm")
        print(f"max ||L_com||:    {Ln.max():.2f} Nms (lim {self.cfg.L_max})")
        print(f"max ||L̇_com||:    {Ldn.max():.2f} Nm (lim {self.cfg.tau_w_max})")
        print(f"Struct drift:     {np.linalg.norm(sp[-1]-sp[0])*100:.1f} cm")
        print(f"Struct rotation:  roll={euler[-1,0]:.2f}° "
              f"pitch={euler[-1,1]:.2f}° yaw={euler[-1,2]:.2f}°")
        print(f"Max |angle|:      {np.max(np.abs(euler)):.2f}°")
        nf_nmpc = sum(1 for x in log.nmpc_ok if not x)
        nf_qp = sum(1 for x in log.qp_ok if not x)
        print(f"NMPC fails:       {nf_nmpc}/{len(log.nmpc_ok)}")
        print(f"QP fails:         {nf_qp}/{len(log.qp_ok)}")
        if log.hw_physical:
            hw_phys = np.array(log.hw_physical)
            hw_norms = np.linalg.norm(hw_phys, axis=1)
            print(f"max ||hw_phys||:  {hw_norms.max():.2f} Nms (lim {self.cfg.hw_max[0]:.1f})")
            n_viol = np.sum(hw_norms > self.cfg.hw_max[0])
            print(f"hw violation:     {n_viol}/{len(hw_norms)} "
                  f"({100*n_viol/max(len(hw_norms),1):.1f}%)")

    # ── Plotting ─────────────────────────────────────────────────────────

    @staticmethod

    # ── Plotting (delegated to plotting module) ──
    @staticmethod
    def plot(log, save_path=None, cfg=None):
        return plot_simulation(log, save_path=save_path, cfg=cfg)
