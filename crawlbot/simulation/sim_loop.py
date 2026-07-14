"""
SimulationLoop — Closed-loop MuJoCo simulation with two-stage controller.

Orchestrates the full locomotion pipeline for the VISPA crawling robot:
    TorsoPlanner -> CoM ref -> CentroidalNMPC -> WholeBodyQP -> MuJoCo

Per NMPC step (10 Hz):
    1. Read MuJoCo state -> Pinocchio (structure frame)
    2. TorsoPlanner.reference_at(t) -> 6D torso ref
    3. CentroidalNMPC.solve(...) -> lambda_ref, a_ff, hw_dot_plan
    4. Inner loop (100 Hz QP):
        a. WholeBodyQP.solve(...) -> tau_q
        b. AOCS wheel torque -> ctrl[12:15]
        c. mj_step

Phase machine per step (M7, two-phase):
    DS (double support, energy-based exit) ->
    SS (single support, T_step-synchronized trajectories) ->
    dock (d<5mm AND ori<5deg)
"""

import numpy as np
import time
from typing import Optional

try:
    import mujoco
except ImportError:
    mujoco = None

try:
    import pinocchio as pin
except ImportError:
    pin = None

from crawlbot.core.robot_interface import RobotInterface, FRAME_TOOL_A, FRAME_TOOL_B
from crawlbot.core.state_conversions import (
    mujoco_to_pinocchio, pinocchio_to_mujoco, quat_wxyz_to_euler_deg)
from crawlbot.core.com_to_torso_mapping import CoMToTorsoMapping
from crawlbot.core.ik import (
    dock_configuration, dock_configuration_fixed_rotation,
    solve_ik, solve_ik_waypoints,
    manipulability_config, manipulability_config_trajectory,
    manipulability_config_mid_waypoint, check_path_feasibility,
    precompute_torso_map,
)
from crawlbot.planning.contact_scheduler import ContactScheduler, read_anchors_from_mujoco
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
    MomentumDisturbanceEstimator, EstimatorConfig, compute_aocs_command)

from .config import SimConfig
from .logging import SimLog, capture_environment
from .plotting import plot_simulation
# ── Simulation loop ──────────────────────────────────────────────────────────

class SimulationLoop:
    """Closed-loop MuJoCo simulation with hierarchical NMPC+QP controller."""

    def __init__(self, mjcf_path: str, urdf_path: str,
                 config: Optional[SimConfig] = None):
        assert mujoco is not None, "mujoco package required"
        assert pin is not None, "pinocchio package required"
        self.mjcf_path = mjcf_path
        self.urdf_path = urdf_path
        self.cfg = config or SimConfig()
        self.n_qp_per_nmpc = int(round(self.cfg.dt_nmpc / self.cfg.dt_qp))

        self.mj_model = None
        self.mj_data = None
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
        self._weld_map = {}
        self._site_ids = {}
        self.plan = None
        self.has_rwa = False  # Set True if model has reaction wheels
        # M6/M7: coarse pre-planner — mandatory, built in setup().
        self.preplanner: Optional[CoarsePrePlanner] = None
        # Most recent pre-planner result for the active step; None in DS.
        self._coarse_plan: Optional[CoarsePlanResult] = None
        # M7: T_step from the pre-planner for the active step.
        self._current_T_step: float = 0.0
        # M7 v19: planned arm-joint trajectory endpoints (captured in
        # _setup_torso_for_step). Used by _planned_arm_config so the
        # mapping can see where the arm *will* be, not where it is.
        self._step_q_start: Optional[np.ndarray] = None
        self._step_q_end: Optional[np.ndarray] = None
        self._step_t_ss_start: float = 0.0
        self._step_T_step: float = 0.0
        # M7 / FK-on-smoothed-q: cached smoothed q-sequence + per-segment
        # tangents for the active SS step. Populated by
        # _setup_torso_for_step under cfg.reference_source='joint_space_fk';
        # consumed by _planned_arm_config so the M5 mapping sees the same
        # q(τ) the planners are tracking.
        self._step_q_seq: Optional[list] = None
        self._step_dq_seg: Optional[list] = None
        # M7 EE-bisection follow-up: torso linear position at SS entry
        # (set in _setup_torso_for_step). Read by _step() when
        # cfg.mapping_bypass_in_ss is True; otherwise unused.
        self._ss_entry_p_torso: Optional[np.ndarray] = None
        # Option A (T12 fix, 2026-04-22): post-dock blend state for
        # the DS torso position reference. Populated at weld
        # activation; cleared on SS entry. See
        # cfg.ds_ramp_duration_s and docs/architecture/M7_T12_MEMO.md §5.
        self._ds_ramp_t_start: Optional[float] = None
        self._ds_ramp_p_start: Optional[np.ndarray] = None
        self._ds_ramp_p_end: Optional[np.ndarray] = None
        # Simulation time at which the active coarse plan was anchored
        # (so r_com_at(t - t0) gives the right reference at current time).
        self._coarse_plan_t0: float = 0.0
        # Per-step telemetry (infeasibilities, solve times, etc.)
        self._preplanner_stats = []
        # ── Diagnostic hooks (runtime-only, not config fields) ────────
        # _diag_disable_aocs: if True, force tau_w_cmd = 0 every QP sub-
        #   step (used to measure the raw robot-disturbance-induced
        #   platform drift without AOCS compensation).
        # _diag_lock_arm_joints: if True, set qvel[arm joints] = 0 after
        #   every mj_step and clear arm joint actuation (used to
        #   measure the contact/weld/MJ baseline drift with the robot
        #   "frozen").
        self._diag_disable_aocs: bool = False
        self._diag_lock_arm_joints: bool = False
        # _diag_pure_pd: strips ALL feedforward terms entering the QP
        #   (a_com_ff → 0, a_torso_ff → 0, λ_ref → 0) and the NMPC's
        #   L_com_ref → 0, leaving only PD feedback on r_b_ref from
        #   the mapping layer. Used to localize feedforward-injected
        #   instabilities vs. PD-loop instabilities.
        self._diag_pure_pd: bool = False
        # Per-step trace of the pure-PD diagnostic (filled in _step).
        self._diag_pure_pd_trace: list = []
        # _diag_freeze_ref: keep r_b_ref / v_b_ref held at the first-
        # sample value during the run. Used to probe the PD loop's
        # stability around a FIXED torso target (no reference motion).
        self._diag_freeze_ref: bool = False
        self._diag_frozen_r_b_ref: Optional[np.ndarray] = None
        self._diag_frozen_R_b_ref: Optional[np.ndarray] = None
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
        # Most recent CoMToTorsoMapping δ(q) output; populated by _step()
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
        self._mapping_nmpc_tick: int = -1
        self._mapping_cache_delta: Optional[np.ndarray] = None
        self._mapping_cache_delta_dot: Optional[np.ndarray] = None
        self._last_r_b_ref_out: Optional[np.ndarray] = None
        # F-SAT telemetry: counts of cycles where the increment was
        # clipped, max clip magnitude, per-step bookkeeping.
        self._sat_total_calls: int = 0
        self._sat_clipped_calls: int = 0
        self._sat_max_clip_mm: float = 0.0

    # ── Setup ────────────────────────────────────────────────────────────

    def setup(self, n_steps: int = 3, start_a: int = 2, start_b: int = 2,
              sequence_path: str = None):
        """Initialize all components.

        If ``sequence_path`` is provided, the gait plan is built from
        that ``.seq`` file (see ``crawlbot.planning.sequence_loader``)
        and ``n_steps`` / ``start_a`` / ``start_b`` are ignored.
        """
        cfg = self.cfg

        # MuJoCo
        self.mj_model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = cfg.dt_qp

        # Detect RWA model (3 reaction wheels)
        rw_jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, 'rw_x')
        self.has_rwa = rw_jid >= 0

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
        self._struct_quat_init = self.mj_data.qpos[3:7].copy()
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Read anchor sites in world frame and convert to structure-local frame
        mj_a_world, mj_b_world = read_anchors_from_mujoco(self.mj_model, self.mj_data)
        p_s0 = self.mj_data.qpos[0:3].copy()
        w, x, y, z = self.mj_data.qpos[3:7]
        R_s0 = pin.Quaternion(w, x, y, z).toRotationMatrix()
        anchors_a_local = [R_s0.T @ (a - p_s0) for a in mj_a_world]
        anchors_b_local = [R_s0.T @ (b - p_s0) for b in mj_b_world]

        # Pinocchio
        self.robot = RobotInterface(
            self.urdf_path, gravity='zero')

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
            early_finish_fraction=cfg.swing_early_finish_fraction,
            model=self.robot.model)

        # Torso planner (reconfigured per step)
        self.torso_planner = TorsoPlanner(
            model=self.robot.model,
            frame_torso=self.robot.frame_torso)
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

        sp = self.mj_data.qpos[0:3].copy()
        sq = self.mj_data.qpos[3:7].copy()
        mj_qpos, _ = pinocchio_to_mujoco(
            self.q_dock_init, np.zeros(self.robot.model.nv), struct_pos=sp, struct_quat=sq,
            rwa=self.has_rwa)
        self.mj_data.qpos[:] = mj_qpos
        self.mj_data.qvel[:] = 0.0

        # Welds
        self._build_weld_map()
        self._deactivate_all_welds()
        self._activate_weld('a', start_a)
        self._activate_weld('b', start_b)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Initial CoM calibration (no settling yet — state is hot from the
        # weld activation, but we only need total_mass + frame IDs for
        # building the NMPC/QP, which are invariant to velocity.)
        rs0 = self.robot.update(
            *mujoco_to_pinocchio(self.mj_data.qpos, self.mj_data.qvel))

        # Site IDs
        self._cache_site_ids()

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
            cfg.ss_alpha_com, cfg.ss_alpha_torso, cfg.ss_alpha_ee,
            cfg.ss_alpha_posture, cfg.ss_alpha_wrench, cfg.ss_alpha_reaction,
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

        # GMO contact estimator
        from crawlbot.estimation.contact_estimator import (
            GeneralizedMomentumObserver, ContactStateMachine,
            ContactObserverConfig, ContactState)
        obs_cfg = ContactObserverConfig(
            K_O=cfg.gmo_K_O, dt=cfg.dt_qp, nv=self.robot.model.nv,
            F_threshold=cfg.gmo_F_threshold,
            d_proximity=cfg.gmo_d_proximity,
            d_contact=cfg.gmo_d_contact,
            d_reset=cfg.gmo_d_reset,
            debounce_count=cfg.gmo_debounce_count)
        self.gmo = GeneralizedMomentumObserver(obs_cfg)
        self.contact_sm = ContactStateMachine(obs_cfg)
        self._contact_confirmed = False

        # ── Two-stage setup settling ──────────────────────────────────
        # Stage 1: strong joint-velocity damping for n_settle_damping_steps
        #          steps to absorb the weld activation impulse.
        # Stage 2: M2 QP with settle_mode + passivity_active, exit when
        #          T_kin < 0.5·epsilon_v²·lambda_min(H).
        #
        # qvel is NEVER zeroed — the passivity-constrained QP dissipates
        # momentum via joint torques, respecting dynamics.
        self._settling_log = self._settle_setup(start_a, start_b)

        print(f"[SimulationLoop] Initialized:")
        print(f"  Robot mass:     {rs0.total_mass:.1f} kg")
        print(f"  RWA model:      {'YES (3 wheels)' if self.has_rwa else 'NO'}")
        print(f"  AOCS estimator: {'H_{r/O}' if cfg.aocs_use_H_estimator else 'L_dot (legacy)'}")
        print(f"  GMO dock:       {'YES' if cfg.use_gmo_dock else 'NO (legacy kinematic)'}")
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
            Apply tau_j = -Kd * dq_j for n_settle_damping_steps steps.
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
        n_j = self.robot.n_joints
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
        pq0, pv0 = mujoco_to_pinocchio(
            self.mj_data.qpos, self.mj_data.qvel)
        rs0 = self.robot.update(pq0, pv0)
        T_initial = _kinetic_energy(rs0.v, rs0.H)
        log['T_start'] = T_initial

        # ── Stage 1: open-loop damping ────────────────────────────────
        Kd = cfg.Kd_settle_damping
        for k in range(cfg.n_settle_damping_steps):
            dq_j = self.mj_data.qvel[
                -n_j - (3 if self.has_rwa else 0):
                None if not self.has_rwa else -3]
            tau_damp = np.clip(-Kd * dq_j, -cfg.tau_max, cfg.tau_max)
            self.mj_data.ctrl[:n_j] = tau_damp
            mujoco.mj_step(self.mj_model, self.mj_data)
            log['stage1_steps'] += 1

            if k % 5 == 0:
                pq, pv = mujoco_to_pinocchio(
                    self.mj_data.qpos, self.mj_data.qvel)
                rs = self.robot.update(pq, pv)
                T = _kinetic_energy(rs.v, rs.H)
                log['t_log'].append(k * dt)
                log['T_log'].append(T)

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
        self.mj_data.ctrl[:] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)
        pq_end, pv_end = mujoco_to_pinocchio(
            self.mj_data.qpos, self.mj_data.qvel)
        rs_end = self.robot.update(pq_end, pv_end)
        log['T_end'] = _kinetic_energy(rs_end.v, rs_end.H)
        log['initial_vcom'] = rs_end.v_com.copy()
        log['initial_Lcom'] = rs_end.L_com.copy()
        # Final sample for the plot
        t_final = (log['stage1_steps'] + log['stage2_steps']) * dt
        log['t_log'].append(t_final)
        log['T_log'].append(log['T_end'])
        return log

    def _run_ds_passivity_loop(
        self,
        *,
        contact_config,
        max_steps: int,
        epsilon_v: float,
        plateau_window: int = 50,
        plateau_ratio: float = 0.999,
        min_steps: int = 0,
        fallback_Kd: float = 20.0,
        t_log: Optional[list] = None,
        T_log: Optional[list] = None,
        t_log_step_offset: int = 0,
        # Phase-B logging hooks (no control effect; logging-only).
        # When `log_obj` is None, no per-tick log rows are written —
        # the setup-Stage-2 settle call site keeps that behaviour.
        # The inter-step DS call site passes log_obj=log + context so
        # the loop emits per-tick rows schema-identical to the SS log.
        log_obj=None,
        log_step_idx: int = -1,
        log_just_landed_arm: str = '',
        log_anchor_a_idx: int = -1,
        log_anchor_b_idx: int = -1,
        log_t_abs: float = 0.0,
    ) -> dict:
        """Run the M2 QP in settle_mode + passivity_active until T<T_settle.

        This is the shared dissipation engine used by BOTH the setup-phase
        stage-2 settling and the inter-step DS settling (spec §7.1.1).

        Assumptions on entry
        --------------------
        - The robot is in DS (both tools welded to their anchors).
        - `self.qp_ss` is built with `use_m2_stack=True`.
        - `self.mj_data` holds the current MuJoCo state.

        The loop runs at dt_qp (100 Hz) and calls `self.qp_ss.solve(...)`
        directly — NMPC is bypassed. Exit conditions (in priority order):
            1. Target met: T_kin < T_settle = 0.5·epsilon_v²·lambda_min(H)
            2. Plateau: no progress over `plateau_window` steps
            3. Max steps reached
        `min_steps` forces at least that many iterations before exits 1/2
        fire — useful for the inter-step call, which should run for at
        least a few dt_qp steps to absorb the post-dock impact.

        Parameters
        ----------
        max_steps : int
            Safety cap on the number of dt_qp iterations.
        epsilon_v : float
            Target ‖dq_full‖ bound [m/s]; defines T_settle via H's
            smallest eigenvalue.
        plateau_window : int
            Window for plateau detection (steps).
        plateau_ratio : float
            Plateau fires when T(k) > plateau_ratio · T(k - plateau_window).
        min_steps : int
            Minimum iterations before target/plateau exits can fire.
        fallback_Kd : float
            Joint-damping gain used when the QP raises an exception.
        t_log, T_log : list or None
            Optional destinations for plot samples (appended every 5 steps).
        t_log_step_offset : int
            Absolute step offset used to compute the plot time stamps
            (t = (t_log_step_offset + k) · dt_qp).

        Returns
        -------
        dict with keys
            n_steps       : int     actual iterations run
            T_start       : float   kinetic energy at entry [J]
            T_end         : float   kinetic energy at exit [J]
            T_settle      : float   target threshold [J]
            lambda_min    : float   min eigenvalue of H at entry [kg·m²]
            exit_reason   : str     'target_met' | 'plateau' | 'max_steps'
        """
        cfg = self.cfg
        n_j = self.robot.n_joints
        dt = cfg.dt_qp
        qp = self.qp_ss

        def _kinetic_energy(v_full, H_robot):
            return 0.5 * float(v_full @ H_robot @ v_full)

        # Threshold from H at entry (stable over the small displacement
        # we expect during settling).
        pq0, pv0 = mujoco_to_pinocchio(
            self.mj_data.qpos, self.mj_data.qvel)
        rs0 = self.robot.update(pq0, pv0)
        eig_H = np.linalg.eigvalsh(rs0.H)
        lambda_min = float(np.min(np.abs(eig_H)))
        T_settle = 0.5 * (epsilon_v ** 2) * lambda_min
        T_start = _kinetic_energy(rs0.v, rs0.H)

        # DS contact config (both anchors active). Caller must pass a
        # ContactConfig whose r_contact_A/B hold the CURRENT structure-
        # frame anchor positions — these are used by compute_momentum_map
        # to build the lever arms for the hw safety constraint.
        cc_ds = contact_config

        hw_current = np.zeros(3) if not self.has_rwa else (
            cfg.rwa_I_w * self.mj_data.qvel[6:9]).copy()

        # Loop-local ω_s history for the inter-step AOCS K_d·ω̇_s term.
        # The _step regulator's `_omega_s_last` is local to _step and
        # invisible here, so the loop tracks its own (init from the entry
        # ω_s ⇒ ω̇_s = 0 on the first tick; updated each iteration before
        # mj_step). Only this history is new — the DS wrench feedforward
        # needs no L_com/v_com history (AOCS-FF audit).
        _omega_s_prev = (self.mj_data.qvel[3:6].copy()
                         if self.has_rwa else np.zeros(3))

        T_history = []
        exit_reason = 'max_steps'
        T = T_start
        for k in range(max_steps):
            pq, pv = mujoco_to_pinocchio(
                self.mj_data.qpos, self.mj_data.qvel)
            rs = self.robot.update(pq, pv)
            T = _kinetic_energy(rs.v, rs.H)
            T_history.append(T)

            if (t_log is not None) and (T_log is not None) and (k % 5 == 0):
                t_log.append((t_log_step_offset + k) * dt)
                T_log.append(T)

            # Exits (only after min_steps)
            if k >= min_steps:
                if T < T_settle:
                    exit_reason = 'target_met'
                    break
                if k >= plateau_window:
                    T_old = T_history[k - plateau_window]
                    if T > plateau_ratio * T_old:
                        exit_reason = 'plateau'
                        break

            Jc, Jdc = self.robot.get_contact_jacobians(True, True)

            # c_curr (J2): refresh the QP's hw_current parameter to the LIVE
            # wheel momentum this tick. With the AOCS active in this loop the
            # wheels move, so the entry-frozen hw_current goes stale and the
            # QP momentum-safety box loses C5 margin. Reading qvel[6:9] here
            # (same source as the _step per-tick refresh) and passing it as
            # the frozen hw parameter mirrors c_simple(k): a fresh numeric
            # value per solve, frozen during the solve, NOT a decision
            # variable (non-co-integration audit — the QP decision vector
            # {qdd_t, qdd, λ, τ_q, slack} is unchanged). Flag OFF ⇒ the
            # entry-frozen value (byte-identical). At k=0 qvel is unchanged
            # since entry ⇒ the first solve is identical either way.
            if self.has_rwa and cfg.interstep_hw_refresh:
                hw_for_qp = (cfg.rwa_I_w * self.mj_data.qvel[6:9]).copy()
            else:
                hw_for_qp = hw_current

            try:
                # lambda_qp_sol captured for logging-only (was `_` before
                # Phase B). No control change — value is not consumed
                # downstream of this branch.
                _, _, lambda_qp_sol, tau, _ = qp.solve(
                    q_t=rs.q_torso, dq_t=rs.dq_torso,
                    q=rs.q_joints, dq=rs.dq_joints,
                    r_com_ref=rs.r_com, v_com_ref=np.zeros(3),
                    lambda_ref=np.zeros(12), a_com_ff=np.zeros(3),
                    H_robot=rs.H, C_robot=rs.C,
                    J_com=rs.J_com, Jdot_dq_com=rs.Jdot_dq_com,
                    contact_config=cc_ds,
                    J_contacts=Jc, Jdot_dq_contacts=Jdc,
                    hw_current=hw_for_qp,
                    hw_min=cfg.hw_min, hw_max=cfg.hw_max,
                    r_com=rs.r_com, L_com_current=rs.L_com,
                    J_torso=rs.J_torso,
                    Jdot_dq_torso=rs.Jdot_dq_torso,
                    p_torso=rs.oMf_torso.translation.copy(),
                    R_torso=rs.oMf_torso.rotation.copy(),
                    p_torso_ref=rs.oMf_torso.translation.copy(),
                    R_torso_ref=rs.oMf_torso.rotation.copy(),
                    v_torso_ref=np.zeros(6),
                    a_torso_ff=np.zeros(6),
                    settle_mode=True,
                    passivity_active=True,
                    settle_alpha_wrench=(cfg.interstep_settle_alpha_wrench
                                         if cfg.interstep_settle_alpha_wrench > 0
                                         else None),
                    settle_alpha_sigf=(cfg.interstep_settle_alpha_sigf
                                       if cfg.interstep_settle_alpha_sigf > 0
                                       else None))
                tau = np.clip(tau, -cfg.tau_max, cfg.tau_max)
            except Exception:
                tau = -fallback_Kd * rs.dq_joints
                tau = np.clip(tau, -cfg.tau_max, cfg.tau_max)
                lambda_qp_sol = np.zeros(12)

            self.mj_data.ctrl[:n_j] = tau
            # AOCS re-activation (free-floating invariant — J2 step 4a).
            # The structure is free-floating every tick, so the AOCS must
            # run in DS too. Flag ON (default): run the canonical
            # legacy_pid_numerical AOCS with the DS wrench FF from the
            # settle-QP wrench (lambda_qp_sol). Flag OFF: the legacy
            # hardcoded-zero path, byte-identical to the pre-fix loop.
            tau_w_applied = np.zeros(3)
            if self.has_rwa:
                if cfg.aocs_active_in_interstep:
                    tau_w_applied = self._interstep_aocs_command(
                        rs, cc_ds, lambda_qp_sol, _omega_s_prev)
                    self.mj_data.ctrl[n_j:n_j + 3] = tau_w_applied
                    # Update the ω_s history BEFORE mj_step (current ω_s
                    # is this tick's pre-step value, the next tick's prev).
                    _omega_s_prev = self.mj_data.qvel[3:6].copy()
                else:
                    self.mj_data.ctrl[n_j:n_j + 3] = 0.0
            mujoco.mj_step(self.mj_model, self.mj_data)

            # ── Phase-B per-tick logging (logging-only; no control change) ──
            # Emit a row schema-identical to the SS log block. NaN sentinels
            # for NMPC-side quantities (NMPC is bypassed in this loop).
            if log_obj is not None:
                t_abs_tick = log_t_abs + (k + 1) * dt
                self._log_ds_tick(
                    log_obj, t_abs_tick,
                    log_step_idx, log_just_landed_arm,
                    log_anchor_a_idx, log_anchor_b_idx,
                    tau, lambda_qp_sol, tau_w_applied)

        n_steps_run = min(k + 1, max_steps) if max_steps > 0 else 0
        # Record final energy (no ctrl reset — caller manages the handoff)
        pq_end, pv_end = mujoco_to_pinocchio(
            self.mj_data.qpos, self.mj_data.qvel)
        rs_end = self.robot.update(pq_end, pv_end)
        T_end = _kinetic_energy(rs_end.v, rs_end.H)

        return {
            'n_steps': n_steps_run,
            'T_start': T_start,
            'T_end': T_end,
            'T_settle': T_settle,
            'lambda_min': lambda_min,
            'exit_reason': exit_reason,
        }

    def _interstep_aocs_command(self, rs, cc_ds, lambda_qp_sol, omega_s_prev):
        """Canonical AOCS wheel-torque command for the inter-step DS loop.

        Re-activates the structure attitude controller inside
        ``_run_ds_passivity_loop`` (J2 step 4a — the loop historically
        zeroed the wheels, violating the free-floating invariant; see the
        AOCS-during-DS audit). Mirrors the ``_step`` AOCS for the canonical
        ``legacy_pid_numerical`` mode, fed entirely from in-loop values:

          - DS WRENCH feedforward  τ_w_FF = −Σ_i (r_Ci × f_i + τ_i)  from
            ``lambda_qp_sol`` (the settle-QP wrench captured this tick),
            with anchor levers r_Ci = cc_ds.r_contact_{A,B} (struct frame).
            This is the welded-loop internal-stress couple the FD-on-L_com
            feedforward is blind to (force_estimator.py:582-592). Same
            construction as _step (sim_loop.py wrench-FF block).
          - attitude PID + desaturation via
            ``compute_aocs_command_legacy_pid_numerical`` — θ_s from the
            geometric SO(3) error ½ vee(R_errᵀ−R_err) (identical to _step),
            ω_s / h_w from MuJoCo state, ω̇_s from the loop-local
            ``omega_s_prev``.

        No NMPC dependency (the loop bypasses the NMPC): the feedforward
        source is the QP wrench, not the NMPC plan (AOCS-FF audit). The QP,
        passivity constraint, and envelope box are untouched. Returns the
        clipped τ_w command (±aocs_tau_w_max).
        """
        from crawlbot.aocs.force_estimator import (
            compute_aocs_command_legacy_pid_numerical)
        cfg = self.cfg
        omega_s = self.mj_data.qvel[3:6].copy()
        hw_phys = (cfg.rwa_I_w * self.mj_data.qvel[6:9]).copy()

        # Geometric SO(3) attitude error θ_s (same as _step).
        qw, qx, qy, qz = self._struct_quat_init
        R_init = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
        qw, qx, qy, qz = self.mj_data.qpos[3:7]
        R_now = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
        R_err = R_init.T @ R_now
        theta_s = 0.5 * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ])

        # DS wrench feedforward from λ_qp (anchor levers in struct frame;
        # λ_qp in world frame — equivalent at the <5° structure-frame
        # rotation we operate in, as in _step).
        _lam = np.asarray(lambda_qp_sol, dtype=float).ravel()
        _r_C = (cc_ds.r_contact_A, cc_ds.r_contact_B)
        tau_struct_ff = np.zeros(3)
        for _ci in range(2):
            if not cc_ds.active_contacts[_ci]:
                continue
            _f = _lam[6 * _ci: 6 * _ci + 3]
            _tq = _lam[6 * _ci + 3: 6 * _ci + 6]
            tau_struct_ff -= np.cross(_r_C[_ci], _f) + _tq

        # L_com/v_com args are unused on the wrench-FF path (the FD branch
        # is `tau_struct_ff is None`), so current values are passed.
        return compute_aocs_command_legacy_pid_numerical(
            L_com=rs.L_com, L_com_prev=rs.L_com,
            r_com=rs.r_com, v_com=rs.v_com, v_com_prev=rs.v_com,
            omega_s=omega_s, omega_s_prev=omega_s_prev,
            theta_s=theta_s,
            hw_current=hw_phys, dt=cfg.dt_qp,
            robot_mass=self.robot._total_mass,
            K_hw=cfg.aocs_K_hw, K_omega=cfg.aocs_K_omega,
            K_d=cfg.aocs_K_d, K_theta=cfg.aocs_K_theta,
            hw_min=cfg.hw_min, hw_max=cfg.hw_max,
            tau_w_max=cfg.aocs_tau_w_max,
            tau_struct_ff=tau_struct_ff)

    def _log_ds_tick(self, log, t_abs, step_idx, just_landed_arm,
                     anchor_a_idx, anchor_b_idx, tau, lambda_qp_sol,
                     tau_w_applied=None):
        """Append one per-tick DS log row (logging-only; no control change).

        Called from ``_run_ds_passivity_loop`` AFTER ``mj_step``. The row
        has the SAME field set as the SS row written by ``_step`` so the
        SimLog stays schema-consistent (every list grows by 1 per tick,
        all lengths equal). NaN sentinels are used for NMPC-side fields
        (NMPC is bypassed in this loop); zeros for the AOCS-side wheel
        torque (wheels are commanded to zero at line ~754).

        Phase-B guarantee: this method only reads MuJoCo / Pinocchio
        state and appends to the log. It does not call qp.solve, does
        not change ``self.mj_data.ctrl``, does not advance MuJoCo time,
        does not mutate ``self.gmo`` / ``self.contact_sm`` / planner
        state. Side-effect-free w.r.t. control.
        """
        cfg = self.cfg
        nq = self.robot.n_joints

        # Recompute Pinocchio state from current MuJoCo state.
        # This duplicates the next iteration's top-of-loop computation
        # by one tick; harmless (the next iteration recomputes anyway).
        pq, pv = mujoco_to_pinocchio(self.mj_data.qpos, self.mj_data.qvel)
        rs = self.robot.update(pq, pv)

        # Torso planner reference at the absolute tick time, clamped
        # into the phase window: during inter-step DS (t past the
        # step's t_end) this holds the quintic's TERMINAL pose p_t1, so
        # the exported p_torso_ref is continuous across SS->DS instead
        # of jumping back to the set_hold pose p_t0 (a full step stride
        # behind). Logging-only — the DS QP does not consume this value
        # (settle_mode gates all torso tasks off).
        tref = self.torso_planner.reference_at_clamped(t_abs)
        p_torso_ref = tref.p.copy()
        R_torso_ref = tref.R.copy()
        q_torso_ref = pin.Quaternion(R_torso_ref)
        # Torso tracking error (geodesic angle).
        R_err = R_torso_ref.T @ rs.oMf_torso.rotation
        angle_err = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))

        # End-effector data for the JUST-LANDED arm (the one whose
        # contact tenure we are measuring). If '' (initial DS), default
        # to arm A so the field is well-defined; the post-processor
        # knows to drop initial-DS rows.
        ee_arm = just_landed_arm if just_landed_arm in ('a', 'b') else 'a'
        J_ee, _, oMf_ee = self._get_ee_data(rs, ee_arm)
        # d_grip_swing = distance to the welded contact anchor (just-landed
        # arm). For initial DS (anchor < 0), report NaN.
        try:
            if ee_arm == 'a' and anchor_a_idx >= 0:
                d_swing = self._gripper_distance('a', anchor_a_idx)
            elif ee_arm == 'b' and anchor_b_idx >= 0:
                d_swing = self._gripper_distance('b', anchor_b_idx)
            else:
                d_swing = float('nan')
        except Exception:
            d_swing = float('nan')
        # d_grip_stance: the OTHER arm (continuing stance).
        try:
            if ee_arm == 'a' and anchor_b_idx >= 0:
                d_stance = self._gripper_distance('b', anchor_b_idx)
            elif ee_arm == 'b' and anchor_a_idx >= 0:
                d_stance = self._gripper_distance('a', anchor_a_idx)
            else:
                d_stance = float('nan')
        except Exception:
            d_stance = float('nan')

        # Wheel momentum from MJ wheel state.
        if self.has_rwa:
            rw_vel = self.mj_data.qvel[6:9].copy()
            hw = (cfg.rwa_I_w * rw_vel).copy()
        else:
            rw_vel = np.zeros(3)
            hw = np.zeros(3)

        # SS convention: tau_max_joint is the SCALAR max|tau| at this tick.

        # ── Schema-identical append ───────────────────────────────────
        log.t.append(float(t_abs))
        log.phase.append('DS')
        log.step_idx.append(int(step_idx))

        # Torso tracking
        log.p_torso.append(rs.oMf_torso.translation.copy())
        log.p_torso_ref.append(p_torso_ref)
        log.e_torso_pos.append(float(
            np.linalg.norm(rs.oMf_torso.translation - p_torso_ref)))
        log.e_torso_ori.append(float(np.degrees(angle_err)))
        q_torso_actual = pin.Quaternion(rs.oMf_torso.rotation)
        log.q_torso.append(np.array([q_torso_actual.w, q_torso_actual.x,
                                      q_torso_actual.y, q_torso_actual.z]))
        log.q_torso_ref.append(np.array([q_torso_ref.w, q_torso_ref.x,
                                          q_torso_ref.y, q_torso_ref.z]))

        # End-effector
        log.d_grip_swing.append(float(d_swing))
        log.d_grip_stance.append(float(d_stance))
        log.swing_arm.append(just_landed_arm)
        log.p_ee.append(oMf_ee.translation.copy())
        # No swing planner reference in DS — anchor IS the target.
        log.p_ee_ref.append(oMf_ee.translation.copy())
        q_ee_a = pin.Quaternion(oMf_ee.rotation)
        log.q_ee.append(np.array([q_ee_a.w, q_ee_a.x, q_ee_a.y, q_ee_a.z]))
        log.q_ee_ref.append(np.array([q_ee_a.w, q_ee_a.x,
                                       q_ee_a.y, q_ee_a.z]))

        # Joint / EE / torso velocities
        v_a = rs.J_tool_a @ rs.v
        v_b = rs.J_tool_b @ rs.v
        v_t = rs.J_torso @ rs.v
        log.qvel_joints_a.append(rs.dq_joints[:nq // 2].copy())
        log.qvel_joints_b.append(rs.dq_joints[nq // 2:].copy())
        log.v_ee_a.append(v_a[:3].copy())
        log.omega_ee_a.append(v_a[3:].copy())
        log.v_ee_b.append(v_b[:3].copy())
        log.omega_ee_b.append(v_b[3:].copy())
        log.v_torso.append(v_t[:3].copy())
        log.omega_torso.append(v_t[3:].copy())

        # CoM (use measured as ref → e_com = 0)
        log.r_com.append(rs.r_com.copy())
        log.r_com_ref.append(rs.r_com.copy())
        log.e_com.append(0.0)
        log.v_com.append(rs.v_com.copy())
        log.v_com_ref.append(np.zeros(3))

        # Momentum (L_dot left as zeros — would require prior tick state)
        log.L_com.append(rs.L_com.copy())
        log.L_com_norm.append(float(np.linalg.norm(rs.L_com)))
        log.L_com_ref.append(np.zeros(3))
        log.L_dot.append(np.zeros(3))
        log.L_dot_norm.append(0.0)
        log.hw.append(hw.copy())

        # RWA physical. With the inter-step AOCS re-activated (flag on),
        # tau_w_applied is the ACTUAL wheel torque commanded this tick;
        # when the AOCS is off (flag off / no RWA) it is zeros — the
        # legacy commanded value. None defaults to zeros for any caller
        # that does not pass it (schema-safe).
        log.hw_physical.append(hw.copy())
        log.tau_w.append((np.zeros(3) if tau_w_applied is None
                          else np.asarray(tau_w_applied, dtype=float)).copy())
        log.rw_speed.append(rw_vel.copy())

        # EE tracking error (vs same-frame ref → 0).
        log.e_ee_pos.append(0.0)
        log.e_ee_ori.append(0.0)

        # GMO / contact estimator: not updated in DS loop; report best-
        # effort current state (the GMO object is stateful; reading it
        # does NOT mutate). Fall back to zeros on any attribute error.
        try:
            log.gmo_residual_norm.append(float(np.linalg.norm(self.gmo.residual)))
        except Exception:
            log.gmo_residual_norm.append(0.0)
        # gmo_swing_residual requires a v-slice; not tracked in DS.
        log.gmo_swing_residual.append(0.0)
        try:
            log.gmo_contact_state.append(self.contact_sm.state.value)
        except Exception:
            log.gmo_contact_state.append(0)

        # H_rO / H estimator diagnostics — H_estimator not advanced here.
        try:
            log.H_rO.append(self.H_estimator.H_rO.copy())
            log.H_dot_est.append(self.H_estimator.H_dot.copy())
        except Exception:
            log.H_rO.append(np.zeros(3))
            log.H_dot_est.append(np.zeros(3))
        log.omega_struct.append(self.mj_data.qvel[3:6].copy())
        log.qfrc_constraint_torque.append(np.zeros(3))

        # Joint torques (clipped value commanded to MJ).
        log.tau.append(tau.copy())
        log.tau_max_joint.append(float(np.max(np.abs(tau))))

        # Structure state.
        log.struct_pos.append(self.mj_data.qpos[0:3].copy())
        log.struct_quat.append(self.mj_data.qpos[3:7].copy())
        # Euler (zyx) from quat — small-angle is fine for diag.
        qw_s, qx_s, qy_s, qz_s = self.mj_data.qpos[3:7]
        log.struct_euler_deg.append(np.degrees(np.array([
            np.arctan2(2*(qw_s*qx_s + qy_s*qz_s), 1 - 2*(qx_s*qx_s + qy_s*qy_s)),
            np.arcsin(np.clip(2*(qw_s*qy_s - qz_s*qx_s), -1, 1)),
            np.arctan2(2*(qw_s*qz_s + qx_s*qy_s), 1 - 2*(qy_s*qy_s + qz_s*qz_s)),
        ])))
        log.omega_s.append(self.mj_data.qvel[3:6].copy())

        # ── NMPC-bypassed sentinels ───────────────────────────────────
        # lambda_ref: NaN sentinel — there is NO plan in this regime.
        log.lambda_ref.append(np.full(12, np.nan))
        log.lambda_ref_norm.append(float('nan'))
        # QP-side wrenches: real values from this loop's QP solve.
        log.lambda_qp.append(np.asarray(lambda_qp_sol, dtype=float).copy())
        log.lambda_qp_norm.append(float(np.linalg.norm(lambda_qp_sol)))
        log.nmpc_ok.append(False)  # not run; not a failure
        log.qp_ok.append(True)
        log.nmpc_time_ms.append(0.0)
        log.qp_time_ms.append(0.0)
        log.nmpc_status.append(3)  # 3 = BYPASSED sentinel (post-proc reads)
        log.nmpc_cost.append(float('nan'))
        log.nmpc_status_str.append('NMPC_BYPASSED')
        log.nmpc_iterations.append(0)
        log.transport_term_mag.append(0.0)

        # Energy / passivity
        log.T_kinetic.append(0.5 * float(rs.v @ rs.H @ rs.v))

    def _build_qp(self, ac, at, ae, ap, aw, ar_react,
                   kpc, kdc, kpt, kdt, kpe, kde,
                   kpe_ang=5.0, kde_ang=3.0):
        cfg = self.cfg
        # M2: when use_m2_stack is on, the explicit CoM task is dropped,
        # the torso 6D task becomes primary P1, EE is null-space projected
        # against the torso task, and a soft CoM residual cost is added.
        c = WholeBodyQPConfig(
            nq=self.robot.n_joints, nc_max=2, dt_qp=cfg.dt_qp,
            tau_max=cfg.tau_max * np.ones(self.robot.n_joints),
            alpha_com=ac if not cfg.use_m2_stack else 0.0,
            alpha_torso=at,
            alpha_ee=ae,
            alpha_posture=ap, alpha_wrench=aw,
            alpha_reaction=ar_react,
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
            use_m2_stack=cfg.use_m2_stack,
            ee_null_space=cfg.use_m2_stack,
            alpha_com_soft=cfg.alpha_com_soft,
            alpha_passivity=cfg.alpha_passivity,
            passivity_W_budget=cfg.passivity_W_budget,
            qp_envelope_exact=cfg.qp_envelope_exact,
            r_tube=cfg.r_tube,
            w_tube_lin=cfg.w_tube_lin,
            cooperative_arms_mode=cfg.cooperative_arms_mode,
            alpha_torso_ang=cfg.ss_alpha_torso_ang,
            alpha_torso_lin=cfg.ss_alpha_torso_lin,
            ss_centroidal_momentum_task=cfg.ss_centroidal_momentum_task,
            ss_alpha_mom=cfg.ss_alpha_mom,
            ss_alpha_tl_weak=cfg.ss_alpha_tl_weak,
            ss_two_task_mode=cfg.ss_two_task_mode,
            alpha_torso_pose=cfg.alpha_torso_pose,
            stance_thrust_correction=cfg.stance_thrust_correction)
        qp = WholeBodyQP(c)
        qp.set_nominal_posture(self.q_dock_init[self.robot.joints_q_slice])
        return qp

    # ── Weld management ──────────────────────────────────────────────────

    def _build_weld_map(self):
        self._weld_map = {}
        for i in range(self.mj_model.neq):
            name = mujoco.mj_id2name(
                self.mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, i)
            if name and name.startswith('grip_'):
                parts = name.split('_to_')
                arm = parts[0].split('_')[1]
                anchor_idx = int(parts[1][0]) - 1
                self._weld_map[(arm, anchor_idx)] = i

    def _deactivate_all_welds(self):
        for eq_id in range(self.mj_model.neq):
            self.mj_data.eq_active[eq_id] = 0

    def _activate_weld(self, arm, anchor_idx):
        key = (arm, anchor_idx)
        if key in self._weld_map:
            self.mj_data.eq_active[self._weld_map[key]] = 1

    def _deactivate_weld(self, arm, anchor_idx):
        key = (arm, anchor_idx)
        if key in self._weld_map:
            self.mj_data.eq_active[self._weld_map[key]] = 0

    def _cache_site_ids(self):
        for name in ['gripper_a', 'gripper_b']:
            sid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
            self._site_ids[name] = sid
        # Cache ALL anchor sites present in the model (not a hardcoded
        # count). A hardcoded range(5) silently dropped anchor_6, so the
        # dock gate read d=inf and never fired on any step targeting the
        # 6th anchor (e.g. step 4 -> dock_timeout despite the EE reaching
        # the anchor). Discovering the count from the model keeps the
        # cache consistent with the gait/MJCF for any anchor grid.
        for arm in ['a', 'b']:
            idx = 0
            while True:
                name = f'anchor_{idx+1}{arm}'
                sid = mujoco.mj_name2id(
                    self.mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
                if sid < 0:
                    break
                self._site_ids[name] = sid
                idx += 1

    def _gripper_distance(self, arm, anchor_idx):
        grip_sid = self._site_ids.get(f'gripper_{arm}', -1)
        anch_sid = self._site_ids.get(f'anchor_{anchor_idx+1}{arm}', -1)
        if grip_sid < 0 or anch_sid < 0:
            return np.inf
        return float(np.linalg.norm(
            self.mj_data.site_xpos[grip_sid] - self.mj_data.site_xpos[anch_sid]))

    def _gripper_speed(self, arm):
        """Swing-EE linear speed relative to the structure [m/s].

        Anchors are static in the structure frame, so this is the EE
        approach speed onto the anchor. Used as a dock-gate criterion:
        a clean dock needs low relative speed, else the weld's inelastic
        impact projection injects a momentum impulse (recoil / force
        spike). Pinocchio relative twist -> J_ee @ v gives EE velocity
        relative to the structure (LOCAL_WORLD_ALIGNED linear part).
        """
        pq, pv = mujoco_to_pinocchio(self.mj_data.qpos, self.mj_data.qvel)
        rs = self.robot.update(pq, pv)
        J_ee, _, _ = self._get_ee_data(rs, arm)
        return float(np.linalg.norm((J_ee @ pv)[0:3]))

    def _gripper_ori_err_deg(self, arm, anchor_idx):
        """Angle between the gripper frame and its target anchor frame.

        Anchor frames are Identity in the structure frame, so this
        reduces to the angle between the gripper's structure-frame
        rotation matrix and I. Returns the angle in degrees.
        """
        pq, pv = mujoco_to_pinocchio(self.mj_data.qpos, self.mj_data.qvel)
        rs = self.robot.update(pq, pv)
        _, _, oMf = self._get_ee_data(rs, arm)
        R_ee = np.asarray(oMf.rotation)
        R_tgt = np.asarray(self.sched.anchor_se3(arm, anchor_idx).rotation)
        R_err = R_ee.T @ R_tgt
        return float(np.degrees(np.linalg.norm(pin.log3(R_err))))

    def _weld_relative_twist(self, arm, anchor_idx):
        """6-D weld-relative twist Jc·v⁻ for one gripper↔anchor pair.

        Returns the 6-vector [relative linear; relative angular] velocity
        of the swing gripper site w.r.t. its target anchor site, over ALL
        qvel (incl. structure base + wheels). This is the exact quantity
        the inelastic impact map projects out at dock.

        Reuses the Fix-A relative-site weld-Jacobian construction
        (the post-dock impact block): J = [jpg-jpa; jrg-jra] via
        ``mj_jacSite`` for the gripper/anchor sites, twist = J @ qvel.
        Unlike the impact block (which iterates ``eq_active``), the gate
        runs BEFORE the weld engages, so the pair is addressed by name.
        Assumes the caller has already run ``mj_forward`` so the site
        Jacobians are current. Returns zeros if either site is missing.
        """
        nv = self.mj_model.nv
        gsid = self._site_ids.get(f'gripper_{arm}', -1)
        asid = self._site_ids.get(f'anchor_{anchor_idx + 1}{arm}', -1)
        if gsid < 0 or asid < 0:
            return np.zeros(6)
        jpg = np.zeros((3, nv)); jrg = np.zeros((3, nv))
        jpa = np.zeros((3, nv)); jra = np.zeros((3, nv))
        mujoco.mj_jacSite(self.mj_model, self.mj_data, jpg, jrg, gsid)
        mujoco.mj_jacSite(self.mj_model, self.mj_data, jpa, jra, asid)
        J = np.vstack([jpg - jpa, jrg - jra])      # (6, nv)
        return J @ self.mj_data.qvel               # 6-D weld-relative twist

    def _dock_gate(self, swing_arm, target_idx, *, log=None, t=0.0,
                   step_idx=-1):
        """Evaluate the dock gate. Returns (docked, d, ori_deg, twist_norm).

        Fix C (J2 #1): the velocity criterion is the 6-D weld-relative
        twist ‖Jc·v⁻‖ < cfg.dock_twist_max (via ``_weld_relative_twist``),
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
            self._weld_relative_twist(swing_arm, target_idx)))
        pos_ok = d < cfg.weld_radius
        ori_ok = ori_err_deg < cfg.dock_ori_threshold_deg
        if cfg.dock_use_6d_twist:
            vel_ok = twist_norm < cfg.dock_twist_max
        else:
            vel_ok = self._gripper_speed(swing_arm) < cfg.dock_vel_max
        if cfg.use_gmo_dock:
            docked = self._contact_confirmed and ori_ok and vel_ok
        else:
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

    def _planned_arm_config(self, t: float, rs):
        """Return (q_planned, dq_planned) at sim time `t` for v19 mapping.

        Mixes the live floating-base state (rs.q / rs.v) with an arm
        portion obtained by quintic interpolation between
        self._step_q_start and self._step_q_end over [t_ss_start,
        t_ss_start + T_step]. The arm slice therefore reflects the
        trajectory the SwingPlanner is commanding at time `t`, not the
        arm's actual (tracking-lagged) configuration.

        Why: delta(q) in the mapping formula is dominated by the arms
        (torso block contributes 0 by construction). Using the live
        arm state makes the mapping a feedback path on arm tracking
        error; using the planned arm state makes it a clean feedforward.
        The floating-base portion is kept at rs.q/rs.v so the world-
        frame assembly of delta stays consistent with where the torso
        really is.
        """
        if self._step_q_start is None or self._step_q_end is None:
            return rs.q.copy(), rs.v.copy()
        T = max(self._step_T_step, 1e-6)
        tau = float(np.clip((t - self._step_t_ss_start) / T, 0.0, 1.0))
        # M7 / FK-on-smoothed-q (cfg.reference_source='joint_space_fk'):
        # use the same task-space-smoothed q-sequence the planners use,
        # so the M5 mapping consumes the same q(τ) the QP is tracking.
        if (self.cfg.reference_source == 'joint_space_fk'
                and self._step_q_seq is not None):
            from crawlbot.planning.constrained_geodesic import q_v_real_at_tau
            q_plan, v_real = q_v_real_at_tau(
                self.robot.model, self._step_q_seq, self._step_dq_seg,
                tau=tau, T_phase=T,
            )
            return q_plan, v_real
        s = 10.0*tau**3 - 15.0*tau**4 + 6.0*tau**5
        sd = (30.0*tau**2 - 60.0*tau**3 + 30.0*tau**4) / T
        sl_q = self.robot.joints_q_slice
        sl_v = self.robot.joints_v_slice
        q_arm_start = self._step_q_start[sl_q]
        q_arm_end = self._step_q_end[sl_q]
        dq_arm = q_arm_end - q_arm_start
        q_plan = rs.q.copy()
        q_plan[sl_q] = q_arm_start + s * dq_arm
        dq_plan = rs.v.copy()
        dq_plan[sl_v] = sd * dq_arm
        return q_plan, dq_plan

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
        pq_live, pv_live = mujoco_to_pinocchio(
            self.mj_data.qpos, self.mj_data.qvel)
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
        traj_w_worst = float('nan')
        traj_w_end = float('nan')
        traj_ik_elapsed_s = float('nan')

        # M7 Manipulability-IK-1 Phase 4 — on-demand trajectory-aware IK.
        # Phase 3 (results/M7_1pct_3step_v22_t15_trajIK/T15_trajIK_report.md
        # §5) showed that the chained-precompute cache's q_start_assumed
        # diverged from live state by 30–60× the tolerance at every SS
        # entry, so the cache was never consumed. Phase 4 bypasses the
        # cache and calls manipulability_config_trajectory with the
        # live pq_live so the optimiser sees the actual SS-entry state.
        # The cache build in setup() is retained unchanged (dead code
        # in this run) per the Phase-4 prompt — future work may
        # repurpose it as a warm-start seed source.
        if cfg.use_trajectory_aware_ik:
            t0 = time.perf_counter()
            try:
                q_end, traj_w_worst, traj_w_end = manipulability_config_trajectory(
                    model,
                    se3_a.translation, se3_b.translation,
                    q_start=pq_live,
                    n_samples=cfg.trajectory_ik_n_samples,
                    w_min_threshold=cfg.trajectory_ik_w_min_threshold,
                )
                if q_end is None:
                    # IK_FORMULATION.md §9.3 safety check rejected
                    # the converged endpoint (w_end below threshold).
                    # Fall through to the fixed_rotation IK below.
                    ik_mode = 'trajectory_aware_rejected_singular'
                    print(
                        f"  [IK] trajectory-aware IK rejected: "
                        f"w_end={traj_w_end:.2e} < threshold "
                        f"{cfg.trajectory_ik_w_min_threshold:.2e}; "
                        f"falling back to fixed_rotation"
                    )
                else:
                    ik_mode = 'trajectory_aware_on_demand'
            except RuntimeError:
                q_end = None
                ik_mode = 'trajectory_aware_on_demand_failed'
            traj_ik_elapsed_s = float(time.perf_counter() - t0)

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
        traj_suffix = ""
        if cfg.use_trajectory_aware_ik:
            traj_suffix = (
                f"  traj_ik_t = {traj_ik_elapsed_s:.2f} s"
                f"  w_worst(traj) = {traj_w_worst:.2e}"
                f"  w_end(traj) = {traj_w_end:.2e}"
            )
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
            'traj_w_worst': traj_w_worst,
            'traj_w_end': traj_w_end,
            'traj_ik_elapsed_s': traj_ik_elapsed_s,
        })

        # ── 1b. Mid-waypoint reshape (Option B per
        #        T15_step2_path_geometry.md §7.3) ─────────────────────────
        # When enabled, sample the planner-style reference path's
        # whole-body Jacobian conditioning. If infeasible (or
        # mid_waypoint_force_on=True), insert a manipulability-aware
        # mid-waypoint into TorsoPlanner and SwingPlanner, generating a
        # piecewise quintic that visits a known-feasible configuration
        # at the midpoint. Default flags off → byte-identical to the
        # IK-fix branch.
        # Always clear any prior swing override so old steps' mid-waypoints
        # don't leak into the next step.
        self.swing_planner.clear_phase_overrides()
        mid_wp_p_t = None
        mid_wp_R_t = None
        mid_wp_p_ee = None
        mid_wp_R_ee = None
        path_feasibility_result = None
        mid_w_worst = float('nan')
        mid_wp_used = False
        if cfg.use_path_feasibility_check or cfg.use_mid_waypoint_reshape:
            should_attempt_reshape = False
            if cfg.use_path_feasibility_check:
                t_pf0 = time.perf_counter()
                path_feasibility_result = check_path_feasibility(
                    model, pq_live, q_end, se3_a, se3_b,
                    swing_arm=target_arm, fid_torso=self.robot.frame_torso,
                    n_samples=21,
                    w_min_threshold=cfg.path_feasibility_w_threshold,
                )
                t_pf = time.perf_counter() - t_pf0
                print(
                    f"  [path-feasibility] feasible="
                    f"{path_feasibility_result['all_samples_feasible']} "
                    f"w_min={path_feasibility_result['w_min']:.2e} "
                    f"@τ={path_feasibility_result['w_min_tau']:.2f} "
                    f"({t_pf:.2f} s)"
                )
                if not path_feasibility_result['all_samples_feasible']:
                    should_attempt_reshape = True
            if (cfg.use_mid_waypoint_reshape and
                    (should_attempt_reshape or cfg.mid_waypoint_force_on)):
                t_mid_ik0 = time.perf_counter()
                try:
                    q_mid, mid_w_worst, mid_success = (
                        manipulability_config_mid_waypoint(
                            model, se3_a, se3_b,
                            q_start=pq_live, q_end=q_end,
                            swing_arm=target_arm,
                            n_interior_samples=5,
                            w_min_threshold=cfg.path_feasibility_w_threshold,
                        )
                    )
                except Exception as e:
                    q_mid = None
                    mid_success = False
                    print(f"  [mid-waypoint] IK raised: {type(e).__name__}: {e}")
                t_mid_ik = time.perf_counter() - t_mid_ik0
                if mid_success and q_mid is not None:
                    # Extract mid-waypoint poses via FK on q_mid.
                    rs_m = self.robot.update(
                        q_mid, np.zeros(self.robot.model.nv))
                    mid_wp_p_t = rs_m.oMf_torso.translation.copy()
                    mid_wp_R_t = rs_m.oMf_torso.rotation.copy()
                    # Swing arm EE at q_mid (for SwingPlanner override).
                    fid_swing = (FRAME_TOOL_B if target_arm == 'b'
                                 else FRAME_TOOL_A)
                    pin.forwardKinematics(self.robot.model,
                                          self.robot.data, q_mid)
                    pin.updateFramePlacements(self.robot.model,
                                              self.robot.data)
                    mid_wp_p_ee = self.robot.data.oMf[fid_swing].translation.copy()
                    mid_wp_R_ee = self.robot.data.oMf[fid_swing].rotation.copy()
                    mid_wp_used = True
                    print(
                        f"  [mid-waypoint] inserted: "
                        f"w_worst_mid={mid_w_worst:.2e} "
                        f"q_mid_torso={mid_wp_p_t.round(3).tolist()} "
                        f"({t_mid_ik:.1f} s)"
                    )
                else:
                    print(
                        f"  [mid-waypoint] IK failed (w_worst="
                        f"{mid_w_worst:.2e}); falling back to single-quintic "
                        f"({t_mid_ik:.1f} s)"
                    )
        # Append mid-waypoint diagnostics to the IK trace entry.
        self._debug_ik_trace[-1].update({
            'mid_wp_used': bool(mid_wp_used),
            'mid_w_worst': float(mid_w_worst),
            'mid_wp_p_t': (mid_wp_p_t.tolist() if mid_wp_p_t is not None
                           else None),
            'path_feasibility_w_min': (
                None if path_feasibility_result is None
                else float(path_feasibility_result['w_min'])),
            'path_feasibility_all_ok': (
                None if path_feasibility_result is None
                else bool(path_feasibility_result['all_samples_feasible'])),
        })

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
        if mid_wp_used:
            t_mid_phase = t_ss_start + 0.5 * T_step
            torso_phase_kwargs.update(
                p_mid=mid_wp_p_t, R_mid=mid_wp_R_t, t_mid=t_mid_phase)

        # M7 / FK-on-smoothed-q reference path
        # (cfg.reference_source='joint_space_fk').
        # Compute the task-space-smoothed constrained geodesic ONCE per
        # SS step. The same q_seq is shared by both planners, ensuring
        # the torso and swing references at every τ derive from a
        # single q satisfying the stance constraint by construction.
        # Phase-0 measurement: ~0.3 s wall-clock for n_tau=21,
        # n_iter=120; eliminates the kinematically-uncoupled-refs
        # failure mode at T15 step 2 (synthesis §6, plan §2.2).
        fk_torso_extra = {}
        fk_swing_extra = {}
        fk_q_seq = None
        if cfg.reference_source == 'joint_space_fk':
            from crawlbot.planning.constrained_geodesic import (
                smoothed_constrained_geodesic,
            )
            fid_stance_fk = (self.robot.frame_tool_a
                             if target_arm == 'b'
                             else self.robot.frame_tool_b)
            fid_swing_fk = (self.robot.frame_tool_b
                            if target_arm == 'b'
                            else self.robot.frame_tool_a)
            t_smooth0 = time.perf_counter()
            fk_q_seq, fk_info = smoothed_constrained_geodesic(
                self.robot.model, pq_live, q_end,
                fid_stance=fid_stance_fk,
                fid_torso=self.robot.frame_torso,
                fid_swing=fid_swing_fk,
                n_tau=cfg.geodesic_n_tau,
                n_iter=cfg.geodesic_n_iter,
                tol=cfg.geodesic_tol,
                verbose=False,
            )
            print(
                f"  [smoother] {fk_info['iters']} iters in "
                f"{time.perf_counter() - t_smooth0:.2f}s, "
                f"converged={fk_info['converged']}, "
                f"fallbacks={fk_info['fallbacks_total']}"
            )
            if fk_info['fallbacks_total'] > 0:
                print(
                    f"  [smoother] WARNING: {fk_info['fallbacks_total']} "
                    f"3-task IK fallbacks during smoothing"
                )
            fk_torso_extra['q_seq'] = fk_q_seq
            fk_swing_extra['q_seq'] = fk_q_seq
            fk_swing_extra['frame_swing'] = fid_swing_fk

        # Orientation: R_start = R_end = R_flat (R_t1 == R_flat via the
        # R_torso_fixed=R_flat dock-IK above), so the SS orientation reference
        # is the constant global R_flat — zero DS<->SS seam, and the QP closes
        # any residual torso drift back to R_flat during the swing. Position
        # (p_t0->p_t1) and delta_com still advance as before.
        self.torso_planner.add_phase(
            t_ss_start, t_ss_start + T_step,
            p_t0, self._R_torso_flat, p_t1, R_t1,
            **torso_phase_kwargs, **fk_torso_extra)
        # Install the swing-EE phase override when mid-waypoint
        # reshape succeeded OR FK mode is on. Endpoints come from FK
        # at pq_live (start) and FK at q_end (end). The legacy
        # SwingPlanner path is taken when no override is registered.
        if cfg.reference_source == 'joint_space_fk':
            fid_swing_for_log = fk_swing_extra['frame_swing']
            pin.forwardKinematics(self.robot.model, self.robot.data, pq_live)
            pin.updateFramePlacements(self.robot.model, self.robot.data)
            p_swing_start_pose = self.robot.data.oMf[fid_swing_for_log].translation.copy()
            R_swing_start_pose = self.robot.data.oMf[fid_swing_for_log].rotation.copy()
            pin.forwardKinematics(self.robot.model, self.robot.data, q_end)
            pin.updateFramePlacements(self.robot.model, self.robot.data)
            p_swing_end_pose = self.robot.data.oMf[fid_swing_for_log].translation.copy()
            R_swing_end_pose = self.robot.data.oMf[fid_swing_for_log].rotation.copy()
            self.swing_planner.add_phase(
                t_start=t_ss_start, t_end=t_ss_start + T_step,
                p_ee_start=p_swing_start_pose, R_ee_start=R_swing_start_pose,
                p_ee_end=p_swing_end_pose, R_ee_end=R_swing_end_pose,
                swing_arm=target_arm,
                **fk_swing_extra,
            )
        elif mid_wp_used:
            fid_swing_for_log = (FRAME_TOOL_B if target_arm == 'b'
                                 else FRAME_TOOL_A)
            pin.forwardKinematics(self.robot.model, self.robot.data, pq_live)
            pin.updateFramePlacements(self.robot.model, self.robot.data)
            p_swing_start_pose = self.robot.data.oMf[fid_swing_for_log].translation.copy()
            R_swing_start_pose = self.robot.data.oMf[fid_swing_for_log].rotation.copy()
            pin.forwardKinematics(self.robot.model, self.robot.data, q_end)
            pin.updateFramePlacements(self.robot.model, self.robot.data)
            p_swing_end_pose = self.robot.data.oMf[fid_swing_for_log].translation.copy()
            R_swing_end_pose = self.robot.data.oMf[fid_swing_for_log].rotation.copy()
            self.swing_planner.add_phase(
                t_start=t_ss_start, t_end=t_ss_start + T_step,
                p_ee_start=p_swing_start_pose, R_ee_start=R_swing_start_pose,
                p_ee_end=p_swing_end_pose, R_ee_end=R_swing_end_pose,
                swing_arm=target_arm,
                p_ee_mid=mid_wp_p_ee, R_ee_mid=mid_wp_R_ee,
                t_mid=t_mid_phase,
            )

        # M7 v19: capture per-step planned joint trajectory endpoints so
        # the CoM->torso mapping can be fed q_planned (arm configuration
        # the swing planner will track) instead of q_current. Turns the
        # mapping from a feedback-on-actual-state path into a feedforward
        # path that anticipates arm motion and is not disturbed by arm
        # tracking errors.
        self._step_q_start = pq_live.copy()
        self._step_q_end = q_end.copy()
        self._step_t_ss_start = float(t_ss_start)
        self._step_T_step = float(T_step)
        # M7 / FK-on-smoothed-q: cache the smoothed q-sequence + per-
        # segment tangents for use by _planned_arm_config (so the M5
        # mapping sees the same q(τ) the planners are tracking).
        # Cleared (set to None) under legacy mode so the legacy quintic
        # path is taken.
        if fk_q_seq is not None:
            from crawlbot.planning.constrained_geodesic import (
                precompute_segment_tangents,
            )
            self._step_q_seq = fk_q_seq
            self._step_dq_seg = precompute_segment_tangents(
                self.robot.model, fk_q_seq)
        else:
            self._step_q_seq = None
            self._step_dq_seg = None
        # Snapshot the torso linear position at SS entry — read by the
        # mapping_bypass_in_ss diagnostic in _step() to freeze the SS
        # linear torso reference. p_t0 is the torso pose computed from
        # the live state above (line ~810), so it equals the actual
        # torso position at the moment SS begins.
        self._ss_entry_p_torso = p_t0.copy()
        # Option A: reset DS ramp state at SS entry; the next weld
        # activation will repopulate _ds_ramp_t_start / _ds_ramp_p_start.
        self._ds_ramp_t_start = None
        self._ds_ramp_p_start = None
        self._ds_ramp_p_end = None

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
        pq_live, pv_live = mujoco_to_pinocchio(
            self.mj_data.qpos, self.mj_data.qvel)
        rs_live = self.robot.update(pq_live, pv_live)
        v_com_0 = rs_live.v_com.copy()
        L_com_0 = rs_live.L_com.copy()
        # Conservation constant c = hw_0 + L_com_0 + r_com_0 × m·v_com_0
        if self.has_rwa:
            hw_0 = cfg.rwa_I_w * self.mj_data.qvel[6:9].copy()
        else:
            hw_0 = np.zeros(3)
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
        log.snapshots.append((
            round(t, 3),
            self.mj_data.qpos.copy(),
            self.mj_data.qvel.copy(),
            label))

    def run(self, verbose=True):
        """Run full multi-step locomotion simulation."""
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
        _pq_flat, _pv_flat = mujoco_to_pinocchio(
            self.mj_data.qpos, self.mj_data.qvel)
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
                    # Uses _run_ds_passivity_loop which drives T_kin <
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
                    ds_result = self._run_ds_passivity_loop(
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
                        pq_d, pv_d = mujoco_to_pinocchio(
                            self.mj_data.qpos, self.mj_data.qvel)
                        rs_d = self.robot.update(pq_d, pv_d)
                        t_dwell_end = t + _dwell_target
                        if cfg.ds_mobile_com_magnitude > 0.0:
                            # α (J2 #2): CoM-mobile DS. Translate the CoM toward
                            # the next swing arm's target anchor (held ori +
                            # posture) via the existing trajectory machinery —
                            # NOT a mode. set_hold is the degenerate constant
                            # case; here the CoM content is non-constant.
                            p0 = rs_d.oMf_torso.translation.copy()
                            R0 = rs_d.oMf_torso.rotation.copy()
                            com0 = rs_d.r_com.copy()
                            anchors = (self.sched.anchors_a if swing_arm == 'a'
                                       else self.sched.anchors_b)
                            anchor_next = np.asarray(anchors[target_idx],
                                                     dtype=float)
                            dirvec = anchor_next - com0
                            nrm = float(np.linalg.norm(dirvec))
                            offset = (cfg.ds_mobile_com_magnitude * dirvec / nrm
                                      if nrm > 1e-6 else np.zeros(3))
                            # CoM + torso position translate by `offset`
                            # (constant δ_com ⇒ r_com(t)=p(t)+R·δ translates);
                            # orientation held at global R_flat (both waypoints).
                            self.torso_planner.set_from_waypoints(
                                t, t_dwell_end,
                                [(p0, self._R_torso_flat),
                                 (p0 + offset, self._R_torso_flat)],
                                [com0, com0 + offset])
                            if verbose:
                                print(f"  DS-mobile: CoM +{nrm and cfg.ds_mobile_com_magnitude:.3f}m "
                                      f"toward anchor {swing_arm}{target_idx}")
                        else:
                            self.torso_planner.set_hold(
                                rs_d.oMf_torso.translation.copy(),
                                self._R_torso_flat.copy(),
                                r_com=rs_d.r_com.copy())
                        # Use stance pair from the upcoming SS phase as the
                        # contact config (both arms welded → DOUBLE).
                        last_sa_d = stance_a
                        last_sb_d = stance_b
                        while t < t_dwell_end:
                            hw, L_com_prev = self._step(
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
                    self._deactivate_weld(swing_arm, old_anchor)
                    self.nmpc.reset_warm_start()
                    pq_r, pv_r = mujoco_to_pinocchio(
                        self.mj_data.qpos, self.mj_data.qvel)
                    rs_r = self.robot.update(pq_r, pv_r)
                    self.gmo.reset(rs_r.H, rs_r.v)
                    self.contact_sm.reset()
                    self._contact_confirmed = False
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
                            mujoco.mj_forward(self.mj_model, self.mj_data)
                            label = (
                                f'frame_step{self._frame_capture_step_idx}_'
                                f'{self._frame_capture_kidx}')
                            self._capture_snapshot(log, t, label)
                            self._frame_capture_times.pop(0)
                            self._frame_capture_kidx += 1
                        hw, L_com_prev = self._step(
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
                            mujoco.mj_forward(self.mj_model, self.mj_data)
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
                                    'method': ('gmo' if cfg.use_gmo_dock
                                               else 'kinematic')})
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
                            hw, L_com_prev = self._step(
                                t, 'SS', step_idx, swing_arm, stance_arm,
                                cc_ss, target_idx, stance_a, stance_b,
                                hw, L_com_prev, log,
                                ss_end=t_ss_start + T_step,
                                # Dock-floor audit: default False (the SS-hold
                                # escape — passivity OFF); the audit forces it
                                # ON to test whether passivity limits the close.
                                passivity_hold=cfg.dock_hold_passivity_on)
                            t += cfg.dt_nmpc

                            mujoco.mj_forward(self.mj_model, self.mj_data)
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
                                    'method': ('gmo' if cfg.use_gmo_dock
                                               else 'kinematic')})
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
                        self._activate_weld(swing_arm, target_idx)
                        mujoco.mj_forward(self.mj_model, self.mj_data)
                        self.nmpc.reset_warm_start()
                        # Option A: capture the SS-exit torso position
                        # and weld time for the post-dock DS blend. The
                        # blend endpoint (_ds_ramp_p_end) is not stored
                        # here — _step() recomputes the live mapping
                        # output each tick and blends it against
                        # _ds_ramp_p_start. See M7_T12_MEMO.md §5.
                        self._ds_ramp_t_start = float(t)
                        if self._ss_entry_p_torso is not None:
                            self._ds_ramp_p_start = (
                                self._ss_entry_p_torso.copy())

                        # ── Inelastic impact: FULL-DOF momentum-consistent ──
                        # Fix A (dock-leak Part 3). The previous map projected in
                        # the robot-only Pinocchio space (structure as fixed
                        # base) and wrote back only qvel[6+off:] via the
                        # setup-only pinocchio_to_mujoco conversion — a one-sided
                        # impulse that injected ~0.2 N·m·s of spurious system
                        # angular momentum at the docks (dock-leak Parts 1–2:
                        # the conversion also drops the structure-coupling terms
                        # since it assumes v_struct≈0, false at a dock). Replace
                        # with the full-DOF projection validated offline in
                        # Part-2 A.1 (leak 0.3565→0.0011 over the 5 docks),
                        # computed ENTIRELY in MuJoCo DOF (no Pinocchio round-
                        # trip) and written back to ALL qvel, so the constraint
                        # impulse is a full action-reaction pair and conserves
                        # subtree_angmom to the O(gap·f) couple residual. The
                        # weld is already active above, so every active
                        # gripper↔anchor relation is in the constraint set.
                        nv = self.mj_model.nv
                        M_full = np.zeros((nv, nv))
                        mujoco.mj_fullM(self.mj_model, self.mj_data, M_full)
                        # Full-DOF weld constraint Jacobian: relative twist of
                        # each welded gripper↔anchor site pair over ALL qvel
                        # (incl. structure base + wheels) — the same relative-
                        # site weld relation Part-2 A.1 validated. Active welds
                        # are read from eq_active (exact welded pairs).
                        inv_weld = {eq: key for key, eq in self._weld_map.items()}
                        rows = []
                        for eq_id in range(self.mj_model.neq):
                            if not self.mj_data.eq_active[eq_id] or eq_id not in inv_weld:
                                continue
                            arm, a_idx = inv_weld[eq_id]
                            gsid = self._site_ids.get(f'gripper_{arm}', -1)
                            asid = self._site_ids.get(f'anchor_{a_idx + 1}{arm}', -1)
                            if gsid < 0 or asid < 0:
                                continue
                            jpg = np.zeros((3, nv)); jrg = np.zeros((3, nv))
                            jpa = np.zeros((3, nv)); jra = np.zeros((3, nv))
                            mujoco.mj_jacSite(self.mj_model, self.mj_data, jpg, jrg, gsid)
                            mujoco.mj_jacSite(self.mj_model, self.mj_data, jpa, jra, asid)
                            rows.append(jpg - jpa)   # relative linear twist
                            rows.append(jrg - jra)   # relative angular twist
                        if rows:
                            J = np.vstack(rows)        # (6·n_weld) × nv
                            v_minus = self.mj_data.qvel.copy()
                            v_pre = J @ v_minus
                            MiJT = np.linalg.solve(M_full, J.T)
                            impulse = np.linalg.solve(J @ MiJT, v_pre)
                            v_plus = v_minus - MiJT @ impulse
                            if verbose:
                                mujoco.mj_subtreeVel(self.mj_model, self.mj_data)
                                H_pre = float(np.linalg.norm(
                                    self.mj_data.subtree_angmom[0]))
                            self.mj_data.qvel[:] = v_plus   # write back ALL DOFs
                            mujoco.mj_forward(self.mj_model, self.mj_data)
                            if verbose:
                                mujoco.mj_subtreeVel(self.mj_model, self.mj_data)
                                H_post = float(np.linalg.norm(
                                    self.mj_data.subtree_angmom[0]))
                                print(f"  Impact(fullDOF): ||dv||="
                                      f"{np.linalg.norm(v_plus - v_minus):.4f}, "
                                      f"||J·v-||={np.linalg.norm(v_pre):.4f}, "
                                      f"|H_sys| {H_pre:.4f}->{H_post:.4f}")

                    # Post-dock energy-based settling is now handled by
                    # the *next* step's DS block (above). One shared
                    # implementation via _run_ds_passivity_loop.

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
                    pq, pv = mujoco_to_pinocchio(
                        self.mj_data.qpos, self.mj_data.qvel)
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
                        hw, L_com_prev = self._step(
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
        """Single NMPC+QP step.  All quantities are in structure frame.

        Parameters
        ----------
        passivity_hold : bool
            M7: when True, activate the QP's passivity inequality even in
            SS. Used during the convergence-hold window (after the
            trajectory has ended and the EE is still converging on the
            dock) so the system dissipates residual kinetic energy.
        passivity_override : Optional[bool]
            Diagnostic (H_DS3). When not None, overrides the phase-based
            passivity gate below. Wired only by the trailing-DS branch
            when `cfg.diag_disable_passivity_on_abort` is set and the
            preceding SS aborted on dock_timeout.
        """
        cfg = self.cfg

        # Torso/CoM references (structure frame — no struct pose needed)
        tref = self.torso_planner.reference_at(t)
        # Query CoM reference at horizon end, not current time.
        # The NMPC uses a constant reference across all N horizon steps,
        # so passing the current-time reference causes systematic lag.
        t_horizon = t + cfg.nmpc_N * cfg.nmpc_dt
        cref = self.torso_planner.com_reference_at(t_horizon)

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
        if (self._coarse_plan is not None) and (not settle_mode):
            tau_rel = t_horizon - self._coarse_plan_t0
            ff = float(getattr(cfg, 'torso_early_finish_fraction', 1.0))
            T_plan = float(self._coarse_plan.T_step)
            if 0.0 < ff < 1.0:
                # Compressed time: profile covers [0, ff·T_plan] in
                # real-time, then holds. Positions accelerate; velocity
                # scales by 1/ff during the active window, goes to 0
                # afterwards (the pre-planner's v_com[-1] ≈ 0 anyway,
                # so clamping is safe).
                if tau_rel <= ff * T_plan:
                    tau_comp = tau_rel / ff
                    rp_coarse = self._coarse_plan.r_com_at(tau_comp)
                    vp_coarse = self._coarse_plan.v_com_at(tau_comp) / ff
                else:
                    rp_coarse = self._coarse_plan.r_com_at(T_plan)
                    vp_coarse = np.zeros(3)
            else:
                rp_coarse = self._coarse_plan.r_com_at(tau_rel)
                vp_coarse = self._coarse_plan.v_com_at(tau_rel)
            cref_r = rp_coarse
            cref_v = vp_coarse
        else:
            cref_r = cref.r_com
            cref_v = cref.v_com

        # Robot state in structure frame.
        # Extract structure angular velocity for non-inertial corrections.
        omega_s = self.mj_data.qvel[3:6].copy()
        pq, pv = mujoco_to_pinocchio(self.mj_data.qpos, self.mj_data.qvel)
        rs = self.robot.update(pq, pv, omega_struct=omega_s)
        if L_com_prev is None:
            L_com_prev = rs.L_com.copy()

        # Contact config from constant structure-frame anchors (no live reading)
        cc_nmpc = ContactConfig.from_phase(
            cc_ss.phase,
            self.sched.anchors_a[stance_a].copy(),
            self.sched.anchors_b[stance_b].copy())

        # NMPC — plans robot motion only; AOCS manages wheels independently.
        nmpc_ok = True
        nmpc_status_code = 0  # 0=ok, 1=max_iter, 2=infeasible
        nmpc_cost_val = np.inf
        t_nmpc_start = time.perf_counter()
        # M3: pass current hw (wheel momentum) so NMPC can compute c_simple
        # for the conservation-law box.
        # M5: pass L_com_ref from the TorsoPlanner — nonzero during SS
        # when the torso is rotating. Prevents the NMPC from treating
        # intentional rotation as a disturbance to be cancelled.
        if self.has_rwa:
            hw_for_nmpc = (cfg.rwa_I_w * self.mj_data.qvel[6:9]).copy()
        else:
            hw_for_nmpc = hw.copy()
        # Query L_com_ref at the horizon midpoint to track the
        # planner's rotation phase reasonably.
        t_mid = t + 0.5 * cfg.nmpc_N * cfg.nmpc_dt
        L_com_ref_nmpc = self.torso_planner.l_com_reference_at(t_mid)
        if self._diag_pure_pd:
            L_com_ref_nmpc = np.zeros(3)
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
                    't_mid': float(t_mid),
                    'L_com_ref': L_com_ref_nmpc.copy(),
                    'norm': float(np.linalg.norm(L_com_ref_nmpc)),
                })
        info_n = None
        # NMPC warm-start diagnosis: tag each solve with locomotion-step
        # index and phase. Read by NMPCSolver.solve when populating
        # step_log. Does not affect control logic.
        try:
            self.nmpc._nmpc.diag_label = (
                f"step{int(step_idx):02d}/{phase}/t={float(t):.2f}")
        except Exception:
            pass
        try:
            rp, vp, _, lr, info_n = self.nmpc.solve(
                r_com=rs.r_com, v_com=rs.v_com, L_com=rs.L_com,
                r_com_ref=cref_r, v_com_ref=cref_v,
                contact_config=cc_nmpc, warm_start=True,
                hw_current=hw_for_nmpc,
                L_com_ref=L_com_ref_nmpc)
            af = self.nmpc.compute_feedforward_acceleration(lr)
            nmpc_ok = info_n.success
            nmpc_cost_val = float(info_n.cost) if np.isfinite(info_n.cost) else np.inf
            if not info_n.success:
                nmpc_status_code = 2 if 'infeasib' in info_n.status.lower() else 1
        except Exception:
            nmpc_ok = False
            nmpc_status_code = 2
            af = np.zeros(3)
        t_nmpc_ms = (time.perf_counter() - t_nmpc_start) * 1000

        # ── M5 Fix 2: infeasibility fallback via receding-horizon shift ─
        # On NMPC failure, do NOT jump to cref.r_com (creates a reference
        # discontinuity that saturates actuators). Instead, warm-shift
        # the previous feasible trajectory by one NMPC step. We do NOT
        # update _last_x_opt in place — repeated failures re-shift the
        # same last-successful trajectory, so the fallback never drifts
        # more than ~1 NMPC step from a real plan.
        x_plan, u_plan, _ = self.nmpc.get_last_trajectory()
        if not nmpc_ok:
            x_shift, u_shift = self.nmpc.get_shifted_fallback()
            if x_shift is not None and u_shift is not None:
                x_plan = x_shift
                u_plan = u_shift
                rp = x_plan[0:3, 1]
                vp = x_plan[3:6, 1]
                lr = u_plan[:, 0]
                af = self.nmpc.compute_feedforward_acceleration(lr)
            else:
                # No previous solve — only possible on the very first
                # NMPC call. Use the reference-level CoM as a last resort
                # (pre-planner if active, else the TorsoPlanner cref).
                rp = np.asarray(cref_r, dtype=float).copy()
                vp = np.asarray(cref_v, dtype=float).copy()
                lr = np.zeros(12)
                af = np.zeros(3)

        # ── M5 Fix 1b: interpolate across QP sub-steps ─────────────────
        # Cache the full trajectory's first two knots for linear
        # interpolation inside the inner QP loop. The control u_0 is
        # piecewise constant over [t, t+dt_nmpc] and is NOT interpolated.
        if x_plan is not None:
            rp_k0 = x_plan[0:3, 0].copy()
            rp_k1 = x_plan[0:3, 1].copy()
            vp_k0 = x_plan[3:6, 0].copy()
            vp_k1 = x_plan[3:6, 1].copy()
        else:
            rp_k0 = rp.copy()
            rp_k1 = rp.copy()
            vp_k0 = vp.copy()
            vp_k1 = vp.copy()

        # M7: single QP variant throughout DS and SS. Synchronized
        # trajectories eliminate the need for gain scheduling.
        qp = self.qp_ss
        tau_last = np.zeros(self.robot.n_joints)
        tau_w_last = np.zeros(3)
        transport_mag_last = 0.0
        _omega_s_last = np.zeros(3)
        qp_ok = True
        t_qp_start = time.perf_counter()

        if ss_end is None:
            ss_end = t + cfg.dt_nmpc  # fallback

        _L_com_qp_prev = rs.L_com.copy()
        # M4: track v_com across QP sub-steps to estimate dv_com for the
        # orbital feedforward term r_com × m·dv_com_est.
        _v_com_qp_prev = rs.v_com.copy()

        for qs in range(self.n_qp_per_nmpc):
            tq = t + qs * cfg.dt_qp
            omega_s = self.mj_data.qvel[3:6].copy()
            pq, pv = mujoco_to_pinocchio(self.mj_data.qpos, self.mj_data.qvel)
            rs = self.robot.update(pq, pv, omega_struct=omega_s)
            Jc, Jdc = self.robot.get_contact_jacobians(
                cc_ss.active_contacts[0], cc_ss.active_contacts[1])

            # M5 Fix 1b: linear interpolation along the NMPC trajectory
            # knot 0 -> knot 1 across the 10 QP sub-steps. alpha goes
            # 0, 0.1, 0.2, ..., 0.9 — at qs=0 we target the current
            # state (knot 0, matches rs), at qs=9 we target 90 % of the
            # way to knot 1. This matches the time parameterisation
            # tq = t + qs*dt_qp for the NMPC reference at that time,
            # avoiding the staircase reference that otherwise creates
            # impulsive torques through the mapping.
            alpha_interp = qs / self.n_qp_per_nmpc
            rp_interp = (1.0 - alpha_interp) * rp_k0 + alpha_interp * rp_k1
            vp_interp = (1.0 - alpha_interp) * vp_k0 + alpha_interp * vp_k1

            # Torso reference (structure frame — no struct pose needed at QP rate).
            #
            # M7 v19: mapping-based torso reference in BOTH SS and DS,
            # but SS uses q_planned (quintic interp between step start
            # and end arm configurations) while DS uses q_current.
            # Rationale:
            #   - v12 (mapping + δ̇, q_current everywhere) had SS torso
            #     peak 22 mm — the mapping's δ(q) term implicitly
            #     anticipates arm-induced base drift.
            #   - v16..v18 (TorsoPlanner direct in SS, no mapping) lost
            #     that compensation and peaked at 120+ mm.
            #   - v19 restores the mapping in SS but feeds it q_planned
            #     so it becomes a feedforward term on the nominal
            #     trajectory, not a feedback term on arm tracking error.
            # Cap the query time to ss_end - ε so reference_at() returns
            # the quintic's terminal pose (p_t1, v=0, a=0) during the
            # post-T_step margin/hold window instead of falling through
            # to _hold_reference() which holds p_t0 (initial) — see
            # TorsoPlanner.reference_at().
            if phase == 'SS' and ss_end is not None:
                tq_planner = min(tq, ss_end - 1e-3)
            else:
                tq_planner = tq
            tr = self.torso_planner.reference_at(tq_planner)
            if (phase == 'SS' and cfg.mapping_bypass_in_ss
                    and self._ss_entry_p_torso is not None):
                # Diagnostic bypass: freeze the linear torso reference at
                # its SS-entry value; angular reference still from
                # TorsoPlanner. Mapping is not called this tick.
                p_torso_ref_used = self._ss_entry_p_torso.copy()
                v_torso_ref_used = np.concatenate([np.zeros(3), tr.v[3:6]])
                a_torso_ff_used = np.concatenate([np.zeros(3), tr.a[3:6]])
            elif (phase in ('SS', 'DS') and self.mapping is not None
                    and cfg.use_m2_stack
                    and not (cfg.ss_two_task_mode and phase == 'SS')):
                # Phase-2.1 two-task: in SS the torso-pose task is fed the RAW
                # TorsoPlanner quintic+SLERP (the `else` branch below, tr.p/v/a)
                # — NO CoMToTorsoMapping δ. (DS still uses the mapping.)
                af_for_mapping = np.zeros(3) if self._diag_pure_pd else af
                # Planned-vs-current diag (commit 64479ab) confirmed
                # q_current is the right q-source for the δ term. But
                # q_current at the WBC rate (100 Hz) closes a mapping
                # feedback loop that oscillates r_b_ref by up to
                # 237 mm/tick on large swings (commit 1b5b841).
                # F-RATE: recompute δ(q) and δ̇(q,q̇) ONCE per NMPC tick
                # (10 Hz). The (m_total/m_b)·rp_interp term still varies
                # smoothly at WBC rate via the existing interpolation.
                ratio = self.mapping.ratio
                m_b = self.mapping.m_b
                if cfg.use_local_delta_mapping:
                    # Loop-free: r_b_ref = r_com_ref - D_local(q)/m_total.
                    # D_local is base-position invariant -> no mapping->q
                    # feedback even with live q (and no q_planned mismatch).
                    # Computed every WBC tick (NOT F-RATE cached): the
                    # 10Hz caching existed only to break the base-position
                    # feedback loop, which D_local does not have. Caching
                    # would instead STEP D_local at NMPC boundaries and
                    # re-introduce reference jitter.
                    m_total = self.mapping.m_total
                    _delta_q = np.asarray(
                        self.mapping.compute_delta_local(rs.q), dtype=float)
                    _delta_dot = np.asarray(
                        self.mapping.compute_delta_local_dot(rs.q, rs.v),
                        dtype=float)
                    # Keep the telemetry cache vars populated (the block
                    # below copies self._mapping_cache_delta).
                    self._mapping_cache_delta = _delta_q.copy()
                    self._mapping_cache_delta_dot = _delta_dot.copy()
                    r_b_ref_m = rp_interp - _delta_q / m_total
                    v_b_ref_m = vp_interp - _delta_dot / m_total
                    a_b_ff_m = af_for_mapping
                else:
                    if qs == 0 or self._mapping_cache_delta is None:
                        self._mapping_cache_delta = np.asarray(
                            self.mapping.compute_delta(rs.q),
                            dtype=float).copy()
                        self._mapping_cache_delta_dot = np.asarray(
                            self.mapping.compute_delta_dot(rs.q, rs.v),
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
                        self.mapping.compute_delta(rs.q),
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

            if self._diag_freeze_ref:
                # Freeze r_b_ref / R_b_ref to the first sample taken.
                # Used to probe PD stability at a STATIC target.
                if self._diag_frozen_r_b_ref is None:
                    self._diag_frozen_r_b_ref = p_torso_ref_used.copy()
                    self._diag_frozen_R_b_ref = tr.R.copy()
                p_torso_ref_used = self._diag_frozen_r_b_ref.copy()
                R_b_ref_frozen = self._diag_frozen_R_b_ref.copy()
                v_torso_ref_used = np.zeros(6)
                a_torso_ff_used = np.zeros(6)

            if self._diag_pure_pd:
                # Strip all feedforward terms entering the QP.
                a_torso_ff_used = np.zeros(6)
                lr = np.zeros(12)
                af = np.zeros(3)
            R_torso_ref_used = (self._diag_frozen_R_b_ref
                                if self._diag_freeze_ref else tr.R)
            tkw = dict(
                J_torso=rs.J_torso, Jdot_dq_torso=rs.Jdot_dq_torso,
                p_torso=rs.oMf_torso.translation,
                R_torso=rs.oMf_torso.rotation,
                p_torso_ref=p_torso_ref_used, R_torso_ref=R_torso_ref_used,
                v_torso_ref=v_torso_ref_used, a_torso_ff=a_torso_ff_used)

            ek = {}
            if phase == 'SS':
                # M7: SwingPlanner plans over [0, T_step] (same horizon
                # as the torso planner). Its quintic timing `tau =
                # clip((t - t_start)/T, 0, 1)` naturally clamps past
                # T_step to the terminal reference (p=p_dock, v=0, a=0),
                # which is exactly what the margin/hold windows need —
                # no separate EXT approach-velocity reference is
                # required.
                sr = self.swing_planner.reference_at(
                    self._swing_query_time(tq, phase, ss_end))
                if sr.is_swinging and sr.swing_arm == swing_arm:
                    J_ee, Jdq_ee, oMf_ee = self._get_ee_data(rs, swing_arm)
                    ek = dict(J_ee=J_ee, Jdot_dq_ee=Jdq_ee,
                              p_ee=oMf_ee.translation, R_ee=oMf_ee.rotation,
                              p_ee_ref=sr.p_ee, R_ee_ref=sr.R_ee,
                              v_ee_ref=np.concatenate([sr.v_ee, sr.omega_ee]),
                              a_ee_ff=np.concatenate([sr.a_ee, sr.alpha_ee]))

            # Reaction null-space: coupling block H_base ← swing_arm
            sw_slice = (self.robot.arm_b_v_slice if swing_arm == 'b'
                        else self.robot.arm_a_v_slice)
            H_bs = rs.H[:6, sw_slice]

            # M2: enable passivity inequality during DS (settling) when the
            # reworked task stack is active. settle_mode already bypasses
            # torso/EE tasks; passivity just adds dq^T*tau_q + 2α*T ≤ 0.
            # M7: passivity active during DS (energy-based settle) and
            # during the SS convergence-hold window (passivity_hold=True).
            # Main loop engages hold mode once the trajectory has ended
            # without docking, so the system dissipates residual energy
            # while the EE closes on the target.
            passivity_active = bool(
                cfg.use_m2_stack and (phase == 'DS' or passivity_hold))
            if passivity_override is not None:
                passivity_active = bool(passivity_override)

            # WBC slack/tube diagnosis: tag each QP solve with locomotion
            # step + phase + sub-step. Read by WholeBodyQP.solve when
            # populating hw_slack_log. Does not affect control logic.
            try:
                qp.diag_label = (
                    f"step{int(step_idx):02d}/{phase}/qs={int(qs):02d}")
            except Exception:
                pass
            # Piste A LOT A: envelope-coupled passivity work-budget. Per tick,
            # W_budget = β·α·max(0, τ_w_max − ‖Ḣ_s(lambda_ref)‖∞), where
            # Ḣ_s(lambda_ref) = Σ r_Cj×f_j_ref + τ_j_ref is the NMPC-EXACT
            # origin-referenced planned momentum-rate (the same quantity the
            # NMPC envelope path constraint enforces). A pre-solve scalar that
            # opens the strict passivity RHS only by the planned envelope
            # headroom. β=0 ⇒ None ⇒ strict (byte-identical).
            _pisteA_W = None
            if cfg.ds_passivity_beta > 0.0 and passivity_active:
                _lr = np.asarray(lr, dtype=float)
                _rCa = np.asarray(self.sched.anchors_a[stance_a], dtype=float)
                _rCb = np.asarray(self.sched.anchors_b[stance_b], dtype=float)
                _Hs_plan = (np.cross(_rCa, _lr[0:3]) + _lr[3:6]
                            + np.cross(_rCb, _lr[6:9]) + _lr[9:12])
                _margin = max(0.0, cfg.tau_w_max
                              - float(np.max(np.abs(_Hs_plan))))
                _pisteA_W = (cfg.ds_passivity_beta * cfg.alpha_passivity
                             * _margin)
            try:
                qdd_t_qp, qdd_qp, lambda_qp_sol, tau, _ = qp.solve(
                    q_t=rs.q_torso, dq_t=rs.dq_torso,
                    q=rs.q_joints, dq=rs.dq_joints,
                    r_com_ref=rp_interp, v_com_ref=vp_interp,
                    lambda_ref=lr, a_com_ff=af,
                    H_robot=rs.H, C_robot=rs.C,
                    J_com=rs.J_com, Jdot_dq_com=rs.Jdot_dq_com,
                    contact_config=cc_ss, J_contacts=Jc, Jdot_dq_contacts=Jdc,
                    hw_current=hw,
                    hw_min=-cfg.hw_qp_tight, hw_max=cfg.hw_qp_tight,
                    r_com=rs.r_com, L_com_current=rs.L_com,
                    H_base_swing=H_bs, swing_v_slice=sw_slice,
                    settle_mode=settle_mode,
                    passivity_active=passivity_active,
                    ds_centroidal_active=ds_centroidal_active,
                    passivity_W_budget=_pisteA_W,
                    **tkw, **ek)
            except Exception as _qp_exc:
                # Surface QP failures (silent swallow -> zero torque is
                # dangerous). Fall back to zero torque of the correct
                # actuator dimension.
                print(f"  [QP-FAIL] {phase} t={t:.2f} qs={qs}: "
                      f"{type(_qp_exc).__name__}: {_qp_exc}")
                tau = np.zeros(self.robot.n_joints)
                lambda_qp_sol = np.zeros(12)
                qdd_t_qp = np.zeros(6)
                qdd_qp = np.zeros(self.robot.model.nv)
                qp_ok = False

            # α (J2 #2): CoM-mobile DS conflict trace (gated; DWELL ticks).
            # Disentangles which constraint binds during the moving-CoM DS:
            #   pass_resid = dqⱼᵀτ_q + 2α·T_kin  (≈0 ⇒ passivity binding),
            #   Hdot_inf   = ‖Σ r_Cj×f_j + τ_j‖∞ from the QP wrench
            #               (→ τ_w_max ⇒ envelope binding),
            #   com_err    = ‖r_com − cref_r‖  (tracking),
            #   qp_ok / nmpc_status  (feasibility).
            if (cfg.ds_mobile_com_magnitude > 0.0 and settle_mode
                    and ds_centroidal_active):
                dqj = rs.dq_joints
                T_kin = 0.5 * float(dqj @ rs.H[6:, 6:] @ dqj)
                pass_resid = float(dqj @ tau + 2.0 * cfg.alpha_passivity * T_kin)
                lam = np.asarray(lambda_qp_sol, dtype=float)
                rCa = np.asarray(self.sched.anchors_a[stance_a], dtype=float)
                rCb = np.asarray(self.sched.anchors_b[stance_b], dtype=float)
                Hdot = (np.cross(rCa, lam[0:3]) + lam[3:6]
                        + np.cross(rCb, lam[6:9]) + lam[9:12])
                # Next swing arm's Jacobian manipulability (byproduct: does
                # "rapprocher" improve conditioning?). Arm-block of the tool
                # Jacobian; sqrt(det(J Jᵀ)).
                J_sw = (np.asarray(rs.J_tool_a) if swing_arm == 'a'
                        else np.asarray(rs.J_tool_b))
                sl_sw = (self.robot.arm_a_v_slice if swing_arm == 'a'
                         else self.robot.arm_b_v_slice)
                J_arm = J_sw[:, sl_sw]
                JJt = J_arm @ J_arm.T
                manip = float(np.sqrt(max(np.linalg.det(JJt), 0.0)))
                log.ds_mobile_trace.append({
                    't': round(float(t), 3),
                    'com_err': float(np.linalg.norm(
                        rs.r_com - np.asarray(cref_r, dtype=float))),
                    'pass_resid': pass_resid,
                    'dq_tau': float(dqj @ tau),   # joint mech. power (>0 ⇒ +work)
                    'W_budget': (float(_pisteA_W) if _pisteA_W is not None
                                 else 0.0),       # Piste A per-tick budget
                    'Hdot_inf': float(np.max(np.abs(Hdot))),
                    'swing_manip': manip,
                    'qp_ok': bool(qp_ok),
                    'nmpc_status': int(nmpc_status_code)})

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
                if self._last_mapping_delta is not None:
                    entry['delta_q'] = self._last_mapping_delta.tolist()
                else:
                    entry['delta_q'] = None
                if self._last_mapping_delta_current is not None:
                    entry['delta_q_current'] = (
                        self._last_mapping_delta_current.tolist())
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

            tau = np.clip(tau, -cfg.tau_max, cfg.tau_max)
            tau_last = tau.copy()
            if self._diag_lock_arm_joints:
                # Diagnostic: zero all joint torques so the arms hold
                # their initial config (qvel[arm] is forced to 0 after
                # every mj_step below).
                tau = np.zeros_like(tau)
                tau_last = tau.copy()
            self.mj_data.ctrl[:self.robot.n_joints] = tau

            # AOCS: reaction wheel torque command.
            if self.has_rwa:
                rw_vel = self.mj_data.qvel[6:9]
                hw_phys = cfg.rwa_I_w * rw_vel
                omega_s = self.mj_data.qvel[3:6]

                # Diagnostic: |ω_s × H_{r/O}| — magnitude of the transport
                # term missing from Mode B's feedforward. Computed but
                # NOT applied to tau_w_cmd (see AOCS_CONCERN_MEMO.md §7).
                H_rO_diag = (rs.L_com
                             + np.cross(rs.r_com,
                                        self.robot._total_mass * rs.v_com))
                transport_mag_last = float(
                    np.linalg.norm(np.cross(omega_s, H_rO_diag)))

                # DS-only per-contact wrench feedforward (legacy_pid_* only):
                # τ_w_FF = −Σ_i (r_Ci × f_i + τ_i) from λ_qp. Captures the
                # internal-stress couple at the welded loop that the FD on
                # L_com misses. r_Ci taken in struct body frame; λ_qp in
                # world frame — equivalent at small structure-frame rotation
                # (the regime we operate in; <5° transient).
                tau_struct_ff_aocs = None
                if (cfg.aocs_use_wrench_ff_in_ds
                        and phase == 'DS'
                        and cfg.aocs_mode in ('legacy_pid_numerical',
                                              'legacy_pid_model')):
                    _r_C = (self.sched.anchors_a[stance_a],
                            self.sched.anchors_b[stance_b])
                    _lam = np.asarray(lambda_qp_sol, dtype=float).ravel()
                    _ff = np.zeros(3)
                    for _ci in range(2):
                        if not cc_nmpc.active_contacts[_ci]:
                            continue
                        _f = _lam[6 * _ci: 6 * _ci + 3]
                        _tq = _lam[6 * _ci + 3: 6 * _ci + 6]
                        _ff -= np.cross(_r_C[_ci], _f) + _tq
                    tau_struct_ff_aocs = _ff

                if cfg.aocs_off_in_ds and phase == 'DS':
                    tau_w_cmd = np.zeros(3)
                elif cfg.aocs_mode == 'H_est' or cfg.aocs_use_H_estimator:
                    # H_{r/O} estimator: feedforward on full robot angular
                    # momentum about O (spin + orbital), with attitude
                    # damping and desaturation feedback.
                    H_dot_est = self.H_estimator.update(
                        r_com=rs.r_com, v_com=rs.v_com,
                        L_com=rs.L_com, omega_s=omega_s)
                    tau_w_cmd = compute_aocs_command(
                        H_dot_est=H_dot_est,
                        omega_s=omega_s,
                        hw_current=hw_phys,
                        hw_target=cfg.aocs_hw_target,
                        K_omega=cfg.aocs_K_omega,
                        K_h=cfg.aocs_K_h,
                        tau_w_max=cfg.aocs_tau_w_max)
                elif cfg.aocs_use_legacy_corrected:
                    # M4: legacy formula + missing orbital term r × m·dv_com.
                    from crawlbot.aocs.force_estimator import (
                        compute_aocs_command_legacy_corrected)
                    tau_w_cmd = compute_aocs_command_legacy_corrected(
                        L_com=rs.L_com, L_com_prev=_L_com_qp_prev,
                        r_com=rs.r_com, v_com=rs.v_com,
                        v_com_prev=_v_com_qp_prev,
                        hw_current=hw_phys, dt=cfg.dt_qp,
                        robot_mass=self.robot._total_mass,
                        K_hw=cfg.aocs_K_hw,
                        hw_min=cfg.hw_min, hw_max=cfg.hw_max,
                        tau_w_max=cfg.aocs_tau_w_max)
                elif cfg.aocs_mode == 'legacy_pd_numerical':
                    # legacy_corrected + PD on ω_s (numerical ω̇_s).
                    from crawlbot.aocs.force_estimator import (
                        compute_aocs_command_legacy_pd_numerical)
                    tau_w_cmd = compute_aocs_command_legacy_pd_numerical(
                        L_com=rs.L_com, L_com_prev=_L_com_qp_prev,
                        r_com=rs.r_com, v_com=rs.v_com,
                        v_com_prev=_v_com_qp_prev,
                        omega_s=omega_s, omega_s_prev=_omega_s_last,
                        hw_current=hw_phys, dt=cfg.dt_qp,
                        robot_mass=self.robot._total_mass,
                        K_hw=cfg.aocs_K_hw, K_omega=cfg.aocs_K_omega,
                        K_d=cfg.aocs_K_d,
                        hw_min=cfg.hw_min, hw_max=cfg.hw_max,
                        tau_w_max=cfg.aocs_tau_w_max)
                elif cfg.aocs_mode == 'legacy_pd_model':
                    # legacy_corrected + PD on ω_s (model-based ω̇_s
                    # from previous τ_w_cmd and structure inertia).
                    from crawlbot.aocs.force_estimator import (
                        compute_aocs_command_legacy_pd_model)
                    tau_w_cmd = compute_aocs_command_legacy_pd_model(
                        L_com=rs.L_com, L_com_prev=_L_com_qp_prev,
                        r_com=rs.r_com, v_com=rs.v_com,
                        v_com_prev=_v_com_qp_prev,
                        omega_s=omega_s, tau_w_prev=tau_w_last,
                        I_struct=self._struct_I,
                        hw_current=hw_phys, dt=cfg.dt_qp,
                        robot_mass=self.robot._total_mass,
                        K_hw=cfg.aocs_K_hw, K_omega=cfg.aocs_K_omega,
                        K_d=cfg.aocs_K_d,
                        hw_min=cfg.hw_min, hw_max=cfg.hw_max,
                        tau_w_max=cfg.aocs_tau_w_max)
                elif cfg.aocs_mode in ('legacy_pid_numerical',
                                       'legacy_pid_model'):
                    # legacy_pd_* + attitude-tracking P term (drives
                    # the structure back to its initial orientation).
                    #
                    # Geometric SO(3) attitude error (Lee–McClamroch):
                    #   e_R = ½(R_init^T R_now − R_now^T R_init)^∨
                    # where (·)^∨ extracts the 3-vector from a skew-
                    # symmetric matrix. Properties vs log3:
                    #   (1) frame-consistent with ω_s (both implicitly
                    #       in current body frame at small angles, with
                    #       only third-order discrepancy from R_now T R_init)
                    #   (2) bounded |e_R| ≤ |sin θ| ≤ 1 — graceful
                    #       saturation under any disturbance
                    #   (3) no singularity at θ = π (log3 would have one)
                    # For our sub-3° drift regime, e_R ≈ θ_s to within
                    # 1% — the K_θ=10 result on log3 transfers directly.
                    # Local import: see note at the H_est branch above.
                    import pinocchio as _pin
                    qw, qx, qy, qz = self._struct_quat_init
                    R_init = _pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
                    qw, qx, qy, qz = self.mj_data.qpos[3:7]
                    R_now = _pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
                    R_err = R_init.T @ R_now
                    # vee of (R_err − R_err^T) / 2 — geometric SO(3) error.
                    theta_s = 0.5 * np.array([
                        R_err[2, 1] - R_err[1, 2],
                        R_err[0, 2] - R_err[2, 0],
                        R_err[1, 0] - R_err[0, 1],
                    ])
                    if cfg.aocs_mode == 'legacy_pid_numerical':
                        from crawlbot.aocs.force_estimator import (
                            compute_aocs_command_legacy_pid_numerical)
                        tau_w_cmd = compute_aocs_command_legacy_pid_numerical(
                            L_com=rs.L_com, L_com_prev=_L_com_qp_prev,
                            r_com=rs.r_com, v_com=rs.v_com,
                            v_com_prev=_v_com_qp_prev,
                            omega_s=omega_s, omega_s_prev=_omega_s_last,
                            theta_s=theta_s,
                            hw_current=hw_phys, dt=cfg.dt_qp,
                            robot_mass=self.robot._total_mass,
                            K_hw=cfg.aocs_K_hw, K_omega=cfg.aocs_K_omega,
                            K_d=cfg.aocs_K_d, K_theta=cfg.aocs_K_theta,
                            hw_min=cfg.hw_min, hw_max=cfg.hw_max,
                            tau_w_max=cfg.aocs_tau_w_max,
                            tau_struct_ff=tau_struct_ff_aocs)
                    else:
                        from crawlbot.aocs.force_estimator import (
                            compute_aocs_command_legacy_pid_model)
                        tau_w_cmd = compute_aocs_command_legacy_pid_model(
                            L_com=rs.L_com, L_com_prev=_L_com_qp_prev,
                            r_com=rs.r_com, v_com=rs.v_com,
                            v_com_prev=_v_com_qp_prev,
                            omega_s=omega_s, theta_s=theta_s,
                            tau_w_prev=tau_w_last,
                            I_struct=self._struct_I,
                            hw_current=hw_phys, dt=cfg.dt_qp,
                            robot_mass=self.robot._total_mass,
                            K_hw=cfg.aocs_K_hw, K_omega=cfg.aocs_K_omega,
                            K_d=cfg.aocs_K_d, K_theta=cfg.aocs_K_theta,
                            hw_min=cfg.hw_min, hw_max=cfg.hw_max,
                            tau_w_max=cfg.aocs_tau_w_max,
                            tau_struct_ff=tau_struct_ff_aocs)
                else:
                    # Legacy AOCS: L_dot feedforward only (spin component).
                    # Desaturation sign matches compute_aocs_command_legacy_corrected
                    # (+K_hw·hw_error). See that function's docstring for the
                    # MuJoCo-convention derivation.
                    L_dot_est = (rs.L_com - _L_com_qp_prev) / cfg.dt_qp
                    hw_error = np.clip(hw_phys, cfg.hw_min, cfg.hw_max) - hw_phys
                    tau_w_cmd = -L_dot_est + cfg.aocs_K_hw * hw_error
                    tau_w_cmd = np.clip(tau_w_cmd, -cfg.aocs_tau_w_max, cfg.aocs_tau_w_max)

                if self._diag_disable_aocs:
                    tau_w_cmd = np.zeros(3)
                self.mj_data.ctrl[self.robot.n_joints:self.robot.n_joints + 3] = tau_w_cmd
                tau_w_last = tau_w_cmd.copy()
                _omega_s_last = omega_s.copy()

            _L_com_qp_prev = rs.L_com.copy()
            _v_com_qp_prev = rs.v_com.copy()
            mujoco.mj_step(self.mj_model, self.mj_data)
            if self._diag_lock_arm_joints:
                # Re-freeze arm joints after the physics step. The arm
                # joints live in the tail of qvel immediately after the
                # structure(6) + RWA(3 if present) + torso(6) block.
                off_rw = 3 if self.has_rwa else 0
                arm_v_start = 6 + off_rw + 6
                arm_v_end = arm_v_start + self.robot.n_joints
                self.mj_data.qvel[arm_v_start:arm_v_end] = 0.0

            omega_s_post = self.mj_data.qvel[3:6].copy()
            rs2 = self.robot.update(
                *mujoco_to_pinocchio(self.mj_data.qpos, self.mj_data.qvel),
                omega_struct=omega_s_post)

            # GMO update (100 Hz, after physics step)
            tau_applied = np.zeros(self.robot.model.nv)
            tau_applied[6:6 + self.robot.n_joints] = tau
            self.gmo.update(rs2.H, rs2.v, rs2.C_matrix, tau_applied)

            # Contact state machine (EXT phase only)
            # M7: GMO-based contact detection runs throughout SS (the
            # EXT phase no longer exists; dock detection is continuous).
            if phase == 'SS' and cfg.use_gmo_dock:
                from crawlbot.estimation.contact_estimator import ContactState
                d_gmo = self._gripper_distance(swing_arm, target_anchor)
                sw_slice = (self.robot.arm_b_v_slice if swing_arm == 'b'
                            else self.robot.arm_a_v_slice)
                r_norm = self.gmo.swing_residual_norm(sw_slice)
                cs = self.contact_sm.update(r_norm, d_gmo)
                if cs == ContactState.CONFIRMED and not self._contact_confirmed:
                    self._contact_confirmed = True

            if self.has_rwa:
                hw = cfg.rwa_I_w * self.mj_data.qvel[6:9].copy()
            else:
                hw -= (rs2.L_com - rs.L_com) / cfg.dt_qp * cfg.dt_qp
            # Do NOT clip hw here. The QP's hw safety constraint is
            # soft (slack variables with heavy quadratic penalty), so
            # the QP stays feasible even when physical hw is beyond
            # the box, and actively generates the maximum corrective
            # wrench it can.

            # Phase-2.1: optional 100 Hz (QP-rate) SS logging of reaction-wheel
            # torque + stored momentum. Gated (default OFF) ⇒ no behavioural
            # change; tau_w_last (init 2495, updated this tick at the AOCS block
            # when has_rwa), hw, tq, phase are all in scope.
            if cfg.log_hifreq_ss and phase == 'SS':
                log.t_ss_hifreq.append(float(tq))
                log.tau_w_ss_hifreq.append(
                    np.asarray(tau_w_last, dtype=float).copy())
                log.hw_ss_hifreq.append(np.asarray(hw, dtype=float).copy())

        t_qp_ms = (time.perf_counter() - t_qp_start) * 1000

        # Logging
        mujoco.mj_forward(self.mj_model, self.mj_data)
        rs_f = self.robot.update(
            *mujoco_to_pinocchio(self.mj_data.qpos, self.mj_data.qvel))
        # Recompute torso reference at the actual logged time (after QP steps)
        t_log = t + cfg.dt_nmpc
        tref_log = self.torso_planner.reference_at(t_log)
        # For the torso-position error metric use the reference the QP
        # actually tracked at the LAST sub-step — i.e. the output of
        # the M5 mapping layer (`p_torso_ref_used`), not the geometric
        # TorsoPlanner quintic (`tref_log.p`). These diverge whenever
        # the mapping is wired, and the old metric was measuring the
        # mapping-vs-planner discrepancy, not the controller's tracking
        # quality. Orientation still comes from the TorsoPlanner SLERP
        # since the mapping only maps position.
        try:
            p_torso_ref_log = p_torso_ref_used.copy()
        except NameError:
            # Sub-loop didn't run (shouldn't happen in production).
            p_torso_ref_log = tref_log.p.copy()
        # DS settle ticks with NO planner phase covering t_log (trailing
        # settle past the last quintic, initial DS before the first): the
        # torso-position task is settle-gated off and p_torso_ref_used
        # carries the live CoM-mapping output, which jumps at the SS->DS
        # switch and ramps during settle. Log the clamped planner pose
        # instead (terminal quintic pose p_t1 / initial hold) so the
        # exported reference is continuous across the whole traversal.
        # When a phase DOES cover t_log (run-B DWELL set_from_waypoints
        # moving reference), p_torso_ref_used is kept — the QP genuinely
        # tracks that moving centroidal reference. NB the guard cannot
        # key on ds_centroidal_active: the locked config runs centroidal
        # DS in the trailing settle too (dca sets ds_centroidal_mode).
        if (phase == 'DS' and settle_mode
                and not self.torso_planner.has_phase_at(t_log)):
            p_torso_ref_log = self.torso_planner.reference_at_clamped(
                t_log).p.copy()
        d_swing = self._gripper_distance(swing_arm, target_anchor)
        d_stance = self._gripper_distance(
            stance_arm, stance_a if stance_arm == 'a' else stance_b)
        L_dot_est = (rs_f.L_com - L_com_prev) / cfg.dt_nmpc
        sq = self.mj_data.qpos[3:7].copy()
        euler = quat_wxyz_to_euler_deg(sq[0], sq[1], sq[2], sq[3])

        log.t.append(t)
        log.phase.append(phase)
        log.step_idx.append(step_idx)
        log.p_torso.append(rs_f.oMf_torso.translation.copy())
        log.p_torso_ref.append(p_torso_ref_log.copy())
        log.e_torso_pos.append(float(np.linalg.norm(
            rs_f.oMf_torso.translation - p_torso_ref_log)))
        R_err = tref_log.R.T @ rs_f.oMf_torso.rotation
        angle_err = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        log.e_torso_ori.append(float(np.degrees(angle_err)))
        log.d_grip_swing.append(d_swing)
        log.d_grip_stance.append(d_stance)
        log.swing_arm.append(swing_arm)

        # EE tracking error (vs planned trajectory reference, not just target).
        # Plan-time (offset-corrected) for the swing planner lookup.
        import pinocchio as pin
        sr_log = self.swing_planner.reference_at(
            self._swing_query_time(t, phase, ss_end))
        _, _, oMf_ee_log = self._get_ee_data(rs_f, swing_arm)
        log.e_ee_pos.append(float(np.linalg.norm(oMf_ee_log.translation - sr_log.p_ee)))
        e_ori_ee = pin.log3(oMf_ee_log.rotation.T @ sr_log.R_ee)
        log.e_ee_ori.append(float(np.degrees(np.linalg.norm(e_ori_ee))))

        # GMO diagnostics
        sw_slice = (self.robot.arm_b_v_slice if swing_arm == 'b'
                    else self.robot.arm_a_v_slice)
        log.gmo_residual_norm.append(float(np.linalg.norm(self.gmo.residual)))
        log.gmo_swing_residual.append(self.gmo.swing_residual_norm(sw_slice))
        log.gmo_contact_state.append(self.contact_sm.state.value)
        log.r_com.append(rs_f.r_com.copy())
        # Log the reference actually fed to the NMPC (pre-planner if
        # active, else TorsoPlanner-derived cref) so diagnostics compare
        # against what the controller was actually tracking.
        log.r_com_ref.append(np.asarray(cref_r, dtype=float).copy())
        log.e_com.append(float(np.linalg.norm(rs_f.r_com - np.asarray(cref_r))))
        log.L_com.append(rs_f.L_com.copy())
        log.L_com_norm.append(float(np.linalg.norm(rs_f.L_com)))
        log.L_dot.append(L_dot_est.copy())
        log.L_dot_norm.append(float(np.linalg.norm(L_dot_est)))
        log.hw.append(hw.copy())
        if self.has_rwa:
            rw_vel_f = self.mj_data.qvel[6:9].copy()
            log.hw_physical.append((cfg.rwa_I_w * rw_vel_f).copy())
            log.tau_w.append(tau_w_last.copy())
            log.rw_speed.append(rw_vel_f.copy())
            log.transport_term_mag.append(transport_mag_last)
        else:
            log.hw_physical.append(hw.copy())
            log.tau_w.append(np.zeros(3))
            log.rw_speed.append(np.zeros(3))
            log.transport_term_mag.append(0.0)

        # H_{r/O} estimator diagnostics
        if self.has_rwa and cfg.aocs_use_H_estimator:
            log.H_rO.append(self.H_estimator.H_rO.copy())
            log.H_dot_est.append(self.H_estimator.H_dot.copy())
            log.omega_struct.append(_omega_s_last.copy())
            # MuJoCo ground truth: constraint torque on structure about O
            mujoco.mj_forward(self.mj_model, self.mj_data)
            log.qfrc_constraint_torque.append(
                self.mj_data.qfrc_constraint[3:6].copy())
        else:
            log.H_rO.append(np.zeros(3))
            log.H_dot_est.append(np.zeros(3))
            log.omega_struct.append(np.zeros(3))
            log.qfrc_constraint_torque.append(np.zeros(3))
        log.tau.append(tau_last.copy())
        log.tau_max_joint.append(float(np.max(np.abs(tau_last))))
        log.struct_pos.append(self.mj_data.qpos[0:3].copy())
        log.struct_quat.append(sq)
        log.struct_euler_deg.append(euler)
        log.nmpc_ok.append(nmpc_ok)
        log.qp_ok.append(qp_ok)
        log.lambda_ref_norm.append(float(np.linalg.norm(lr)))
        log.nmpc_time_ms.append(t_nmpc_ms)
        log.qp_time_ms.append(t_qp_ms)

        # ── M0.2 enrichment ────────────────────────────────────────────
        # Torso orientation (actual vs ref) as quaternion wxyz
        R_torso = rs_f.oMf_torso.rotation
        q_t_actual = pin.Quaternion(R_torso)  # xyzw
        log.q_torso.append(np.array([q_t_actual.w, q_t_actual.x,
                                      q_t_actual.y, q_t_actual.z]))
        R_ref = tref_log.R
        q_t_ref = pin.Quaternion(R_ref)
        log.q_torso_ref.append(np.array([q_t_ref.w, q_t_ref.x,
                                          q_t_ref.y, q_t_ref.z]))

        # EE position/orientation (actual and ref)
        _, _, oMf_ee_f = self._get_ee_data(rs_f, swing_arm)
        log.p_ee.append(oMf_ee_f.translation.copy())
        q_ee_actual = pin.Quaternion(oMf_ee_f.rotation)
        log.q_ee.append(np.array([q_ee_actual.w, q_ee_actual.x,
                                   q_ee_actual.y, q_ee_actual.z]))
        sr_f = self.swing_planner.reference_at(
            self._swing_query_time(t_log, phase, ss_end))
        log.p_ee_ref.append(sr_f.p_ee.copy())
        q_ee_r = pin.Quaternion(sr_f.R_ee)
        log.q_ee_ref.append(np.array([q_ee_r.w, q_ee_r.x,
                                       q_ee_r.y, q_ee_r.z]))

        # Joint, EE, and torso velocities (T15-post-2 instrumentation).
        # Joint rates: Pinocchio-ordered v[arm_{a,b}_v_slice]; same convention
        # as q_joints slicing. EE/torso velocities: J @ v in LOCAL_WORLD_ALIGNED
        # (matches the frame of oMf_*.translation = p_ee / p_torso).
        log.qvel_joints_a.append(
            rs_f.v[self.robot.arm_a_v_slice].copy())
        log.qvel_joints_b.append(
            rs_f.v[self.robot.arm_b_v_slice].copy())
        V_tool_a = rs_f.J_tool_a @ rs_f.v
        V_tool_b = rs_f.J_tool_b @ rs_f.v
        V_torso_body = rs_f.J_torso @ rs_f.v
        log.v_ee_a.append(V_tool_a[:3].copy())
        log.omega_ee_a.append(V_tool_a[3:].copy())
        log.v_ee_b.append(V_tool_b[:3].copy())
        log.omega_ee_b.append(V_tool_b[3:].copy())
        log.v_torso.append(V_torso_body[:3].copy())
        log.omega_torso.append(V_torso_body[3:].copy())

        # CoM velocity (actual from Pinocchio, ref from NMPC)
        log.v_com.append(rs_f.v_com.copy())
        log.v_com_ref.append(vp.copy())

        # L_com reference (NMPC-planned)
        log.L_com_ref.append(rs_f.L_com.copy())  # best available: actual (no L plan)

        # Platform angular velocity
        log.omega_s.append(self.mj_data.qvel[3:6].copy())

        # NMPC solver diagnostics
        log.nmpc_status.append(nmpc_status_code)
        log.nmpc_cost.append(nmpc_cost_val)
        log.nmpc_status_str.append(
            info_n.status if info_n is not None else "no_info")
        log.nmpc_iterations.append(
            int(info_n.solver_stats.get('iter_count', -1))
            if info_n is not None and info_n.solver_stats else -1)

        # Contact wrenches
        log.lambda_ref.append(lr.copy())
        _lqp = (lambda_qp_sol.copy() if hasattr(lambda_qp_sol, 'copy')
                else np.zeros(12))
        log.lambda_qp.append(_lqp)
        log.lambda_qp_norm.append(float(np.linalg.norm(_lqp)))

        # Kinetic energy: 0.5 * v^T H v (relative to structure)
        v_rel = rs_f.v[:rs_f.H.shape[0]]
        T_kin = 0.5 * float(v_rel @ rs_f.H @ v_rel)
        log.T_kinetic.append(T_kin)

        return hw, rs_f.L_com.copy()

    def _get_ee_data(self, rs, arm):
        """Return (J_ee, Jdq_ee, oMf_ee) for the given arm."""
        if arm == 'b':
            return rs.J_tool_b, rs.Jdot_dq_tool_b, rs.oMf_tool_b
        else:
            return rs.J_tool_a, rs.Jdot_dq_tool_a, rs.oMf_tool_a

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
