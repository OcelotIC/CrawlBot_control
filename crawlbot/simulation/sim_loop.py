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

Phase machine per step:
    DS (double support) -> SS (single support) -> EXT (extension) -> dock
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
    dock_configuration, solve_ik, solve_ik_waypoints,
    manipulability_config, precompute_torso_map,
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
from crawlbot.solvers.contact_phase import ContactConfig
from crawlbot.aocs.force_estimator import (
    MomentumDisturbanceEstimator, EstimatorConfig, compute_aocs_command)

from .config import SimConfig
from .logging import SimLog
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
        self.nmpc = None
        self.qp_ss = None
        self.qp_ext = None
        self._weld_map = {}
        self._site_ids = {}
        self.plan = None
        self.has_rwa = False  # Set True if model has reaction wheels
        # M6: coarse pre-planner (optional, created in setup if enabled)
        self.preplanner: Optional[CoarsePrePlanner] = None
        # Most recent pre-planner result for the active step; None when
        # the pre-planner is disabled or the step is in DS/EXT.
        self._coarse_plan: Optional[CoarsePlanResult] = None
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

    # ── Setup ────────────────────────────────────────────────────────────

    def setup(self, n_steps: int = 3, start_a: int = 2, start_b: int = 2):
        """Initialize all components."""
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

        # Scheduler (anchors in structure-local frame)
        self.sched = ContactScheduler(
            anchors_a=anchors_a_local, anchors_b=anchors_b_local,
            dt_ds=cfg.t_ds, dt_ss=cfg.t_swing)
        self.plan = self.sched.plan_traversal(
            start_a=start_a, start_b=start_b, n_steps=n_steps)

        # Swing planner (anchors already in structure frame — no transforms needed)
        self.swing_planner = SwingPlanner(self.sched, clearance=cfg.swing_clearance)

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
        self.torso_map = {}
        for (ai, bi) in needed_pairs:
            se3_a = self.sched.anchor_se3('a', ai)
            se3_b = self.sched.anchor_se3('b', bi)
            try:
                q_opt, w = manipulability_config(
                    self.robot.model, se3_a, se3_b)
                self.torso_map[(ai, bi)] = q_opt
            except RuntimeError:
                pass

        # Initial IK
        self.q_dock_init = dock_configuration(
            self.robot.model,
            self.sched.anchor_se3('a', start_a),
            self.sched.anchor_se3('b', start_b))

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
            tau_struct_max=cfg.tau_struct_max,
            p_max=cfg.nmpc_p_max,
            Wv=cfg.nmpc_Wv * np.ones(3),
            enforce_hw_conservation=cfg.enforce_hw_conservation,
            h_max_tight=cfg.h_max_tight,
            w_L=cfg.w_L_nmpc,
            kappa_terminal=cfg.kappa_terminal))
        self.nmpc.build()

        # M6: coarse pre-planner (optional). Built once; solved per step.
        if cfg.use_coarse_preplanner:
            # Match the TorsoPlanner's timeline: plan over SS + EXT
            # minus the torso_delay, anchored at t_torso_start so the
            # first ~torso_delay*t_swing seconds naturally hold (the
            # interpolator clamps to r_com[0]).
            T_plan_default = (cfg.t_swing + cfg.t_ext_max
                              - cfg.torso_delay * cfg.t_swing)
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
                T_step_default=T_plan_default,
                ipopt_max_iter=cfg.preplanner_max_iter,
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

        # QP variants
        self.qp_ss = self._build_qp(
            cfg.ss_alpha_com, cfg.ss_alpha_torso, cfg.ss_alpha_ee,
            cfg.ss_alpha_posture, cfg.ss_alpha_wrench, cfg.ss_alpha_reaction,
            cfg.ss_Kp_com, cfg.ss_Kd_com,
            cfg.ss_Kp_torso, cfg.ss_Kd_torso,
            cfg.ss_Kp_ee, cfg.ss_Kd_ee,
            cfg.ss_Kp_ee_ang, cfg.ss_Kd_ee_ang)
        self.qp_ext = self._build_qp(
            cfg.ext_alpha_com, cfg.ext_alpha_torso, cfg.ext_alpha_ee,
            cfg.ext_alpha_posture, cfg.ext_alpha_wrench, cfg.ext_alpha_reaction,
            cfg.ext_Kp_com, cfg.ext_Kd_com,
            cfg.ext_Kp_torso, cfg.ext_Kd_torso,
            cfg.ext_Kp_ee, cfg.ext_Kd_ee,
            cfg.ext_Kp_ee_ang, cfg.ext_Kd_ee_ang)

        # QP for close approach (d < 20mm): relax CoM/Torso to let EE converge
        self.qp_approach = self._build_qp(
            cfg.ext_alpha_com * 0.1, cfg.ext_alpha_torso * 0.1,
            cfg.ext_alpha_ee * 10, cfg.ext_alpha_posture,
            cfg.ext_alpha_wrench, cfg.ext_alpha_reaction,
            cfg.ext_Kp_com, cfg.ext_Kd_com,
            cfg.ext_Kp_torso, cfg.ext_Kd_torso,
            cfg.ext_Kp_ee, cfg.ext_Kd_ee,
            cfg.ext_Kp_ee_ang, cfg.ext_Kd_ee_ang)

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
        print(f"  Gait:           {n_steps} step(s), T_swing={cfg.t_swing}s")
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

            try:
                _, _, _, tau, _ = qp.solve(
                    q_t=rs.q_torso, dq_t=rs.dq_torso,
                    q=rs.q_joints, dq=rs.dq_joints,
                    r_com_ref=rs.r_com, v_com_ref=np.zeros(3),
                    lambda_ref=np.zeros(12), a_com_ff=np.zeros(3),
                    H_robot=rs.H, C_robot=rs.C,
                    J_com=rs.J_com, Jdot_dq_com=rs.Jdot_dq_com,
                    contact_config=cc_ds,
                    J_contacts=Jc, Jdot_dq_contacts=Jdc,
                    hw_current=hw_current,
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
                    passivity_active=True)
                tau = np.clip(tau, -cfg.tau_max, cfg.tau_max)
            except Exception:
                tau = -fallback_Kd * rs.dq_joints
                tau = np.clip(tau, -cfg.tau_max, cfg.tau_max)

            self.mj_data.ctrl[:n_j] = tau
            if self.has_rwa:
                self.mj_data.ctrl[n_j:n_j + 3] = 0.0
            mujoco.mj_step(self.mj_model, self.mj_data)

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
            alpha_torque=1e0, alpha_reg=1e-2,
            Kp_com=np.diag([kpc]*3), Kd_com=np.diag([kdc]*3),
            Kp_torso=np.array([kpt]*3 + [kpt*0.6]*3),
            Kd_torso=np.array([kdt]*3 + [kdt*0.6]*3),
            Kp_ee=kpe * np.ones(3), Kd_ee=kde * np.ones(3),
            Kp_ee_ang=kpe_ang * np.ones(3), Kd_ee_ang=kde_ang * np.ones(3),
            Kp_posture=1.0, Kd_posture=1.5,
            L_max=cfg.L_max, tau_w_max=cfg.tau_w_max,
            use_m2_stack=cfg.use_m2_stack,
            ee_null_space=cfg.use_m2_stack,
            alpha_com_soft=cfg.alpha_com_soft,
            alpha_passivity=cfg.alpha_passivity)
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
        for arm in ['a', 'b']:
            for idx in range(5):
                name = f'anchor_{idx+1}{arm}'
                sid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
                if sid >= 0:
                    self._site_ids[name] = sid

    def _gripper_distance(self, arm, anchor_idx):
        grip_sid = self._site_ids.get(f'gripper_{arm}', -1)
        anch_sid = self._site_ids.get(f'anchor_{anchor_idx+1}{arm}', -1)
        if grip_sid < 0 or anch_sid < 0:
            return np.inf
        return float(np.linalg.norm(
            self.mj_data.site_xpos[grip_sid] - self.mj_data.site_xpos[anch_sid]))

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

    # ── Torso planner setup per step ─────────────────────────────────────

    def _setup_torso_for_step(self, t_ss_start, t_ss_end, swing_arm,
                              stance_a, stance_b, target_arm, target_idx):
        """Plan torso trajectory for a crawling step.

        Uses the precomputed manipulability-optimal configuration as the
        endpoint. The trajectory is a quintic interpolation from the
        current state to the optimal endpoint, extending through SS + EXT.
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

        # End configuration from manipulability map.
        # The target anchor pair after docking:
        if target_arm == 'b':
            end_a, end_b = stance_a, target_idx
        else:
            end_a, end_b = target_idx, stance_b

        q_end = self.torso_map.get((end_a, end_b))
        if q_end is None:
            # Fallback: dock_configuration from neutral
            se3_a = self.sched.anchor_se3('a', end_a)
            se3_b = self.sched.anchor_se3('b', end_b)
            q_end = dock_configuration(model, se3_a, se3_b)

        rs_e = self.robot.update(q_end, np.zeros(self.robot.model.nv))
        p_t1 = rs_e.oMf_torso.translation.copy()
        R_t1 = rs_e.oMf_torso.rotation.copy()
        r_com1 = rs_e.r_com.copy()
        delta1 = R_t1.T @ (r_com1 - p_t1)

        # Trajectory: quintic from current to optimal endpoint.
        # The trajectory covers SS and the first portion of EXT, giving
        # the controller time to converge before docking.
        t_torso_start = t_ss_start + cfg.torso_delay * cfg.t_swing
        t_torso_end = t_ss_end + cfg.t_ext_max

        self.torso_planner.clear_phases()
        self.torso_planner.set_hold(p_t0, R_t0, r_com=r_com0)
        self.torso_planner.add_phase(
            t_torso_start, t_torso_end,
            p_t0, R_t0, p_t1, R_t1,
            delta_com_start=delta0, delta_com_end=delta1)

        # M6: run the coarse pre-planner over the TORSO TRAJECTORY HORIZON.
        # Matching the torso planner's timeline (t_torso_start ..
        # t_torso_end ≈ 14.8 s for default t_swing=6, t_ext_max=10,
        # torso_delay=0.2) ensures the pre-planner's velocity profile
        # is consistent with the rest of the stack. Anchoring at
        # t_torso_start gives the same ~1.2 s initial hold the torso
        # planner uses (the interpolator clamps to r_com[0] before the
        # first knot), then a smooth ramp from r_com0 → r_com1.
        if self.preplanner is not None:
            self._run_preplanner(
                t_plan_start=t_torso_start,
                t_plan_end=t_torso_end,
                stance_arm='a' if target_arm == 'b' else 'b',
                stance_a=stance_a, stance_b=stance_b,
                r_com_0=r_com0, r_com_goal=r_com1,
            )

        return q_end

    def _run_preplanner(
        self,
        t_plan_start: float,
        t_plan_end: float,
        stance_arm: str,
        stance_a: int,
        stance_b: int,
        r_com_0: np.ndarray,
        r_com_goal: np.ndarray,
    ) -> None:
        """Solve the coarse pre-planner for the upcoming step.

        Anchored at `t_plan_start` with T_step = t_plan_end - t_plan_start.
        This matches the TorsoPlanner's timeline so the pre-planner and
        the 6D torso orientation reference share a common velocity
        profile.

        The solve uses the live Pinocchio state for (v_com_0, L_com_0)
        and the current wheel momentum for c_const. Stance contact point
        is the anchor of the stance arm in structure frame (constant
        through SS). On success, caches `_coarse_plan` and
        `_coarse_plan_t0 = t_plan_start` so `_step()` can evaluate the
        reference at current sim time via `r_com_at(t - t0)`.
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
        # During SS the swing arm lifts → stance arm is the other one.
        if stance_arm == 'a':
            r_C = self.sched.anchors_a[stance_a].copy()
        else:
            r_C = self.sched.anchors_b[stance_b].copy()

        T_step = max(0.1, t_plan_end - t_plan_start)
        result = self.preplanner.solve(
            r_com_0=r_com_0,
            v_com_0=v_com_0,
            L_com_0=L_com_0,
            r_com_goal=r_com_goal,
            r_C_stance=r_C,
            c_const=c_const,
            T_step=T_step,
            h_max=np.asarray(cfg.h_max_tight, dtype=float).reshape(3),
        )
        self._preplanner_stats.append({
            'success': result.success,
            'solve_ms': result.solve_time_ms,
            'iter_count': result.iter_count,
            'cost': result.cost,
            'status': result.status,
            't_plan_start': t_plan_start,
        })
        if result.success:
            self._coarse_plan = result
            self._coarse_plan_t0 = t_plan_start
            peak_v = float(max(np.linalg.norm(v) for v in result.v_com))
            peak_L = float(max(np.linalg.norm(L) for L in result.L_com))
            print(f"[CoarsePrePlanner] success in "
                  f"{result.solve_time_ms:.1f} ms "
                  f"({result.iter_count} iters, cost={result.cost:.3f}, "
                  f"T_step={T_step:.2f}s, peak |v|={peak_v:.3f} m/s, "
                  f"peak |L|={peak_L:.3f} Nms)")
        else:
            # On failure, clear the plan so _step() falls back to the
            # TorsoPlanner-derived reference.
            self._coarse_plan = None
            print(f"[CoarsePrePlanner] FAILED: {result.status}")

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

        # Parse phases: DS-SS pairs
        phases = plan.phases
        step_idx = 0
        i = 0
        t_offset = 0.0   # Cumulative time offset from inter-step settling
        self._t_plan_offset = 0.0  # mirror for _step's swing-planner query
        while i < len(phases):
            gp = phases[i]
            if gp.phase.value == 'double':
                # DS phase (offset plan timing for settle delays)
                t_ds_start = plan.t_start[i] + t_offset
                t_ds_end = plan.t_end[i] + t_offset

                # Look ahead for SS phase
                if i + 1 < len(phases) and phases[i+1].phase.value != 'double':
                    ss_gp = phases[i+1]
                    t_ss_start = plan.t_start[i+1] + t_offset
                    t_ss_end = plan.t_end[i+1] + t_offset

                    swing_arm = ss_gp.swing_arm
                    stance_arm = 'a' if swing_arm == 'b' else 'b'
                    stance_a = ss_gp.anchor_a_idx
                    stance_b = ss_gp.anchor_b_idx
                    target_idx = ss_gp.swing_to_idx

                    if verbose:
                        print(f"\n[Step {step_idx}] swing={swing_arm}, "
                              f"stance=({stance_a}a,{stance_b}b), "
                              f"target={target_idx}{swing_arm}")

                    # Torso planner
                    q_dock = self._setup_torso_for_step(
                        t_ss_start, t_ss_end, swing_arm,
                        stance_a, stance_b, swing_arm, target_idx)
                    self.qp_ss.set_nominal_posture(q_dock[self.robot.joints_q_slice])
                    self.qp_ext.set_nominal_posture(q_dock[self.robot.joints_q_slice])
                    self.qp_approach.set_nominal_posture(q_dock[self.robot.joints_q_slice])
                    cc_ss = self.sched.contact_config_at(plan.t_start[i+1] + 0.1)

                    # DS (use original plan time for contact config lookup)
                    cc_ds = self.sched.contact_config_at(plan.t_start[i] + 0.1)
                    if step_idx == 0 and len(log.snapshots) == 0:
                        self._capture_snapshot(log, t, 'initial')
                    if verbose:
                        print(f"  DS: [{t_ds_start:.2f}, {t_ds_end:.2f}]")
                    while t < t_ds_end:
                        hw, L_com_prev = self._step(
                            t, 'DS', step_idx, swing_arm, stance_arm,
                            cc_ds, target_idx, stance_a, stance_b,
                            hw, L_com_prev, log, ss_end=t_ss_end)
                        t += cfg.dt_nmpc

                    # SS: release swing arm
                    self._capture_snapshot(log, t,
                                           f'release_step{step_idx}')
                    old_anchor = ss_gp.swing_from_idx
                    self._deactivate_weld(swing_arm, old_anchor)
                    # M3: reset NMPC warm start at the DS→SS transition —
                    # contact configuration has changed, so the previous
                    # solution is no longer feasible.
                    self.nmpc.reset_warm_start()
                    # Reset GMO on phase transition (contact force discontinuity)
                    pq_r, pv_r = mujoco_to_pinocchio(
                        self.mj_data.qpos, self.mj_data.qvel)
                    rs_r = self.robot.update(pq_r, pv_r)
                    self.gmo.reset(rs_r.H, rs_r.v)
                    self.contact_sm.reset()
                    self._contact_confirmed = False
                    # Set EE orientation at swing release for SLERP trajectory
                    _, _, oMf_release = self._get_ee_data(rs_r, swing_arm)
                    self.swing_planner.set_swing_orientation(oMf_release.rotation)
                    if verbose:
                        print(f"  SS: [{t_ss_start:.2f}, {t_ss_end:.2f}] "
                              f"released {swing_arm}@{old_anchor}")
                    while t < t_ss_end:
                        hw, L_com_prev = self._step(
                            t, 'SS', step_idx, swing_arm, stance_arm,
                            cc_ss, target_idx, stance_a, stance_b,
                            hw, L_com_prev, log, ss_end=t_ss_end)
                        t += cfg.dt_nmpc

                    # EXT: torso trajectory continues toward optimal endpoint.
                    # Freeze torso when EE is close (< 10mm) to stabilize
                    # the final approach and let the EE QP close the gap.
                    if verbose:
                        print(f"  EXT: {t:.2f} → dock or +{cfg.t_ext_max}s")

                    t_ext_start = t
                    docked = False
                    torso_frozen = False
                    close_approach = False
                    while t < t_ext_start + cfg.t_ext_max and not docked:
                        ext_phase = 'EXT_CLOSE' if close_approach else 'EXT'
                        hw, L_com_prev = self._step(
                            t, ext_phase, step_idx, swing_arm, stance_arm,
                            cc_ss, target_idx, stance_a, stance_b,
                            hw, L_com_prev, log, ss_end=t_ss_end)
                        t += cfg.dt_nmpc

                        mujoco.mj_forward(self.mj_model, self.mj_data)
                        d = self._gripper_distance(swing_arm, target_idx)
                        ori_err_deg = self._gripper_ori_err_deg(
                            swing_arm, target_idx)

                        # Latch close-approach mode: once d < 20mm, stay in
                        # EE-dominant QP for the rest of the EXT phase
                        if d < 0.020:
                            close_approach = True

                        # Freeze torso when EE is close to stabilize approach
                        if not torso_frozen and d < 0.010:
                            pq_snap, pv_snap = mujoco_to_pinocchio(
                                self.mj_data.qpos, self.mj_data.qvel)
                            rs_snap = self.robot.update(pq_snap, pv_snap)
                            self.torso_planner.set_hold(
                                rs_snap.oMf_torso.translation.copy(),
                                rs_snap.oMf_torso.rotation.copy(),
                                r_com=rs_snap.r_com.copy())
                            torso_frozen = True

                        # Dock detection: BOTH position AND orientation
                        # must be within their thresholds (spec §2 —
                        # MuJoCo's weld is position-gated only, so this
                        # is the only safeguard against welding at a
                        # large orientation misalignment that corrupts
                        # the next step's initial conditions).
                        pos_ok = d < cfg.weld_radius
                        ori_ok = ori_err_deg < cfg.dock_ori_threshold_deg
                        if cfg.use_gmo_dock:
                            docked = self._contact_confirmed and ori_ok
                        else:
                            docked = pos_ok and ori_ok

                        if docked:
                            log.dock_events.append({
                                't': round(t, 3), 'step': step_idx,
                                'd_mm': round(d*1000, 2),
                                'ori_deg': round(ori_err_deg, 2),
                                'arm': swing_arm, 'anchor': target_idx,
                                'method': 'gmo' if cfg.use_gmo_dock else 'kinematic'})
                            self._capture_snapshot(
                                log, t, f'dock_step{step_idx}')
                            if verbose:
                                print(f"  *** DOCK step {step_idx}: t={t:.2f}s "
                                      f"d={d*1000:.1f}mm "
                                      f"ori={ori_err_deg:.2f}° ***")

                    if not docked and verbose:
                        recent = log.d_grip_swing[-20:] if len(log.d_grip_swing) >= 20 else log.d_grip_swing
                        min_d = min(recent) * 1000
                        # Report the ori at the same time we report d
                        ori_at_timeout = self._gripper_ori_err_deg(
                            swing_arm, target_idx)
                        print(f"  TIMEOUT step {step_idx}: "
                              f"min d={min_d:.1f}mm "
                              f"ori_at_exit={ori_at_timeout:.1f}°")

                    # Post-dock: activate weld + inelastic impact projection
                    if docked:
                        self._activate_weld(swing_arm, target_idx)
                        mujoco.mj_forward(self.mj_model, self.mj_data)
                        # M3: reset NMPC warm start at SS→DS transition too.
                        self.nmpc.reset_warm_start()

                        # Inelastic impact: project velocity onto new constraint
                        # manifold. The weld creates a bilateral constraint that
                        # the pre-dock velocity likely violates.
                        pq_dock, pv_dock = mujoco_to_pinocchio(
                            self.mj_data.qpos, self.mj_data.qvel)
                        rs_dock = self.robot.update(pq_dock, pv_dock)
                        Jc_both, _ = self.robot.get_contact_jacobians(True, True)
                        if Jc_both is not None:
                            # v_constraint = Jc @ v (should be ~0 after projection)
                            v_pre = Jc_both @ pv_dock
                            # v_post = v - M^{-1} Jc^T (Jc M^{-1} Jc^T)^{-1} Jc v
                            MiJcT = np.linalg.solve(rs_dock.H, Jc_both.T)
                            Lambda_inv = Jc_both @ MiJcT
                            impulse = np.linalg.solve(Lambda_inv, v_pre)
                            pv_post = pv_dock - MiJcT @ impulse

                            dv = np.linalg.norm(pv_post - pv_dock)
                            if verbose:
                                print(f"  Impact: ||dv||={dv:.4f}, "
                                      f"||Jc@v_pre||={np.linalg.norm(v_pre):.4f}")

                            # Write corrected velocity back to MuJoCo
                            _, mj_qvel_post = pinocchio_to_mujoco(
                                pq_dock, pv_post,
                                struct_pos=self.mj_data.qpos[0:3],
                                struct_quat=self.mj_data.qpos[3:7],
                                rwa=self.has_rwa)
                            # Only overwrite torso + joint velocities, keep structure
                            off_v = 3 if self.has_rwa else 0
                            self.mj_data.qvel[6+off_v:] = mj_qvel_post[6+off_v:]
                            mujoco.mj_forward(self.mj_model, self.mj_data)

                    # Inter-step DS passivity settle (spec §7.1.1).
                    # Energy-based exit: remain in DS with passivity_active
                    # until T_kin < T_settle_inter = 0.5·epsilon_v²·lambda_min(H).
                    # NMPC is bypassed; the QP in settle_mode dissipates
                    # momentum through joint torques, respecting dynamics.
                    # The warm start was already reset at the SS→DS edge
                    # above (see `self.nmpc.reset_warm_start()`), so the
                    # first NMPC solve of the next step starts clean.
                    if (docked
                            and cfg.use_m2_stack
                            and cfg.use_energy_settle_inter):
                        t_settle_start = t
                        # Post-dock DS config. The swing arm is now welded
                        # to target_idx; the stance arm is still at its
                        # original index (stance_a / stance_b).
                        if swing_arm == 'a':
                            new_anchor_a = target_idx
                            new_anchor_b = stance_b
                        else:
                            new_anchor_a = stance_a
                            new_anchor_b = target_idx
                        from crawlbot.solvers.contact_phase import ContactPhase
                        cc_settle = ContactConfig.from_phase(
                            ContactPhase.DOUBLE,
                            r_contact_A=self.sched.anchors_a[new_anchor_a].copy(),
                            r_contact_B=self.sched.anchors_b[new_anchor_b].copy(),
                        )
                        min_steps = max(
                            0,
                            int(round(cfg.t_settle_inter_min / cfg.dt_qp)))
                        settle_result = self._run_ds_passivity_loop(
                            contact_config=cc_settle,
                            max_steps=cfg.n_settle_inter_max_steps,
                            epsilon_v=cfg.settle_inter_epsilon_v,
                            plateau_window=50,
                            plateau_ratio=cfg.settle_plateau_ratio,
                            min_steps=min_steps,
                            fallback_Kd=cfg.Kd_settle_damping,
                        )
                        dt_elapsed = settle_result['n_steps'] * cfg.dt_qp
                        t += dt_elapsed
                        t_offset += dt_elapsed
                        self._t_plan_offset = t_offset
                        log.inter_step_settles.append({
                            'step_idx': int(step_idx),
                            't_start': float(t_settle_start),
                            't_end': float(t),
                            'n_steps': int(settle_result['n_steps']),
                            'T_start': float(settle_result['T_start']),
                            'T_end': float(settle_result['T_end']),
                            'T_settle': float(settle_result['T_settle']),
                            'lambda_min': float(settle_result['lambda_min']),
                            'exit_reason': settle_result['exit_reason'],
                        })
                        if verbose:
                            print(
                                f"  Inter-step settle: {t_settle_start:.2f}"
                                f" → +{dt_elapsed:.3f}s "
                                f"({settle_result['n_steps']} steps, "
                                f"exit={settle_result['exit_reason']}, "
                                f"T: {settle_result['T_start']:.2e} → "
                                f"{settle_result['T_end']:.2e} J, "
                                f"T_settle={settle_result['T_settle']:.2e} J)")

                    step_idx += 1
                    i += 2  # skip SS phase (already processed)
                else:
                    # Trailing DS (end of gait): run settling phase
                    t_ds_start = plan.t_start[i] + t_offset
                    t_ds_settle = t + cfg.t_settle_final
                    cc_ds = self.sched.contact_config_at(plan.t_start[i] + 0.1)

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
                    try:
                        anchor_a_se3 = self.sched.anchor_se3('a', last_sa)
                        anchor_b_se3 = self.sched.anchor_se3('b', last_sb)
                        q_eq = dock_configuration(
                            self.robot.model, anchor_a_se3, anchor_b_se3,
                            q_init=pq)
                        rs_eq = self.robot.update(q_eq, np.zeros(self.robot.model.nv))
                        self.torso_planner.set_hold(
                            rs_eq.oMf_torso.translation.copy(),
                            rs_eq.oMf_torso.rotation.copy(),
                            r_com=rs_eq.r_com.copy())
                    except RuntimeError:
                        # IK failed — fall back to current state
                        self.torso_planner.set_hold(
                            rs_hold.oMf_torso.translation.copy(),
                            rs_hold.oMf_torso.rotation.copy(),
                            r_com=rs_hold.r_com.copy())
                    while t < t_ds_settle:
                        hw, L_com_prev = self._step(
                            t, 'DS', step_idx - 1, last_swing, last_stance,
                            cc_ds, 0, last_sa, last_sb,
                            hw, L_com_prev, log, ss_end=t,
                            settle_mode=True)
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

    def _step(self, t, phase, step_idx, swing_arm, stance_arm,
              cc_ss, target_anchor, stance_a, stance_b,
              hw, L_com_prev, log, ss_end=None, settle_mode=False):
        """Single NMPC+QP step.  All quantities are in structure frame."""
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
        if (self._coarse_plan is not None) and (not settle_mode):
            tau_rel = t_horizon - self._coarse_plan_t0
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

        # QP inner loop: select QP variant for current phase
        if phase == 'EXT_CLOSE':
            qp = self.qp_approach
            phase = 'EXT'   # downstream logic uses 'EXT'
        elif phase == 'EXT':
            qp = self.qp_ext
        else:
            qp = self.qp_ss
        tau_last = np.zeros(12)
        tau_w_last = np.zeros(3)
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
            # M5.1: position/velocity/accel come from the mapping layer
            # evaluated at the CURRENT configuration (arms at their
            # current positions, not those from the start of the NMPC
            # interval). Orientation comes from the TorsoPlanner SLERP.
            tr = self.torso_planner.reference_at(tq)
            if self.mapping is not None and cfg.use_m2_stack:
                # In pure-PD diagnostic mode, zero the CoM feedforward
                # so the mapping's a_b_ff comes only from -δ̈(q)/m_b;
                # then we also clobber it to zero below. This keeps the
                # position reference (r_b_ref from the mapping) intact.
                af_for_mapping = np.zeros(3) if self._diag_pure_pd else af
                r_b_ref_m, v_b_ref_m, a_b_ff_m, _ = self.mapping.compute(
                    r_com_ref=rp_interp, v_com_ref=vp_interp,
                    a_com_ff=af_for_mapping, q_current=rs.q, dq_current=rs.v)
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
                # SwingPlanner uses the scheduler's plan timeline (which
                # is NOT offset-adjusted), so we query it at plan-time
                # = sim-time − cumulative inter-step settle offset.
                tq_plan = tq - self._t_plan_offset
                ss_end_plan = ss_end - self._t_plan_offset
                sr = self.swing_planner.reference_at(
                    min(tq_plan, ss_end_plan - 0.01))
                if sr.is_swinging and sr.swing_arm == swing_arm:
                    J_ee, Jdq_ee, oMf_ee = self._get_ee_data(rs, swing_arm)
                    ek = dict(J_ee=J_ee, Jdot_dq_ee=Jdq_ee,
                              p_ee=oMf_ee.translation, R_ee=oMf_ee.rotation,
                              p_ee_ref=sr.p_ee, R_ee_ref=sr.R_ee,
                              v_ee_ref=np.concatenate([sr.v_ee, sr.omega_ee]),
                              a_ee_ff=np.concatenate([sr.a_ee, sr.alpha_ee]))
            elif phase == 'EXT':
                # Target anchor in structure frame (constant)
                if swing_arm == 'b':
                    p_tgt = self.sched.anchors_b[target_anchor].copy()
                else:
                    p_tgt = self.sched.anchors_a[target_anchor].copy()
                J_ee, Jdq_ee, oMf_ee = self._get_ee_data(rs, swing_arm)
                p_ee = oMf_ee.translation

                # Approach velocity: proportional with minimum floor.
                d_ee = np.linalg.norm(p_tgt - p_ee)
                if d_ee > 1e-6:
                    direction = (p_tgt - p_ee) / d_ee
                    v_mag = max(0.5 * d_ee, 0.002)
                    v_approach = v_mag * direction
                else:
                    v_approach = np.zeros(3)

                # During close approach, match orientation reference to actual
                # pose. This zeros the orientation error so all DOFs serve
                # position convergence, while keeping the 6D Jacobian for
                # Coriolis compensation and regularization.
                if d_ee < 0.020:
                    R_tgt = oMf_ee.rotation.copy()
                else:
                    R_tgt = np.eye(3)
                ek = dict(J_ee=J_ee, Jdot_dq_ee=Jdq_ee,
                          p_ee=p_ee, R_ee=oMf_ee.rotation,
                          p_ee_ref=p_tgt, R_ee_ref=R_tgt,
                          v_ee_ref=np.concatenate([v_approach, np.zeros(3)]),
                          a_ee_ff=np.zeros(6))

            # Reaction null-space: coupling block H_base ← swing_arm
            sw_slice = (self.robot.arm_b_v_slice if swing_arm == 'b'
                        else self.robot.arm_a_v_slice)
            H_bs = rs.H[:6, sw_slice]

            # M2: enable passivity inequality during DS (settling) when the
            # reworked task stack is active. settle_mode already bypasses
            # torso/EE tasks; passivity just adds dq^T*tau_q + 2α*T ≤ 0.
            passivity_active = bool(cfg.use_m2_stack and phase == 'DS')

            try:
                _, _, lambda_qp_sol, tau, _ = qp.solve(
                    q_t=rs.q_torso, dq_t=rs.dq_torso,
                    q=rs.q_joints, dq=rs.dq_joints,
                    r_com_ref=rp_interp, v_com_ref=vp_interp,
                    lambda_ref=lr, a_com_ff=af,
                    H_robot=rs.H, C_robot=rs.C,
                    J_com=rs.J_com, Jdot_dq_com=rs.Jdot_dq_com,
                    contact_config=cc_ss, J_contacts=Jc, Jdot_dq_contacts=Jdc,
                    hw_current=hw, hw_min=cfg.hw_min, hw_max=cfg.hw_max,
                    r_com=rs.r_com, L_com_current=rs.L_com,
                    H_base_swing=H_bs, swing_v_slice=sw_slice,
                    settle_mode=settle_mode,
                    passivity_active=passivity_active,
                    **tkw, **ek)
            except Exception:
                tau = np.zeros(12)
                lambda_qp_sol = np.zeros(12)
                qp_ok = False

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

                if cfg.aocs_mode == 'H_est' or cfg.aocs_use_H_estimator:
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
                # Re-freeze arm joints after the physics step. The
                # arm joints start at qvel[15:27] in the 27-DOF layout
                # (structure 0..5, wheels 6..8, torso 9..14, arms 15..26).
                self.mj_data.qvel[15:27] = 0.0

            omega_s_post = self.mj_data.qvel[3:6].copy()
            rs2 = self.robot.update(
                *mujoco_to_pinocchio(self.mj_data.qpos, self.mj_data.qvel),
                omega_struct=omega_s_post)

            # GMO update (100 Hz, after physics step)
            tau_applied = np.zeros(self.robot.model.nv)
            tau_applied[6:6 + self.robot.n_joints] = tau
            self.gmo.update(rs2.H, rs2.v, rs2.C_matrix, tau_applied)

            # Contact state machine (EXT phase only)
            if phase == 'EXT' and cfg.use_gmo_dock:
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
        sr_log = self.swing_planner.reference_at(t - self._t_plan_offset)
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
        else:
            log.hw_physical.append(hw.copy())
            log.tau_w.append(np.zeros(3))
            log.rw_speed.append(np.zeros(3))

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
        sr_f = self.swing_planner.reference_at(t_log - self._t_plan_offset)
        log.p_ee_ref.append(sr_f.p_ee.copy())
        q_ee_r = pin.Quaternion(sr_f.R_ee)
        log.q_ee_ref.append(np.array([q_ee_r.w, q_ee_r.x,
                                       q_ee_r.y, q_ee_r.z]))

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

        # Contact wrenches
        log.lambda_ref.append(lr.copy())
        log.lambda_qp.append(lambda_qp_sol.copy() if hasattr(lambda_qp_sol, 'copy')
                              else np.zeros(12))

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
