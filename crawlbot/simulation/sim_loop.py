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

from crawlbot.core.robot_interface import RobotInterface
from crawlbot.core.state_conversions import (
    mujoco_to_pinocchio, pinocchio_to_mujoco, quat_wxyz_to_euler_deg)
from crawlbot.core.ik import dock_configuration, solve_ik
from crawlbot.planning.contact_scheduler import ContactScheduler, read_anchors_from_mujoco
from crawlbot.planning.locomotion_planner import LocomotionPlanner
from crawlbot.planning.swing_planner import SwingPlanner
from crawlbot.planning.torso_planner import TorsoPlanner
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

    # ── Setup ────────────────────────────────────────────────────────────

    def setup(self, n_steps: int = 3, start_a: int = 2, start_b: int = 2):
        """Initialize all components."""
        cfg = self.cfg

        # MuJoCo
        self.mj_model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = cfg.dt_qp

        # Detect RWA model (3 reaction wheels → nq=29, nv=27, nu=15)
        rw_jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, 'rw_x')
        self.has_rwa = rw_jid >= 0
        if self.has_rwa:
            assert self.mj_model.nq == 29, f"RWA model expects nq=29, got {self.mj_model.nq}"
            assert self.mj_model.nu == 15, f"RWA model expects nu=15, got {self.mj_model.nu}"

        # Verify torso mass matches expectations
        tid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        assert abs(self.mj_model.body_mass[tid] - 40.0) < 0.1, \
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

        # Initial IK
        self.q_dock_init = dock_configuration(
            self.robot.model,
            self.sched.anchor_se3('a', start_a),
            self.sched.anchor_se3('b', start_b))

        sp = self.mj_data.qpos[0:3].copy()
        sq = self.mj_data.qpos[3:7].copy()
        mj_qpos, _ = pinocchio_to_mujoco(
            self.q_dock_init, np.zeros(18), struct_pos=sp, struct_quat=sq,
            rwa=self.has_rwa)
        self.mj_data.qpos[:] = mj_qpos
        self.mj_data.qvel[:] = 0.0

        # Welds
        self._build_weld_map()
        self._deactivate_all_welds()
        self._activate_weld('a', start_a)
        self._activate_weld('b', start_b)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        for _ in range(cfg.n_settle_steps):
            mujoco.mj_step(self.mj_model, self.mj_data)

        # CoM calibration
        rs0 = self.robot.update(
            *mujoco_to_pinocchio(self.mj_data.qpos, self.mj_data.qvel))
        am = sum(self.robot.model.inertias[i].mass for i in range(2, 8))
        self.loco_planner = LocomotionPlanner(
            self.sched, arm_mass=am, total_mass=rs0.total_mass)
        self.loco_planner.calibrate_from_config(rs0.r_com)

        # Site IDs
        self._cache_site_ids()

        # NMPC
        self.nmpc = CentroidalNMPC(CentroidalNMPCConfig(
            robot_mass=rs0.total_mass,
            N=cfg.nmpc_N, dt=cfg.nmpc_dt,
            f_max=cfg.nmpc_f_max, tau_max=cfg.nmpc_tau_max,
            hw_min=cfg.hw_min, hw_max=cfg.hw_max,
            L_max=cfg.L_max, tau_w_max=cfg.tau_w_max,
            W_hw=cfg.nmpc_W_hw,
            Wv=cfg.nmpc_Wv * np.ones(3)))
        self.nmpc.build()

        # QP variants
        self.qp_ss = self._build_qp(
            cfg.ss_alpha_com, cfg.ss_alpha_torso, cfg.ss_alpha_ee,
            cfg.ss_alpha_posture, cfg.ss_alpha_wrench,
            cfg.ss_Kp_com, cfg.ss_Kd_com,
            cfg.ss_Kp_torso, cfg.ss_Kd_torso,
            cfg.ss_Kp_ee, cfg.ss_Kd_ee)
        self.qp_ext = self._build_qp(
            cfg.ext_alpha_com, cfg.ext_alpha_torso, cfg.ext_alpha_ee,
            cfg.ext_alpha_posture, cfg.ext_alpha_wrench,
            cfg.ext_Kp_com, cfg.ext_Kd_com,
            cfg.ext_Kp_torso, cfg.ext_Kd_torso,
            cfg.ext_Kp_ee, cfg.ext_Kd_ee)

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

        print(f"[SimulationLoop] Initialized:")
        print(f"  Robot mass:     {rs0.total_mass:.1f} kg")
        print(f"  RWA model:      {'YES (3 wheels)' if self.has_rwa else 'NO'}")
        print(f"  AOCS estimator: {'H_{r/O}' if cfg.aocs_use_H_estimator else 'L_dot (legacy)'}")
        print(f"  NMPC:           {1/cfg.dt_nmpc:.0f} Hz, N={cfg.nmpc_N}")
        print(f"  QP:             {1/cfg.dt_qp:.0f} Hz, {self.n_qp_per_nmpc} per NMPC")
        print(f"  Gait:           {n_steps} step(s), T_swing={cfg.t_swing}s")
        print(f"  Constraints:    L_max={cfg.L_max} Nms, tau_w={cfg.tau_w_max} Nm, "
              f"tau_joint={cfg.tau_max} Nm")
        print(f"  hw bounds:      [{cfg.hw_min[0]:.1f}, {cfg.hw_max[0]:.1f}] Nms")
        print(f"  Dock threshold: {cfg.weld_radius*1000:.1f} mm")

    def _build_qp(self, ac, at, ae, ap, aw, kpc, kdc, kpt, kdt, kpe, kde):
        cfg = self.cfg
        c = WholeBodyQPConfig(
            nq=12, nc_max=2, dt_qp=cfg.dt_qp,
            tau_max=cfg.tau_max * np.ones(12),
            alpha_com=ac, alpha_torso=at, alpha_ee=ae,
            alpha_posture=ap, alpha_wrench=aw,
            alpha_torque=1e0, alpha_reg=1e-2,
            Kp_com=np.diag([kpc]*3), Kd_com=np.diag([kdc]*3),
            Kp_torso=np.array([kpt]*3 + [kpt*0.6]*3),
            Kd_torso=np.array([kdt]*3 + [kdt*0.6]*3),
            Kp_ee=kpe * np.ones(3), Kd_ee=kde * np.ones(3),
            Kp_posture=1.0, Kd_posture=1.5,
            L_max=cfg.L_max, tau_w_max=cfg.tau_w_max)
        qp = WholeBodyQP(c)
        qp.set_nominal_posture(self.q_dock_init[7:19])
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

    # ── Torso planner setup per step ─────────────────────────────────────

    def _setup_torso_for_step(self, t_ss_start, t_ss_end, swing_arm,
                              stance_a, stance_b, target_arm, target_idx):
        """Plan torso trajectory for a crawling step.

        All computation is in structure frame (Pinocchio outputs and scheduler
        anchors are both in this frame).  No live-anchor reading or
        structure-pose capture is needed.
        """
        cfg = self.cfg
        model = self.robot.model

        # IK start: use live MuJoCo torso state (structure frame via mujoco_to_pinocchio)
        pq_live, pv_live = mujoco_to_pinocchio(
            self.mj_data.qpos, self.mj_data.qvel)
        rs_s = self.robot.update(pq_live, pv_live)
        p_t0   = rs_s.oMf_torso.translation.copy()     # struct frame
        R_t0   = rs_s.oMf_torso.rotation.copy()         # struct frame
        r_com0 = rs_s.r_com.copy()                       # struct frame
        delta0 = R_t0.T @ (r_com0 - p_t0)
        q_start = pq_live.copy()

        # IK end: use constant structure-frame anchors (no live reading needed)
        se3_a = self.sched.anchor_se3('a', stance_a)
        se3_b = self.sched.anchor_se3('b', stance_b)
        if target_arm == 'b':
            se3_b_end = self.sched.anchor_se3('b', target_idx)
            q_end = dock_configuration(model, se3_a, se3_b_end)
        else:
            se3_a_end = self.sched.anchor_se3('a', target_idx)
            q_end = dock_configuration(model, se3_a_end, se3_b)

        rs_e = self.robot.update(q_end, np.zeros(18))
        p_t1_full = rs_e.oMf_torso.translation.copy()   # struct frame
        R_t1_full = rs_e.oMf_torso.rotation.copy()
        r_com1_full = rs_e.r_com.copy()
        delta1_full = R_t1_full.T @ (r_com1_full - p_t1_full)

        frac = cfg.torso_frac
        dp = p_t1_full - p_t0
        dR = R_t0.T @ R_t1_full
        omega = pin.log3(dR)
        p_t1 = p_t0 + frac * dp
        R_t1 = R_t0 @ pin.exp3(frac * omega)
        delta1 = (1 - frac) * delta0 + frac * delta1_full

        # Trajectory stored directly in structure frame (no Fix 3 conversion)
        t_torso_start = t_ss_start + cfg.torso_delay * cfg.t_swing
        self.torso_planner.clear_phases()
        self.torso_planner.set_hold(p_t0, R_t0, r_com=r_com0)
        self.torso_planner.add_phase(
            t_torso_start, t_ss_end,
            p_t0, R_t0, p_t1, R_t1,
            delta_com_start=delta0, delta_com_end=delta1)

        return q_start

    # ── Run ──────────────────────────────────────────────────────────────

    def run(self, verbose=True):
        """Run full multi-step locomotion simulation."""
        cfg = self.cfg
        log = SimLog()
        plan = self.plan

        hw = cfg.hw_init.copy()
        t = 0.0
        L_com_prev = None

        # Parse phases: DS-SS pairs
        phases = plan.phases
        step_idx = 0
        i = 0
        while i < len(phases):
            gp = phases[i]
            if gp.phase.value == 'double':
                # DS phase
                t_ds_start = plan.t_start[i]
                t_ds_end = plan.t_end[i]

                # Look ahead for SS phase
                if i + 1 < len(phases) and phases[i+1].phase.value != 'double':
                    ss_gp = phases[i+1]
                    t_ss_start = plan.t_start[i+1]
                    t_ss_end = plan.t_end[i+1]

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
                    self.qp_ss.set_nominal_posture(q_dock[7:19])
                    self.qp_ext.set_nominal_posture(q_dock[7:19])
                    cc_ss = self.sched.contact_config_at(t_ss_start + 0.1)

                    # DS
                    cc_ds = self.sched.contact_config_at(t_ds_start + 0.1)
                    if verbose:
                        print(f"  DS: [{t_ds_start:.2f}, {t_ds_end:.2f}]")
                    while t < t_ds_end:
                        hw, L_com_prev = self._step(
                            t, 'DS', step_idx, swing_arm, stance_arm,
                            cc_ds, target_idx, stance_a, stance_b,
                            hw, L_com_prev, log, ss_end=t_ss_end)
                        t += cfg.dt_nmpc

                    # SS: release swing arm
                    old_anchor = ss_gp.swing_from_idx
                    self._deactivate_weld(swing_arm, old_anchor)
                    if verbose:
                        print(f"  SS: [{t_ss_start:.2f}, {t_ss_end:.2f}] "
                              f"released {swing_arm}@{old_anchor}")
                    while t < t_ss_end:
                        hw, L_com_prev = self._step(
                            t, 'SS', step_idx, swing_arm, stance_arm,
                            cc_ss, target_idx, stance_a, stance_b,
                            hw, L_com_prev, log, ss_end=t_ss_end)
                        t += cfg.dt_nmpc

                    # EXT: capture torso hold (already in structure frame)
                    pq, pv = mujoco_to_pinocchio(
                        self.mj_data.qpos, self.mj_data.qvel)
                    rs_snap = self.robot.update(pq, pv)
                    self.torso_planner.set_hold(
                        rs_snap.oMf_torso.translation.copy(),
                        rs_snap.oMf_torso.rotation.copy(),
                        r_com=rs_snap.r_com.copy())

                    if verbose:
                        print(f"  EXT: {t:.2f} → dock or +{cfg.t_ext_max}s")

                    t_ext_start = t
                    docked = False
                    while t < t_ext_start + cfg.t_ext_max and not docked:
                        hw, L_com_prev = self._step(
                            t, 'EXT', step_idx, swing_arm, stance_arm,
                            cc_ss, target_idx, stance_a, stance_b,
                            hw, L_com_prev, log, ss_end=t_ss_end)
                        t += cfg.dt_nmpc

                        mujoco.mj_forward(self.mj_model, self.mj_data)
                        d = self._gripper_distance(swing_arm, target_idx)
                        if d < cfg.weld_radius:
                            docked = True
                            log.dock_events.append({
                                't': round(t, 3), 'step': step_idx,
                                'd_mm': round(d*1000, 2),
                                'arm': swing_arm, 'anchor': target_idx})
                            if verbose:
                                print(f"  *** DOCK step {step_idx}: t={t:.2f}s "
                                      f"d={d*1000:.1f}mm ***")

                    if not docked and verbose:
                        recent = log.d_grip_swing[-20:] if len(log.d_grip_swing) >= 20 else log.d_grip_swing
                        print(f"  TIMEOUT step {step_idx}: "
                              f"min d={min(recent)*1000:.1f}mm")

                    # Post-dock: activate weld
                    if docked:
                        self._activate_weld(swing_arm, target_idx)
                        mujoco.mj_forward(self.mj_model, self.mj_data)

                    step_idx += 1
                    i += 2  # skip SS phase (already processed)
                else:
                    # Trailing DS (end of gait): run settling phase
                    t_ds_start = plan.t_start[i]
                    t_ds_settle = t + cfg.t_settle_final
                    cc_ds = self.sched.contact_config_at(t_ds_start + 0.1)

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
                            torso_pos=rs_hold.oMf_torso.translation.copy())
                        rs_eq = self.robot.update(q_eq, np.zeros(18))
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
                            hw, L_com_prev, log, ss_end=t)
                        t += cfg.dt_nmpc

                    i += 1
            else:
                # Standalone SS phase (shouldn't happen in normal plan)
                i += 1

        if verbose:
            self._print_summary(log)
        return log

    # ── Single NMPC+QP step ──────────────────────────────────────────────

    def _step(self, t, phase, step_idx, swing_arm, stance_arm,
              cc_ss, target_anchor, stance_a, stance_b,
              hw, L_com_prev, log, ss_end=None):
        """Single NMPC+QP step.  All quantities are in structure frame."""
        cfg = self.cfg

        # Torso/CoM references (structure frame — no struct pose needed)
        tref = self.torso_planner.reference_at(t)
        cref = self.torso_planner.com_reference_at(t)

        # Robot state in structure frame
        pq, pv = mujoco_to_pinocchio(self.mj_data.qpos, self.mj_data.qvel)
        rs = self.robot.update(pq, pv)
        if L_com_prev is None:
            L_com_prev = rs.L_com.copy()

        # Contact config from constant structure-frame anchors (no live reading)
        cc_nmpc = ContactConfig.from_phase(
            cc_ss.phase,
            self.sched.anchors_a[stance_a].copy(),
            self.sched.anchors_b[stance_b].copy())

        # NMPC
        nmpc_ok = True
        t_nmpc_start = time.perf_counter()
        try:
            rp, vp, _, lr, hw_dot_plan, info_n = self.nmpc.solve(
                r_com=rs.r_com, v_com=rs.v_com, L_com=rs.L_com,
                hw_current=hw, r_com_ref=cref.r_com, v_com_ref=cref.v_com,
                contact_config=cc_nmpc, warm_start=True)
            af = self.nmpc.compute_feedforward_acceleration(lr)
            nmpc_ok = info_n.success
        except Exception:
            rp, vp, lr, af = cref.r_com, cref.v_com, np.zeros(12), np.zeros(3)
            hw_dot_plan = np.zeros(3)
            nmpc_ok = False
        t_nmpc_ms = (time.perf_counter() - t_nmpc_start) * 1000

        # QP inner loop
        qp = self.qp_ext if phase == 'EXT' else self.qp_ss
        tau_last = np.zeros(12)
        tau_w_last = np.zeros(3)
        _omega_s_last = np.zeros(3)
        qp_ok = True
        t_qp_start = time.perf_counter()

        if ss_end is None:
            ss_end = t + cfg.dt_nmpc  # fallback

        _L_com_qp_prev = rs.L_com.copy()

        for qs in range(self.n_qp_per_nmpc):
            tq = t + qs * cfg.dt_qp
            pq, pv = mujoco_to_pinocchio(self.mj_data.qpos, self.mj_data.qvel)
            rs = self.robot.update(pq, pv)
            Jc, Jdc = self.robot.get_contact_jacobians(
                cc_ss.active_contacts[0], cc_ss.active_contacts[1])

            # Torso reference (structure frame — no struct pose needed at QP rate)
            tr = self.torso_planner.reference_at(tq)
            tkw = dict(
                J_torso=rs.J_torso, Jdot_dq_torso=rs.Jdot_dq_torso,
                p_torso=rs.oMf_torso.translation,
                R_torso=rs.oMf_torso.rotation,
                p_torso_ref=tr.p, R_torso_ref=tr.R,
                v_torso_ref=tr.v, a_torso_ff=tr.a)

            ek = {}
            if phase == 'SS':
                sr = self.swing_planner.reference_at(min(tq, ss_end - 0.01))
                if sr.is_swinging and sr.swing_arm == swing_arm:
                    J_ee, Jdq_ee, p_ee = self._get_ee_data(rs, swing_arm)
                    ek = dict(J_ee=J_ee, Jdot_dq_ee=Jdq_ee,
                              p_ee=p_ee, p_ee_ref=sr.p_ee,
                              v_ee_ref=sr.v_ee, a_ee_ff=sr.a_ee)
            elif phase == 'EXT':
                # Target anchor in structure frame (constant)
                if swing_arm == 'b':
                    p_tgt = self.sched.anchors_b[target_anchor].copy()
                else:
                    p_tgt = self.sched.anchors_a[target_anchor].copy()
                J_ee, Jdq_ee, p_ee = self._get_ee_data(rs, swing_arm)
                ek = dict(J_ee=J_ee, Jdot_dq_ee=Jdq_ee,
                          p_ee=p_ee, p_ee_ref=p_tgt,
                          v_ee_ref=np.zeros(3), a_ee_ff=np.zeros(3))

            try:
                _, _, _, tau, _ = qp.solve(
                    q_t=rs.q_torso, dq_t=rs.dq_torso,
                    q=rs.q_joints, dq=rs.dq_joints,
                    r_com_ref=rp, v_com_ref=vp,
                    lambda_ref=lr, a_com_ff=af,
                    H_robot=rs.H, C_robot=rs.C,
                    J_com=rs.J_com, Jdot_dq_com=rs.Jdot_dq_com,
                    contact_config=cc_ss, J_contacts=Jc, Jdot_dq_contacts=Jdc,
                    hw_current=hw, hw_min=cfg.hw_min, hw_max=cfg.hw_max,
                    r_com=rs.r_com, L_com_current=rs.L_com,
                    **tkw, **ek)
            except Exception:
                tau = np.zeros(12)
                qp_ok = False

            tau = np.clip(tau, -cfg.tau_max, cfg.tau_max)
            tau_last = tau.copy()
            self.mj_data.ctrl[:12] = tau

            # AOCS: reaction wheel torque command.
            if self.has_rwa:
                rw_vel = self.mj_data.qvel[6:9]
                hw_phys = cfg.rwa_I_w * rw_vel
                omega_s = self.mj_data.qvel[3:6]

                if cfg.aocs_mode == 'nmpc_plan':
                    # NMPC-planned feedforward: use hw_dot from the NMPC's
                    # own trajectory (self-consistent, no estimation needed).
                    # hw_dot_plan is computed once per NMPC step and held
                    # constant across the 10 QP substeps.
                    hw_error = np.clip(hw_phys, cfg.hw_min, cfg.hw_max) - hw_phys
                    tau_w_cmd = hw_dot_plan - cfg.aocs_K_hw * hw_error
                    tau_w_cmd = np.clip(tau_w_cmd, -cfg.aocs_tau_w_max, cfg.aocs_tau_w_max)
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
                else:
                    # Legacy AOCS: L_dot feedforward only (spin component).
                    L_dot_est = (rs.L_com - _L_com_qp_prev) / cfg.dt_qp
                    hw_error = np.clip(hw_phys, cfg.hw_min, cfg.hw_max) - hw_phys
                    tau_w_cmd = -L_dot_est - cfg.aocs_K_hw * hw_error
                    tau_w_cmd = np.clip(tau_w_cmd, -cfg.aocs_tau_w_max, cfg.aocs_tau_w_max)

                self.mj_data.ctrl[12:15] = tau_w_cmd
                tau_w_last = tau_w_cmd.copy()
                _omega_s_last = omega_s.copy()

            _L_com_qp_prev = rs.L_com.copy()
            mujoco.mj_step(self.mj_model, self.mj_data)

            rs2 = self.robot.update(
                *mujoco_to_pinocchio(self.mj_data.qpos, self.mj_data.qvel))

            if self.has_rwa:
                hw = cfg.rwa_I_w * self.mj_data.qvel[6:9].copy()
            else:
                hw -= (rs2.L_com - rs.L_com) / cfg.dt_qp * cfg.dt_qp
            hw = np.clip(hw, cfg.hw_min, cfg.hw_max)

        t_qp_ms = (time.perf_counter() - t_qp_start) * 1000

        # Logging
        mujoco.mj_forward(self.mj_model, self.mj_data)
        rs_f = self.robot.update(
            *mujoco_to_pinocchio(self.mj_data.qpos, self.mj_data.qvel))
        # Recompute torso reference at the actual logged time (after QP steps)
        t_log = t + cfg.dt_nmpc
        tref_log = self.torso_planner.reference_at(t_log)
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
        log.p_torso_ref.append(tref_log.p.copy())
        log.e_torso_pos.append(float(np.linalg.norm(
            rs_f.oMf_torso.translation - tref_log.p)))
        R_err = tref_log.R.T @ rs_f.oMf_torso.rotation
        angle_err = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        log.e_torso_ori.append(float(np.degrees(angle_err)))
        log.d_grip_swing.append(d_swing)
        log.d_grip_stance.append(d_stance)
        log.swing_arm.append(swing_arm)
        log.r_com.append(rs_f.r_com.copy())
        log.r_com_ref.append(cref.r_com.copy())
        log.e_com.append(float(np.linalg.norm(rs_f.r_com - cref.r_com)))
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

        return hw, rs_f.L_com.copy()

    def _get_ee_data(self, rs, arm):
        if arm == 'b':
            return rs.J_tool_b, rs.Jdot_dq_tool_b, rs.oMf_tool_b.translation
        else:
            return rs.J_tool_a, rs.Jdot_dq_tool_a, rs.oMf_tool_a.translation

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
