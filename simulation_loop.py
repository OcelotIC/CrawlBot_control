"""
SimulationLoop — Closed-loop MuJoCo simulation with two-stage controller.

Orchestrates full locomotion pipeline for the VISPA crawling robot:

    TorsoPlanner → CoM ref → CentroidalNMPC → WholeBodyQP → MuJoCo

Architecture per NMPC step (10 Hz):
    1. Read MuJoCo state → Pinocchio
    2. TorsoPlanner.reference_at(t) → 6D torso ref
    3. TorsoPlanner.com_reference_at(t) → CoM ref (derived from torso)
    4. CentroidalNMPC.solve(r_com_ref, ...) → λ_ref, a_ff
    5. Inner loop (100 Hz QP):
        a. WholeBodyQP.solve(torso, EE, CoM, wrenches, momentum) → τ
        b. τ → MuJoCo actuators → mj_step
        c. hw update (simplified AOCS)

Phase machine per step:
    DS (double support, 0.5s) → SS (single support, T_swing) → EXT (extension)
    EXT ends when d_ee < WELD_R (real dock) or timeout → DS of next step

Constraints:
    |L_com| ≤ L_max           (robot angular momentum box)
    |L̇_com| ≤ τ_w_max         (reaction wheel torque box)
    h_min ≤ h_w ≤ h_max       (wheel storage box)
    |τ_joint| ≤ τ_max          (joint actuator limits)

Usage:
    sim = SimulationLoop(
        mjcf_path='VISPA_crawling.xml',
        urdf_path='VISPA_crawling_fixed.urdf',
    )
    sim.setup(n_steps=3, start_a=2, start_b=2)
    log = sim.run()
    sim.plot(log)
"""

import numpy as np
import time
import json
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path

try:
    import mujoco
except ImportError:
    mujoco = None

try:
    import pinocchio as pin
except ImportError:
    pin = None

from robot_interface import RobotInterface
from contact_scheduler import ContactScheduler, read_anchors_from_mujoco
from locomotion_planner import LocomotionPlanner
from swing_planner import SwingPlanner
from torso_planner import TorsoPlanner
from ik import dock_configuration
from solvers.centroidal_nmpc import CentroidalNMPC, CentroidalNMPCConfig
from solvers.wholebody_qp import WholeBodyQP, WholeBodyQPConfig
from solvers.contact_phase import ContactConfig


# ── State conversions ────────────────────────────────────────────────────────

def mujoco_to_pinocchio(mj_qpos, mj_qvel):
    """Convert MuJoCo state to Pinocchio convention **in structure frame**.

    The torso pose and twist are expressed relative to the floating structure
    body.  This makes all Pinocchio outputs (r_com, v_com, L_com, Jacobians,
    etc.) invariant to structure drift in the inertial frame.

    Supports both original (nq=26) and RWA-3 (nq=29) layouts.

    Returns
    -------
    pin_q : ndarray (19,)
        [torso_pos_struct(3), torso_quat_xyzw_struct(4), joints(12)]
    pin_v : ndarray (18,)
        [torso_vel_struct(3), torso_omega_struct(3), joint_vel(12)]
    """
    rwa = len(mj_qpos) >= 29
    off_q = 3 if rwa else 0
    off_v = 3 if rwa else 0

    # Structure pose / twist in world
    p_s = mj_qpos[0:3]
    qw_s, qx_s, qy_s, qz_s = mj_qpos[3:7]
    R_s = pin.Quaternion(qw_s, qx_s, qy_s, qz_s).toRotationMatrix()
    v_s = mj_qvel[0:3]
    omega_s = mj_qvel[3:6]

    # Torso pose / twist in world
    p_t = mj_qpos[7 + off_q : 10 + off_q]
    qw_t, qx_t, qy_t, qz_t = mj_qpos[10 + off_q : 14 + off_q]
    R_t = pin.Quaternion(qw_t, qx_t, qy_t, qz_t).toRotationMatrix()
    v_t = mj_qvel[6 + off_v : 9 + off_v]
    omega_t = mj_qvel[9 + off_v : 12 + off_v]

    # Transform torso to structure frame
    dp = p_t - p_s
    p_local = R_s.T @ dp
    R_local = R_s.T @ R_t
    v_local = R_s.T @ (v_t - v_s - np.cross(omega_s, dp))
    omega_local = R_s.T @ (omega_t - omega_s)

    # Pinocchio quaternion convention: xyzw
    q_local = pin.Quaternion(R_local)
    coeffs = q_local.coeffs()  # [x, y, z, w]

    pin_q = np.zeros(19)
    pin_v = np.zeros(18)
    pin_q[0:3] = p_local
    pin_q[3:7] = coeffs
    pin_q[7:19] = mj_qpos[14 + off_q : 26 + off_q]
    pin_v[0:3] = v_local
    pin_v[3:6] = omega_local
    pin_v[6:18] = mj_qvel[12 + off_v : 24 + off_v]
    return pin_q, pin_v


def pinocchio_to_mujoco(pin_q, pin_v, struct_pos=None, struct_quat=None,
                        rwa=False):
    """Convert Pinocchio state (structure frame) to MuJoCo convention (world frame).

    pin_q/pin_v are in the structure body frame.  The torso is transformed back
    to the MuJoCo world frame using the provided structure pose.

    If rwa=True, produces the RWA-3 layout (nq=29, nv=27) with zero wheel angles/vels.
    """
    off_q = 3 if rwa else 0
    off_v = 3 if rwa else 0
    nq = 29 if rwa else 26
    nv = 27 if rwa else 24

    s_pos  = struct_pos  if struct_pos  is not None else np.zeros(3)
    s_quat = struct_quat if struct_quat is not None else np.array([1, 0, 0, 0])
    qw_s, qx_s, qy_s, qz_s = s_quat
    R_s = pin.Quaternion(qw_s, qx_s, qy_s, qz_s).toRotationMatrix()

    # Torso in structure frame → world frame
    p_local = pin_q[0:3]
    x, y, z, w_ = pin_q[3:7]                       # Pinocchio xyzw
    R_local = pin.Quaternion(w_, x, y, z).toRotationMatrix()  # ctor: (w,x,y,z)

    p_world = s_pos + R_s @ p_local
    R_world = R_s @ R_local
    q_world = pin.Quaternion(R_world)
    cw = q_world.coeffs()                           # [x, y, z, w]

    mj_qpos = np.zeros(nq)
    mj_qvel = np.zeros(nv)
    mj_qpos[0:3] = s_pos
    mj_qpos[3:7] = s_quat
    mj_qpos[7 + off_q : 10 + off_q] = p_world
    mj_qpos[10 + off_q : 14 + off_q] = [cw[3], cw[0], cw[1], cw[2]]  # wxyz
    mj_qpos[14 + off_q : 26 + off_q] = pin_q[7:19]

    # Velocities: struct → world (assumes v_struct ≈ 0 at setup)
    mj_qvel[6 + off_v : 9 + off_v]   = R_s @ pin_v[0:3]
    mj_qvel[9 + off_v : 12 + off_v]  = R_s @ pin_v[3:6]
    mj_qvel[12 + off_v : 24 + off_v] = pin_v[6:18]
    return mj_qpos, mj_qvel


# MuJoCo index slices for RWA-3 layout (nq=29, nv=27)
MJ_IDX_STRUCT_POS  = slice(0, 3)
MJ_IDX_STRUCT_QUAT = slice(3, 7)
MJ_IDX_RW_ANGLES   = slice(7, 10)
MJ_IDX_TORSO_POS   = slice(10, 13)
MJ_IDX_TORSO_QUAT  = slice(13, 17)
MJ_IDX_JOINTS      = slice(17, 29)


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class SimConfig:
    """Full simulation configuration."""

    # Timing
    dt_nmpc: float = 0.1          # NMPC period [s] (10 Hz)
    dt_qp: float = 0.01           # QP/MuJoCo period [s] (100 Hz)
    t_ds: float = 0.5             # Double-support duration [s]
    t_swing: float = 6.0          # Single-support (swing) duration [s]
    t_ext_max: float = 10.0       # Max extension phase before timeout [s]

    # Torso trajectory
    torso_frac: float = 0.70      # Fraction of full IK displacement
    torso_delay: float = 0.20     # Delay before torso starts (fraction of t_swing)

    # Joint limits
    tau_max: float = 20.0         # Joint torque limit [Nm]

    # Docking
    weld_radius: float = 0.005    # Real dock threshold [m]

    # Momentum constraints
    hw_init: np.ndarray = field(default_factory=lambda: np.zeros(3))
    hw_min: np.ndarray = field(default_factory=lambda: np.full(3, -5.0))
    hw_max: np.ndarray = field(default_factory=lambda: np.full(3, 5.0))
    L_max: float = 10.0           # Robot angular momentum limit [Nms]
    tau_w_max: float = 5.0        # Reaction wheel torque limit [Nm]

    # AOCS parameters (for physical RWA model)
    aocs_K_hw: float = 2.0        # Feedback gain [1/s]
    aocs_tau_w_max: float = 5.0   # Max wheel torque [Nm] (matches NMPC tau_w_max)
    rwa_I_w: float = 0.01         # Wheel spin inertia [kg.m2]

    # Passivity penalty on hw (drives wheels toward zero in NMPC)
    nmpc_W_hw: float = 0.0        # Penalty weight on ‖hw‖² in terminal cost (0=disabled)

    # NMPC parameters
    nmpc_N: int = 8
    nmpc_dt: float = 0.1
    nmpc_f_max: float = 25.0
    nmpc_tau_max: float = 8.0
    nmpc_Wv: float = 10.0         # NMPC velocity tracking weight (default)
    t_settle_final: float = 20.0   # Duration of final DS settling phase [s]

    # QP weights — Single-support phase
    ss_alpha_com: float = 2e2
    ss_alpha_torso: float = 5e2
    ss_alpha_ee: float = 3e3
    ss_alpha_posture: float = 2e1
    ss_alpha_wrench: float = 1e2

    # QP weights — Extension phase (freeze torso, max EE)
    ext_alpha_com: float = 1e2
    ext_alpha_torso: float = 5e1
    ext_alpha_ee: float = 1e4
    ext_alpha_posture: float = 5e0
    ext_alpha_wrench: float = 1e2

    # QP gains — Single-support
    ss_Kp_com: float = 3.0
    ss_Kd_com: float = 3.0
    ss_Kp_torso: float = 6.0
    ss_Kd_torso: float = 5.0
    ss_Kp_ee: float = 10.0
    ss_Kd_ee: float = 7.0

    # QP gains — Extension
    ext_Kp_com: float = 2.0
    ext_Kd_com: float = 2.0
    ext_Kp_torso: float = 3.0
    ext_Kd_torso: float = 3.0
    ext_Kp_ee: float = 40.0
    ext_Kd_ee: float = 22.0

    # Swing planner
    swing_clearance: float = 0.03  # [m]

    # MuJoCo settling
    n_settle_steps: int = 500


# ── Simulation log ───────────────────────────────────────────────────────────

@dataclass
class SimLog:
    """Comprehensive logged data from a simulation run."""

    t: list = field(default_factory=list)
    phase: list = field(default_factory=list)
    step_idx: list = field(default_factory=list)

    # Torso
    p_torso: list = field(default_factory=list)
    p_torso_ref: list = field(default_factory=list)
    e_torso_pos: list = field(default_factory=list)
    e_torso_ori: list = field(default_factory=list)  # orientation error [deg]

    # End-effector
    d_grip_swing: list = field(default_factory=list)
    d_grip_stance: list = field(default_factory=list)
    swing_arm: list = field(default_factory=list)

    # CoM
    r_com: list = field(default_factory=list)
    r_com_ref: list = field(default_factory=list)
    e_com: list = field(default_factory=list)

    # Momentum
    L_com: list = field(default_factory=list)
    L_com_norm: list = field(default_factory=list)
    L_dot: list = field(default_factory=list)
    L_dot_norm: list = field(default_factory=list)
    hw: list = field(default_factory=list)

    # RWA physical (from MuJoCo wheel speeds)
    hw_physical: list = field(default_factory=list)
    tau_w: list = field(default_factory=list)
    rw_speed: list = field(default_factory=list)

    # Torques
    tau: list = field(default_factory=list)
    tau_max_joint: list = field(default_factory=list)

    # Structure (free-floating body)
    struct_pos: list = field(default_factory=list)
    struct_quat: list = field(default_factory=list)
    struct_euler_deg: list = field(default_factory=list)

    # Solver diagnostics
    nmpc_ok: list = field(default_factory=list)
    qp_ok: list = field(default_factory=list)
    lambda_ref_norm: list = field(default_factory=list)

    # Solver timing
    nmpc_time_ms: list = field(default_factory=list)
    qp_time_ms: list = field(default_factory=list)

    # Dock events
    dock_events: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            d[k] = [x.tolist() if hasattr(x, 'tolist') else x for x in v]
        return d

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f)

    @staticmethod
    def load(path: str) -> 'SimLog':
        with open(path) as f:
            d = json.load(f)
        log = SimLog()
        for k, v in d.items():
            if hasattr(log, k):
                setattr(log, k, v)
        return log


# ── Helper ───────────────────────────────────────────────────────────────────

def quat_wxyz_to_euler_deg(qw, qx, qy, qz):
    """Quaternion (w,x,y,z) → Euler (roll,pitch,yaw) in degrees."""
    sinr = 2 * (qw * qx + qy * qz)
    cosr = 1 - 2 * (qx**2 + qy**2)
    roll = np.arctan2(sinr, cosr)
    sinp = np.clip(2 * (qw * qy - qz * qx), -1, 1)
    pitch = np.arcsin(sinp)
    siny = 2 * (qw * qz + qx * qy)
    cosy = 1 - 2 * (qy**2 + qz**2)
    yaw = np.arctan2(siny, cosy)
    return np.degrees(np.array([roll, pitch, yaw]))


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

        print(f"[SimulationLoop] Initialized:")
        print(f"  Robot mass:     {rs0.total_mass:.1f} kg")
        print(f"  RWA model:      {'YES (3 wheels)' if self.has_rwa else 'NO'}")
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

                    # Capture torso hold for settling (already in structure frame)
                    pq, pv = mujoco_to_pinocchio(
                        self.mj_data.qpos, self.mj_data.qvel)
                    rs_hold = self.robot.update(pq, pv)
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
            rp, vp, _, lr, info_n = self.nmpc.solve(
                r_com=rs.r_com, v_com=rs.v_com, L_com=rs.L_com,
                hw_current=hw, r_com_ref=cref.r_com, v_com_ref=cref.v_com,
                contact_config=cc_nmpc, warm_start=True)
            af = self.nmpc.compute_feedforward_acceleration(lr)
            nmpc_ok = info_n.success
        except Exception:
            rp, vp, lr, af = cref.r_com, cref.v_com, np.zeros(12), np.zeros(3)
            nmpc_ok = False
        t_nmpc_ms = (time.perf_counter() - t_nmpc_start) * 1000

        # QP inner loop
        qp = self.qp_ext if phase == 'EXT' else self.qp_ss
        tau_last = np.zeros(12)
        tau_w_last = np.zeros(3)
        qp_ok = True
        t_qp_start = time.perf_counter()

        if ss_end is None:
            ss_end = t + cfg.dt_nmpc  # fallback

        _L_com_qp_prev = rs.L_com.copy()  # for AOCS L_dot estimate

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

            # AOCS: compute and apply reaction wheel torques
            # Feedforward: compensate centroidal L_dot (not total — the orbital
            # term m*(r_com × v_com) is too large/noisy for the wheel budget).
            if self.has_rwa:
                rw_vel = self.mj_data.qvel[6:9]
                hw_phys = cfg.rwa_I_w * rw_vel
                L_dot_est_qp = (rs.L_com - _L_com_qp_prev) / cfg.dt_qp
                hw_error = np.clip(hw_phys, cfg.hw_min, cfg.hw_max) - hw_phys
                tau_w_cmd = -L_dot_est_qp - cfg.aocs_K_hw * hw_error
                tau_w_cmd = np.clip(tau_w_cmd, -cfg.aocs_tau_w_max, cfg.aocs_tau_w_max)
                self.mj_data.ctrl[12:15] = tau_w_cmd
                tau_w_last = tau_w_cmd.copy()

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
    def plot(log, save_path=None, cfg=None):
        """Generate 8-panel diagnostic plot."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        t = np.array(log.t)
        pt = np.array(log.p_torso)
        pt_ref = np.array(log.p_torso_ref)
        d = np.array(log.d_grip_swing)
        tau = np.array(log.tau)
        ecom = np.array(log.e_com)
        rcom = np.array(log.r_com)
        rcom_ref = np.array(log.r_com_ref)
        Lcom = np.array(log.L_com)
        Lnorm = np.array(log.L_com_norm)
        sp = np.array(log.struct_pos)
        euler = np.array(log.struct_euler_deg)
        ph = log.phase

        L_max = cfg.L_max if cfg else 5.0
        tw = cfg.tau_w_max if cfg else 2.0
        wr = cfg.weld_radius if cfg else 0.005
        tm = cfg.tau_max if cfg else 10.0

        def shade(ax):
            for i in range(len(t)):
                if ph[i] == 'DS':
                    ax.axvspan(t[i]-.04, t[i]+.04, alpha=.08, color='blue')
                elif ph[i] == 'EXT':
                    ax.axvspan(t[i]-.04, t[i]+.04, alpha=.08, color='red')
            for i in range(1, len(ph)):
                if ph[i] != ph[i-1]:
                    ax.axvline(t[i], color='gray', ls=':', alpha=.5)

        fig, axes = plt.subplots(9, 1, figsize=(14, 36), sharex=True)
        nd = len(log.dock_events)
        fig.suptitle(
            f'VISPA — $L_{{max}}$={L_max}, $\\tau_w$={tw} Nm, '
            f'$\\tau_j$={tm} Nm | {nd} dock(s)',
            fontsize=14, fontweight='bold')

        ax = axes[0]; shade(ax)
        ax.semilogy(t, d*100, 'r-', lw=2.5, label='||grip−anchor||')
        ax.axhline(wr*100, color='g', ls='--', lw=2, label=f'seuil {wr*1000:.0f}mm')
        for ev in log.dock_events:
            ax.axvline(ev['t'], color='green', ls='-', lw=2, alpha=.4)
        ax.set_ylabel('Distance [cm] (log)'); ax.set_title('① Distance EE → ancre')
        ax.legend(fontsize=9); ax.grid(True, alpha=.3, which='both'); ax.set_ylim([0.1, 200])

        ax = axes[1]; shade(ax)
        ax.plot(t, pt[:,0]*100, 'r-', lw=2.5, label='torse x')
        ax.plot(t, pt_ref[:,0]*100, 'r--', lw=1.5, alpha=.5, label='ref')
        ax.set_ylabel('[cm]'); ax.set_title('② Avancement torse')
        ax.legend(fontsize=9); ax.grid(True, alpha=.3)

        ax = axes[2]; shade(ax)
        ax.plot(t, rcom[:,0]*100, 'r-', lw=2, label='CoM x')
        ax.plot(t, rcom_ref[:,0]*100, 'r--', lw=1.5, alpha=.6, label='ref')
        ax.plot(t, ecom*100, 'k-', lw=2, label='||e_com||')
        ax.set_ylabel('[cm]'); ax.set_title('③ Suivi CoM')
        ax.legend(fontsize=9); ax.grid(True, alpha=.3)

        ax = axes[3]; shade(ax)
        ax.plot(t, Lcom[:,0], 'r-', lw=1.5, alpha=.7, label='$L_x$')
        ax.plot(t, Lcom[:,1], 'g-', lw=1.5, alpha=.7, label='$L_y$')
        ax.plot(t, Lcom[:,2], 'b-', lw=1.5, alpha=.7, label='$L_z$')
        ax.plot(t, Lnorm, 'k-', lw=2.5, label='$||L||$')
        ax.axhline(L_max, color='r', ls='--', lw=2); ax.axhline(-L_max, color='r', ls='--', lw=2)
        ax.fill_between(t, -L_max, L_max, alpha=.05, color='green')
        ax.set_ylabel('[Nms]'); ax.set_title('④ Moment cinétique robot')
        ax.legend(fontsize=9, ncol=3); ax.grid(True, alpha=.3)

        ax = axes[4]; shade(ax)
        for j in range(6): ax.plot(t, tau[:,j], '-', color='C0', alpha=.3, lw=1)
        for j in range(6,12): ax.plot(t, tau[:,j], '-', color='C1', alpha=.3, lw=1)
        ax.plot(t, np.max(np.abs(tau),axis=1), 'k-', lw=2, label='max |τ|')
        ax.axhline(tm, color='r', ls='--', lw=1.5); ax.axhline(-tm, color='r', ls='--', lw=1.5)
        ax.set_ylabel('[Nm]'); ax.set_title('⑤ Couples articulaires')
        ax.legend(fontsize=9); ax.grid(True, alpha=.3)

        ax = axes[5]; shade(ax)
        sd = np.linalg.norm(sp - sp[0], axis=1) * 100
        ax.plot(t, sd, 'k-', lw=2)
        ax.set_ylabel('[cm]'); ax.set_title('⑥ Dérive structure (translation)')
        ax.grid(True, alpha=.3)

        ax = axes[6]; shade(ax)
        ax.plot(t, euler[:,0], 'r-', lw=1.5, label='roll')
        ax.plot(t, euler[:,1], 'g-', lw=1.5, label='pitch')
        ax.plot(t, euler[:,2], 'b-', lw=1.5, label='yaw')
        ax.plot(t, np.max(np.abs(euler), axis=1), 'k-', lw=2, label='max |angle|')
        ax.set_ylabel('[deg]')
        ax.set_title('⑦ Orientation structure (Euler)')
        ax.legend(fontsize=9); ax.grid(True, alpha=.3)

        # ⑧ Torso position tracking error (components + norm)
        e_pos_vec = (pt - pt_ref) * 100  # (N,3) in cm
        e_pos_norm = np.array(log.e_torso_pos) * 100  # [cm]
        ax = axes[7]; shade(ax)
        ax.plot(t, e_pos_vec[:,0], 'r-', lw=1.2, alpha=.7, label='$e_x$')
        ax.plot(t, e_pos_vec[:,1], 'g-', lw=1.2, alpha=.7, label='$e_y$')
        ax.plot(t, e_pos_vec[:,2], 'b-', lw=1.2, alpha=.7, label='$e_z$')
        ax.plot(t, e_pos_norm, 'k-', lw=2.5, label='$\\|e_{pos}\\|$')
        ax.set_ylabel('[cm]')
        ax.set_title('⑧ Erreur tracking torso — position')
        ax.legend(fontsize=9, ncol=4); ax.grid(True, alpha=.3)

        # ⑨ Torso orientation tracking error (geodesic angle)
        e_ori = np.array(log.e_torso_ori) if log.e_torso_ori else np.zeros(len(t))
        ax = axes[8]; shade(ax)
        ax.plot(t, e_ori, 'b-', lw=2.5)
        ax.set_ylabel('[deg]')
        ax.set_xlabel('Temps [s]')
        ax.set_title('⑨ Erreur tracking torso — orientation (angle géodésique)')
        ax.grid(True, alpha=.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
