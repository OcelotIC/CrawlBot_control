"""Single-step Lutze baseline simulation in MuJoCo.

Architecture (Lutze et al. 2023, simplified centroidal model):
    CoM trajectory -> feedforward wrenches -> LutzeQP (10Hz) -> J^T -> tau -> MuJoCo

NO WholeBodyQP: the Lutze approach uses the simplified rigid-body model,
optimizes contact wrenches directly, and maps them to joint torques via
Jacobian transpose.  This is the fundamental difference from the MPC approach
which uses CentroidalNMPC -> WholeBodyQP (full multi-body dynamics).

Gait: DS (0.5s) -> SS (6s) -> EXT (up to 5s) -> dock
Structure: 2040 kg free-floating satellite (Lutze's LUVOIR model).

Usage:
    python -m lutze_baseline.sim_lutze \\
        --mjcf models/VISPA_crawling.xml \\
        --urdf models/VISPA_crawling_fixed.urdf
"""

import argparse
import json
import time
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import mujoco
except ImportError:
    mujoco = None

try:
    import pinocchio as pin
except ImportError:
    pin = None

from lutze_baseline.centroidal_model import compute_centroidal_state
from lutze_baseline.contact_adjoint import compute_dual_contact_adjoints
from lutze_baseline.momentum_map import compute_momentum_map
from lutze_baseline.lutze_feedforward import compute_feedforward, LutzeFeedforwardConfig
from lutze_baseline.lutze_qp import LutzeQP, LutzeQPConfig
from lutze_baseline.lutze_joint_torques import (
    compute_joint_torques, get_contact_jacobians,
)
from lutze_baseline.lutze_swing_controller import (
    compute_swing_torques, SwingImpedanceConfig,
)


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class SimLutzeConfig:
    """Configuration for single-step Lutze simulation."""

    # Timing
    dt_ctrl: float = 0.1           # Lutze QP period [s] (10 Hz, same as NMPC)
    dt_sim: float = 0.01           # MuJoCo step [s] (100 Hz)
    t_ds: float = 0.5              # Double-support duration [s]
    t_swing: float = 6.0           # Single-support duration [s]
    t_ext_max: float = 5.0         # Max extension phase [s]

    # Gait (single step)
    start_a: int = 2               # Initial anchor for arm A (0-indexed)
    start_b: int = 2               # Initial anchor for arm B
    swing_arm: str = 'b'           # Which arm swings
    target_idx: int = 3            # Target anchor for swing arm

    # Structure (Lutze's LUVOIR satellite)
    struct_mass: float = 2040.0    # [kg]

    # Joint limits
    tau_max: float = 10.0          # Joint torque limit [Nm]

    # Docking
    weld_radius: float = 0.005     # Dock threshold [m]

    # Momentum constraints (for logging/comparison, not enforced by Lutze)
    L_max: float = 5.0             # Robot angular momentum limit [Nms]
    tau_w_max: float = 2.0         # Reaction wheel torque limit [Nm]
    hw_init: np.ndarray = field(default_factory=lambda: np.array([2., -1., 0.5]))
    hw_min: np.ndarray = field(default_factory=lambda: np.full(3, -50.))
    hw_max: np.ndarray = field(default_factory=lambda: np.full(3, 50.))

    # Lutze QP weights
    Qr_diag: float = 1.0           # Robot tracking weight
    Qb_diag: float = 10.0          # Momentum minimization weight
    Qc: float = 0.01               # Wrench regularization
    F_max: float = 3000.0          # SI force limit [N]
    si_tau_max: float = 300.0      # SI torque limit [Nm]

    # Feedforward gains
    Kr_trans: float = 25.0
    Dr_trans: float = 5.0
    Kb: float = 40.0
    Db: float = 8.0

    # Swing impedance
    swing_Kp: float = 200.0
    swing_Kd: float = 20.0

    # CoM trajectory
    torso_frac: float = 0.70       # Fraction of full IK displacement
    torso_delay: float = 0.20      # Delay fraction before torso moves

    # Clearance for swing trajectory
    swing_clearance: float = 0.03  # [m]

    # MuJoCo settling
    n_settle_steps: int = 200


# ── State Conversion ─────────────────────────────────────────────────────────

def mujoco_to_pinocchio(mj_qpos, mj_qvel):
    """Convert MuJoCo state to Pinocchio convention.

    MuJoCo qpos: [struct_pos(3), struct_quat_wxyz(4),
                   torso_pos(3), torso_quat_wxyz(4), joints(12)]
    Pinocchio q: [torso_pos(3), torso_quat_xyzw(4), joints(12)]
    """
    pin_q = np.zeros(19)
    pin_v = np.zeros(18)
    # Torso position
    pin_q[0:3] = mj_qpos[7:10]
    # Torso quaternion: MuJoCo wxyz -> Pinocchio xyzw
    w, x, y, z = mj_qpos[10:14]
    pin_q[3:7] = [x, y, z, w]
    # Joints
    pin_q[7:19] = mj_qpos[14:26]
    # Velocities (torso)
    pin_v[0:3] = mj_qvel[6:9]    # linear
    pin_v[3:6] = mj_qvel[9:12]   # angular
    # Joint velocities
    pin_v[6:18] = mj_qvel[12:24]
    return pin_q, pin_v


def quat_wxyz_to_euler_deg(qw, qx, qy, qz):
    """Quaternion (w,x,y,z) -> Euler (roll,pitch,yaw) in degrees."""
    sinr = 2 * (qw * qx + qy * qz)
    cosr = 1 - 2 * (qx**2 + qy**2)
    roll = np.arctan2(sinr, cosr)
    sinp = np.clip(2 * (qw * qy - qz * qx), -1, 1)
    pitch = np.arcsin(sinp)
    siny = 2 * (qw * qz + qx * qy)
    cosy = 1 - 2 * (qy**2 + qz**2)
    yaw = np.arctan2(siny, cosy)
    return np.degrees(np.array([roll, pitch, yaw]))


# ── Trajectory Helpers ───────────────────────────────────────────────────────

def quintic_interp(t, t0, t1, x0, x1):
    """5th-order polynomial interpolation with zero vel/accel at boundaries.

    Returns (pos, vel, acc) at time t.
    """
    if t <= t0:
        return x0.copy(), np.zeros_like(x0), np.zeros_like(x0)
    if t >= t1:
        return x1.copy(), np.zeros_like(x0), np.zeros_like(x0)

    T = t1 - t0
    tau = (t - t0) / T
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    sd = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T
    sdd = (60 * tau - 180 * tau**2 + 120 * tau**3) / T**2

    dx = x1 - x0
    pos = x0 + s * dx
    vel = sd * dx
    acc = sdd * dx
    return pos, vel, acc


def swing_trajectory(t, t0, t1, p_start, p_end, clearance=0.03):
    """Swing arm trajectory with vertical clearance bump.

    Returns (pos, vel) at time t.
    """
    p_mid = 0.5 * (p_start + p_end)
    # Determine "up" direction: perpendicular to swing direction in the
    # plane containing both points and the torso z-axis
    d = p_end - p_start
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-6:
        up = np.array([0., 0., 1.])
    else:
        # Use z-axis as up for clearance
        up = np.array([0., 0., 1.])
        # Project out the swing direction component
        up = up - np.dot(up, d / d_norm) * (d / d_norm)
        up_norm = np.linalg.norm(up)
        if up_norm < 1e-6:
            up = np.array([0., 1., 0.])
        else:
            up = up / up_norm

    T = t1 - t0
    if t <= t0:
        return p_start.copy(), np.zeros(3)
    if t >= t1:
        return p_end.copy(), np.zeros(3)

    tau = (t - t0) / T
    # Quintic for along-path
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    sd = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T

    # Sine bump for clearance
    h = clearance * np.sin(np.pi * tau)
    hd = clearance * np.pi / T * np.cos(np.pi * tau)

    pos = p_start + s * (p_end - p_start) + h * up
    vel = sd * (p_end - p_start) + hd * up
    return pos, vel


# ── Inverse Kinematics ───────────────────────────────────────────────────────

def dock_configuration(model, oMf_a, oMf_b, q_init=None, max_iter=2000, eps=1e-4):
    """Compute joint configuration that places tool_a at oMf_a and tool_b at oMf_b.

    Uses Pinocchio's inverse kinematics (iterative Jacobian-based).
    Supports both fixed-base and free-flyer models.

    Parameters
    ----------
    model : pin.Model
    oMf_a : pin.SE3 - Desired placement for tool_a.
    oMf_b : pin.SE3 - Desired placement for tool_b.
    q_init : (nq,) or None - Initial guess.

    Returns
    -------
    q : (nq,) - Joint configuration.
    """
    data = model.createData()
    fid_a = model.getFrameId('tool_a')
    fid_b = model.getFrameId('tool_b')

    q = q_init if q_init is not None else pin.neutral(model)
    q = q.copy()

    # For free-flyer models, set initial base position to anchor midpoint
    has_ff = (model.nq > model.nv)  # free-flyer adds 1 extra qpos (quaternion)
    if has_ff and q_init is None:
        q[0:3] = 0.5 * (oMf_a.translation + oMf_b.translation)

    dt_ik = 0.1
    damp = 1e-6

    for i in range(max_iter):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        # Compute errors
        err_a = pin.log6(data.oMf[fid_a].actInv(oMf_a)).vector
        err_b = pin.log6(data.oMf[fid_b].actInv(oMf_b)).vector

        err = np.concatenate([err_a, err_b])
        if np.linalg.norm(err) < eps:
            break

        # Jacobians
        pin.computeJointJacobians(model, data, q)
        J_a = pin.getFrameJacobian(model, data, fid_a, pin.LOCAL)
        J_b = pin.getFrameJacobian(model, data, fid_b, pin.LOCAL)

        dq = np.zeros(model.nv)

        if has_ff:
            # Partitioned IK: solve arms first, then base (conservative)
            Ja_arm = J_a[:, 6:12]
            dq[6:12] = np.linalg.solve(
                Ja_arm.T @ Ja_arm + damp * np.eye(6), Ja_arm.T @ err_a)
            Jb_arm = J_b[:, 12:18]
            dq[12:18] = np.linalg.solve(
                Jb_arm.T @ Jb_arm + damp * np.eye(6), Jb_arm.T @ err_b)
            # Base contribution (conservative scaling)
            J_base = np.vstack([J_a[:, :6], J_b[:, :6]])
            dq[:6] = np.linalg.solve(
                J_base.T @ J_base + 1e-3 * np.eye(6),
                J_base.T @ err) * 0.3
        else:
            J = np.vstack([J_a, J_b])
            JtJ = J.T @ J + damp * np.eye(model.nv)
            dq = np.linalg.solve(JtJ, J.T @ err)

        q = pin.integrate(model, q, dt_ik * dq)

        # Clamp revolute joints to [-pi, pi] to respect joint limits
        joint_start = 7 if has_ff else 0
        q[joint_start:] = np.remainder(
            q[joint_start:] + np.pi, 2 * np.pi) - np.pi

    return q


# ── Simulation Log ───────────────────────────────────────────────────────────

@dataclass
class SimLog:
    """Logged data from the Lutze simulation."""
    t: list = field(default_factory=list)
    phase: list = field(default_factory=list)

    # Torso
    p_torso: list = field(default_factory=list)
    p_torso_ref: list = field(default_factory=list)
    e_torso_pos: list = field(default_factory=list)

    # End-effector
    d_grip_swing: list = field(default_factory=list)
    d_grip_stance: list = field(default_factory=list)

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

    # Torques
    tau: list = field(default_factory=list)
    tau_max_joint: list = field(default_factory=list)

    # Structure
    struct_pos: list = field(default_factory=list)
    struct_quat: list = field(default_factory=list)
    struct_euler_deg: list = field(default_factory=list)

    # Solver
    qp_status: list = field(default_factory=list)
    lambda_ref_norm: list = field(default_factory=list)

    # Dock events
    dock_events: list = field(default_factory=list)

    def to_dict(self):
        d = {}
        for k, v in self.__dict__.items():
            d[k] = [x.tolist() if hasattr(x, 'tolist') else x for x in v]
        return d

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f)

    @staticmethod
    def load(path):
        with open(path) as f:
            d = json.load(f)
        log = SimLog()
        for k, v in d.items():
            if hasattr(log, k):
                setattr(log, k, v)
        return log


# ── Main Simulation ──────────────────────────────────────────────────────────

class SimulationLutze:
    """Single-step Lutze baseline simulation."""

    def __init__(self, mjcf_path, urdf_path, config=None):
        assert mujoco is not None, "mujoco package required"
        assert pin is not None, "pinocchio package required"
        self.mjcf_path = mjcf_path
        self.urdf_path = urdf_path
        self.cfg = config or SimLutzeConfig()
        self.n_sim_per_ctrl = int(round(self.cfg.dt_ctrl / self.cfg.dt_sim))

    def setup(self):
        """Initialize MuJoCo, Pinocchio, and compute IK for start/end configs."""
        cfg = self.cfg

        # --- MuJoCo ---
        self.mj_model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = cfg.dt_sim

        # Override structure mass to 2040 kg (Lutze's satellite)
        struct_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, 'structure')
        old_mass = self.mj_model.body_mass[struct_id]
        ratio = cfg.struct_mass / old_mass
        self.mj_model.body_mass[struct_id] = cfg.struct_mass
        self.mj_model.body_inertia[struct_id] *= ratio
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # --- Pinocchio (free-flyer for centroidal dynamics) ---
        self.pin_model = pin.buildModelFromUrdf(
            self.urdf_path, pin.JointModelFreeFlyer())
        self.pin_data = self.pin_model.createData()

        # --- Cache site IDs ---
        self._site_ids = {}
        for name in ['gripper_a', 'gripper_b']:
            sid = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
            self._site_ids[name] = sid
        for arm in ['a', 'b']:
            for idx in range(6):
                name = f'anchor_{idx+1}{arm}'
                sid = mujoco.mj_name2id(
                    self.mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
                if sid >= 0:
                    self._site_ids[name] = sid

        # --- Build weld map ---
        self._weld_map = {}
        for i in range(self.mj_model.neq):
            name = mujoco.mj_id2name(
                self.mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, i)
            if name and name.startswith('grip_'):
                parts = name.split('_to_')
                arm = parts[0].split('_')[1]
                anchor_idx = int(parts[1][0]) - 1  # 1-indexed in MJCF
                self._weld_map[(arm, anchor_idx)] = i

        # --- Deactivate all welds, activate start pair ---
        for eq_id in range(self.mj_model.neq):
            self.mj_data.eq_active[eq_id] = 0
        self._activate_weld('a', cfg.start_a)
        self._activate_weld('b', cfg.start_b)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # --- Read anchor positions from MuJoCo ---
        self._anchor_pos = {}
        for arm in ['a', 'b']:
            for idx in range(6):
                sname = f'anchor_{idx+1}{arm}'
                if sname in self._site_ids:
                    sid = self._site_ids[sname]
                    self._anchor_pos[(arm, idx)] = (
                        self.mj_data.site_xpos[sid].copy())

        # --- IK: start configuration ---
        p_a_start = self._anchor_pos[('a', cfg.start_a)]
        p_b_start = self._anchor_pos[('b', cfg.start_b)]
        oMf_a_start = pin.SE3(np.eye(3), p_a_start)
        oMf_b_start = pin.SE3(np.eye(3), p_b_start)
        self.q_start = dock_configuration(
            self.pin_model, oMf_a_start, oMf_b_start)

        # --- Compute CoM at start for trajectory planning ---
        pin.forwardKinematics(self.pin_model, self.pin_data, self.q_start)
        pin.centerOfMass(self.pin_model, self.pin_data, self.q_start)
        self.r_com_start = self.pin_data.com[0].copy()

        # CoM target: shift by fraction of anchor displacement
        swing = cfg.swing_arm
        target = cfg.target_idx
        from_idx = cfg.start_b if swing == 'b' else cfg.start_a
        p_from = self._anchor_pos[(swing, from_idx)]
        p_to = self._anchor_pos[(swing, target)]
        anchor_disp = p_to - p_from
        self.r_com_end = self.r_com_start + cfg.torso_frac * anchor_disp

        # --- Set MuJoCo initial state from IK ---
        struct_pos = self.mj_data.qpos[0:3].copy()
        struct_quat = self.mj_data.qpos[3:7].copy()
        mj_qpos = np.zeros(26)
        mj_qpos[0:3] = struct_pos
        mj_qpos[3:7] = struct_quat
        mj_qpos[7:10] = self.q_start[0:3]       # torso pos
        # Pinocchio quat xyzw -> MuJoCo wxyz
        x, y, z, w_ = self.q_start[3:7]
        mj_qpos[10:14] = [w_, x, y, z]
        mj_qpos[14:26] = self.q_start[7:19]      # joints
        self.mj_data.qpos[:] = mj_qpos
        self.mj_data.qvel[:] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Settle
        for _ in range(cfg.n_settle_steps):
            mujoco.mj_step(self.mj_model, self.mj_data)
        self.mj_data.qvel[:] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # --- Build controllers ---
        self.lutze_qp = LutzeQP(LutzeQPConfig(
            Qr=np.eye(6) * cfg.Qr_diag,
            Qb=np.eye(3) * cfg.Qb_diag,
            Qc=cfg.Qc,
            F_max=cfg.F_max,
            tau_max=cfg.si_tau_max,
            h_dot_max=cfg.tau_w_max,
        ))
        self.ff_config = LutzeFeedforwardConfig(
            Kr_trans=np.diag([cfg.Kr_trans] * 3),
            Dr_trans=np.diag([cfg.Dr_trans] * 3),
            Kb=np.diag([cfg.Kb] * 3),
            Db=np.diag([cfg.Db] * 3),
        )
        self.swing_config = SwingImpedanceConfig(
            Kp=np.diag([cfg.swing_Kp] * 3),
            Kd=np.diag([cfg.swing_Kd] * 3),
        )

        # --- Swing anchor positions (from MuJoCo, after settling) ---
        mujoco.mj_forward(self.mj_model, self.mj_data)
        swing = cfg.swing_arm
        from_idx = cfg.start_b if swing == 'b' else cfg.start_a
        self.p_swing_start = self._anchor_site_pos(swing, from_idx)
        self.p_swing_end = self._anchor_site_pos(swing, cfg.target_idx)

        # --- IK: end configuration for joint-space swing control ---
        p_a_start = self._anchor_pos[('a', cfg.start_a)]
        p_b_start = self._anchor_pos[('b', cfg.start_b)]
        if swing == 'b':
            p_b_end = self._anchor_pos[('b', target)]
            oMf_b_end = pin.SE3(np.eye(3), p_b_end)
            self.q_end = dock_configuration(
                self.pin_model,
                pin.SE3(np.eye(3), p_a_start),
                oMf_b_end, q_init=self.q_start)
        else:
            p_a_end = self._anchor_pos[('a', target)]
            oMf_a_end = pin.SE3(np.eye(3), p_a_end)
            self.q_end = dock_configuration(
                self.pin_model, oMf_a_end,
                pin.SE3(np.eye(3), p_b_start),
                q_init=self.q_start)

        # Extract swing arm target joint angles for joint-space PD
        if swing == 'b':
            self.q_swing_start = self.q_start[13:19].copy()  # arm B joints
            self.q_swing_end = self.q_end[13:19].copy()
        else:
            self.q_swing_start = self.q_start[7:13].copy()   # arm A joints
            self.q_swing_end = self.q_end[7:13].copy()

        # During EXT phase, also use Cartesian impedance for final approach
        # (more effective near the target than joint-space PD)

        total_mass = pin.computeTotalMass(self.pin_model)
        print(f"[SimulationLutze] Initialized:")
        print(f"  Robot mass:     {total_mass:.1f} kg")
        print(f"  Structure:      {cfg.struct_mass:.0f} kg")
        print(f"  Lutze QP:       {1/cfg.dt_ctrl:.0f} Hz")
        print(f"  MuJoCo:         {1/cfg.dt_sim:.0f} Hz "
              f"({self.n_sim_per_ctrl} per ctrl)")
        print(f"  Gait:           1 step, swing={cfg.swing_arm}, "
              f"target={cfg.target_idx}")
        print(f"  SI limits:      F={cfg.F_max}N, tau={cfg.si_tau_max}Nm")
        print(f"  Dock threshold: {cfg.weld_radius*1000:.1f} mm")

    # ── Weld management ──────────────────────────────────────────────────

    def _activate_weld(self, arm, anchor_idx):
        key = (arm, anchor_idx)
        if key in self._weld_map:
            self.mj_data.eq_active[self._weld_map[key]] = 1

    def _deactivate_weld(self, arm, anchor_idx):
        key = (arm, anchor_idx)
        if key in self._weld_map:
            self.mj_data.eq_active[self._weld_map[key]] = 0

    def _gripper_distance(self, arm, anchor_idx):
        grip_sid = self._site_ids.get(f'gripper_{arm}', -1)
        anch_sid = self._site_ids.get(f'anchor_{anchor_idx+1}{arm}', -1)
        if grip_sid < 0 or anch_sid < 0:
            return np.inf
        return float(np.linalg.norm(
            self.mj_data.site_xpos[grip_sid] -
            self.mj_data.site_xpos[anch_sid]))

    def _anchor_site_pos(self, arm, anchor_idx):
        anch_sid = self._site_ids.get(f'anchor_{anchor_idx+1}{arm}', -1)
        if anch_sid < 0:
            return np.zeros(3)
        return self.mj_data.site_xpos[anch_sid].copy()

    # ── Run ──────────────────────────────────────────────────────────────

    def run(self, verbose=True):
        """Run the single-step simulation: DS -> SS -> EXT -> dock."""
        cfg = self.cfg
        log = SimLog()
        hw = cfg.hw_init.copy()
        t = 0.0
        L_com_prev = None

        swing_arm = cfg.swing_arm
        stance_arm = 'a' if swing_arm == 'b' else 'b'
        from_idx = cfg.start_b if swing_arm == 'b' else cfg.start_a

        # Phase timing
        t_ds_end = cfg.t_ds
        t_ss_start = t_ds_end
        t_ss_end = t_ss_start + cfg.t_swing
        t_com_start = t_ss_start + cfg.torso_delay * cfg.t_swing
        t_com_end = t_ss_end

        # ── DS Phase ─────────────────────────────────────────────────────
        if verbose:
            print(f"\n  DS: [0.00, {t_ds_end:.2f}]")

        while t < t_ds_end:
            hw, L_com_prev = self._step(
                t, 'DS', hw, L_com_prev, log,
                active_a=True, active_b=True,
                t_com_start=t_com_start, t_com_end=t_com_end,
                t_swing_start=t_ss_start, t_swing_end=t_ss_end)
            t += cfg.dt_ctrl

        # ── SS Phase: release swing arm ──────────────────────────────────
        self._deactivate_weld(swing_arm, from_idx)
        if verbose:
            print(f"  SS: [{t_ss_start:.2f}, {t_ss_end:.2f}] "
                  f"released {swing_arm}@{from_idx}")

        while t < t_ss_end:
            hw, L_com_prev = self._step(
                t, 'SS', hw, L_com_prev, log,
                active_a=(stance_arm == 'a' or swing_arm != 'a'),
                active_b=(stance_arm == 'b' or swing_arm != 'b'),
                t_com_start=t_com_start, t_com_end=t_com_end,
                t_swing_start=t_ss_start, t_swing_end=t_ss_end)
            t += cfg.dt_ctrl

        # ── EXT Phase ────────────────────────────────────────────────────
        if verbose:
            print(f"  EXT: {t:.2f} -> dock or +{cfg.t_ext_max}s")

        t_ext_start = t
        docked = False

        # During SS, only stance arm is in contact
        active_a_ext = (stance_arm == 'a')
        active_b_ext = (stance_arm == 'b')

        while t < t_ext_start + cfg.t_ext_max and not docked:
            hw, L_com_prev = self._step(
                t, 'EXT', hw, L_com_prev, log,
                active_a=active_a_ext, active_b=active_b_ext,
                t_com_start=t_com_start, t_com_end=t_com_end,
                t_swing_start=t_ss_start, t_swing_end=t_ss_end)
            t += cfg.dt_ctrl

            mujoco.mj_forward(self.mj_model, self.mj_data)
            d = self._gripper_distance(swing_arm, cfg.target_idx)
            if d < cfg.weld_radius:
                docked = True
                log.dock_events.append({
                    't': round(t, 3),
                    'd_mm': round(d * 1000, 2),
                    'arm': swing_arm,
                    'anchor': cfg.target_idx,
                })
                if verbose:
                    print(f"  *** DOCK t={t:.2f}s d={d*1000:.1f}mm ***")

        if not docked and verbose:
            recent = log.d_grip_swing[-20:] if len(log.d_grip_swing) >= 20 \
                else log.d_grip_swing
            if recent:
                print(f"  TIMEOUT: min d={min(recent)*1000:.1f}mm")

        # Post-dock: activate weld
        if docked:
            self._activate_weld(swing_arm, cfg.target_idx)
            mujoco.mj_forward(self.mj_model, self.mj_data)

        if verbose:
            self._print_summary(log)

        return log

    # ── Single control step ──────────────────────────────────────────────

    def _step(self, t, phase, hw, L_com_prev, log,
              active_a, active_b,
              t_com_start, t_com_end,
              t_swing_start, t_swing_end):
        """One Lutze QP step (10Hz) with inner MuJoCo loop (100Hz)."""
        cfg = self.cfg
        swing_arm = cfg.swing_arm
        stance_arm = 'a' if swing_arm == 'b' else 'b'

        # --- Read state ---
        pin_q, pin_v = mujoco_to_pinocchio(
            self.mj_data.qpos, self.mj_data.qvel)
        cs = compute_centroidal_state(
            self.pin_model, self.pin_data, pin_q, pin_v)

        if L_com_prev is None:
            L_com_prev = cs.L_com.copy()

        # --- CoM reference ---
        r_com_ref, v_com_ref, _ = quintic_interp(
            t, t_com_start, t_com_end,
            self.r_com_start, self.r_com_end)

        # --- Structure state (for stabilization feedback) ---
        struct_quat_wxyz = self.mj_data.qpos[3:7].copy()
        struct_omega = self.mj_data.qvel[3:6].copy()

        # --- Feedforward wrenches ---
        F_d_r, F_d_b = compute_feedforward(
            cs.r_com, cs.v_com, r_com_ref, v_com_ref,
            struct_quat_wxyz=struct_quat_wxyz,
            struct_omega=struct_omega,
            cfg=self.ff_config)

        # --- Contact adjoints and momentum map ---
        # Determine which contacts are active for the QP
        qp_active_a = active_a and (swing_arm != 'a' or phase == 'DS')
        qp_active_b = active_b and (swing_arm != 'b' or phase == 'DS')

        Ad_a, Ad_b = compute_dual_contact_adjoints(
            self.pin_model, self.pin_data, pin_q,
            active_a=qp_active_a, active_b=qp_active_b)

        # Contact positions for momentum map
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        fid_a = self.pin_model.getFrameId('tool_a')
        fid_b = self.pin_model.getFrameId('tool_b')
        r_a = self.pin_data.oMf[fid_a].translation.copy() if qp_active_a else None
        r_b = self.pin_data.oMf[fid_b].translation.copy() if qp_active_b else None
        M_lambda = compute_momentum_map(cs.r_com, r_a, r_b)

        # --- Solve Lutze QP ---
        Fc_a, Fc_b, qp_info = self.lutze_qp.solve(
            Ad_a, Ad_b, M_lambda, F_d_r, F_d_b)

        # --- Inner loop: apply torques at 100Hz ---
        tau_last = np.zeros(12)
        for qs in range(self.n_sim_per_ctrl):
            tq = t + qs * cfg.dt_sim

            # Re-read state and recompute Jacobians for fresh J^T mapping
            pin_q_inner, pin_v_inner = mujoco_to_pinocchio(
                self.mj_data.qpos, self.mj_data.qvel)
            J_a, J_b = get_contact_jacobians(
                self.pin_model, self.pin_data, pin_q_inner)

            # Wrench -> joint torques via J^T
            tau = compute_joint_torques(
                Fc_a, Fc_b, J_a, J_b,
                active_a=qp_active_a, active_b=qp_active_b,
                tau_max=cfg.tau_max)

            # Swing arm joint-space PD during SS/EXT
            if phase in ('SS', 'EXT'):
                # Joint-space PD: interpolate from start to end joint angles
                # This works better than Cartesian impedance for free-floating
                # robots because it avoids the base counter-rotation issue.
                if phase == 'SS':
                    # Quintic interpolation of swing arm joint angles
                    q_sw_ref, dq_sw_ref, _ = quintic_interp(
                        tq, t_swing_start, t_swing_end,
                        self.q_swing_start, self.q_swing_end)
                else:  # EXT: drive toward IK end configuration
                    q_sw_ref = self.q_swing_end.copy()
                    dq_sw_ref = np.zeros(6)

                # Current swing arm joint state from MuJoCo
                if swing_arm == 'b':
                    q_sw_cur = self.mj_data.qpos[20:26].copy()
                    dq_sw_cur = self.mj_data.qvel[18:24].copy()
                    sw_slice = slice(12, 18)  # in generalized force vector
                else:
                    q_sw_cur = self.mj_data.qpos[14:20].copy()
                    dq_sw_cur = self.mj_data.qvel[12:18].copy()
                    sw_slice = slice(6, 12)

                # PD torques on swing arm joints
                tau_sw = (cfg.swing_Kp * (q_sw_ref - q_sw_cur) +
                          cfg.swing_Kd * (dq_sw_ref - dq_sw_cur))
                tau_sw = np.clip(tau_sw, -cfg.tau_max, cfg.tau_max)
                tau[sw_slice] += tau_sw


            # Clip total and apply
            tau_joints = np.clip(tau[6:18], -cfg.tau_max, cfg.tau_max)
            tau_last = tau_joints.copy()
            self.mj_data.ctrl[:12] = tau_joints
            mujoco.mj_step(self.mj_model, self.mj_data)

            # Update wheel momentum (simplified)
            pin_q2, pin_v2 = mujoco_to_pinocchio(
                self.mj_data.qpos, self.mj_data.qvel)
            cs2 = compute_centroidal_state(
                self.pin_model, self.pin_data, pin_q2, pin_v2)
            hw -= (cs2.L_com - cs.L_com)
            hw = np.clip(hw, cfg.hw_min, cfg.hw_max)

        # --- Logging ---
        mujoco.mj_forward(self.mj_model, self.mj_data)
        pin_q_f, pin_v_f = mujoco_to_pinocchio(
            self.mj_data.qpos, self.mj_data.qvel)
        cs_f = compute_centroidal_state(
            self.pin_model, self.pin_data, pin_q_f, pin_v_f)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

        # Torso position from Pinocchio
        fid_torso = self.pin_model.getFrameId('Link_0')
        if fid_torso >= self.pin_model.nframes:
            # Fallback: use CoM as torso proxy
            p_torso = cs_f.r_com.copy()
        else:
            p_torso = self.pin_data.oMf[fid_torso].translation.copy()

        d_swing = self._gripper_distance(swing_arm, cfg.target_idx)
        stance_idx = cfg.start_a if stance_arm == 'a' else cfg.start_b
        d_stance = self._gripper_distance(stance_arm, stance_idx)

        L_dot_est = (cs_f.L_com - L_com_prev) / cfg.dt_ctrl
        sq = self.mj_data.qpos[3:7].copy()
        euler = quat_wxyz_to_euler_deg(sq[0], sq[1], sq[2], sq[3])

        Fc_stacked = np.concatenate([Fc_a, Fc_b])

        log.t.append(t)
        log.phase.append(phase)
        log.p_torso.append(p_torso)
        log.p_torso_ref.append(r_com_ref.copy())
        log.e_torso_pos.append(float(np.linalg.norm(p_torso - r_com_ref)))
        log.d_grip_swing.append(d_swing)
        log.d_grip_stance.append(d_stance)
        log.r_com.append(cs_f.r_com.copy())
        log.r_com_ref.append(r_com_ref.copy())
        log.e_com.append(float(np.linalg.norm(cs_f.r_com - r_com_ref)))
        log.L_com.append(cs_f.L_com.copy())
        log.L_com_norm.append(float(np.linalg.norm(cs_f.L_com)))
        log.L_dot.append(L_dot_est.copy())
        log.L_dot_norm.append(float(np.linalg.norm(L_dot_est)))
        log.hw.append(hw.copy())
        log.tau.append(tau_last.copy())
        log.tau_max_joint.append(float(np.max(np.abs(tau_last))))
        log.struct_pos.append(self.mj_data.qpos[0:3].copy())
        log.struct_quat.append(sq)
        log.struct_euler_deg.append(euler)
        log.qp_status.append(qp_info.get('status', 'unknown'))
        log.lambda_ref_norm.append(float(np.linalg.norm(Fc_stacked)))

        return hw, cs_f.L_com.copy()

    # ── Summary ──────────────────────────────────────────────────────────

    def _print_summary(self, log):
        t = np.array(log.t)
        Ln = np.array(log.L_com_norm)
        Ldn = np.array(log.L_dot_norm)
        euler = np.array(log.struct_euler_deg)
        sp = np.array(log.struct_pos)

        print(f"\n{'='*60}")
        print(f"LUTZE BASELINE — SIMULATION SUMMARY")
        print(f"{'='*60}")
        print(f"Duration:        {t[-1]:.1f}s")
        print(f"Dock events:     {len(log.dock_events)}")
        for ev in log.dock_events:
            print(f"  t={ev['t']}s d={ev['d_mm']}mm arm={ev['arm']}")
        print(f"max |tau_joint|:  {max(log.tau_max_joint):.2f} Nm "
              f"(lim {self.cfg.tau_max})")
        print(f"max ||L_com||:    {Ln.max():.2f} Nms "
              f"(lim {self.cfg.L_max})")
        print(f"max ||L_dot||:    {Ldn.max():.2f} Nm "
              f"(lim {self.cfg.tau_w_max})")
        print(f"Struct drift:     {np.linalg.norm(sp[-1]-sp[0])*100:.1f} cm")
        print(f"Struct rotation:  roll={euler[-1,0]:.2f} "
              f"pitch={euler[-1,1]:.2f} yaw={euler[-1,2]:.2f} deg")
        print(f"Max |angle|:      {np.max(np.abs(euler)):.2f} deg")
        nf = sum(1 for s in log.qp_status if s != 'optimal')
        print(f"QP non-optimal:   {nf}/{len(log.qp_status)}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Single-step Lutze baseline simulation')
    parser.add_argument('--mjcf', default='models/VISPA_crawling.xml')
    parser.add_argument('--urdf', default='models/VISPA_crawling_fixed.urdf')
    parser.add_argument('--output', default='sim_lutze_log.json',
                        help='Output log file')
    parser.add_argument('--swing-arm', default='b', choices=['a', 'b'])
    parser.add_argument('--start-a', type=int, default=2)
    parser.add_argument('--start-b', type=int, default=2)
    parser.add_argument('--target', type=int, default=3)
    parser.add_argument('--struct-mass', type=float, default=2040.0)
    args = parser.parse_args()

    cfg = SimLutzeConfig(
        swing_arm=args.swing_arm,
        start_a=args.start_a,
        start_b=args.start_b,
        target_idx=args.target,
        struct_mass=args.struct_mass,
    )

    sim = SimulationLutze(args.mjcf, args.urdf, cfg)
    sim.setup()

    t0 = time.time()
    log = sim.run()
    elapsed = time.time() - t0
    print(f"\nSimulation time: {elapsed:.1f}s")

    log.save(args.output)
    print(f"Log saved to {args.output}")


if __name__ == '__main__':
    main()
