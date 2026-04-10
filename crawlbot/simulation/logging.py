"""Simulation data logger.

SimLog collects time-series data from each NMPC step for post-processing
and plotting. All fields are lists of scalars or numpy arrays.
"""

import json
import numpy as np
from dataclasses import dataclass, field


@dataclass
class SimLog:
    """Comprehensive logged data from a simulation run."""

    t: list = field(default_factory=list)
    phase: list = field(default_factory=list)
    step_idx: list = field(default_factory=list)

    # Torso tracking
    p_torso: list = field(default_factory=list)
    p_torso_ref: list = field(default_factory=list)
    e_torso_pos: list = field(default_factory=list)
    e_torso_ori: list = field(default_factory=list)
    q_torso: list = field(default_factory=list)       # actual torso quat (wxyz)
    q_torso_ref: list = field(default_factory=list)    # reference torso quat (wxyz)

    # End-effector
    d_grip_swing: list = field(default_factory=list)
    d_grip_stance: list = field(default_factory=list)
    swing_arm: list = field(default_factory=list)
    p_ee: list = field(default_factory=list)           # EE position actual (3,)
    p_ee_ref: list = field(default_factory=list)       # EE position ref (3,)
    q_ee: list = field(default_factory=list)           # EE orientation actual quat (wxyz)
    q_ee_ref: list = field(default_factory=list)       # EE orientation ref quat (wxyz)

    # CoM
    r_com: list = field(default_factory=list)
    r_com_ref: list = field(default_factory=list)
    e_com: list = field(default_factory=list)
    v_com: list = field(default_factory=list)           # CoM velocity actual (3,)
    v_com_ref: list = field(default_factory=list)       # CoM velocity ref (3,)

    # Momentum
    L_com: list = field(default_factory=list)
    L_com_norm: list = field(default_factory=list)
    L_com_ref: list = field(default_factory=list)       # NMPC-planned L_com (3,)
    L_dot: list = field(default_factory=list)
    L_dot_norm: list = field(default_factory=list)
    hw: list = field(default_factory=list)

    # RWA physical
    hw_physical: list = field(default_factory=list)
    tau_w: list = field(default_factory=list)
    rw_speed: list = field(default_factory=list)

    # EE tracking error (vs planned trajectory, not just target distance)
    e_ee_pos: list = field(default_factory=list)
    e_ee_ori: list = field(default_factory=list)

    # GMO contact estimator
    gmo_residual_norm: list = field(default_factory=list)
    gmo_swing_residual: list = field(default_factory=list)
    gmo_contact_state: list = field(default_factory=list)

    # H_{r/O} estimator diagnostics
    H_rO: list = field(default_factory=list)
    H_dot_est: list = field(default_factory=list)
    omega_struct: list = field(default_factory=list)
    qfrc_constraint_torque: list = field(default_factory=list)

    # Joint torques
    tau: list = field(default_factory=list)
    tau_max_joint: list = field(default_factory=list)

    # Structure state
    struct_pos: list = field(default_factory=list)
    struct_quat: list = field(default_factory=list)
    struct_euler_deg: list = field(default_factory=list)
    omega_s: list = field(default_factory=list)         # platform angular velocity (3,)

    # Solver diagnostics
    nmpc_ok: list = field(default_factory=list)
    qp_ok: list = field(default_factory=list)
    lambda_ref_norm: list = field(default_factory=list)
    nmpc_time_ms: list = field(default_factory=list)
    qp_time_ms: list = field(default_factory=list)
    nmpc_status: list = field(default_factory=list)     # 0=ok, 1=max_iter, 2=infeasible
    nmpc_cost: list = field(default_factory=list)       # NMPC objective value

    # Contact wrenches
    lambda_ref: list = field(default_factory=list)      # NMPC planned wrench (12,)
    lambda_qp: list = field(default_factory=list)       # QP contact wrench (12,)

    # Energy / passivity
    T_kinetic: list = field(default_factory=list)       # 0.5 * dq^T H dq

    # Setup-phase settling (populated by _settle_setup)
    settling_t: list = field(default_factory=list)      # (n,) time [s]
    settling_T: list = field(default_factory=list)      # (n,) kinetic energy [J]
    settling_T_target: float = 0.0                      # T_settle threshold
    settling_stage1_steps: int = 0
    settling_stage2_steps: int = 0
    settling_exit_reason: str = ''

    # Dock events
    dock_events: list = field(default_factory=list)

    # MuJoCo snapshots for offline rendering
    snapshots: list = field(default_factory=list)       # [(t, qpos, qvel, label)]

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if k == 'snapshots':
                # Snapshots contain large arrays; serialize specially
                d[k] = [(t, q.tolist() if hasattr(q, 'tolist') else q,
                          v_.tolist() if hasattr(v_, 'tolist') else v_, lbl)
                         for t, q, v_, lbl in v]
            elif isinstance(v, list):
                d[k] = [x.tolist() if hasattr(x, 'tolist') else x for x in v]
            else:
                # Scalar fields (e.g. settling_T_target, settling_stage1_steps)
                d[k] = v
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
