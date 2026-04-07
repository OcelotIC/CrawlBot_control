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

    # RWA physical
    hw_physical: list = field(default_factory=list)
    tau_w: list = field(default_factory=list)
    rw_speed: list = field(default_factory=list)

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

    # Solver diagnostics
    nmpc_ok: list = field(default_factory=list)
    qp_ok: list = field(default_factory=list)
    lambda_ref_norm: list = field(default_factory=list)
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
