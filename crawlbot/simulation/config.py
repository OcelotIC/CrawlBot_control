"""Simulation configuration dataclass.

All tunable parameters for the NMPC+QP+AOCS pipeline in one place.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class SimConfig:
    """Full simulation configuration.

    Parameters are grouped by subsystem. Defaults match the validated
    configuration at 8% mass ratio (888 kg structure, 3/3 docks, 10°).
    """

    # ── Timing ──────────────────────────────────────────────────
    dt_nmpc: float = 0.1          # NMPC period [s] (10 Hz)
    dt_qp: float = 0.01           # QP/MuJoCo period [s] (100 Hz)
    t_ds: float = 0.5             # Double-support duration [s]
    t_swing: float = 6.0          # Single-support (swing) duration [s]
    t_ext_max: float = 10.0       # Max extension phase before timeout [s]

    # ── Torso trajectory ────────────────────────────────────────
    torso_delay: float = 0.20     # Delay before torso starts (fraction of t_swing)

    # ── Actuator limits ─────────────────────────────────────────
    tau_max: float = 20.0         # Joint torque limit [Nm]

    # ── Docking ─────────────────────────────────────────────────
    weld_radius: float = 0.005    # Real dock threshold [m]

    # ── Momentum constraints (NMPC + QP) ────────────────────────
    hw_init: np.ndarray = field(default_factory=lambda: np.zeros(3))
    hw_min: np.ndarray = field(default_factory=lambda: np.full(3, -5.0))
    hw_max: np.ndarray = field(default_factory=lambda: np.full(3, 5.0))
    L_max: float = 10.0           # Robot angular momentum limit [Nms]
    tau_w_max: float = 5.0        # Reaction wheel torque limit [Nm]

    # ── AOCS ────────────────────────────────────────────────────
    aocs_K_hw: float = 2.0        # Legacy feedback gain [1/s]
    aocs_tau_w_max: float = 5.0   # Max wheel torque [Nm]
    rwa_I_w: float = 0.01         # Wheel spin inertia [kg·m²]

    # Mode: 'legacy' | 'H_est' | 'nmpc_plan'
    aocs_mode: str = 'legacy'
    aocs_use_H_estimator: bool = False     # use aocs_mode to select
    aocs_filter_tau: float = 0.016
    aocs_K_omega: float = 50.0
    aocs_K_h: float = 0.5
    aocs_hw_target: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # ── NMPC solver ─────────────────────────────────────────────
    nmpc_N: int = 8
    nmpc_dt: float = 0.1
    nmpc_f_max: float = 25.0
    nmpc_tau_max: float = 8.0
    nmpc_Wv: float = 10.0
    nmpc_p_max: float = 50.0      # Linear momentum bound [kg·m/s]
    t_settle_final: float = 20.0

    # ── QP weights — Single-support ─────────────────────────────
    ss_alpha_com: float = 2e2
    ss_alpha_torso: float = 5e2
    ss_alpha_ee: float = 3e3
    ss_alpha_posture: float = 2e1
    ss_alpha_wrench: float = 1e2
    ss_alpha_reaction: float = 0.0   # Reaction null-space (0 = disabled)

    # ── QP weights — Extension ──────────────────────────────────
    ext_alpha_com: float = 1e2
    ext_alpha_torso: float = 5e1
    ext_alpha_ee: float = 1e4
    ext_alpha_posture: float = 5e0
    ext_alpha_wrench: float = 1e2
    ext_alpha_reaction: float = 0.0  # Reaction null-space (0 = disabled)

    # ── QP gains — Single-support ──────────────────────────────
    ss_Kp_com: float = 3.0
    ss_Kd_com: float = 3.0
    ss_Kp_torso: float = 6.0
    ss_Kd_torso: float = 5.0
    ss_Kp_ee: float = 10.0
    ss_Kd_ee: float = 7.0

    # ── QP gains — Extension ───────────────────────────────────
    ext_Kp_com: float = 2.0
    ext_Kd_com: float = 2.0
    ext_Kp_torso: float = 3.0
    ext_Kd_torso: float = 3.0
    ext_Kp_ee: float = 40.0
    ext_Kd_ee: float = 22.0

    # ── Swing planner ──────────────────────────────────────────
    swing_clearance: float = 0.03  # [m]

    # ── MuJoCo settling ────────────────────────────────────────
    n_settle_steps: int = 500
