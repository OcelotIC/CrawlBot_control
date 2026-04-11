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

    # ── Contact estimator (GMO) ────────────────────────────────
    gmo_K_O: float = 80.0              # Observer gain [1/s]
    gmo_F_threshold: float = 5.0       # Residual norm threshold [N]
    gmo_d_proximity: float = 0.020     # -> PROXIMITY [m]
    gmo_d_contact: float = 0.005       # -> CONTACT [m] (matches weld_radius)
    gmo_d_reset: float = 0.030         # -> NO_CONTACT [m]
    gmo_debounce_count: int = 3        # -> CONFIRMED [cycles @ 100Hz = 30ms]
    use_gmo_dock: bool = False          # False=legacy kinematic, True=GMO

    # ── Momentum constraints (NMPC + QP) ────────────────────────
    hw_init: np.ndarray = field(default_factory=lambda: np.zeros(3))
    hw_min: np.ndarray = field(default_factory=lambda: np.full(3, -5.0))
    hw_max: np.ndarray = field(default_factory=lambda: np.full(3, 5.0))
    L_max: float = 10.0           # Robot angular momentum limit [Nms]
    tau_w_max: float = 5.0        # Reaction wheel torque limit [Nm]
    tau_struct_max: float = np.inf  # Structure disturbance torque limit [Nm]

    # ── AOCS ────────────────────────────────────────────────────
    aocs_K_hw: float = 2.0        # Legacy feedback gain [1/s]
    aocs_tau_w_max: float = 5.0   # Max wheel torque [Nm]
    rwa_I_w: float = 0.01         # Wheel spin inertia [kg·m²]

    # Mode: 'legacy' | 'legacy_corrected' | 'H_est' | 'nmpc_plan'
    aocs_mode: str = 'legacy'
    aocs_use_H_estimator: bool = False     # use aocs_mode to select
    aocs_use_legacy_corrected: bool = False  # M4: add r_com × m·dv_com term
    aocs_filter_tau: float = 0.016
    aocs_K_omega: float = 50.0
    aocs_K_h: float = 0.5
    aocs_hw_target: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # ── M2: reworked QP task stack ──────────────────────────────
    use_m2_stack: bool = False    # Enable reworked QP (torso P1 + EE null-space P2 + soft CoM)
    alpha_com_soft: float = 5.0   # Weight for the soft CoM residual
    alpha_passivity: float = 1.0  # DS passivity decay rate [1/s]

    # ── M3: NMPC conservation-law box constraint ────────────────
    enforce_hw_conservation: bool = False  # Enable B2 Option B hw box
    h_max_tight: np.ndarray = field(default_factory=lambda: np.full(3, 5.0))  # Tightened [Nms]
    w_L_nmpc: float = 1.0         # Cost weight on ||L_com - L_com_ref||²
    kappa_terminal: float = 1.0   # Terminal margin multiplier

    # ── M6: coarse pre-planner ──────────────────────────────────
    # Runs once per step before SS starts. Solves a centroidal NLP
    # over the full step horizon to produce a momentum-feasible CoM
    # reference that replaces the TorsoPlanner's geometric path as
    # the NMPC reference. See crawlbot/planning/coarse_preplanner.py
    # and spec §6.2.
    use_coarse_preplanner: bool = False
    preplanner_M: int = 15                  # collocation intervals
    preplanner_kappa: float = 0.7           # terminal margin multiplier (< 1)
    preplanner_f_max: float = 25.0          # [N] per active contact
    preplanner_tau_max: float = 8.0         # [Nm] per active contact
    preplanner_w_L: float = 1.0             # cost weight on ||L_com||²
    preplanner_w_u: float = 1e-2            # cost weight on ||[f; τ]||²
    preplanner_max_iter: int = 300          # IPOPT max iterations

    # ── NMPC solver ─────────────────────────────────────────────
    nmpc_N: int = 8
    nmpc_dt: float = 0.1
    nmpc_f_max: float = 25.0
    nmpc_tau_max: float = 8.0
    nmpc_Wv: float = 10.0
    nmpc_p_max: float = 50.0      # Linear momentum bound [kg·m/s]
    t_settle_final: float = 20.0
    # Inter-step settle (between successive locomotion steps).
    # Spec §7.1.1 mandates an energy-based exit, not a fixed timer: after a
    # dock, the controller enters DS with passivity_active=True and remains
    # there until T_kin < T_settle_inter = 0.5 · epsilon_v² · lambda_min(H).
    # The fixed-timer version (`t_settle_inter`) is kept as a dead knob for
    # backwards compatibility but is no longer read.
    t_settle_inter: float = 0.0                     # [DEPRECATED] ignored
    use_energy_settle_inter: bool = True            # spec §7.1.1
    settle_inter_epsilon_v: float = 1e-3            # target ‖dq_full‖ [m/s]
    n_settle_inter_max_steps: int = 500             # safety cap (5 s @ 100 Hz)
    t_settle_inter_min: float = 0.1                 # min runtime [s]

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
    ss_Kd_ee: float = 12.0
    ss_Kp_ee_ang: float = 6.0
    ss_Kd_ee_ang: float = 4.5

    # ── QP gains — Extension ───────────────────────────────────
    ext_Kp_com: float = 2.0
    ext_Kd_com: float = 2.0
    ext_Kp_torso: float = 3.0
    ext_Kd_torso: float = 3.0
    ext_Kp_ee: float = 25.0
    ext_Kd_ee: float = 15.0
    ext_Kp_ee_ang: float = 10.0
    ext_Kd_ee_ang: float = 5.0

    # ── Swing planner ──────────────────────────────────────────
    swing_clearance: float = 0.03  # [m]

    # ── MuJoCo settling ────────────────────────────────────────
    n_settle_steps: int = 500

    # ── Setup-phase settling (weld-snap absorption + passivity decay) ──
    # Stage 1: pure joint damping (~100 steps) to absorb the weld
    # activation impulse (~300 N) that pure physics can't handle cleanly.
    # Stage 2: M2 QP with passivity_active=True, drives kinetic energy
    # toward T_settle = 0.5 * epsilon_v^2 * lambda_min(H).
    # Exit stage 2 when EITHER:
    #   (a) T < T_settle, or
    #   (b) T stops decreasing (plateau detection), or
    #   (c) n_settle_max_steps reached (safety cap).
    n_settle_damping_steps: int = 100       # stage 1: hard-damped steps
    Kd_settle_damping: float = 20.0         # Nm·s/rad per joint (stage 1)
    n_settle_max_steps: int = 1000          # stage 2: safety cap
    settle_epsilon_v: float = 1e-3          # target ‖dq_full‖ bound [m/s]
    settle_plateau_ratio: float = 0.999     # T(k+50) > ratio·T(k) → plateau
