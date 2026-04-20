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

    # ── Timing (M7: two-phase state machine) ────────────────────
    # Per spec §6 and HANDOFF §M7: there is no EXT phase, no fixed
    # swing duration, and no torso delay. DS exits on T < T_settle
    # (energy-based, spec §7.1.1). SS runs for T_step + t_ss_margin
    # where T_step is produced by the coarse pre-planner.
    dt_nmpc: float = 0.1          # NMPC period [s] (10 Hz)
    dt_qp: float = 0.01           # QP/MuJoCo period [s] (100 Hz)
    t_ss_margin: float = 1.0      # Extra SS time beyond T_step before timeout [s]
    t_hold_max: float = 3.0       # Convergence hold after timeout before aborting step [s]
    dock_check_delay: float = 0.5 # Skip dock checks for first N s of SS (avoid release noise) [s]
    n_ds_max_steps: int = 1000    # Safety cap on energy-based DS settle (10 s @ 100 Hz)

    # ── Actuator limits ─────────────────────────────────────────
    tau_max: float = 20.0         # Joint torque limit [Nm]

    # ── Docking ─────────────────────────────────────────────────
    weld_radius: float = 0.005    # Real dock threshold [m]
    # Orientation gate for dock activation. The anchor frame is Identity
    # in the structure frame (verified 2026-04-11), so this is the angle
    # between the gripper's rotation matrix and I. MuJoCo's weld is
    # position-gated only — without this gate, docking at large ori
    # misalignment corrupts the next step's initial conditions.
    dock_ori_threshold_deg: float = 5.0

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
    # Tight QP-level hw safety bounds: the main SS QP uses these in its
    # soft-slack momentum safety constraint so the chosen contact wrench
    # stays well within the physical ±hw_max envelope, leaving AOCS
    # headroom. NMPC (`h_max_tight`) and AOCS still see the physical
    # ±5 Nms limit, so the momentum handoff remains consistent.
    hw_qp_tight: np.ndarray = field(default_factory=lambda: np.full(3, 3.0))
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
    alpha_com_soft: float = 0.0   # Soft CoM residual disabled — redundant with torso 6D position task; 5.0 was fighting torso tracking
    alpha_passivity: float = 1.0  # DS passivity decay rate [1/s]

    # ── M3: NMPC conservation-law box constraint ────────────────
    enforce_hw_conservation: bool = False  # Enable B2 Option B hw box
    h_max_tight: np.ndarray = field(default_factory=lambda: np.full(3, 5.0))  # Tightened [Nms]
    w_L_nmpc: float = 1.0         # Cost weight on ||L_com - L_com_ref||²
    kappa_terminal: float = 1.0   # Terminal margin multiplier

    # ── M6/M7: coarse pre-planner (mandatory) ────────────────────
    # Runs once per step before SS starts. Solves a centroidal NLP
    # over [0, T_step] to produce (a) a momentum-feasible CoM
    # trajectory and (b) the T_step that the TorsoPlanner and
    # SwingPlanner use to synchronize their trajectories. The
    # pre-planner is mandatory for M7 — there is no use_* flag to
    # disable it. On solver failure the sim loop logs the failure
    # and skips the step (no silent heuristic fallback); unit tests
    # that want to avoid the IPOPT dependency use
    # CoarsePlanResult.from_heuristic() directly.
    preplanner_M: int = 15                  # collocation intervals
    preplanner_kappa: float = 0.7           # terminal margin multiplier (< 1)
    preplanner_f_max: float = 25.0          # [N] per active contact
    preplanner_tau_max: float = 8.0         # [Nm] per active contact
    preplanner_w_L: float = 1.0             # cost weight on ||L_com||²
    preplanner_w_u: float = 1e-2            # cost weight on ||[f; τ]||²
    preplanner_max_iter: int = 300          # IPOPT max iterations
    preplanner_a_cruise_max: float = 0.0     # [m/s²] cruise accel limit (0=off)
    preplanner_cruise_ramp_frac: float = 0.2 # ramp fraction for cruise window

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
    ss_alpha_wrench: float = 1e-2  # pure regularisation; 1e2 was penalising contact forces (the only actuation path through the stance weld) and attenuating the torso task 7x (see scripts/test_qp_tracking.py)
    ss_alpha_reaction: float = 0.0   # Reaction null-space (0 = disabled)

    # ── QP gains — Single-support ──────────────────────────────
    ss_Kp_com: float = 3.0
    ss_Kd_com: float = 3.0
    ss_Kp_torso: float = 6.0
    ss_Kd_torso: float = 5.0
    ss_Kp_ee: float = 10.0
    ss_Kd_ee: float = 12.0
    ss_Kp_ee_ang: float = 6.0
    ss_Kd_ee_ang: float = 4.5

    # ── Swing planner ──────────────────────────────────────────
    swing_clearance: float = 0.03  # [m]
    # Symmetric sin²(πτ) bump (peak at τ=0.5). The τ=0.25 shift was
    # a workaround for the folded-arm (N_torso·J_ee) singularity
    # around mid-swing; with the manipulability-optimized init
    # keeping κ(N_t·J_ee) ≤ 6.5 throughout SS (v8 trace), that
    # workaround is no longer needed.
    swing_bump_peak_tau: float = 0.5

    # ── M7 change A: minimize torso reorientation per step ──────
    # The IK per step first tries to solve with torso rotation held
    # at R_start ("the robot crawls forward, it doesn't pirouette").
    # Only if the resulting manipulability product w_a*w_b falls
    # below `ik_fixed_rotation_w_min` do we fall back to the
    # manipulability-optimized configuration from the torso_map.
    # Threshold is in the same units as w_a*w_b (dimensionally
    # m^6 for two 6D Jacobians); 1e-4 is a conservative floor —
    # well above near-singularity (w < ~1e-6).
    ik_fixed_rotation: bool = True
    ik_fixed_rotation_w_min: float = 1e-4

    # ── Torso-vs-swing velocity profile ─────────────────────────
    # 1.0 = torso quintic runs over the full [0, T_step] alongside
    # the swing. The 0.7 stagger was a workaround for the folded-arm
    # (N_torso·J_ee) singularity around the bump peak; with the
    # manipulability-optimized init keeping κ(N_t·J_ee) ≤ 6.5
    # throughout SS (v8 trace), that workaround is no longer needed.
    torso_early_finish_fraction: float = 1.0

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
    n_settle_damping_steps: int = 0         # stage 1: skipped — manipulability-optimized init places arms near weld equilibrium, no impulse to absorb; stage 2 passivity QP holds posture
    Kd_settle_damping: float = 20.0         # Nm·s/rad per joint (stage 1)
    n_settle_max_steps: int = 1000          # stage 2: safety cap
    settle_epsilon_v: float = 1e-3          # target ‖dq_full‖ bound [m/s]
    settle_plateau_ratio: float = 0.999     # T(k+50) > ratio·T(k) → plateau

    # ── Post-abort DS diagnostic flags (2026-04-17) ─────────────
    # All three default False. When enabled individually they apply
    # ONLY to the trailing-DS phase entered after a dock_timeout
    # abort — never to SS, never to the pre-SS DS settle, never to
    # the pre-planner. See docs/architecture/M7_DS_DIAGNOSTIC_EXPERIMENTS.md.
    diag_freeze_torso_ref_on_abort: bool = False
    # Diagnostic for H_DS2 (POST_ABORT_DIVERGENCE.md).
    # When True, skip dock_configuration + set_hold at sim_loop.py:1365-1375
    # on trailing-DS entry after dock_timeout, and freeze the TorsoPlanner
    # hold target to the actual oMf_torso at the last SS sample.

    diag_force_single_contact_on_abort: bool = False
    # Diagnostic for H_DS1 (POST_ABORT_DIVERGENCE.md).
    # When True, force cc_ds = ContactConfig.from_phase(
    #   ContactPhase.SINGLE_A, r_contact_a, r_contact_b) at sim_loop.py:1343
    # on trailing-DS entry after dock_timeout, matching the physical state.

    diag_disable_passivity_on_abort: bool = False
    # Diagnostic for H_DS3 (POST_ABORT_DIVERGENCE.md).
    # When True, pass passivity_active=False to the QP during trailing DS
    # entered after dock_timeout, overriding the phase=='DS' gate at
    # sim_loop.py:1712.

    mapping_bypass_in_ss: bool = False
    # M7 EE bisection follow-up. When True, sim_loop bypasses the
    # CoM->torso mapping during SS only: the QP receives
    #   r_torso_ref = r_torso(t = t_ss_start)   (frozen at SS entry)
    #   v_torso_ref_lin = 0
    #   a_torso_ff_lin  = 0
    # for the linear components of the torso reference. Angular
    # reference still comes from TorsoPlanner (orientation tracking
    # unchanged). DS phase is unchanged (mapping still active there).
