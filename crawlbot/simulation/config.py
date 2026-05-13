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

    # Diagnostic: zero wheel torque during DS (Mode B still runs in SS).
    aocs_off_in_ds: bool = False

    # When True, the gait-phase loop breaks out of the multi-step
    # traversal as soon as a step ABORTs (dock_timeout or
    # preplanner_infeasible). Subsequent steps would only observe
    # post-failure cascade dynamics, not the controller in normal
    # operation, so running them is wasted compute. Default True
    # (fail-fast diagnostics); set False to reproduce the legacy
    # full-cascade runs.
    stop_on_failed_step: bool = True

    # ── M2: reworked QP task stack ──────────────────────────────
    use_m2_stack: bool = False    # Enable reworked QP (torso P1 + EE null-space P2 + soft CoM)
    alpha_com_soft: float = 0.0   # Soft CoM residual disabled — redundant with torso 6D position task; 5.0 was fighting torso tracking
    alpha_passivity: float = 1.0  # DS passivity decay rate [1/s]

    # ── Cooperative-arms mode (deviation from spec §4.3) ────────
    # When True, the WBC task stack splits the torso 6D task into:
    #   P1 — torso ANGULAR 3D (strict-equality strength, alpha_torso_ang)
    #   P2 — torso LINEAR 3D + EE 6D as weighted-LS co-contributors,
    #        BOTH projected through N_torso_ang (no projection between them)
    #   P3 — posture, projected through N(combined P1+P2)
    # This drops the strict EE→torso null-space hierarchy: the EE task can
    # now pull the body forward in the linear DOFs when its reach margin
    # demands it. The torso angular DOF stays strict (protects AOCS
    # budget). Stance-arm thrust still flows passively through the
    # dynamics constraint (no explicit QP cost on stance thrust —
    # documented limitation; cooperative reaction is structurally
    # rewarded by the LS arbitration, not by a dedicated task).
    # Default False preserves the legacy M2 strict-priority stack.
    cooperative_arms_mode: bool = False

    # ── Stance-thrust inertial-coupling correction (experimental, OFF) ─
    # Negative-result experiment: see WholeBodyQPConfig docstring and
    # results/diag_cooperative_arms_thrust/. Adds an explicit
    # M_fj · ddq_j feedforward to the stance wrench reference. Was
    # hypothesized to close the 2.4 mm step-2 gap by absorbing the
    # joint-acceleration coupling at the contact; in practice it
    # regressed step 0 to TIMEOUT 12.1 mm. Kept available behind this
    # flag for future investigation.
    stance_thrust_correction: bool = False

    # ── Option D: torso linear soft tube ────────────────────────
    # When r_tube > 0, the WBC torso P1 task is split:
    #   (a) angular 3D — always enforced at the full α_torso weight
    #       (hard-equality-strength task)
    #   (b) linear 3D — gated. Skipped while ||p_torso - p_ref|| ≤ r_tube;
    #       outside the tube, added with weight w_tube_lin · (|e|² - r²)
    #       so the cost grows with the violation magnitude.
    # When the linear task is skipped, the null-space projection for
    # the EE / posture / soft-CoM tasks is computed against the 3D
    # angular Jacobian only, freeing the linear-base DOFs for EE
    # tracking. Default 0 disables the tube (legacy 6D equality task).
    r_tube: float = 0.0           # [m] tube radius for torso linear position
    w_tube_lin: float = 50.0      # cost scale on (|e|² - r²) when violated

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
    preplanner_f_max: float = 25.0          # [N] per active contact (also used by F-SAT clamp)
    preplanner_tau_max: float = 8.0         # [Nm] per active contact
    preplanner_w_L: float = 1.0             # cost weight on ||L_com||²
    preplanner_w_u: float = 1e-2            # cost weight on ||[f; τ]||²
    preplanner_max_iter: int = 300          # IPOPT max iterations
    preplanner_a_cruise_max: float = 0.0     # [m/s²] cruise accel limit (0=off)
    preplanner_cruise_ramp_frac: float = 0.2 # ramp fraction for cruise window

    # ── NMPC solver ─────────────────────────────────────────────
    nmpc_N: int = 8
    nmpc_dt: float = 0.1
    nmpc_f_max: float = 300.0
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
    ss_alpha_torso_ang: float = 5e2  # Cooperative-arms P1 torso angular weight
    ss_alpha_torso_lin: float = 5e2  # Cooperative-arms P2 torso linear weight (co-equal w/ EE)
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

    # ── M7 Manipulability-IK-1: trajectory-aware IK (Candidate 1) ──
    # When True, sim_loop builds an additional torso_map_traj dict
    # whose q_end is optimised for worst-case σ_min(J_a)·σ_min(J_b)
    # across K interior samples of the planned SS swing, chained by
    # q_start = q_end_prev. Falls back to the endpoint torso_map when
    # the chain drift exceeds `trajectory_ik_qstart_tolerance`.
    # Default False preserves existing behaviour (bit-for-bit ablation).
    # See docs/architecture/T15_MANIPULABILITY_IK_DESIGN.md.
    use_trajectory_aware_ik: bool = False
    trajectory_ik_qstart_tolerance: float = 0.05  # [rad] chain-drift fallback
    trajectory_ik_n_samples: int = 5              # K samples along τ∈(0,1]
    # Below this σ_min(J_a)·σ_min(J_b) at the converged trajectory-aware
    # IK endpoint, the result is considered pathological (likely landed
    # in a singular branch despite §9.1–9.2 fixes). The caller should
    # fall back to fixed_rotation IK. Spec: IK_FORMULATION.md §9.3.
    trajectory_ik_w_min_threshold: float = 1e-3

    # ── Mid-waypoint reshape (Option B per T15_step2_path_geometry §7.3) ──
    # When True, after the SS-entry IK returns q_end, sample the implied
    # planner-style reference path's whole-body Jacobian conditioning at
    # 21 points (check_path_feasibility). If any sample's σ_min product
    # falls below path_feasibility_w_threshold, attempt a mid-waypoint
    # reshape via manipulability_config_mid_waypoint and pass the result
    # to TorsoPlanner / SwingPlanner as a piecewise quintic. This
    # addresses the H2 finding (T15_step2_path_geometry §6.2): the
    # single-quintic reference for some anchor pairs visits a
    # near-singular interior region that the QP cannot track.
    use_path_feasibility_check: bool = False
    use_mid_waypoint_reshape: bool = False
    # When True (and use_mid_waypoint_reshape is also True), bypass the
    # feasibility check and ALWAYS attempt mid-waypoint reshape. Useful
    # for validation when the simplified planner-style references in
    # check_path_feasibility underreport vs the actual mapped references
    # (T15_step2_path_geometry §7 caveat: the M5 CoM-mapping layer is
    # not modelled in the runtime check). Default False preserves the
    # gated behaviour.
    mid_waypoint_force_on: bool = False
    # Threshold below which a sample's σ_min product is considered
    # pathologically singular. Per IK_FORMULATION.md §9.3.
    path_feasibility_w_threshold: float = 1e-3

    # ── Reference generation source (M7 / FK-on-smoothed-q) ─────
    # 'task_space'      : legacy independent SLERP per planner. Default;
    #                     keeps existing test/sim outputs byte-identical.
    # 'joint_space_fk'  : derive task-space refs from FK on a single
    #                     task-space-smoothed constrained geodesic.
    #                     Eliminates the kinematically-uncoupled-refs
    #                     failure mode at T15 step 2 (synthesis §6,
    #                     plan §2.2). Adds ~0.3 s smoother overhead at
    #                     each SS-entry; zero runtime cost in the QP loop.
    # See docs/architecture/T15_step2_diagnosis_and_resolution.md and
    # results/diagnostic/stance_deviation_along_geodesic/PHASE0_FINDINGS.md.
    reference_source: str = 'task_space'
    # Smoothed-geodesic discretisation parameters
    # (used only when reference_source='joint_space_fk').
    geodesic_n_tau: int = 21
    geodesic_n_iter: int = 120
    geodesic_tol: float = 1e-5

    # ── Torso-vs-swing velocity profile ─────────────────────────
    # 1.0 = torso quintic runs over the full [0, T_step] alongside
    # the swing. The 0.7 stagger was a workaround for the folded-arm
    # (N_torso·J_ee) singularity around the bump peak; with the
    # manipulability-optimized init keeping κ(N_t·J_ee) ≤ 6.5
    # throughout SS (v8 trace), that workaround is no longer needed.
    torso_early_finish_fraction: float = 1.0

    # Swing-planner analogue (M7 v22). When < 1.0, the swing trajectory
    # (position, clearance bump, orientation SLERP) completes at
    # t = t_ss_start + swing_early_finish_fraction · T_step and then
    # HOLDS (v=0, a=0) for the remainder of the planned window. Paired
    # with a gate in sim_loop that prevents dock activation before the
    # swing trajectory has finished, so contact is made only after the
    # commanded reference has settled.
    swing_early_finish_fraction: float = 1.0

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

    ds_ramp_duration_s: float = 2.0
    # Option A (2026-04-22): duration over which the torso linear
    # position reference is ramped from the SS-exit pose
    # (_ss_entry_p_torso) to the live DS mapping output
    # (mapping.compute(q_current)) after weld activation. Quintic
    # shape function s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5, C^2
    # continuous with s(0)=0, s(1)=1, s'(0)=s'(1)=s''(0)=s''(1)=0.
    # Set to 0.0 to disable (reverts to the pre-Option-A step
    # behavior). Introduced to close the T12 DS1 divergence;
    # see docs/architecture/M7_T12_MEMO.md §5.
