# Control Stack Overview (CODE-GROUND TRUTH)

**Date:** 2026-07-26 · **Commit:** `eecbf94` (branch
`claude/review-closure-bloc-2-uwu1x7`; = `main`/`bfd5509` + the 51-commit
hygiene chantier)
**Status:** Code-verified reference. **This document supersedes the stale
intent in `brainstorming_reworked_architecture.md` and
`CLAUDE_CODE_HANDOFF.md` wherever they disagree.** Those describe *intended*
architecture and an outdated milestone state; this file describes what the
code in the canonical run **actually does**, grounded in `file:line`.

**Refresh note (review-closure Step 0).** The previous revision of this file
was dated 2026-05-27 and described the **pre-freeze cooperative-split QP**
(strict-P1 torso-angular, co-equal P2 via `N_torso_ang`, soft-CoM, α_ee=3000,
`w_hw_slack=1e4`, dead ∞ momentum boxes) and the **`legacy_corrected` AOCS**.
None of that is the canonical controller any more. Every section below was
re-read against the source at `eecbf94`; the discrepancy list in §3 is the
new one.

**Verification legend:** [V] read directly from the cited source at `eecbf94`
this session · [A] asserted by an earlier audit, not re-read here — treat as
provisional.

---

## 0. What "canonical" means — the config chain (read this first)

`SimConfig` **dataclass defaults are not the canonical values.** The canonical
operating point is the result of a four-layer override chain, and every layer
must be read before quoting a number:

| layer | file | what it sets |
|---|---|---|
| 1. dataclass defaults | `crawlbot/simulation/config.py` | the 8 %-mass-ratio historical baseline |
| 2. M7 base config | `scripts/run_m7_single_step.py:31-60` `_make_m7_config()` | `use_m2_stack=True`, `enforce_hw_conservation=True`, `h_max_tight=5`, `preplanner_M=15`, `preplanner_kappa=0.7`, `preplanner_f_max=25`, `preplanner_tau_max=8` |
| 3. runner overrides | `scripts/diag_cooperative_arms.py:242-…` `main()` | `use_com_z_standoff=True`/`com_z_standoff=-0.35`, `t_ss_margin=5.0`, `log_hifreq_ss=True`, `aocs_use_wrench_ff_in_ds=True`, `ds_torso_ref_from_state=True`, `ss_alpha_lambda_int=1.0`, **`ds_centroidal_mode=True`** (`:347`), `ik_level_axis=[0,0,1]`, `ik_w_posture=0.2`, `frames_per_step=5` |
| 4. canonical kwargs | `gate/replay_canonical.py:37-47` (verbatim from `Misc/scripts/diag_canonical2p5_run.py:126-140`, the script that produced the committed artifacts) | the frozen 2.5 operating point — table below |

**Canonical `dca.main(**kwargs)` — the frozen 2.5 point** [V]:

```
legacy=False              alpha_torso_lin=0.0        anchor_dx=0.8
mass_ratio=0.01           n_steps=6                  settle_seconds=20.0
aocs_mode='legacy_pid_numerical'   K_theta=1.0   K_omega=50.0   tau_w_max=2.5
ss_two_task=True          ss_alpha_mom=400.0         alpha_torso_pose=2000.0
ss_alpha_ee=1000.0        ss_alpha_posture=20.0      ss_alpha_wrench=1.0
ss_kp_torso=3.0           ss_kd_torso=2.5            qp_envelope_exact=True
interstep_settle_alpha_wrench=3.0   interstep_settle_epsilon_v=5e-3
```

plus `HierarchicalQP.regularization = 1e-6` pinned explicitly
(`gate/replay_canonical.py:20-29`).

Three traps this chain has already sprung on readers:

- **`ss_Kp_torso`/`ss_Kd_torso` are 6.0/5.0 in `config.py:351-352` but 3.0/2.5
  on the canonical.** The paper's `K_p=3, K_d=2.5` is right; the dataclass
  default is not the canonical. [V]
- **`aocs_mode` is `legacy_pid_numerical`**, not the `legacy_corrected` that
  both `_make_m7_config()` (`:49`) and `dca.main`'s own signature default
  (`:243`) declare. The kwarg overrides it. [V]
- **`qp_envelope_exact=True`** on the canonical, so the QP momentum-rate box
  uses the origin-referenced exact Ḣ_s, not the `M_λ` proxy. [V]

**Models** [V]: PLANT `models/VISPA_crawling_rwa3.xml` (MuJoCo; nq=31,
structure + 3 RWA + welds) · CONTROLLER `models/VISPA_crawling_fixed.urdf`
(Pinocchio; nq=21). Both named in `gate/run_gate.py:45-46`,
`scripts/diag_cooperative_arms.py:51,488`. `models/VISPA_crawling.xml` and
`URDF_models/VISPA_crawling*.urdf` exist but **no code path loads them**. The
plant MJCF is mutated in place per run by `_mutate_mjcf(damping, armature,
anchor_dx, mass_ratio)` and restored under an md5 assert.

---

## 1. Effective data flow (what actually runs)

```
DS exit (once per step)
  CoarsePrePlanner  ── momentum-constrained IPOPT NLP (M=15, RK4 multiple
        │              shooting) → T_step + CoM traj; cruise-accel cap OFF;
        │              terminal margin κ=0.7
  ┌─────┼───────────────────────────┐
  ▼     ▼                           ▼
 TorsoPlanner       SwingPlanner     ContactScheduler
 (quintic pose,     (scheduler-      (anchor pairs;
  ω, α; L_com_ref)   driven quintic   DS/SS; T_step
        │            + bump + SLERP)  installed per-step)
        ▼  (10 Hz)       │
  Centroidal NMPC ──── 9-state [r_com, v_com, L_com], 12-control [f,τ]×2,
        │              N=8, dt=0.1, RK4, IPOPT/mumps
        │              momentum-conservation box ACTIVE (enforce_hw=True):
        │              c_simple − L_com − r_com×m·v_com ∈ [−5, +5] Nms
        │              rate cap |Ḣ_s,i| ≤ τ_w_max = 2.5 Nm
        │  r_com, v_com, a_com_ff, λ_ref, L_com_ref
        ▼  (100 Hz WBC)
  Whole-body QP ── TWO-TASK WEIGHTED STACK, no null-space projection anywhere:
        │            SS:  T-MOM linear (α=400) + torso-pose 6-D (α=2000)
        │                 + swing-EE 6-D (α=1000) + posture (α=20)
        │            all: wrench-track (α=1) + torque-min (α=5)
        │                 + accel-reg (α=1) + h_w slack (w=800)
        │            SS torso reference = RAW TorsoPlanner quintic+SLERP
        │                 (the CoM→torso δ-mapping is BYPASSED in SS;
        │                  DS still routes through δ(q_current)+F-SAT)
        │            DS: centroidal (CoM 3-D α=100 + torso-ori 3-D α=200
        │                 + posture α=50) with a passivity INEQUALITY
        │  τ_q (joints), λ (contact wrenches)
        ▼
  AOCS (legacy_pid_numerical, 100 Hz):
        τ_w = τ_ff − K_θ·θ_s − K_ω·ω_s − K_d·ω̇_s + K_h·(sat_±5(h_w) − h_w)
        K_θ=1.0, K_ω=50.0, K_d=25.0, K_h=0.5; clipped to ±2.5 Nm
        → STRUCTURE reaction wheels (the robot has NO attitude wheels)
        ▼
  MuJoCo: robot + floating structure (7110 kg, +3 RWA) + welds.
  Plant wheel actuators are ALSO capped at ±2.5 Nm by MJCF ctrlrange
  (`VISPA_crawling_rwa3.xml:324-326`) — the third, physical enforcement.
```

The robot has **no attitude wheels**; `rwa_x/y/z` are children of the
`structure` body (`VISPA_crawling_rwa3.xml:114-137`, inside the `structure`
body opened at `:80`, whose inertial is declared at `:82-83` —
`mass="7110" fullinertia="597 1493 1777 0 0 0"`). Robot torso attitude during
SS is held only through the single stance-arm weld. [V]

---

## 2. Per-layer reference (effective canonical values)

### 2.1 Coarse pre-planner — `crawlbot/planning/coarse_preplanner.py`
- Full RK4 multiple-shooting IPOPT NLP, **once per step**
  (`sim_loop._run_preplanner`). [A]
- Momentum box enforced at all collocation points; terminal tightened by
  `preplanner_kappa = 0.7`. [A]
- Cruise-accel constraint **OFF** (`preplanner_a_cruise_max = 0.0`,
  `config.py:215`). HANDOFF §M7's claim that `a_cruise_max=0.01` is an active
  fix is **stale**. [V]
- Outputs `T_step` + CoM trajectory. `f_max = 25 N`, `tau_max = 8 Nm`, M = 15. [V]

### 2.2 TorsoPlanner — `crawlbot/planning/torso_planner.py`
- Single quintic `s = 10τ³ − 15τ⁴ + 6τ⁵`; full 6-D pose + `L_com_ref`. [A]
- `torso_early_finish_fraction = 1.0` (`config.py:433`), no mid-waypoint. [V]
- **In SS its output is consumed RAW by the QP's 6-D torso-pose task** — see
  §2.5. `L_com_ref` still feeds the NMPC cost (`w_L_nmpc = 1.0`). [V]

### 2.3 SwingPlanner — `crawlbot/planning/swing_planner.py`
- Scheduler-driven (`reference_at`, `:230`): quintic + symmetric clearance
  bump + delayed-cosine SLERP, `rotation_delay_ratio = 0.2` (constructor
  default, `:84`). [V]
- `swing_early_finish_fraction = 1.0`, `swing_bump_peak_tau = 0.5`. [V]
- Reference query is **clamped to the SS window** at both the control and the
  logging site via `_swing_query_time()` (`sim_loop.py:2079`, used at
  `sim_loop.py:2508` and `tick_logging.py:404,476`) — this fixes a phantom
  820 mm `e_ee_pos` that came from the unclamped logging path. [V]

### 2.4 Centroidal NMPC — `crawlbot/solvers/centroidal_nmpc.py`, `nmpc_solver.py`
- State 9 `[r_com, v_com, L_com]`, control 12 `[f, τ]×2`, RK4, IPOPT/mumps. [A]
- **N = 8, dt = 0.1** — passed from `SimConfig` at `sim_loop.py:416-418`. The
  `CentroidalNMPCConfig` dataclass still defaults to `N=20, dt=0.05`
  (`centroidal_nmpc.py:84-85`); those defaults are **overridden and never
  used** on the canonical. The module docstring (`:11`) now states 8/0.1
  correctly. [V]
- **Momentum-conservation box ACTIVE**: `enforce_hw_conservation=True`
  (`run_m7_single_step.py:44`). Path constraint
  `c_simple − L_com − r_com×m·v_com ∈ [−h_max, h_max]`, `h_max = 5 Nms`. [V flag / A expr]
- Cost weights hoisted into `SimConfig` (CLEANUP-2 F5): `nmpc_Wr=100`,
  `nmpc_Wu_f=0.01`, `nmpc_Wu_tau=0.001`, `nmpc_Qf_r=1000`, `nmpc_Qf_v=100`,
  `nmpc_Qf_L=10` (`config.py:254-259`). [V]

### 2.5 Torso reference routing — `crawlbot/core/com_to_torso_mapping.py` + `sim_loop`
- **SS: the δ-mapping is BYPASSED.** The routing branch at
  `sim_loop.py:2378-2380` excludes SS when `ss_two_task_mode` is on
  (`... and not (cfg.ss_two_task_mode and phase == 'SS')`), so SS falls to
  the `else` at `sim_loop.py:2468-2471` and the QP receives `tr.p / tr.v /
  tr.a` — the raw TorsoPlanner quintic+SLERP. [V]
  *(CLAUDE.md still cites `sim_loop.py:2573-2576` for this; the anchor has
  drifted to `2378-2380` / `2468-2471`.)*
- **DS: the mapping is live.** World-frame `δ(q) = Σ_{i≠torso} m_i r_i(q)`
  (`compute_delta`), `r_b_ref = ratio·r_com_ref − δ/m_b`, δ recomputed **once
  per NMPC tick** (F-RATE, `sim_loop.py:2398-2406`), plus the **F-SAT**
  per-tick rate clamp at `(|v_b_ref| + fsat_jitter_margin)·dt_qp` with
  `fsat_jitter_margin = 0.05` (`sim_loop.py:2421-2435`) and the Option-A
  quintic DS blend over `ds_ramp_duration_s = 2.0` (`:2451-2464`). [V]
- The loop-free `compute_delta_local` variant and its flag were **removed**
  (CLEANUP-14). [V]

### 2.6 Whole-body QP — `crawlbot/solvers/wholebody_qp.py`
**There is no null-space projection anywhere in this file.** `weight_ratio =
1.0` (`:94`), so each task enters the cost at its face-value α and the
`priority=` arguments are nominal labels. The α magnitudes **are** the
hierarchy. [V]

- **SS two-task stack** (`:422-466`), gated by `ss_two_task_mode and not
  settle_mode`:
  1. T-MOM linear — `A_com` from `_com_task_rows` (`:888-912`), the **linear**
     CoM-Jacobian rows only, α = `ss_alpha_mom` = **400**, `Kp_com = Kd_com = 3`. [V]
  2. Torso-pose 6-D on `J_torso`, α = `alpha_torso_pose` = **2000**,
     `a_t_des = a_ff + Kp·e6 + Kd·(v_ref − v_act)` with
     `e6 = [p_ref − p ; log3(Rᵀ R_ref)]`, `Kp = 3·I₆`, `Kd = 2.5·I₆`. [V]
  3. Swing-EE 6-D on `J_ee`, α = `alpha_ee` = **1000**, same FF+PD form,
     `Kp_ee = 10`, `Kd_ee = 12`, `Kp_ee_ang = 6`, `Kd_ee_ang = 4.5`. [V]
  4. Posture, α = `alpha_posture` = **20**, `Kp_posture = 1.0`,
     `Kd_posture = 1.5` (`sim_loop.py:968` — **not** the dataclass 25/10). [V]
- **All phases**: wrench-tracking α = **1.0** (`:553`), internal-stress
  regularization `alpha_lambda_int = 1.0` (DS only — SS has no internal-stress
  null space), torque-min α = **5** and accel-reg α = **1**, both QP-construction
  literals at `sim_loop.py:951`; h_w slack penalty `w_hw_slack = 800` (`:159`). [V]
- **The inequality boxes are LIVE, not ∞** (`_add_inequality_constraints`,
  `:731-840`):
  - momentum safety `h_min ≤ h_w − dt·M_λ·λ ≤ h_max` with soft slacks (`:760-776`);
  - robot angular-momentum box `|L_com + dt·M_λ·λ| ≤ L_max`, **`L_max = 10 Nms`**
    piped from `cfg.L_max` (`sim_loop.py:969`) (`:779-791`);
  - **momentum-rate envelope box `|Ḣ_s| ≤ τ_w_max = 2.5 Nm`**, and because
    `qp_envelope_exact = True` it uses the **exact origin-referenced**
    `M_exact = compute_momentum_map(0, cc)` rather than the `M_λ` proxy
    (`:799-815`);
  - passivity `dq_jᵀτ_q + 2α·T_kin ≤ W_budget`, **DS only**, `α = 1.0`,
    `W_budget = 0` (strict) (`:826-838`). [V]
- **DS centroidal mode is ON** (`ds_centroidal_mode=True`): the joint-velocity
  damping cost is replaced by CoM 3-D (α = 100) + torso-angular 3-D (α = 200)
  at P1 plus posture (α = 50) (`:502-540`), with dissipation carried by the
  passivity inequality. [V]

### 2.7 AOCS — `crawlbot/aocs/force_estimator.py` + `sim_loop`
- **Active mode `legacy_pid_numerical`** (`compute_aocs_command_legacy_pid_numerical`,
  `force_estimator.py:514-596`):
  `pid_term = K_θ·θ_s + K_ω·ω_s + K_d·ω̇_s` (`:577`), ω̇_s by one-step finite
  difference of measured ω_s. `K_θ = 1.0`, `K_ω = 50.0`, `K_d = 25.0`,
  `K_h = 0.5`, `h_w` saturation at ±5 Nms, output clipped to
  `aocs_tau_w_max = 2.5`. [V]
- The feedforward carries the **orbital term** (`−L̇_com − r_com×m·a_com`);
  spec §5.8 / M4's claim that it is missing is **stale**. [V]
- `aocs_use_wrench_ff_in_ds = True`: in DS the FF is the contact-wrench couple
  `−Σ_i (r_Ci×f_i + τ_i)` from `λ_qp`, not the FD-on-`L_com` estimate. [V]
- `aocs_active_in_interstep = True`: the AOCS runs during the inter-step DS
  settle too (it used to be hardcoded to zero there). [V]
- Controls the **structure** wheels at 100 Hz. The H-estimator path exists but
  is off. [V]

### 2.8 Constant CoM-z standoff — `crawlbot/core/ik.py` + `sim_loop`
- `use_com_z_standoff = True`, `com_z_standoff = −0.35 m`: the fixed-rotation
  dock IK pins CoM-z and the initial config is re-solved at the standoff. Set
  by the runner, **not** by the dataclass default (which is `False`). [V]
- The startup IK also gets `ik_level_axis = [0,0,1]`, an `ik_q_nominal` arm
  posture and `ik_w_posture = 0.2`. [V]

### 2.9 Orchestration — `crawlbot/simulation/sim_loop.py` (+ `tick_logging.py`)
- Two-phase DS↔SS (no EXT). DS exit is energy-based (`T_kin < T_settle`);
  inter-step settle uses `interstep_settle_epsilon_v = 5e-3` on the canonical. [V]
- **Dock gate** (`config.py:35-58`): `d < weld_radius (5 mm)` **AND**
  `ori < dock_ori_threshold_deg (5°)` **AND** — since Fix C —
  `‖J_c·v⁻‖ < dock_twist_max (0.05)`, the full 6-D **weld-relative twist**
  (`dock_use_6d_twist = True`), not the legacy linear EE speed. Plus
  `swing_done`; else convergence hold; else `dock_timeout`. [V]
- `e_torso_ori` is logged in **degrees** (`np.degrees(angle_err)`,
  `tick_logging.py:208,395`). Older session diagnostics double-converted it
  and inflated readings 57×. [V]
- The per-tick recorders were split out of `sim_loop` into
  `crawlbot/simulation/tick_logging.py` (CLEANUP-32). [V]

---

## 3. Doc-vs-code discrepancies (current)

| Stale claim | Where it still appears | Code reality | Ref |
|---|---|---|---|
| cooperative split / strict-P1 / soft-CoM QP | this file's pre-2026-07 revision; HANDOFF | two-task **weighted** stack, no projection; `alpha_com_soft` field **removed** (CLEANUP-6) | [V] |
| AOCS mode `legacy_corrected` | `_make_m7_config():49`, `dca.main` signature `:242` | canonical kwarg forces **`legacy_pid_numerical`** | [V] |
| torso gains 6.0 / 5.0 | `config.py:351-352` | canonical **3.0 / 2.5** via `ss_kp_torso`/`ss_kd_torso` | [V] |
| NMPC N=20, dt=0.05 | `centroidal_nmpc.py:84-85` dataclass defaults | overridden to **8 / 0.1** at `sim_loop.py:418` | [V] |
| hard `L_max` / `τ_w` QP boxes are ∞ and dead | this file's previous revision | **both live**: `L_max = 10`, `τ_w_max = 2.5`, exact-Ḣ_s variant | [V] |
| `w_hw_slack = 1e4` | this file's previous revision | **800** (Add-5 freeze, `wholebody_qp.py:159`) | [V] |
| SS torso reference routed through δ-mapping | HANDOFF §M7, older CLAUDE.md text | **bypassed in SS**; DS only | [V] |
| `a_cruise_max = 0.01` active | HANDOFF §M7 | `0.0` (off) | [V] |
| AOCS orbital term missing | spec §5.8 | present & active | [V] |
| planning module `constrained_geodesic.py` | REPO_STATE (pre-refresh), older docs | **deleted** (CLEANUP-17) | [V] |
| dock gate is position+orientation only | older docs | third term: 6-D weld-relative twist `< 0.05` | [V] |
| `sim_loop.py:2573-2576` = SS torso-ref routing | CLAUDE.md | now `2378-2380` / `2468-2471` | [V] |

---

## 4. Disabled / dead paths (present but OFF on the canonical)

`mapping_bypass_in_ss` · H-estimator AOCS · the `legacy` / `legacy_corrected`
/ `legacy_pd_*` / `legacy_pid_model` / `nmpc_plan` AOCS modes ·
`preplanner_a_cruise_max` (0) · `preplanner_tstep_*` diagnostic knobs ·
`dock_hold_passivity_on` / `passivity_W_budget` (0) / `log_dock_work` ·
`diag_*_on_abort` (3 flags) · `t_settle_inter` (deprecated, unread) ·
`dt_ds = 0.5` ⇒ the centroidal **DWELL** never fires · `sequence_loader`
(`sim.setup(sequence_path=…)` — a documented user-facing path verified by
neither gate). [V]

**Removed outright** (do not look for a flag): soft-CoM residual task and
`alpha_com_soft` · the cooperative split and strict-P1 stacks · the Option-D
torso tube · `use_local_delta_mapping` / `compute_delta_local` ·
trajectory-aware IK, mid-waypoint reshape, `joint_space_fk` ·
`constrained_geodesic.py` · the manipulability-IK Option-B path
(`manipulability_config_trajectory`, `manipulability_config_mid_waypoint`,
`check_path_feasibility`, `precompute_torso_map` — CLEANUP-30; **consequence:
there is no interior path-feasibility guard at all**). [V]

---

## 5. Open issues

Not duplicated here — they drift. The live lists are:

- `CLAUDE.md` §"Known Issues" and §"Remaining Work" (repo + paper side);
- `results/j2_adjconv/CLEANUP_CARRYOVER.md` (chantier ledger, §A6 = the
  canonical run does not honour Rule 3, §A7 = `solve_ik_waypoints` has no
  callers);
- `results/j2_adjconv/PHASE_CLEANUP_OVERVIEW.md` (what the chantier did);
- `docs/architecture/PORT_AUDIT.md` / `PORT_SYNTHESIS.md` (portability —
  **semantic map only; every `file:line` in them predates the package
  restructuring and must be re-located before being quoted**).

---

## 6. How to use this file

Before changing a layer: confirm its row here against the source
(`file:line`), upgrade any `[A]` to `[V]` by reading the code, and update this
file. **If this file and a `file:line` disagree, the code wins and this file
is wrong — fix it.**

And before quoting any number: walk the §0 config chain. A `SimConfig` default
read in isolation has been wrong about the canonical three times.
