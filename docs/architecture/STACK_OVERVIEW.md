# Control Stack Overview (CODE-GROUND TRUTH)

**Date:** 2026-05-27
**Status:** Code-verified reference. **This document supersedes the stale
intent in `brainstorming_reworked_architecture.md` and
`CLAUDE_CODE_HANDOFF.md` wherever they disagree.** Those docs describe
*intended* architecture and an outdated milestone state; this file
describes what the code in the canonical run **actually does**, grounded
in `file:line`.

**Canonical run:** `scripts/diag_cooperative_arms.py` →
`_make_m7_config()` (`scripts/run_m7_single_step.py`) + script overrides.
"Effective value" below = after that override chain, **not** the
`SimConfig` dataclass default (defaults are misleading — verify the
chain).

**Verification legend:** [V] verified directly from source this session ·
[A] reported by a code-reading agent, spot-checked · [?] not yet
independently verified — treat as provisional.

---

## 1. Effective data flow (what actually runs)

```
DS exit (once per step)
  CoarsePrePlanner  ── full momentum-constrained IPOPT NLP → T_step + CoM traj
        │                (cruise-accel cap OFF; terminal margin κ=0.7)
  ┌─────┼───────────────────────────┐
  ▼     ▼                           ▼
 TorsoPlanner       SwingPlanner     ContactScheduler
 (quintic R_ref,    (scheduler-      (anchor pairs;
  ω,α; L_com_ref;    driven quintic   DS/SS; T_step
  position folded    + bump + SLERP)  installed per-step)
  via mapping)
        │                │
        ▼  (10 Hz)       │
  Centroidal NMPC ──── 9-state[r_com,v_com,L_com], 12-ctrl[f,τ×2], N=8 dt=0.1 RK4
        │                  momentum-conservation box ACTIVE (enforce_hw=True)
  r_com,v_com,a_ff,λ_ref ; L_com_ref tracked
        ▼
  CoM→Torso Mapping ── world-frame δ(q_current): r_b_ref=(m_t/m_b)r_com−δ/m_b
        │                + F-SAT per-tick rate clamp (band-aid)
        ▼  (~100 Hz WBC)
  Cooperative WB-QP ── P1 torso-ANGULAR(strict)+hw-slack ;
        │              P2 torso-LINEAR + EE-6D (co-equal, via N_torso_ang) ;
        │              P3 posture ; soft-CoM OFF ; passivity DS-only
  τ_q (joints)
        │              AOCS (legacy_corrected): τ_w=−L̇_com−r_com×m·v̇_com+K·clip(hw)
        ▼                  → STRUCTURE reaction wheels (NOT on the robot)
  MuJoCo: robot + floating ASTROHUB structure (+ its RWA) + welds
```

The robot has **no attitude wheels**; the RWA are children of the
`structure` body in `models/VISPA_crawling_rwa3.xml` (rwa_x/y/z ~L114–134,
inside the `structure` body opened ~L80). Robot torso attitude during SS
is held only through the **single stance-arm weld**. [V]

---

## 2. Per-layer reference (effective canonical config)

### 2.1 Coarse pre-planner — `crawlbot/planning/coarse_preplanner.py`
- Full RK4 multiple-shooting IPOPT NLP, **once per step** (`sim_loop._run_preplanner`). [A]
- Momentum box enforced at all collocation pts; terminal tightened by
  `preplanner_kappa=0.7`. [A]
- Cruise-accel constraint **OFF** (`preplanner_a_cruise_max=0.0`). [A]
  - NB: HANDOFF §M7 claims `a_cruise_max=0.01` is an active fix — **stale**. [V]
- Outputs `T_step` + CoM trajectory; `f_max=25 N`, `tau_max=8 Nm`, M=15. [A]

### 2.2 TorsoPlanner — `crawlbot/planning/torso_planner.py`
- Single quintic `s=10τ³−15τ⁴+6τ⁵`; produces full 6D pose + `L_com_ref`. [A]
- `early_finish_fraction=1.0`, no mid-waypoint (single-quintic). [A]
- Orientation → QP P1; `L_com_ref` → NMPC cost; position folded via mapping. [A]

### 2.3 SwingPlanner — `crawlbot/planning/swing_planner.py`
- Scheduler-driven (`reference_at`, ~L515) quintic + symmetric clearance
  bump + delayed-cosine SLERP (`rotation_delay_ratio=0.2`, hardcoded). [A]
- FK-mode and mid-waypoint paths exist but **OFF** by default. [A]
- **Reference query is clamped to the SS window** at both control and
  logging sites via `_swing_query_time()` (sim_loop) — fixes a phantom
  820mm `e_ee_pos` from the unclamped logging path. [V]

### 2.4 Centroidal NMPC — `crawlbot/solvers/centroidal_nmpc.py`, `nmpc_solver.py`
- State 9 `[r_com,v_com,L_com]`, control 12 `[f,τ]×2`, N=8, dt=0.1, RK4,
  IPOPT/mumps. [A]
- **Momentum-conservation box ACTIVE**: `enforce_hw_conservation=True`
  (`_make_m7_config:49`). Path constraint
  `c_simple−L_com−r_com×m·v_com ∈ [−h_max,h_max]`, `h_max=5 Nms`. [V flag / A expr]
- Tracks `L_com_ref` from TorsoPlanner (`w_L_nmpc=1.0`). [V flag]
- Docstring claims N=20/dt=0.05 — **stale**. [A]

### 2.5 CoM→Torso mapping — `crawlbot/core/com_to_torso_mapping.py`
- **Active formula:** world-frame `δ(q)=Σ_{i≠torso} m_i r_i(q)` (`compute_delta`,
  `oMi.act(lever)`), `r_b_ref=(m_t/m_b)r_com−δ/m_b`, fed `q_current`. [V]
- `use_local_delta_mapping=False` → the **loop-free base-relative**
  reformulation (`compute_delta_local`) is present but **OFF**. [V]
- **F-SAT clamp** on per-tick `r_b_ref` increment, threshold
  `(|v_b_ref|+fsat_jitter_margin)·dt_qp`, `fsat_jitter_margin=0.05`
  (sim_loop ~L2237-2257). Band-aid for δ(q_current) jitter. [V]
- ⚠ **Doc-vs-code:** HANDOFF §M7 + CLAUDE.md say "planned-δ `δ(q_planned)`
  active" — **the code uses `δ(q_current)`** (reverted in commit 64479ab).
  F-SAT is the undocumented patch that followed. [V]

### 2.6 Whole-body QP — `crawlbot/solvers/wholebody_qp.py` (cooperative mode)
- P1 (strict): torso **angular** 3D, `α=ss_alpha_torso_ang=500`
  (~L645). `N_torso` = null-space of **angular-only** torso (~L728). [A]
- P2 (co-equal, via `N_torso_ang`): torso **linear** 3D (`α=ss_alpha_torso_lin=500`,
  ~L809) + EE 6D (`α=ss_alpha_ee=3000`, ~L788). 6:1 EE-over-linear. [A/V]
- P3 posture, projected through combined P1+P2 (`rcond=1e-4`). [A]
- **Soft-CoM residual OFF** (`alpha_com_soft=0`, ~L826) → QP has **no
  direct CoM/momentum feedback**; trusts the mapping. [V]
  - ⚠ **Two confounds before re-engaging it (see §5):** (a) the only
    sweep on record (`Misc/runs/M5_alpha_sweep/`) ran
    `cooperative_arms_mode=False` — the **wrong stack** — so its
    "every non-zero α diverges" is not evidence against soft-CoM here;
    (b) the projection basis `null(A_torso)∩null(A_ee)` (~L842)
    collapses to **angular-only ∩ EE** in cooperative mode
    (`A_torso`=angular, ~L655), so it **no longer excludes torso-linear**
    (a co-equal P2 task). [V]
- hw box present only as **soft slack** (`w_hw_slack=1e4`); hard
  `L_max`/`τ_w` boxes are ∞ (dead). [A]
- Passivity active **DS only**; `tau_max=20 Nm` effective. [A]

### 2.7 AOCS — `crawlbot/aocs/force_estimator.py` + `sim_loop`
- **Active mode `legacy_corrected`** (`_make_m7_config:54-55`):
  `τ_w=−L̇_com−r_com×m·v̇_com+K_hw·clip(hw)` — **orbital term ON**;
  `K_hw=2`, `hw_max=±5 Nms`. [V flag / A expr]
- H-estimator path present but **OFF**. Controls **structure** wheels at
  ~100 Hz. [A]
- ⚠ **Doc-vs-code:** spec §5.8/M4 says the orbital term was *missing*;
  it is **present and active** in the canonical run. [V]

### 2.8 Constant CoM-z standoff — `ik.py` + `sim_loop` (our PR)
- `use_com_z_standoff=True`, `com_z_standoff=−0.35`: dock IK pins CoM-z
  (`dock_configuration_fixed_rotation(com_z_target=...)`) and the initial
  config is re-solved at the standoff. [V]

### 2.9 Orchestration — `crawlbot/simulation/sim_loop.py`
- Two-phase DS↔SS (no EXT). DS exit energy-based (T<T_settle). [A]
- SS dock gate: `d<5mm AND ori<5°` AND `swing_done`; else convergence
  hold; else `dock_timeout`. [A]
- `e_torso_ori` is logged in **degrees** (`np.degrees(angle_err)`,
  ~L2591) — earlier session diagnostics double-converted it, inflating
  readings 57×. [V]

---

## 3. Doc-vs-code discrepancies (stale-doc claims to ignore)

| Stale claim (docs) | Code reality | Ref |
|---|---|---|
| planned-δ mapping active (HANDOFF §M7, CLAUDE.md) | world-frame `δ(q_current)` + F-SAT | [V] |
| `a_cruise_max=0.01` active (HANDOFF §M7) | `preplanner_a_cruise_max=0.0` (off) | [V] |
| AOCS orbital term missing (spec §5.8) | present & active (`legacy_corrected`) | [V] |
| "torso-ori 45° is the last blocker" (HANDOFF/CLAUDE.md) | docks 0–3 at e_torso_ori ~1–3°; 45° was partly a units artifact | [V] |
| NMPC N=20, dt=0.05 (docstring) | N=8, dt=0.1 | [A] |
| F-SAT, CoM-z standoff | not in any doc (code-only) | [V] |

---

## 4. Disabled / dead paths (present but OFF in canonical run)

soft-CoM residual (`α=0`) · loop-free mapping (`use_local_delta_mapping=False`) ·
mid-waypoint reshape · trajectory-aware IK · torso-linear soft tube
(`r_tube=0`) · mapping bypass · H-estimator AOCS · hard QP momentum/τ_w
boxes (∞). [V/A]

---

## 5. Open issues, mapped to the real stack

- **Angular/CoM-z drift (loop-free experiment):** stance-arm motion
  degrades P1 torso-angular → cascades to P2-EE via `N_torso_ang`. This is
  the spec's own §3 *constrained dynamic singularity* prediction; the §6
  mitigation (condition-number monitor + damped null-space) was **never
  implemented**. Robot-side WB-QP issue (no robot wheels). [V reasoning]
- **Step 4 dock timeout: RESOLVED (commit 7ac9124).** Was 3 bugs, not
  control: (a) `_cache_site_ids` cached only anchors 1–5 → `_gripper_distance`
  returned inf for anchor 6 → dock gate never fired; (b) stale `tau=zeros(12)`
  (6-DOF) crashed the terminal DS settle on `ctrl[:14]`; (c) `_coop_A_lin`
  UnboundLocalError in `settle_mode`. All 5 steps now dock end-to-end.
- **Mapping debt:** world-frame `δ(q_current)` + F-SAT is the live combo;
  loop-free `δ_local` removes the feedback but exposed the angular drift
  above (committed OFF).
- **Soft-CoM re-engagement — blocked on attribution, not a knob-sweep.**
  The campaign's 10.8× QP↔NMPC wrench mismatch (`CAMPAIGN_…md` §6/§7) is
  attributed to soft-CoM-off, but that is a **hypothesis** (peak
  `lambda_qp` not yet decomposed). Two reasons a sweep is premature: (1)
  soft-CoM enforces `a_com_des = a_com_ff(NMPC)+PD` (~L578), but the
  point-mass NMPC has **no arm-momentum term** → the reference is
  structurally wrong; soft-CoM (feedback) and a CMM-feedforward into the
  NMPC (reference fix) are duals, and **feedback against a wrong
  reference cannot win**. (2) Candidate mechanisms not yet ruled out:
  arm-momentum (→ CMM-feedforward), **internal/squeeze force** in the
  null space of the 12→6 centroidal map (soft-CoM constitutively cannot
  touch it), or **F-SAT/δ(q_current) mapping debt** (49.6% tick clip).
  The two prior-sweep confounds (§2.6) further mean no clean evidence
  exists yet. **Next: attribution experiments, not α-tuning.** The
  projection-basis question in §2.6 is a **hierarchy redesign**, not a
  harness fix — do not merge it as cleanup.

---

## 6. How to use this file

Before changing a layer: confirm its row here against the source
(`file:line`), upgrade any `[?]`/`[A]` to `[V]` by reading the code, and
update this file. **If this file and a `file:line` disagree, the code
wins and this file is wrong — fix it.**
