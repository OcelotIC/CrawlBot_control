# INTERNAL — α (J2 #2): CoM-mobile DS reference — implement (no modes) + characterize the conflict

**Implementation brief, then characterize — NO success threshold.** Raw numbers; the conflict's character,
the needed budget, and the ε/translation choice are decided by Idriss + reviewing Claude on these data →
this dimensions Piste A (step 3). Branch `j2/ds-active-rework` (pushed, never merged). Base `ae0673e`.

**Headline (raw, no verdict): the moving-CoM-under-strict-passivity conflict is PASSIVITY-DOMINATED.**
Across the whole magnitude sweep (CoM translation 0.02 → 0.20 m toward the next anchor) the **passivity
inequality binds (61–100 % of moving-DS ticks)**, the **envelope has massive headroom** (‖Ḣ_s‖∞ ≤ 0.82 vs
`τ_w,max=5`, never ≥ 90 % of cap), and the QP/NMPC **never go infeasible** (always docks, 0 timeouts). The
CoM reaches the target but **lags during the translation** (peak error ≈ the full offset; final error ≤
1 cm), the lag growing with magnitude — passivity rate-limits the tracking. ⇒ **Piste A (a passivity
work-budget) is the indicated lever; the envelope headroom confirms the conflict is NOT envelope-limited.**

---

## Part 0 — Verification stop-gate: **WIRING** (proceed), with a routing nuance

**(1) Is the DS-centroidal CoM-3D task fed a per-tick moving CoM target?** **YES — WIRING.** The chain
(verified in code, `sim_loop.py`):
```
com_reference_at(t_horizon) → cref_r  (settle_mode branch, :2460)
   → NMPC.solve(r_com_ref=cref_r) → rp  (:2524)
   → qp.solve(r_com_ref=rp_interp)  (:2841) → DS CoM-3D task (wholebody_qp.py:1068)
```
Exactly analogous to the per-tick `R_torso_ref` the torso-ori task takes. **Nuance (the wiring-vs-building
point the TorsoPlanner audit did not nail):** this per-tick path lives in the **DWELL** (`_step`,
`passivity_override=True`), gated `_dwell_target > 1.0`. With the canonical `dt_ds=0.5` the DWELL **never
fires** — the inter-step DS is `_run_ds_passivity_loop` (NMPC bypassed, CoM held). So the path is **wiring,
but dormant**; routing the inter-step DS through it is a **config change** (lengthen `dt_ds`), **not new
plumbing**. (Driving a moving CoM inside `_run_ds_passivity_loop` itself *would* be building — not done.)

**(2) Is the translation target available at the DS hook?** **YES.** The gait loop peeks at the next SS
phase `ss_gp = phases[i+1]` (`sim_loop.py:1836`) → `ss_gp.swing_arm`, `ss_gp.swing_to_idx`; the anchor
position is `self.sched.anchors_{a,b}[target_idx]` (structure frame). No new plumbing.

## Part 1 — Implementation (no mode flag) — committed `e402dcd`

`set_hold` stays the **degenerate constant** case; a DS window is one where the **CoM** content is
non-constant. No mode flag, no hold↔mobile switch.
- **`config.py`:** `dt_ds` (DS phase duration; default 0.5 ⇒ DWELL dormant / unchanged) + `ds_mobile_com_magnitude`
  (CoM translation [m] toward the next anchor; default 0.0 ⇒ hold). Both opt-in ⇒ default byte-identical.
- **`sim_loop.py`:** scheduler uses `cfg.dt_ds`; at the **DWELL hook**, when `magnitude>0`,
  `set_from_waypoints` translates the CoM from the docked value toward the next anchor (orientation held —
  `R0` at both waypoints; constant `δ_com` ⇒ `r_com(t)=p(t)+R·δ` translates), via the **existing** quintic
  trajectory tooling; `clear_phases` after. Orientation: held. Posture: `q_nominal` (the existing P3 posture
  task, unchanged). NMPC tracks the moving `com_reference_at`/`l_com_reference_at` and plans `lambda_ref`
  under its envelope, as usual.
- **Instrumentation (`ds_mobile_trace`, gated):** per DWELL tick — `pass_resid = dqⱼᵀτ_q + 2α·T_kin` (→0 ⇒
  passivity binding), `Hdot_inf = ‖Σ r_Cj×f_j + τ_j‖∞` from the QP wrench (→ `τ_w,max` ⇒ envelope binding),
  `com_err`, swing-arm `manip = √det(J_arm J_armᵀ)`, `qp_ok`, `nmpc_status`.
- **diag CLI:** `--ds-mobile-com-magnitude`, `--dt-ds`. Tooling: `Misc/scripts/run_alpha_sweep.sh`,
  `Misc/scripts/audit_alpha.py`.

**Strict passivity only** (`passivity_active=True`, RHS ≤ 0 — no Piste A budget, no envelope-box change, no
FLAG-2 fix). Orientation + posture held (arm-posture-as-moving-DOF is BUILDING, out of scope).

**Regression (`pytest tests/`):** **220 passed, 1 failed.** The single failure is the **pre-existing FK
test** `test_E7_t15_step2_dock_under_fk_mode` (verified identical on clean `ae0673e` in the J2 #1 work; an
FK-mode preplanner/timeout issue, unrelated to α). **No new failures.** The α paths are
default-off (`ds_mobile_com_magnitude=0`, `dt_ds=0.5`) and the only always-on edit is
`dt_ds=self.cfg.dt_ds` (= the prior literal 0.5) ⇒ default behaviour byte-identical; the remaining
integration tests exercise that dormant default. (C6 OFF in every sweep run above is BIT-IDENTICAL to the
Fix-A baseline, independently confirming the flag-OFF determinism.)

## Part 2 — Run under STRICT passivity + characterize (measure, do NOT judge)

Sweep = the **initial-DS DWELL** (n-steps 2, `dt_ds=2.5` to trigger the DWELL) translating the CoM toward
step-0's anchor. The trace also captures the trailing-DS settle ("hold" segment, no translation) as a
within-run baseline. Metrics are the **MOVING** segment (the moving-CoM-under-passivity conflict).

### 2.1 / 2.2  Magnitude sweep — CoM tracking + which constraint binds  (strict passivity, `dt_ds=2.5`)

| `magnitude` [m] | CoM err max / final [m] | passivity p_frac (→bind) | envelope ‖Ḣ_s‖∞ (cap 5) | feasibility | docks/timeouts | **binds** |
|---|---|---|---|---|---|---|
| 0.02 | 0.0194 / 0.0012 | **1.00** | 0.103 (2 %) | qpf 0, nin 0 | 2 / 0 | **passivity** |
| 0.05 | 0.0484 / 0.0029 | **0.79** | 0.252 (5 %) | qpf 0, nin 0 | 2 / 0 | **passivity** |
| 0.10 | 0.0968 / 0.0058 | **0.62** | 0.501 (10 %) | qpf 0, nin 0 | 2 / 0 | **passivity** |
| 0.20 | 0.1943 / 0.0099 | **0.61** | 0.816 (16 %) | qpf 0, nin 0 | 2 / 0 | **passivity** |
| (hold baseline, trailing-DS) | 0.0034 / 0.0016 | 0.97–0.98 | ≤ 0.52 | 0, 0 | — | passivity (settle) |

- **Which binds first: PASSIVITY, at every magnitude.** Envelope never reaches 90 % of cap (max 16 % at
  0.20 m); QP/NMPC never infeasible; the DS never times out and both arms always dock — even at a 20 cm
  translation. So the conflict is **passivity-dominated**, not envelope-dominated and not a feasibility wall.
- **CoM tracking:** peak error ≈ the full commanded offset (the CoM lags the moving reference during the
  ramp; the reference is also queried at the NMPC horizon `t+0.8 s`, so part of the peak is horizon lead).
  **Final** error ≤ 1 cm even at 20 cm — the CoM *reaches* the target but *slowly*. Final error **grows**
  with magnitude (0.0012 → 0.0099), i.e. larger translations under-track more within the DWELL.
- **Passivity binding falls as magnitude rises** (1.00 → 0.61): the larger demanded motion makes the QP
  dissipate harder (slack more often) but `dqⱼᵀτ_q ≤ −2α T_kin ≤ 0` **always holds** — under strict
  passivity the joints never inject net positive work, so the translation is realized only by a
  passivity-compatible (energy-non-increasing) reconfiguration, which is what limits the rate. **This is the
  exact lever Piste A would relax** (allow `≤ W_budget > 0` so positive work can drive the CoM faster).

### 2.x  Speed axis (`magnitude=0.05`, `dt_ds` = DWELL duration)

| `dt_ds` [s] | CoM err max / final | passivity p_frac | envelope ‖Ḣ_s‖∞ | feasibility | binds |
|---|---|---|---|---|---|
| 2.5 (slower) | 0.0484 / 0.0029 | 0.79 | 0.252 | 0,0 | passivity |
| 1.5 (faster) | 0.0496 / 0.0015 | 0.91 | 0.360 | 0,0 | passivity |

Faster translation ⇒ passivity binds **more** (0.79 → 0.91) and envelope usage rises (0.25 → 0.36) but stays
≪ cap. Still passivity-dominated, still feasible. (Speed and magnitude both load passivity first.)

### 2.3  C1–C5 (full 5-step traversal, raw — no pass/fail verdict by me)

Hold baseline (`m=0`, `dt_ds=2.5`) vs moving (`m=0.10`, `dt_ds=2.5`), both vs the dcda974 gate baseline:

| criterion | hold `m=0` | moving `m=0.10` | note |
|---|---|---|---|
| C1 docking | **PASS** d=[4.94,4.51,4.79,4.64,4.85] | **FAIL** d=[4.90,4.44,4.87,4.50,**4.99**] | 5/5 dock, **0 timeouts** both; FAIL is a *margin* trip |
| C2 torso-track | PASS (ori_rms 0.068, pos_pk 14.1mm) | PASS (ori_rms 0.070, pos_pk 13.6mm) | unchanged |
| C3 envelope ‖Ḣ_s‖∞ SS | PASS 5.00 (per-axis 3.12/4.13/5.0) | PASS 5.00 (3.40/4.08/5.0) | z-axis binds in **SS** (AOCS), unrelated to the DS-CoM move |
| C4 attitude θ_s | PASS peak 0.50 / final 0.10 | PASS peak 0.45 / final 0.07 | unchanged / slightly better |
| C5 h_w∞ | PASS 4.133 (≤4.5) | PASS 4.175 (≤4.5) | +0.04 Nms with the move |
| C6 OFF determinism | **BIT-IDENTICAL** | **BIT-IDENTICAL** | flag-OFF byte-identical to Fix-A baseline both runs |

**The only C-criterion the moving CoM trips is C1, and it is a marginal trade, not a docking failure.** C1's
verdict = (all 5 dock ≤5 mm) **and** (worst dock margin ≥ baseline's worst). Both hold and moving dock all 5
with 0 timeouts; the moving run's **worst dock margin is 0.01 mm** (one dock at 4.99 mm) vs the baseline's
0.06 mm (4.94 mm) — a **~0.05 mm** degradation of the tightest dock, which trips the "no-worse-than-baseline"
check. So the moving-CoM DS costs ~0.05 mm of dock margin on the tightest step (and +0.04 Nms of `h_w`),
while C2/C3/C4/C6 are unchanged. (C3's SS z-axis = 5.0 is the AOCS envelope during *swing*, not the DS-CoM
move — the DS envelope usage is ≤0.82, §2.1.)

### 2.4  Byproduct — arm-Jacobian conditioning before/after the translation ("rapprocher")

Swing-arm manipulability `√det(J_arm J_armᵀ)` within the moving DS segment (first → last tick):

| magnitude | manip before → after | Δ |
|---|---|---|
| 0.02 | 0.2541 → 0.2544 | +0.0003 |
| 0.05 | 0.2541 → 0.2549 | +0.0008 |
| 0.10 | 0.2541 → 0.2556 | +0.0015 |
| 0.20 | 0.2541 → 0.2571 | +0.0030 |

The "rapprocher" **slightly improves** the next swing arm's manipulability, scaling with magnitude (+1.2 %
at 20 cm). The physical objective is served in the right direction but the effect is **small** at these
magnitudes. (Comparing across the *whole* trace would wrongly show a "drop" — that is the initial-DS arm vs
the trailing-DS arm, two different configs; the per-segment first→last above is the correct before/after.)

---

## Flags / divergences vs the J2 / TorsoPlanner / envelope audit facts

1. **DWELL-routing nuance (new):** the per-tick moving-CoM path is the DWELL (`_step`), dormant at
   `dt_ds=0.5` (canonical inter-step DS = `_run_ds_passivity_loop`, NMPC bypassed). Exercising the moving CoM
   requires `dt_ds` large (config). The TorsoPlanner audit said "moving CoM = WIRING (com_reference_at →
   NMPC → QP)"; this refines it: wiring exists **but is dormant** until the DS is routed through the DWELL.
2. **Envelope audit (Piste A) corroborated:** the envelope has huge headroom in DS (‖Ḣ_s‖∞ ≤ 0.82 ≪ 5), so
   the moving-CoM conflict is **passivity-limited, not envelope-limited** — Piste A (passivity budget), not a
   translation-magnitude cap, is the indicated fix. Consistent with the envelope audit's "the conflict is
   passivity-dominated → Piste A" framing.
3. **Strict passivity confirmed binding (J1/Bloc-1 passivity facts):** `dqⱼᵀτ_q + 2α T_kin ≤ 0` is active
   61–100 % of moving-DS ticks — the moving reference genuinely fights it (the conflict the audits predicted
   is real and now quantified).

## Reproduce
```
bash Misc/scripts/run_alpha_sweep.sh    # magnitude {0,0.02,0.05,0.10,0.20} + dt_ds{1.5} (n=2) + C1-C5 (n=5)
MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/audit_alpha.py LABEL=results/<dir> ...
```
Supporting: `Misc/runs/j2_alpha/sweep_binding.log`, `gate_C1-C5.log`. Raw per-run sim dirs reproducible from
the script, not committed (bulk).

**STOP after the report.** No success threshold applied; the conflict character (passivity vs envelope), the
needed budget, and the ε/translation choice are yours → this dimensions Piste A (step 3). No merge, no PR.
