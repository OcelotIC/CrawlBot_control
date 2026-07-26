# M7 Technical Log — Complete History and Current State

**Date:** 2026-04-17
**Branch:** `claude/m7-core-task-replace-3Hmf4`
**Last commit:** `68d3b18` (v19)
**Tests:** 191/191 passing

---

## 0. Correction Notice (2026-04-17)

This log was written against global-max metrics (a single `np.max(np.abs(·))` over the full simulation window) and, in consequence, drew conclusions that do not survive the per-phase re-diagnostic carried out on 2026-04-17. The refactored diagnostic suite (`crawlbot/diagnostics/metrics.py`, Step 1) splits every peak metric into an SS window (useful locomotion), a DS window (hold/recovery), and a `_global` value flagged `WARN` when the run aborted. Re-running the existing archived `sim_log.json` files for v17, v19, v20, v21 through this refactored suite produced `Misc/runs/archive_rediagnostic.md`. Sections making claims that the per-phase data contradicts have been amended in place; sections whose diagnosis remains correct (notably §2) are preserved unchanged. Amended paragraphs are tagged `(amended 2026-04-17)`.

Corrected picture: SS torso orientation tracking is 0.53 ± 0.01° across v17–v21 — below the 0.72° standalone floor and well inside the 5° dock threshold — and was never the blocker; the remaining real problem in SS is EE position tracking (24 mm standalone vs. 153–165 mm closed-loop peak, 25–41 mm closest approach at dock_timeout, ~6.7× inflation); a second, orthogonal issue is a post-abort DS divergence triggered by the gait scheduler entering a pre-planned trailing-DS phase regardless of whether the preceding SS docked. (amended 2026-04-17)

---

## 1. What Works (Validated)

### QP Standalone — PROVEN CORRECT
- **Result:** 15.6mm EE position, 3.9° EE orientation, 24.7mm torso position
- **Config:** α_wrench=0.01, α_com_soft=0, 800mm/45°/7.3s swing, 7-DOF arms
- **Torque used:** 0.87 Nm of 20 Nm budget (4.4%)
- **Conclusion:** The QP task stack (torso P1, EE null-space P2, posture P3) works correctly in isolation

### B_const_torso — NMPC + static torso reference
- **Result:** 13.5mm EE, 21mm torso
- **Config:** NMPC active (λ_ref + a_com_ff flowing to QP), mapping bypassed, torso held at initial pose
- **Conclusion:** NMPC momentum management works through λ constraint alone, without needing the mapping to move the torso

### Individual Components — All Sound
- **NMPC conservation law (M3):** 0% infeasibility standalone, B2 formulation correct
- **AOCS (M4):** Desaturation sign fixed, orbital term added. Platform rotation at 1%: 3-5° when controller tracks well
- **7-DOF arms:** κ(N_torso·J_ee) < 7 throughout SS (was 10⁵ with folded 6-DOF arms)
- **Manipulability-optimized init:** Stance arm reach 750mm (55% of neutral), singularity eliminated
- **Coarse pre-planner (M6):** Solves in ~60ms, produces momentum-feasible T_step
- **Two-phase state machine (M7):** DS (energy-based exit) → SS (synchronized trajectories), EXT eliminated
- **QP hw safety with soft slack:** 0 QP failures even when hw exceeds bounds
- **Dock gate:** d < 5mm AND ori < 5° (both required)

---

## 2. The Core Problem — Dynamic Coupling During SS

When the torso and swing arm move simultaneously, their motions couple through the kinematic chain and compete for actuator budget. The standalone QP tracks to 15.6mm; the closed loop produces 137-193mm on the same geometry. The inflation has two identified sources:

### Source 1: CoM→Torso Mapping (A→B: 16mm → 57mm, 3.6×)

The mapping computes `r_b_ref = (m_total/m_b) · r_com_ref - δ(q)/m_b`.

**Problem with live δ(q_current):** The arm's tracking error creates fluctuations in δ(q) that get amplified (×1.78) into torso reference noise. The EE tracks imperfectly → δ changes → torso reference jitters → QP fights jitter → EE tracking degrades. A feedback loop.

**Problem without mapping (TorsoPlanner direct, v16-v18):** Torso tracking degrades from 22mm to 121mm because the geometric quintic doesn't know the arm is pushing on the base. The torso PD carries the full burden without feedforward compensation for arm-induced CoM drift.

**Current best (v19): Mapping with planned δ(q_planned).** Uses swing planner's planned arm configuration instead of actual. Produces smooth feedforward torso reference that anticipates arm motion. Torso tracks to 42mm (vs 22mm with live δ, 121mm with no mapping).

### Source 2: Null-Space Projection Drift (C→D: 56mm → 193mm, 3.4×)

The null-space projector `N_torso` changes as the torso moves. At acceleration level, EE and torso commands are orthogonal. At velocity/position level, the EE drifts because the null space rotates. Drift accumulates proportionally to torso velocity.

**Mitigation applied (v17):** Task-consistent feedforward `a_ee_ff += J_ee · J_torso† · a_torso_des`. Gave 11mm improvement (36→25mm closest approach). Correct direction but insufficient alone.

---

## 3. Version History — What Was Tried

### Weight/Configuration Fixes (all retained)
| Fix | Effect | Status |
|---|---|---|
| weight_ratio = 1.0 (was 1000) | Tasks below P1 were invisible (EE effective weight 3 vs torso 500) | **KEPT** |
| α_wrench = 0.01 (was 100) | Wrench reg was consuming 20% of QP budget, blocking torso/EE authority | **KEPT** |
| α_com_soft = 0 (was 5) | Redundant with mapping; at P4 was invisible anyway | **KEPT** |
| AOCS desaturation sign flip | Was amplifying disturbances 3.7× instead of rejecting | **KEPT** |
| 7-DOF arms | 6-DOF has zero EE orientation margin → can never dock at 5° | **KEPT** |
| Manipulability-optimized init | Prevents folded-arm internal singularity (κ: 10⁵ → 7) | **KEPT** |
| Stage-1 settling skip | With good init, no weld snap impulse to absorb | **KEPT** |
| δ̇ correction in mapping | Position-only δ/m_b was asymmetric with velocity feedforward | **KEPT** |
| EE task-consistent FF | `a_ee_ff += J_ee · J_torso† · a_torso_des` compensates first-order coupling | **KEPT** |
| Planned δ(q_planned) in mapping | Feedforward instead of feedback; eliminates jitter from arm tracking errors | **KEPT** (v19) |
| Quintic TorsoPlanner | Continuous a_ff everywhere (trapezoidal had zero-FF cruise plateau) | **KEPT** |

### Trajectory Shaping (all reverted — were compensating for the folded-arm singularity)
| Attempt | Effect | Status |
|---|---|---|
| Torso stagger (finish at 0.6·T_step) | −21% EE error but compressed torso motion → higher peak κ | **REVERTED** — singularity is gone with proper init |
| Early bump (peak at τ=0.25) | −40% EE error but peak κ was 4× worse | **REVERTED** — was treating symptom |
| CoM displacement scale ×0.5 | EE: 137→70mm (diagnostic only) | **REVERTED** — not a fix |
| Torso freeze at d<50mm | Reference discontinuity → energy injection | **NEVER APPLIED** |

### Approaches That Failed
| Attempt | Why It Failed |
|---|---|
| Raising angular Kp (3.6→6) | Only 18% improvement; disturbance-dominated, not gain-limited |
| Passivity during hold window | Prevents positive work needed to close the dock gap |
| Bypass mapping in hold → use p_t1 | 591mm reference discontinuity → catastrophic |
| Gain scheduling on EE approach | Rejected on principle — synchronized trajectories should eliminate the need |
| TorsoPlanner-only (no mapping, v16-v18) | Torso tracking 6× worse (121mm vs 22mm) — loses arm-compensation |
| Trapezoidal → quintic swap alone (v18) | Higher peak velocity → more disturbance, negated the FF benefit |

---

## 4. Current State (v19)

### Configuration
```
α_torso = 500, α_ee = 3000, α_posture = 20
α_wrench = 0.01, α_com_soft = 0
weight_ratio = 1.0 (face-value weights + null-space projection)
Kp_torso = 6, Kd_torso = 5 (uniform, no 0.6× angular scaling)
Kp_ee = 10, Kd_ee = 12, Kp_ee_ang = 6, Kd_ee_ang = 4.5
7-DOF arms, manipulability-optimized init, stage-1 skip
TorsoPlanner quintic, SwingPlanner quintic + clearance bump (peak τ=0.5)
Mapping with planned δ(q_planned) for torso ref during SS
EE task-consistent feedforward active
Two-phase state machine (DS energy-based / SS synchronized)
```

### v19 SS Metrics (1%, single step)
```
torso_pos_peak:     42.0 mm    (v12: 22, v16: 121, target: 10)  [global max]
torso_ori_peak:     31.1°      (v12: 29, target: 5)             [global max — post-abort DS]
  └─ torso_ori_peak_SS (2026-04-17 re-diag):  0.5432°           [SS-only, actual]
ee_pos_peak:       165.4 mm    (v12: 137, standalone: 15.6, target: 10)
ee_ori_peak:       437°        (accumulated, not peak instant)
min_d_swing:        40.8 mm    (v17: 24.9, target: 5)
NMPC infeasibility:  1.9%      (target: 2%)
hw_sat_peak:        ~0.95      (PASS)
platform_rotation:   ~5°       (marginal)
dock:               NO (timeout)
```
Note (amended 2026-04-17): `torso_ori_peak` above was a global `np.max(np.abs(·))`; the SS-only value (0.5432°) comes from `Misc/runs/archive_rediagnostic.md`. Other entries were not re-split; see §10's per-phase table for v19's EE peak in each phase.

### Achievable Floor (from standalone/bisection)
```
QP standalone:           15.6 mm EE
B_const_torso (NMPC on): 13.5 mm EE, 21 mm torso
```

### Gap Analysis
```
Achievable:   15.6 mm EE
Current:     165.4 mm EE (v19), 40.8 mm closest approach
Ratio:       10.6× inflation from standalone to closed-loop
```

---

## 5. Remaining Unknowns

1. **Standalone with v19 config not yet run.** We proposed running the standalone QP test with the v19 moving-torso (planned-δ mapping) and EE feedforward active. This would establish the new achievable floor. If it's ~20mm, the 40mm closed-loop gap is from NMPC/AOCS dynamics. If it's ~35mm, gains need adjustment.

2. **Why v17 got closer (24.9mm) than v19 (40.8mm).** v17 had no mapping (TorsoPlanner direct) — the torso drifted 120mm toward the arm, effectively shortening the remaining reach distance. v19 holds the torso at its planned position, forcing the arm to cover the full 800mm. The arm may not be able to reach 800mm in 7.3s with these gains against the coupling disturbance.

3. **The 31° torso orientation figure** was a `_global` metric artefact, not an SS tracking failure. The per-phase re-diagnostic (2026-04-17, `Misc/runs/archive_rediagnostic.md`) shows SS torso orientation peak at 0.53 ± 0.01° across v17–v21 — below the 0.72° standalone floor. The 29–45° values came from the post-abort DS hold window and are accounted for separately (see §11). SS orientation is tracked to the standalone floor and is not an open unknown. (amended 2026-04-17)

4. **Anchor spacing / step geometry.** DEFAULT_DX=0.8, DEFAULT_DY=0.3 gives a 1.0m diagonal step. The arm reach is 1.375m. The step is 73% of reach — ambitious. Shorter steps would help but the architecture should handle the designed geometry.

5. **The δ̈ term** is still dropped from the mapping's acceleration feedforward. Adding it would make the mapping fully consistent across position, velocity, and acceleration. Currently the PD handles the acceleration mismatch.

---

## 6. Architectural Insights (Lessons Learned)

### The Mapping Dilemma
The CoM→torso mapping is algebraically exact (T3 proved at machine precision). But during SS with a swinging arm:
- **Live δ(q_current):** Best torso tracking (22mm) but feeds arm tracking errors back into the torso reference → 137mm EE error
- **Planned δ(q_planned):** Smooth feedforward, 42mm torso, but EE still at 165mm because the torso's planned motion still couples dynamically
- **No mapping (TorsoPlanner):** 121mm torso (6× worse), EE at 156mm — the geometric quintic can't compensate for arm-induced base drift
- **Static torso (B_const):** 0mm torso, 13.5mm EE — but no torso motion at all

### NMPC Routing
The NMPC's contribution to the QP should be primarily through the **hw constraint on λ** (bounding contact forces to respect the momentum envelope), not through **torso displacement commands**. The momentum management works through constraining what wrenches the QP can use, which shapes the entire solution including both torso and EE accelerations.

### Null-Space Projection Limitations
Acceleration-level null-space projection guarantees orthogonality at each instant but not across time. When the torso moves, the null-space projector N_torso rotates in configuration space, creating velocity-level EE drift proportional to torso velocity. The task-consistent feedforward (v17) partially compensates but doesn't eliminate the drift.

### Priority of Evidence
The standalone QP test (scripts/test_qp_tracking.py) is the single most valuable diagnostic tool. It establishes the achievable floor. Any closed-loop degradation beyond that floor is from the cascade, not the QP itself. Every architectural change should be validated standalone first.

---

## 7. Open Directions for Discussion

### A. Dynamically-consistent null-space projection
Replace kinematic projector `N = I - J†J` with dynamic projector `N_dyn = I - J^T·M·J·M⁻¹` that accounts for inertia coupling. More expensive but eliminates mass-dependent drift. Standard in the whole-body control literature (Khatib 1987, Sentis & Khatib 2005).

### B. Operational-space formulation
Instead of projecting EE into torso null-space at acceleration level, solve directly in operational space: compute the torque needed for each task accounting for the dynamics, then superpose in a priority-consistent way. This avoids the projection drift entirely but requires a more complex QP formulation.

### C. Reduce step ambition
Shorter steps (DX=0.4, DY=0.15) mean the arm covers 500mm instead of 1000mm, the torso moves 300mm instead of 591mm, and the coupling disturbance scales down quadratically with distance/time. More steps per traversal but each step docks reliably. Pragmatic but doesn't fix the fundamental coupling issue.

### D. Whole-body trajectory optimization
Replace the decoupled (TorsoPlanner + SwingPlanner + NMPC) planning with a single whole-body trajectory optimizer that plans all joint motions simultaneously, respecting momentum constraints, joint limits, and dock precision requirements. This is what state-of-the-art terrestrial humanoid locomotion does (Ponton et al., 2021; Dantec et al., 2022). Much more complex but eliminates the coupling problem at the planning level.

### E. Improve the planned-δ quality
The current planned δ(q_planned) uses a linear interpolation of arm joints between start and end. A quintic interpolation matching the swing planner's actual timing would produce a more accurate δ prediction, reducing the gap between planned and actual δ. Also add the δ̈ term to complete the feedforward chain.

### F. Torso orientation management
~~Struck 2026-04-17: SS orientation is at the standalone floor. No management needed.~~

### G. EE gain investigation
Kp_ee=10 with Kd_ee=12 gives ω_n≈3.2 rad/s, ζ≈1.9 (overdamped). The standalone tracks to 15.6mm at these gains. In closed-loop the effective gains are reduced by the null-space projection. Higher Kp_ee (20-30) might close the gap, but needs to be tested against the coupling disturbance.

### H. Actuator-aware trajectory generation (FUNDAMENTAL OPEN QUESTION)

The coarse pre-planner checks **momentum feasibility** (hw box) but not **joint torque feasibility**. It produces a CoM trajectory that respects the momentum envelope but may require accelerations that exceed actuator limits when distributed through the kinematic chain. The torso and swing quintics are geometric — smooth and well-timed but nobody verifies that the peak joint torques they require are within ±τ_max.

The standalone test with v19 config proved this: the mapping-based moving torso reference saturates joints at 20 Nm, leaving zero torque for the EE task. The quintic demands ~13 Nm on the peak stance joint just for the torso (40 kg × 0.67 m/s² peak accel × 0.5m moment arm), leaving 7 Nm for everything else. The EE needs 5-7 Nm. Budget: maxed.

**The trajectory generation must be actuator-aware.** Two levels of solution:
- **Pragmatic (v20):** Reshape the acceleration profile (trapezoidal) so peak joint demand stays within budget. Combined with planned-δ mapping to maintain feedforward during low-acceleration phases.
- **Principled (future):** Whole-body trajectory optimizer that includes joint torque bounds alongside momentum constraints. Plans all joint motions simultaneously, guaranteeing both momentum and actuator feasibility by construction. This is the state of the art in terrestrial humanoid locomotion (Ponton et al., 2021; Dantec et al., 2022). Significantly more complex but eliminates the coupling problem at the planning level.

This is the most important architectural lesson from M7: **individual component validation (QP standalone, NMPC standalone, mapping T3) is necessary but insufficient. The references generated by the planning layer must be jointly feasible within the actuator budget of the coupled system.**

---

## 9. v20 Plan — Trapezoidal + Planned-δ Combined Approach

### Rationale

Two ideas that failed independently should work together:
- **Trapezoidal alone (v18):** Zero `a_ff` during cruise → 120mm torso drift (no feedforward against arm disturbance)
- **Planned-δ alone (v19 quintic):** Smooth feedforward → 42mm torso, but quintic peak acceleration saturates joints at 20 Nm

**Combined:** trapezoidal velocity profile + planned-δ mapping. During ramp-up/down, the torso accelerates with planned-δ providing smooth compensation. During cruise, `a_torso_demand ≈ 0`, the planned-δ handles arm disturbance through the mapping, and the full torque budget is freed for EE tracking.

### Key timing

The EE's precision approach (last 30-40% of swing) must coincide with the torso's cruise phase (where torque is available). Configure the trapezoidal ramp fraction so that cruise covers τ ∈ [0.4, 1.0] — ramp in the first 40%, cruise+decel in the second 60%. This ensures the torso is at near-zero acceleration during the arm's final approach to the dock.

### Expected effect
- Peak joint torque: ~13 Nm during ramp → ~3-5 Nm during cruise (frees 15+ Nm for EE)
- EE tracking during cruise phase: should approach the B_const_torso floor (13.5mm) since the torso is moving at constant velocity (minimal coupling disturbance)
- Torso tracking: planned-δ maintains feedforward compensation even during cruise, preventing the 120mm drift seen in v18

### v20 result — trapezoidal on TorsoPlanner (FAILED)
Trapezoidal on TorsoPlanner had no effect on closed-loop because the SS linear torso reference comes from the mapping (not TorsoPlanner). TorsoPlanner only drives angular reference in SS, and with fixed-rotation IK the angular motion is ~0°.

### v21 result — CoM shaping at pre-planner level (POSITION SOLVED, ORIENTATION ISOLATED)
Added acceleration constraints at the pre-planner: `‖a_com(k)‖ ≤ 0.01 m/s²` for knots k ∈ [0.2·M, 0.8·M]. Forces trapezoidal-like CoM profile.

| Metric | v19 standalone | v21 standalone | v21 closed-loop |
|---|---|---|---|
| torso_pos peak | 42 mm | **34 mm** | 44 mm |
| torso_ori peak | — | **0.72°** | **45.5°** |
| ee_pos peak | 165 mm | **24 mm** | no dock |
| |τ|_∞ peak | 20 Nm (saturated) | **1.17 Nm** | 2.2 Nm |

**The position tracking chain is solved.** 24mm EE at 1.2 Nm in standalone. Actuator saturation eliminated.

~~The remaining problem is purely torso orientation: 0.72° standalone → 45° closed-loop... (the next investigation must focus exclusively on why the closed-loop cascade inflates torso orientation error ... vs the standalone).~~ Struck 2026-04-17: this reading is wrong. The per-phase re-diagnostic below shows SS torso orientation is at the standalone floor across v17–v21; the remaining closed-loop problem is EE position, and the large orientation numbers came from the post-abort DS window. (amended 2026-04-17)

### Per-phase re-diagnostic (2026-04-17)

Re-running v17, v19, v20, v21 `sim_log.json` through the per-phase diagnostic refactor (no re-simulation, no parameter change) gives:

| version | SS peak ori [°] | DS peak ori [°] | ori at SS end [°] | q_ref jump [°] | EE pos peak SS [mm] | EE pos peak DS [mm] | abort? | abort d_mm | abort ori_deg |
|---|---|---|---|---|---|---|---|---|---|
| v17 | 0.5348 | 179.5354 | 0.1431 | 2.9440 | 153.91 | 3811.24 | yes | 24.88 | 6.30 |
| v19 | 0.5432 | 171.9536 | 0.2022 | 3.3173 | 165.37 | 2314.60 | yes | 40.78 | 6.97 |
| v20 | 0.5432 | 118.0349 | 0.2022 | 3.3173 | 165.37 | 2512.07 | yes | 40.78 | 6.97 |
| v21 | 0.5334 | 45.4672 | 0.1990 | 3.4162 | 162.38 | 933.31 | yes | 40.84 | 6.97 |

The SS orientation column is essentially constant at 0.53° across all four versions — the 29–45° figures reported previously (v19 §4, v21 §10 above) were the `_global` column, dominated by post-abort DS divergence. The v20/v21 rationale for CoM shaping at the pre-planner — freeing actuator budget so the EE position task can use it — remains valid and is confirmed by the standalone column (τ peak 1.17 Nm, EE 24 mm). The architectural work in v20/v21 was sound; the characterization of what it had and had not achieved was not. (amended 2026-04-17)

---

## 11. Current State (v21) — Position Inflation in SS Is the Remaining Blocker

### What's solved

- QP task stack: correct (standalone proves 24 mm EE, 0.72° torso ori)
- Actuator budget: no longer saturated (1.17 Nm of 20 Nm in standalone)
- Position tracking chain: CoM shaping + planned-δ mapping + EE feedforward
- Momentum management: NMPC conservation law + hw constraint on λ
- Singularity: eliminated with manipulability init (κ < 7)
- AOCS: sign fixed, orbital term present, exonerated by bisection
- SS torso orientation: at the standalone floor (0.53° closed-loop vs 0.72° standalone) across v17–v21 — not a blocker. (amended 2026-04-17)

### What's broken (revised)

1. **EE position inflation in SS.** Across v17–v21 the SS peak is 153–165 mm and the closest approach at dock_timeout is 25–41 mm; the standalone floor is 24 mm. Roughly 6.7× inflation from standalone to closed-loop. This is the locus §2 already identified (mapping 3.6× + null-space drift 3.4×). It is the primary blocker for M7 dock success. (amended 2026-04-17)

2. **Post-abort DS divergence.** In every archived run the simulation aborts on dock_timeout and then enters a trailing-DS "settle" window in which the torso tumbles 45–180° and both joint and wheel torques saturate. The root cause is at the gait-scheduler level: the main loop at `sim_loop.py:1337-1338` executes `step_idx += 1; i += 2` unconditionally after the SS phase, so the pre-planned trailing-DS entry at `sim_loop.py:1339-1390` is reached regardless of whether SS docked. The trailing DS is a scheduled plan entry, not a post-dock consequence. When SS aborts, trailing-DS runs with arm B un-welded in MuJoCo while the plan, the QP, and the reference generator all treat the system as double-support. Three downstream symptoms follow from this one cause:

   - **H_DS1 (contact config).** `cc_ds = self.sched.contact_config_at(plan.t_start[i]+0.1)` at `sim_loop.py:1343` returns `ContactConfig.DOUBLE`; `get_contact_jacobians(True, True)` at `sim_loop.py:1597-1598` stacks both tool Jacobians against a single-weld physical state.
   - **H_DS2 (reference discontinuity).** `dock_configuration(anchor_a, anchor_b, q_init=pq)` at `sim_loop.py:1365-1375` IKs a pose with both tools at their anchors; `set_hold(rs_eq.oMf_torso, …)` then injects a 2.9–3.4° step in `q_torso_ref` at the SS→DS boundary.
   - **H_DS3 (passivity).** `passivity_active = True` at `sim_loop.py:1712` is gated on `phase == 'DS'` (plan string), not on physical contact state; the passivity inequality's `T_kin` formulation at `wholebody_qp.py:444-459` assumes both welds are rigid.

   See `docs/architecture/POST_ABORT_DIVERGENCE.md` for line-by-line evidence and conclusions. (amended 2026-04-17)

### Next investigation (revised)

**Scheduler-level.** The scheduler needs an `if docked` branch before trailing-DS setup. The *architectural* decision — what the abort-DS semantics should be (freeze swing arm / attempt retraction / declare step failure and stop) — is Idriss's to specify; it is not derivable from the diagnostic data. The three orthogonal experiments in `POST_ABORT_DIVERGENCE.md` (cc_ds → SINGLE_A, set_hold on actual pose, passivity_active off) are retained as **diagnostic decomposition**: they quantify how much of the 45° is attributable to each symptom and are useful for scientific characterization, but they are not the candidate fix.

**Position (primary track).** Apply the A/B/C/D cascade bisection originally drafted for orientation to `ee_pos_peak_SS` as the primary metric (standalone → +NMPC/mapping → +AOCS → full sim_loop). The standalone-with-v21-config floor is 24 mm; the closed-loop peak is 162 mm; bisection isolates which cascade stage contributes each factor in the 6.7× inflation. Until the position chain closes and a step docks cleanly, the post-abort DS question does not gate a 1 % traversal run — but the scheduler-level fix is still required for operational robustness at the operating point. (amended 2026-04-17)

---

## 12. Files Reference

### Key source files
```
crawlbot/simulation/sim_loop.py          — main simulation loop, phase machine
crawlbot/simulation/config.py            — SimConfig (all parameters)
crawlbot/core/com_to_torso_mapping.py    — CoM→torso mapping with δ/δ̇
crawlbot/solvers/wholebody_qp.py         — QP with null-space projection
crawlbot/solvers/centroidal_nmpc.py      — NMPC with conservation law
crawlbot/planning/torso_planner.py       — TorsoPlanner (quintic SLERP)
crawlbot/planning/swing_planner.py       — SwingPlanner (quintic + bump)
crawlbot/planning/coarse_preplanner.py   — Pre-planner (T_step, CoM traj)
crawlbot/aocs/force_estimator.py         — AOCS with orbital correction
crawlbot/core/ik.py                      — IK, manipulability_config
```

### Diagnostic scripts
```
scripts/test_qp_tracking.py              — standalone QP tracking test
scripts/bisect_qp_cascade.py             — A/B/C/D cascade bisection
scripts/run_m7_single_step.py            — closed-loop single step
scripts/diag_platform_rotation.py        — 3-case AOCS isolation
scripts/diag_m7_swing_velocity.py        — velocity profile comparison
```

### Results directories
```
Misc/runs/M7_1pct_1step_v11/  through  Misc/runs/M7_1pct_1step_v19/
Misc/runs/qp_tracking_test/
Misc/runs/M6_platform_diag/
```
## 13. Armature Mismatch — Root Cause of EE Orientation Drift (2026-04-20)

### Finding

The MJCF arm joints carry `armature=0.05` and `damping=0.05` by inheritance
from the `<default class="robot_joint">` block. The URDF used by Pinocchio
does not represent armature (URDF has no equivalent field), and no
post-load assignment was installing it. Pinocchio's inertia matrix `H_pin`
therefore lacked the diagonal rotor-inertia contribution that MuJoCo
integrates. On wrist joints — where link inertia is small and rotor
inertia is a significant fraction of the effective diagonal — the QP's
commanded `q̈` was producing an actual angular acceleration attenuated by
roughly `H_pin/(H_pin + 0.05)`, silently scaled down at integration time.

The bisection leading to this diagnosis worked bottom-up from the QP-vs-MuJoCo
6D residual audit. Key intermediate numbers (from
`ee_full_6d_qp_vs_mujoco.md`, t=3.6 s of A_swing):

| quantity | value |
|---|---|
| QP-commanded angular Z accel | −0.847 rad/s² |
| MuJoCo-integrated angular Z accel | −0.006 rad/s² |
| Wrist torque commanded | 4.6 mNm (0.023% of τ_max) |

The QP was commanding correct angular deceleration to hold EE orientation
while the swing arm executed its 800 mm / 45° reconfiguration; MuJoCo was
integrating an angular acceleration two orders of magnitude smaller. No
torque saturation, no QP failure, no solver degeneracy — purely a model
consistency issue at the mass-matrix level.

### Fix

One block added to `crawlbot/core/robot_interface.py`:

```python
# After pin.buildModelFromUrdf(...), before createData():
armature = np.zeros(model.nv)
armature[6:20] = 0.05  # 14 arm joints, 0 on 6-DOF floating base
model.armature = armature
data = model.createData()
```

### Verification

A_swing (standalone QP, constant torso reference, no NMPC, no mapping,
no AOCS, SwingPlanner EE reference on full step geometry) before and
after the fix:

| metric | before | after | factor |
|---|---|---|---|
| `ee_ori_peak_SS` [deg] | 16.74 | 0.88 | 19× |
| `ee_ori_at_T_step` [deg] | 16.36 | 0.41 | 40× |
| `ee_pos_peak_SS` [mm] | 3.82 | 3.78 | unchanged |
| `tau_q_peak_wrist_b` [mNm] | 4.6 | 51 | 11× |

The wrist-torque increase is the QP now being asked to produce the
angular acceleration the physics actually requires; the controller was
previously undercommanding the wrist because its model disagreed with
the simulator.

EE position was essentially unchanged because position tracking is
driven by shoulder/elbow joints where link inertia dominates over
armature. The mismatch was wrist-specific.

**Commit:** `63a072f` — `fix(robot): install MJCF armature in Pinocchio model`

## 14. Damping Is Not Load-Bearing (Part 2 Sweep, 2026-04-20)

With Pinocchio armature installed, the settle sweep from §12 was
re-examined with damping and armature decoupled. Seven variants, all
with Pinocchio armature matched to the MJCF value per variant:

| variant | MJCF damping | MJCF armature | T_end [J] | exit |
|---|---|---|---|---|
| a0_d0 | 0 | 0 | 0.191 | plateau |
| a0p01_d0 | 0 | 0.01 | 0.150 | plateau |
| a0p02_d0 | 0 | 0.02 | 0.116 | plateau |
| a0p03_d0 | 0 | 0.03 | 2.1e-9 | target_met |
| a0p04_d0 | 0 | 0.04 | 1.6e-10 | target_met |
| a0p05_d0 | 0 | 0.05 | 1.5e-11 | target_met |
| a0_d0p05 | 0.05 | 0 | 0.189 | plateau |

The damping-only variant (`a0_d0p05`) plateaus at the same T_end as the
zero-everything variant (`a0_d0`). **Damping alone does not stabilize
the DS passivity settle.** The stabilizing mechanism is rotor inertia,
not dissipation. Minimum viable armature at 1 ms timestep is between
0.02 and 0.03; `a = 0.05` provides >60% margin.

This result allows MJCF damping to be set to zero without affecting
settle convergence, simplifying the physics to a conservative rigid-body
system (modulo armature) that is consistent with the centroidal NMPC
theorem's frictionless assumption.

**Commit:** (Part 2 sweep commit on current branch)

## 15. Mapping Bypass in SS — Position Tracking Resolved (2026-04-20)

The EE position bisection (§M7_EE_POSITION_BISECTION.md) identified the
CoM→Torso mapping as producing the bulk of the position inflation when
comparing standalone QP (24 mm EE peak) to full closed-loop (162 mm
EE peak). The chain:

| case | description | EE pos peak SS [mm] | Δ from previous |
|---|---|---|---|
| A_swing | standalone, SwingPlanner EE | 3.82 | — |
| B_minus | + NMPC, torso constant | 4.59 | +0.77 |
| B_v21 | + mapping (planned-δ) | 164.79 | +160.20 |
| D | full sim_loop SS | 162.38 | −2.41 |

Adding NMPC alone contributes under 1 mm of additional inflation;
adding the mapping (planned-δ, v21 configuration) adds 160 mm. The
mapping was providing a moving torso reference that consumed QP torque
budget and prevented the EE task from tracking its reference.

### Fix

`SimConfig.mapping_bypass_in_ss = True` causes the SS-phase torso
reference to hold at its SS-entry pose:

```python
# In sim_loop.py, SS reference construction:
if cfg.mapping_bypass_in_ss:
    r_b_ref = self._ss_entry_p_torso      # frozen at SS start
    v_b_ref = np.zeros(3)
    a_b_ff = np.zeros(3)
else:
    r_b_ref, v_b_ref, a_b_ff = self.mapping.compute(...)
```

Angular reference still comes from TorsoPlanner (unchanged).

### Verification

Closed-loop v21 with mapping bypass: EE position peak 32 mm, closest
approach at abort 8 mm. A 5× reduction vs mapping active (162 mm /
41 mm).

## 16. Swing Early-Finish — Clean Dock Kinematics (2026-04-20)

With armature, damping, and mapping all resolved, the closed-loop v21
achieved `d = 3.89 mm, ori = 0.03°` at dock activation — both gate
thresholds met. However, a post-run kinematic audit revealed the weld
was activating while the SwingPlanner was still in its terminal
deceleration ramp (dock fired at t = 7.21 s, before T_step = 7.28 s),
with approach-direction closing velocity of −10.6 mm/s.

### Mechanism

The SwingPlanner's quintic profile reaches zero velocity exactly at
`τ = 1` (i.e., `t = T_step`). When the dock gate fires before T_step,
the reference is in its terminal deceleration ramp and the actual
gripper velocity follows with some lag. The weld activates on a moving
gripper, producing a Baumgarte transient that DS passivity then has to
absorb.

### Fix

`SimConfig.swing_early_finish_fraction = 0.80`. The SwingPlanner reaches
its target pose at `ef · T_step = 0.80 · 7.284 = 5.83 s`, then holds
zero-velocity reference through the remaining 20% of T_step. Dock gate
is augmented to require `t ≥ ef · T_step` alongside the position and
orientation thresholds.

Analogous to the `torso_early_finish_fraction` mechanism tested in v14
for the torso planner; applied here to the swing trajectory.

### Verification

| quantity | before (v21) | after (v22, ef=0.80) |
|---|---|---|
| dock event | no (timeout) | yes |
| `d` at activation [mm] | 40.84 (abort) | 2.70 |
| `ori` at activation [deg] | 6.97 | 0.06 |
| `\|\|v_rel_lin\|\|` at activation [mm/s] | — | 13.6 (receding) |
| `\|\|v_rel_ang\|\|` at activation [mrad/s] | — | 4.6 |

A longer hold (`ef = 0.70`) was also tested and produced *larger*
relative velocity at dock (23.6 mm/s) — the PD residual is a bounded
oscillation around the dock pose, and longer hold catches a different
phase rather than reducing amplitude. ef = 0.80 is the chosen value.

The residual 13.6 mm/s receding velocity is absorbed by MuJoCo weld
Baumgarte (time constant 3 ms) within ~10 ticks, followed by the DS
passivity QP settling within ~100 ms. No velocity gate is specified in
the validation plan (§8 of the brainstorming doc); the residual is
within the implicit tolerance of the dock specification.

## 17. T11 Closed (2026-04-20)

First closed single-step dock in the project. Configuration:

- Pinocchio armature installed (§13)
- MJCF damping = 0, armature = 0.05 on arm joints (§14)
- `mapping_bypass_in_ss = True` (§15)
- `swing_early_finish_fraction = 0.80` (§16)

All other v21 fixes retained: pre-planner cruise-box, EE task-consistent
feedforward, 7-DOF arms, manipulability init, α_wrench = 0.01,
α_com_soft = 0, weight_ratio = 1.0.

### Result

```
SS-phase metrics:
  torso_pos_peak_SS       = 36.5 mm
  torso_ori_peak_SS       =  1.05°
  ee_pos_peak_SS          = 32.4 mm
  ee_ori_peak_SS          =  9.37°
  ee_ori_at_T_step        =  0.06°

Dock event:
  t = 6.01 s, d = 2.70 mm, ori = 0.06°, kinematic
  relative velocity at dock: 13.6 mm/s (receding), 4.6 mrad/s
  approach distance: 0.115 mm

Aborted steps: 0
pytest: 192/192
```

All T11 validation gate criteria met: `d < 5 mm`, `ori < 5°`, `h_w` in
box, 7-DOF arms, no QP failures, no aborted steps.

### Path from v21 to T11 closed

This session's findings are cumulative. The same closed-loop that
plateaued at `d = 40.84 mm` at the start of the session now docks
cleanly with three independent architectural changes applied together,
each diagnosed through targeted bisections and committed with evidence:

1. Pinocchio armature — closed 16° of EE orientation drift (§13)
2. MJCF damping → 0, armature alone retained — theorem-consistent
   physics on both sides (§14)
3. Mapping bypass in SS — closed 130 mm of EE position inflation (§15)
4. Swing early-finish — ensured dock activates after reference completes (§16)

The controller logic was not changed during this session. All four
findings are model-consistency or reference-generation fixes at the
simulation-controller boundary. The QP and NMPC algorithms as
implemented at the start of the session were correct; they were being
asked to run against an inconsistent model of the robot and presented
with references that exceeded what was dynamically feasible.

## 18. Current State (2026-04-20) — T11 closed, T12 open

### What's solved
- T11: 1% mass ratio, single step, 800 mm / 45° / 7.3 s geometry
  — dock achieved, all gate criteria met
- Pinocchio-MuJoCo model consistency on armature
- Frictionless idealization (damping = 0) with discrete-time stability
- EE position tracking chain (standalone floor effectively reached)
- EE orientation tracking chain (standalone floor effectively reached)
- Dock kinematics (swing completes, PD settles, weld activates)

### What's open
- T12: 14% mass ratio, single step. Same geometry. Tests whether
  the fixes in §13-16 generalize across the mass-ratio envelope.
- T15/T16: three-step traversal at 1% and 14%. Tests the scheduler
  logic across multiple DS→SS→DS cycles and compounding errors.
- T17: EE orientation at dock < 5° across traversal. Current result
  at 0.06° suggests comfortable margin, but untested at 14%.
- T18: NMPC solve rate > 95% within 50 ms. Instrumented but not
  aggregated as pass/fail across traversal.
- T19/T20: zero QP failures, dynamics residual < 1e-8 across
  traversal. Same instrumentation status as T18.

### Orthogonal open items
- Post-abort DS divergence. The scheduler-level `if docked` gate
  identified in §M7_DS_DIAGNOSTIC_EXPERIMENTS.md is not needed for
  T11 (which docks successfully) but remains a robustness issue for
  T12 and beyond if a step fails to dock at higher mass ratio.
  Architectural decision on abort-DS semantics still pending.

### Next investigation
T12 run with identical configuration to the T11 closed run. Expected
outcome: either T12 passes with no change (the fixes generalize) or
the new mass ratio exposes a physical coupling not present at 1%
(pre-planner infeasibility, NMPC saturation, dock kinematics
degradation). The three candidate failure modes are separable and
each would point to a different next investigation.

---
