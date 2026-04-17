# M7 Technical Log — Complete History and Current State

**Date:** 2026-04-17
**Branch:** `claude/m7-core-task-replace-3Hmf4`
**Last commit:** `68d3b18` (v19)
**Tests:** 191/191 passing

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
torso_pos_peak:     42.0 mm    (v12: 22, v16: 121, target: 10)
torso_ori_peak:     31.1°      (v12: 29, target: 5)
ee_pos_peak:       165.4 mm    (v12: 137, standalone: 15.6, target: 10)
ee_ori_peak:       437°        (accumulated, not peak instant)
min_d_swing:        40.8 mm    (v17: 24.9, target: 5)
NMPC infeasibility:  1.9%      (target: 2%)
hw_sat_peak:        ~0.95      (PASS)
platform_rotation:   ~5°       (marginal)
dock:               NO (timeout)
```

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

3. **The 31° torso orientation error** persists across all versions (v12: 29°, v16: 31°, v19: 31°). With zero planned rotation (fixed-rotation IK), this is pure arm-reaction disturbance on the torso orientation. The torso PD (Kp=6) can't reject it. But the singularity (κ<7) and torque headroom (4.4% used) suggest this should be fixable with higher angular gains — yet the v3 experiment showed only 18% improvement from doubling Kp. Needs investigation.

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
The 31° orientation error from arm reaction is independent of all the position-tracking work. With κ<7 and 4.4% torque usage, there should be room for better orientation rejection. Possible: arm-reaction feedforward for orientation (predict the torque the swing arm will exert on the torso from the planned swing trajectory, and include it in the orientation feedforward).

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

**The remaining problem is purely torso orientation:** 0.72° standalone → 45° closed-loop. This is an independent problem from position tracking that has persisted at 29-45° across ALL versions (v12-v21). The next investigation must focus exclusively on why the closed-loop cascade inflates torso orientation error 60× vs the standalone.

---

## 11. Current State (v21) — Orientation Is The Last Blocker

### What's solved
- QP task stack: correct (standalone proves 24mm EE, 0.72° torso ori)
- Actuator budget: no longer saturated (1.2 Nm of 20 Nm)
- Position tracking chain: CoM shaping + planned-δ mapping + EE feedforward
- Momentum management: NMPC conservation law + hw constraint on λ
- Singularity: eliminated with manipulability init (κ < 7)
- AOCS: sign fixed, orbital term present, exonerated by bisection

### What's broken
- **Torso orientation error: 45° in closed-loop vs 0.72° in standalone**
- This has been 29-45° across all versions since v12
- With zero planned rotation (fixed-rotation IK), this is pure arm-reaction disturbance
- The QP has authority (0.72° standalone proves it) but something in the cascade prevents it from using that authority

### Next investigation
The same bisection approach that identified the mapping as the position culprit should be applied to orientation:
- (A) Standalone QP: 0.72° — baseline
- (B) + NMPC + mapping: measure
- (C) + AOCS: measure
- (D) Full sim_loop: 45°
Identify which cascade component inflates orientation error from 0.72° to 45°.

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
results/M7_1pct_1step_v11/  through  results/M7_1pct_1step_v19/
results/qp_tracking_test/
results/M6_platform_diag/
```
