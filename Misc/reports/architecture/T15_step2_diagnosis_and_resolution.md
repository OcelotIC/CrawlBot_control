# T15 step 2 — diagnosis arc and the FK-references resolution

**Branch of record:** `claude/step2-path-diagnostic` @ `cd9a2e8`+
(this document).
**Source of all data cited below:** the per-stage diagnostic
reports listed in §11.
**Date:** 2026-04-25.
**Status:** diagnosis complete; root cause identified; fix
designed; implementation pending (separate prompt).

This document is the synthesis of a multi-week investigation
into why T15 step 2 (anchor-pair (3,4) at 1% mass-ratio) does
not dock under the M7 controller. It rolls together six
diagnostic stages, names the root cause that cuts across all of
them, and proposes the concrete architectural change that
addresses it. The supporting per-stage reports remain
authoritative for their own data; this document is the navigator.

---

## §0  TL;DR

T15 step 2 fails to dock because the controller is asked to
follow a reference path through a configuration that no q ∈
ℝ²¹ satisfies — i.e. the references commanded to the QP are
**kinematically infeasible** along an interior τ ≈ 0.25 of the
SS window. The QP cannot reconcile the three task-space targets
(stance arm at anchor, swing arm on its quintic, torso on its
quintic) and produces large tracking errors that compound until
abort.

Five investigations confirmed this from independent angles:

1. **Phase-4 IK anomaly** (`results/M7_1pct_3step_v22_t15_trajIK_ondemand/T15_trajIK_ondemand_report.md`):
   trajectory-aware IK reported `w_end = 4.09e-8` at step 2 vs
   fixed-rotation IK's `w_product = 1.55e-2`. Apparent IK bug.
2. **IK-anomaly diagnostic** (the
   `claude/manipulability-ik-diagnostic` branch report):
   metric-mismatch artifact + path-dependent cost function.
   Real IK pathology, but specific to the trajectory IK's
   internal warm-start.
3. **IK-fix run** (`Misc/runs/M7_1pct_3step_v22_t15_ik_fix/T15_ik_fix_report.md`):
   four IK fixes applied; trajectory-aware IK now produces
   `w_end = 5.10e-2` at step 2. Steps 0 and 1 dock at the best
   distances of any run (3.20 mm / 3.43 mm). **Step 2 still
   aborts at 429 mm** despite the fixed IK.
4. **Path-geometry diagnostic** (`Misc/runs/diagnostic/T15_step2_path_geometry.md`):
   H2 confirmed — the reference path the planners hand to the
   QP visits a near-singular interior at τ=0.25, where
   `w_ideal` collapses 6 orders of magnitude (4.2e-2 → 2.8e-8)
   and 16 of 21 sample points have no q satisfying torso +
   swing + stance simultaneously.
5. **Mid-waypoint reshape (Option B)** (`Misc/runs/M7_1pct_3step_v22_t15_midwaypoint/T15_midwaypoint_report.md`):
   inserting a manipulability-aware q_mid regresses every
   step (step 0: 3.20 → 34 mm; step 1: 3.43 → 333 mm with 177°
   flip; step 2: skipped). The mid-waypoint cost optimizes
   kinematic conditioning without a trackability term; the
   commanded detours destabilise the cascade.
6. **Trackability diagnostic Q1/Q2** (`Misc/runs/q1_q2/Q1_Q2_trackability_report.md`):
   - Q1: step-1's 177° orientation error is *not* a
     representation/SLERP-hemisphere bug. The reference is
     well-formed (≤12° commanded reorient, 0 sign flips); the
     actual diverges within 1.3 s of SS entry.
   - Q2a (single-step scenario): closed-loop tolerates 2× the
     natural displacement along the q_start→q_end axis. Distance
     alone is not the trackability bound.
   - Q2b (T15 step 2 specifically): **0 of 12 perturbations of
     step-2 q_end dock**. Sweep A (along-axis α ∈ [0.5, 1.5])
     and Sweep B (orthogonal y β ∈ [−0.45, +0.45] m) both fail
     uniformly at ≈430 mm. Step 2 is unreachable for any q_end
     in the tested neighbourhood.

The convergent finding: the failure is not in the IK output, not
in q_end choice, not in displacement magnitude, not in
quaternion/SLERP representation, not in the gait-level (3,4)
geometry per se. **It is in the way references are generated**:
two independent task-space quintics (TorsoPlanner produces a
torso 6D quintic; SwingPlanner produces a swing-EE 6D quintic
+ bump) with no constraint that they be simultaneously
reachable at any τ in the interior.

The fix is to derive both task-space references from the same
joint-space quintic via forward kinematics:

```
q(τ)  = pin.interpolate(model, q_start, q_end, s(τ))
v(τ)  = pin.difference(model, q_start, q_end) · ṡ(τ)        # tangent vector
torso_ref(τ) = FK[fid_torso](q(τ));  twist = J_torso · v(τ)
swing_ref(τ) = FK[fid_swing](q(τ));  twist = J_swing · v(τ)
                                              + clearance · n̂ · bump(τ)  (additive)
```

Every reference triple is simultaneously reachable by
construction. The QP receives consistent setpoints and never
sees the kinematic-infeasibility transient.

This is **not** a controller architecture change in the broader
sense:

- The **cascaded centroidal NMPC + whole-body QP** stays
  intact. Same 9-D centroidal state, same contact-wrench
  inputs, same NMPC ↔ QP interface.
- The **QP task stack** stays intact. Same tasks (stance, torso,
  swing, CoM, L_com, posture, wrench reg), same priorities, same
  weights, same null-space projection.
- Only the *source* of the reference signals changes — same
  dataclass shape (`TorsoReference`, `SwingReference`), same
  units, same frames.

Implementation scope: ~5–7 days, gated behind
`cfg.reference_source: str = 'task_space'` (default, byte-
identical to current behaviour) vs `'joint_space_fk'` (new path,
opt-in for ablation; default flips after validation).

---

## §1  Origin — the Phase-4 anomaly that turned out not to be an anomaly

The investigation began with a sharp, surprising number.

In Phase 4 of the Manipulability-IK-1 program (the "on-demand
trajectory-aware IK" run, `results/M7_1pct_3step_v22_t15_trajIK_ondemand/`),
the per-step IK trace showed:

| Step | pair  | mode                       | θ [°] | dp [mm] | t_ik [s] | w_worst | w_end |
|-----:|:-----:|----------------------------|------:|--------:|---------:|--------:|------:|
| 0    | (2,3) | trajectory_aware_on_demand | 0.57  | 361.1   | 18.66    | 7.63e-2 | 7.63e-2 |
| 1    | (3,3) | trajectory_aware_on_demand | 1.46  | 534.8   | 12.07    | 6.04e-2 | 6.04e-2 |
| 2    | (3,4) | trajectory_aware_on_demand | 8.01  | 1176.7  | 18.52    | **4.09e-08** | **4.09e-08** |

The step-2 output `w_end = 4.09e-8` is six orders of magnitude
below the other steps and below any reasonable threshold for a
non-singular configuration. Compared to the same run's
fixed-rotation IK at step 2 reporting `w_product = 1.55e-2`,
the discrepancy looked like an IK-formulation bug: same
(q_start, anchor_pair (3,4)), wildly different `w` values.

The Phase-4 report initially attributed this to "step-2
singularity intrinsic to anchor-pair (3,4) geometry on this
mass-ratio". Closed-loop step-2 aborted at 460.7 mm; that
seemed consistent with a geometric infeasibility.

**This framing was wrong, in three ways the IK-anomaly
diagnostic later untangled.**

### §1.1  The metric mismatch

The two IKs reported *different scalars* under the same name
"manipulability":

- `dock_configuration_fixed_rotation.w_product`
  = `√det(J_a J_aᵀ) · √det(J_b J_bᵀ)` (Yoshikawa, product of all
  6 task-direction singular values per arm — 12 SVs total).
- `manipulability_config_trajectory.w_end`
  = `σ_min(J_a) · σ_min(J_b)` (product of the two minimum
  singular values).

For typical well-conditioned arms these scalars differ by a
factor of 3–5×. Comparing 1.55e-2 (Yoshikawa) with 4.09e-8
(σ_min product) was an apples-to-oranges category error. Under
a common metric (σ_min product at the same q_end), both IKs
sat in the 5–6 × 10⁻² regime — same order of magnitude.

### §1.2  The path-dependent cost

`manipulability_config_trajectory`'s cost function had an
internal `_cache['q_prev']` warm-start for the inner
`solve_ik`. The same torso xyz could yield `cost(xyz)` values
separated by 7 orders of magnitude depending on whether
`solve_ik` was warm-started from a previous Nelder-Mead
iteration's q or cold-started from `pin.neutral`. The cost was
not a deterministic function of its decision variable. The
4.09e-8 was a Phase-4-specific Nelder-Mead trajectory falling
into the cold-start basin at one specific torso pose.

### §1.3  The narrow multi-start

The trajectory IK ran three multi-start seeds at
`midpoint + dz ∈ {0, −0.3, −0.6}` (vertical-axis only). A
brute-force grid search (the IK-anomaly diagnostic's §3.2) found
better basins at torso xyz roughly 0.9 m from any of the three
seeds. For some (q_start, anchor_pair) inputs the multi-start
missed feasible high-w basins entirely.

### §1.4  Implication for what came next

Three IK pathologies, all real, all narrow:

- A → metric reporting (cosmetic but misleading).
- C → path-dependent cost (the actual mechanism behind 4.09e-8).
- B → narrow multi-start (auxiliary).

The diagnostic verdict was **not** "step 2 is geometrically
infeasible". It was "the IK code has fixable defects; fix them
and re-test." That fix became the IK-fix branch — see §2.

---

## §2  The IK fix and what it didn't fix

The IK fix (the `claude/manipulability-ik-fix` branch, merged
into the lineage that became this branch) applied four
priorities derived from the IK-anomaly diagnostic and codified
in `docs/architecture/IK_FORMULATION.md` §9:

1. **Deterministic inner-solve seed (§9.1)**: removed
   `_cache['q_prev']`; inner `solve_ik` always seeded from
   `q_start` with torso xyz overwritten. `cost(xyz)` is now a
   pure function of xyz.
2. **Broadened multi-start (§9.2)**: 7 seeds spanning all three
   Cartesian axes plus q_start torso xyz and the
   fixed-rotation IK output. Falls back to 6 seeds if the
   fixed-rotation pre-solve fails.
3. **Unified metric reporting (§10)**: `dock_configuration_fixed_rotation`
   now returns both Yoshikawa and σ_min-product. Downstream
   logging shows both. The metric mismatch is no longer
   silently misleading.
4. **Post-convergence safety check (§9.3)**: rejects converged
   trajectory-IK results with `w_end < cfg.trajectory_ik_w_min_threshold`
   (default 1e-3) and falls back to `dock_configuration_fixed_rotation`.

Test suite: 200/200 pass on the IK-fix tip
(192 pre-existing + 4 trajectory-aware + 4 new mid-waypoint
regression added later on this branch).

### §2.1  IK-fix run results

`Misc/runs/M7_1pct_3step_v22_t15_ik_fix/T15_ik_fix_report.md`
captures the run; the key numbers:

| Step | pair  | w_end (post-fix) | w_end (Phase 4)  | ratio   |
|-----:|:-----:|----------------:|-----------------:|--------:|
| 0    | (2,3) | 9.20e-2         | 7.63e-2          | 1.21×   |
| 1    | (3,3) | 6.85e-2         | 6.04e-2          | 1.13×   |
| 2    | (3,4) | **5.10e-02**    | **4.09e-08**     | **1.25 × 10⁶ ×** |

Step 2's IK output went from singular (4.09e-08) to
well-conditioned (5.10e-02) — the IK-output pathology is
*completely* resolved. No singular endpoint, no 8° pirouette
(post-fix step-2 reorient is 3.90°).

Closed-loop dock outcomes:

| Run            | Step 0                      | Step 1                      | Step 2 |
|:---------------|:----------------------------|:----------------------------|:-------|
| Baseline       | DOCKED t=6.01s d=3.82mm     | DOCKED t=13.02s d=4.84mm    | ABORT t=28.49s d=374.4mm |
| Phase 4        | DOCKED t=6.21s d=4.97mm     | DOCKED t=13.07s d=4.72mm    | ABORT t=33.35s d=460.7mm |
| **IK-fix**     | DOCKED t=6.21s d=**3.20mm** | DOCKED t=17.72s d=**3.43mm**| **ABORT t=35.58s d=429.5mm** |

Steps 0 and 1 dock at the *best* distances of any run. Step 2
*still aborts* at 429 mm — slightly better than baseline's
374 mm at the abort instant in absolute terms, but still
two orders of magnitude beyond the 5 mm dock criterion.

### §2.2  What the IK fix proved

The IK fix was a methodological success and a clinical failure:

- **Success**: the diagnostic correctly identified three real
  IK defects; the four-priority fix resolved all three; the
  `w_end = 4.09e-8` value can no longer occur.
- **Failure of expectation**: step 2 was supposed to dock once
  the IK gave a well-conditioned q_end. It didn't. The
  controller cannot reach a q_end with `w_end ≥ 5e-2` either.

This mismatch — IK output is fine, controller still fails — was
the data point that rerouted the investigation. If the IK is
not the bottleneck, what is? The path-geometry diagnostic
(§3) was scoped to answer exactly that.

---

## §3  Path-geometry diagnostic — H2 confirmed

`Misc/runs/diagnostic/T15_step2_path_geometry.md` operationalises
three hypotheses:

- **H1 (time budget)**: closed-loop σ_min stays well-conditioned,
  EE just doesn't reach the anchor in `T_step`.
- **H2 (reference singular)**: σ_min drops in the *idealised
  reference path*; the QP follows the references; both go
  singular together. Fix is in the reference shape.
- **H3 (QP-induced detour)**: σ_min stays well-conditioned along
  the idealised reference path but drops along the actual
  closed-loop path. The QP is doing more than the references
  demand.

### §3.1  Method

For each step's SS window 21 evenly-spaced τ samples are drawn.
At each τ:

- The reference triple `(p_torso_ref(τ), R_torso_ref(τ),
  p_swing_ref(τ), R_swing_ref(τ))` is sampled from `sim_log` at
  sim time `t(τ)`.
- A **3-task IK** (torso pose + swing-arm pose + stance-arm pose)
  is run from a joint-space-interpolation seed, with damped LS +
  backtracking line search.
- If it converges, the resulting `q_ideal(τ)` lies on the
  reference path. `w_ideal(τ) = σ_min(J_a) · σ_min(J_b)` at
  `q_ideal`.
- If it does *not* converge within 1e-6 task error, the
  references at that τ are not simultaneously satisfiable to IK
  tolerance — itself diagnostic.
- The actual closed-loop `q_actual(τ)` is then sampled from
  `physics_trace.pkl` and the same `w_actual(τ)` computed.

### §3.2  The decisive datum

For step 2 (anchor pair (3,4), SS window 18.48 → 35.48 s):

| τ    | t [s]  | w_ideal     | IK conv | e_swing_pos [mm] |
|-----:|-------:|------------:|:-------:|-----------------:|
| 0.00 | 18.48  | 4.238e-02   | ✔       |  2.9             |
| 0.05 | 19.33  | 4.456e-02   | ✔       | 37.3             |
| 0.10 | 20.18  | 4.355e-02   | ✔       | 46.2             |
| 0.15 | 21.03  | 3.753e-02   | ✔       | 28.2             |
| 0.20 | 21.88  | 2.315e-02   | ✔       | 12.3             |
| **0.25** | **22.73** | **2.802e-08** | **✘** | 31.5     |
| 0.30 | 23.58  | 5.621e-07   | ✘       | 94.7             |
| 0.40 | 25.28  | 1.102e-07   | ✘       | 261.6            |
| 0.50 | 26.98  | 3.584e-06   | ✘       | 399.8            |
| 0.75 | 31.23  | 1.767e-06   | ✘       | 436.6            |
| 1.00 | 35.48  | 2.196e-06   | ✘       | 452.6            |

Two regimes, sharp transition at τ=0.25:

- **τ ∈ [0, 0.20]** (5/21 samples): IK converges to ~1e-9 task
  error. `w_ideal ∈ [2.3e-2, 4.5e-2]`. References are mutually
  satisfiable.
- **τ ≥ 0.25** (16/21 samples): IK fails (residual 0.02–0.34).
  `w_ideal` collapses 6 orders of magnitude (4.2e-2 → 2.8e-8) in
  one 0.85 s interval. The references at these τ are not
  simultaneously satisfiable to IK tolerance.

The closed-loop `w_actual` enters the singular regime one τ
sample later (τ=0.30), and the swing-EE tracking error grows
**immediately after** the singular collapse (31 mm at τ=0.25 →
95 mm at τ=0.30 → 453 mm at τ=1.0, matching the abort distance).

Steps 0 and 1 (which dock cleanly) do **not** show this
pattern: their `w_ideal` stays in [10⁻², 10⁻¹] for the full SS
window. Step 2 is *qualitatively* different — not a bigger
version of the steps-0/1 pattern.

### §3.3  Verdict

- H1 rejected: w_ideal is singular for 75% of step-2 SS, not a
  clean path running out of time.
- **H2 confirmed**: w_ideal collapses 6 orders at τ=0.25; the
  3-task IK can't satisfy the references; `w_actual` tracks
  `w_ideal` into the same singular regime with one-sample lag.
- H3 rejected: `w_actual ≥ w_ideal` everywhere from τ=0.25
  onward (ratio 30 to 7×10⁴). The QP is *regularising* the
  singular reference, not making it worse — it just can't fully
  recover.

### §3.4  What "singular reference" actually means here

The reference triple at τ=0.25 demands:

- Torso at `p_torso_ref(τ)`, `R_torso_ref(τ)`: midway along the
  TorsoPlanner quintic between (p_t0, R_t0) and (p_t1, R_t1).
- Swing-EE at `p_ee_ref(τ)`, `R_ee_ref(τ)`: midway along the
  SwingPlanner quintic between anchor_b[3] and anchor_b[4],
  plus the clearance bump.
- Stance-A at `anchor_a[3]`: held throughout.

These three are produced by **independent task-space planners**.
Nothing in the architecture requires them to be jointly
satisfiable at every τ. At τ=0.25 of step 2, they are not.
There is no q ∈ ℝ²¹ that places all three at their commanded
poses simultaneously — `pin.log6(oMf actInv tgt)` for at least
one frame stays at residual 0.02–0.34 m·rad regardless of where
the IK starts.

This is **the structural finding**. Everything that follows is
either evidence for it or a failed attempt to work around it.

---

## §4  Mid-waypoint reshape (Option B) — regression on every step

The path-geometry diagnostic recommended (§7.3 of that report)
three candidate fixes ordered by invasiveness:

- **Option A** — planning-time path-singularity guard.
- **Option B** — reshape the reference between (3,3) and (3,4)
  by inserting a manipulability-aware mid-waypoint q_mid such
  that piecewise quintics (q_start → q_mid → q_end) stay
  well-conditioned throughout.
- **Option C** — full short-horizon trajectory optimisation.

Option B was implemented and validated.
`Misc/runs/M7_1pct_3step_v22_t15_midwaypoint/T15_midwaypoint_report.md`
captures the run.

### §4.1  Implementation

Six commits on this branch:

| Commit     | Content |
|------------|---------|
| `a9ff933`  | `manipulability_config_mid_waypoint` IK in `crawlbot/core/ik.py`. Decision variable: torso xyz at the mid-waypoint. Cost: worst-case `σ_min(J_a) · σ_min(J_b)` over 5 interior τ samples per sub-quintic. Multi-start: 7 seeds (same set as `manipulability_config_trajectory` post-fix). Returns `(q_mid, w_worst, success)` with `success = w_worst ≥ threshold`. |
| `9546c1e`  | `TorsoPlanner.add_phase` extended with optional `(p_mid, R_mid, t_mid)` for piecewise quintic; legacy single-quintic preserved. |
| `ba87ee3`  | `SwingPlanner.add_phase` added — explicit phase override for swing-EE references; piecewise mode supports `(p_ee_mid, R_ee_mid, t_mid)` with bump applied additively over the full phase. |
| `5a06861`  | `sim_loop._setup_torso_for_step` wired with `cfg.use_path_feasibility_check`, `cfg.use_mid_waypoint_reshape`, `cfg.mid_waypoint_force_on`, `cfg.path_feasibility_w_threshold`. |
| `a38e9fd`  | `check_path_feasibility` helper in `ik.py` (3-task IK at 21 τ samples — runtime version of the path-geometry diagnostic). |
| `7878af6`  | 4 regression tests in `tests/test_mid_waypoint_reshape.py`. |

Test suite under flags off: 200/200 pass. The implementation is
clean and exercises every code path. The architectural change
itself is correct.

### §4.2  Phase-7 validation results

T15 with `use_mid_waypoint_reshape=True` and
`mid_waypoint_force_on=True` (per-step IK trace from
`Misc/runs/M7_1pct_3step_v22_t15_midwaypoint/ik_trace.json`):

| Step | pair  | mid_used | w_worst_mid | q_mid torso xyz [m] |
|-----:|:-----:|:--------:|------------:|---------------------|
| 0    | (2,3) | ✔        | 8.88e-2     | [**−0.655**, 0.013, −0.235] |
| 1    | (3,3) | ✔        | 6.59e-2     | [0.898, **−0.444**, −0.128] |
| 2    | (3,4) | ✔        | 3.98e-2     | [0.332, −0.369, −0.347] |

Closed-loop dock outcomes:

| Step | IK-fix outcome              | midwaypoint outcome                    |
|-----:|:----------------------------|:---------------------------------------|
| 0    | DOCKED d=3.20 mm            | **ABORT d=34.4 mm**                    |
| 1    | DOCKED d=3.43 mm            | **ABORT d=333.2 mm, ori=176.8°**       |
| 2    | ABORT d=429.5 mm            | **SKIP** (preplanner_infeasible)       |

All three steps regressed. The IK output itself was
well-conditioned at every step (w_worst_mid ∈ [4e-2, 9e-2], all
≫ the 1e-3 threshold). The mid-waypoint IK did exactly what it
was asked to do.

### §4.3  Why Option B failed — the cost function blind spot

`manipulability_config_mid_waypoint`'s cost is purely kinematic:

```
cost(p_t_mid) = − min over τ of σ_min(J_a(q(τ))) · σ_min(J_b(q(τ)))
```

It optimises kinematic conditioning along the piecewise quintic
without any path-length, trackability, momentum-loading, or
torque-envelope penalty. With 7 seeds spanning a wide
neighbourhood, Nelder-Mead naturally finds a high-w basin —
which can be geometrically far from the start-to-end geodesic.

Step 0's q_mid_torso[0] = −0.655 sits **0.78 m backwards** from
q_start[:3] ≈ +0.12 and 1.4 m from the IK-fix step-0 q_end
(≈+0.75). The closed-loop QP is asked to drive the torso +0.12
→ −0.655 → +0.75 in 6 s. It cannot. Step 0 mistracks by 30 mm.

Step 1 then enters from an off-nominal state; the mid-waypoint
optimiser commands a 10° torso reorient (vs IK-fix's 3.76°);
the closed-loop falls 333 mm behind and ends with a 177°
orientation flip. By step 2 the state is so far off-nominal
that the coarse pre-planner cannot find a momentum-feasible
plan at all, and the step is skipped.

### §4.4  The runtime gate that didn't fire

The `check_path_feasibility` runtime gate (which would
nominally decide whether to insert a mid-waypoint at all)
reported "feasible" at every step:

| Step | gate verdict | gate w_min |
|-----:|:------------:|-----------:|
| 0    | feasible     | 8.68e-2    |
| 1    | feasible     | 6.39e-2    |
| 2    | feasible     | 1.44e-2    |

The gate operates on a *simplified* planner reference
(linear quintic + symmetric `sin²(πτ)` bump, no M5 CoM-mapping
layer). The actual closed-loop tracks the *mapped* reference
where w_ideal collapses to 2.8e-8 at step 2 (per the
path-geometry diagnostic). The gate underreports by 6 orders
of magnitude. Without `mid_waypoint_force_on=True` the
mid-waypoint code path would never have been exercised in a
T15 run.

This is itself diagnostic: a runtime gate that approximates the
planner refs cannot be made faithful without integrating the
mapping layer. A faithful gate is more expensive than just
fixing the underlying problem.

### §4.5  What §4 told us

- **Option B as scoped does not work.** Six clean commits, 200/200
  tests pass, a thorough Phase-7 validation, and the dock
  outcomes are uniformly worse than baseline. The
  implementation cannot be tweaked into success without changing
  the cost function.
- **The cost function blind spot is real and fundamental**: any
  cost over q_mid that ignores trackability will find off-axis
  optima the controller can't track. A trackability term would
  need to encode QP/NMPC bandwidth and AOCS torque envelope —
  approaching the complexity of full TO.
- **The mid-waypoint code stays in tree, gated off by default**.
  All three flags default to False. 200/200 tests pass with
  flags off (byte-identical to the IK-fix tip).

But §4 alone didn't rule out simpler fixes, like a path-length
penalty on q_mid. Q1/Q2 (§5) was scoped to nail down whether
distance is the binding trackability constraint.

---

## §5  Q1/Q2 trackability — distance is not the bound

`Misc/runs/q1_q2/Q1_Q2_trackability_report.md` answers two related
questions left open after §4:

- **Q1**: Is the 177° step-1 orientation in §4.2 a
  representation/SLERP-hemisphere bug? If yes, a small fix in
  the planner's quaternion handling could rescue Option B.
- **Q2**: How far can the IK place q_end from q_start before
  the closed-loop fails to track? Quantifies the trackability
  bound directly.

### §5.1  Q1 — the 177° is real, not representational

From `sim_log.json` of the mid-waypoint run, step-1 SS:

| metric                                | torso  | swing-EE |
|---------------------------------------|-------:|---------:|
| max commanded reorient (ref vs ref(0))| 10.0°  | 12.1°    |
| max ref-vs-actual error (geodesic)    | 178.6° | 178.8°   |
| time of first divergence (>30°)       | t=13.52 s (1.30 s into SS) | similar |
| q_torso_ref / q_ee_ref sign-flips     | **0** / **0** | **0** / **0** |
| q_torso_actual / q_ee_actual sign-flips | 3 / 6 | — |

The reference quaternion stream is well-formed (zero sign
flips across 142 SS samples; commanded reorient never exceeds
12°). The piecewise-SLERP replay (using the same
`pin.log3 / pin.exp3` machinery the planners use, with the
actual step-1 mid-waypoint quaternions) produces a path that
peaks at 9.5° from start — geodesic-bounded, no hemisphere
ambiguity. All three pairwise quaternion inner products
(`⟨q_start, q_mid⟩`, `⟨q_mid, q_end⟩`, `⟨q_start, q_end⟩`) are
+0.997+, same sign throughout.

The 177° is genuine open-loop divergence. Within 1.3 s of SS
entry, the actual orientation drifts off a small (≤12°)
reference because the controller cannot track the dynamics
induced by the mid-waypoint's aggressive 0.78-m torso translation
detour. Once divergence starts, errors compound: by t=22 s the
orientation error oscillates in [100°, 180°] for the rest of
the SS window. The 177° at exit is the integral of that
uncontrolled rotation.

Verdict **Q1-C** (genuine kinematic divergence). No cheap
representation fix exists — the failure is dynamic.

### §5.2  Q2a — single-step sweep (misleading positive)

Following the original brief literally, Q2a ran a single-step
T15-equivalent simulation (`scripts/run_m7_single_step.py`,
n_steps=1, `start_a=2, start_b=2`) with
`dock_configuration_fixed_rotation` monkey-patched to return
`q_perturbed = q_start + α · (q_natural − q_start)` over the
whole 21-dim configuration (torso pos linear, torso quat
SLERP-extrap via log/exp, arm joints linear). α ∈ {0.5, 0.75,
1.0, 1.25, 1.5, 1.75, 2.0}.

Result: **all 7 α DOCKED**, including α=2.0 with a torso
displacement of 1.18 m and arm-joint displacement of 5.38 rad
(2× the natural step's 0.59 m / 2.69 rad).

The mid-waypoint step-1 had `||q_mid_torso − q_start_torso||`
= 1.093 m — within the empirically-validated trackable range.
On its face, this said: distance alone is not the trackability
bound; Option B's mid-waypoint was within distance bounds; the
failure mode must be off-axis path shape, and a "snap mid-waypoint
back toward the geodesic midpoint" cost-function tweak (Option
B'') would help.

**This conclusion was wrong, and the brief's Q2.3 anticipated
why**: the single-step (2,2)→(2,3) scenario has fundamentally
different kinematics from T15 step 2's (3,3)→(3,4). Q2a
generalised the wrong way.

### §5.3  Q2b — T15 step 2 specifically (rigorous test)

Q2b (`Misc/scripts/diagnostic_q2b_step2_trackability.py`) runs the
full 3-step T15 scenario and intercepts
`dock_configuration_fixed_rotation` *only on its 3rd call*
(step 2). Steps 0 and 1 run with the natural IK so the
closed-loop state at step-2 entry matches the IK-fix baseline.
Two sweeps:

- **Sweep A (along-axis)**: q_perturbed = q_start + α(q_natural
  − q_start), α ∈ {0.5, 0.75, 1.0, 1.25, 1.5}.
- **Sweep B (orthogonal lateral-y)**: q_perturbed = q_natural
  with q_perturbed[1] += β, β ∈ {−0.45, −0.30, −0.15, 0.00,
  +0.15, +0.30, +0.45} m. Tests the empirically-observed
  failure direction (the Phase-7 mid-waypoint q_mid for step 1
  was at torso y = −0.44 m vs natural ≈ 0).

**Result: 0 of 12 perturbations docked.**

| α     | step-2 e_swing_peak [mm] | docked? |
|------:|-------------------------:|:-------:|
| 0.50  | 439.8                    | ✘       |
| 0.75  | 431.7                    | ✘       |
| 1.00  | 432.8 (= IK-fix baseline)| ✘       |
| 1.25  | 414.3                    | ✘       |
| 1.50  | 397.2                    | ✘       |

| β [m] | step-2 e_swing_peak [mm] | docked? |
|------:|-------------------------:|:-------:|
| −0.45 | 423.5                    | ✘       |
| −0.30 | 427.7                    | ✘       |
| −0.15 | 431.4                    | ✘       |
|  0.00 | 432.8 (= α=1, sanity)    | ✘       |
| +0.15 | 431.9                    | ✘       |
| +0.30 | 430.4                    | ✘       |
| +0.45 | 423.0                    | ✘       |

Along-axis range: 397–440 mm (slight monotone improvement with
α, no knee). Orthogonal-y range: 423–433 mm (total spread only
10 mm; nearly flat).

### §5.4  Q2 verdict

| Verdict | Description                                                                | Status |
|:-------:|:---------------------------------------------------------------------------|:------:|
| Q2-A    | Sharp trackability threshold along the natural axis                        | rejected |
| Q2-B    | Smooth degradation along axis OR orthogonal direction has a tractable dock | rejected |
| **Q2-C** | **Step 2 is infeasible from this q_start regardless of q_end choice** within the tested neighbourhood | **confirmed** |

**The decisive finding: T15 step 2 cannot be made to dock by
choosing a different q_end within ±50% along-axis OR ±0.45 m
laterally.** The failure is not in the q_end target; it is in
the *trajectory* the planners generate from this q_start to any
q_end within the tested neighbourhood.

This rules out:

- Option B' (`||q_mid − q_start||²` distance penalty): contradicted
  by Q2a — that distance is empirically tracked.
- Option B'' (off-axis penalty): contradicted by Q2b — no
  off-axis q_end choice fixes step 2.
- Mid-waypoint reshape of any variant: ruled out by data.

What it does *not* rule out:

- Option C (full trajectory optimisation): the trajectory shape
  must be the optimisation variable, not a single waypoint or
  endpoint.
- The architectural fix in §7 below: change *how* the references
  are generated so the singular reference interior never exists.

---

## §6  Root-cause synthesis — kinematically-uncoupled task-space refs

The five investigations converge on a single mechanism. This
section names it precisely.

### §6.1  The current reference architecture

`crawlbot/simulation/sim_loop.py::_setup_torso_for_step` is
called at every DS→SS boundary. After running the IK to obtain
`q_end`, it configures two **independent** task-space planners:

```python
# torso_planner.py:155-201
self.torso_planner.add_phase(
    t_ss_start, t_ss_start + T_step,
    p_t0, R_t0,    # torso pose at q_start (FK)
    p_t1, R_t1,    # torso pose at q_end   (FK)
    delta_com_start=δ0, delta_com_end=δ1,
    early_finish_fraction=cfg.torso_early_finish_fraction,
)

# swing_planner.py reference_at(t):
#   reads scheduler.anchors_b[swing_to_idx] etc. at every t
#   (no add_phase call needed under the legacy non-override path).
```

`TorsoPlanner.reference_at(t)` then returns
`(p_torso(τ), R_torso(τ), v_torso(τ), a_torso(τ))` via:

- Position: `p = p_t0 + s(τ)·(p_t1 − p_t0)` (linear quintic).
- Rotation: `R = R_t0 · exp3(s(τ) · log3(R_t0ᵀ R_t1))` (SLERP
  on SO(3) via the same s(τ)).
- Velocity / acceleration: analytic derivatives via the chain
  rule on s(τ).

`SwingPlanner.reference_at(t)` returns
`(p_ee(τ), R_ee(τ), v_ee(τ), ω_ee(τ), a_ee(τ), α_ee(τ))` via:

- Position: `p_ee = p_anchor_start + s(τ)·dp + clearance·n̂·bump(τ)`
  (linear quintic in EE-Cartesian + asymmetric `sin²(πτ)`
  clearance bump in the away-normal direction).
- Rotation: `R_ee = R_release · exp3(σ_r(τ_d) · log3(R_releaseᵀ R_target))`
  (SLERP with delayed-cosine timing σ_r ≠ s).

Two independent planners. No shared state, no IK between them,
no kinematic-feasibility check at any τ.

### §6.2  Why this works at steps 0 and 1

For "easy" anchor pairs — where the natural geodesic on the
q-manifold between q_start and q_end happens to keep the torso
and swing-EE references roughly co-located in their respective
task spaces — the two independent quintics produce reference
triples that *happen* to be jointly satisfiable at every τ.
The 3-task IK in the path-geometry diagnostic confirms this:
21/21 IK-converge for step 0 and 20/21 for step 1 (the lone
failure at τ=1 of step 1 is a dock-handoff artifact).

The architecture relies on a **happy accident**: that
independent SLERP-in-task-space and SLERP-in-EE-Cartesian
produce kinematically consistent triples. For most anchor pairs
on this robot at this mass-ratio, they do.

### §6.3  Why this breaks at step 2

For the (3,4) transition, the natural geodesic on the q-manifold
sweeps through configurations where the two task-space SLERPs
diverge. Specifically: the linear-in-s torso pose and the
linear-in-s swing-EE pose, evaluated at τ=0.25, demand a
configuration where the swing arm has moved roughly halfway in
EE-Cartesian (~400 mm) but the torso has moved roughly halfway
in pose space (~590 mm of displacement, plus ~2° of reorient).
Combined with the held stance arm at anchor_a[3], no q ∈ ℝ²¹
can simultaneously satisfy all three constraints to IK
tolerance.

The path-geometry diagnostic measured this directly: w_ideal at
τ=0.25 collapses 6 orders of magnitude, and the 3-task IK
residual stays in [0.02, 0.34] m·rad for the rest of the SS.
Steps 0 and 1's reference triples don't visit such a region;
step 2's does.

### §6.4  What every other investigation was actually measuring

Each of the five diagnostics is a different probe of the same
mechanism:

- **Phase-4 IK 4.09e-08**: an IK code defect (path-dependent
  cost) that *also* happens to surface when the cost evaluation
  walks through the singular region. Once the IK was fixed, the
  defect went away — but the underlying singular-reference
  problem remained.
- **IK-fix step 2 abort at 429 mm**: the controller follows the
  reference triple from the IK-fix's q_end, and that triple
  visits the singular interior just like Phase 4's did. Same
  abort distance (429 vs 461 mm); same mechanism.
- **Path-geometry w_ideal = 2.8e-8 at τ=0.25**: the smoking gun.
  Direct measurement of the singular reference interior.
- **Mid-waypoint Option B regresses every step**: the
  mid-waypoint optimiser tries to reshape the *path*, not fix
  the *reference architecture*. It produces kinematically-
  consistent waypoints (w_worst_mid ≥ 4e-2 at every step) but
  the closed-loop dynamics — under the same independent-
  task-space-planner architecture — can't follow the kinked
  trajectory.
- **Q2b 0/12 perturbations dock**: confirms that no choice of
  q_end (within a wide neighbourhood) routes the references
  away from the singular interior, because the singular
  interior is created by the **planner architecture**, not by
  q_end's value.

### §6.5  The structural statement

> The closed-loop system fails when the **two independent
> task-space planners produce, at some interior τ of an SS
> window, a reference triple (torso, swing, stance) that admits
> no kinematically-feasible q**. The QP cannot satisfy all three
> tasks; tracking error grows; AOCS cannot recover; the step
> aborts. This failure mode is invisible to the IK (which only
> looks at endpoints), invisible to the controllers (which take
> references as ground truth), and invisible to any single-
> waypoint reshape of the IK output. **It can only be fixed by
> generating references that are kinematically consistent by
> construction.**

§7 spells out what that means concretely.

---

## §7  The fix — references derived by FK from one joint-space quintic

Replace the two independent task-space quintics with a single
joint-space quintic, and derive both task-space references via
forward kinematics:

```
q(τ)  = pin.interpolate(model, q_start, q_end, s(τ))
v(τ)  = pin.difference(model, q_start, q_end) · ṡ(τ)
a(τ)  = pin.difference(model, q_start, q_end) · s̈(τ)

torso_ref(τ): (p, R) = FK[fid_torso](q(τ))
              twist  = J_torso(q(τ)) · v(τ)
              accel  = J_torso(q(τ)) · a(τ) + getFrameAcceleration|qdd=0

swing_ref(τ): (p, R) = FK[fid_swing](q(τ))   (+ clearance · n̂ · bump(τ) on p)
              twist  = J_swing(q(τ)) · v(τ)  (+ clearance · n̂ · bump_dot/T_eff on linear)
              accel  = J_swing(q(τ)) · a(τ) + getFrameAcceleration|qdd=0
                       (+ clearance · n̂ · bump_ddot/T_eff² on linear)

stance_ref(τ) = anchor pose (held by construction since q_start
                and q_end both place the stance arm at the anchor)
```

Where:
- `s(τ) = 10τ³ − 15τ⁴ + 6τ⁵` is the quintic time-scaling already
  used by both planners (boundary conditions
  `s(0)=0, s(1)=1, ṡ(0)=ṡ(1)=s̈(0)=s̈(1)=0`).
- `pin.interpolate` handles the SE(3) free-flyer geodesic +
  linear interpolation on revolute joints natively; for
  s(τ)∈[0,1] it produces a valid q on the configuration
  manifold.
- `pin.difference(model, q_start, q_end)` is the manifold-aware
  "q_end − q_start" — an `nv`-length tangent vector that
  satisfies `pin.integrate(model, q_start, dq_geo) = q_end`. For
  free-flyer + revolute models, the free-flyer block is the
  SE(3) log of the relative transform; revolute blocks are
  element-wise subtraction. **`dq_geo` is constant along the
  geodesic** — geometric acceleration is identically zero, so
  the only acceleration in tangent space is the chain-rule s̈
  term.

### §7.1  Kinematic consistency by construction

By construction, the reference triple `(torso, swing, stance)`
at every τ comes from FK on a single q. There is one q that
satisfies all three references simultaneously: q(τ) itself.

The 3-task IK that the path-geometry diagnostic ran at every
τ would converge in zero iterations to q(τ) under this
architecture, with task residual zero. The "no q satisfies
torso + swing + stance" failure mode of the current
architecture (§6) cannot occur.

This is the entire technical content of the fix. Everything
else is plumbing.

### §7.2  What changes and what doesn't

**Changes** — only the reference *generation* path inside the
two planners:

- `TorsoPlanner.add_phase` accepts `q_start, q_end` (in addition
  to legacy `p_start/R_start/p_end/R_end` for backwards-compat).
- `TorsoPlanner.reference_at(t)` branches: legacy path produces
  refs by independent linear-quintic + SLERP; new path produces
  refs by FK on `pin.interpolate(model, q_start, q_end, s(τ))`.
- `SwingPlanner.add_phase` and `SwingPlanner.reference_at(t)`
  same pattern.
- `TorsoPlanner.l_com_reference_at(t)` upgrades from the
  torso-only `I_torso · ω_torso_ref` formula to the full
  centroidal momentum `pin.computeCentroidalMomentum(q, v)`,
  capturing the limb contribution that was a documented ~20%
  approximation in the current code.
- `sim_loop._setup_torso_for_step` passes `(pq_live, q_end)` to
  both planners' `add_phase` calls. The IK chain that produces
  q_end is unchanged.
- `sim_loop._planned_arm_config(t, rs)` extends to full-q
  interpolation (not just arm slice) under FK mode, so the M5
  CoM-mapping layer sees the same q(τ) the planners see.
- A new `cfg.reference_source: str` flag selects between
  `'task_space'` (default, byte-identical legacy) and
  `'joint_space_fk'` (new path). Default flips after T15 step-2
  validation passes.

**Does not change**:

- The **cascaded centroidal NMPC + whole-body QP architecture**.
  Same 9-D centroidal state in NMPC (CoM pos, vel, angular
  momentum). Same contact-wrench inputs. Same dt_nmpc=0.1 s,
  dt_qp=0.01 s. Same NMPC ↔ QP interface contract.
- The **QP task stack**. Same tasks (stance contact constraint,
  torso 6D, swing-EE 6D, CoM tracking, L_com tracking, posture
  regularisation, wrench regularisation). Same priorities.
  Same `weight_ratio = 1.0` and `α_wrench = 0.01` from
  `CLAUDE.md`. Same null-space projection.
- The **NMPC state, dynamics, or cost**. The L_com_ref values
  the NMPC consumes become more accurate (full-body vs
  torso-only) but the NMPC's formulation is unchanged.
- The **IK functions** (`dock_configuration_fixed_rotation`,
  `manipulability_config`, `manipulability_config_trajectory`).
  They produce the same q_end they do today.
- The **gait scheduler** (anchor sequence, contact phases,
  T_step). The same (3,4) transition is attempted; it just
  becomes feasible because the references are now reachable.
- The **MJCF**, mass-ratio, AOCS hardware envelope (hw_max=5
  Nms, tau_w_max=5 Nm), or any external interface.

### §7.3  Velocity and acceleration math, in detail

The chain-rule construction gives well-defined v and a along
the geodesic. The key facts:

1. `dq_geo = pin.difference(model, q_start, q_end)` is computed
   *once* at `add_phase` time and cached on the phase dict.
   It does not depend on τ.
2. `v_full(τ) = dq_geo · ṡ(τ)` is the tangent-space velocity at
   τ. For the free-flyer block, this is the LOCAL body twist;
   for revolute joints, this is the joint rate. This is the
   convention `pin.forwardKinematics(model, data, q, v)`
   expects.
3. `a_full(τ) = dq_geo · s̈(τ)` is the tangent-space acceleration.
   The geometric acceleration along a geodesic is identically
   zero — only the chain-rule s̈ term contributes.
4. World-frame frame twist:
   `v6 = J_torso(q(τ)) · v_full` (or J_swing). Use
   `LOCAL_WORLD_ALIGNED` to match the convention in
   `crawlbot/core/robot_interface.py::RobotState`.
5. World-frame frame acceleration:
   `a6 = J_torso · a_full + (J̇_torso · v_full)`. The second
   term is computed by calling
   `pin.forwardKinematics(model, data, q, v_full, np.zeros(nv))`
   and then `pin.getFrameAcceleration(model, data, fid, LOCAL_WORLD_ALIGNED)`
   — at zero generalized acceleration, this returns exactly
   `J̇·v_full`.
6. Bump on swing position is **additive**:
   `p_ee = FK[swing](q(τ)).translation + clearance · n̂ · bump(τ)`,
   with the bump derivative added to the linear velocity and
   acceleration components only (rotation untouched).

### §7.4  Centroidal momentum reference upgrade

The current `TorsoPlanner.l_com_reference_at(t)` computes
`L_com = R(t) · I_torso · R(t)ᵀ · ω_torso_ref(t)` — torso-only,
with a documented ~20% error from missing limb contribution
(see comment at `crawlbot/planning/torso_planner.py:333`).

Under the new architecture, the planner has access to `q(τ)`
and `v_full(τ)`, so the full centroidal angular momentum is
computable directly:

```
pin.computeCentroidalMomentum(model, data, q(τ), v_full(τ))
L_com_ref = data.hg.vector[3:6]
```

This includes all limb contributions and is exact for the
quintic reference. The NMPC consumes this as a 3-vector at
every solve cycle (~10 Hz). Cost: ~one extra
`computeCentroidalMomentum` call per NMPC tick, which is ~20 µs
in Pinocchio's C++ implementation — negligible.

The legacy `set_torso_inertia` API becomes deprecated in the
FK path and emits a `DeprecationWarning`. The legacy path
keeps the existing formula for byte-identical reproduction.

### §7.5  Why this is the right shape of fix

The fix is uniquely cheap relative to the alternatives and
uniquely targeted at the root cause:

- **Versus Option C (full TO)**: TO would *find* an optimal
  joint-space trajectory under explicit dynamics constraints.
  This fix *uses* the natural joint-space geodesic (which the
  diagnostic showed stays well-conditioned at step 2 with
  w ≥ 2e-2 throughout). No optimisation; cheaper to implement
  and faster to evaluate.
- **Versus mid-waypoint reshape (Option B)**: ruled out by data
  (§4, §5).
- **Versus gait-level fix (transit anchor, multi-segment SS)**:
  routes around the geometric problem rather than solving it.
  The next problematic anchor pair would fail similarly. This
  fix solves the underlying generation issue, so it works for
  any anchor pair.
- **Versus QP/NMPC re-tuning**: the QP is *correctly* tracking
  the references it's given. Re-tuning the QP to track an
  infeasible reference faster wouldn't help (and might
  destabilise existing working steps).

The fix is exactly as invasive as it needs to be and not more.

---

## §8  Implementation scope

The full implementation plan is `/root/.claude/plans/magical-munching-book.md`
(internal). The summary here.

### §8.1  Files touched

| File | Edit |
|------|------|
| `crawlbot/planning/torso_planner.py` | New `_interpolate_phase_fk` and `_com_reference_fk` helpers. `add_phase` accepts optional `q_start`, `q_end`, `q_mid`. Constructor takes optional `model`/`frame_torso`. `reference_at` and `com_reference_at` branch on `phase['use_fk']`. Legacy path preserved. |
| `crawlbot/planning/swing_planner.py` | Same pattern: FK-based override branch in `_override_reference_at`. Bump remains additive on FK position. `set_swing_orientation` becomes no-op under FK mode (deprecated). Delayed-cosine SLERP dropped under FK (the joint-space geodesic supplies a kinematically-consistent rotation profile by construction). |
| `crawlbot/planning/torso_planner.py::l_com_reference_at` | Replace torso-only formula with `pin.computeCentroidalMomentum(q, v).vector[3:6]`. `set_torso_inertia` deprecated under FK mode. |
| `crawlbot/simulation/sim_loop.py` | `_setup_torso_for_step` passes `(pq_live, q_end)` to both planners' `add_phase` calls. Wire planners with `model` and frame IDs at construction. `_planned_arm_config` extends to full-q `pin.interpolate` under FK mode. |
| `crawlbot/simulation/config.py` | Add `reference_source: str = 'task_space'` flag. Optional `swing_fk_with_delayed_cosine: bool = False` for ablation. |

No edits to `crawlbot/core/ik.py` (the `_interpolate_q_quintic` helper at line 488 is reused as-is). No edits to NMPC, QP, MJCF, or scheduler.

### §8.2  Tests

A new `Misc/tests/test_fk_reference_consistency.py` with 8 tests:

1. TorsoPlanner FK endpoint exactness (FK at q_start matches
   ref at τ=0; same at τ=1; v=0 and a=0 at both).
2. TorsoPlanner FK interior velocity numerical match
   (finite-difference on FK(q(τ ± ε)) matches analytic v
   from chain rule; tolerance 1e-5).
3. Centroidal momentum reference accuracy
   (`L_new` vs legacy `L_old = R·I·Rᵀ·ω`; non-trivial limb
   contribution measurable at a swing-arm-active configuration).
4. SwingPlanner FK endpoint exactness with bump
   (at τ=0.5 with default clearance 0.03 and `bump_peak_tau=0.5`,
   `ref.p_ee − FK[swing](q(0.5)).translation = 0.03 · n̂`).
5. `set_swing_orientation` is no-op in FK mode.
6. T15 baseline byte-identical under `reference_source='task_space'`
   (snapshot regression against current 200/200-pass tip).
7. **T15 step-2 dock under FK mode** (the critical pass criterion):
   - `min(w_actual)` over step-2 SS ≥ 1e-2 (vs current 1.6e-4).
   - Step 2 docks at d ≤ 5 mm.
   - At least 8/12 of the Q2b q_end perturbations dock under FK
     (vs current 0/12).
8. Acceleration finite-difference cross-check (numerical a
   matches the analytic `J·a_full + J̇·v_full` decomposition;
   tolerance 1e-3).

### §8.3  Effort estimate

Total **5–7 focused engineering days**:

| Task | Effort |
|------|--------|
| TorsoPlanner FK path (new helpers + add_phase + constructor) | 0.75 d |
| SwingPlanner FK path (override branch + bump + deprecation) | 0.75 d |
| `l_com_reference_at` rewrite to centroidal momentum | 0.25 d |
| `sim_loop._setup_torso_for_step` plumbing | 0.5 d |
| `_planned_arm_config` extension to full-q | 0.5 d |
| `cfg.reference_source` flag + dispatch | 0.25 d |
| Unit tests E.1–E.5, E.8 | 1.0 d |
| Integration test E.6 (byte-identical) | 0.5 d |
| Integration test E.7 (T15 step-2 dock validation) | 1.0 d |
| Pinocchio convention checks (R1, R2 risk mitigation) | 0.5 d |
| Documentation + commit messages | 0.25 d |
| Buffer for Pinocchio API surprises | 0.5–1.0 d |

The Pinocchio API conventions (`pin.difference` for free-flyer
returning local body twist; `pin.getFrameAcceleration` semantics
under zero qdd) are the only real unknowns. Front-loading
tests E.1–E.2 as TDD-style convention validators on day 1 locks
these down before touching the planners.

### §8.4  Risks and mitigations

1. **`pin.difference` convention for the free-flyer**: returns
   `[v_lin_local, ω_local]`. The `v_full` we pass to FK and to
   centroidal-momentum must match this convention. *Mitigation*:
   write E.2 finite-difference test first (TDD); run on a
   configuration with `R_torso ≠ I` so a missed transformation
   surfaces.
2. **`pin.getFrameAcceleration` at zero qdd**: returns
   `J̇·v + J·qdd` in some Pinocchio builds; semantics depend on
   which `pin.forwardKinematics` call last ran. *Mitigation*:
   always call `pin.forwardKinematics(q, v, np.zeros(nv))`
   immediately before `getFrameAcceleration`. Cross-check with
   `getFrameJacobianTimeVariation @ v_full` in a unit test.
3. **`q_end` with non-IK-consistent free-flyer pose**: if any
   IK code path produces a `q_end` whose torso rotation is left
   at neutral while position is set by the optimiser, FK at q_end
   would not match the IK target rotation. *Mitigation*: the
   existing `rs_e = self.robot.update(q_end, ...)` and `R_t1 =
   rs_e.oMf_torso.rotation` already extract the canonical
   FK-derived endpoints from q_end; the FK reference path
   returns them by construction.
4. **`computeCentroidalMomentum` cost in NMPC inner loop**: now
   does a full FK + centroidal computation per query instead of
   a 3×3 mat-vec. *Mitigation*: profile under the existing M7
   baseline. Pinocchio's centroidal computation is ~20 µs; not
   a bottleneck.
5. **Mid-waypoint piecewise mode interaction**: the existing
   piecewise-quintic add_phase machinery (used by the Phase-7
   mid-waypoint code, gated off by default) needs to compose
   with the FK path. *Mitigation*: per-segment chain rule is
   identical; `pin.difference` for `(q_start, q_mid)` and
   `(q_mid, q_end)` are both well-defined. Test by extending E.7
   with `mid_waypoint_force_on=True`.
6. **Diagnostic-script drift**: the path-geometry diagnostic
   samples `torso_planner.reference_at(t)` and assumes the
   legacy SLERP. Under FK mode it would silently get FK-derived
   refs and report different `w_min` numbers. *Mitigation*: print
   the `reference_source` flag in the diagnostic header; keep
   both numbers in the regression record.

---

## §9  What's not addressed and what remains open

The fix in §7 addresses the structural failure of step 2. It
does not address every question raised by the investigation.

### §9.1  Within-scope but not part of the §7 fix

- **Step 0 / step 1 robustness margins**: the IK-fix run docks
  steps 0 and 1 cleanly at d=3.20 mm and d=3.43 mm. Under FK-mode
  these should remain at least as good (the FK path is a
  refinement of an already-feasible reference, not a regression).
  But the FK-mode T15 validation should report the per-step
  dock distances to confirm.
- **NMPC infeasibility in the post-abort DS hold**: the IK-fix
  run shows 32 `Infeasible_Problem_Detected` events vs baseline's
  9. If step 2 docks under FK mode, step 2 doesn't enter the
  post-abort DS hold and these events don't accumulate. If it
  doesn't dock, the increase is residual and worth a second
  look. Either way, this is a downstream consequence, not an
  independent issue.
- **AOCS desaturation behavior**: outside the scope of the
  reference-shape fix. The current implementation already
  handles desaturation per spec; nothing about the FK references
  changes the AOCS layer's role.

### §9.2  Out-of-scope but worth noting

- **Higher mass ratios (T16 = 14%)**: the (3,4) anchor pair's
  natural geodesic at 14% mass-ratio may visit different
  configurations than at 1%. The FK-mode references are
  guaranteed kinematically consistent at any mass ratio — the
  fix is mass-ratio-agnostic in that sense. But T16 has its own
  AOCS / dynamics envelope that hasn't been validated yet.
- **Longer N (multi-step T17, T18)**: cascading state drift
  across more steps could expose other singular-reference modes
  not seen in T15. The FK fix removes one source of cascading
  drift (reference infeasibility) but is not a guarantee against
  all of them.
- **Off-anchor-grid trajectories**: any future scenario that
  uses non-grid anchor positions (e.g., curved hull surfaces)
  would interact with FK-derived references through the same
  channel — the fix should generalise, but is not validated for
  it.
- **Bump shape and clearance**: the additive bump in §7
  preserves current behavior. If bump-induced perturbations to
  the FK trajectory cause any new tracking issues at higher mass
  ratios, the bump shape (asymmetric `bump_peak_tau`) may need
  re-tuning. Out of scope here.

### §9.3  Methodological reflections

The investigation took six diagnostic stages because each stage
ruled out one hypothesis cleanly. In retrospect:

- **The IK-anomaly diagnostic** (Phase 4 → IK fix) was the
  right call: the path-dependent cost was a real defect that
  would have re-surfaced in unrelated scenarios.
- **The path-geometry diagnostic** could have been run earlier,
  before the mid-waypoint Option B implementation. It already
  contained enough evidence (w_ideal collapses 6 orders at
  τ=0.25; 16/21 IK failures) to predict that any
  single-waypoint reshape would fail. The Option B implementation
  was clean and correct, but the data already said it was the
  wrong target.
- **Q1 and Q2** confirmed what §3 strongly implied: distance
  isn't the bound; q_end choice isn't the bound; the bound is
  trajectory shape, and trajectory shape is determined by the
  reference architecture. Q2b's 0/12 dock count was the
  cleanest possible negative result.

The convergent finding (kinematically-uncoupled refs) is
visible in retrospect from any single stage. It took six to
rule out everything else.

---

## §10  Artifact index

All paths are relative to the repository root.

### §10.1  Per-stage reports (authoritative for their own data)

| Stage | Report | Branch |
|-------|--------|--------|
| Phase 4 anomaly | `results/M7_1pct_3step_v22_t15_trajIK_ondemand/T15_trajIK_ondemand_report.md` | `claude/trajectory-aware-ik-pWRpA` (closed) |
| IK-anomaly diagnostic | `IK_ANOMALY_REPORT.md` | `claude/manipulability-ik-diagnostic` (closed) |
| IK formulation spec | `docs/architecture/IK_FORMULATION.md` | this branch |
| IK-fix validation | `Misc/runs/M7_1pct_3step_v22_t15_ik_fix/T15_ik_fix_report.md` | `claude/manipulability-ik-fix` (merged into this branch) |
| Path-geometry diagnostic | `Misc/runs/diagnostic/T15_step2_path_geometry.md` | this branch |
| Mid-waypoint reshape (Option B) | `Misc/runs/M7_1pct_3step_v22_t15_midwaypoint/T15_midwaypoint_report.md` | this branch |
| Q1/Q2 trackability | `Misc/runs/q1_q2/Q1_Q2_trackability_report.md` | this branch |
| **Synthesis (this document)** | `Misc/reports/architecture/T15_step2_diagnosis_and_resolution.md` | this branch |

### §10.2  Run output directories

| Run | Directory |
|-----|-----------|
| Baseline T15 (3-step, 1%, fixed-rotation IK only) | `Misc/runs/M7_1pct_3step_v22_t15_bug1fix_vel/` |
| Phase 4 trajIK on-demand | `results/M7_1pct_3step_v22_t15_trajIK_ondemand/` |
| IK-fix validation | `Misc/runs/M7_1pct_3step_v22_t15_ik_fix/` |
| Phase 7 mid-waypoint | `Misc/runs/M7_1pct_3step_v22_t15_midwaypoint/` |
| Q2a single-step sweep | `Misc/runs/diagnostic_q2/alpha_{0.5..2.0}/` |
| Q2b T15 step-2 sweep | `Misc/runs/diagnostic_q2b/{A_alpha_*, B_betay_*}/` |

### §10.3  Diagnostic scripts

| Script | Purpose |
|--------|---------|
| `Misc/scripts/diagnostic_step2_path_geometry.py` | 3-task IK along reference path; produces step{0,1,2}_data.json + figures |
| `Misc/scripts/diagnostic_q1_slerp_repro.py` | Replays piecewise SLERP with actual step-1 quaternions; checks hemisphere consistency |
| `Misc/scripts/diagnostic_q2_trackability.py` | Single-step α-sweep (Q2a) — monkey-patches `dock_configuration_fixed_rotation` |
| `Misc/scripts/diagnostic_q2b_step2_trackability.py` | Full T15 with step-2 IK intercepted; α and β sweeps |
| `Misc/scripts/run_m7_v22_1pct_3step_t15_ik_fix.py` | T15 runner with IK-fix flags |
| `Misc/scripts/run_m7_v22_1pct_3step_t15_midwaypoint.py` | T15 runner with mid-waypoint flags (Option B) |

### §10.4  Branch lineage

```
origin/main
  └── claude/trajectory-aware-ik-pWRpA      (Phase 1-4: cache + on-demand trajIK)
       └── claude/manipulability-ik-diagnostic   (read-only IK-anomaly probe)
       └── claude/manipulability-ik-fix          (IK_FORMULATION §9 fixes; 200/200)
            └── claude/step2-path-diagnostic     (THIS branch)
                  ├── path-geometry diagnostic
                  ├── Option B implementation (gated off by default)
                  ├── Phase 7 validation
                  ├── Q1/Q2 trackability diagnostic
                  └── this synthesis document
```

The implementation of the §7 fix would land on a new branch
`claude/joint-space-fk-references` forked from this branch, per
the implementation plan at `/root/.claude/plans/magical-munching-book.md`.

### §10.5  Key commits on this branch

| Commit | Subject |
|--------|---------|
| `31f548e` | path-geometry diagnostic — H2 confirmed |
| `a9ff933` | mid-waypoint IK function (Option B) |
| `9546c1e` | TorsoPlanner piecewise-quintic add_phase |
| `ba87ee3` | SwingPlanner piecewise add_phase override |
| `5a06861` | sim_loop wiring + 4 config flags for Option B |
| `a38e9fd` | check_path_feasibility runtime helper |
| `7878af6` | 4 mid-waypoint regression tests |
| `f35a1a1` | Phase 7 validation — Option B regresses |
| `3ce416e` | Q1 — orientation failure is genuine |
| `b29d9fe` | Q2a — single-step sweep |
| `cd9a2e8` | Q2b — T15 step 2 unreachable from any q_end |
| (pending) | this synthesis document |

---

**End of synthesis.** The next prompt should be the
implementation of §7 per the plan in
`/root/.claude/plans/magical-munching-book.md`. Stop here.
