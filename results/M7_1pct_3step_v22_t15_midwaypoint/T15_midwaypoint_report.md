# T15 — mid-waypoint reshape (Option B) validation

**Branch:** `claude/step2-path-diagnostic` @ `b8688fe`+ this report.
**Scenario:** T15 (3-step, 1% mass-ratio, `aocs_off_in_ds=True`,
`swing_early_finish_fraction=0.80`, `mapping_bypass_in_ss=True`,
MJCF transient mutation: damping=0.0, armature=0.05).
**Run directory:** `results/M7_1pct_3step_v22_t15_midwaypoint/`
**Date:** 2026-04-25.
**MJCF md5 (pre = post):** `96d229250ca882951f1c0d2516391421` ✔ restored byte-exactly.

---

## §0  TL;DR

**Option B as implemented is a regression.** The mid-waypoint
reshape inserts a manipulability-aware q_mid that maximises
`σ_min(J_a)·σ_min(J_b)` worst-case over the piecewise quintic, but
the optimiser has **no path-length / trackability constraint**. It
selects q_mid configurations geometrically far from `q_start` (one
step proposes a backward 0.78 m torso detour; another puts the
torso 44 cm in `−y`). The closed-loop QP / NMPC cannot follow the
resulting reference within SS time budgets, and step-0 — which
docked cleanly at d=3.20 mm under the IK-fix — now times out at
d=34 mm. Step 1 ends with a **177° orientation flip**. Step 2's
pre-planner becomes infeasible entirely (no momentum-feasible plan
exists from the post-step-1 state) and the step is skipped.

The mid-waypoint q_mid w-product values are themselves
well-conditioned (8.88e-2, 6.59e-2, 3.98e-2 — all >> 1e-3
threshold). The IK is doing what it was asked. The cost function
is wrong: it optimises a kinematic invariant without any dynamic
trackability term.

**Verdict:** Option B-as-specified does not close step 2; it
strictly regresses on previously-working steps. Per the brief,
do not attempt further fixes. Next investigation needs to go
beyond mid-waypoint to a real trajectory optimisation that
includes path-length and tracking-bandwidth penalties (Option C
or full TO).

---

## §1  Configuration delta

| Run                | use_traj_aware_ik | use_path_feas_check | use_mid_wp_reshape | mid_wp_force_on |
|--------------------|------------------:|--------------------:|-------------------:|----------------:|
| Baseline (bug1fix) | False             | False               | False              | False           |
| Phase 4 on-demand  | True              | False               | False              | False           |
| IK-fix             | True              | False               | False              | False           |
| **midwaypoint (this)** | True          | True                | True               | **True**        |

`mid_waypoint_force_on=True` because §3 below confirms the runtime
gate underreports — its w_min ≥ 1.4e-2 on every step, well above the
1e-3 threshold. The gate would never fire on its own, so the
mid-waypoint code path would never be exercised in a closed-loop
T15 if we relied on it. Forcing always-on lets us validate the
reshape's closed-loop effect directly.

All other config (preplanner, mapping, swing timing, NMPC, QP) is
byte-identical to the IK-fix run.

Test suite: 200/200 pass on commit `7878af6` (4 new mid-waypoint
regression tests, all green; runtime ~50 s).

---

## §2  Per-step IK trace

From `ik_trace.json`:

| Step | pair  | θ_reorient | dp [mm] | w_end (q_end) | mid_used | w_worst_mid | q_mid torso xyz [m] |
|-----:|:-----:|-----------:|--------:|--------------:|:--------:|------------:|---------------------|
| 0    | (2,3) | 2.33°      |  637.3  | 9.20e-2       | ✔        | 8.88e-2     | [**−0.655**, 0.013, −0.235] |
| 1    | (3,3) | **10.00°** |  812.2  | 6.85e-2       | ✔        | 6.59e-2     | [0.898, **−0.444**, −0.128] |
| 2    | (3,4) | **30.37°** | 1330.3  | 4.29e-2       | ✔        | 3.98e-2     | [0.332, −0.369, −0.347] |

**The IK output is well-conditioned.** Every w_end and w_worst_mid
sits in the 4–9 × 10⁻² regime, far above any singular threshold.
The trajectory IK and the mid-waypoint IK both did exactly what
they were asked to do.

**The mid-waypoint torso choices are pathological w.r.t. the
closed-loop, not the IK metric.** Step 0's q_mid_torso x = −0.655
sits 0.78 m **backwards** from `q_start[0]≈+0.12` and 1.4 m from
the IK-fix-equivalent step-0 q_end (≈+0.75). The mid-waypoint
optimiser found a manipulability-positive basin that requires the
torso to traverse +0.12 → −0.655 → +X in 6 s — a path the
controller can't track within bandwidth.

**Cascade in θ:** step 0's commanded reorient is 2.33° (small);
step 1 jumps to 10° because the live state at step 1 entry is
already off-nominal from step 0's failed dock; step 2 reaches
30.37° because two prior cascading failures have left the robot
nowhere near its planned path. Compare to the IK-fix run: steps
0/1/2 commanded 2.33°/3.76°/3.90°. The cascade is purely a
consequence of the closed-loop tracking failures upstream.

---

## §3  Per-step path-feasibility metrics

The runtime gate (`check_path_feasibility`) reports per-step:

| Step | gate verdict       | gate w_min | gate τ@min | mid-waypoint inserted? |
|-----:|:------------------:|-----------:|-----------:|:----------------------:|
| 0    | feasible           | 8.68e-2    | 0.00       | ✔ (forced on)         |
| 1    | feasible           | 6.39e-2    | 0.00       | ✔ (forced on)         |
| 2    | feasible           | 1.44e-2    | 0.10       | ✔ (forced on)         |

**The gate would not have fired on any step.** Without
`force_on=True`, the mid-waypoint path is never taken. The gate
operates on the **simplified planner reference** (linear quintic +
sin²(πτ) bump, no M5 CoM-mapping); the
`T15_step2_path_geometry.md` diagnostic measured the **actual
mapped reference** and found w_ideal collapsing to 2.8e-8 at
τ=0.25 of step 2. The gate underreports by 6 orders of magnitude.

This is a useful negative finding: the runtime gate as currently
constructed cannot detect the H2 failure mode it was designed for.

---

## §4  Closed-loop dock outcomes

From `sim_log.dock_events` and `aborted_steps`:

| Run            | Step 0                      | Step 1                              | Step 2 |
|:---------------|:----------------------------|:------------------------------------|:-------|
| Baseline       | DOCKED t=6.01s d=3.82mm     | DOCKED t=13.02s d=4.84mm           | ABORT t=28.49s d=374.4mm |
| Phase 4        | DOCKED t=6.21s d=4.97mm     | DOCKED t=13.07s d=4.72mm           | ABORT t=33.35s d=460.7mm |
| IK-fix         | DOCKED t=6.21s d=**3.20mm** | DOCKED t=17.72s d=**3.43mm**       | ABORT t=35.58s d=429.5mm |
| **midwaypoint**| **ABORT** t=11.71s d=**34.4mm** | **ABORT** t=26.42s d=**333.2mm**, ori=**176.8°** | **SKIP** preplanner_infeasible at t=26.93s |

Every step is worse:

- **Step 0** regressed from a 3.20 mm dock to a 34 mm timeout
  (10× worse separation; never closed).
- **Step 1** went from a 3.43 mm dock to a 333 mm timeout, and
  the robot ended in a near-180° orientation flip — the
  trajectory-tracking layer lost stability under the imposed
  reference detour.
- **Step 2** was already a timeout under IK-fix; under
  midwaypoint, the post-step-1 state is so far off-nominal that
  the **coarse pre-planner cannot find a momentum-feasible
  trajectory** for any T_step. Step 2 is skipped entirely.

---

## §5  Closed-loop interior manipulability (qualitative)

Not formally re-run on this branch (the analysis script for
interior σ_min sampling lives on the diagnostic-only side and the
brief did not require numerical sampling). Qualitatively, from the
sim_log tracking errors and the abort patterns:

- Step 0: closed-loop EE tracking failed by ~30 mm despite
  well-conditioned IK output and a non-singular w_actual interior
  — the failure is **trackability**, not geometric infeasibility
  along the path.
- Step 1: 177° final orientation indicates the QP / NMPC entered
  a regime where the structure-frame coordinates wrap, plausibly
  through a high-momentum transient produced by the aggressive
  torso detour.
- Step 2: not analysable (no SS executed; pre-planner refused).

The interior `w_actual` along the closed-loop trajectory is not
the limiting factor. The reference, even reshaped, demands
configurations the controller cannot reach.

---

## §6  NMPC health

From `sim_log.nmpc_status_str`:

| Status                         | Baseline | IK-fix | **midwaypoint** |
|--------------------------------|---------:|-------:|----------------:|
| Solve_Succeeded                | 429      | 487    | 433             |
| Solved_To_Acceptable_Level     | 21       | 23     | 20              |
| Infeasible_Problem_Detected    | 9        | 32     | 5               |
| Maximum_Iterations_Exceeded    | 0        | 1      | 0               |
| Total NMPC ticks               | 459      | 543    | 458             |
| infeas + max-iter rate         | 0.020    | 0.061  | 0.011           |

The infeasibility rate is **misleadingly low** for the midwaypoint
run — the run aborts earlier (458 vs 543 ticks) so post-abort DS-
hold infeasibility events don't accumulate. The lower count
reflects shorter elapsed time, not better health. Per-second
infeasibility during the SS windows themselves is harder to
extract without phase-aligned sampling but is not the bottleneck
here — the failure is in the upstream reference shape and the
QP's inability to track it, not in the NMPC's solver health on
the references it does receive.

---

## §7  Verdict — does step 2 dock?

**No, and steps 0 and 1 stop docking too.**

Three observations:

1. **The IK output is well-conditioned at every step.**
   `w_worst_mid ∈ [3.98e-2, 8.88e-2]` for all three steps, all
   > the 1e-3 safety threshold by 1.5 orders of magnitude. The
   mid-waypoint IK produces *kinematically* good waypoints.

2. **The mid-waypoint optimiser has no trackability constraint.**
   Its cost is `min over τ of σ_min(Ja)·σ_min(Jb)`, evaluated
   along the piecewise quintic. It will pick *any* q_mid with
   high w_worst, including configurations that require the torso
   to traverse 0.78 m backwards in 3 s or jump 44 cm laterally
   in mid-SS. The closed-loop dynamics — payload momentum,
   AOCS authority, NMPC bandwidth — do not enter the cost.

3. **Cascading failure across steps.** Step 0's mis-track puts the
   live state off-nominal at step 1 entry; step 1's near-flip
   leaves a state that the pre-planner can't even build a
   feasible trajectory from for step 2. Per-step closed-loop
   sensitivity to reference shape is much higher than the
   diagnostic's standalone path measurements suggested.

The original H2 finding from `T15_step2_path_geometry.md` — that
the single-quintic reference between (3,3) and (3,4) visits a
near-singular interior — remains correct as a kinematic
observation. But fixing that interior with a kinematics-only
mid-waypoint cost makes things worse, because the cost ignores
the dynamics.

**Disposition.** Per the brief: "If step 2 still fails, the
failure mode has moved beyond what mid-waypoint reshape can fix,
and we'd need to scope full trajectory optimisation (Option C
from the diagnostic). Do NOT attempt further fixes in this
prompt. Report cleanly and stop."

The Option B implementation as committed (commits `a9ff933` …
`7878af6`) is left in place but **gated off by default**
(`use_mid_waypoint_reshape = False`). All 200/200 tests pass with
flags off. The IK-fix-only run remains the best closed-loop
behaviour to date.

**Recommendations for the next prompt's scoping (out of scope here):**

1. Add a path-length penalty to `manipulability_config_mid_waypoint`
   (e.g., quadratic in `||q_mid_torso − 0.5·(q_start_torso + q_end_torso)||`).
   This would discourage the backward-detour basins that destroyed
   step 0.
2. Move from waypoint optimisation to short-horizon trajectory
   optimisation (e.g., direct collocation over the SS) with an
   explicit cost on tracking error and torque envelope. This is
   Option C territory.
3. Re-examine the gait sequence itself — for 1% mass-ratio, the
   (3,3)→(3,4) anchor pair may simply be infeasible without a
   transit anchor. A scheduler change (multi-segment SS, virtual
   anchors) could route around the geometric problem rather than
   trying to squeeze through it.

**Stopping per the task brief.**
