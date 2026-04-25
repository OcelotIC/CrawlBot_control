# T15 — mid-waypoint reshape (Option B) validation

**Branch:** `claude/step2-path-diagnostic` @ HEAD `7878af6`+ this report.
**Scenario:** T15 (3-step, 1% mass-ratio, `aocs_off_in_ds=True`,
`swing_early_finish_fraction=0.80`, `mapping_bypass_in_ss=True`,
MJCF transient mutation: damping=0.0, armature=0.05).
**Run directory:** `results/M7_1pct_3step_v22_t15_midwaypoint/`
**Date:** _filled when the run completes_

WIP — populated from the live run as it produces output.

## §1 Configuration delta

| Run                | use_trajectory_aware_ik | use_path_feasibility_check | use_mid_waypoint_reshape | mid_waypoint_force_on |
|--------------------|------------------------:|---------------------------:|-------------------------:|----------------------:|
| Baseline (bug1fix) | False                   | False                      | False                    | False                 |
| Phase 4 on-demand  | True                    | False                      | False                    | False                 |
| IK-fix             | True                    | False                      | False                    | False                 |
| **midwaypoint (this)** | True                | True                       | True                     | True                  |

`force_on=True` because the runtime gate (`check_path_feasibility`)
operates on simplified planner refs (no M5 CoM-mapping) and was
shown in §5/§7 of `T15_step2_path_geometry.md` to underreport
relative to the actual mapped sim_log refs the diagnostic
measured. Forcing the mid-waypoint inserts the reshape on every
step, validating the closed-loop effect rather than gating on a
proxy.

All other config (preplanner, mapping, swing timing, MJCF
mutation, NMPC, QP) byte-identical to baseline + IK-fix.

## §2 Per-step IK trace (mid-waypoint output)

_To be populated from `ik_trace.json`._

| Step | pair  | mid_used | w_end (single-q) | w_worst_mid | q_mid torso xyz |
|-----:|:-----:|:--------:|-----------------:|------------:|-----------------|
| 0    | (2,3) |          |                  |             |                 |
| 1    | (3,3) |          |                  |             |                 |
| 2    | (3,4) |          |                  |             |                 |

## §3 Per-step path-feasibility metrics

_From the `[path-feasibility]` log lines._

| Step | feasible (gate) | gate w_min | gate τ | mid-waypoint inserted? |
|-----:|:---------------:|-----------:|-------:|:----------------------:|
| 0    |                 |            |        |                        |
| 1    |                 |            |        |                        |
| 2    |                 |            |        |                        |

## §4 Closed-loop dock outcomes

_From `sim_log.dock_events` and `aborted_steps`._

| Run            | Step 0 | Step 1 | Step 2 |
|:---------------|:-------|:-------|:-------|
| Baseline       | DOCKED t=6.01s d=3.82mm  | DOCKED t=13.02s d=4.84mm | ABORT t=28.49s d=374.4mm |
| Phase 4        | DOCKED t=6.21s d=4.97mm  | DOCKED t=13.07s d=4.72mm | ABORT t=33.35s d=460.7mm |
| IK-fix         | DOCKED t=6.21s d=3.20mm  | DOCKED t=17.72s d=3.43mm | ABORT t=35.58s d=429.5mm |
| **midwaypoint**|                          |                          |                          |

## §5 Closed-loop interior manipulability

_5 sample points per SS along the actual closed-loop trajectory._

| Step | pair  | Baseline w_min | IK-fix w_min | midwaypoint w_min |
|-----:|:-----:|---------------:|-------------:|------------------:|
| 0    | (2,3) | 5.92e-2        | (similar)    |                   |
| 1    | (3,3) | 2.52e-2        | (similar)    |                   |
| 2    | (3,4) | 1.64e-4        | (similar)    |                   |

## §6 NMPC health

_From `sim_log.nmpc_status_str`._

| Status                         | Baseline | IK-fix | midwaypoint |
|--------------------------------|---------:|-------:|------------:|
| Solve_Succeeded                | 429      | 487    |             |
| Solved_To_Acceptable_Level     | 21       | 23     |             |
| Infeasible_Problem_Detected    | 9        | 32     |             |
| Maximum_Iterations_Exceeded    | 0        | 1      |             |
| Total NMPC ticks               | 459      | 543    |             |
| infeas + max-iter rate         | 0.020    | 0.061  |             |

## §7 Verdict

_To be answered when the run completes._
