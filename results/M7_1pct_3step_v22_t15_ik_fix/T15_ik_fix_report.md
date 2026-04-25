# T15 — Manipulability-IK fix validation

**Branch:** `claude/manipulability-ik-fix`  (HEAD pending; will be filled in)
**Scenario:** T15 (3-step, 1% mass-ratio, `aocs_off_in_ds=True`,
`swing_early_finish_fraction=0.80`, `mapping_bypass_in_ss=True`,
MJCF transient mutation: damping=0.0, armature=0.05)
**Run directory:** `results/M7_1pct_3step_v22_t15_ik_fix/`
**Date:** _filled when the run completes_

This document is **work in progress** — the §§ below are populated
from the live run as it produces output.

## §1  Configuration

Three runs to compare:

| Run                | branch                                  | use_trajectory_aware_ik | IK code           |
|--------------------|-----------------------------------------|------------------------:|-------------------|
| Baseline           | `claude/trajectory-aware-ik-pWRpA`      | False                   | fixed_rotation only |
| Phase 4 on-demand  | `claude/trajectory-aware-ik-pWRpA`      | True                    | pre-fix trajIK     |
| **This run**       | `claude/manipulability-ik-fix`          | True                    | **post-fix trajIK** (IK_FORMULATION §9.1–9.3) |

Delta of post-fix vs Phase 4 on-demand:

- §9.1: deterministic inner-solve seed (q_start; no `_cache['q_prev']`).
- §9.2: 7-seed multi-start (was 3).
- §9.3: post-convergence safety check (w_min_threshold = 1e-3); fallback to fixed_rotation if rejected.
- §10:  `dock_configuration_fixed_rotation` now reports both Yoshikawa and σ_min product.

All other config (preplanner, mapping, swing timing, MJCF mutation,
NMPC, QP) byte-identical to baseline + Phase 4.

## §2  Per-step IK trace (post-fix)

_To be populated from `results/M7_1pct_3step_v22_t15_ik_fix/ik_trace.json`._

| Step | pair  | mode | θ [°] | dp [mm] | t_ik [s] | w_worst | w_end | w_sigma_min_fixed |
|-----:|:-----:|------|------:|--------:|---------:|--------:|------:|------------------:|
| 0    | (2,3) |      |       |         |          |         |       |                   |
| 1    | (3,3) |      |       |         |          |         |       |                   |
| 2    | (3,4) |      |       |         |          |         |       |                   |

## §3  Per-step closed-loop interior manipulability

_5 sample points per SS, σ_min(J_a)·σ_min(J_b) along the actual
closed-loop path. Compare to baseline._

| Step | pair  | baseline w_min | post-fix w_min | ratio |
|-----:|:-----:|---------------:|---------------:|------:|
| 0    | (2,3) | 5.92e-2        |                |       |
| 1    | (3,3) | 2.52e-2        |                |       |
| 2    | (3,4) | 1.64e-4        |                |       |

## §4  Closed-loop dock outcomes

_Filled from `sim_log.json`'s `dock_events` and `aborted_steps`._

| Run        | Step 0 | Step 1 | Step 2 |
|------------|--------|--------|--------|
| Baseline   | DOCKED t=6.01 s, d=3.82 mm, ori=0.08° | DOCKED t=13.02 s, d=4.84 mm, ori=0.22° | ABORTED t=28.49 s, d=374.4 mm |
| Phase 4    | DOCKED t=6.21 s, d=4.97 mm, ori=0.13° | DOCKED t=13.07 s, d=4.72 mm, ori=0.20° | ABORTED t=33.35 s, d=460.7 mm |
| **post-fix** |        |        |        |

## §5  NMPC health

_Filled from `sim_log.nmpc_status_str` Counter._

| Status                         | Baseline | Phase 4 | post-fix |
|--------------------------------|---------:|--------:|---------:|
| Solve_Succeeded                | 429      | 437     |          |
| Solved_To_Acceptable_Level     | 21       | 14      |          |
| Infeasible_Problem_Detected    | 9        | 54      |          |
| Maximum_Iterations_Exceeded    | 0        | 0       |          |
| nmpc_solve_rate_50ms           | 0.963    | 0.780   |          |
| nmpc_infeasibility_rate        | 0.020    | 0.107   |          |

## §6  Wall-clock cost

_Trajectory-aware IK time per step. Phase 4 §2 totalled 49.25 s
across 3 calls. Per-call cost expected ~10–15 s post-fix (no
warm-start) per IK_FORMULATION §9.1._

| Step | t_ik [s]  | seed picked        |
|-----:|----------:|--------------------|
| 0    |           |                    |
| 1    |           |                    |
| 2    |           |                    |

## §7  Verdict

**Question:** does step 2 dock?

_To be answered when the run completes._
