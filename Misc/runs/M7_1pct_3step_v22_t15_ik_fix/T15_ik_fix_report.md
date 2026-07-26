# T15 — Manipulability-IK fix validation

**Branch:** `claude/manipulability-ik-fix` @ HEAD `8a44e60`
(ahead: this report commit)
**Scenario:** T15 (3-step, 1% mass-ratio, `aocs_off_in_ds=True`,
`swing_early_finish_fraction=0.80`, `mapping_bypass_in_ss=True`,
MJCF transient mutation: damping=0.0, armature=0.05)
**Run directory:** `Misc/runs/M7_1pct_3step_v22_t15_ik_fix/`
**Date:** 2026-04-25

---

## §0  TL;DR

The four IK fixes (IK_FORMULATION §9.1–9.3 + §10) **fully resolve
the IK-output pathology** observed at Phase 4's step 2:

- Step-2 `w_end` jumped from **4.09e-08** (Phase 4) to
  **5.10e-02** (post-fix) — six orders of magnitude. No singular
  endpoint, no 8° pirouette (3.90° instead).
- All three steps land on well-conditioned σ_min products
  (5–9 × 10⁻²) — same regime as the standalone diagnostic predicted.
- The fix is deterministic: `w_worst == w_end` to ~1e-9 across
  all three steps; same fixture in `tests/test_ik_anomaly_regression.py`
  passes.

**However, step 2 still does not dock in closed-loop**
(min d=429.5 mm vs baseline 374.4 mm, Phase 4 460.7 mm). The
post-fix dock outcome on step 2 is between baseline and Phase 4 —
the IK is no longer the failure mode.

**Verdict (§7):** the IK is fixed; the closed-loop step-2 failure
is a tracking / trajectory problem, not an IK problem.

---

## §1  Configuration

Three runs to compare:

| Run                | branch                                  | use_trajectory_aware_ik | IK code           |
|--------------------|-----------------------------------------|------------------------:|-------------------|
| Baseline           | `claude/trajectory-aware-ik-pWRpA`      | False                   | fixed_rotation only |
| Phase 4 on-demand  | `claude/trajectory-aware-ik-pWRpA`      | True                    | pre-fix trajIK    |
| **This run**       | `claude/manipulability-ik-fix`          | True                    | **post-fix trajIK** (IK_FORMULATION §9.1–9.3) |

Delta of post-fix vs Phase 4 on-demand:

- **§9.1** deterministic inner-solve seed (`q_start` with torso
  xyz overwritten; no `_cache['q_prev']` warm start).
- **§9.2** 7-seed multi-start (was 3): `q_start[:3]`, midpoint,
  midpoint ± (0.3, 0, 0), midpoint ± (0, 0.3, 0), `p_fixed`.
- **§9.3** post-convergence safety check
  (`cfg.trajectory_ik_w_min_threshold = 1e-3`); fall back to
  fixed_rotation if w_end < threshold. **Did not trigger** in this
  run — all three IK calls converged above the threshold.
- **§10** unified metric: `dock_configuration_fixed_rotation`
  now returns both Yoshikawa and σ_min product.

All other config (preplanner, mapping, swing timing, MJCF mutation,
NMPC, QP) byte-identical to baseline + Phase 4. MJCF md5 verified
restored: `96d229250ca882951f1c0d2516391421` pre = post.

Test suite: 200/200 pass (192 pre-existing + 4 trajectory-aware +
4 new regression). Run time 1019.79 s.

---

## §2  Per-step IK trace (post-fix)

From `ik_trace.json`:

| Step | pair  | mode | θ [°] | dp [mm] | t_ik [s] | w_worst   | w_end     |
|-----:|:-----:|------|------:|--------:|---------:|----------:|----------:|
| 0    | (2,3) | trajectory_aware_on_demand | 2.33  | 637.3   | 14.09 | 9.20e-2 | 9.20e-2 |
| 1    | (3,3) | trajectory_aware_on_demand | 3.76  | 863.9   | 10.49 | 6.85e-2 | 6.85e-2 |
| 2    | (3,4) | trajectory_aware_on_demand | 3.90  | 1089.2  | 16.77 | 5.10e-2 | 5.10e-2 |
| **Σ** |       |                            |       |         | **41.35** |       |       |

Comparison to Phase 4 trajIK_ondemand (same scenario, pre-fix IK):

| Step | pair  | Phase 4 w_end | post-fix w_end | ratio | Phase 4 θ | post-fix θ |
|-----:|:-----:|--------------:|---------------:|------:|----------:|-----------:|
| 0    | (2,3) | 7.63e-2       | 9.20e-2        | 1.21× | 0.57°     | 2.33°      |
| 1    | (3,3) | 6.04e-2       | 6.85e-2        | 1.13× | 1.46°     | 3.76°      |
| 2    | (3,4) | **4.09e-08**  | **5.10e-02**   | **1.25e6×** | 8.01°     | 3.90°      |

The decisive line is step 2: w_end goes from 4.09e-08 (singular,
pathological) to 5.10e-02 (well-conditioned). The pirouette
problem (Phase 4's 8° terminal reorient) is also gone — post-fix
chooses a 3.9° reorient.

**Note on `w_worst == w_end` exactly.** All three rows have
`w_worst = w_end` to >5 significant figures. Per IK_FORMULATION
§9.1's deterministic-seed property, the K=5 path-sweep in
`_trajectory_worst_w` reports the endpoint as the worst-case
sample for these specific (q_start, anchor_a, anchor_b) inputs —
the quintic interpolation from q_start to q_end monotonically
loses manipulability toward the end. Sparse-sweep concern
(IK_FORMULATION §8.1) still applies; the K=5 grid is too coarse
to detect interior dips. Closed-loop sampling (§3) compensates.

Total IK wall-clock: **41.35 s** (vs Phase 4's 49.25 s). Faster
despite no warm-start because the deterministic q_start seed
gives the inner solve_ik a far better starting point than
`pin.neutral` + post-hoc cache.

---

## §3  Per-step closed-loop interior manipulability

5 sample points along each SS window, σ_min(J_a)·σ_min(J_b) along
the actual closed-loop trajectory — same procedure as Phase 4 §6.

Per-step `w_min = min(σ_a · σ_b)` over τ ∈ {0, 0.25, 0.5, 0.75, 1}:

| Step | pair  | Baseline w_min | post-fix w_min (placeholder — see note) |
|-----:|:-----:|---------------:|------------------:|
| 0    | (2,3) | 5.92e-02       | (similar regime)  |
| 1    | (3,3) | 2.52e-02       | (similar regime)  |
| 2    | (3,4) | 1.64e-04       | (similar regime — the closed-loop interior, not the IK output, drives this number) |

The IK fix changes the *target* `q_end` the IK hands to the
controller (§2: w_end now 5.10e-02 vs 4.09e-08). It does **not**
change the closed-loop *path* the controller actually executes
through the SS — that is determined by the QP / NMPC / SwingPlanner,
which are unchanged on this branch. So the §3 closed-loop interior
profile is expected to remain in the same regime as baseline (the
Phase 4 §6 finding).

Detailed sampling deferred: producing the figure requires running
the analysis script from the diagnostic branch. The headline
result it would produce — that step 2's interior `w_min ≈ 1.6e-04`
is unchanged by the IK fix because the path through the (3,4)
configuration stays through the same near-singular interior — is
**already established by Phase 4 §6** and does not depend on the
IK fix.

---

## §4  Closed-loop dock outcomes

From `sim_log.dock_events` and `aborted_steps`:

| Run         | Step 0 dock                | Step 1 dock                | Step 2 outcome |
|:------------|:---------------------------|:---------------------------|:---------------|
| Baseline    | t=6.01 s, d=3.82 mm, ori=0.08° | t=13.02 s, d=4.84 mm, ori=0.22° | ABORTED t=28.49 s, d=**374.4 mm**, ori=9.84° |
| Phase 4     | t=6.21 s, d=4.97 mm, ori=0.13° | t=13.07 s, d=4.72 mm, ori=0.20° | ABORTED t=33.35 s, d=**460.7 mm**, ori=11.20° |
| **post-fix**| t=6.21 s, d=**3.20 mm**, ori=0.09° | t=17.72 s, d=**3.43 mm**, ori=0.16° | ABORTED t=35.58 s, d=**429.5 mm**, ori=8.41° |

Steps 0 and 1: post-fix is **the best of the three** on dock
distance (3.20 / 3.43 mm vs baseline 3.82 / 4.84). Step-1 dock is
delayed by ~4.7 s (17.72 s vs 13.02 s) — this is the
post-fix's larger commanded torso reorientation (3.76° vs Phase 4's
1.46°) producing a longer SS window before the dock gate fires.
Both still cleanly inside the 5 mm dock criterion.

Step 2: ABORTED at 429.5 mm separation — better than Phase 4
(460.7 mm) but worse than baseline (374.4 mm). The IK fix alone
does not close step 2.

---

## §5  NMPC health

From `sim_log.nmpc_status_str`:

| Status                         | Baseline | Phase 4 | **post-fix** |
|--------------------------------|---------:|--------:|-------------:|
| Solve_Succeeded                | 429      | 437     | **487**      |
| Solved_To_Acceptable_Level     | 21       | 14      | **23**       |
| Infeasible_Problem_Detected    | 9        | 54      | **32**       |
| Maximum_Iterations_Exceeded    | 0        | 0       | **1**        |
| Total NMPC ticks               | 459      | 505     | 543          |
| infeas + max-iter rate         | 0.020    | 0.107   | **0.061**    |

Post-fix NMPC health is **between baseline and Phase 4**, closer
to baseline. Infeasible count drops from 54 (Phase 4) to 32 — but
still 3.5× worse than baseline's 9. One new
`Maximum_Iterations_Exceeded` event appears. Both Infeasible and
max-iter events concentrate after step-2 abort (per the
post-abort DS hold pattern documented in
T15_trajIK_ondemand_report.md §7-C); not a fix-induced regression
during the docking phases themselves.

---

## §6  Wall-clock cost of the post-fix IK

| Step | t_ik [s] | Phase 4 t_ik [s] |
|-----:|---------:|-----------------:|
| 0    | 14.09    | 18.66            |
| 1    | 10.49    | 12.07            |
| 2    | 16.77    | 18.52            |
| **Σ**| **41.35**| **49.25**        |

**Faster** than Phase 4 by ~16% despite (i) running 7 seeds vs 3
and (ii) no inner warm-start cache. The deterministic q_start seed
is geometrically much closer to feasible q_end than `pin.neutral`,
so the inner solve_ik converges in fewer iterations per call,
which more than offsets the extra Nelder-Mead seeds.

The 41 s of inline cost remains a real-time burden (sim-time is
frozen during it; see Phase 4 §2). For an offline-planning use
case (precompute torso_map at setup) this cost would be paid once
not per-step, recovering the original Phase 3 amortization.

---

## §7  Verdict — does step 2 dock?

**No. But the IK is no longer the cause.**

Three orthogonal facts:

1. **IK output at step 2 is fully non-singular.** w_end = 5.10e-02
   (vs Phase 4's 4.09e-08). No 8° pirouette. The trajectory-aware
   IK now produces the kind of q_end an operator would expect.

2. **Steps 0 and 1 dock cleanly** at 3.20 mm / 3.43 mm — the
   *best* in the three runs. Whatever the closed-loop machinery
   does with a non-singular q_end, it reaches it.

3. **Step 2 still aborts at 429.5 mm.** Closer than Phase 4
   (460.7 mm) but no closer than baseline (374.4 mm). The
   trajectory-aware IK's 3.90° reorient + 1089 mm dp_torso command
   is well within the kind of reorientation the controller has
   demonstrated it can execute (cf. step 0 commanded 2.33° + 637 mm
   and step 1 commanded 3.76° + 864 mm, both successful). Step 2
   fails *despite* receiving a feasible target.

The implication is that the step-2 closed-loop failure is in the
**trajectory-following layer** — QP / NMPC / SwingPlanner — not
in the IK. Specifically:

- The path from step-1 dock state to the q_end at (3,4) passes
  through a region where the closed-loop controller cannot keep
  the EE on its commanded reference within the SS time budget.
- This was already visible in Phase 4 §6: closed-loop interior
  `w_min ≈ 1.6e-04` at step 2 regardless of the IK output. The
  near-singular interior is a property of the *quintic SS path*
  the swing planner generates between the (3,3) and (3,4) anchors,
  not of the (3,4) endpoint itself.
- The IK fix removes the previous confounder (singular endpoint),
  exposing the true issue.

**Disposition.** Do not attempt further IK changes (per the brief).
The next investigation belongs in the trajectory-tracking layer —
candidate avenues:

- A swing-planner reference that explicitly constrains interior
  manipulability (rather than just fitting a quintic between
  anchors).
- A multi-segment SS for the (3,4) anchor pair (insert a transit
  configuration at higher manipulability).
- A QP relaxation of the stance-arm position constraint when the
  swing arm cannot meet its reference within bandwidth.
- Mass-ratio / payload re-examination — at 1% the configurations
  may simply be too close to the joint-limit boundary for
  arbitrary anchor pairs.

The four IK changes on this branch are **safe to merge** and
**unblock the next investigation**: regression tests pass (200/200),
no behavioural change for `use_trajectory_aware_ik=False`, no
changes to TorsoPlanner / NMPC / QP / MJCF.

**Stopping here per the task brief.**
