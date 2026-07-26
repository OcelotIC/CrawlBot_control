# T15-fix-1 Phase 1 — Timeline desync diagnosis

**Scope.** Read-only localisation of the mechanism that makes
`SwingPlanner.reference_at` return step-0's trajectory/fallback
during the first 2.3 s of step 1's SS window.

**Branch.** `claude/t15-bug1-fix` (based on `origin/main` @
`4435c5d`). No code edits in this phase.

---

## 1. `t_plan_offset` update site (and ± 10 surrounding lines)

File: `crawlbot/simulation/sim_loop.py`. The cumulative plan-time
offset is maintained by two sites:

### 1.1 Initialisation (start of main loop)

`sim_loop.py:1086–1090`:

```python
phases = plan.phases
step_idx = 0
i = 0
t_offset = 0.0   # Cumulative time offset from inter-step settling
self._t_plan_offset = 0.0  # mirror for _step's swing-planner query
```

### 1.2 Increment during DS settle (the only mutation site after init)

`sim_loop.py:1133–1145` (update is lines 1137–1139):

```python
                        min_steps=min_steps_ds,
                        fallback_Kd=cfg.Kd_settle_damping,
                    )
                    dt_ds_elapsed = ds_result['n_steps'] * cfg.dt_qp
                    t += dt_ds_elapsed
                    t_offset += dt_ds_elapsed
                    self._t_plan_offset = t_offset
                    log.inter_step_settles.append({
                        'step_idx': int(step_idx),
                        't_start': float(t_ds_start_wall),
                        't_end': float(t),
                        'n_steps': int(ds_result['n_steps']),
```

### 1.3 Declared contract

`sim_loop.py:140–147` (comment block on `self._t_plan_offset`):

```
# Cumulative plan-time offset from inter-step settling. The sim
# clock `t` advances with settle time, but the ContactScheduler
# plan's t_start fields are frozen at the nominal plan times.
# SwingPlanner queries the plan by time via plan.phase_at(t), so
# it must be fed `t - _t_plan_offset` to stay in sync. The torso
# planner and coarse pre-planner already receive offset-adjusted
# times when they are set up per-step, so they use `t` directly.
```

`t_plan_offset` is incremented **only** by `dt_ds_elapsed` during
DS settles. There is no site that advances it in response to dock
events, SS early-finish, or SS abort.

---

## 2. All `plan.set_step_duration(...)` call sites in `sim_loop.py`

Single call site.

### 2.1 `sim_loop.py:905–912`

```python
        # 3. Install T_step in the scheduler's SS phase. This updates
        #    GaitPhase.duration and cascades t_start/t_end for all
        #    subsequent phases, so the SwingPlanner (which reads
        #    gp.duration in reference_at) plans over [0, T_step] —
        #    identical to the torso planner's horizon below.
        self.plan.set_step_duration(ss_phase_idx, T_step)
        self._current_T_step = T_step
```

The call lives inside `_setup_torso_for_step(…, ss_phase_idx)`
(sim_loop.py:794–951), which is invoked once per step at
`sim_loop.py:1167–1170`:

```python
                    q_dock, T_step, step_feasible = self._setup_torso_for_step(
                        t_ss_start, swing_arm,
                        stance_a, stance_b, swing_arm, target_idx,
                        ss_phase_idx=ss_phase_idx)
```

Per call: updates one phase's duration (`ss_phase_idx` = the SS
phase about to begin), triggers cascaded recompute of
`t_start`/`t_end` for all phases via
`contact_scheduler.GaitPlan.set_step_duration`
(`contact_scheduler.py:135–160`).

### 2.2 Per-step table (T15 run)

| step_idx | ss_phase_idx | T_step installed (s) | sim-time at install | scheduler phase mutated |
|---:|---:|---:|---:|---|
| 0 | 1 | 7.284 | 0.110 s | phase 1 SS (step 0 SS) |
| 1 | 3 | 7.929 | 6.520 s | phase 3 SS (step 1 SS) |
| 2 | 5 | 9.299 | 19.030 s | phase 5 SS (step 2 SS) |

`set_step_duration` is never called for a completed or in-progress
phase. In particular, at step 0's dock instant (`t = 6.01 s`) no
call fires: phase 1's duration stays at the pre-planner-derived
**7.284 s** even though the step docked at SS-elapsed-plan-time
**5.400 s**.

---

## 3. `plan_query_t = t − t_plan_offset` at three instants

### 3.1 Plan state and timeline

After `set_step_duration` for step 0 (fired at `t = 0.110 s`, see
§2.2), before `set_step_duration` for step 1:

| phase | kind | duration (s) | `t_start` (s, plan) | `t_end` (s, plan) |
|---:|---|---:|---:|---:|
| 0 | DS | 0.500 | 0.000 | 0.500 |
| 1 | SS (step 0) | **7.284** | 0.500 | **7.784** |
| 2 | DS | 0.500 | 7.784 | 8.284 |
| 3 | SS (step 1, stub) | 0.000 | 8.284 | 8.284 |
| 4 | DS | 0.500 | 8.284 | 8.784 |
| 5 | SS (step 2, stub) | 0.000 | 8.784 | 8.784 |
| 6 | DS trailing | 0.500 | 8.784 | 9.284 |

After `set_step_duration` for step 1 (fired at `t = 6.520 s`):

| phase | kind | duration (s) | `t_start` (s, plan) | `t_end` (s, plan) |
|---:|---|---:|---:|---:|
| 0 | DS | 0.500 | 0.000 | 0.500 |
| 1 | SS (step 0) | **7.284** | 0.500 | **7.784** |
| 2 | DS | 0.500 | 7.784 | 8.284 |
| 3 | SS (step 1) | **7.929** | 8.284 | **16.213** |
| 4 | DS | 0.500 | 16.213 | 16.713 |
| 5 | SS (step 2, stub) | 0.000 | 16.713 | 16.713 |
| 6 | DS trailing | 0.500 | 16.713 | 17.213 |

### 3.2 `t_plan_offset` budget at the three instants

From the T15 `log.inter_step_settles`:

| settle | step_idx | t_start (sim) | t_end (sim) | n_steps | dt_ds_elapsed |
|---|---:|---:|---:|---:|---:|
| [0] | 0 | 0.000 | 0.110 | 11 | 0.110 |
| [1] | 1 | 6.010 | 6.520 | 51 | 0.510 |

Therefore `t_plan_offset` at each instant:

| instant | `t_plan_offset` (s) | why |
|---|---:|---|
| t = 5.910 s (k = 58, last SS tick of step 0) | 0.110 | only settle [0] has fired |
| t = 6.010 s (step 0 dock) | 0.110 | settle [1] starts **after** the dock |
| t = 6.520 s (k = 59, first SS tick of step 1) | 0.620 | settle [1] added +0.510 |

### 3.3 `plan_query_t` trace

| instant | `t_sim` | `t_plan_offset` | `plan_query_t` | `phase_at(plan_query_t)` | comment |
|---|---:|---:|---:|---|---|
| last SS tick of step 0 | 5.910 | 0.110 | **5.800** | phase 1 (SS, swing_arm='b') | correct — still in step 0 SS |
| step 0 dock | 6.010 | 0.110 | **5.900** | phase 1 (SS, swing_arm='b') | plan-time phase 1 is only 5.400 s in; still 1.884 s of plan phase 1 remains |
| first SS tick of step 1 | 6.520 | 0.620 | **5.900** | phase 1 (SS, swing_arm='b') | **unchanged from the dock instant** — the DS settle advanced `t` and `t_plan_offset` in lockstep by 0.510 s, so `plan_query_t` is invariant under DS settles |

### 3.4 Logging-path offset (swing_planner query at `t_log`)

`sim_loop.py:2058`: `t_log = t + cfg.dt_nmpc`.
`sim_loop.py:2174`: `sr_f = self.swing_planner.reference_at(t_log − self._t_plan_offset)`.
`sim_loop.py:2175`: `log.p_ee_ref.append(sr_f.p_ee.copy())`.

At the logged tick `k = 59` (`log.t = 6.520`), the
swing-planner query uses `t_log = 6.620`, so
`plan_query_t_log = 6.620 − 0.620 = 6.000 s → phase 1 SS`, with
`τ_phase1 = (6.000 − 0.500) / (0.8 · 7.284) = 0.9439`, `s_quintic
≈ 0.9985`. Evaluated trajectory:

`p_ee_ref ≈ anchors_b[2] + 0.9985 · (anchors_b[3] − anchors_b[2]) +
(clearance · n · bump ≈ small)`
`           ≈ (−400 + 798.8, −300, 25 − 0.9) ≈ (+398.8, −300, +24.1)`

Observed: `log.p_ee_ref[59] = (+398.70, −300.00, +24.07) mm`.
Match (the trajectory-evaluation branch of `reference_at`, not the
DOUBLE fallback).

### 3.5 When does `plan_query_t` cross into phase 3 (step 1 SS)?

Target plan-time: 8.284 s (start of phase 3 after step-1 setup).
Using `plan_query_t_log = t_log − 0.620 = t + 0.100 − 0.620 =
t − 0.520`. Solve `t − 0.520 = 8.284` → `t = 8.804 s`.

| k | `log.t` | `t_log` | `plan_query_t_log` | phase | observed `log.p_ee_ref` |
|---:|---:|---:|---:|---|---|
| 81 | 8.720 | 8.820 | 8.200 | phase 2 DS | `(+400, −300, +25)` ≡ `anchors_b[3]` (via `_last_swing_position`) |
| 82 | 8.820 | 8.920 | 8.300 | phase 3 SS (step 1, swing_arm='a') | `(−400.0, +300.0, +25.0)` ≡ `anchors_a[2]` (`τ = 0.0025`, `s ≈ 0`) |

The transition is consistent with the plan-time crossing of
`phase 2 → phase 3` at plan-time 8.284, i.e. sim-time 8.804 s.
Time from step-1 SS entry (`t = 6.520 s`) to transition
(`t ≈ 8.820 s`) = **2.300 s** — matches the empirical lag reported
in T15_post1_mechanism §Q1.5.

---

## 4. Root-cause classification

Per the prompt's four candidate labels:

| label | description | verdict |
|---|---|---|
| (a) | `t_plan_offset` not incremented on dock — planner's clock fails to absorb the "early dock" slack | **TRUE**: `t_plan_offset` is updated only by `dt_ds_elapsed` during DS settles (§1.2); no update at dock. At step 0 dock, plan-phase 1 has 1.884 s of remaining nominal duration that is never absorbed. |
| (b) | `plan.set_step_duration` not called at dock to truncate the completed SS phase | **TRUE**: the single `set_step_duration` call site (§2.1) runs once per step at SS *setup* time and targets the *upcoming* SS phase. Phase 1 is never retro-truncated at step-0 dock. |
| (c) | both | **chosen** — (a) and (b) are two equivalent surface forms of one underlying invariant violation (see §5). |
| (d) | something else | not selected. The plan-time trace (§3.3) maps the 995 mm `p_ee_ref` at step 1 SS entry directly to "phase_at returns phase 1 SS at step 1 SS entry", which is explained entirely by (a) + (b). |

### 4.1 Verdict: (c) both

`plan.phase_at(plan_query_t)` at step 1 SS entry resolves to phase
1 SS (step 0's trajectory) because **neither** of the two possible
reconciliation paths fires at step 0's dock instant. Fixing either
one alone resolves the symptom; they represent the same
underlying invariant.

### 4.2 The underlying invariant (what either fix must restore)

Let `plan_query_t(t_sim) = t_sim − t_plan_offset`. The intended
contract of the plan is:

*When the sim is executing SS phase `k` at sim-time `t_sim`, and
that SS phase is about to begin (`t_sim = t_ss_start_k`), the
query `plan_query_t(t_ss_start_k)` must land at or just past
`plan.t_start[k]` (phase `k`'s plan-time start).*

For step 0 this holds trivially because `plan.t_start[1]` (=0.5)
and `t_ss_start_0 − t_plan_offset_0 = 0.110 − 0.110 = 0.000` —
off by 0.5 s of the initial DS's nominal duration, but the
observed tick-1 plan-query-time (5.8 s at `t = 5.91 s`) falls
inside phase 1. The invariant holds (weakly) because phase 1 has
7.284 s of runway.

For step 1 the invariant is violated: `t_ss_start_1 −
t_plan_offset_1 = 6.520 − 0.620 = 5.900`, but
`plan.t_start[3] = 8.284`. Gap = **2.384 s**, which is almost
exactly `(7.284 − 5.400) + (0.500 − 0.510) ≈ 1.884 − 0.010 ≈
1.874 s` of unabsorbed step-0 SS slack plus a ~0.010 s
over-absorption of DS settle[1] (its actual duration 0.510 s
exceeded the nominal 0.500 s).

### 4.3 Two equivalent fix targets (design options restated)

Both fixes land the same invariant. Option X operates on
`t_plan_offset`; Option Y operates on `plan.phases[ss_phase].duration`.

**Option X (adjust `t_plan_offset` on dock).** At step 0's dock
instant, advance `t_plan_offset` so that `plan_query_t` snaps to
`plan.t_end[1]` (the end of step 0's SS phase). Increment:

```
Δ_X = plan.t_end[ss_phase_idx_of_just_completed_step] − plan_query_t_at_dock
    = 7.784 − 5.900 = 1.884 s
```

Then `t_plan_offset` becomes `0.110 + 1.884 = 1.994`; after DS
settle[1] (+0.510) it is `2.504`; at `t = 6.520` the
`plan_query_t = 6.520 − 2.504 = 4.016`. This is too small
(still ≥ 0.5 inside phase 1) — so Option X requires a second
adjustment during the DS settle to account for the difference
between actual DS elapsed and plan nominal DS width. Net: two
coupled adjustments, fragile.

**Option Y (truncate `plan.phases[ss_phase].duration` on dock).**
At step 0's dock instant, call

```
plan.set_step_duration(ss_phase_idx_just_completed,
                       plan_query_t_at_dock − plan.t_start[ss_phase_idx_just_completed])
                     = 5.900 − 0.500 = 5.400
```

so phase 1's duration becomes 5.400 s. Timeline cascades:

| phase | new `t_start` | new `t_end` | duration |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.500 | 0.500 |
| 1 | 0.500 | 5.900 | **5.400** |
| 2 | 5.900 | 6.400 | 0.500 |
| 3 (stub) | 6.400 | 6.400 | 0.000 |

Then during DS settle[1] the existing `t_offset += dt_ds_elapsed`
update pushes `t_plan_offset` from 0.110 → 0.620. At `t =
6.520 s`, `plan_query_t = 5.900` — now at **phase 2 DS start**,
not mid-phase-1.

This is still one residual step off: plan-time needs to be at
phase 3's start (6.400 s) at SS entry, but is at phase 2's start
(5.900 s). Gap = 0.5 s (= nominal DS width).

Equivalently: during the DS settle, `t_offset` should advance by
`dt_ds_elapsed − nominal_DS_width` (i.e. only the *slack* over
the plan's DS nominal), not by the full `dt_ds_elapsed`. With
phase 1 truncated (Option Y) *and* the DS-settle increment
changed from `+= dt_ds_elapsed` to `+= dt_ds_elapsed − phase_ds.duration`,
the invariant holds exactly.

### 4.4 Conclusion for Phase 1

- Root cause is **both (a) and (b)**: an invariant violation that
  can be repaired at either site; (a) and (b) are redundant
  descriptions.
- The minimal consistent fix is **Option Y + adjusted DS-settle
  increment**:
  1. At step 0's dock instant, call
     `plan.set_step_duration(ss_phase_idx, plan_query_t_at_dock −
     plan.t_start[ss_phase_idx])` to truncate the just-completed
     SS phase to its actual plan-time length.
  2. Change `t_offset += dt_ds_elapsed` (sim_loop.py:1138) to
     `t_offset += dt_ds_elapsed − nominal_DS_width` so the DS
     settle advances `plan_query_t` by exactly the plan's nominal
     DS duration.
- An alternative that avoids (2) is to instead, at SS *entry*
  (step 1's `_setup_torso_for_step` moment), set
  `self._t_plan_offset = t_ss_start − plan.t_start[ss_phase_idx]`
  directly. This closed-form reset at SS entry subsumes (1) and
  (2) into a single assignment and does not require plan-timeline
  mutation at dock. `plan.t_start[ss_phase_idx]` at that moment
  is already well-defined (phase 3's `t_start` = post-step-0
  cascade end of phase 2 = 8.284 s in the current behaviour; with
  Option Y applied, = 6.400 s).

A concrete fix recommendation (Option Y preferred over Option X)
is deferred to Phase 2. The diagnosis is sufficient to proceed.

---

## 5. Summary

| item | finding |
|---|---|
| `t_plan_offset` update site | `sim_loop.py:1138–1139` (DS-settle-only) |
| `plan.set_step_duration` call site | `sim_loop.py:911` (SS-setup-only, for upcoming phase) |
| `plan_query_t` at step 0 dock | 5.900 s (phase 1 SS, mid-trajectory) |
| `plan_query_t` at step 1 SS entry (main loop) | 5.900 s (phase 1 SS, mid-trajectory — identical to dock) |
| `plan_query_t_log` at step 1 SS entry (logging) | 6.000 s (phase 1 SS, `τ ≈ 0.944`, s ≈ 0.9985) |
| Observed `log.p_ee_ref[59]` | (+398.70, −300.00, +24.07) mm = phase-1 SS eval, not DOUBLE fallback |
| Transition to phase 3 (step 1 SS trajectory) | sim `t ≈ 8.804 s`, i.e. `k = 82` in `log.p_ee_ref` (2.300 s after SS entry) |
| Root cause classification | **(c) both**: `t_plan_offset` not advanced on dock AND `set_step_duration` not called to truncate completed SS phase |
| Recommended fix | Option Y (`set_step_duration` at dock to truncate completed SS) + `t_offset += dt_ds_elapsed − nominal_DS_width` during DS settle. Alternative: single-assignment reset of `_t_plan_offset` at SS entry. |

*(End of Phase 1 diagnosis. Phase 2 — implementation — awaits human review.)*
