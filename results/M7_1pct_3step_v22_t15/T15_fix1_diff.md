# T15-fix-1 Phase 2 — Fix diff summary (Option Z)

**Branch.** `claude/t15-bug1-fix` (based on `origin/main` @ `4435c5d`).

**Scope.** One file touched: `crawlbot/simulation/sim_loop.py`.
Total additions: **+9 lines** (two insertions, no deletions, no
refactoring).

**Behaviour change.** `self._t_plan_offset` and the local mirror
`t_offset` are now reset at every SS entry to
`t_ss_start − plan.t_start[ss_phase_idx]`, independent of prior
DS-settle or early-dock slack. This restores the invariant

> `plan_query_t(t_ss_start_k) == plan.t_start[k]`

which the prior implementation violated whenever the previous
SS phase docked before its pre-planner-assigned `T_step` expired.

---

## 1. Edits

### 1.1 `crawlbot/simulation/sim_loop.py:914–918`

Inside `_setup_torso_for_step`, immediately after the existing
`plan.set_step_duration(ss_phase_idx, T_step)` call and the
`self._current_T_step = T_step` assignment:

```python
        self.plan.set_step_duration(ss_phase_idx, T_step)
        self._current_T_step = T_step

        # Option Z: reset plan-time offset at SS entry so that
        # plan_query_t(t_ss_start) aligns with plan.t_start[ss_phase_idx].
        # This absorbs both (a) unused SS runway from prior-step early
        # dock and (b) any inter-step DS-settle slack vs nominal DS
        # duration. Idempotent per SS entry.
        self._t_plan_offset = t_ss_start - self.plan.t_start[ss_phase_idx]
```

### 1.2 `crawlbot/simulation/sim_loop.py:1178–1179`

Inside the main loop, immediately after the
`self._setup_torso_for_step(...)` call returns, before the
`if not step_feasible:` branch:

```python
                    q_dock, T_step, step_feasible = self._setup_torso_for_step(
                        t_ss_start, swing_arm,
                        stance_a, stance_b, swing_arm, target_idx,
                        ss_phase_idx=ss_phase_idx)
                    # Mirror Option Z reset into the outer loop's offset.
                    t_offset = self._t_plan_offset
                    if not step_feasible:
```

---

## 2. Why the mirror in the caller is needed

The outer loop maintains a local `t_offset` float
(`sim_loop.py:1089`) that is incremented during DS settles
(`sim_loop.py:1138–1139`):

```python
t_offset += dt_ds_elapsed
self._t_plan_offset = t_offset
```

Without the mirror, `self._t_plan_offset` would be overwritten at
each SS entry (the new behaviour) but the next DS settle would
reconstruct `self._t_plan_offset` from the stale local `t_offset`,
re-introducing the desync. The mirror `t_offset =
self._t_plan_offset` keeps the two in sync at SS entry so that
subsequent `t_offset += dt_ds_elapsed` updates build on the reset
value.

---

## 3. Pre-run sanity

Numeric values at the first two SS entries in the T15 scenario
(with `set_step_duration` installing step 0's `T_step = 7.284 s`
and later step 1's `T_step = 7.929 s`):

| SS entry | `t_ss_start` (s) | `plan.t_start[ss_phase_idx]` (s) | `_t_plan_offset` after reset (s) |
|---|---:|---:|---:|
| step 0 (`ss_phase_idx = 1`) | 0.110 | 0.500 | **−0.390** |
| step 1 (`ss_phase_idx = 3`) | 6.520 | 8.284 | **−1.764** |
| step 2 (`ss_phase_idx = 5`) | 19.030 | 16.713 + T_step_1_unchanged_during_step_2_setup | (to be verified in Phase 3) |

Consequence at step-1 SS first tick (`t = 6.520 s`):

```
plan_query_t       = 6.520 − (−1.764) = 8.284   ← phase 3 t_start
plan_query_t_log   = 6.620 − (−1.764) = 8.384   ← phase 3 + 0.100 s
τ_phase3           = (8.384 − 8.284) / (0.8 · 7.929) = 0.0158
s_quintic(τ=0.0158) ≈ 0
p_ee_ref (logged)  ≈ anchors_a[2] = (−400, +300, +25) mm
```

i.e. the step-1 SS first-tick `p_ee_ref` is now at the
SwingPlanner's `p_start = anchors_a[2]`, not at step 0's dock
target `anchors_b[3]`. Expected `|p_ee_ref − p_ee_actual|` at
step-1 SS entry: **< 5 mm**, down from 995 mm.

Import sanity:

```
$ PYTHONPATH=. MUJOCO_GL=disabled python3 -c "import crawlbot.simulation.sim_loop; print('import OK')"
import OK
```

---

## 4. Full diff (verbatim)

```diff
diff --git a/crawlbot/simulation/sim_loop.py b/crawlbot/simulation/sim_loop.py
index 9ace30a..5f966eb 100644
--- a/crawlbot/simulation/sim_loop.py
+++ b/crawlbot/simulation/sim_loop.py
@@ -911,6 +911,13 @@ class SimulationLoop:
         self.plan.set_step_duration(ss_phase_idx, T_step)
         self._current_T_step = T_step
 
+        # Option Z: reset plan-time offset at SS entry so that
+        # plan_query_t(t_ss_start) aligns with plan.t_start[ss_phase_idx].
+        # This absorbs both (a) unused SS runway from prior-step early
+        # dock and (b) any inter-step DS-settle slack vs nominal DS
+        # duration. Idempotent per SS entry.
+        self._t_plan_offset = t_ss_start - self.plan.t_start[ss_phase_idx]
+
         # 4. Torso planner over the SAME [t_ss_start, t_ss_start + T_step].
         #    No torso_delay, no EXT extension.
         #    M7 change (B): the torso trajectory completes in
@@ -1168,6 +1175,8 @@ class SimulationLoop:
                         t_ss_start, swing_arm,
                         stance_a, stance_b, swing_arm, target_idx,
                         ss_phase_idx=ss_phase_idx)
+                    # Mirror Option Z reset into the outer loop's offset.
+                    t_offset = self._t_plan_offset
                     if not step_feasible:
                         log.aborted_steps.append({
                             'step_idx': int(step_idx),
```

---

## 5. What this fix does **not** change

- No change to `ContactScheduler.plan_traversal` or
  `GaitPlan.set_step_duration`.
- No change to how `set_step_duration` is called on the upcoming
  SS phase.
- No change to the DS-settle logic (`t += dt_ds_elapsed;
  t_offset += dt_ds_elapsed`) — the DS settles still advance both
  `t` and `t_offset` in lockstep, because the SS-entry reset
  supersedes whatever value they hold.
- No new config fields, no new logging fields, no new import.

*(End of Phase 2 diff summary. Commit next; Phase 3 — simulation
rerun — awaits human review.)*
