# T12-fix — Option A implementation diff

Spec: `docs/architecture/M7_T12_MEMO.md` §5 (Option A — smooth the
post-dock position-reference release).

## Files modified
- `crawlbot/simulation/config.py` — S1 (new config field).
- `crawlbot/simulation/sim_loop.py` — S2 (init + weld-time capture),
  S3 (DS blend in reference selection), S4 (reset at SS entry).

## Deviation from spec (§5.7)

S2 requested capturing **both** `_ds_ramp_p_start` and `_ds_ramp_p_end`
at weld activation. `_ds_ramp_p_end` is **not** captured.

Reason: at weld scope the required `a_com_ff` argument to
`mapping.compute(r_com_ref, v_com_ref, a_com_ff, q_current, dq_current)`
has no clean source.
- The SS-path call at `sim_loop.py:1716–1718` supplies
  `af_for_mapping = af` where `af = self.nmpc.compute_feedforward_acceleration(lr)`
  is a local in `_step()` and not visible from the outer loop.
- `TorsoPlanner` does not expose CoM acceleration; its public API on
  CoM is `com_reference_at(t) -> ComReference(r_com, v_com)` only.
- Passing a substituted value (e.g. `np.zeros(3)`) would be a guess.

Per §5.7 ("Do not guess arguments") the capture is skipped rather than
hacked around. `_ds_ramp_p_end` is left as `None` on the sim object.
This is functionally inert: the S3 blend in `_step()` recomputes the
live mapping output `r_b_ref_m` each tick and never reads
`_ds_ramp_p_end`. The attribute is still defined (init + SS reset) so
future diagnostics can populate it if a clean source becomes available.

## Diffs (3-line context)

### `crawlbot/simulation/config.py`
```
@@ -233,3 +233,14 @@ class SimConfig:
     # for the linear components of the torso reference. Angular
     # reference still comes from TorsoPlanner (orientation tracking
     # unchanged). DS phase is unchanged (mapping still active there).
+
+    ds_ramp_duration_s: float = 2.0
+    # Option A (2026-04-22): duration over which the torso linear
+    # position reference is ramped from the SS-exit pose
+    # (_ss_entry_p_torso) to the live DS mapping output
+    # (mapping.compute(q_current)) after weld activation. Quintic
+    # shape function s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5, C^2
+    # continuous with s(0)=0, s(1)=1, s'(0)=s'(1)=s''(0)=s''(1)=0.
+    # Set to 0.0 to disable (reverts to the pre-Option-A step
+    # behavior). Introduced to close the T12 DS1 divergence;
+    # see docs/architecture/M7_T12_MEMO.md §5.
```
**One-line summary.** Adds `ds_ramp_duration_s: float = 2.0` with
inline comments matching the style of `mapping_bypass_in_ss`
immediately above it.

### `crawlbot/simulation/sim_loop.py`

Site 1 — `__init__`, after `_ss_entry_p_torso` declaration:
```
@@ -101,6 +101,13 @@ class SimulationLoop:
         # (set in _setup_torso_for_step). Read by _step() when
         # cfg.mapping_bypass_in_ss is True; otherwise unused.
         self._ss_entry_p_torso: Optional[np.ndarray] = None
+        # Option A (T12 fix, 2026-04-22): post-dock blend state for
+        # the DS torso position reference. Populated at weld
+        # activation; cleared on SS entry. See
+        # cfg.ds_ramp_duration_s and docs/architecture/M7_T12_MEMO.md §5.
+        self._ds_ramp_t_start: Optional[float] = None
+        self._ds_ramp_p_start: Optional[np.ndarray] = None
+        self._ds_ramp_p_end: Optional[np.ndarray] = None
         # Simulation time at which the active coarse plan was anchored
         # (so r_com_at(t - t0) gives the right reference at current time).
         self._coarse_plan_t0: float = 0.0
```
**One-line summary.** Declares three blend-state attributes alongside
the existing `_ss_entry_p_torso`, all initialised to `None`.

Site 2 — SS entry, after `_ss_entry_p_torso = p_t0.copy()`:
```
@@ -937,6 +944,11 @@ class SimulationLoop:
         # the live state above (line ~810), so it equals the actual
         # torso position at the moment SS begins.
         self._ss_entry_p_torso = p_t0.copy()
+        # Option A: reset DS ramp state at SS entry; the next weld
+        # activation will repopulate _ds_ramp_t_start / _ds_ramp_p_start.
+        self._ds_ramp_t_start = None
+        self._ds_ramp_p_start = None
+        self._ds_ramp_p_end = None

         return (q_end, T_step, True)
```
**One-line summary.** Clears the three blend-state attributes at every
SS entry so the next dock captures fresh endpoints (S4).

Site 3 — Weld activation, after `nmpc.reset_warm_start()`:
```
@@ -1326,6 +1338,16 @@ class SimulationLoop:
                         self._activate_weld(swing_arm, target_idx)
                         mujoco.mj_forward(self.mj_model, self.mj_data)
                         self.nmpc.reset_warm_start()
+                        # Option A: capture the SS-exit torso position
+                        # and weld time for the post-dock DS blend. The
+                        # blend endpoint (_ds_ramp_p_end) is not stored
+                        # here — _step() recomputes the live mapping
+                        # output each tick and blends it against
+                        # _ds_ramp_p_start. See M7_T12_MEMO.md §5.
+                        self._ds_ramp_t_start = float(t)
+                        if self._ss_entry_p_torso is not None:
+                            self._ds_ramp_p_start = (
+                                self._ss_entry_p_torso.copy())

                         # Inelastic impact: project velocity onto new
                         # constraint manifold.
```
**One-line summary.** At weld activation captures `_ds_ramp_t_start = t`
and `_ds_ramp_p_start = _ss_entry_p_torso.copy()`; `_ds_ramp_p_end`
deliberately not captured (see "Deviation from spec" above).

Site 4 — Reference selection (DS blend), inside the
`elif phase in ('SS', 'DS') and self.mapping is not None and cfg.use_m2_stack:`
branch, after the existing `mapping.compute(...)` call:
```
@@ -1716,6 +1738,25 @@ class SimulationLoop:
                 r_b_ref_m, v_b_ref_m, a_b_ff_m, _ = self.mapping.compute(
                     r_com_ref=rp_interp, v_com_ref=vp_interp,
                     a_com_ff=af_for_mapping, q_current=q_map, dq_current=dq_map)
+                # Option A: post-dock blend of the DS torso linear
+                # position reference from the SS-exit pose to the live
+                # mapping output over cfg.ds_ramp_duration_s. Quintic
+                # shape s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5.
+                # Orientation reference (tr.R below) is not blended.
+                if phase == 'DS':
+                    T_ramp = cfg.ds_ramp_duration_s
+                    if (T_ramp > 0.0
+                            and self._ds_ramp_t_start is not None
+                            and self._ds_ramp_p_start is not None):
+                        tau_blend = (tq - self._ds_ramp_t_start) / T_ramp
+                        if tau_blend <= 0.0:
+                            r_b_ref_m = self._ds_ramp_p_start.copy()
+                        elif tau_blend < 1.0:
+                            s_blend = (10.0 * tau_blend ** 3
+                                       - 15.0 * tau_blend ** 4
+                                       + 6.0 * tau_blend ** 5)
+                            r_b_ref_m = ((1.0 - s_blend) * self._ds_ramp_p_start
+                                         + s_blend * r_b_ref_m)
                 p_torso_ref_used = r_b_ref_m
                 v_torso_ref_used = np.concatenate([v_b_ref_m, tr.v[3:6]])
                 a_torso_ff_used = np.concatenate([a_b_ff_m, tr.a[3:6]])
```
**One-line summary.** In the DS branch only, blends the mapping's
`r_b_ref_m` with the frozen `_ds_ramp_p_start` via the quintic
`s(tau) = 10τ³ − 15τ⁴ + 6τ⁵`; when `tau_blend ≥ 1.0` the mapping
output passes through unmodified. `v_b_ref_m`, `a_b_ff_m`, and the
orientation components from `tr` are not touched. SS branch and
`mapping_bypass_in_ss = True` SS branch are unchanged. Uses `tq` (the
QP sub-tick time, defined at `sim_loop.py:1659`), not `self.t`.

## Import-test results

```
$ PYTHONPATH=. python3 -c "import crawlbot.simulation.config; print('config OK')"
config OK
$ PYTHONPATH=. python3 -c "import crawlbot.simulation.sim_loop; print('sim_loop OK')"
sim_loop OK
```

Both modules import cleanly with no syntax or binding errors.

## What is NOT in scope of this diff

Per §5 prohibitions of the implementation task:
- No commits, no pushes.
- No simulation runs.
- No edits to any file outside `config.py` and `sim_loop.py`.
- No logging of the new ramp state.
- No assertions.
- No `try/except` around the blend logic.
- No refactoring of adjacent code.

## Stop

Diff summary written. Awaiting human review before T12-fix simulation.
