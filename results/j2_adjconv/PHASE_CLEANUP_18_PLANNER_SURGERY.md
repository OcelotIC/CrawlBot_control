# PHASE CLEANUP-18 — planner surgery (steps 3–5 of the CLEANUP-16 plan)

Executes the remainder of the `crawlbot/planning/` removal plan. Steps 3 and 4 landed.
**Step 5 was executed and then reverted on measurement** — see §3, the one substantive
finding of this pass.

Gate-verified byte-identical, and — new this pass — the physical canonical result is now
reported explicitly rather than inferred from the hash (§4).

---

## 1. What was removed

| file | HEAD | now | Δ |
|---|---|---|---|
| `swing_planner.py` | 615 | 337 | **−278** |
| `torso_planner.py` | 586 | 480 | **−106** |
| `sim_loop.py` | 3390 | 3386 | −4 |

### Step 3 — the SwingPlanner phase-override mechanism (one coherent unit)

`add_phase`, `_override_reference_at`, `clear_phase_overrides`, the `_phase_overrides` list,
and the dispatch loop at the head of `reference_at`. Zero production callers since CLEANUP-15
removed the mid-waypoint reshape. `reference_at()` now falls straight through to the
scheduler-driven gait plan — the path the canonical has always taken. The `clear_phase_overrides()`
call in `sim_loop` went with it.

Note the name collision that survives: **`torso_planner.add_phase` is live** (`sim_loop:1544`)
and is untouched. Only the *swing* planner's `add_phase` was dead.

### Step 4 — methods orphaned by earlier passes

| method | why dead |
|---|---|
| `SwingPlanner.adaptive_reference_at` | zero callers anywhere in `crawlbot/`, `scripts/`, `tests/` |
| `SwingPlanner.swing_trajectory` | only caller is `Misc/scripts/test_integration.py` (legacy) |
| `TorsoPlanner.set_from_waypoints` | orphaned by CLEANUP-14 (the `ds_mobile_com_magnitude` block) |
| `TorsoPlanner._trapezoidal_params` | zero callers, including internally |

Residue cleared: the orphaned `Optional` / `GaitPhase` imports in `swing_planner`, and a
`sim_loop:3174` comment still citing `set_from_waypoints`.

### Post-removal orphan scan

Cleanup makes its own dead code — that is how `set_from_waypoints` became removable in the
first place. Re-scanned both classes afterwards for methods whose only callers had just been
deleted: **none**. Every remaining `SwingPlanner` / `TorsoPlanner` method has a live caller.

---

## 2. A tooling bug worth recording

The first surgery script cut method spans by scanning for the next same-indent `def`. That is
wrong for a **multi-line signature** whose closing `) -> None:` sits at the method's own indent —
the span cut mid-signature and produced `SyntaxError: unmatched ')'`. The `ast.parse` guard
before write caught it; nothing was written. Spans are now taken from the AST
(`node.lineno` / `node.end_lineno`), which is correct by construction.

Keep the parse-guard-before-write pattern. It is the reason this was a non-event.

---

## 3. Step 5 REVERTED — `locomotion_planner.py` is kept

CLEANUP-16 ranked step 5 as *"delete `locomotion_planner.py` (205 lines) — low risk, but breaks
2 legacy scripts"*, and asserted those scripts were already non-functional per
`CLEANUP_CARRYOVER` §C3. **Both halves of that were wrong**, and measurement — not reading —
is what showed it.

Import-resolving every `crawlbot.*` symbol of each consumer against HEAD:

| consumer | status at HEAD |
|---|---|
| `Misc/scripts/test_integration.py` | **all imports resolve** |
| `Misc/scripts/sim_torso6d.py` | **all imports resolve** |
| **`Misc/lutze_baseline/sim_lutze.py`** | **all imports resolve** |

There are **three** consumers, not two; none was on the §C3 list; and none was already broken.

The decisive one is the third. `Misc/lutze_baseline/` is a *package*, not a research script, and it
carries the **M0 / Lutze comparison baseline** — the one backing the paper's §II differentiation
table against Lutze [2023]. `LocomotionPlanner` is load-bearing there: constructed at
`sim_lutze.py:175`, calibrated at `:176`, and evaluated at `:231` and `:266`.

So this is the same KEEP class the CLEANUP-16 audit itself applied to `sequence_loader.py`:
**unused on the canonical ≠ retired research.** Deleting it would have traded 205 lines for a
broken paper baseline. Reverted; `__init__` export restored.

Recorded in `CLEANUP_CARRYOVER` §C5. Revisiting it is not a code question — it is "is the Lutze
baseline still to be re-run", which is Idriss's call.

**Method note.** The audit's error was inferring a script's health from a plausible-sounding
memory of a list. The same class of error as the CLEANUP-2 F1 retraction (a dataclass default
taken for the canonical value). Both are cured the same way: run the check.

---

## 4. The invariant, reported rather than implied

The gate proves byte-identity of the exported CSV, which *entails* identical docks but never
displayed them. Since the standing instruction is to keep checking that the canonical run still
passes — docks and main results — that is now a first-class, re-runnable tool,
`gate/dock_check.py`:

```
MUJOCO_GL=disabled PYTHONPATH=. python3 gate/dock_check.py [log.json]
```

It reports the at-weld `dock_events` d_mm (Rule 10 — never min-over-swing) against the frozen
2.5 table, plus θ_s, h_w, e_com and the QP-failure count, and exits non-zero on divergence.

### Gate

```
[4] environment pin           : PASS
[3] two-model consistency     : PASS  (15 links, 14 joints, total 71.056 kg)
[1] canonical replay + export : replay rc=0 (286.2s), export rc=0
[2] artifact identity         : PASS  (2077 rows × 132928 fields,
                                       excl ['nmpc_time_ms', 'qp_time_ms'])
VERDICT: PASS   (env PASS, 291.5s)
```

### Canonical result, post-surgery

```
at-weld docks  6/6  ALL UNDER 5 mm
   step 1:    4.02 mm   (frozen  4.02, delta +0.0000)   margin  0.98 mm
   step 2:    4.89 mm   (frozen  4.89, delta +0.0000)   margin  0.11 mm
   step 3:    4.99 mm   (frozen  4.99, delta +0.0000)   margin  0.01 mm
   step 4:    4.97 mm   (frozen  4.97, delta +0.0000)   margin  0.03 mm
   step 5:    4.95 mm   (frozen  4.95, delta +0.0000)   margin  0.05 mm
   step 6:    4.62 mm   (frozen  4.62, delta +0.0000)   margin  0.38 mm
   worst margin: 0.01 mm

theta_s peak [deg]          0.540   frozen   0.540   OK
h_w peak axis [Nms]         4.102   frozen   4.100   OK
h_w peak norm [Nms]         4.243   frozen   4.240   OK
e_com peak [m]              0.154   frozen   0.154   OK
qp_fail                         0   frozen       0   OK

CANONICAL RESULTS: MATCH frozen 2.5
```

Every dock reproduces to `delta +0.0000` mm — the run is bit-for-bit the frozen canonical, not
merely close to it. The step-3 margin remains the binding 0.01 mm.

---

## 5. What remains in `crawlbot/planning/`

| file | lines | disposition |
|---|---|---|
| `coarse_preplanner.py` | 539 | 45 dead statements, predominantly **fallback branches** — same KEEP class as `get_shifted_fallback`. Audit line-by-line before touching. |
| `contact_scheduler.py` | 349 | 16 dead, same class |
| `sequence_loader.py` | 254 | never imported on the canonical, but backs `sim.setup(sequence_path=...)`. KEEP |
| `locomotion_planner.py` | 205 | KEEP — §3 |
| `torso_planner.py` | 480 | live |
| `swing_planner.py` | 337 | live |

`constrained_geodesic.py` (470 lines) was deleted in CLEANUP-17.

The planning module is done to the point where the next removal would require judging
fallback branches, which is a different and riskier exercise than removing research sediment.
