# PHASE CLEANUP-16 — `crawlbot/planning/` audit (READ-ONLY)

Audit of the planning module, unblocked by CLEANUP-15 (which removed the FK-reference path
from `sim_loop` and thereby orphaned its planner-side wiring). **No code changed.**

Coverage was re-measured **after** CLEANUP-15 — the previous planner numbers predated it and
would have been wrong. (Working from stale coverage nearly caused a bad cut in CLEANUP-15; the
rule now is to re-measure after every removal.)

## Scale

| file | lines | statements | dead | cover |
|---|---|---|---|---|
| `swing_planner.py` | 728 | 307 | **163** | **47 %** |
| `torso_planner.py` | 702 | 221 | **99** | 55 % |
| `coarse_preplanner.py` | 539 | 242 | 45 | 81 % |
| **`constrained_geodesic.py`** | **470** | — | **never imported** | **0 %** |
| `contact_scheduler.py` | 349 | 120 | 16 | 87 % |
| `sequence_loader.py` | 254 | — | never imported | — |
| `locomotion_planner.py` | 205 | 72 | **60** | **17 %** |

Two files **do not appear in the coverage report at all**, which means they were never imported
during the canonical run — a stronger statement than "0 % covered".

## The big finding: `constrained_geodesic.py` is now 100 % dead

470 lines, 7 public functions (`project_to_stance`, `ik_three_tasks`,
`smoothed_constrained_geodesic`, `precompute_segment_tangents`, `frame_reference_at_tau`,
`q_v_real_at_tau`). **Every import of it is a function-local import inside a dead FK method**:

```
swing_planner.py:231, 375        (inside _override_reference_at_fk / add_phase FK branch)
torso_planner.py:359, 384, 405, 427  (inside _add_phase_fk / _reference_at_fk / ...)
```

`sim_loop`'s imports were removed in CLEANUP-15. Because all remaining imports are inside
methods that never execute, the module is never loaded. It is the single largest removable
unit found in this chantier.

## `SwingPlanner`: the phase-override mechanism is dead as of CLEANUP-15

`swing_planner.add_phase()` now has **zero production callers** (verified: the only
`.add_phase(` in `sim_loop` is `torso_planner.add_phase` at 1551). `add_phase` only ever
registered a *phase override*; with none registered, `reference_at()` falls through to the
scheduler-driven gait plan — the path the canonical has always used.

Dead as a consequence, and coherent as one unit:

| method | dead/span | note |
|---|---|---|
| `_override_reference_at` | 57/75 | the override evaluator |
| `_override_reference_at_fk` | 15/38 | its FK variant |
| `add_phase` (body) | 21/126 | no callers |
| `set_swing_orientation` | 5/20 | partly |
| `clear_phase_overrides` | — | still **called** (`sim_loop:1481`) but now always a no-op on an empty list |

Independently dead, unrelated to the override mechanism:

- **`adaptive_reference_at`** — 42 dead statements over 84 lines, **zero callers anywhere** in
  `crawlbot/`, `scripts/` or `tests/`.
- **`swing_trajectory`** — 14 dead; only caller is `scripts/test_integration.py`.

## `TorsoPlanner`: FK methods + methods orphaned by our own earlier passes

| method | dead/span | status |
|---|---|---|
| `_trapezoidal_params` | 22/57 | **zero callers, even internally** — only its own `def` line matches |
| `set_from_waypoints` | 18/49 | **orphaned by CLEANUP-14** — its only production caller was the `ds_mobile_com_magnitude` block |
| `_add_phase_fk` | 6/32 | FK |
| `_com_reference_at_fk` | 8/19 | FK |
| `_l_com_reference_at_fk` | 6/19 | FK |
| `_reference_at_fk` | 5/16 | FK |
| `add_phase` `q_seq` branch | part of 15/146 | FK dispatch |
| `set_torso_inertia` | 5/27 | the FK-mode warning path |

`torso_planner.add_phase` itself is **live** (`sim_loop:1551`) — only its FK branch is dead.

## `LocomotionPlanner`: a retired module still exported

72 statements, **60 dead (83 %)**. `sim_loop.py:46` carries the comment *"LocomotionPlanner
removed — CoM reference comes from TorsoPlanner"* — and indeed `sim_loop` never constructs it.
Its only references are `planning/__init__.py` (the export) and two legacy scripts
(`test_integration.py`, `sim_torso6d.py`). `full_trajectory` has zero callers.

## KEEP

- **`sequence_loader.py`** — never imported on the canonical, but it backs a *legitimate
  feature*: `sim.setup(sequence_path=...)` (used by `dca` when a scenario file is given). Unused
  ≠ retired research. Leave it.
- **`coarse_preplanner.py`'s 45 dead statements** and `contact_scheduler.py`'s 16 — these are
  predominantly failure/fallback branches, same class as `get_shifted_fallback`. Not audited
  line-by-line here; do that before touching them.

## Removal plan (ranked by value ÷ risk)

| # | target | ~lines | risk |
|---|---|---|---|
| 1 | **delete `constrained_geodesic.py` entirely** | **470** | lowest — never imported; only referents are the dead FK methods removed in step 2 |
| 2 | TorsoPlanner + SwingPlanner **FK methods** and the `q_seq` dispatch | ~180 | low — dead, and step 1 removes their import target |
| 3 | SwingPlanner **phase-override mechanism** (`add_phase`, `_override_reference_at*`), and make `clear_phase_overrides` a no-op or drop its call site | ~200 | low — zero production callers |
| 4 | `adaptive_reference_at`, `_trapezoidal_params`, `set_from_waypoints`, `swing_trajectory` | ~150 | low — zero or legacy-script-only callers |
| 5 | **delete `locomotion_planner.py`** + its `__init__` export | **205** | low, but breaks 2 legacy scripts |

Total ≈ **1200 lines**, comfortably the largest remaining block in the chantier — and unusually
safe, because most of it is not merely unreachable but *never even imported*.

Steps 1–2 should land together (step 2 removes the only referents of step 1). Each step
gate-verified byte-identical as usual.

**Collateral, as before:** `scripts/test_integration.py` and `scripts/sim_torso6d.py` import
`LocomotionPlanner`; `test_integration.py` also calls `swing_trajectory`. Both are legacy
research scripts already in the "non-functional after cleanup" list in
`CLEANUP_CARRYOVER.md` §C3.
