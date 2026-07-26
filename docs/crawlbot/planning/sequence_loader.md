# `crawlbot.planning.sequence_loader`

**File**: `crawlbot/planning/sequence_loader.py` — **255 lines** — canonical coverage **0 %**

> Module docstring: *"Locomotion-sequence file loader."*

Loads a `.seq` scenario file and turns it into a gait plan — an alternative to
specifying a traversal through arguments.

---

## Public API

| symbol | signature | canonical? |
|---|---|---|
| **`SwingTarget`** *(dataclass)* |  |  |
|   `arm` |  | _field_ |
|   `anchor_idx` |  | _field_ |
|   `dwell_after` | `0.0` | _field_ |
| **`LoadedSequence`** *(dataclass)* |  |  |
|   `start_a` |  | _field_ |
|   `start_b` |  | _field_ |
|   `swing_targets` |  | _field_ |
|   `source_path` |  | _field_ |
| `load_sequence` | `(path, n_anchors)` | not exercised |
| `plan_from_sequence` | `(sched, seq)` | not exercised |

### Module constants

| name | value |
|---|---|
| `_ANCHOR_RE` | `re.compile('^anchor_(\\d+)([ab])$')` |

---

---

## 1. What it does

`load_sequence(path)` parses the scenario file; `plan_from_sequence(...)` turns
it into the same `GaitPlan` structure `ContactScheduler.plan_traversal` produces,
so everything downstream is unchanged.

Entry point: `sim.setup(sequence_path=...)`, used by `dca` whenever a scenario
file is given. `dca` then routes output into a subdirectory named after the
scenario stem.

Available scenarios in `scenarios/`: `canonical_3step.seq`, `canonical_5step.seq`,
`multi_traversal_2x.seq`, `multi_traversal_10x.seq`,
`multi_traversal_10x_dwell.seq`.

## 2. ⚠ 0 % coverage — and kept anyway

The module never executes on the canonical run, which uses `n_steps=6` directly.
It is nevertheless a **real user-facing feature**, not abandoned research.

This is a distinction the chantier applies throughout:

> **Unused on the canonical is not the same as retired.**

Compare with genuine research sediment — the alternative AOCS modes, the planner
FK path — which sits behind opt-in flags from closed experiments. Here the flag
is a documented user input.

## 3. Practical consequence

No gate coverage. A regression introduced here will be caught by nothing, so the
scenario path must be exercised by hand if it is modified.

## See also

- package overview: [`planning.md`](planning.md)
