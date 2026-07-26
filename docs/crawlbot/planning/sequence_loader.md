# `crawlbot.planning.sequence_loader`

**File**: [`crawlbot/planning/sequence_loader.py`](../../../crawlbot/planning/sequence_loader.py) — **255 lines** — canonical coverage **0 %**

> Module docstring: *"Locomotion-sequence file loader."*

Loads a `.seq` scenario file and turns it into a gait plan — an alternative to
specifying a traversal through arguments.

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`SwingTarget`** *(dataclass)* |  |  | [L58](../../../crawlbot/planning/sequence_loader.py#L58) |
|   `arm` | `` | _field_ | [L59](../../../crawlbot/planning/sequence_loader.py#L59) |
|   `anchor_idx` | `` | _field_ | [L60](../../../crawlbot/planning/sequence_loader.py#L60) |
|   `dwell_after` | `0.0` | _field_ | [L61](../../../crawlbot/planning/sequence_loader.py#L61) |
| **`LoadedSequence`** *(dataclass)* |  |  | [L65](../../../crawlbot/planning/sequence_loader.py#L65) |
|   `start_a` | `` | _field_ | [L66](../../../crawlbot/planning/sequence_loader.py#L66) |
|   `start_b` | `` | _field_ | [L67](../../../crawlbot/planning/sequence_loader.py#L67) |
|   `swing_targets` | `` | _field_ | [L68](../../../crawlbot/planning/sequence_loader.py#L68) |
|   `source_path` | `` | _field_ | [L69](../../../crawlbot/planning/sequence_loader.py#L69) |
| `load_sequence` | `(path, n_anchors)` | — | [L88](../../../crawlbot/planning/sequence_loader.py#L88) |
| `plan_from_sequence` | `(sched, seq)` | — | [L193](../../../crawlbot/planning/sequence_loader.py#L193) |

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

## Code map

| unit | source |
|---|---|
| `class SwingTarget` | [L58-61](../../../crawlbot/planning/sequence_loader.py#L58-L61) |
| `class LoadedSequence` | [L65-69](../../../crawlbot/planning/sequence_loader.py#L65-L69) |
| `load_sequence()` | [L88-190](../../../crawlbot/planning/sequence_loader.py#L88-L190) |
| `plan_from_sequence()` | [L193-254](../../../crawlbot/planning/sequence_loader.py#L193-L254) |

---

## See also

- package overview: [`planning.md`](planning.md)
