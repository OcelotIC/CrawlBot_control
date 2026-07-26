# `crawlbot.diagnostics.plots`

**File**: [`crawlbot/diagnostics/plots.py`](../../../crawlbot/diagnostics/plots.py) — **689 lines** — canonical coverage **5 %**

> Module docstring: *"Generate the fixed set of 8 diagnostic figures from SimLog."*

Diagnostic figure plates generated from a simulation log.

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| `generate_plots` | `(log, output_dir, cfg=None, dpi=150)` | not exercised | [L67](../../../crawlbot/diagnostics/plots.py#L67) |

### Module constants

| name | value |
|---|---|
| `_PHASE_COLORS` | `{'DS': 'blue', 'SS': 'orange'}` |

---

---

## 1. Usage

`generate_plots(log, output_dir, cfg=None, dpi=150)` — called by
`run_diagnostics`.

## 2. ⚠ These are not the paper figures

Published figures come from `scripts/export_figure_data.py` and
`scripts/diag_full_diag_export.py`, reading the same `sim_log.json` through a
different chain. Do not assume a plate from here matches a published figure —
the quantities are recomputed independently.

---

## Package-wide caveat

`crawlbot/diagnostics/` is **not exercised by the canonical run**, although
CLAUDE.md rule 3 requires it:

> *Every simulation produces diagnostics. Call `run_diagnostics()` at the end of
> every sim. "It docked" is not a pass criterion.*

Measured: `run_diagnostics` 0/56 lines, `compute_metrics` 0/287,
`generate_plots` 0/26, `capture_snapshots` 0/59. The canonical import closure
pulls in `crawlbot/diagnostics/__init__.py` (which re-exports `run_diagnostics`)
but **none of the four modules**, and neither `dca` nor `sim_loop` calls it.

Reported, not fixed: this is a rule-compliance question, orthogonal to the code
(CLEANUP-20 section 5.3). What is authoritative today is the gate
(`gate/run_gate.py`, `gate/dock_check.py`) and the export scripts.

Practical consequence: **no gate coverage anywhere in this package**. A
regression introduced here will be caught by nothing.

## Code map

| unit | source |
|---|---|
| `generate_plots()` | [L67-93](../../../crawlbot/diagnostics/plots.py#L67-L93) |

---

## See also

- package overview: [`diagnostics.md`](diagnostics.md)
