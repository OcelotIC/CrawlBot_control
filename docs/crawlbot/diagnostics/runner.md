# `crawlbot.diagnostics.runner`

**File**: [`crawlbot/diagnostics/runner.py`](../../../crawlbot/diagnostics/runner.py) — **71 lines** — canonical coverage **15 %**

> Module docstring: *"Single entry point for the diagnostic suite."*

Orchestrator for the diagnostic suite: metrics, plots, snapshots.

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| `run_diagnostics` | `(log, output_dir, cfg=None, thresholds=None, model=None,...)` | not exercised | [L14](../../../crawlbot/diagnostics/runner.py#L14) |

---

---

## 1. Usage

```python
from crawlbot.diagnostics import run_diagnostics
run_diagnostics(log, output_dir, cfg=None, thresholds=None, model=None, data=None)
```

Chains `compute_metrics` -> `print_metrics` / `save_metrics_csv` ->
`generate_plots`, then `capture_snapshots` when `model` and `data` are supplied.

Usable on demand against an existing log:

```bash
MUJOCO_GL=osmesa PYTHONPATH=. python3 -c "
from crawlbot.diagnostics import run_diagnostics
import json
log = json.load(open('results/<log>.json'))
run_diagnostics(log, 'results/<output_dir>/')
"
```

## 2. The intent

Rule 3 exists because "it docked" is a weak pass criterion — a run can dock while
saturating the wheels, drifting the structure, or hitting a joint limit. The
suite is meant to turn every such quantity into a thresholded verdict, so a pass
is a statement about all of them rather than about the last 5 mm.

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
| `run_diagnostics()` | [L14-70](../../../crawlbot/diagnostics/runner.py#L14-L70) |

---

## See also

- package overview: [`diagnostics.md`](diagnostics.md)
