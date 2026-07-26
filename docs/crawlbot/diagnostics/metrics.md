# `crawlbot.diagnostics.metrics`

**File**: `crawlbot/diagnostics/metrics.py` — **424 lines** — canonical coverage **5 %**

> Module docstring: *"Compute scalar summary metrics from SimLog time series."*

Thresholded metrics: computes each quantity, compares it to its bound, returns
a verdict.

---

## Public API

| symbol | signature | canonical? |
|---|---|---|
| `compute_metrics` | `(log, cfg=None, thresholds=None)` | not exercised |
| `print_metrics` | `(results, file=None)` | not exercised |
| `save_metrics_csv` | `(results, path)` | not exercised |

### Module constants

| name | value |
|---|---|
| `DEFAULT_THRESHOLDS` | `{'torso_pos_err_peak_mm': 10.0, 'torso_ori_e` |

---

---

## 1. Principle

`compute_metrics(log, cfg, thresholds)` returns a dict
`name -> (value, threshold, verdict)`. `print_metrics` formats it for the
console, `save_metrics_csv` writes it out.

The design follows rule 3: a run passes because every measured quantity is under
its bound, not because it reached the target.

## 2. The largest unexercised block in the repository

287 lines with no coverage. Its size says nothing about its validity — only that
it is unverified. If it is ever reconnected, its thresholds should be
cross-checked against the frozen canonical values in CLAUDE.md, since the
controller has been retuned substantially since this was written.

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

## See also

- package overview: [`diagnostics.md`](diagnostics.md)
