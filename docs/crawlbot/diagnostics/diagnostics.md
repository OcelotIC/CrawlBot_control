# `crawlbot.diagnostics`

**Metrics and figures — and a rule-compliance gap.**

| file | lines | canonical coverage | document |
|---|---:|---:|---|
| `plots.py` | 689 | 5 % | [plots.md](plots.md) |
| `metrics.py` | 424 | 5 % | [metrics.md](metrics.md) |
| `runner.py` | 71 | 15 % | [runner.md](runner.md) |
| `snapshots.py` | 71 | **0 %** | [snapshots.md](snapshots.md) |

## The main fact about this package

**It is not exercised by the canonical run**, although CLAUDE.md rule 3 requires
it:

> *Every simulation produces diagnostics. Call `run_diagnostics()` at the end of
> every sim. "It docked" is not a pass criterion.*

Measured: `run_diagnostics` 0/56 lines, `compute_metrics` 0/287,
`generate_plots` 0/26, `capture_snapshots` 0/59. The canonical import closure
pulls in `__init__.py` but none of the four modules, and nothing calls the
function.

Reported, not fixed — it is a rule-compliance question, orthogonal to the code
(CLEANUP-20 section 5.3).

## What is authoritative instead

The gate (`gate/run_gate.py` for byte-identity, `gate/dock_check.py` for the
physical numbers) and the export scripts. Those are what the frozen canonical
results are measured with.

## Consequence

No gate coverage anywhere in this package. A regression introduced here will be
caught by nothing.
