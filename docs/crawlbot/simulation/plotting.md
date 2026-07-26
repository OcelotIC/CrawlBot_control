# `crawlbot.simulation.plotting`

**File**: `crawlbot/simulation/plotting.py` — **154 lines** — canonical coverage **2 %**

> Module docstring: *"9-panel diagnostic plot for simulation results."*

Simulation plots — **not used by the canonical run** (2 % coverage).

---

## Public API

| symbol | signature | canonical? |
|---|---|---|
| `plot_simulation` | `(log, save_path=None, cfg=None)` | not exercised |

---

---

## 1. Status

`plot_simulation(log, save_path, cfg)` is called neither by `dca` nor by
`sim_loop` on the canonical path. Its only entry point is
`SimulationLoop.plot`, itself unexercised.

## 2. What actually produces figures

| use | tool |
|---|---|
| paper figures | `scripts/export_figure_data.py` |
| 66-column fulldiag export | `scripts/diag_full_diag_export.py` |
| diagnostic plates | `crawlbot/diagnostics/plots.py` (also unexercised) |

⚠ Do not assume a plate produced here corresponds to a published figure: it is a
different chain, reading the same `sim_log.json` but computing its own
quantities.

## 3. Consequence

No gate coverage. A regression introduced here will be caught by nothing.

## See also

- package overview: [`simulation.md`](simulation.md)
