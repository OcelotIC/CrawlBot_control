# `crawlbot.diagnostics.snapshots`

**File**: `crawlbot/diagnostics/snapshots.py` — **71 lines** — canonical coverage **0 %**

> Module docstring: *"Render MuJoCo frames at key simulation instants for visual diagnostics."*

MuJoCo image captures at the instants recorded in the log.

---

## Public API

| symbol | signature | canonical? |
|---|---|---|
| `capture_snapshots` | `(model, data, sim_log, output_dir, width=1280, height=72...)` | not exercised |

---

---

## 1. Usage

`capture_snapshots(model, data, sim_log, output_dir, width=1280, height=720,
camera=None)` renders the poses stored in `log.snapshots`.

Requires a render context: `MUJOCO_GL=osmesa`, or `disabled` when rendering is
unavailable. Project rule: never run a simulation without one of the two set.

The canonical captures 44 poses when `cfg.frames_per_step > 0`; the rendering
itself goes through `scripts/render_traversal.py`, not this module.

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
