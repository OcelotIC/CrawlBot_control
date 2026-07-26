# PHASE CLEANUP-23 — `lutze_baseline/` audit (READ-ONLY)

Thorough investigation of `lutze_baseline/` (10 files, 1277 lines) on the
hypothesis that it is entirely dead.

**Verdict: the hypothesis is correct in the strongest sense — nothing in the
package can run.** All three entry points fail, all for the same reason, and
the failure predates the chantier by three months.

**No code changed.** This also **retracts a CLEANUP-18 decision** (§5).

---

## 1. What the package is

The Lutze et al. (2023) single-step QP wrench optimiser, adapted to VISPA as the
**M0 comparison baseline** for the hierarchical MPC controller.

| file | lines | external consumers |
|---|---:|---|
| `sim_lutze.py` | 441 | 2 scripts in `Misc/scripts/` |
| `plot_comparison.py` | 203 | **none** |
| `lutze_qp.py` | 189 | `tests/test_phase1.py` |
| `lutze_feedforward.py` | 99 | `tests/test_phase1.py` |
| `lutze_swing_controller.py` | 82 | `tests/test_phase1.py` |
| `contact_adjoint.py` | 80 | `tests/test_phase{0,1}.py` |
| `momentum_map.py` | 63 | `tests/test_phase{0,1}.py` |
| `centroidal_model.py` | 61 | `tests/test_phase{0,1}.py` |
| `lutze_joint_torques.py` | 52 | `tests/test_phase1.py` |
| `__init__.py` | 7 | — |

Last modified **2026-04-10** (`6b37a58`, "M0 baseline: 3-step simulation
diagnostic report"). Nothing has touched it since.

---

## 2. The apparent consumers are not what they look like

### `tests/test_phase0.py` / `test_phase1.py` are not tests

They define `run_test(urdf_path)` — **not** `test_*`. With
`pytest.ini: python_files = test_*.py`, pytest matches the *filename* and then
collects nothing from inside:

```
$ pytest tests/test_phase0.py tests/test_phase1.py --collect-only -q
no tests collected in 0.54s
```

So **the pytest suite does not exercise `lutze_baseline` at all.** They are
standalone scripts that happen to live in `tests/`, invoked as
`python tests/test_phase0.py --urdf ...`.

Their docstrings tell you to run `python -m lutze_baseline.test_phase0` — a
module that does not exist in the package. Stale since they were moved.

### `Misc/scripts/` consumers are documented as superseded

`sim_lutze.py`'s only referents are `Misc/scripts/run_r6_full_sim.py` and
`run_r7_figures.py`. The repository's own figure-pipeline inventory
(`Misc/runs/j2_closure_curves/FIGURE_PIPELINE_INVENTORY.md`) says of the second:

> *The publication-quality plotter `run_r7_figures.py` exists but is **R7-era and
> superseded** — it plots Lutze-vs-MPC single/3-step from old logs
> (`sim_lutze_log.json`, …), **not** the J2 cooperative-arms run, and its outputs
> are **not committed**.*

Both live in `Misc/`, which is slated for removal.

---

## 3. Every entry point is broken — measured, not inferred

The package predates the **6-DOF → 7-DOF arm upgrade** and was never migrated.

### `tests/test_phase0.py`

```
ValueError: wrong argument size: expected 20, got 18
hint: The velocity vector is not of right size
```

Root cause, `test_phase0.py:18` and `:32`:

```python
from crawlbot.core.robot_interface import RobotInterface, NQ, NV
v0 = np.zeros(NV)
```

This is **exactly the trap documented in `docs/crawlbot/core/robot_interface.md`
§2**, caught in the wild: `NQ`/`NV` are module-level defaults from the 6-DOF era
(19/18), rebound by `global` to 21/20 at construction. A `from … import NV`
snapshots **18 forever**, so the script builds an 18-vector for a 20-DOF model.

### `tests/test_phase1.py`

Identical failure, identical cause.

### `lutze_baseline/sim_lutze.py`

```
ValueError: could not broadcast input array from shape (12,) into shape (14,)
  at crawlbot/core/state_conversions.py:150
```

Same root cause one level up: it assumes **12 arm joints** (6 x 2) where the
current model has **14** (7 x 2).

**Three entry points, three failures, one cause.** The package has been
unrunnable since the 7-DOF upgrade.

---

## 4. The saved baseline output is also from the old architecture

`Misc/runs/logs/sim_lutze_log.json` exists — the M0 result, preserved as data.
But:

```
tau[0] shape = (12,)            <- 6-DOF era
current model: 14 arm joints
```

So the stored baseline is **not dimensionally comparable** to any current
canonical result. Anyone using it as an M0 reference against the frozen 2.5
canonical would be comparing two different robots.

This is worth stating plainly because it is the tempting middle path — *"delete
the code, keep the log"*. The log is not a usable baseline either.

---

## 5. ⚠ This retracts a CLEANUP-18 decision

CLEANUP-18 proposed deleting `crawlbot/planning/locomotion_planner.py`
(205 lines) and **I reversed that on the strength of this package**, writing:

> *The decisive one is `lutze_baseline/sim_lutze.py` — a package, not a research
> script, carrying the M0/Lutze comparison … `LocomotionPlanner` is load-bearing
> there.*

The measurement behind that was **import resolution**: every `crawlbot.*` symbol
in `sim_lutze.py` resolves at HEAD. That was true then and is true now.

**It was the wrong instrument.** Imports resolving says nothing about whether a
program runs. `sim_lutze.py` imports cleanly and then dies on the first state
conversion. I checked that the door opened, not that the room existed.

Consequence: `locomotion_planner.py` has **no working consumer** and the
CLEANUP-18 reversal does not stand. It should go back on the removal list.

This is the second time in this chantier that a "measure it" claim was measured
with an instrument too weak for the question — the first was `link_audit`'s
basename heuristic raising 35 false alarms. The lesson is narrower than "measure":
**state what would falsify the claim, then measure that.** For "is this code
alive", the falsifier is *does it run*, not *does it import*.

---

## 6. What the paper actually needs

CLAUDE.md's remaining-work list has:

> *§II differentiation table vs Lutze[2023]/Rognant[2025] (**sourced**).*

"Sourced" — a **literature comparison**, built from the published papers and
`refs_master.bib`, not from re-running this code. No current paper figure depends
on `lutze_baseline` or its log: the J2 figure chain is
`export_figure_data.py` → tidy CSV, and the Lutze-vs-MPC plotter is documented as
superseded with uncommitted outputs (§2).

So removal does not touch the paper.

---

## 7. Recommendation

| target | lines | note |
|---|---:|---|
| `lutze_baseline/` | **1277** | unrunnable; no working consumer |
| `crawlbot/planning/locomotion_planner.py` | **205** | its only consumer was `sim_lutze.py` (§5) |
| `tests/test_phase0.py`, `test_phase1.py` | ~250 | not collected by pytest, broken as scripts |
| **total** | **~1730** | |

Removing `locomotion_planner.py` also drops the last non-canonical import from
`crawlbot/planning/__init__.py`.

### The one argument for keeping it

A runnable M0 baseline has scientific value independent of whether *this* code
runs: if a Lutze comparison at 7 DOF is ever wanted, this is the starting point.
Migration is not large — the failures are DOF-arity, not algorithmic.

Against that: git history retains every line, so deleting at HEAD loses nothing
recoverable, and `git show 6b37a58:lutze_baseline/sim_lutze.py` is a perfectly
good starting point too.

**This is a project decision, not a measurement one.** The measurement is
finished: nothing here runs, nothing working depends on it, and the paper does
not need it.

### If it is kept instead

Then it should be *fixed*, not left broken — a baseline that cannot run is worse
than no baseline, because it reads as available. The fix is mechanical: replace
`from … import NQ, NV` with instance attributes (`robot.model.nq`,
`robot.model.nv`) in the two scripts, and the hard-coded 12 with `robot.n_joints`
in `sim_lutze.py`.
