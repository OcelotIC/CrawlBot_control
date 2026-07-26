# PHASE CLEANUP-32 — telemetry split out of the controller

The structural item from the readability audit. `sim_loop.py` was 3488 lines and
28 methods in one class; the per-tick recorders now live in their own module.

```
crawlbot/simulation/sim_loop.py      3488 -> 3014 lines
crawlbot/simulation/tick_logging.py          530 lines (new)
```

---

## 1. Why logging, and why not setup

The audit offered two candidate groups. They are not equally good, and the
measurement says so plainly:

| group | lines | calls back into the class | `self.<attr>` used | module imports needed |
|---|---:|---|---:|---:|
| **logging** (`_log_ds_tick`, `_log_ss_tick`) | 413 | **3** | 10 | **5** |
| setup (`setup`, `_setup_torso_for_step`, `_run_preplanner`, `_settle_setup`) | 721 | **6** | 42 | **18** |

Logging is nearly self-contained: three geometry queries and ten attributes.
Setup reaches back into six methods — including `_run_ds_passivity_loop`, itself
236 lines — touches 42 attributes, and drags in eighteen of the module's imports.

Splitting setup would **relocate** code without **separating** concerns: the new
file would need most of what the old one needs, and a reader would gain a second
place to look rather than a smaller thing to read. Not done. Recorded here so the
next person does not re-derive it.

## 2. The organising principle

`SimulationLoop` is the controller; this is its telemetry. They are separated
because **they fail differently**: a mistake here corrupts a plot, a mistake
there corrupts the robot.

That is also what makes the split cheap to verify. Both recorders are pure with
respect to control — they read `self`, the MuJoCo/Pinocchio state and their
arguments, and write only to `log`. Two calls mutate (`mujoco.mj_forward`,
`robot.update`) but only recompute derived quantities from unchanged
`qpos`/`qvel`, so the blocks moved **verbatim and in order** rather than being
tidied on the way.

⚠ The one naming hazard: `logging.py` already exists in this package and defines
`SimLog`. That module is the **container**; this one is the **writers**. Both
docstrings say so, because adjacent names with different jobs is exactly how a
reader gets lost.

## 3. What moved

```
TickState          (65 lines)  the boundary record from CLEANUP-31
_log_ds_tick      (210 lines)  double support
_log_ss_tick      (203 lines)  single support
```

as `TickState` plus `TickLoggingMixin`, mixed into `SimulationLoop`, so `self.*`
resolves against the finished class exactly as before. Three calls resolve there
rather than in the new module — `_get_ee_data`, `_gripper_distance`,
`_swing_query_time` — because they are geometry queries the loop owns.
Duplicating them to make the module standalone would trade a little coupling for
a lot of drift risk, and CLEANUP-21's fixture miss is what that failure looks
like.

## 4. Verification

```
gate/run_gate.py
  [1] canonical replay + export   rc=0 (80.4 s)
  [2] artifact identity           PASS — 2077 rows × 132 928 fields
  VERDICT: PASS

gate/dock_check.py
  at-weld docks 6/6 — 4.02 / 4.89 / 4.99 / 4.97 / 4.95 / 4.62 mm
  every step delta +0.0000     θ_s 0.540     qp_fail 0
  CANONICAL RESULTS: MATCH frozen 2.5

gate/run_suite.py          PASS   200 tests, 199 passed, 1 xfail, 94 s
sync_docs --check / verify_docs / verify_params / verify_roots / link_audit  OK
```

### Three checkers bit on this pass

Worth listing, because it is the whole argument for having built them:

1. **`sync_docs --check`** — `tick_logging.md: MISSING — module has no document`.
   A new module without a document is exactly the failure Rule 15 exists to
   prevent, and it was refused at the gate rather than noticed later.
2. **`verify_params`** — CLAUDE.md's `α torque-min` / `α accel-reg` rows cited
   `sim_loop.py:1197`; moving 474 lines out sent the literals to `:926`. Second
   time in two passes this checker has caught my own drift.
3. **`verify_docs`** — `contact_estimator.md` cited `sim_loop.py:3123` in a file
   that is now 3014 lines. Real site `:2913`.

None of the three would have been caught by reading.

### And one rot-proofing fix

`sim_loop.md`'s coupling profile was written in CLEANUP-31 with absolute line
numbers, which this pass immediately invalidated. Rewritten as **percentages
through the method**. The same lesson as the CLAUDE.md prose citations: a
measurement worth documenting should be expressed so that moving code cannot
falsify it.

---

## 5. State of the module

```
sim_loop.py   3014 lines, 26 methods
  _step        851   run  600   setup  308
  _run_ds_passivity_loop 236   _setup_torso_for_step 221
```

Still the largest file, and the two remaining structural items are the ones the
audit priced as medium risk:

- **`_step`'s core** — coupling climbs monotonically with no cheap seam, so it
  needs a state object threaded through ~650 lines, and 99 of its statements are
  never executed by the canonical, so the gate cannot prove that part either way.
- **`run()`** — 600 lines of which **546 are a single `while`**. Its problem is
  nesting depth, not sequence; extraction has to come from inside the loop body.

The zero-risk items from the audit remain and are still worth doing first:
section banners (`_step` has **1** across 851 lines, `run` has 6 across 600), a
module-level map, and pruning the 34 comment lines that narrate removed
architecture. Plus `_planned_arm_config`: 37 lines, zero call sites, 5 % covered.

---

## 6. ⚠ POSTSCRIPT — §4's verification above was VACUOUS (CLEANUP-33)

Everything in §4 was reported in good faith and was **wrong**. Two faults, and
the second is much worse than the first.

### Fault 1 — the move dropped a decorator

The span moved for `TickState` started at `class TickState:`. `@dataclass` was on
the line **before** it. So the new module got a plain class, and `sim_loop.py`
kept an orphaned `@dataclass` that silently landed on `SimulationLoop`.

`SimulationLoop` survived by luck: `dataclasses` uses `_set_new_attribute`, which
does not overwrite an existing `__init__`. `TickState` did not — every
`TickState(t=..., ...)` raised `TypeError: TickState() takes no arguments`, so
`_step` failed on its first single-support tick.

### Fault 2 — and the gate reported PASS anyway

`gate/replay_canonical.py` wrapped `dca.main()` in `except Exception` with the
comment *"dca.main may raise after the sim_log is written"* — then printed
`[replay] done` and exited **0**. `run_gate.py` checked only that a
`sim_log.json` **existed**.

Nothing deleted the previous run's log. So:

```
sim_log.json   19:01:37   <- written by CLEANUP-31's gate run
last_verdict   19:34:33   <- the run that "validated" it, 33 minutes later
replay time    80.4 s     <- vs 127-142 s for a real run. The tell, missed.
```

The replay died on tick one, the export re-exported the **stale** artifact, and
identity passed against it. `dock_check` read the same stale log and reported all
six docks at delta +0.0000. The coverage replay died the same way, which is why
`_step`'s unreached count appeared to jump 99 -> 140 and `sim_loop.md` was
committed claiming 63 % coverage instead of 81 %.

**A gate that can pass on a stale artifact is worse than no gate**, because it
launders a broken commit as verified. This one did, and it was pushed.

### The fix, and proof it bites

- `replay_canonical.py` now **deletes the target log before running**, and an
  exception raised *before* the log exists exits **1** instead of printing
  `done`.
- `run_gate.py` now fails on a non-zero replay return code, and independently
  asserts the log's mtime is **newer than the replay start** — belt and braces,
  so any other cause of staleness is caught too.

Proven on the real fault rather than a synthetic one: with the missing decorator
re-injected,

```
[1] canonical replay + export : replay rc=1 (82.4s), export rc=None
[2] artifact identity         : FAIL  (replay exited 1)
VERDICT: FAIL
```

Then, decorator restored:

```
[1] canonical replay + export : replay rc=0 (139.4s)     <- full duration again
[2] artifact identity         : PASS  (2077 rows × 132 928 fields)
VERDICT: PASS
sim_log.json 20:11:54, verdict 20:11:55                  <- fresh, by one second
docks 6/6, every step delta +0.0000, MATCH frozen 2.5
suite 200 tests, 199 passed, 1 xfail, 95 s
```

So the CLEANUP-32 split **is** inert — but that is only now established. The
coverage and every doc percentage were regenerated from the valid run.

### What this says about the method

The chantier's rule is that a checker is proven to bite before being trusted.
`run_gate.py` was proven that way in CLEANUP-0 — against a *wrong number*, by
injecting an F1 arithmetic fault. It was never proven against a **missing run**,
and that is the hole a wrong-number probe cannot find.

The lesson generalises past this repo: verifying that a check rejects bad data
does not verify that it rejects *absent* data. Every artifact-comparing gate
needs a freshness assertion, not just a comparison.
