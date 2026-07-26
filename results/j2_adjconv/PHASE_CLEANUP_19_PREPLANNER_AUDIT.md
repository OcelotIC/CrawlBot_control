# PHASE CLEANUP-19 — `coarse_preplanner.py` + `contact_scheduler.py` audit (READ-ONLY)

The last two unaudited files in `crawlbot/planning/`. CLEANUP-16 deferred them with the note
that their dead statements are *"predominantly failure/fallback branches, same class as
`get_shifted_fallback`"* — i.e. dead **because the system is healthy**, which is the one class
of dead code that must be kept. This pass checks that claim statement by statement instead of
accepting it.

**No code changed.** Coverage re-measured after CLEANUP-18 (rule: re-measure after every
removal).

---

## 1. Method

Two independent sources, because neither alone is sufficient:

1. **Line coverage of the full canonical replay** (`gate/_run/cov_replay.sh`) — the only
   admissible live/dead evidence. Reading cannot establish liveness (CLEANUP-2 F1).
2. **Whole-history caller search** — `git grep` across every reachable commit, not just HEAD.
   Coverage says "not executed on the canonical"; the caller search distinguishes *"unused
   because the system is healthy"* from *"unused because nothing has ever called it"*.

### Coverage of `crawlbot/planning/`, post-CLEANUP-18

```
Name                                      Stmts   Miss  Cover
crawlbot/planning/__init__.py                 5      0   100%
crawlbot/planning/coarse_preplanner.py      242     45    81%
crawlbot/planning/contact_scheduler.py      120     16    87%
crawlbot/planning/locomotion_planner.py      72     60    17%
crawlbot/planning/sequence_loader.py        107    107     0%
crawlbot/planning/swing_planner.py          138      7    95%
crawlbot/planning/torso_planner.py          132     25    81%
```

Worth recording as evidence the last two passes did what they claimed:
**`swing_planner.py` went from 307 statements / 163 dead / 47 % (CLEANUP-16) to 138 / 7 / 95 %**,
and `torso_planner.py` from 221 / 99 / 55 % to 132 / 25 / 81 %. The dead code was removed, not
merely moved.

(`sequence_loader.py` at 0 % and `locomotion_planner.py` at 17 % are both deliberate KEEPs —
see §7.)

---

## 2. The finding: `from_heuristic` is dead, and four comments say otherwise

`CoarsePlanResult.from_heuristic` is **83 lines / 20 statements** — 57 % of the
`CoarsePlanResult` dataclass (145 lines), and the single largest dead block in either file. Its
docstring says:

> **Test fixtures only** — production paths always call `CoarsePrePlanner.solve()`, and
> `sim_loop.py` skips the step if the solve fails rather than falling back here.

Four separate comments repeat the claim that unit tests use it:

| site | text |
|---|---|
| `sim_loop.py:1369` | *"No heuristic fallback inside sim_loop — from_heuristic is a test fixture only."* |
| `sim_loop.py:1599` | *"`CoarsePlanResult.from_heuristic` exists for unit tests only and is NOT called here"* |
| `sim_loop.py:1633` | *"CoarsePlanResult.from_heuristic is test-only and must not appear on this code path"* |
| `config.py:207` | *"unit tests that want to avoid the IPOPT dependency use `CoarsePlanResult.from_heuristic()` directly"* |

**No test calls it.** The string `heuristic` does not appear anywhere in `tests/`. Nor does any
script. Nor is there dynamic access (`getattr`, string dispatch).

Searched across **every reachable commit**, not just HEAD:

```
git grep -n "\.from_heuristic(" $(git rev-list --all)
  -> only ever  config.py:207:  # CoarsePlanResult.from_heuristic() directly.   (a comment)
```

It has **never had a caller in the history of this repository**. It is a fixture written for
tests that were never written.

The comments are not merely redundant — they are load-bearing and false. `sim_loop:1632-1640`
**re-implements `from_heuristic`'s envelope formula inline**, and the comment justifying that
duplication cites the existence of the test fixture. Remove the method and those comments must
be rewritten to keep the real design statement (*a failed NLP skips the step; there is
deliberately no silent heuristic fallback*) while dropping the false premise.

---

## 3. The rest of `coarse_preplanner.py` — the CLEANUP-16 claim holds

The 45 dead statements decompose exactly as:

| owner | stmts | lines |
|---|---|---|
| `from_heuristic` | **20** | 215–252 |
| `solve` | 12 | 453, 457, 459, 500–502, 511–514, 519–520 |
| `build` | 7 | 350–356 |
| `hw_at_knots` | 4 | 160–162, 167 |
| `L_com_at` | 1 | 139 |
| `_interp` | 1 | 143 |

Every region outside `from_heuristic` is a health-conditional branch. **KEEP all of it.**

| region | lines | why dead | class |
|---|---|---|---|
| `except RuntimeError` → IPOPT failure handler | 500–502 | all six solves succeeded | (b) healthy-system |
| `except Exception:` → `X_init` fallback | 511–514 | `source.value()` never raised | (b) |
| `except Exception:` → `iter_count = 0` | 519–520 | same | (b) |
| `if not self._built: self.build()` | 453 | `sim_loop:421` builds explicitly at setup — **and `test_lazy_build` covers it** | (b) defensive |
| `if T_step is None` / `if h_max is None` | 457, 459 | `sim_loop` always passes both | (b) API default |
| `_interp` lower clamp | 143 | reference never evaluated at/below `t_grid[0]` | (b) |
| cruise-accel block | 350–356 | `a_cruise_max = 0.0` | **(a) opt-in — see §5** |

Note `solve`'s dead set is *entirely* the failure-handling ladder plus two API defaults. That is
the signature of a healthy system, and it is exactly what CLEANUP-16 predicted.

`hw_at_knots` (15 lines) and `L_com_at` (3 lines) are production-dead but **KEEP**:
`hw_at_knots` is the momentum-box residual check used by `test_coarse_preplanner.py:139/150/296`
— test infrastructure for a real invariant, not sediment. `L_com_at` is one of three symmetric
accessors (`r_com_at` / `v_com_at` / `L_com_at`); deleting one of three for 3 lines costs more
in readability than it saves.

---

## 4. `contact_scheduler.py` — essentially clean

349 lines, 16 dead statements. Measured owner by owner — **15 of 16 are validation and
fallback**, all KEEP:

| owner | stmts | what |
|---|---|---|
| `read_anchors_from_mujoco` | 5 | `ImportError`, `except Exception: break`, `RuntimeError("No anchor sites")` |
| `make_anchor_grid` body | 3 | reached only via the `__init__` fallback below |
| `set_step_duration` | 2 | `IndexError` (bad idx) / `ValueError` (non-positive `T_step`) |
| `plan_traversal` | 2 | the two `break`s — anchor grid exhausted |
| `__init__` | 1 | `anchors_a is None` → `make_anchor_grid()` |
| `plan` property | 1 | `RuntimeError("Call plan_traversal() first.")` |
| `phase_at` | 1 | the exactly-at-end edge-case return |

`make_anchor_grid` is dead *on the canonical* (anchors are read from MuJoCo) but live in
`Misc/scripts/diag_m7_swing_velocity.py` — dead-here is not dead-everywhere.

**One genuinely dead public method:** `contact_sequence_over_horizon` (19 lines, 1 statement,
zero callers in `crawlbot/`, `scripts/`, or `tests/`). It reads as NMPC-horizon plumbing that
the NMPC never took up.

Everything else in the file is live: `anchor_se3` (4 call sites), `contact_config_at` (4+),
`set_step_duration`, `phase_at`, `plan_traversal`, `read_anchors_from_mujoco`.

This file does not need a cleanup pass. One 19-line method is not worth a gate cycle on its own.

---

## 5. Rule-5 gap: five more silent canonical values

`CoarsePrePlannerConfig` fields that `sim_loop` **never overrides**, so the dataclass default
*is* the canonical value — and none is in CLAUDE.md:

| field | default | what it does |
|---|---|---|
| `eps_v_terminal` | 5e-3 m/s | **hard box** on terminal CoM velocity of every step's plan |
| `eps_L_terminal` | 5e-2 Nms | **hard box** on terminal centroidal momentum |
| `w_v_terminal` | 1e2 | soft penalty on the same residual |
| `w_L_terminal` | 1e2 | soft penalty on the same residual |
| `ipopt_tol` | 1e-6 | NLP convergence tolerance |

The first four set the terminal boundary condition of *every* pre-planner solve — the constraint
that decides where each step is allowed to end. That is a physics parameter living in a
dataclass default, the same class as the eight `WholeBodyQPConfig` values in
`CLEANUP_CARRYOVER` §C4, and it should be merged into that item.

Separately, `T_step_default = 6.0` is **never used** — `sim_loop` always passes `T_step`
explicitly, and its own comment at `sim_loop:403` calls it "only a bootstrap". Its `if T_step is
None` branch is dead.

### The `a_cruise_max` question — needs a ruling, not a decision

The cruise-accel block (`348-356`, M7 v21) is gated on `a_cruise_max > 0.0`. `SimConfig.
preplanner_a_cruise_max` defaults to `0.0`, is **not** exposed by `dca.main` or
`run_m7_single_step`, and CLAUDE.md records it as *"CoM shaping — a_cruise_max=0.0 (off)"*.

So it is reachable only by hand-editing `SimConfig`: research sediment by the chantier's usual
test. But it is also a *documented* CLAUDE.md parameter, and removing it would delete a
documented capability rather than dead plumbing. Flagging rather than deciding — this is
Idriss's call, not a measurement question.

---

## 6. Removal plan

| # | target | lines | risk | note |
|---|---|---|---|---|
| 1 | `CoarsePlanResult.from_heuristic` + the 4 false comments | **~85** | **lowest** — never called in repo history | comments must be rewritten, not just deleted (§2) |
| 2 | `ContactScheduler.contact_sequence_over_horizon` | 19 | low | zero callers |
| 3 | `CoarsePrePlannerConfig.T_step_default` + its `None` branch | ~4 | low | unused default |
| 4 | cruise-accel block + its 2 config fields | ~12 | low | **blocked on a ruling** (§5) |

Total ≈ 120 lines — an order of magnitude less than CLEANUP-17/18, which is the honest finding:
**these two files are close to clean.** The CLEANUP-16 characterisation was right; the only
thing it missed is that the single largest dead block here is not a fallback branch at all, but
a test fixture with no tests.

Steps 1–3 are ordinary. Step 4 is a product decision.

---

## 7. After this, `crawlbot/planning/` is audited end to end

| file | lines | state |
|---|---|---|
| `coarse_preplanner.py` | 539 | audited — ~105 removable (§6) |
| `torso_planner.py` | 480 | cleaned (CLEANUP-17/18) |
| `contact_scheduler.py` | 349 | audited — 19 removable |
| `swing_planner.py` | 337 | cleaned (CLEANUP-17/18) |
| `sequence_loader.py` | 254 | KEEP — backs `sim.setup(sequence_path=…)` |
| `locomotion_planner.py` | 205 | KEEP — M0/Lutze paper baseline (CLEANUP-18 §3) |
| ~~`constrained_geodesic.py`~~ | ~~470~~ | deleted (CLEANUP-17) |

The module went from **3258 lines to 2175** across CLEANUP-17/18 (measured: `git show
<pre-17>:crawlbot/planning/` vs HEAD), a **33 %** reduction, with the remaining removable surface
down to ~120 lines and the two cleaned files at 95 % / 81 % coverage.

The next meaningful target in the chantier is `sim_loop.py`'s `_step` decomposition, which
`CLEANUP_CARRYOVER` §A already scopes and which needs its own coupling measurement first.
