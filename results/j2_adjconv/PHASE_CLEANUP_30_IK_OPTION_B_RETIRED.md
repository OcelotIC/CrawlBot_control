# PHASE CLEANUP-30 — the Option-B IK path retired; `diagnostics/` kept

The two decisions left open after CLEANUP-29. One was a genuine bet and went to
Idriss; the other resolved itself on measurement and never needed asking.

---

## 1. `crawlbot/diagnostics/` — KEEP, and the real finding

Framed as "626 statements, 60 % of the suite's unique coverage, and the canonical
never runs it". Two measurements settle it:

```
$ grep -rn run_diagnostics --include=*.py .        (live paths only)
scripts/run_m7_single_step.py:25, :294             <- one of only 5 live scripts
tests/test_diagnostics.py:206
```

and CLAUDE.md **Rule 3**:

> *Every simulation produces diagnostics. Call `run_diagnostics()` at the end of
> every sim. "It docked" is not a pass criterion.*

So the package has a live consumer **and** is mandated by project rule. There was
never a deletion question here.

**The real finding is why the canonical never runs it.** The gate's export step
invokes `scripts/diag_full_diag_export.py` (`run_gate.py:226`), a bespoke
exporter — not `run_diagnostics()`. So the canonical run, the one run that must
be defensible, is the one run that **does not honour Rule 3**. That is a wiring
gap, not dead code, and it explains the coverage result entirely.

Deliberately **not** fixed in this pass: changing the gate's own path in the same
commit that removes 695 lines from `ik.py` would destroy the gate's ability to
prove the removal inert. Carried to `CLEANUP_CARRYOVER` §A6.

---

## 2. The Option-B IK path — RETIRED

### The decision, and the evidence it rested on

| question | measured |
|---|---|
| callers in `crawlbot/` | **0** for all four. The one grep hit (`sim_loop.py:1407`) was a *comment* |
| lines executed by the canonical replay | **0** for all four |
| live consumers of any kind | their own 10 tests, plus scripts already retired to `Misc/` |
| cost to the suite | **644 s of 743 s — 87 % of total runtime** |
| does the live path depend on them? | **no** — `manipulability_config` (IK 2, `sim_loop.py:307`) touches no orphan |

The distinction that mattered: `sim_loop` calls **`manipulability_config`**, a
different and fully live function. The four retirees are IK 3 — the
*trajectory-aware* escalation built for the T15 step-2 path singularity, where an
endpoint-only IK can return a configuration whose interior passes near a
singularity. The capability that ships was never theirs.

### What was removed — computed, not chosen by eye

Removing the four approved functions strands four private helpers. Which ones is
a call-graph question, so it was answered with one:

```
ik.py defines 17 module-level functions
consumed from outside ik.py (excluding the leaving tests):
  dock_configuration, dock_configuration_fixed_rotation,
  manipulability_config, solve_ik

reachable from live roots            : 8   <- must stay
reachable ONLY from the four retirees: 8   <- removable
```

| removed | lines |
|---|---:|
| `check_path_feasibility` | 171 |
| `manipulability_config_mid_waypoint` | 167 |
| `manipulability_config_trajectory` | 143 |
| `precompute_torso_map` | 83 |
| `_ik_three_tasks` (stranded) | 51 |
| `_trajectory_worst_w` (stranded) | 34 |
| `_interpolate_q_quintic` (stranded) | 19 |
| `_sigma_min_pair` (stranded) | 10 |

```
crawlbot/core/ik.py   1469 -> 774 lines   (-695, 47 %)
statements            592  -> 328
```

Spans came from the AST and the result was `ast.parse`-checked before writing —
the CLEANUP-16 lesson, where a hand-computed span cut a signature in half.

### A fifth orphan, found and deliberately left

The same computation surfaced `solve_ik_waypoints` (118 lines, `ik.py:1352-1469`
pre-removal): reachable from **neither** the live roots nor the four retirees —
zero callers anywhere in `crawlbot/`, `scripts/` or `tests/`.

Not removed. The approved scope was four functions, and widening it silently is
how a scope becomes untraceable. Recorded in `CLEANUP_CARRYOVER` §A7 as the
obvious next candidate; it is the largest single block of the 173 statements in
`ik.py` the canonical still never reaches.

### One near-miss worth recording

`CLEANUP_CARRYOVER` §A3's lint list says `sim_loop.py:40` imports
`precompute_torso_map`, `solve_ik` and `solve_ik_waypoints` unused. Had that been
true, deleting `precompute_torso_map` would have broken `sim_loop` on import —
the exact CLEANUP-6 failure mode, where 9 deleted config fields took 8 tests down
with them.

It was checked rather than assumed. The list is **stale**: those imports were
already cleaned up (CLEANUP-9), and `sim_loop.py:40-42` now imports only
`dock_configuration`, `dock_configuration_fixed_rotation`, `manipulability_config`.
§A3 corrected.

### Tests: 11 retired, 1 rescued

```
tests/test_trajectory_aware_ik.py     (4)  ->  Misc/tests/
tests/test_ik_anomaly_regression.py   (4)  ->  Misc/tests/
tests/test_mid_waypoint_reshape.py    (3)  ->  Misc/tests/
tests/fixtures/step2_ss_entry_fixture.npz  ->  Misc/tests/fixtures/
```

The fixture follows because those three were its only consumers — the same
principle CLEANUP-26 applied, and the reason CLEANUP-29 put it under `tests/` in
the first place is now void.

**`test_torso_planner_piecewise_continuous` did not retire.** It was the 4th test
in `test_mid_waypoint_reshape.py` and exercises the **live** `TorsoPlanner` (C0
continuity at `t_mid`, `v = a = 0` at all three waypoints). Lifted into
`tests/test_planners_6d.py`, which already imports `TorsoPlanner` — the move
CLEANUP-27 recommended and nobody had executed. `17 passed`.

### Documentation, per Rule 15

`IK_FORMULATION.md` is now half a description of code that does not exist, which
is exactly how `docs/api/` died. It gets a banner rather than a rewrite:

| section | status |
|---|---|
| §1–§4 (config space, Jacobians, anchors, metrics) | live |
| §5 (IK 1 `dock_configuration_fixed_rotation`) | live |
| §6 (IK 2 `manipulability_config`) | live — what `sim_loop` calls |
| **§7–§9** (IK 3, its pathologies, the corrected formulation) | **RETIRED** |
| **§11** (implementation plan) | **RETIRED** — carried out, then retired |

§7–§9 are kept on purpose: they are the record of *why* the path was built and
the recipe for reviving it. Revival starts at
`git show d61e1a0:crawlbot/core/ik.py`, with the tests in `Misc/tests/`.

Also updated: `docs/crawlbot/core/ik.md` §4 (rewritten — "what is not on the
canonical path" went from six entries to two), the `ik.py` module docstring,
`Misc/tests/README.md`, and a 13-line comment block at `sim_loop.py:1402` that
described the Phase-4 wiring of a path removed twice over.

---

## 3. Verification

The removal is **provably inert** — not argued, measured:

```
gate/run_gate.py
  [1] canonical replay + export   rc=0 (127.2 s)
  [2] artifact identity           PASS — 2077 rows × 132 928 fields
  [3] two-model consistency       PASS — 15 links, 14 joints, 71.056 kg
  [4] environment pin             PASS
  VERDICT: PASS (128.7 s)

gate/dock_check.py
  at-weld docks 6/6 — 4.02 / 4.89 / 4.99 / 4.97 / 4.95 / 4.62 mm
  every step delta +0.0000 vs frozen; worst margin 0.01 mm
  theta_s 0.540  h_w 4.102 / 4.243  e_com 0.154  qp_fail 0
  CANONICAL RESULTS: MATCH frozen 2.5
```

695 lines out of the controller's IK module, and the six docks reproduce to four
decimal places.

```
gate/run_suite.py --fast    PASS   199 tests, 198 passed, 1 xfail,  23 s
gate/run_suite.py           PASS   200 tests, 199 passed, 1 xfail,  89 s
gate/sync_docs.py --check   in sync (3 documents regenerated)
gate/verify_docs.py         OK
gate/verify_params.py       14/14 rows OK
gate/verify_roots.py        OK
gate/link_audit.py          0 BROKEN BY MOVE
```

Coverage was regenerated **before** `sync_docs` (`bash gate/_run/cov_replay.sh`),
so the `canonical?` column reflects the post-removal architecture rather than the
previous one — the CLEANUP-15 near-miss that rule exists for.

One note for anyone diffing `cov.json` across this commit: `ik.py`'s *executed*
statement count also fell (234 → 155). That cannot be a behaviour change —
artifact identity is byte-exact — so it is a measurement artefact of the two
coverage runs, and it has not been chased further.

---

### This retires most of CLEANUP-29's fast/slow argument

The full suite went **743 s → 89 s**, 8.3× faster, because the 644 s it spent on
this path left with the path. The new duration profile:

```
66.71s  test_reworked_qp::TestMomentumTask::test_jdot_assembly_and_per_axis_tracking
 2.79s  test_diagnostics::TestRunDiagnostics::test_full_pipeline
 2.70s  test_diagnostics::TestGeneratePlots::test_creates_all_figures
 2.60s  test_reworked_qp::TestDSPassivity::test_energy_decay[settle]
 2.14s  ... and everything else below this
```

Exactly **one** test now exceeds 5 s. CLEANUP-29 justified the `--fast` split at
32× (743 s → 23 s); at 89 s → 23 s it is 3.9×, and the entire remaining gap is
that single T-MOM test. The split is kept — 23 s is still the better per-commit
figure and the marker costs nothing — but the honest position is that the full
suite is now cheap enough to run every time, and the argument for the split has
mostly evaporated. Worth revisiting if that one test is ever made cheaper.

---

## 4. Carried forward

| item | where |
|---|---|
| `solve_ik_waypoints` — 118 lines, zero callers, same class as the four | `CARRYOVER` §A7 |
| the canonical run does not honour Rule 3 (bespoke exporter, never `run_diagnostics()`) | `CARRYOVER` §A6 |
| §A3's lint list was stale for three `sim_loop` imports | corrected in place |

And one consequence to hold onto rather than file: `check_path_feasibility` was
the architecture's only interior path-feasibility guard, and it was **already
disconnected** — nothing called it. Retiring it removed no protection that was
running, but there is now no such check at all. The canonical anchor geometry does
not need one. A new one might, and the derivation is in §7–§9 when it does.
