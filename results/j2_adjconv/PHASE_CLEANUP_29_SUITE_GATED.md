# PHASE CLEANUP-29 — the suite is a gate

Two items, per the recommendation at the end of CLEANUP-28: retire the
`test_mid_waypoint_reshape` trio, then gate the suite without waiting on the
`crawlbot/diagnostics/` decision.

**The first item was wrong, and finding out why is the more useful half of this
report.** The trio was not testing a removed feature. It was a citation that
CLEANUP-21 forgot to rewrite, and the cost of not checking was seven disabled
tests carried across six passes of the chantier.

---

## 1. The reshape trio: repaired, not retired

The recommendation rested on an inherited claim. Its history:

| pass | what it said about the 3 errors | checked? |
|---|---|---|
| CLEANUP-25 | listed them as `FileNotFoundError`, "fixtures for the reshape path removed in CLEANUP-15" | no |
| CLEANUP-26 §3 | *"the same class as §2 and probably the same verdict, but it deserves its own look rather than being swept along"* | **explicitly deferred** |
| CLEANUP-27 | classified class D, "testing removed or disabled features", clear retiree | no — inherited |
| CLEANUP-28 reco | "retire the reshape trio (mechanical — fixtures deleted in CLEANUP-15)" | no — inherited again |

Two minutes of checking:

```
$ find . -name step2_ss_entry_fixture.npz
./Misc/runs/q1_q2/step2_ss_entry_fixture.npz          # 3 KB, git-tracked

$ python3 -c "import crawlbot.core.ik as ik; ..."     # the 7 imported symbols
manipulability_config_mid_waypoint   present
manipulability_config_trajectory     present
check_path_feasibility               present
_interpolate_q_quintic               present
_get_tool_frames                     present
_arm_v_slice                         present
_sigma_min_pair                      present
```

Nothing was removed. **CLEANUP-21 moved the fixture out of `diagnostic/` and
rewrote its citation in the documentation (`IK_FORMULATION.md:684`) but not in
either test file.** Both still pointed at the pre-migration path.

### The cost, which was larger than the visible symptom

| file | consumers of the fixture | symptom |
|---|---|---|
| `tests/test_mid_waypoint_reshape.py` | 3 of 4 tests | 3 hard **errors** — visible |
| `tests/test_ik_anomaly_regression.py` | **4 of 4** tests | 4 **silent skips** — invisible |

The second file guards its fixture with `pytest.skip(...)` when the file is
absent, so it reported success by not running. The suite's entire *"4 skipped"*
count — carried in every report since CLEANUP-25 as if it were a deliberate
condition — was this one stale path.

**Seven tests, one line.** Repaired by moving the fixture to
`tests/fixtures/step2_ss_entry_fixture.npz` — under `tests/`, so it cannot be
separated from its only consumers, and so it survives the eventual `Misc/`
deletion — and pointing both files at it:

```
$ pytest tests/test_mid_waypoint_reshape.py tests/test_ik_anomaly_regression.py
8 passed in 57.16s
```

`IK_FORMULATION.md` now cites the new location and names both consumers.

### The pattern, for the fourth and seventh time

This is the **fourth** CLEANUP-21 citation miss — after the string-literal-only
first pass, the `os.path.join(...)` class, and the f-string branches. Each was
found by widening the instrument rather than by re-reading the code.
`gate/link_audit.py` did not catch this one either: it audits citations in
**prose**, and this was a path assembled in Python. That is a real hole and it is
now known.

It is also the **seventh** instance in this chantier of an inherited claim
failing on measurement — and the second in three passes where the claim was
mine rather than the repository's. The tell was on the page the whole time:
CLEANUP-26 wrote *"deserves its own look"*, and nobody looked. A deferral that
is not tracked becomes a fact.

---

## 2. The last failure: made visible, not green

`test_far_infeasible_under_tight_rate` is the only failure that predates the
chantier (proven at `4e2e8da^` in CLEANUP-28). Its subject is a genuine open
question — whether the far case *should* be infeasible at `tau_w_max = 2.5` —
already on CLAUDE.md's Remaining Work.

Marked `@pytest.mark.xfail(strict=True)` with that reasoning in the `reason=`
string, not deleted and not loosened. `strict=True` is the point: if the far
case ever goes infeasible again, the test **fails** rather than quietly turning
green, because that would mean the envelope semantics changed and nobody wrote it
down.

`gate/run_suite.py` reports xfails on their own line rather than folding them
into the skip count — junit files an xfail under `<skipped>`, and "1 skipped"
reads as "1 test not run" when it means "1 documented open question, behaving as
documented". Precisely the reporting sloppiness that let the 4 real skips hide.

---

## 3. `gate/run_suite.py` — the suite as an enforceable gate

```
PYTHONPATH=. python3 gate/run_suite.py --fast          # per-commit
PYTHONPATH=. python3 gate/run_suite.py                 # pre-merge
PYTHONPATH=. python3 gate/run_suite.py tests/test_x.py # while iterating
```

Design decisions and why:

| decision | reason |
|---|---|
| forces `MUJOCO_GL=disabled` | under `osmesa` this container aborts pytest **collection** (`PyOpenGL: 'NoneType' has no attribute 'glGetError'` via `test_diagnostics`). An aborted collection is indistinguishable from a broken suite in CI output. CLAUDE.md's startup step 3 told everyone to use `osmesa`, so the documented command could not run the suite at all |
| parses `--junitxml`, not the summary line | counts come from a machine-readable artifact; the verdict lands in `gate/_run/suite_verdict.json` next to the other gate verdicts |
| PASS = 0 failed **and** 0 errors **and** 0 XPASS | an unexpectedly-passing strict xfail gets its own message telling the reader not to just delete the marker |
| `--durations=25` always on | a test that drifts past ~5 s shows up in the gate's own output rather than silently inflating the per-commit run until people stop running it |
| a path argument narrows the run | and the mode line then says `NARROWED … not a gate run`, so a narrowed run cannot be mistaken for a gate pass |
| `--strict-markers` in `pytest.ini` | a typo'd marker silently selects nothing. For `-m "not slow"` that means silently running a test the fast gate meant to skip — or skipping one it meant to run |

### Proven to bite — three modes, before being trusted

Per the chantier's standing rule that a new checker is proven, not assumed:

| injected | gate response |
|---|---|
| malformed edit → collection error | `VERDICT: FAIL`, names the module, `errors: 1` |
| `assert 1 == 2` in a real test | `VERDICT: FAIL`, `[failure] tests.test_frame_conversions::test_gate_bite_probe`, `failed_ids` in the JSON |
| `xfail(strict=True)` that passes | `VERDICT: FAIL` **and** routed to the XPASS message, not the failure list |

All three probes reverted; `git diff` on the probe file is empty.

---

## 4. The fast / slow split — and what the durations exposed

Marked from measurement (`--durations=25` on the full run), at the >5 s line
`pytest.ini` documents. The distribution is not close to uniform:

| test | s | marked |
|---|---:|---|
| `test_trajectory_aware_ik::test_chain_consistency` | **526.17** | slow |
| `test_reworked_qp::TestMomentumTask::test_jdot_assembly_and_per_axis_tracking` | 73.45 | slow |
| `test_trajectory_aware_ik::test_trajectory_ik_improves_worst_case` | 31.02 | slow |
| `test_trajectory_aware_ik::test_trajectory_ik_matches_endpoint_for_k1` | 23.48 | slow |
| `test_ik_anomaly_regression` ×4 (module-level mark) | 11.82 / 11.43 / 5.95 / 5.69 | slow |
| `test_mid_waypoint_reshape` ×3 | 10.05 / 9.47 / 9.44 | slow |
| everything else (200 items) | ≤ 3.24 each | — |

**11 tests hold 718 of the 743 s — 97 %.** One test holds 526 s, 71 % of the
entire suite on its own. Deselecting the 11 gives:

```
full :  211 tests, 210 passed, 0 failed, 0 errors, 0 skipped, 1 xfail   743 s
fast :  200 tests, 199 passed, 0 failed, 0 errors, 0 skipped, 1 xfail    23 s
```

**32× faster while still running 200 of the 211 items (95 %).** That is why the
per-commit gate is worth having: the split costs almost no coverage because the
cost is almost all in one place.

### The figure that matters more than the split

Grouping the marked tests by subject rather than by file:

```
test_trajectory_aware_ik   3 tests   580.7 s
test_ik_anomaly_regression 4 tests    34.9 s
test_mid_waypoint_reshape  3 tests    29.0 s
                                     ------
                                      644.5 s  =  87 % of the whole suite
```

All ten exercise the **manipulability-IK path** — `manipulability_config_trajectory`,
`manipulability_config_mid_waypoint`, `check_path_feasibility`,
`precompute_torso_map`. Measured call sites for those four inside `crawlbot/`,
excluding `ik.py` itself:

```
manipulability_config_trajectory    0   (the one grep hit, sim_loop.py:1407, is a COMMENT)
manipulability_config_mid_waypoint  0
check_path_feasibility              0
precompute_torso_map                0
```

**Zero.** So 87 % of the suite's runtime tests code the canonical controller
never calls — orphaned from `sim_loop` by CLEANUP-15 and kept on the correct
CLEANUP-16 principle that unused ≠ retired.

That is not an argument for deleting the tests; it is the decision-relevant
number for the *code*. CLEANUP-27 filed `test_trajectory_aware_ik` as "fate
follows its subject" and left the subject undecided. This quantifies the cost of
leaving it undecided: 11 of the 12 minutes.

---

## 5. Result

The suite is green and enforced.

| | before CLEANUP-28 | after CLEANUP-28 | now |
|---|---|---|---|
| failed | 9 | 1 | **0** |
| errors | 3 | 3 | **0** |
| skipped | 4 | 4 | **0** |
| xfail (tracked open question) | 0 | 0 | **1** |
| passed | 196 | 203 | **210** |
| gated? | no | no | **yes** |

Nothing was retired to get there. The 9 → 0 came from porting 6 tests
(CLEANUP-28), repairing 7 (§1), and marking 1 open question honestly (§2).

```
PYTHONPATH=. python3 gate/run_suite.py --fast     PASS   200 tests,  23 s
PYTHONPATH=. python3 gate/run_suite.py            PASS   211 tests, 743 s
gate/sync_docs.py --check                         in sync
gate/verify_params.py                             14/14 rows OK
gate/verify_docs.py                               33 documents OK
gate/verify_roots.py                              all root expressions OK
gate/link_audit.py                                0 BROKEN BY MOVE; 12/119 DELETED/DANGLING
                                                  — identical to the pre-change baseline,
                                                    so the fixture move added no dangles
```

`crawlbot/` was **not** touched: this pass is `tests/`, `gate/`, `pytest.ini`,
CLAUDE.md and reports. The canonical run cannot have moved.

CLAUDE.md updated in two places — startup step 3 now calls
`gate/run_suite.py --fast` instead of raw `pytest` under `osmesa` (which could
not run the suite at all), and the Rule-15 routine gains step 5 with both modes
and the reason both gates are needed rather than either.

---

## 6. What gating does *not* settle

- **`crawlbot/diagnostics/` is still undecided**, and it is 626 of the suite's
  1039 unique statements (60 %). Gating went ahead without that decision on
  purpose: the drift this chantier keeps uncovering came from a suite nobody ran,
  not from a suite that was too small.
- **`link_audit.py` cannot see computed paths.** §1's miss was
  `os.path.join(_root, 'diagnostic', ...)`. Auditing assembled paths means
  either an AST pass over `os.path.join` literals or a convention that fixtures
  live in one directory. `tests/fixtures/` is now the start of the latter.
- **The suite still proves nothing about the paper's numbers.** That is
  `run_gate.py` + `dock_check.py`, and it stays that way. Two gates, different
  jobs: one protects the past, the other protects a change.
