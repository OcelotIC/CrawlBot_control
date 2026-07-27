# C2.2 — Solver instrumentation

**Review-Closure Bloc 2, Phase C2.2 (revised brief).** Logging-only. No control
path changed, and the canonical replay proves it.

---

## Header (mandatory)

| item | value |
|---|---|
| commit worked on | branch `claude/review-closure-bloc-2-uwu1x7`, parent **`1c3cc37`** (C3.4) |
| date | 2026-07-26 |
| python / mujoco / pinocchio | 3.11.15 / 3.10.0 / 3.9.0 |
| casadi + ipopt / numpy / scipy | 3.7.2 + IPOPT-MUMPS / 2.3.5 / 1.17.1 |
| env vs `gate/environment.lock` | exact match |
| host (now recorded, C2.2.4) | Intel Xeon @ 2.10 GHz, 4 logical CPUs, 15 GiB, Linux 6.18.5 |

**Artifacts produced**

- `results/review_closure/c2/c25_c2_fulldiag.csv` — 73 columns (66 + 7)
- `results/review_closure/c2/c25_c2_fulldiag_meta.json` — pre-planner + host
- `results/review_closure/c2/c2_neutrality_verdict.json`
- `results/review_closure/c2/c2_solver_stats_v2.json`
- scripts: `c2_neutrality_check.py`, `c2_solver_stats_v2.py`, `c2_docsync_repair.py`

---

## 1. Neutrality — **PASS, twice**

The brief's criterion is the 66 existing columns byte-identical with new
columns appended only. `gate/run_gate.py` cannot express that criterion: its
`diff_csv` fails on `R[0] != B[0]` (`gate/run_gate.py:265-267`), so **any**
appended column is a "header mismatch" FAIL regardless of the data. Rather than
weaken a shared gate, the new columns are **opt-in** (`--solver-diag`, default
OFF) and neutrality is shown two ways:

**(a) Flag OFF — `gate/run_gate.py`, unmodified:**

```
[4] environment pin           : PASS
[3] two-model consistency     : PASS  (15 links, 14 joints, total 71.056 kg)
[1] canonical replay + export : replay rc=0 (270.3s), export rc=0
[2] artifact identity         : PASS  (2077 rows × 132928 fields,
                                       excl ['nmpc_time_ms', 'qp_time_ms'])
VERDICT: PASS   (env PASS, 275.1s)
```

Identical field count to the C0 run. `gate/dock_check.py`: **6/6 at-weld, every
step `delta +0.0000`**, θ_s 0.540°, h_w 4.102/4.243, e_com 0.154, qp_fail 0 —
`CANONICAL RESULTS: MATCH frozen 2.5`.

**(b) Flag ON — `c2_neutrality_check.py`, the brief's criterion exactly:**

```
rows      : 2077
columns   : 66 baseline + 7 appended = 73
appended  : qp_solve_ms_sum, qp_solve_ms_max, qp_iter_sum, qp_n_solves,
            qp_n_failed, qp_status_worst, nmpc_status_str
excluded  : nmpc_time_ms, qp_time_ms   (wall-clock, per gate/EXCEPTIONS.md)
compared  : 132928 fields
VERDICT: PASS
```

The checker also fails if a baseline column *moves* (not just changes), which is
the control-path-leak test the brief asks for. `gate/run_suite.py` (full,
not `--fast`): PASS.

**Why default OFF, and what it defers.** Emitting the columns unconditionally
would break `gate/run_gate.py` for every caller until
`results/j2_adjconv/c25_fulldiag.csv` — a frozen paper artifact — is
regenerated. That regeneration, and whether the gate should instead learn to
tolerate appended columns, is a governance decision (`gate/EXCEPTIONS.md`
territory) and is not this stream's to take. Flipping the default is a
one-character change once that decision is made.

---

## 2. New-column dictionary

Appended after column 66, in this order. Present only with `--solver-diag`.

| # | column | type | content |
|---:|---|---|---|
| 67 | `qp_solve_ms_sum` | float | Σ `QPSolveInfo.solve_time_ms` over the tick's QP solves |
| 68 | `qp_solve_ms_max` | float | max over the tick's solves — the budget-relevant one |
| 69 | `qp_iter_sum` | int | Σ qpOASES working-set recalculations (`stats()['iter_count']`) |
| 70 | `qp_n_solves` | int | number of solves aggregated into this row |
| 71 | `qp_n_failed` | int | solves whose backend reported not-success, or which raised |
| 72 | `qp_status_worst` | int | worst outcome over the tick (ordering below) |
| 73 | `nmpc_status_str` | string | IPOPT `return_status`, verbatim |

`qp_status_worst` ordering, ascending severity:

| code | meaning |
|---:|---|
| `-1` | **not measured** — no solve offered to the accumulator. Never an outcome |
| `0` | the backend reported success |
| `1` | the backend reported **not** success — the case `qp_ok` cannot see (§3) |
| `2` | the solve raised |

**Sentinel convention.** A not-measured row carries
`qp_solve_ms_sum = qp_solve_ms_max = 0.0`,
`qp_iter_sum = qp_n_solves = qp_n_failed = 0`, `qp_status_worst = -1`.
**Test `qp_n_solves == 0`**, not the timers against 0.0 — testing timers is
exactly how the existing `nmpc_time_ms`/`qp_time_ms` convention traps readers.

**On the canonical there are zero sentinel rows.** 1368 ticks × 1 solve
(inter-step settle) + 709 ticks × 10 solves (SS, DS-terminal) = **8458 solves,
all recorded.** The inter-step loop was reachable from the same logging path, so
per the brief it is recorded there rather than left hardcoded.

Metadata added to `*_fulldiag_meta.json`, unconditionally (no gate byte-compares
it): `preplanner_stats` (C2.2.3), `host` and `library_versions` (C2.2.4).

---

## 3. What the instrumentation immediately revealed

### `qp_ok` is structurally incapable of reporting a QP failure

`_get_solver_options` sets `error_on_fail: False`, so CasADi **returns
normally** when the backend fails. But `_solve_qp_raw` derives its outcome from
whether the call *raised* (`hierarchical_qp.py`, `except RuntimeError`).
Verified directly: a QP with `lba = uba = 5` against `lbx = ubx = ±1` makes
qpOASES print *"Premature homotopy termination because QP is infeasible"* and
the Python call returns without raising, so `success = True`, `exitflag = 0`.

Combined with the hardcoded `log.qp_ok.append(True)` on the 1368 inter-step
ticks, **"0 QP failures" was never a claim about the run** — it is the only
value that column can take. C3.4 reported it with the coverage caveat; the
cause is worse than the coverage.

**Now it is a measurement.** `casadi.Function.stats()` carries the backend's own
verdict, read into three additive `QPSolveInfo` fields (`solver_success`,
`return_status`, `n_iter`). Result over the canonical:

```
qp_status_worst : {0: 2077}      (0 = backend reported success)
qp_n_failed     : 0              over all 8458 solves
```

Zero failures — now established rather than assumed, across the whole run
instead of a third of it. Note `n_iter` was **declared on `QPSolveInfo` and
never assigned**: it read 0 for every solve before this phase.

The honest fix — deriving `info.success` from `stats()` — is deliberately *not*
applied: it would change the frozen `qp_ok` column. The code comment says so and
points at the replacement.

### IPOPT status, now in the CSV

`nmpc_status_str` was in `sim_log.json` but not the CSV, so the committed
artifact could not distinguish the 5 degraded solves from the 704 clean ones:

```
Solve_Succeeded             704
Solved_To_Acceptable_Level    5
NMPC_BYPASSED              1368   (sentinel, not a solver outcome)
```

### Coarse pre-planner — six solves that appeared in no artifact

| | value |
|---|---|
| solves | 6 (one per step), **all `Solve_Succeeded`** |
| solve time | median **40.2 ms**, mean 48.2, max **99.0 ms**, total **289.3 ms** |
| IPOPT iterations | median 13, range 11–21 |

Per step: `99.0 ms/14 it` · `35.2/12` · `45.2/18` · `28.7/11` · `50.9/21` ·
`30.4/12`. Cost varies by two orders of magnitude across steps
(0.81 → 162.98), tracking the short/long swing alternation.

---

## 4. C3.4 table, re-run with true per-QP-solve timing

Replaces the `qp_time_ms / 10` upper bound. **Measured on this container**,
which C3.4 established runs ~25 % slower than the (unrecorded) machine that
produced the committed artifacts.

| quantity | n | median | mean | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| **QP solve, per-solve mean** [ms] | 2077 | **5.641** | 5.873 | 7.755 | 8.895 | 14.702 |
| **QP solve, worst in tick** [ms] | 2077 | 5.806 | 6.157 | 8.116 | 10.034 | **97.959** |
| **qpOASES iters per solve** | 2077 | **90.0** | 93.3 | 116.8 | 123.0 | 132.2 |
| QP share of the WBC block [%] | 709 | **70.9** | 70.7 | 73.6 | 74.7 | 84.3 |

Total qpOASES working-set recalculations over the traversal: **788 854**.

Per phase (per-solve mean):

| phase | ticks | solves | median [ms] | max [ms] | iters/solve median |
|---|---:|---:|---:|---:|---:|
| SS | 508 | 5080 | **5.091** | 8.012 | 85.6 |
| DS_terminal | 201 | 2010 | 6.349 | 14.702 | 115.1 |
| DS_interstep | 1368 | 1368 | 5.783 | 10.097 | 92.0 |

**C3.4's upper bound was good.** It estimated 7.26 ms median per solve from
`block/10`; the truth is 5.641 ms — the bound was 1.29× pessimistic, because
the QP is ~71 % of the block and the Pinocchio/MuJoCo/AOCS remainder is the
other ~29 %. Any earlier statement built on the bound was conservative, not
wrong.

---

## 5. Deliverable 3 — the 10 ms budget, answered

**No. The controller-only QP cost is not within budget on all ticks.** Stated
plainly, because the brief asks for the answer either way.

| check (dt_qp = 10 ms) | result |
|---|---|
| ticks containing a solve > 10 ms | **22 / 2077 (1.06 %)** |
| worst single solve | **97.959 ms (9.8× budget)** |
| worst excluding that one outlier | **13.937 ms (1.39× budget)** |
| ticks over budget, SS only | 3 / 508 (**0.59 %**) |
| ticks over budget, DS_terminal | 18 / 201 (**8.96 %**) |
| ticks over budget, DS_interstep | 1 / 1368 (0.07 %) |

And at the NMPC rate, controller-only (all QP solves + the NMPC solve, with
physics and logging removed): **19 / 2077 ticks over the 100 ms budget**, median
6.5 ms.

Three qualifications, all of which cut in the paper's favour but none of which
makes the answer yes:

1. **The exceedances are marginal and concentrated in the trailing settle.**
   18 of 22 are in `DS_terminal` — the 20 s station-keeping tail after the
   traversal — whose per-solve iteration count is ~35 % above SS's (115 vs 86).
   In **SS, the phase that does the locomotion**, 0.59 % of ticks contain a
   solve over budget and the worst is 13.94 ms.
2. **One outlier dominates the maximum.** The 97.959 ms solve at t = 65.54 s is
   **7× the next-worst** in the entire run. It is not the first solve of the run
   nor of its phase, so a cold `ca.conic` construction is *possible* — the
   solver cache is keyed `(n_vars, n_constraint_rows)` at
   `hierarchical_qp.py:453`, and the passivity constraint adds a row when it
   toggles, which would mint a new key mid-settle — but **this is a hypothesis,
   not established from this data.**
3. **This container is slow.** C3.4 measured it ~25 % above the artifact
   machine on identical work. On the reference machine the picture would be
   better — by how much is unmeasurable, because that machine is unrecorded.
   `environment['host']` now closes that gap for every future run.

Per the brief, whether the paper makes a real-time feasibility claim is not a
consequence of these numbers being favourable. What the data supports is a
statement of the form *"median controller-side QP cost is 5.6 ms against a
10 ms budget, with 1.1 % of ticks exceeding it, concentrated in the
station-keeping tail"* — not an unqualified real-time claim.

---

## 6. ⚠ Blocker found in the mandated documentation routine (hygiene stream)

C0 reported that `gate/sync_docs.py --check` fails on a fresh clone. Complying
with Rule 15 this phase revealed something worse: **the fix it tells you to run
destroys data.**

`load_cov()` prints

```
note: gate/_run/cov/cov.json absent — coverage columns left as-is.
      regenerate with gate/_run/cov_replay.sh
```

and then does the opposite. `build_blocks` sets `cov = None`, so the header
becomes `canonical coverage **not measured**` and `live()` returns `—` for every
symbol. Running the mandated `PYTHONPATH=. python3 gate/sync_docs.py` on this
branch rewrote **26 documents, −335/+348 lines, all of it coverage data replaced
by placeholders.** Neither `cov.json` nor the `cov_replay.sh` that would rebuild
it is tracked, so this is the behaviour on **every** clone. (`coverage` is not
installed here either, so the file cannot be reconstructed without a further
dependency and a full instrumented replay.)

Second defect, found the same way: **`gate/sync_docs.py` has no
`if __name__ == '__main__'` guard.** Its entire body runs at module level, so
`import gate.sync_docs` — which any tooling around it would do — rewrites all 26
documents as an import side effect.

Both are `gate/` property. Per the brief's cross-stream rule: reported, not
fixed. Rule 15 was satisfied instead by `c2_docsync_repair.py`, which harvests
the committed coverage cells via `git show HEAD:<doc>` **before** triggering the
import, regenerates the structural half for the four changed modules only, and
substitutes the coverage back by symbol name. New symbols get a cell only where
it is measured (`add_raised` → `not exercised`, because the canonical has zero
QP failures); nothing is invented. The other 22 documents are restored untouched.

`gate/verify_docs.py`, `verify_params.py`, `verify_roots.py`, `link_audit.py`:
all PASS. `CLAUDE.md`'s parameter table needed three anchors moved
(`sim_loop.py:951 → :957` for α torque-min and α accel-reg,
`hierarchical_qp.py:98 → :112` for ε) — caught by `verify_params.py`, which is
the one checker that does bite.

---

## 7. Files changed

| file | change |
|---|---|
| `crawlbot/solvers/hierarchical_qp.py` | `QPSolveInfo`: +`solver_success`, +`return_status`; `_solve_qp_raw` reads `stats()`; `_solve_weighted` populates them and `n_iter`. Legacy `success`/`exitflag`/`cost` untouched |
| `crawlbot/simulation/logging.py` | `SimLog`: +6 `qp_*` lists, +`preplanner_stats`; `capture_environment()`: +`host` |
| `crawlbot/simulation/tick_logging.py` | +`QPStatAccumulator`, +`_qp_stats_from_info`; `TickState` +6 fields; both recorders log them |
| `crawlbot/simulation/sim_loop.py` | accumulator in the WBC sub-loop and the inter-step loop; `run()` persists `_preplanner_stats` |
| `scripts/diag_full_diag_export.py` | `--solver-diag` (default OFF) appends 7 columns; meta gains `preplanner_stats`, `host`, `library_versions` |
| `docs/crawlbot/{solvers/hierarchical_qp,simulation/logging,simulation/tick_logging,simulation/sim_loop}.md` | Rule 15 — measured half regenerated, prose half written |
| `CLAUDE.md` | 3 parameter-table anchors moved |

---

## STOP

C2.2 is complete: neutrality PASS by both routes, seven columns appended with a
documented sentinel convention, the C3.4 table re-run on true QP timings, and
the 10 ms budget question answered — **not within budget on all ticks**, 1.06 %
exceedance, 0.59 % in SS, one unexplained 98 ms outlier.

Two items for Idriss:

1. **Governance:** flipping `--solver-diag` on by default requires regenerating
   `c25_fulldiag.csv`, or teaching `gate/run_gate.py` to allow appended columns.
2. **Hygiene stream:** `gate/sync_docs.py` erases coverage data on every clone
   and rewrites all documents on import. The mandated Rule-15 routine currently
   degrades the repository it is meant to protect.
