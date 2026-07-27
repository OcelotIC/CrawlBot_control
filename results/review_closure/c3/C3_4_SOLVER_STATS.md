# C3.4 — Solver statistics for the paper

**Review-Closure Bloc 2, Phase C3.4.** Post-processing only — **no new physics
run**. Every number below comes from artifacts already committed, plus the C0
replay whose 64 non-timing columns are byte-identical to the committed
canonical.

---

## Header (mandatory)

| item | value |
|---|---|
| commit worked on | **`d6e7d77`** (C1), branch `claude/review-closure-bloc-2-uwu1x7`; controller code identical to `eecbf94` |
| date | 2026-07-26 |

**Artifacts cited**

- `results/j2_adjconv/c25_fulldiag.csv` — committed canonical, managed (C)
- `results/j2_adjconv/u25_fulldiag.csv` — committed canonical, unmanaged (U)
- `gate/_run/c25_replay_fulldiag.csv` — the C0 replay **on this container**
- `results/gate_run_scratch/sim_log.json` — IPOPT return-status strings
- `results/review_closure/c3/c3_4_solver_stats.py` → `c3_4_solver_stats.json`

---

## 0. Read this before quoting any number

Three facts about how these channels are measured govern the whole table. All
three were verified in source, and two of them invalidate the obvious reading.

**(a) `qp_time_ms` is not a QP solve time.** The timer opens at
`crawlbot/simulation/sim_loop.py:2316` and closes at `:2947`. Between them sits
`for qs in range(self.n_qp_per_nmpc):` at **`:2326`**, with
`n_qp_per_nmpc = round(dt_nmpc / dt_qp) = 10` (`sim_loop.py:102`). The interval
therefore contains **ten** QP solves *plus* ten Pinocchio `computeAllTerms`
updates, ten torso/EE reference constructions, ten AOCS evaluations, ten
MuJoCo `mj_step` calls and the per-tick logging. It is a **WBC block time**.
Quoting it as "QP solve time" overstates by ~10× plus the physics. Divide by
10 for a per-solve **upper bound**, which is what the table reports.

**(b) `nmpc_time_ms` is a fair NMPC-solve proxy.** Timer `sim_loop.py:2198` →
`:2256`, wrapping `l_com_reference_at`, the IPOPT `nmpc.solve()` and
`compute_feedforward_acceleration`. Dominated by the solve.

**(c) One third of the ticks must be excluded, or every statistic collapses.**
On `DS_interstep` ticks the NMPC is bypassed and **both timers are written as
`0.0` sentinels** (`crawlbot/simulation/tick_logging.py:318-319`). That is
1368 of 2077 rows in C and 1283 of 1905 in U — **66 % and 67 %**. The script
asserts the bypass set is exactly the `DS_interstep` phase rather than assuming
it. Including those rows would report a median NMPC time of 0.00 ms.

**(d) Wall-clock is machine-dependent and the committed values' machine is
unrecorded.** `qp_time_ms` and `nmpc_time_ms` are the two columns the gate
excludes from byte-comparison precisely because they are nondeterministic
(`gate/EXCEPTIONS.md`). The table below therefore reports the committed run
**and** a re-measurement on hardware this session can describe; see §3.

---

## 1. Hardware and software

**This container** (where `C_replay_here` was measured):

| item | value |
|---|---|
| CPU | Intel(R) Xeon(R) Processor @ 2.10 GHz |
| architecture | x86_64, **4 vCPU**, 1 thread/core, 1 socket |
| notable ISA | AVX-512 (f/dq/cd/bw/vl, vnni, bf16), AMX, SHA-NI |
| RAM | 15 GiB |
| OS / kernel | Linux 6.18.5 |

**The machine that produced `c25_fulldiag.csv` / `u25_fulldiag.csv` is not
recorded anywhere in the repository.** No committed artifact, metadata JSON or
report carries a hardware string; `gate/environment.lock` pins the *software*
stack only. This is a gap the paper must close by other means — see §5.

**Software stack** (identical for all three columns; matches
`gate/environment.lock` exactly, verified in C0):

| library | version |
|---|---|
| Python | 3.11.15 |
| MuJoCo | 3.10.0 |
| Pinocchio (`pin`) | 3.9.0 |
| CasADi | 3.7.2 |
| NLP solver | **IPOPT** via CasADi `nlpsol`, linear solver **MUMPS** |
| QP solver | **qpOASES** via CasADi `conic` (`WholeBodyQPConfig.solver = 'qpoases'`, `wholebody_qp.py:86`) |
| NumPy / SciPy | 2.3.5 / 1.17.1 |
| qpsolvers / OSQP | 4.13.0 / 1.1.3 (installed, not on the canonical path) |

**Problem sizes**, for the table's caption:

| solver | rate | decision vars | horizon | notes |
|---|---|---|---|---|
| Centroidal NMPC | 10 Hz (`dt_nmpc = 0.1 s`) | nx = 9, nu = 12, **N = 8**, RK4 multiple shooting | 0.8 s | `sim_loop.py:416-418` |
| Whole-body QP | 100 Hz (`dt_qp = 0.01 s`) | **z ∈ ℝ⁴⁶** = q̈_t(6) + q̈(14) + λ(12) + τ_q(14), plus 6 h_w slacks ⇒ **52** | instantaneous | `wholebody_qp.py:914-921` |
| Coarse pre-planner | once per step (6 total) | M = 15 collocation intervals, IPOPT NLP | per-step | `config.py:208` |

---

## 2. The table

Statistics over **NMPC-active ticks only** (SS + DS_terminal). Percentiles are
linear-interpolated and were cross-checked against `numpy.percentile` (exact
agreement).

### 2.1 Managed canonical (C) — the paper's run

| quantity | n | median | mean | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| **NMPC solve** [ms] | 709 | **16.87** | 16.97 | 22.75 | 27.75 | 38.56 |
| **WBC block** (10 QP + physics) [ms] | 709 | **56.92** | 59.34 | 72.01 | 85.22 | 107.12 |
| ↳ per-QP-solve **upper bound** [ms] | 709 | **5.69** | 5.93 | 7.20 | 8.52 | 10.71 |
| **NMPC IPOPT iterations** | 709 | **11** | 10.87 | 15 | 17 | 18 |

Per phase:

| phase | n | NMPC median / p95 / max [ms] | WBC median / p95 / max [ms] | iters median / max |
|---|---:|---|---|---|
| SS | 508 | 18.53 / 23.82 / 38.56 | 55.13 / 64.76 / 107.12 | 13 / 18 |
| DS_terminal | 201 | 12.80 / 16.85 / 19.93 | 66.10 / 74.83 / 94.40 | 7 / 9 |

### 2.2 Unmanaged (U) — for the ablation caption

| quantity | n | median | mean | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| NMPC solve [ms] | 622 | **12.18** | 13.74 | 23.12 | 29.48 | 44.94 |
| WBC block [ms] | 622 | 57.26 | 59.64 | 72.06 | 93.84 | 136.00 |
| ↳ per-QP-solve upper bound [ms] | 622 | 5.73 | 5.96 | 7.21 | 9.38 | 13.60 |
| NMPC IPOPT iterations | 622 | **7** | 8.56 | 16.95 | 21 | 23 |

### 2.3 Solver outcomes — zero failures, but the CSV cannot say so

| quantity | C | U |
|---|---:|---:|
| NMPC solves attempted | 709 | 622 |
| `nmpc_ok == False` among **NMPC-active** ticks | **0** | **0** |
| `nmpc_ok == False` in the raw CSV (**all** rows) | 1368 | 1283 |
| `qp_ok == False`, all rows | **0** | **0** |
| ticks where QP status is **not recorded at all** | 1368 | 1283 |

IPOPT return status, from `sim_log.json` (C):

| `return_status` | count |
|---|---:|
| `Solve_Succeeded` | **704** |
| `Solved_To_Acceptable_Level` | **5** |
| `NMPC_BYPASSED` (sentinel, not a solver outcome) | 1368 |
| any failure / restoration status | **0** |

⚠ **Two traps in this block.**

1. **`nmpc_ok == False` on 1368 rows is not 1368 failures.** The bypass path
   writes `log.nmpc_ok.append(False)` with the in-source comment *"not run; not
   a failure"* (`tick_logging.py:317`). A naive CSV read reports a **66 %
   NMPC failure rate**. The true failure count is zero. Any sentence in the
   paper quoting `nmpc_ok` must filter on phase first.
2. **`qp_ok == 0 failures` is only true where it is measured.** The same block
   hardcodes `log.qp_ok.append(True)` (`tick_logging.py:318`) even though the
   inter-step settle loop *does* run a QP every tick. So QP status is
   unrecorded on 66 % of ticks and the honest claim is **"0 QP failures across
   the 709 ticks where QP status is recorded"**, not "0 across the run".

---

## 3. Wall-clock is machine-dependent — quantified

Same physics, same commit, same software stack, **different machine**.
`nmpc_iterations` is one of the 64 byte-compared columns and is *identical*
between the two columns below (histogram
`{6:4, 7:163, 8:33, 9:78, 10:57, 11:49, 12:43, 13:112, 14:99, 15:43, 16:16, 17:10, 18:2}`
in both) — so the solver did **exactly the same work**, and the entire
difference is hardware.

| quantity (C, NMPC-active) | committed artifact | this container | delta |
|---|---:|---:|---:|
| NMPC solve, median [ms] | 16.87 | **20.83** | **+23 %** |
| NMPC solve, max [ms] | 38.56 | 39.50 | +2 % |
| WBC block, median [ms] | 56.92 | **72.64** | **+28 %** |
| WBC block, max [ms] | 107.12 | 143.43 | +34 % |
| per-QP-solve upper bound, median [ms] | 5.69 | 7.26 | +28 % |
| IPOPT iterations, median / total | 11 / 7705 | 11 / 7705 | **0 %** |

**Recommendation for the paper: lead with iteration counts, not milliseconds.**
Iterations are deterministic and reproduce byte-identically under the gate;
milliseconds move ~25 % between two machines running the same code. If a
timing table is wanted, it must carry a hardware row — and the committed run's
hardware is unknown (§5).

---

## 4. Real-time feasibility — what the data does and does not support

The control rates imply budgets of **100 ms** per NMPC tick and **10 ms** per
QP tick.

| budget check (C, NMPC-active ticks) | committed | this container |
|---|---:|---:|
| WBC block > 100 ms | **1 / 709** (0.14 %) | 21 / 709 (3.0 %) |
| WBC block + NMPC solve > 100 ms | **12 / 709** (1.7 %) | 208 / 709 (29 %) |
| per-QP-solve upper bound > 10 ms | **1 / 709** (0.14 %) | 21 / 709 (3.0 %) |
| median (WBC + NMPC) [ms] | 75.26 | 95.80 |
| max (WBC + NMPC) [ms] | 129.03 | 168.29 |

Read carefully:

- The measured block **includes MuJoCo integration and logging**, which a
  deployed controller would not pay. So these are upper bounds on the
  controller-only cost, not estimates of it.
- Even so, on the committed machine the combined per-tick cost exceeds the
  100 ms budget on **1.7 %** of ticks, and on this container on **29 %**.
- **The simulation is not run in real time and nothing in these artifacts
  claims otherwise.** The paper should either omit a real-time claim or state
  it as "median controller cost is within budget on the reference machine,
  with a tail that is not" — the second is defensible from this table, the
  first is not defensible at all.

---

## 5. Channels that are genuinely missing (input to C2.2)

C2.2 was told to add only what is absent. Checked against the source, here is
what is and is not there. This supersedes guessing at C2 time.

| channel | status | where |
|---|---|---|
| NMPC solve time | **present** (CSV col 55) | — |
| WBC block time | **present** (CSV col 54), but **mislabelled** `qp_time_ms` | see §0(a) |
| NMPC iteration count | **present** (CSV col 56) | from `solver_stats['iter_count']`, `tick_logging.py:515-517` |
| **IPOPT return status** | **present in `sim_log.json`, ABSENT from the CSV** | `log.nmpc_status_str`, `tick_logging.py:513-514`. The CSV carries only the `nmpc_ok` boolean, which cannot distinguish `Solve_Succeeded` from `Solved_To_Acceptable_Level` — 5 ticks that are degraded successes are indistinguishable from the other 704. **Promote the string to a CSV column.** |
| **True per-QP-solve time** | **computed and thrown away** | `HierarchicalQP.solve` sets `info.solve_time_ms` at `hierarchical_qp.py:245` and `info.n_iter`; `WholeBodyQP.solve` returns the `QPSolveInfo`; **`sim_loop` never persists it**. The instrumentation already exists — it just needs logging. This is the single highest-value C2.2 addition, because it is the only way to replace the ×10-plus-physics upper bound with a real number. |
| **QP solver status / exitflag** | **computed and thrown away** | same `QPSolveInfo` (`.success`, `.exitflag`, `.failed_priority`, `hierarchical_qp.py:55-67`). Currently collapsed into one `qp_ok` bool, itself hardcoded `True` on 66 % of ticks. |
| **Coarse pre-planner solve time / iterations / status** | **collected but not exported** | `sim_loop._preplanner_stats` (`sim_loop.py:1437-1443`) holds `success`, `solve_ms`, `iter_count`, `cost`, `status` per step, and the values are printed to stdout — but only `preplanner_T_steps` reaches `sim_log.json`. No committed artifact carries them (grepped `results/j2_adjconv/*.json`, `results/j2_figdata/*.json`: no `solve_ms` / `iter_count`). **A solver table that omits the once-per-step IPOPT NLP is incomplete** — six solves that gate every step. |
| IPOPT restoration-phase count | **not available** | not a CasADi `stats()` key; `solver_stats` is kept on `NMPCSolveInfo` (`nmpc_solver.py:487`) but is not persisted. Reporting "0 restorations" is currently unsupported; "0 failure statuses" is supported. |

---

## 6. One result worth its own line in the ablation

Because iteration counts are deterministic, this comparison is
**gate-reproducible** in a way the timings are not:

| | C (managed) | U (unmanaged) | delta |
|---|---:|---:|---:|
| NMPC solves | 709 | 622 | |
| total IPOPT iterations | **7705** | 5326 | |
| mean iterations / solve | **10.87** | 8.56 | **+26.9 %** |
| mean iterations / solve, SS only | **12.24** | 9.61 | **+27.4 %** |
| median iterations | 11 | 7 | |

**The momentum envelope costs ~27 % more IPOPT iterations per NMPC solve.**
That is a clean, deterministic, hardware-independent statement of the
constraint's computational price — and it is a much better sentence than the
9.5 s traversal claim C1.6 found to be misattributed. It belongs in the C4
ablation table alongside the physical metrics.

Note the tails run the other way: U's iteration *maximum* is higher (23 vs 18)
and its p99 is 21 vs 17. The unmanaged NMPC is usually cheaper but occasionally
works harder — consistent with an unconstrained problem whose solution
wanders further between warm starts.

---

## STOP

C3.4 is complete: the table is producible today from committed data, with the
two mislabelling traps and the two "0 failures" caveats made explicit, and
§5 hands C2.2 a verified list of what is actually missing rather than a guess.

Two items the paper cannot get from this repository and must obtain elsewhere:
the **hardware** the committed artifacts were timed on, and — if a solver table
is to be complete — the **pre-planner** solve statistics, which are collected
in memory and printed but never persisted.

Awaiting explicit GO before C2 (reduced scope: 6-D dock twist, AOCS torque
decomposition, and the three solver channels identified in §5).
