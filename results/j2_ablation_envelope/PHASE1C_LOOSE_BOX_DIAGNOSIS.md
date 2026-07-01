# Phase-1c — loose-but-active storage box (h_max_tight=10, enforce=True): result

> **⚠️ CORRECTED BY PHASE-1D (`PHASE1D_ROOT_CAUSE.md`).** The conclusion below — "the
> pre-planner is numerically fragile to the box VALUE" — is **confounded and wrong**. The
> sim's `T_step_guess` heuristic (sim_loop:1888-1890) inversely couples the planned step
> duration to `h_max`, so the h_max=10 run also ran with **half the step time** (2.78→1.39 s).
> With the step time held fixed, box=10 converges to the *identical* optimum as box=5
> (`Optimal Solution Found`). The failure is the shortened step time, **not** the box value or
> the landscape. The methodology note below (blaming a "missing warm-start") is likewise wrong:
> the discarded sweep failed because it inherited the h_max=10 step time (T10), not for lack of
> a warm-start. Read `PHASE1D_ROOT_CAUSE.md` for the corrected root cause; the raw run facts
> below (h_max=10 fails, h_max=5 succeeds, verbatim EXITs, s_hw=0) remain accurate.


**Hypothesis (Idriss):** Task-2's storage blocker was the box's ABSENCE (removing
`enforce_hw_conservation` + setting `h_max=1e6` left `h_w` under-regularized), NOT its
bound value. Keeping the box ACTIVE but LOOSE (2× the real 5 bound) should converge
fine, because the box then acts as a regularizer that never binds (C peaks at 4.886 <
5 < 10). **Falsifiable.**

**Verdict: FALSIFIED.** The loose-active box (h_max_tight=10, enforce=True, everything
else canonical) **fails at the step-0 pre-planner** in the real sim, while canonical
h_max=5 succeeds there. The pre-planner is numerically sensitive to the box **value**,
not just its presence. This does not overturn Task 2 — it **extends** it: removing the
box *and* merely loosening it both drive IPOPT to the same "local infeasibility" class.
The storage box at 5 is a **numerically-tuned operating point**, not a physical
necessity (it never binds — h_w peaks 4.886 < 5).

---

## The decisive test — real-sim solve path (real warm-start), same env, single variable

The only difference between the two runs is `cfg.h_max_tight` (5 vs 10). The pre-planner's
initial guess is **h_max-independent** (`coarse_preplanner.py:479-489`: linear r_com
interpolation, constant v, zero L/u — no `h_max` term), so this is a clean single-variable
comparison of IPOPT's local convergence.

| run (real sim, this env) | h_max_tight | step-0 pre-planner | evidence |
|---|---|---|---|
| **nominal (= canonical C)** | **5** | **SUCCESS** — `success in 77.0 ms (12 iters, cost=11.205, T_step=2.78s, peak \|v\|=0.106 m/s, peak \|L\|=0.134 Nms)` | `canon5_control.log`; `diag_nominal_shw.py` |
| **loose-active** | **10** | **FAIL** — `RuntimeError: Error in Opti::solve` → step skipped | `loose10_run.log`; `diag_loose_box.py` |

Nominal (h_max=5) also reproduces committed C **bit-for-bit**: docks
**[4.94, 4.41, 4.88, 4.42, 4.76, 4.92] mm** (identical to C), 6/6 pre-planner solves —
so the h_max=10 failure is a genuine box-value effect, not environment drift.

### Verbatim IPOPT EXIT for the real-path h_max=10 step-0 failure (`loose10_realverbose.log`)
```
Number of Iterations....: 133
Objective...............:   1.9491405224373173e+02
Dual infeasibility......:   9.9999849289655263e+00
Constraint violation....:   9.1339744365410480e-02
Variable bound violation:   0.0000000000000000e+00      <-- the ±10 box is NEVER touched
Overall NLP error.......:   9.9999849289655263e+00
EXIT: Converged to a point of local infeasibility. Problem may be infeasible.
```
Same class as Task 2: objective bounded (194.9, vs the true optimum 11.2 that h_max=5
reaches), variable bound violation **0.0** (the box does not bind), small constraint
residual (0.091) → **local-solver / restoration failure, not unbounded, not the box**.

### Mechanism
With the same h_max-independent warm-start, the box at **5** constrains IPOPT's iterates
tightly enough to keep them in the basin of the true low-momentum optimum
(‖L‖≈0.13 Nms, cost 11.2). Loosening the box to **10** removes that regularizing
pressure near the warm-start; IPOPT wanders and converges to a point of local
infeasibility (cost 194.9) instead. The box's operative role here is **conditioning /
regularization of the pre-planner NLP**, not physical momentum enforcement.

---

## Methodology honesty — a warm-start-free sweep was run first and DISCARDED

Before the real-path control above, a sweep re-solved the captured step-0 problem at
h_max ∈ {5,6,7,8,10,20,100,1e6} with a **fresh** pre-planner (`diag_loose_box_verbose.py`).
Its **positive control (h_max=5) FAILED** — landing at cost ≈193, "local infeasibility",
identical last-iterate (‖h_w‖=2.579, ‖L‖=3.468) at *every* h_max — even though the real
sim converges at h_max=5. The tell: that re-solve omits the sim's warm-start / initial
guess, so it fails regardless of the box value. **The sweep is unfaithful and is not used**
to characterize the box-value sensitivity; only the real-sim runs (with warm-start) are.
(Raw sweep data: `loose10_sweep.json`.) Note this also means Task 2's re-solve was
warm-start-free; Task 2's conclusion nonetheless stands because the REAL sim independently
fails at storage-off — the re-solve only corroborated the local-infeasibility EXIT.

---

## Side-ask — max|s_hw| over the full NOMINAL traversal (QP soft-box slack is inert)

`nominal_hw_slack.json` (from `diag_nominal_shw.py`, canonical h_max=5, full 6-step run):

| quantity | value |
|---|---|
| QP instances | 1 (single `qp_ss`) |
| slack-log entries (QP solves) | **5832** |
| max\|s_hw\| upper / lower / norm | **0.0000 / 0.0000 / 0.0000 Nms** |
| QP soft box `hw_max` | 5 |

The momentum soft-slack `s_hw` (penalty `w_hw_slack=1e4`) is **exactly 0** across all 5832
nominal QP solves — h_w never exceeds ±5, so the soft box never activates. **The Task-1
numbers (planned Ḣ_s, realized τ_w, θ_s) are not distorted by the soft slack.**

---

## Artifacts
| file | content |
|---|---|
| `scripts/diag_loose_box.py` | real-sim loose-active run (h_max=10, enforce=True); step-0 fails |
| `scripts/diag_loose_box_realverbose.py` | real-path h_max=10 step-0 with `ipopt_print_level=5` → verbatim EXIT |
| `scripts/diag_nominal_shw.py` | genuine nominal C (h_max=5) full run + `qp.hw_slack_log` capture |
| `scripts/diag_loose_box_verbose.py` | **superseded/unfaithful** warm-start-free sweep (positive control fails) |
| `loose10_run.log` | real loose-box run: step-0 pre-planner failure |
| `loose10_realverbose.log` | verbatim IPOPT EXIT (real path, h_max=10) |
| `loose10_sweep.json` | discarded warm-start-free sweep data |
| `nominal_hw_slack.json` | max\|s_hw\| = 0 over 5832 nominal QP solves |

Raw sim dumps (`figU_loose10/`, `figC_nominal/`, `_loose_verbose/`, `_loose_realverbose/`,
`_canon5_control/`) are gitignored (regenerable; tidy-artifact convention). No `crawlbot/`
or MJCF change. No paper text (Phase 2). Task 3 (soft `w·‖h_w‖²` penalty counterfactual)
remains gated on approval.
