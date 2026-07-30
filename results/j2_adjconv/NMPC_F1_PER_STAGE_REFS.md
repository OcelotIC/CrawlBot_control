# F1 — per-knot NMPC references, then N = 20 at 10 Hz

**Branch** `claude/com-gain-semantics-audit-j0u6yr`
**Parent** `64ab4d2` (NMPC audit)
**Final config** `nmpc_N=20`, `nmpc_dt=0.1`, `dt_nmpc=0.1`, `nmpc_per_stage_refs=True`
— a 2.0 s horizon at 10 Hz with the F3 timing invariant satisfied.

**F2 was not touched.** `enforce_hw_conservation` is still `False`, as agreed.

---

## 1. What F1 was

The NLP carried **one** parameter block shared by every stage
(`nmpc_solver.build`: the same `p_param` at every `k`), so `r_ref`, `v_ref` and
`L_ref` were necessarily **constant across the horizon**. The NMPC was a
regulator to a fixed setpoint, not a trajectory tracker.

`sim_loop` compensated by sampling that setpoint in the *future* — horizon end
for CoM (`t + N·dt`), midpoint for `L_com` (`t + N·dt/2`) — because a constant
reference sampled at the current time lags systematically. **That is what made
`nmpc_N` two knobs**: raising the horizon also pushed the target further ahead,
so no clean horizon ablation existed (`NMPC_HORIZON_N15.md` §1).

The coarse pre-planner produces a momentum-feasible CoM *trajectory* over the
full `T_step`; the NMPC consumed exactly **one point of it per solve**.

## 2. What changed

`NMPCSolver.set_parameters(np_, per_stage=True)` gives the NLP **N+1 parameter
blocks** — `p_k` for stage `k`, `p_N` for the terminal cost/constraint. The
per-stage symbol handed to the user callbacks is unchanged in shape, so
`set_stage_cost`, `set_path_constraints` etc. are written identically and the
decision-variable and constraint counts are untouched. Only the
parameterization grows.

`CentroidalNMPC._assemble_params` now accepts a `(3,)` setpoint **or** a
`(K, 3)` per-knot reference and emits one block per knot.
`sim_loop._com_ref_at(t_query, settle_mode)` — extracted verbatim from the old
inline block, including the pre-planner override and its compressed-time
mapping — is called once per knot at `t + k·nmpc_dt`.

Selected by `SimConfig.nmpc_per_stage_refs`.

### 2.1 Proven inert before it was proven useful

`scripts/audit_nmpc_f1_equivalence.py` — **PASS**:

| check | result |
|---|---|
| structure: blocks 1 → N+1, params 27 → 171, **decision vars and constraints unchanged** (177 / 169 both) | PASS |
| equivalence: per-stage NLP fed ONE broadcast reference vs legacy NLP | **Δcost = 0.000e+00, Δr_com_plan = 0.000e+00, Δλ₀ = 0.000e+00** |
| effect: a *ramped* reference must change the plan | Δ = 7.0e-04 m ✓ |
| guard: per-knot reference into a legacy NLP | raises |
| guard: N rows instead of N+1 | raises |

Exact zeros, not "within tolerance". And at sim level, `F1off_N15` reproduces
the committed N=15 run on every metric (docks 4.40/4.53/4.93/4.48/2.12/4.40,
θ_s 0.554, h_w 3.992/4.172, e_com 0.190). **So every difference below is
attributable to the reference varying, not to the refactor.**

---

## 3. Results — the Rule-12 ladder

Each column differs from the one to its left in **exactly one field**.

| metric | frozen N=8 | F1 **off** N=15 *(= committed)* | F1 **on** N=15 | **F1 on, N=20** *(final)* |
|---|---|---|---|---|
| docks [mm] | 4.02/4.89/4.99/4.97/4.95/4.62 | 4.40/4.53/4.93/4.48/2.12/4.40 | 3.81/4.61/4.92/4.59/2.54/4.74 | 4.19/4.70/4.95/4.69/2.62/4.55 |
| under 5 mm | 6/6 | 6/6 | 6/6 | **6/6** |
| worst margin [mm] | 0.01 | 0.07 | 0.08 | 0.05 |
| **θ_s peak [deg]** | 0.540 | 0.554 | 0.511 | **0.455** |
| h_w peak axis [Nms] | 4.100 | 3.992 | 3.836 | **3.815** |
| h_w peak norm [Nms] | 4.240 | 4.172 | 4.088 | **4.087** |
| e_com peak [m] ⚠ | 0.154 | 0.190 | 0.093 | 0.092 |
| qp_fail | 0 | 0 | 0 | **0** |
| NMPC solves | 709 | 634 | 638 | 639 |
| solve median / max [ms] | 22.0 / 61.9 | 25.4 / 43.6 | 24.1 / 41.5 | 30.4 / **57.1** |
| solves over the 100 ms period | 0 | **0** | **0** | **0** |
| `Solved_To_Acceptable_Level` | — | 2 | 1 | **0** |

Figure: `results/j2_adjconv/nmpc_sweep/nmpc_f1.png`.

### 3.1 Reading it

**θ_s improves monotonically down the ladder** — 0.554 → 0.511 → 0.455, i.e.
**0.540 → 0.455 (−16 %)** against the frozen canonical. `h_w` peak falls the
same way. Both are definition-stable metrics, so this is the solid part of the
result.

**All six docks hold**, worst margin 0.05 mm vs the frozen 0.01 mm.

**At N=20 every solve converged fully** — 639/639 `Solve_Succeeded`, zero
acceptable-tolerance exits, the cleanest solver behaviour of any configuration
measured in this campaign. Max solve 57.1 ms against a 100 ms period.

⚠ **The `e_com` column is not comparable across the flag.** The exported
`r_com_ref` is knot 0, which is the *horizon-end setpoint* under F1-off and the
*current-time reference* under F1-on. Each run's `e_com` is therefore "error
against what that controller was actually told to reach" — a fair statement per
run, but the 0.190 → 0.093 drop is part architecture and part definition, and
this campaign cannot separate them. **Do not quote it as a pure tracking
improvement.** The F1 verdict rests on θ_s, h_w and the docks.

### 3.2 A correction to the N=15 report

`NMPC_HORIZON_N15.md` §3 recorded a **117.9 ms** max solve at N=15 and flagged
"1 solve in 634 over the 100 ms period — a hard real-time violation". Re-running
that same configuration here (`F1off_N15`) gives **max 43.6 ms, 0 over budget**,
with byte-identical trajectory metrics — so the control path is the same and the
117.9 ms was **machine contention, not an algorithmic property**.

**That retracts the real-time concern.** No configuration in this campaign,
including N=20, exceeds its control period on a quiet machine. Solve-time
outliers on this container are not evidence about the algorithm; only medians
and the general trend are.

---

## 4. Verification

| gate | result |
|---|---|
| `gate/run_suite.py` (full) | **PASS** — 211 tests, 210 passed, 0 failed, 0 errors, 1 xfail |
| `scripts/audit_nmpc_f1_equivalence.py` | **PASS** — exact-zero equivalence, effect confirmed, both guards raise |
| `gate/run_gate.py` | FAIL on artifact identity — expected, the controller changed |
| `gate/sync_docs.py --check` / `verify_docs` / `link_audit` / `verify_params` | PASS |

Five new suite tests (`TestPerStageReferences`) pin the block count, the
broadcast equivalence, that a varying reference actually changes the plan, and
that both malformed-reference cases raise instead of silently truncating.

The strict `xfail` did not flip.

---

## 5. What this unblocks

`nmpc_N` is now **one knob**. The three reference-sampling expressions no longer
depend on it, so a horizon ablation is finally clean — the N=15 → N=20 column
above is the first honest one in this campaign.

F3 (the `nmpc_dt == dt_nmpc` invariant) is **still unfixed** and still
unasserted. It does not bite at the committed config because `nmpc_dt` and
`dt_nmpc` are both 0.1, but it remains a trap for anyone who changes one.

F2 (`enforce_hw_conservation`) is untouched and awaiting your call.
