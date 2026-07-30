# F3 — prediction step vs control period, decoupled

**Branch** `claude/com-gain-semantics-audit-j0u6yr`
**Parent** `50a432d` (F1 + N=20)
**Committed config unchanged**: `nmpc_N=20`, `nmpc_dt=0.1`, `dt_nmpc=0.1`,
`nmpc_per_stage_refs=True`. F3 is a correctness fix, not a retune.

**F2 still untouched.**

---

## 1. Why there are two `dt` fields

They are **different quantities**, and the split is legitimate MPC practice —
not a duplicate:

| field | meaning | where it is used |
|---|---|---|
| `dt_nmpc` = 0.1 s | **control period** — how often the NMPC re-solves and the plan is applied | `n_qp_per_nmpc` (`sim_loop.py:102`), time advance `t += dt_nmpc` (×4), the L̇ finite-difference window (`tick_logging.py:382`), the "10 Hz" banner |
| `nmpc_dt` = 0.1 s | **prediction step** — the RK4 step inside the NLP, i.e. the horizon's knot spacing | the NLP's `dt` (`sim_loop.py:418`), horizon lookahead `N·nmpc_dt`, per-knot reference times |

A controller that re-solves at 10 Hz while predicting on a 20 ms grid is normal
and often desirable. So the two fields *should* exist.

**But the split was never exercised.** Git archaeology: both fields were
introduced in the same commit, and across the whole reachable history of
`config.py` they have **never held different values** — always 0.1 / 0.1. Three
code paths quietly assumed that would remain true.

Two aggravating factors worth naming, because they are the reason this survived:

1. **The names are near-anagrams.** `nmpc_dt` and `dt_nmpc` differ only in the
   order of two tokens. Nothing in a diff, a review, or a grep makes them look
   distinct, and code using the wrong one reads perfectly.
2. **Equal values hide the bug.** With `nmpc_dt == dt_nmpc` every incorrect
   expression is numerically correct, so no test, gate or run could expose it.
   The defect only appears the first time someone changes one — which is
   exactly what happened when "set dt = 0.05" was requested.

---

## 2. What was wrong

Three paths advanced the plan by **one prediction knot per control period**:

| site | expression | correct only if |
|---|---|---|
| `sim_loop` plan interpolation | `alpha = qs / n_qp_per_nmpc` between knots 0 and 1 | `nmpc_dt == dt_nmpc` |
| `CentroidalNMPC.get_shifted_fallback` | shift by 1 knot | same |
| `NMPCSolver.shift_warm_start` | shift by 1 knot | same |

With `nmpc_dt = 0.05` and `dt_nmpc = 0.1`, the interpolation walked knot 0 → 1
over 0.1 s of wall time, but that segment is only 0.05 s of plan time — so the
QP tracked a CoM reference running at **half speed**.

## 3. The fix

**Interpolation** — index the plan by elapsed time on its own knot grid:

```
u = qs / qp_per_knot        qp_per_knot = round(nmpc_dt / dt_qp)   [integer]
k = min(floor(u), N-1)      a = clamp(u - k, 0, 1)
r_ref = (1-a)·plan[:,k] + a·plan[:,k+1]
```

The whole plan is cached now, not just its first two knots. `qp_per_knot` is
formed as an **integer** deliberately: the algebraically equivalent
`(qs·dt_qp)/nmpc_dt` is off by 1 ULP (`0.01/0.1 != 0.1` in IEEE double), which
is enough to break bit-identity over a 2000-tick run. `SimSetup.__init__` now
rejects an `nmpc_dt` that is not an integer multiple of `dt_qp`, since the
indexing would otherwise drift.

**Shift arithmetic** — `NMPCSolver.shift_warm_start(n_steps=1)` and
`CentroidalNMPC.get_shifted_fallback` advance by
`n = round(control_period / dt)` knots, from the new
`CentroidalNMPCConfig.control_period` (`None` ⇒ `dt` ⇒ n = 1, the legacy
behaviour). `sim_loop` passes `control_period=cfg.dt_nmpc`.

---

## 4. Proof

### 4.1 Unit level — `scripts/audit_nmpc_f3_timing.py` **PASS**

| check | result |
|---|---|
| **A. reduction**: new form vs old when `nmpc_dt == dt_nmpc` | max \|new − old\| = **0.000e+00** over all 10 sub-steps — exact, not "within tolerance" |
| **B. effect**: `nmpc_dt=0.05`, `dt_nmpc=0.1`, plan at 1.0 m/s | new tracks truth to 1.4e-17 m; **old lags to 0.0400 m** at qs=8 — exactly the 2× dilation |
| **C. shift count** | 0.1/0.1 → 1, 0.1/0.05 → 2, 0.1/0.025 → 4; `control_period=None` → 1 |
| **C2. fallback** | `shifted[:,0] == original[:,n]` for n = 1 and 2 |

### 4.2 Sim level — inert at the committed config

`F3_N20_dt10` (post-fix) vs `F1on_N20` (pre-fix, identical configuration):

```
physics/state columns compared : 64 x 1967 rows = 125 888 fields
differing physics fields       : 0
VERDICT: PHYSICALLY BYTE-IDENTICAL
```

Only `qp_time_ms` and `nmpc_time_ms` differ — the two wall-clock columns that
`gate/run_gate.py:66` itself excludes from artifact identity as
nondeterministic. A first pass with a plain `cmp` reported "1278 differing
lines" and looked like a regression; it was comparing the timing columns the
gate deliberately ignores. **Compare with the gate's exclusion list, not with
`cmp`.**

### 4.3 Sim level — the differing-rate configuration now runs correctly

`F3_N20_dt05` — `N=20`, `nmpc_dt=0.05`, `dt_nmpc=0.1` (1.0 s horizon at 10 Hz),
a configuration the pre-fix code could not execute correctly:

| metric | pre-fix `N20_dt05_p10` | **post-fix `F3_N20_dt05`** | committed `F1on_N20` |
|---|---|---|---|
| docks under 5 mm | 6/6 | **6/6** | 6/6 |
| worst margin [mm] | 0.05 | 0.06 | 0.05 |
| **θ_s peak [deg]** | **0.494** | **0.774** | **0.455** |
| h_w peak norm [Nms] | 4.233 | 4.043 | 4.087 |
| qp_fail | 0 | 0 | 0 |

**The dilation was flattering that configuration.** With it removed, the same
`nmpc_dt=0.05` at 10 Hz goes from θ_s 0.494° to **0.774°** — the worst of any
run in this campaign. That corroborates the earlier call not to adopt it: it
looked good *because of* the bug.

⚠ Two caveats on that column. The pre-fix run also had `per_stage_refs=False`,
so F1 and F3 both differ between those two columns — the direction is
unambiguous but it is not a single-variable isolation. And `F3_N20_dt05` has a
**1.0 s** horizon against `F1on_N20`'s 2.0 s (same N, half the step), so that
comparison is horizon length, not step resolution. Both point the same way:
keep `nmpc_dt = 0.1`, `N = 20`.

---

## 5. Verification

| gate | result |
|---|---|
| `gate/run_suite.py` (full) | **PASS** — 216 tests, 215 passed, 0 failed, 0 errors, 1 xfail |
| `scripts/audit_nmpc_f3_timing.py` | **PASS** |
| `scripts/audit_nmpc_f1_equivalence.py` | **PASS** (still) |
| `gate/sync_docs.py --check` / `verify_docs` / `link_audit` / `verify_params` | PASS |

Five new suite tests (`TestControlPeriodVsPredictionStep`) pin the shift count
against the ratio, the legacy default, the floor at 1, that the fallback
advances by that count and preserves the horizon length, and the no-previous-
solve path.

---

## 6. State of the audit findings

| finding | status |
|---|---|
| **F1** — reference was a constant setpoint | **fixed** (`50a432d`), awaiting your validation |
| **F2** — RWA conservation box off | **untouched**, awaiting your call |
| **F3** — `nmpc_dt == dt_nmpc` assumed in 3 places | **fixed here**; both rates now free, with the non-integer-multiple case rejected rather than silently drifting |
| F4 — wheel-torque cap is the only binding constraint | informational, unchanged |
| F5 — `get_shifted_fallback` never executed by the canonical | **still uncovered by the canonical run**, but now covered by unit tests (§5) rather than by nothing |
| F6 — `Solved_To_Acceptable_Level` counts as success | informational, unchanged |
| F7 — `L_com` bounded by a state box | informational, unchanged |
