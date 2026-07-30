# F2 — reactivating the RWA conservation box

**Branch** `claude/com-gain-semantics-audit-j0u6yr`
**Outcome: the box is NOT enabled by default.** It is feasible and harmless, but
a bite test shows it does not bound the quantity it appears to bound, and
shipping a constraint that looks like a guarantee but is not one is worse than
shipping no constraint. The machinery, the staging switch and the evidence are
committed; the default stays `False` pending §5.

---

## 1. What was done

`enforce_hw_conservation` gated **both** the path box and the terminal
constraint, so they could not be staged. Split:

| field | meaning |
|---|---|
| `enforce_hw_conservation` | the **path** box `h_w(k) ∈ [−h', h']`, k = 0..N−1 |
| `enforce_hw_terminal` | the **terminal** set `\|h_w(N)\| ≤ κ·h'`. `None` = follow the path box (historical coupling), `True`/`False` overrides |

Verified against all four combinations:

| box | terminal | `ng_path` | `ng_term` | |
|---|---|---|---|---|
| False | None | 11 | 0 | current default, unchanged |
| True | None | 17 | 6 | legacy coupling, preserved exactly |
| True | False | 17 | 0 | step A |
| True | True | 17 | 6 | step B |

`(True, None)` reproducing `17/6` is incidentally the configuration the stale
`centroidal_nmpc.md` §3 was describing — it was a real configuration, just never
the canonical one.

## 2. Step A — path box on, `h_max_tight = 5.0`

Full 6-step replay: docks **6/6** (4.19/4.70/4.95/4.69/2.62/4.55, worst margin
0.05 mm), θ_s 0.455°, h_w 3.815/4.087, e_com 0.092, **639/639 `Solve_Succeeded`**.
Every headline metric identical to F2-off.

Cost: **none measurable.** See §3.1 — an earlier draft of this report claimed
+25 %, comparing against a run from a different session. Corrected.

## 3. Step B — terminal set on top

Same metrics again, 639/639 success.

### 3.1 Solve cost: correcting a cross-session comparison

The `F2off_ctl_N20` control run exists precisely so the cost is measured in
the SAME session as the treatments. All four:

| run | median | p95 | max |
|---|---|---|---|
| **F2 off (control)** | **38.36** | 52.81 | 83.53 |
| box | 37.57 | 50.55 | 67.08 |
| box + terminal | 37.33 | 52.93 | 95.19 |
| bite (h_max 3.5) | 37.98 | 51.83 | 78.73 |

**Enabling the box costs nothing measurable** — box-on is marginally *faster*
at the median than box-off, i.e. the spread is machine noise, not signal.
0/639 over the 100 ms period in every case.

⚠ An earlier draft reported "+25 %" by comparing against `F3_N20_dt10`
(median 30.05 ms), which was recorded in an earlier session on a quieter
machine. That is the same error as the retracted 117.9 ms real-time claim
(`NMPC_HORIZON_N15` §3.2): **solve times are only comparable within one
session.** The maxima here (67–95 ms) span 40 % for configurations whose
medians differ by 1 ms, which is the size of the noise.

## 4. Why "identical metrics" is the whole story

Solver-level, one representative problem under all three configurations:

```
off       ng_path=11 ng_term=0 rows=409  cost=4.285326e+01 it=15
box       ng_path=17 ng_term=0 rows=529  cost=4.285326e+01 it=15
box+term  ng_path=17 ng_term=6 rows=535  cost=4.285326e+01 it=15
   off vs box      max|Δr_plan| = 5.8e-11
   box vs box+term max|Δr_plan| = 4.8e-12
```

**Same optimum, same iteration count.** The constraints are added and are
**inactive**. Closed-loop, the 1e-11 perturbation amplifies chaotically: the
box-only run differs from F2-off in 46 698 of 125 888 exported fields, yet the
largest physical difference is 9 µm of CoM, 1.3e-5° of θ_s, 1.6e-4 Nms of h_w.
Total IPOPT iterations: 6794 / 6798 / 6794.

⚠ `box+term` came out **byte-identical** to `off` while `box` did not. That is a
rounding coincidence in a chaotic loop, not evidence about the terminal set —
recorded here so nobody later reads it as one.

## 5. The bite test — and why the default stays OFF

A constraint that never activates proves nothing. Tightened to
`h_max_tight = 3.5`, **below** the realized h_w peak of 3.815:

| | expected | observed |
|---|---|---|
| NMPC solves | some infeasible | **639/639 `Solve_Succeeded`** |
| realized h_w peak | ≤ 3.5 | **3.8146** (+9.0 %, +0.315 Nms) |
| SS ticks above the box | 0 | **9 / 438**, t = 44.92 … 45.62 s |
| docks / θ_s / h_w | changed | **identical to h_max = 5.0** |

**The tightened box did not bite.** But the machinery is provably correct — the
same NLP, given `hw_current = 3.8146` with `h_max = 3.5` in isolation, returns
`Infeasible_Problem_Detected`:

```
h_max=5.0  |h_w(0)|=3.8146 -> success=True   Solve_Succeeded
h_max=3.5  |h_w(0)|=3.8146 -> success=False  Infeasible_Problem_Detected
h_max=3.0  |h_w(0)|=3.8146 -> success=False  Infeasible_Problem_Detected
```

So the constraint works, and in the closed loop the NMPC never saw a violating
state — while the plant reached one. `h_w(0) ≡ hw_current` identically (the
`c_simple` terms cancel at k=0), and the fulldiag's SS rows *are* the NMPC solve
instants (438 rows at exactly 0.100 s; 438 SS + 201 DS_terminal = 639 solves).
So at t = 45.32 s the NMPC solved successfully at a tick whose logged h_w was
3.8146 against its own 3.5 box.

**That gap is unexplained and is the reason F2 stays off.** The candidate is
that `hw_for_nmpc = rwa_I_w · qvel[6:9]` (read at `_step` entry,
`sim_loop.py:2256`) is not the quantity exported as `hw_*_Nms`
(`rwa_I_w · rw_vel_f`, a *filtered* wheel velocity captured later in the tick —
`tick_logging.py:429`, and `diag_full_diag_export.py:102` exports
`hw_physical`). Measured h_w drift is ~0.003 Nms per 10 ms tick in the quiet
regions but up to 0.113 Nms, i.e. **up to ~1.13 Nms across one control period** —
the same order as the 0.315 Nms discrepancy. That would mean the box is anchored
to a stale or differently-filtered h_w, so it bounds neither the realized wheel
momentum nor the value the diagnostics report.

Until that is resolved, enabling the box would advertise an envelope guarantee
the system does not have. **The correct action is to leave it off and close the
gap first.**

## 6. What this says about the audit findings

- **F2** — machinery correct, feasible at 5.0, **no measurable solve cost** (§3.1),
  but **does not bound realized h_w**. Not enabled. New sub-question above.
- **F7** (`L_com` bounded by a state box, not the wheel envelope) — F2 was the
  candidate remedy. It is not one until §5 is resolved, so F7 stands.
- The wheel envelope therefore still reaches the NLP **only** through the
  `|Ḣ_s| ≤ τ_w,max` rate cap (F4), which is a bound on the derivative. Nothing
  bounds accumulated `h_w` inside the horizon.

## 7. Next step

Instrument one replay to log `hw_for_nmpc` alongside the exported `hw_physical`
at every solve and difference them. If they diverge by the ~0.3 Nms seen here,
the fix is to feed the NMPC the same filtered quantity the AOCS and the
diagnostics use — after which the bite test should be re-run and must bite.

## 8. Artifacts

| path | what |
|---|---|
| `nmpc_sweep/F2off_ctl_N20/` | F2 off, current tree — control, confirms the baseline is still valid |
| `nmpc_sweep/F2box_N20/` | step A, box only, h_max 5.0 |
| `nmpc_sweep/F2boxterm_N20/` | step B, box + terminal |
| `nmpc_sweep/F2bite_h35_N20/` | bite test, h_max 3.5 |

No committed canonical artifact modified; `SimConfig` defaults unchanged.
