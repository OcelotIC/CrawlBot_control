# C4 — clean three-way envelope ablation

**Review-Closure Bloc 2, Phase C4.** Three runs, one variable. No tuning.

---

## Header (mandatory)

| item | value |
|---|---|
| commit worked on | branch `claude/review-closure-bloc-2-uwu1x7`, parent **`5f55615`** (C2.1+C2.3) |
| date | 2026-07-27 |
| python / mujoco / pinocchio | 3.11.15 / 3.10.0 / 3.9.0 |
| casadi + ipopt / numpy / scipy | 3.7.2 + IPOPT-MUMPS / 2.3.5 / 1.17.1 |
| env vs `gate/environment.lock` | exact match |
| host | Intel Xeon @ 2.10 GHz, 4 logical CPUs, 15 GiB, Linux 6.18.5 |

**Artifacts**: `results/review_closure/c4_ablation/{none,rate,full}/` — each
`c4_<arm>_fulldiag.csv` (92 cols, `--solver-diag`) + `_meta.json` + the
per-step text summaries. Metrics: `c4_metrics.json`. Scripts:
`c4_run_ablation.py`, `c4_analyse.py`, `c4_config_diff.py`.

The raw `sim_log.json` / `step_log.json` (~11 MB per arm, 33 MB total) are
**not committed**: everything this report cites is in the CSV and the meta —
`dock_events`, `dock_gate_trace` and `preplanner_stats` are mirrored there by
the C2.1 exporter change — and `c4_analyse.py` reads the meta when the raw log
is absent, so **every number above reproduces from the committed artifacts
alone** (verified after deletion). To regenerate a raw log:

```
MUJOCO_GL=disabled PYTHONPATH=. python3 \
    results/review_closure/c4_ablation/c4_run_ablation.py {none|rate|full}
```

---

## 1. `u25` is not this ablation's `none` arm — measured, not assumed

`c4_config_diff.py` rebuilds both published configs exactly as `dca.main` does
and diffs all 131 `SimConfig` fields.

**2 fields differ — and they move 3 independent physical constraints:**

| field | U | C | consumed by |
|---|---:|---:|---|
| `tau_w_max` | 1e6 | 2.5 | NMPC path constraint `\|Ḣ_s,i\| ≤ τ_w_max` **and** the whole-body QP envelope box |
| `aocs_tau_w_max` | 1e6 | 2.5 | the AOCS output clip |

Both are set from the **single** `tau_w_max` kwarg of `dca.main`
(`_run('U', 1e6, …)` vs `_run('C', 2.5, …)`; every other kwarg is
literal-identical). So the published comparison lifts the planner's rate
constraint, the QP's envelope box **and the actuator's clip** together.

And `enforce_hw_conservation` stays **True** in both — the storage box is on in
the published "unmanaged" run, so it cannot isolate storage either.

**Verdict: `u25` is a fourth configuration, not a member of this design.** Not
reused. All three arms below were produced fresh under one design.

### What this brief's arms vary

| arm | NMPC rate bound | NMPC storage bound |
|---|---|---|
| `none` | **off** (`nmpc_tau_w_max=inf`) | **off** (`enforce_hw_conservation=False`) |
| `rate` | on (2.5) | off |
| `full` | on (2.5) | on — **= canonical** |

Held fixed in all three: **the AOCS clip at ±2.5**, the QP envelope box at 2.5,
every weight, gain, gait timing and threshold. Because `cfg.tau_w_max` feeds all
three consumers, isolating the planner required a new NMPC-only override,
`SimConfig.nmpc_tau_w_max` (default `None` ⇒ canonical path unchanged).

**Control check — the knob is inert.** `gate/run_gate.py` with the field added
and defaulted: **PASS**, 2077 rows × **132 928 fields byte-identical**. And the
`full` arm's own export, run through `c2_neutrality_check.py` against the
committed `c25_fulldiag.csv`: **PASS**, 132 928 fields. The harness did not
drift, and `full` reproduces the canonical.

---

## 2. Results

### Completion

| | none | rate | full |
|---|---:|---:|---:|
| docked steps / 6 | **6** | **6** | **6** |
| aborts | 0 | 0 | 0 |

All three complete. `rate` and `full` dock at identical times and distances
(4.020 / 4.890 / 4.990 / 4.970 / 4.950 / 4.620 mm), matching the canonical.

### Envelope — peak per-axis ‖L̇_s,i‖∞ [N·m]

| step | none planned | rate planned | full planned | none real. | rate real. | full real. |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7.8221 | 2.5000 | 2.5000 | 2.0569 | 1.9586 | 1.9586 |
| 1 | 3.8035 | 2.5000 | 2.5000 | 2.0925 | 2.0378 | 2.0378 |
| 2 | 9.4305 | 2.5000 | 2.5000 | 2.2381 | 2.1878 | 2.1878 |
| 3 | 2.5365 | 2.4999 | 2.4999 | 1.5467 | 1.5019 | 1.5019 |
| 4 | **12.7410** | 2.5000 | 2.5000 | 2.5000 | 2.5000 | 2.5000 |
| 5 | 3.3562 | 2.5000 | 2.5000 | 1.5147 | 1.4687 | 1.4687 |

**`full` reproduces 2.500 on all six steps** — the brief's harness check passes.
Unconstrained, the planner asks for up to **12.741 N·m, 5.1× the envelope**.
Realized peaks are similar across all arms because the QP box (held at 2.5 in
every arm) catches what the planner does not.

### Attitude, storage, saturation, solver

| | none | rate | full |
|---|---:|---:|---:|
| θ_s peak, **traversal window** [deg] | **0.3664** | 0.3668 | **0.3668** |
| θ_s peak, full log [deg] | 0.3664 | 0.5346 | 0.5346 |
| ‖h_w‖ peak per-axis [N·m·s] | 3.5050 | 4.1019 | 4.1019 |
| — as % of the ±5 box | **70.1 %** | **82.0 %** | **82.0 %** |
| ticks outside the box | 0 | 0 | 0 |
| saturation, any-axis per tick | **10.15 %** | 4.57 % | 4.57 % |
| saturation, axis-sample | 3.63 % | 1.59 % | 1.59 % |
| — SS | 3.32 % | 3.94 % | 3.94 % |
| — DS_interstep | **13.73 %** | **5.48 %** | **5.48 %** |
| — DS_terminal | 0.00 % | 0.00 % | 0.00 % |
| peak pre-clip demand [N·m] | 3.5158 | 3.8990 | 3.8990 |
| NMPC solves | 623 | 709 | 709 |
| IPOPT iters, total | 4302 | 7677 | **7705** |
| IPOPT iters, median / mean | 6 / **6.905** | 11 / 10.828 | 11 / **10.867** |

Saturation is **more frequent in the inter-step settles than in the swings** in
every arm — 13.7 % vs 3.3 % unconstrained, 5.5 % vs 3.9 % managed — confirming
C2.3 against three configurations rather than one.

### Traversal time, with the gate trace that explains it

| | none | rate | full |
|---|---:|---:|---:|
| log end [s] | 76.03 | 84.54 | 84.54 |
| last dock [s] | 56.03 | 64.54 | 64.54 |
| gate evaluations | 98 | 181 | 181 |
| refused | 92 | 175 | 175 |
| **refused while POSE-VALID** | **0** | **4** | **4** |

Per-step single-support duration, `none` → `full`:

| step | none | full | Δ | |
|---:|---:|---:|---:|---|
| 0 | 2.7 | 2.8 | +0.1 | |
| 1 | 9.0 | 8.9 | −0.1 | |
| 2 | 8.7 | 10.2 | **+1.5** | ← twist-refused |
| 3 | 9.2 | 9.1 | −0.1 | |
| 4 | 3.2 | 10.2 | **+7.0** | ← twist-refused |
| 5 | 8.8 | 9.0 | +0.2 | |
| **total** | **41.6** | **50.2** | **+8.6** | |

**Steps 2 and 4 alone account for 8.5 s of the 8.6 s — 99 %.** They are exactly
the two steps where the capture gate refused a pose-valid approach:

```
step 2  t=21.90  d=4.334 mm  ori=0.105°  twist=0.060522  -> refused (twist)
step 2  t=22.00  d=4.941 mm  ori=0.234°  twist=0.057386  -> refused (twist)
step 4  t=46.58  d=3.409 mm  ori=0.157°  twist=0.057616  -> refused (twist)
step 4  t=46.68  d=3.326 mm  ori=0.230°  twist=0.050607  -> refused (twist)
```

The `none` arm has **zero** pose-valid refusals; all 92 of its refusals are
position. So the +8.5 s is not a slower plan — it is four refusals against
`eps_twist = 0.05`, one of which misses by 1.2 %. C1.6 found this on the
published pair; it reproduces exactly in the clean design.

---

## 3. ⚠ The attitude benefit is the AOCS clip's, not the planner's

Same metric, same convention, across all four configurations:

| configuration | θ_s peak, traversal [deg] | θ_s peak, full log [deg] |
|---|---:|---:|
| published `u25` (planner **and** clip lifted) | **0.8949** | 1.1513 |
| **`none`** (planner lifted, **clip held**) | **0.3664** | 0.3664 |
| `rate` | 0.3668 | 0.5346 |
| `full` = `c25` canonical | **0.3668** | 0.5346 |

Removing the planner's entire envelope constraint set, with the actuator cap
held, changes the traversal attitude peak from **0.3668° to 0.3664°** — a 0.1 %
difference, in the *unconstrained* arm's favour. The 2.4× degradation visible in
the published pair (0.8949 vs 0.3668) is produced by lifting the **AOCS clip**,
which `u25` does simultaneously.

**A sentence attributing attitude performance to the momentum envelope is not
supported by this ablation.** What the rate constraint demonstrably buys is
**less actuator saturation** — 10.15 % → 4.57 % of ticks, peak demand 3.52 →
3.90 N·m — which is a planner-level claim about staying inside the actuator's
authority, not an attitude-accuracy claim. That is a narrower statement than the
paper currently makes, and it is the one the data carries.

Note also that the rate constraint **increases** peak storage use, 3.505 →
4.102 N·m·s (70 % → 82 % of the box): spreading the momentum transfer over time
to respect the rate cap parks more of it in the wheels.

---

## 4. What the `rate` column says about the storage constraint

**Storage is inactive at the 1 % ratio, and `rate` vs `full` is the measurement
that proves it.**

The two arms are indistinguishable in every physical metric: identical docks
(all six, same times, same distances to 3 dp), identical θ_s (0.3668 traversal,
0.5346 full log), identical ‖h_w‖ peak (4.1019), identical saturation (4.574 %),
identical per-step envelope and per-step durations. They are *not* bit-identical
— 42 011 of 182 776 compared fields differ — but every difference is in the last
significant digit of the 6-figure export, the largest relative deviation being
**7.9e-3 on `hw_x` at an absolute magnitude of 6e-7 N·m·s**, and the largest
solver effect being **±0.36 % on total IPOPT iterations** (7705 vs 7677, with
206 of 2077 ticks taking a different iteration count).

The physical reason is direct: **‖h_w‖ peaks at 4.1019 N·m·s, 82.0 % of the ±5
box, and no arm ever leaves it.** The storage constraint is present in the NLP
and never active. Its removal perturbs IPOPT's path through the problem — a
different active set at intermediate iterates — without moving the solution.

So the constraint is **inactive, not absent**: it costs 0.36 % of solver effort
and buys nothing measurable at this operating point. That is the expected
outcome the brief named, it is a clean result rather than a null one, and it is
what a narrowed storage claim would rest on. Demonstrating the constraint
binding requires an operating point where it does — RATIO-A at 2.1 % predicts
storage demand ≈ 8.6 against the ±5 box, which is Gate D's Option B.

---

## 5. Cost of the rate constraint, three-way

The +27 % iteration figure from the two-arm comparison was against `u25`, which
also lifted the clip. Clean three-way, over NMPC-active ticks:

| | none | rate | full |
|---|---:|---:|---:|
| mean IPOPT iterations / solve | **6.905** | 10.828 | **10.867** |
| vs `none` | — | **+56.8 %** | **+57.4 %** |
| vs `rate` (= storage alone) | — | — | **+0.36 %** |

**The rate constraint costs ~57 % more IPOPT iterations per solve; the storage
constraint costs 0.36 %.** These are deterministic and gate-reproducible, unlike
wall-clock, so they are the most citable cost measure available.

---

## STOP

Deliverables complete: three arms under one design, the `u25` config diff, the
gate verdict on `full` (PASS, byte-identical), and one table per metric family.

Three findings for the decision, none of which follows automatically to a paper
change:

1. **Storage is inactive at 1 %** — `rate` and `full` differ by 0.36 % of solver
   iterations and nothing physical. Narrowing the claim, or un-parking RATIO-A
   to find a binding point, is Idriss's call.
2. **The attitude benefit in the published pair is the AOCS clip's**, not the
   planner's (§3). This is a live correction risk in the same class as C1.6, and
   larger.
3. **The +8.5 s cost is four capture refusals**, 99 % of it on two steps, against
   an `eps_twist` documented in-source as untuned (§2).
