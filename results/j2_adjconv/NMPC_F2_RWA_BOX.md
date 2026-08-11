# F2 — RETRACTED. The RWA conservation box was never off.

**Branch** `claude/com-gain-semantics-audit-j0u6yr`

---

## 0. Retraction

**NMPC_AUDIT finding F2 was wrong.** It claimed the RWA conservation box and its
terminal set are disabled in the canonical run. They are not, and never were.

Measured from the config the canonical actually uses
(`scripts/audit_nmpc_structure.py`, now building from `_make_m7_config()`):

```
[ON] SOC ||f_j||^2, ||tau_j||^2          4 rows
[ON] wheel-torque cap |Hdot_s,i|         6 rows   tau_w_max = 2.5
[ON] linear momentum ||m v||^2           1 row    p_max = 50
[ON] RWA conservation box h_w(k)         6 rows   h_max_tight = [5,5,5]
[ON] terminal |h_w(N)| <= kappa*h_max    6 rows   kappa = 1.0
ng_path = 17   ng_term = 6   rows = 535
```

### How the error happened

`SimConfig.enforce_hw_conservation` defaults to `False`, and I built the
structural audit from a bare `SimConfig()`. But `dca.main` does not use the
defaults — it calls `run_m7_single_step._make_m7_config()`
(`diag_cooperative_arms.py:268`), which constructs `SimConfig(...)` with
**explicit kwargs**:

```python
enforce_hw_conservation=True,
h_max_tight=np.full(3, 5.0),
kappa_terminal=1.0,
```

So the audit measured a configuration nothing runs. This is precisely the trap
`docs/crawlbot/solvers/centroidal_nmpc.md` §3 already documented — *a dataclass
default is not the canonical value* — and I walked into it, then **overwrote
that correct warning with a wrong "correction"**. Both are now restored, and
`audit_nmpc_structure.py` builds from `_make_m7_config()` so it cannot recur.

### What was consequently also wrong

| earlier claim | truth |
|---|---|
| "12 documented constraint rows are never emitted" | all 12 are emitted |
| "`c_simple` is computed every solve and read by nothing" | it is read by the box, every solve |
| "`h_max_tight`, `kappa_terminal` are inert" | both live |
| "F7: the envelope enters only through the disabled box and the rate cap" | the box is live, so accumulated `h_w` **is** bounded inside the horizon |
| "`ng_path=11`, `ng_term=0`" | 17 and 6 |
| the doc "asserted the OPPOSITE of the code" | the doc was right; my correction was the error |

## 1. What the F2 experiment actually measured

Four replays were run believing they varied the box. They did not: the sweep
driver patches `SimConfig` **defaults**, which `_make_m7_config()` overrides for
`enforce_hw_conservation` and `h_max_tight`. So every run had box ON at
h_max = 5.0, and **the bite test never ran at 3.5** — its premise was void.

One field *was* varied, because `_make_m7_config()` does not set it:
`enforce_hw_terminal`.

| run | box | h_max | terminal | what it really was |
|---|---|---|---|---|
| `F2off_ctl_N20` | ON | 5.0 | ON (`None` → follows box) | = canonical |
| `F2box_N20` | ON | 5.0 | **OFF** | **the only real variation** |
| `F2boxterm_N20` | ON | 5.0 | ON | = canonical |
| `F2bite_h35_N20` | ON | 5.0 | ON | = canonical (not 3.5) |

That explains every anomaly at once: three runs were byte-identical because they
*were the same configuration*, and `F2box` was the sole outlier because removing
the terminal set is the only thing that changed. The "bite test didn't bite"
because the box was at 5.0 and the h_w peak is 3.815 — comfortably inside.

**Salvageable result:** removing the terminal constraint (`F2box`) perturbs the
run by 46 698 of 125 888 fields at ≤ 6e-4 relative — 9 µm of CoM, 1.3e-5° of
θ_s — with docks, θ_s and h_w unchanged to printed precision and 639/639 solves
succeeding either way. So the terminal set is **inactive at the current
operating point**, consistent with the path box being non-binding.

## 2. What is genuinely established

- The box **is** live in the canonical, per-axis, at ±5 Nms on each component.
- It is **non-binding** at the current operating point: realized per-axis peaks
  are x 0.583, y 2.339, **z 3.815** —
  24 % headroom on the worst axis. Curves: `nmpc_f2_peraxis.png`.
- `hw_current` handed to the NMPC matches the exported `hw_*_Nms` exactly
  (per-axis peaks identical to 4 dp), so there is **no filtered/stale-telemetry
  problem** — that hypothesis is refuted.
- The constraint rejects violating states when they occur: the same NLP given
  `hw_current` with an axis at 3.8146 against `h_max = 3.5` returns
  `Infeasible_Problem_Detected` (and at 3.0 too).

## 2.1 Hard in the NMPC, soft in the QP — and why that matters here

⚠ **Terminology.** Earlier drafts of this report called the NMPC box "slack",
meaning *non-binding* in the NLP sense. In this codebase that reads as
"implemented with slack variables", which is the opposite of the truth. The two
tiers are genuinely different:

| tier | mechanism | site |
|---|---|---|
| **NMPC** | **hard** per-axis inequality — `lbg = −∞`, `ubg = 0`, no slack variables | `nmpc_solver.py:394-395` (path), `:421-422` (terminal) |
| **QP** | **soft** — 6 slack variables `slack_hw_up/lo` added as a penalised task at `w_hw_slack = 800` | `wholebody_qp.py:680-687` |

That split is deliberate and correct: the *plan* must be feasible against the
wheel envelope (so the NMPC refuses to propose an inadmissible trajectory), while
the *instantaneous tracker* must never itself become infeasible (so the QP is
allowed to exceed the box at a price rather than fail).

**The consequence for F2 is the important part.** Because the NMPC box is hard,
it guarantees the **planned** `h_w` stays inside ±5 Nms at every knot. It does
*not* guarantee the **realized** `h_w` does, because the plant does not follow
the plan exactly — the QP only weakly tracks `lambda_ref` (`alpha_wrench = 1.0`,
a regulariser), and the AOCS acts on top. So:

- a hard NMPC box bounds the plan;
- the realized `h_w` is bounded only softly, by the QP's penalised box;
- the two can and do differ.

That is why "does the box bind?" and "is realized `h_w` inside the envelope?" are
separate questions, and why the bite test — had it actually applied 3.5 — would
have answered only the first.

## 2.2 The box is NOT redundant — and "24 % headroom" was the wrong unit

Measured by `scripts/audit_nmpc_momentum_budget.py`.

The two momentum constraints bound the same physical quantity at different
orders: `|Ḣ_s,i| ≤ τ_w,max` is the moment the wheels must absorb *now*;
`|h_w,i| ≤ h'` is what they have *accumulated*, and h_w is the integral of that
moment. So the level bound is implied by the rate bound whenever
`T·τ_w,max ≤ h'`. Here:

```
T·τ_w,max = 2.0 × 2.5 = 5.0 Nms     h_max_tight = 5.0 Nms     ->  CRITICALLY BALANCED
```

From `h_w(0) = 0` the rate cap alone already implies the box. **The box
therefore adds information only through the initial condition** — which is
exactly why it reads as non-binding, and exactly why that reading is
misleading.

Expressed in the units that matter — how long `τ_w` could stay saturated in one
direction before the box is reached:

| axis | \|τ_w\| peak | ticks at cap | \|h_w\| peak | headroom | **= saturated seconds** |
|---|---|---|---|---|---|
| x | 2.5000 | 6 / 1967 | 0.583 | 4.417 | 1.77 s |
| y | 2.5000 | 52 / 1967 | 2.339 | 2.661 | 1.06 s |
| **z** | 2.5000 | **274 / 1967** | **3.815** | 1.185 | **0.47 s** |

All three are **reachable inside the 2.0 s horizon**, and z — the axis that is
at the torque cap 14 % of the time — has under half a second of margin. So the
box is genuinely load-bearing; it simply was not exercised on this trajectory.
Reporting it as "24 % headroom" understated it badly: 24 % of the box is 0.47 s
of saturated authority.

## 2.3 The term the reconstruction does not carry

`c_simple = h_w0 + L_com0 + r_com0 × m·v_com0` attributes all momentum change to
the wheels. The structure itself also carries angular momentum, `I_s·ω_s`, and
it is not small:

| axis | \|ω_s\| peak [rad/s] | \|I_s·ω_s\| peak [Nms] |
|---|---|---|
| x | 6.82e-4 | **1.212** |
| y | 9.09e-4 | **1.357** |
| z | 1.34e-3 | 0.800 |

(structure inertia diag `[1777, 1493, 597]` kg·m², mass 7110 kg.)

**Peak structure momentum 1.357 Nms vs the tightest box headroom 1.185 Nms —
the same order.** So the inferred `h_w` and the physical wheel momentum can
differ by about as much as the margin the box exists to protect.

⚠ Whether this is an omission or is accounted for elsewhere is **not settled
here**. `compute_c_simple`'s docstring states that drag terms
(`I_robot·ω_s`, `m·r_com×v_s`) "cancel algebraically when forming c_simple from
the full c" — it does not mention `I_s·ω_s`, and the authority is spec §4.5-4.6,
which has not been re-derived against the code in this pass. That check is the
natural next step, and it matters more than re-running the bite test: if the
inference carries a ~1 Nms error, the hard box is bounding a quantity that is
offset from the wheels' true state by a third of its own margin.

## 2.4 The box defends a within-step EXCURSION, not a traversal-scale wind-up

Measured by `scripts/audit_nmpc_momentum_budget.py`; curves
`nmpc_sweep/nmpc_momentum_budget.png`.

§2.2 established the box is load-bearing. It does **not** follow that `h_w`
creeps upward over the traversal — and the distinction changes what the box is
*for*. Start, peak and end of `h_w` over the full 6-step run:

| axis | start | peak | end | **net drift** | peak / drift |
|---|---|---|---|---|---|
| x | +0.000 | −0.583 | +0.001 | **+0.001** | 489× |
| y | +0.000 | +2.339 | +0.098 | **+0.098** | 24× |
| **z** | +0.000 | **−3.815** | −0.239 | **−0.239** | **16×** |

**The wheels come back to ~0 after six steps.** Even on z the residue is 0.24
N·m·s against a 3.82 N·m·s excursion. So the momentum taken on to reject a
contact wrench during a step is handed back when the step reverses — the
traversal is momentum-neutral to 6 % per step, and there is no secular
accumulation for the box to catch.

That means the box's job is **one long push inside a single step**, not slow
wind-up across many. It is a *transient* bound. Which is consistent with §2.2:
`T·τ_w,max = h_max` exactly, so the box and the rate cap describe the same
2.0 s window — there is no longer timescale on which the level bound could
speak.

### Would pushing the robot further from the structure CoM make it bind?

The intuition is right in its first step and wrong in its second. A longer
lever arm does raise `Ḣ_s`, because the contact **force** acts about the
structure CoM — that is a moment `r × f`, and `r` is what grows. (Contact
*torques* get no such amplification; only the force term has a lever.)

But the *delivered* rate is capped, and on z it is already at the cap:

| axis | median \|Ḣ_s\| | p95 | max | ticks at cap (of 1967) |
|---|---|---|---|---|
| x | 0.075 | 0.799 | 2.500 | 9 |
| y | 0.159 | 1.569 | 2.500 | 64 |
| **z** | 0.276 | **2.500** | 2.500 | **302 (15 %)** |

z's *95th percentile* is the cap. So a longer lever raises the **demand**, not
the delivered `τ_w` — the wheels cannot absorb faster than 2.5 N·m regardless
of how big the moment gets. `h_w` therefore does not fill faster.

What grows instead is the **duration** of saturation, and `h_w` integrates
duration:

```
|h_w,z| peak 3.815 Nms = 1.53 s of saturated tau_w
reaching h_max = 5.0  = 2.00 s continuous   (+31 % longer saturation window)
```

So the answer is yes — but through the time axis, not the amplitude axis, and
the required change is modest: 31 % more continuous saturation on z, well
inside a step.

### The catch that makes such a demonstration hard to read

Every second of saturation is a second in which the wheels **cannot fully
reject the applied moment**. The unrejected excess goes into the structure:
`ω_s` grows, and with it `I_s·Δω_s` — which is exactly the frozen-platform
reconstruction error quantified in `NMPC_CONSERVATION_DRAG.md` §3, already at
**107 % of the z margin** today.

**Binding the box and degrading the estimate it is enforced on are driven by
the same quantity.** A "push it further out until the box bites" experiment
would therefore be measuring a hard constraint on a number whose error is
growing at least as fast as the signal. Before running one, the reconstruction
residual `ĥ_w − h_w` and `v_s` must be export channels (§7 of the drag report),
otherwise the result is uninterpretable: a box that appears to bind may be
binding on estimator drift.

## 2.5 Correction to §2.4 — the crawl DOES carry; the terminal settle hides it

Measured by `scripts/audit_nmpc_hw_per_step.py`; curves
`nmpc_sweep/nmpc_hw_per_step.png`.

§2.4's "net drift ≈ 0, therefore no accumulation" is **contaminated by the
trailing settle** and must not be used to predict a longer traversal. The run's
phase structure is:

```
step 0..5 = SS (2.8 .. 10.1 s) + DS_interstep (2.0 .. 3.2 s)     <- the crawl
step 5 ends in DS_terminal, 20.0 s                               <- the settle
```

That 20 s settle is long enough to hand the wheel momentum back; the 2–3 s
`DS_interstep` between crawl steps is not. Measuring the carry as `h_w` at
successive **SS entries**, with the terminal settle excluded:

| axis | carry / crawl step [Nms] | \|mean\|/std | verdict | settle returns |
|---|---|---|---|---|
| x | −0.020 | 0.36 | zero-mean | +0.013 (14 %) |
| y | +0.150 | 0.87 | zero-mean | −0.430 (57 %) |
| **z** | **−0.314** | **1.18** | **SYSTEMATIC** | **+0.999 (64 %)** |

z's per-step carries are −0.092, −0.688, +0.030, −0.291, −0.527 — four of five
negative, mean −0.314 ± 0.119 (standard error, n = 5), ≈ 2.6σ from zero.
**Suggestive, not conclusive at n = 5**, and the whole point of a longer run
would be to settle it.

The mechanism is a DS-duration effect: `DS_interstep` is too short to dump what
the SS took on, so each crawl step hands the next a more negative starting
point. The SS-entry sequence walks down monotonically: −0.029, −0.122, −0.809,
−0.780, −1.071, −1.598.

### Extrapolation — a 20 m path is ~8× more than needed

Modelling `peak(N) ≈ |carry|·N + excursion`, with the worst observed
within-step excursion recurring on top of an accumulated entry:

| axis | carry/step | worst excursion | peak now | N to 3.1 | N to 5.0 | ≈ metres |
|---|---|---|---|---|---|---|
| x | −0.020 | 0.625 | 0.583 | 126 | 222 | 82 m |
| y | +0.150 | 1.764 | 2.339 | 9 | 22 | 8 m |
| **z** | **−0.314** | **2.744** | **3.815** | **already** | **~7** | **2.6 m** |

CoM travel is 0.143 / 0.546 / 0.251 / 0.563 / 0.172 / 0.529 m per step, mean
**0.367 m**, so a 20 m path is ≈ **54 steps** — z would reach the full ±5 box
around step **7**, roughly 2.6 m, if the terminal settle is removed.

⚠ The excursion varies 0.14 … 2.74 Nms step to step, so "7" is an
order-of-magnitude estimate, not a prediction. What it does establish is that
the required scale is **single-digit steps**, not tens of metres.

### The estimator does NOT degrade with path length

`compute_c_simple` is called from `_assemble_params`
(`centroidal_nmpc.py:699`), i.e. **every solve**, from the live `hw_current`.
So the conservation constant is re-anchored every 100 ms and the
frozen-platform residual `I_s·Δω_s` accrues over **one horizon**, not over the
traversal. A longer path therefore grows the signal without growing the
reconstruction error proportionally.

This is the key asymmetry between the two routes to making the box bind:

| route | grows `h_w` by | grows `I_s·Δω_s`? |
|---|---|---|
| longer lever / further from structure CoM | longer saturation duration | **yes** — same driver (§2.4) |
| **more crawl steps, no settle** | **systematic carry** | **no** — c re-anchored per solve |

**More steps is the cleaner demonstration.** It is the one route that does not
degrade the estimate the box is enforced on.

## 3. What is NOT established, and needs a real bite test

Whether the box would bind under stress is **still untested**, because the test
meant to show it never applied its tightened bound. A valid bite test must
override `_make_m7_config()`'s explicit kwarg — patching the `SimConfig` default
is not enough.

⚠ The same applies to **every** field `_make_m7_config()` sets explicitly:
`use_m2_stack`, `alpha_passivity`, `enforce_hw_conservation`, `h_max_tight`,
`w_L_nmpc`, `kappa_terminal`, `aocs_mode`, `aocs_use_legacy_corrected`,
`aocs_use_H_estimator`, and the `preplanner_*` group. All are **immune to
default-patching**.

Recommended: give the sweep driver a second patch target in
`run_m7_single_step.py`, and **assert the value from inside the run** rather
than inferring it from the file. The probe here only caught the problem because
it recorded `h_max` per solve and the recorded value disagreed with the intent —
that assertion is the durable fix.

## 4. Standing changes from this work

The `enforce_hw_terminal` split is kept — it is correct, and it is what allowed
the terminal set to be isolated at all:

| field | meaning |
|---|---|
| `enforce_hw_conservation` | the path box, k = 0..N−1 |
| `enforce_hw_terminal` | the terminal set. `None` = follow the path box (historical coupling), `True`/`False` overrides |

`SimConfig` defaults are unchanged, and since `_make_m7_config()` leaves
`enforce_hw_terminal` at `None`, the canonical is unaffected: terminal follows
the box, exactly as before.

## 5. Effect on the other findings

- **F1, F3** — unaffected. `nmpc_N`, `nmpc_pred_dt`, `nmpc_period` and
  `nmpc_per_stage_refs` are **not** set by `_make_m7_config()`, so the driver's
  default-patching did reach them. Verified: the canonical config reports
  `nmpc_N=20`, `nmpc_pred_dt=0.1`, `nmpc_per_stage_refs=True`.
- **F4** — unaffected; the rate cap still binds ~58 % of SS ticks.
- **F7** — **retracted in part.** The wheel envelope does reach the NLP as a
  bound on accumulated `h_w`, not only through the rate cap. `L_com`'s state box
  remains a separate, coarser bound.
- **F5, F6** — unaffected.

## 6. Credit

The thread that unravelled this was Idriss's question about `hw_current` being
scalar. It is not — it is a proper 3-vector and the box is component-wise — but
checking it forced a per-axis re-analysis, which forced instrumenting what the
NMPC actually receives, which is what exposed that the audit had been reading
the wrong config all along.
