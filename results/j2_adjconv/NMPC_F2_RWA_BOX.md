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
