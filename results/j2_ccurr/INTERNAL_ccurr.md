# INTERNAL — J2 c_curr: per-tick inter-step QP hw_current refresh (implement + characterize)

**Implementation brief, then characterize — NO success threshold.** Raw numbers; the C5 verdict and whether
to keep it on are decided by Idriss + reviewing Claude. Branch `j2/ds-active-rework` (pushed, never merged).
Base `ae0673e`. Default-on; OFF reproduces the entry-frozen path byte-identical.

---

## DECISIVE OUTPUT

**(a) How much C5 margin does c_curr recover on its own (2A, proxy box)? — ZERO.** With AOCS-on, proxy box,
`interstep_hw_refresh` OFF vs ON the runs are **byte-identical** (C5 4.483 → 4.483; whole sim worst |Δ|=0).
The per-tick refresh removes a real staleness (worst **0.387 Nms** of intra-settle hw drift the entry-frozen
value missed) **but recovers no C5 margin** — because the inter-step h_w∞ is only **1.58 Nms (≪ the 4.5 cap)**,
so the QP momentum-safety box **never binds** in the inter-step loop; refreshing a non-binding constraint's
RHS changes nothing. The C5 binding is entirely in SS, which c_curr does not touch.

**(b) Does exact box + AOCS-on + c_curr hold C5 (2B)? — NO.** C5 = **4.949 (FAIL)**, residual gap **0.449
Nms** over the 4.5 budget — **unchanged** from exact+AOCS-on *without* c_curr (4.949). c_curr does not help,
and the h_w breakdown says why: the exact-box C5 h_w∞ = the **SS** value exactly (SS 4.949 = whole;
inter-step only **1.30**). **The C5 pressure is 100% the SS exact-box binding; the inter-step — c_curr's
domain — is far from the cap.** C1 (dock 4.99) and C2 (pos-peak 28.0 mm) also still FAIL, unchanged. **⇒ the
next lever for the exact-box milestone is the SS working point, NOT the inter-step.** c_curr is
architecturally correct (matches the `_step` QP / `c_simple(k)`, removes the staleness, future-proofs against
the inter-step box ever binding) but inert at this working point.

---

## Part 1 — Implementation (committed `cef47d1`; default-on, gated, OFF byte-identical)

- **`cfg.interstep_hw_refresh`** (default **True**). In `_run_ds_passivity_loop`, before each per-iteration
  `qp.solve`, the QP's `hw_current` is refreshed to the **live** wheel momentum
  `hw_for_qp = rwa_I_w · qvel[6:9]` (same source as the `_step` per-tick refresh), instead of the
  loop-entry-frozen value (`sim_loop.py:701`). When False, the entry-frozen value is used.
- **Parameter refresh, not a co-solve** (non-co-integration audit): `hw_current` is the numeric RHS of the
  momentum-safety box `b = hw_max − hw_current`; refreshing it is a fresh value frozen at solve, **h_w is not
  a decision variable**. **The QP decision vector `{qdd_t, qdd, λ, τ_q, slack}` is unchanged** (confirmed —
  no new variable, no constraint/cost/AOCS change). Mirrors `c_simple(k)`'s per-MPC-step refresh exactly.
- AOCS-on (`aocs_active_in_interstep=True`) for all c_curr runs (c_curr only matters because the wheels move).
- Plumbing: `--no-interstep-hw-refresh` (diag A/B); forced off under `--baseline_ds_rework`.

## Part 2A — c_curr isolated (proxy box, AOCS-on, refresh OFF vs ON), raw

| metric | refresh OFF | refresh ON |
|---|---|---|
| **C5 h_w∞** (per-axis, cap 4.5) | 4.483 (PASS) | 4.483 (PASS) |
| h_w∞ split: SS / inter-step / DWELL [Nms] | 4.483 / 1.576 / 1.445 | 4.483 / 1.576 / 1.445 |
| worst intra-settle staleness ‖hw_live−hw_entry‖ [Nms] | **0.387** | 0.387 |
| settle durations (ticks) | 11/175/51/104/51 | 11/175/51/104/51 |
| residual (traversal-final) [N·m·s] | 0.004022 | 0.004022 |
| C1 dock [mm] | [4.94,4.51,4.91,4.72,4.85] PASS | identical PASS |
| C2 / C3 / C4 | PASS / PASS / PASS | PASS / PASS / PASS |
| **whole-sim diff OFF vs ON** | — | **BIT-IDENTICAL (worst \|Δ\|=0)** |

- **c_curr is a no-op at this working point.** OFF and ON are byte-identical on every physical field. The
  staleness it *would* remove (0.387 Nms, in settles 1 & 3 where the AOCS slews the wheels) is real, but it
  lives where the inter-step h_w is small (entry ‖hw‖ 0.4–1.7 → max 0.5–1.7 Nms, all ≪ 5), so the QP
  momentum box is slack there and the RHS value is irrelevant to the solution.
- **The staleness, quantified (what c_curr fixes if it ever matters):** the entry-frozen hw_current lagged
  the live value by up to 0.387 Nms within a settle (settle 3); the longer settles (n=174, n=103, where the
  AOCS works) accrue the most. c_curr removes this lag, but at hw ≤ 1.7 Nms it buys no C5 margin.
- Settle convergence, dock, residual, C1–C4 all unchanged (byte-identical). C6 (below) bit-identical.

## Part 2B — TARGET combination: exact box + AOCS-on + c_curr ON (the decisive C5 verdict), raw

| criterion | exact + AOCS-on + c_curr ON | (vs exact+AOCS-on, no c_curr) |
|---|---|---|
| **C5 h_w∞** (cap 4.5) | **4.949 — FAIL** (gap +0.449) | 4.949 (unchanged) |
| h_w∞ split: **SS / inter-step / DWELL** [Nms] | **4.949 / 1.300 / 2.498** | — |
| C1 dock [mm] | [4.94,4.44,**4.99**,4.64,4.73] — FAIL (margin 0.01) | 4.99 (unchanged) |
| C2 torso | ori_rms 0.111 / **pos-peak 28.0 mm** — FAIL | 28.0 (unchanged) |
| C3 envelope | ‖Ḣ_s‖∞_SS 5.00, per-axis [3.3, 5.0, 5.0] — PASS | unchanged |
| C4 attitude | peak 0.62 / final 0.11 — PASS | unchanged |
| residual (traversal-final) [N·m·s] | 0.002987 | — |

- **c_curr does NOT bring C5 under 4.5.** It stays at 4.949 (FAIL), identical to the no-c_curr exact run.
- **The h_w breakdown is decisive: C5 h_w∞ = the SS value exactly (4.949 = SS = whole).** The inter-step
  (c_curr's domain) is only 1.300 Nms, the DWELL/terminal DS 2.498 — both well under 4.5. **So the entire C5
  pressure is the SS exact-box binding** (C3 confirms two SS axes saturate the envelope: [3.3, 5.0, 5.0]).
  c_curr operates only on the inter-step QP, which is nowhere near the cap, so it cannot move C5.
- **C1 / C2 also unchanged-FAIL** — the dock 4.99 (margin 0.01) and torso pos-peak 28 mm are exact-box SS
  effects (the corrected envelope loads the wheels and perturbs the SS torso solution); the cleaner inter-step
  hw does not help them either, for the same reason (they are SS-driven).
- **Next lever:** the residual gap is **0.449 Nms on C5**, located **in SS**. The exact-box milestone (full
  traversal with correct envelope physics, all-green) requires an **SS-working-point** adjustment (the SS
  momentum/torso weights or the SS envelope handling) — not an inter-step change.

## C6 — flag-OFF determinism

**BIT-IDENTICAL.** New-code flag-off (`--no-interstep-hw-refresh`) vs old-code (`7862e7f`) canonical 5-step:
worst |Δ| = **0.000e+00** on every physical field (only wall-clock timers differ). The gate also reports C6
OFF/test BIT-IDENTICAL and `test_reworked_qp` 8/8 for all three runs.

## Regression (`pytest tests/`)

**220 passed, 1 failed** (916 s, new code at `cef47d1`). The single failure is the **pre-existing** FK test
`test_E7_t15_step2_dock_under_fk_mode` (known/unrelated, identical on clean `ae0673e`) — **no NEW failures**.
The flag-off path is byte-identical (C6) and, at this working point, the flag-ON path is *also* byte-identical
to flag-off (the inter-step box never binds), so the default-on change is provably inert here.

---

## Flags / divergences vs prior audit facts

1. **Confirms the non-co-integration audit:** c_curr is a parameter refresh — the QP decision vector
   `{qdd_t, qdd, λ, τ_q, slack}` is unchanged (no new variable, h_w never in the decision vector); refreshing
   the frozen RHS is architecturally identical to `c_simple(k)`. The decentralization framing holds.
2. **Refutes this brief's premise** that the inter-step staleness *loses C5 margin*. The staleness is real
   (0.387 Nms) but inconsequential at this working point: the inter-step h_w (≤1.58 Nms) is far below the box,
   so the box is slack there and c_curr is byte-identical to entry-frozen (worst |Δ|=0). C5 margin is set by
   SS, not the inter-step.
3. **Sharpens the step-4a C5 finding:** step-4a flagged that exact+AOCS-on gives C5 4.949 and asked whether
   c_curr recovers it. Answer: **no** — and the h_w breakdown localizes the entire 0.449-Nms gap to the **SS
   exact-box binding** (SS 4.949 = whole; inter-step 1.30). The next lever is the SS working point. This is
   consistent with the Piste-A finding that the exact box's cost comes from the SS binding (~+0.5–0.76 Nms),
   not the DS.
4. **c_curr is correct-but-inert here:** it should stay (it matches `_step`/`c_simple(k)` and future-proofs
   the inter-step box), but it is *not* the lever for the exact-box milestone. No divergence from AOCS-FF /
   Piste-A facts; the τ_w-not-a-QP-variable and 9-D-centroidal invariants are untouched.

## Reproduce
```
bash scripts/run_ccurr.sh   # proxy OFF/ON + exact ON, 5-step, analysis + residual + gate
MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_ccurr.py \
  proxy_off=results/ccurr_proxy_off proxy_on=results/ccurr_proxy_on exact_on=results/ccurr_exact_on
```
Supporting: `results/j2_ccurr/{ccurr_analysis.log, ccurr_gate.log, ccurr_residual.log}`. C6 diff: new-code
flag-off vs old-code `7862e7f` (bit-identical). Raw per-run dirs reproducible from the script, not committed.

**STOP after the report.** No success threshold; the C5 verdict (exact-box milestone needs the SS working
point) and keep-default-on are decided by Idriss + reviewing Claude. No merge, no PR.
