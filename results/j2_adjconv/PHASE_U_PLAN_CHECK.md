# Phase U-PLAN-CHECK — VERDICT: **reframe VALIDATED.** The rate-off U-plan EXCEEDS the envelope (z up to **10.0 = 2×**).

**Branch** `j2/ds-active-rework` · **measurement only, `crawlbot/` untouched, NO canonical change** · pushed, never merged.
Data: `results/j2_adjconv/uplan_check_result.json`; runner `scripts/diag_uplan_check.py`. Raw run
`results/figU_plan_check_mom400_Tw1e6` (gitignored). C reference: `nmpc_plan_sat_result.json` (mom 400).

## Verdict

With the rate cap **OFF** (`τ_w_max = 1e6`; NMPC Ḣ_s constraint inactive + WQP box inactive; **only** change vs
the C run), the NMPC plans a z-axis Ḣ_s of **up to 10.008 Nm — 2× the ±5 envelope — on exactly the arm-a steps
where the constrained C-plan pins z = 5.000.** The constraint IS what caps C. So C's planned saturation is a
**genuine, active-constraint momentum-management demonstration**, not the problem dynamics coincidentally
touching 5. **REFRAME VALIDATED.**

## The comparison — C (capped) vs U (rate-off), same Add-5 base, only `τ_w_max` differs

Fixed both: torso 2000 · EE 1000 · hw-slack 800 · posture 20 · torque 5 · wrench 1 · accel-reg 1 · ε 1e-6 ·
`ss_alpha_mom` 400. C: `τ_w_max = 5`. U: `τ_w_max = 1e6`.

| per arm-a step | C-plan z (capped) | **U-plan z (rate-off)** | exceeds? |
|---|---|---|---|
| step 0 (arm a) | 5.000 | **7.831** | **+2.83** |
| step 2 (arm a) | 5.000 | **8.009** | **+3.01** |
| step 4 (arm a) | 5.000 | **10.008** | **+5.01** |

| per-axis SS peak | C-plan [x,y,z] | **U-plan [x,y,z]** |
|---|---|---|
| planned ‖·‖∞ | [1.89, 3.53, **5.00**] (z capped) | [1.65, 3.69, **10.008**] (z 2× over) |

**Every arm-a step where C pins z = 5.0, U shoots to 7.8–10.0.** The unconstrained NMPC genuinely wants far more
aggressive momentum; the `|Ḣ_s,i| ≤ 5` constraint is what holds C at the envelope.

## Internal consistency — the active-constraint signature

| per-step planned z (arm a b a b a b) | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| C (capped) | **5.00** | 3.62 | **5.00** | 2.51 | **5.00** | 3.53 |
| U (rate-off) | **7.83** | 3.68 | **8.01** | 3.58 | **10.01** | 2.95 |

On the **arm-b steps (1, 3, 5)**, where C-plan does **not** saturate (~2.5–3.6), the U-plan **agrees** (~2.9–3.7) —
the constraint is slack there in both. The plans diverge **only** on the arm-a steps where C saturates. That is
the textbook signature of a binding constraint: **plans identical where the constraint is inactive,
C-clipped-below-U where it is active.** If C's z = 5.0 were mere problem dynamics, U would match it on arm-a
too; instead U doubles it.

## Per-axis ‖·‖∞ vs Euclidean

The binding axis is **z** (U 10.008); x, y (1.65, 3.69) stay well under 5. The constraint is per-component, so
only z hits/exceeds the cap; the Euclidean ‖Ḣ_s‖₂ ≈ 10.7 is neither computed nor bounded. Per-axis is the
convention that exposes the binding; Euclidean would misread it.

## U-realized + reconciliation of the cited ~7.48

- **My Add-5-base U-realized** (box off): per-axis [1.03, 1.94, **3.39**], ‖·‖₂ = 3.53.
- **The cited ~7.48** is the **realized z-axis peak of the committed `figU_rateoff` run** (canonical/different
  weights): realized [1.42, 3.51, **7.48**], ‖·‖₂ = 7.93 (recomputed from its on-disk sim_log with the same
  method). Its plan z-peak is 6.27 — also **> 5**.
- The realized Ḣ_s is **base-dependent** (NMPC-PLAN-SATURATION: realized doesn't track the plan; it's set by the
  WQP task demand + the full weight vector). At the Add-5 base the realized stays gentle (z 3.39) **even with the
  box removed** (r/p = 0.34) — the box was not the limiter here. Both U runs' **plans** exceed 5 (Add-5 z 10.0,
  figU z 6.27), so the reframe holds across weight sets.

## Verbatim IPOPT (`return_status`, SS solves)

**421 / 421 `Solve_Succeeded` (100%).** The z = 10.0 plan is **IPOPT-optimal** — genuinely what the unconstrained
NMPC plans, not a solver artifact. `swing_ref_pk = 0.0` (stance-only = exact full Ḣ_s). `qp_fail = 0`. Docks
**6/6** (rate-off still docks at this base; struct drift 21.6 mm vs C's ~10 mm — the more aggressive momentum
shows up as extra base motion, but not a dock failure).

## Bottom line

NMPC-PLAN-SATURATION and U-PLAN-CHECK together close the loop: the unconstrained NMPC **wants** z-Ḣ_s up to 10 Nm
(U), the `|Ḣ_s,i| ≤ 5` constraint **caps** it at exactly 5 (C), and this holds at every WQP momentum weight. The
momentum management is a **real, active NMPC constraint** — the planned-saturation reframe is a valid
demonstration of it, not an artifact of the dynamics.

## Files

| artifact | path |
|---|---|
| U result JSON (per-axis, per-step, IPOPT) | `results/j2_adjconv/uplan_check_result.json` |
| runner | `scripts/diag_uplan_check.py` |
| C reference (mom 400) | `results/j2_adjconv/nmpc_plan_sat_result.json` |
| prior canonical rate-off run (source of the 7.48) | `results/figU_rateoff/sim_log.json` |

**STOP for cross-check.** Push only, never merge. `crawlbot/` untouched.
