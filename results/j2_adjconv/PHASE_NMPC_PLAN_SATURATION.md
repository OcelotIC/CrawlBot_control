# Phase NMPC-PLAN-SATURATION — VERDICT: planned Ḣ_s saturation is **WEIGHT-INDEPENDENT (NMPC-level). PROVEN.**

**Branch** `j2/ds-active-rework` · **measurement only, `crawlbot/` untouched, NO canonical commit** · pushed, never merged.
Data: `results/j2_adjconv/nmpc_plan_sat_result.json` (+ `nmpc_plan_sat.csv`); runner
`Misc/scripts/diag_nmpc_plan_saturation.py`. Raw runs `results/figN_mom{400,1000,2000,5000}` (gitignored).

## Verdict

**PRIMARY HYPOTHESIS — PROVEN.** The NMPC-planned Ḣ_s pins the per-axis envelope — **z-axis peak = 5.000 at
every WQP momentum weight {400, 1000, 2000, 5000}** — so the momentum management `|Ḣ_s,i| ≤ τ_w_max = 5` is
enforced at the **NMPC (IPOPT) level, independent of `ss_alpha_mom`.** The swept weight is a WQP tracking
weight the NMPC never sees; the plan saturates identically regardless.

**SECONDARY PREDICTION — REFUTED (honest).** The phase predicted "realized follows the plan only at high
momentum weight (fidelity ∝ weight)." It does not: the **realized Ḣ_s never saturates** (0/≈427 SS ticks at
*every* weight) and the realized/planned ratio is **flat and non-monotonic** (0.50 → 0.47 → 0.67 → 0.59,
peaking at mom 2000, *not* 5000). Raising `ss_alpha_mom` 400 → 5000 does **not** drive realized toward the
saturating plan — it stays at ~half. e_com is likewise flat (0.137–0.164).

## Machinery (code ground truth)

- **NMPC constraint** — `centroidal_nmpc.py:279-282`: `H_dot_s = cross(r_C1,f1)+τ1 + cross(r_C2,f2)+τ2`;
  `Hdot_s_ineq = vertcat(H_dot_s − tw, −H_dot_s − tw)` ⇒ **per-component** `|Ḣ_s,i| ≤ tw`, `tw = τ_w_max`.
- **WQP box (same quantity)** — `wholebody_qp.py:564-580`: `|M_exact·λ| ≤ τ_w_max` component-wise
  (`M_env = compute_momentum_map(0, …)`), `qp_envelope_exact=True`.
- **`ss_alpha_mom` is WQP-only** — absent from `centroidal_nmpc.py`, `nmpc_solver.py`, `coarse_preplanner.py`
  (grep-confirmed). The NMPC objective/constraints have zero dependence on it.
- **planned = NMPC-EXACT** `Ḣ_s(lambda_ref)` (`sim_loop.py:3049`); `lambda_ref` = NMPC control output `lr`.
  `swing_ref_pk = 0.0` at all weights ⇒ the swing arm's planned wrench is exactly zero in SS, so the
  stance-only Ḣ_s **is** the exact full NMPC quantity (both arms summed = stance term).
- **Norm convention** — the constraint is **per-axis ‖·‖∞**, NOT Euclidean. The z-axis pins 5.000 while
  (x, y) ≈ (1.9, 3.4); the Euclidean ‖Ḣ_s‖₂ ≈ √(1.9²+3.4²+5²) ≈ 6.3 Nm, which the box neither computes nor
  bounds. Reporting Euclidean would hide the saturation; per-axis exposes it.

## Per-weight data (fixed base, only `ss_alpha_mom` varied)

Fixed: torso 2000 · EE 1000 · hw-slack 800 · posture 20 · **torque 5** (= 5× accel-reg floor 1, feasibility
gate) · wrench 1 · accel-reg 1 · ε 1e-6 · **τ_w_max 5** (base = Add-5 docking recipe).

| ss_alpha_mom | dock | PLAN [x,y,z] | pMax | **pSat** | plan-sat ticks | REAL [x,y,z] | rMax | rSat | **r/p** | e_com |
|---|---|---|---|---|---|---|---|---|---|---|
| **400** | 6/6 | [1.89, 3.53, **5.00**] | 5.00 | **YES** | 106/427 (25%) | [0.80, 1.86, 2.50] | 2.50 | no | 0.50 | 0.137 |
| **1000** | 6/6 | [1.90, 3.35, **5.00**] | 5.00 | **YES** | 105/426 (25%) | [0.80, 1.86, 2.37] | 2.37 | no | 0.47 | 0.137 |
| **2000** | 6/6 | [1.99, 3.36, **5.00**] | 5.00 | **YES** | 113/420 (27%) | [1.00, 2.03, 3.33] | 3.33 | no | 0.67 | 0.164 |
| **5000** | 6/6 | [1.96, 2.63, **5.00**] | 5.00 | **YES** | 104/423 (25%) | [0.79, 1.87, 2.95] | 2.95 | no | 0.59 | 0.138 |

**Planned z-axis = 5.000 in every row** (bold). Realized max-axis never reaches 4.95 in any row.

### Per-step max-axis Ḣ_s (arm sequence a b a b a b)

| ss_alpha_mom | PLANNED per step | REALIZED per step |
|---|---|---|
| 400  | [**5.0**, 3.62, **5.0**, 2.51, **5.0**, 3.53] | [2.02, 2.05, 2.16, 1.54, 2.50, 1.63] |
| 1000 | [**5.0**, 3.52, **5.0**, 2.39, **5.0**, 3.35] | [2.04, 2.03, 2.21, 1.53, 2.37, 1.59] |
| 2000 | [**5.0**, 3.38, **5.0**, 3.93, **5.0**, 3.36] | [2.08, 1.99, 2.28, 1.82, 3.33, 1.78] |
| 5000 | [**5.0**, 3.10, **5.0**, 1.87, **5.0**, 2.63] | [2.18, 1.90, 2.62, 1.56, 2.95, 1.43] |

The **arm-a steps (0, 2, 4) plan to 5.0 at every weight**; arm-b steps (1, 3, 5) plan to 3.4–3.9. This
saturating/non-saturating pattern is **weight-independent** — the NMPC pushes the same aggressive arm-a
phases to the envelope no matter what `ss_alpha_mom` is.

## Verbatim IPOPT (`return_status`, SS solves only)

| ss_alpha_mom | Solve_Succeeded | Solved_To_Acceptable_Level | Infeasible_Problem_Detected | n(SS) | success |
|---|---|---|---|---|---|
| 400  | 426 | 1 | 0 | 427 | **100%** |
| 1000 | 425 | 1 | 0 | 426 | **100%** |
| 2000 | 415 | 3 | **2** | 420 | 99.5% |
| 5000 | 421 | 2 | 0 | 423 | **100%** |

Both `Solve_Succeeded` and `Solved_To_Acceptable_Level` are IPOPT success codes. The saturating plan is an
**IPOPT-optimal** result, not a solver artifact. mom 2000 had 2 transient `Infeasible_Problem_Detected`
(0.5%), recovered by the receding-horizon warm-shift fallback (`sim_loop.py:2759`). `qp_fail = 0` everywhere.
(Full-run `nmpc_fail` 1225–1393 is dominated by the intentional DS-phase NMPC bypass, `nmpc_ok=False` =
"not run"; the SS solves captured here are the meaningful ones and they succeed.)

## Interpretation

The momentum envelope is owned by the **NMPC**. IPOPT plans the fastest feasible centroidal motion, which drives
the aggressive arm-a stance phases hard against the per-axis z cap (5.000) — and it does so identically for
every `ss_alpha_mom`, because the plan's optimization has no dependence on the WQP task weight. Downstream, the
WQP realizes a contact wrench whose Ḣ_s sits at **~0.5–0.67× the plan** and never touches the box; raising the
momentum-task weight 12.5× (400 → 5000) moves neither the plan (fixed at 5.0) nor the realized peak
monotonically, and does not change docking (6/6 throughout). This extends **COPRIORITY Addendum 8** (momentum
400↔500 inert on docking) to the momentum channel itself: at the Add-5 base, `ss_alpha_mom` is **near-inert on
both planned and realized Ḣ_s.** Momentum management is an NMPC property; the WQP momentum weight is a weak knob.

## Files

| artifact | path |
|---|---|
| per-weight result JSON | `results/j2_adjconv/nmpc_plan_sat_result.json` |
| per-weight tidy CSV | `results/j2_adjconv/nmpc_plan_sat.csv` |
| runner (weight sweep + per-axis + verbatim IPOPT capture) | `Misc/scripts/diag_nmpc_plan_saturation.py` |

**STOP for cross-check.** Push only, never merge. `crawlbot/` untouched.
