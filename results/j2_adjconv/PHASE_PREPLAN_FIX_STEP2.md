# Phase PREPLAN-FIX · STEP 2 — apply the fix + revalidate canonical C & U

Applied the surgical one-line fix (pre-planner rate cap → about-origin `Ḣ_s`, matching the NMPC) and
re-ran the canonical C (h_max=5, τ_w_max=5) and U (rate-off, τ_w_max=1e6) on `j2/ds-active-rework`.

## STEP 0 — safety tag
`pre-preplan-fix-canonical` → commit `a9877e6` (pre-fix state) created **locally**. **Tag push blocked
(HTTP 403** — this environment permits branch pushes, not tag refs). Pre-fix state is still durably
recoverable **by hash**: `a9877e6` (pushed on `origin/j2/ds-active-rework`), canonical data commits
`5ab2c91` (C=runfix) / `be76c9c` (U).

## STEP 1 — applied diff (commit `3faae82`)
`crawlbot/planning/coarse_preplanner.py:342` — moment arm `(p_r_C - rk)` → `p_r_C`; `rk` removed:
```diff
-            rk = xk[0:3]
-            L_dot = ca.cross(p_r_C - rk, fk) + tauk
-            opti.subject_to(opti.bounded(-cfg.tau_w_max, L_dot, cfg.tau_w_max))
+            H_dot_s = ca.cross(p_r_C, fk) + tauk   # about-origin, matches NMPC :279
+            opti.subject_to(opti.bounded(-cfg.tau_w_max, H_dot_s, cfg.tau_w_max))
```
The state ODE `:317` (`L_dot = cross(p_r_C - r, f) + tau`, centroidal) is **UNCHANGED** — confirmed.

## STEP 2 — canonical C (fixed) vs pre-fix (runfix @5ab2c91 = figC_qpcond)
### Feasibility FIRST — 6/6 pre-planner solve, 6/6 dock (verbatim T_step, cost)
| step (release) | T_step | cost | iters | IPOPT |
|---|---|---|---|---|
| 0 (b@2) | 2.78 s | 11.205 | 13 | Optimal |
| 1 (a@2) | 8.40 s | 0.721 | 10 | Optimal |
| 2 (b@3) | 2.87 s | **25.065** | 16 | Optimal |
| 3 (a@3) | 7.98 s | 0.783 | 11 | Optimal |
| 4 (b@4) | 3.06 s | **53.022** | 14 | Optimal |
| 5 (a@4) | 6.11 s | 1.307 | 14 | Optimal |

All 6 **Optimal**, 0 skipped. Step 0 is **identical** to pre-fix (cost 11.205 = Phase-1h R1). The
short high-standoff steps (2, 4) show **elevated cost** (25.1, 53.0) — the fix *binds* there (the
about-origin cap now includes the orbital term), exactly as Step-1 predicted. No infeasibility.

### Before/after (closed-loop)
| metric | PRE (runfix) | FIX | Δ |
|---|---|---|---|
| n_ticks | 1080 | 1080 | same |
| T_step per step | [2.78, 8.40, 2.87, 7.98, 3.06, 6.10] | [2.78, 8.40, 2.87, 7.98, 3.06, 6.11] | **identical** |
| **realized \|Ḣ_s\| peak/axis** | [3.358, 5.000, 5.000] | [3.358, 5.000, 5.000] | **identical** (≤5) |
| realized \|Ḣ_s\| SS-swing peak | [1.325, 3.370, 5.000] | [1.321, 3.355, 5.000] | ~identical |
| NMPC planned \|Ḣ_s\| peak | [3.485, 5.000, 5.000] | [3.461, 5.000, 5.000] | ~identical |
| h_w peak/axis [N·m·s] | [0.711, 2.356, 4.885] | [0.710, 2.364, 4.884] | ~identical |
| θ_s peak / settled [deg] | 0.591 / 0.108 | 0.594 / 0.107 | ~identical |
| **docks [mm]** | [4.94, 4.41, 4.88, 4.42, 4.76, **4.92**] | [4.94, 4.41, 4.90, 4.44, 4.76, **5.00**] | steps 2,3,5 shifted |
| **worst dock (precise)** | 4.9246 mm (step5) | **4.9999 mm** (step5) | **margin 0.075 → 0.0001 mm** |

**6/6 dock, all `< 5 mm`** (strict gate `d < weld_radius`, `sim_loop:1322`; the gate fired 6×). All
figure-relevant metrics (realized `Ḣ_s`, h_w, θ_s) are **essentially identical** — the NMPC (already
correct, PREPLAN-ROLE) governs the realized motion, so the pre-planner's internal reshaping barely
propagates.

### ⚠ Step-5 dock margin collapsed to the gate edge
The one material change: **step 5 docks at 4.9999 mm** (fixed) vs 4.9246 mm (pre-fix) — a **0.0001 mm
margin** to the 5 mm gate. It still passes (deterministic, MuJoCo), but the margin is razor-thin. The
fix's reshaped reference on the high-standoff steps (cost 53 at step 4) propagates to step 5's swing
ending ~0.075 mm further out. A tiny perturbation could tip it over. **Flagged for decision.**

## STEP 3 — U (rate-off, τ_w_max=1e6): BIT-IDENTICAL
`figU_preplanfix` vs committed U (`figU_rateoff` = runU@`be76c9c`): realized `|Ḣ_s|` peak
[4.079, 15.987, 31.149], **SS-swing peak 7.48**, h_w [0.713, 2.831, 4.484], θ_s, and docks
[4.94, 4.06, 4.78, 4.47, 4.79, 4.70] **all match exactly**. Confirms the fix is **inactive at 1e6**
(the constraint `|Ḣ_s| ≤ 1e6` never binds), and validates that the run command faithfully reproduces
the canonical (U-fixed ≡ committed U to display precision).

## Verdict
- **Feasibility: PASS** — 6/6 pre-planner Optimal, 6/6 dock `< 5 mm`, both runs.
- **U: unchanged** (bit-identical; fix inactive at 1e6).
- **C: essentially numerically identical on all figures** (realized `Ḣ_s`, h_w, θ_s unchanged; T_step
  identical) — **EXCEPT the step-5 dock margin tightened from 0.075 mm to 0.0001 mm** (4.9246 → 4.9999).
  So C is a *marginal* new baseline: figures don't change, but the dock table shifts and step 5 sits at
  the gate edge.
- The fix is a **correctness/reproducibility improvement** (pre-planner now constrains the same
  about-origin `Ḣ_s` as the NMPC, resolving the CSTR-ORBITAL inconsistency) with near-zero closed-loop
  effect — but it exposes a **step-5 dock fragility** that needs a decision.

## Awaiting Idriss (STOP-GATE)
1. **Accept the step-5 4.9999 mm** as the new canonical (6/6 still pass) → re-export ablation CSVs for
   the new C + tag a new canonical commit? OR
2. Treat the step-5 razor-thin margin as a **blocker** needing a follow-up (e.g., a small T_step / dock
   tolerance / tuning adjustment) before re-baselining?

Runs (`figC_preplanfix`, `figU_preplanfix`) are gitignored raw dumps. No re-export / new-canonical
commit made yet — awaiting the decision on the step-5 fragility.
