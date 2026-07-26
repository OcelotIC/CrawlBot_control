# INTERNAL — chatter FIX (settle-QP regularization) + Σf-artifact + SS orbital scaling

**Applies a fix + characterizes two things.** Branch `j2/ds-active-rework` (pushed, never merged). Touches
`crawlbot/` (the settle QP) — minimal, localized to the inter-step settle, **default-off (byte-identical)**.
Reproducer `scripts/diag_cooperative_arms.py --interstep-settle-alpha-wrench <ε>` + `Misc/scripts/audit_chatter.py`
+ `Misc/scripts/audit_ss_orbital.py`.

## DECISIVE OUTPUT

1. **Task 1 (fix works):** a **settle-only α_wrench boost** (the inter-step QP's wrench-tracking weight)
   **eliminates the chatter** — flip-frac 0.94→0.04, at-±5 0.86→0, commanded **Δτ_w 14.6→0.03 N·m**. Smallest
   clean value **ε ≈ 3** (ε=1 partial, ε=0.1 no effect). Settle still **converges and the dock holds**
   (0.008 mm); C1 docking PASS. Default-off is **byte-identical** (C6, |Δ|=0).
2. **Task 2 (the arm-a envelope demand is a CHATTERING ARTIFACT):** at the unique min-norm vertex the net
   contact force **Σf collapses to ≈0** (6.9→0.09–0.9 N), the **exact ‖Ḣ_s‖ collapses 7.27→0.07–0.71 N·m
   (≪5)**, and the **orbital term 5.99→0.07–0.74**. The "arm-a exceeds the ±5 envelope" was the chattering
   ±6 N phantom Σf, **not a real demand** → **problem 2 dissolves; the gros 4b is NOT needed for it.**
3. **Task 3 (SS orbital scaling — a REAL limit, for the paper):** orbital ‖r_com×Σf‖ ≈ **2.14·|r_com|**
   (slope ≈ mean |Σf|; **|Σf| stays ~3.1 N — the NMPC does NOT gentle the swing as r_com grows**), reaching
   the per-axis ±5 envelope at **|r_com| ≈ 2.34 m**. The SS exact ‖Ḣ_s‖ already binds (5.27) at step 4
   (r_com 1.71 m). This bounds achievable swing dynamics vs crawl distance — motivates the predictive
   approach (connects to Rognant's min-step-time-from-capacity).

---

## Task 1 — the fix (STOP-GATE 1)

**Mechanism.** `cfg.alpha_wrench ≈ 0.01` (the existing `α_wrench‖λ−λ_ref‖²`, with `λ_ref=0` in the settle, is
already a Tikhonov on ‖λ‖) is **below the active-set solver's degeneracy tolerance**, so when the exact
envelope box binds the solver alternates between the two equal-**norm** saturating vertices A≈−B. Raising the
settle-only weight makes the λ-cost **strictly convex above tolerance** ⇒ unique **min-norm** wrench = the
midpoint of [A,B], where Σf≈0. Implemented as `solve(settle_alpha_wrench=…)`, passed **only** from
`_run_ds_passivity_loop` (SS and the `_step` DWELL untouched); `cfg.interstep_settle_alpha_wrench` (default
0.0 = off).

**ε sweep (arm-a settles; the chatterers):**

| ε | step2 flip | at-±5 | exact ‖Ḣ_s‖ | Σf | **Δτ_w** | step4 flip | step4 exact ‖Ḣ_s‖ |
|---|---|---|---|---|---|---|---|
| **0 (baseline)** | 0.939 | 0.86 | 7.274 | 6.90 | **14.57** | 0.980 | 7.112 |
| 0.1 | 0.980 | 0.86 | 7.236 | 6.84 | 14.52 | 0.980 | 7.113 |
| 1 | 0.980 | 0.44 | 5.588 | 5.30 | 11.38 | 0.673 | 4.141 |
| **3** | **0.041** | **0.00** | **0.710** | **0.93** | **0.027** | **0.030** | **0.091** |
| 5 | 0.041 | 0.00 | 0.625 | 0.88 | 0.027 | 0.016 | 0.070 |
| 10 | 0.007 | 0.00 | 0.019 | ≈0 | 0.000 | 0.009 | 0.028 |

ε=0.1 inert; ε=1 partial (amplitude down, sign still flips); **ε=3 is the smallest that breaks the cycle**
(flip 0.94→0.04, Δτ_w 14.6→0.03). The actuator-health win is decisive: the wheels stop slamming ±5.

**Settle convergence + dock hold (the legitimate-dynamics check):**

| ε | inter-step settles (n / exit / conv) | dock hold (max d_grip_stance) |
|---|---|---|
| 0 | settles 1–4 **plateau, conv=False** (T_end 7e-5…2e-3) at n=51–102 | 0.010 mm |
| 3 | s1 172/targ, **s2 51/plateau**, s3 206/targ, s4 205/targ | 0.008 mm |
| 10 | **all target_met** (conv=True, T_end≈2.5e-8), n=344–569 | 0.008 mm |

**Key finding: the chatter was *masking non-convergence*.** At baseline the limit cycle pumps T_kin so every
inter-step settle **plateaus** (conv=False) and bails early; removing it lets the settle reach the target.
At ε=3–5 the settle stays fast (step2 keeps its n=51 plateau, like baseline, but now **chatter-free**); at
ε=10 every settle runs to the tight target (n≈350–570 — correct but slow, the brief's "over-damp" regime).
**The dock holds (≈0.008 mm) and the arms stay seated at every ε** — the regularization does not break the
weld-holding dynamics. **Recommended ε ≈ 3–5** (kills the chatter, preserves the fast settle; ε=10 over-converges).

**C1–C5 (ε=5) + C6:** **C1 docking PASS** (5/5 dock, [4.94,4.46,4.94,4.65,4.92] mm); C3 PASS (SS ‖Ḣ_s‖∞=5.00);
C4 PASS (0.56/0.09). **C2 (22.9 mm) and C5 (4.909) still FAIL — but these are the pre-existing *exact-box SS*
issues** (the ccurr brief: exact box → C2 28 mm, C5 4.949), *slightly improved* by removing the chattering
inter-step Σf (28→22.9 mm, 4.949→4.909) and **orthogonal to the chatter fix** (they live in SS, explained by
Task 3). **C6 BIT-IDENTICAL** — default-off (ε=0) is byte-identical to pre-fix `af2f64a` (|Δ|=0).

## Task 2 — is the arm-a envelope demand a chattering artifact? **YES.**

At the unique min-norm vertex (regularized), in the arm-a settles (steps 2,4 = the post-arm-a-dock settles):

| quantity | baseline (chatter) | ε=3 | ε=10 | verdict |
|---|---|---|---|---|
| net Σf [N] | 6.90 / 6.88 | 0.93 / 0.09 | ≈0 | **collapses to ≈0** |
| exact ‖Ḣ_s‖ [N·m] | 7.27 / 7.11 (>5) | 0.71 / 0.09 | 0.02 / 0.03 | **≪ 5 — envelope NOT exceeded** |
| orbital ‖r_com×Σf‖ | 5.99 / 5.97 | 0.74 / 0.07 | 0.02 / 0.02 | **collapses ~6 → ~0** |

**Verdict: the "arm-a exceeds the envelope" was entirely a chattering artifact.** Physically: the settle holds
the welded arms while dissipating joint KE, so the correct net force is Σf = m·a_com ≈ 0 (no CoM accel); the
chatter injected a phantom ±6 N oscillating Σf, whose orbital lever drove the exact Ḣ_s to 7.27. With the
unique min-norm wrench, Σf≈0 and the exact Ḣ_s collapses to ~0 — **the inter-step DS does NOT demand >5.**
⇒ **the gros 4b (predictive-DS for problem 2) is NOT needed** — regularization dissolves problem 2.

## Task 3 — SS orbital scaling with r_com (the structural limit, for the paper)

Per SS phase (planned wrench `lambda_ref`, real crawl motion):

| SS step | \|r_com\| mean [m] | ‖Σf‖ med [N] | orbital ‖r_com×Σf‖ med | proxy ‖Ḣ_s‖ | exact ‖Ḣ_s‖ |
|---|---|---|---|---|---|
| 0 | 0.685 | 3.33 | 2.17 | 2.45 | 3.49 |
| 1 | 0.807 | 3.28 | 1.88 | 1.15 | 2.09 |
| 2 | 1.083 | 2.70 | 2.51 | 2.18 | 4.41 |
| 3 | 1.377 | 3.12 | 1.71 | 0.95 | 1.75 |
| 4 | 1.705 | 3.29 | **4.15** | 3.36 | **5.27 (binds)** |

- **Law:** orbital ≈ **2.14·|r_com|** (through-origin; the slope ≈ the lever-perpendicular component of Σf);
  free fit orbital ≈ 1.59·|r_com| + 0.69. It **grows ~linearly with crawl distance.**
- **Σf trend:** |Σf| = [3.33, 3.28, 2.70, 3.12, 3.29] N — **roughly constant ~3.1 N; the NMPC does NOT reduce
  the swing net force as r_com grows.** So the orbital term grows purely through the lever |r_com|.
- **Extrapolation:** the orbital term alone reaches the per-axis envelope ±5 N·m at **|r_com| ≈ 2.34 m** (at
  the representative swing |Σf|≈3.1 N). The SS exact ‖Ḣ_s‖ **already binds (5.27 ≥ 5) at step 4 (r_com 1.71
  m)** — i.e. the demonstration is *already* at the SS envelope at its furthest step. Beyond ~2 m crawl the
  orbital lever saturates the wheel momentum-rate envelope regardless of the centroidal (proxy) part.
- **Implication (paper):** achievable swing dynamics are **lever-bounded by crawl distance** — to crawl
  further the NMPC must *gentle the swing* (reduce |Σf|) or the envelope must be re-sized. This is a genuine
  physical scaling limit (distinct from the dissolved inter-step artifact) and motivates a predictive /
  capacity-aware swing planner (cf. Rognant min-step-time-from-capacity).

---

## Recommendation
- **Keep the fix; set the production weight ε ≈ 3–5** (kills the chatter, preserves the fast settle, dock
  holds). ε=10 over-converges (correct but slow). Enabling it as the default needs a C2/C5 re-gate — but
  **C2/C5 are SEPARATE exact-box SS issues, not the chatter** (the fix slightly improves them).
- **Problem 2 (arm-a inter-step envelope) is dissolved** by this regularization (Task 2) — **no gros 4b
  needed** for it.
- **The SS orbital scaling (Task 3) is a real, separate limit** — a predictive/swing-gentling approach is
  motivated for crawls beyond ~2 m, independent of problem 2.

## Regression
**220 passed, 1 failed** (new code at `01bfc27`) — the single failure is the **pre-existing** FK test
`test_E7_t15_step2_dock_under_fk_mode` (known/unrelated); **no new failures**. Default-off is **byte-identical**
(C6 |Δ|=0; ε=0 reproduces the diagnosis baseline exactly), so the dormant fix cannot affect the default path.

## Reproduce
```
# ε sweep (chatter + Σf/exact/orbital):
for e in 0 0.1 1 3 5 10; do diag_cooperative_arms.py <canonical exact-box> --interstep-settle-alpha-wrench $e --out-dir chat_e$e; done
audit_chatter.py base=results/chat_e0 e3=results/chat_e3 ...    # chatter metrics
audit_ss_orbital.py results/chat_e0                              # SS scaling law
```
Supporting: `Misc/runs/j2_chatter_fix/{chat_sweep_analysis.log}`. Raw run dirs reproducible, not committed.
**No merge, no PR.**
