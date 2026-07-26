# INTERNAL — Is the baseline's low h_w a consequence of poor torso positioning?

**Status: reviewer-rebuttal RESERVE. NOT paper content.** The paper reports h_w as a
self-contained result (peak value, % of ±5 N·m·s budget, within limit throughout) with no
δ-mapping comparison and no C5-style criterion. This note justifies our confidence in the 4.37
figure internally. Read-only diagnostic on existing runs (`scripts/diag_hw_vs_torso_motion.py`,
figure `Misc/runs/phase3_wp/phase3_plots/8_hw_vs_torso_motion.png`,
data `Misc/runs/phase3_wp/hw_vs_torso_motion.json`). Branch HEAD `9939503`.

## Hypothesis
The baseline (5bca42c, δ-mapping) has lower h_w peak (3.38) than the two-task working point
(4.37) **because it positions the torso poorly** — a torso that tracks its target less accurately
moves less → less reaction → less stored wheel momentum. If true, the baseline's low h_w is an
artifact of under-performance, and the gate criterion "h_w ≤ baseline" is ill-founded.

## Verdict: **NOT CONFIRMED** (the *specific* motion mechanism is refuted; the broader "low h_w =
artifact of under-performance" claim survives via a *different*, verified mechanism — §3).

## 1. The hypothesis's premise is FALSE — the baseline moves the torso MORE, not less

Per-swing torso motion (world frame **and** structure-relative, i.e. crawl-removed), binding
swings 2 & 4 where the C5 miss lives:

| swing | torso path world [mm] | torso path REL [mm] | h_w peak | torso arrival err [mm] |
|---|---|---|---|---|
| baseline step 2 (BIND) | 687.6 | 686.7 | 2.16 | **313.4** |
| two-task step 2 (BIND) | 143.5 | 145.9 | 3.77 | 7.8 |
| baseline step 4 (BIND) | 397.2 | 409.2 | 3.38 | 102.0 |
| two-task step 4 (BIND) | 153.5 | 158.7 | 4.37 | 5.4 |

The baseline moves the torso **3-5× MORE** (path), in both frames, yet has **LOWER** h_w. Its
poor positioning shows as a long, wandering path that ends far from p_t1 (arrival 52-313 mm vs
the two-task's 2.7-10.2 mm) — poor positioning means *more* (inefficient) motion here, not less.
The premise "poor positioning → less motion" is contradicted by the data (figure, middle panel).

## 2. h_w does NOT track torso motion — excess ~100% unexplained by motion

Pooled correlation of h_w peak vs every torso-motion measure across the 10 swings is ≈ 0:

| measure | path (world) | path (rel) | netdisp (rel) | peak speed (rel) | ang-path | peak ω | peak acc |
|---|---|---|---|---|---|---|---|
| R | −0.07 | −0.07 | −0.03 | 0.00 | −0.13 | −0.17 | −0.21 |
| R² | 0.005 | 0.005 | 0.001 | 0.000 | 0.018 | 0.029 | **0.044** |

Best correlate (peak_acc) has R²=0.04 and the wrong sign. Fitting the baseline's own h_w-vs-motion
trend and predicting the two-task swings: on the **C5-limiting swing (step 4)** the trend predicts
h_w=2.27 vs actual **4.37** — it explains *none* of the excess over baseline's 3.38 (unexplained
fraction **+212%**; step 2 **+93%**). **Torso motion explains essentially 0% of the h_w excess.**

## 3. The cause, sought elsewhere (per brief): the h_w ↔ θ_s disturbance-budget tradeoff

The disturbance momentum is absorbed by **either** the reaction wheels (h_w) **or** platform
attitude tilt (θ_s). Matched-swing, two-task vs baseline, on the swings with non-negligible
attitude (2,3,4):

| swing | h_w base→two-task | θ_s base→two-task [deg] | tradeoff |
|---|---|---|---|
| step 2 | 2.16 → 3.77 (+1.61) | 1.09 → 0.47 (−0.62) | ↑h_w, ↓θ_s |
| step 3 | 1.81 → 2.93 (+1.12) | 1.65 → 0.55 (−1.10) | ↑h_w, ↓θ_s |
| step 4 (C5-limiting) | 3.38 → 4.37 (+0.99) | **1.88 → 0.62 (−1.26)** | ↑h_w, ↓θ_s |

On the C5-limiting swing the two-task spends **+0.99 N·m·s** of wheel momentum to hold the platform
**3.0× tighter** (θ_s 1.88°→0.62°). The baseline's lower h_w is bought by letting the platform
**tilt more** — it offloads the disturbance into attitude instead of storing it in the wheels.
(Steps 0,1 have θ_s < 0.25° — almost nothing to trade — so the tradeoff is visible on 3/5 swings,
exactly the high-disturbance ones. Pooled R(h_w,θ_s)=+0.08 is uninformative because both rise with
per-swing disturbance *within* a run; the *cross-run, matched-swing* comparison is the test.)

## 4. Implication for the C5 criterion (decision deferred to review)
The broader thesis **survives**: the baseline's low h_w *is* an artifact of under-performance — but
the under-performance is in **attitude regulation** (it lets θ_s reach 1.88° vs the two-task's
0.62°, gate criterion C4 where the two-task wins decisively), not torso positioning. The criterion
"h_w ≤ baseline" therefore penalizes the controller that regulates attitude *better*: lower h_w is
achievable only by spending attitude. Both runs keep h_w within the ±5 N·m·s hardware budget
(4.37 = 87%). **No criterion redefinition is made here** — that follows the review of this verdict.
For the paper, h_w stands alone (peak, % of budget, within limit); this mechanism is rebuttal
reserve only.
