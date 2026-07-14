# Phase TSTEP-DIAG-ALL — dock error vs T_step for all 6 steps

Measurement-only sweep: each step's T_step scaled ×[1.0, 1.1, 1.25, 1.5, 2.0, 2.5] (others at canonical
formula value), 36 runs, all feasible. NO canonical change (default gain/scale stay off). Mechanism:
per-step scale `if step_idx==scale_step: T_step *= scale_factor` (`config.py`/`sim_loop.py`, CLI
`--preplanner-tstep-scale-{step,factor}`), with `--n-steps k+1` so sweeping step k only simulates
0..k (step k terminal ⇒ its dock is identical to the full run; steps 0..k−1 bit-identical). Data:
`results/j2_adjconv/tstep_sweep.json`; scripts `diag_tstep_sweep_{run,extract}.py`.

## Full table (step × scale): T_step [s], dock [mm], standoff [m], distance [m], realized \|Ḣ_s\|pk, feasible
| step (swing) | metric | ×1.0 | ×1.1 | ×1.25 | ×1.5 | ×2.0 | ×2.5 |
|---|---|---|---|---|---|---|---|
| 0 (b) | T_step / **dock** / Ḣ_s | 2.78/**4.94**/4.82 | 3.05/**4.87**/4.10 | 3.47/**4.77**/3.31 | 4.16/**3.79**/2.40 | 5.55/**2.37**/1.39 | 6.94/**1.53**/0.90 |
| 1 (a) | | 8.40/**4.41**/1.92 | 9.24/**3.41**/1.60 | 10.50/**2.72**/1.24 | 12.60/**1.79**/0.87 | 16.80/**0.90**/0.51 | 21.00/**0.51**/0.34 |
| 2 (b) | | 2.87/**4.90**/**5.00** | 3.16/**4.98**/**5.00** | 3.59/**4.72**/4.38 | 4.30/**3.48**/3.30 | 5.74/**2.18**/1.90 | 7.17/**1.45**/1.22 |
| 3 (a) | | 7.98/**4.44**/1.94 | 8.78/**3.63**/1.61 | 9.98/**2.75**/1.25 | 11.97/**1.82**/0.89 | 15.96/**0.91**/0.52 | 19.95/**0.52**/0.36 |
| 4 (b) | | 3.06/**4.05**/**5.00** | 3.36/**4.87**/**5.00** | 3.82/**4.59**/4.91 | 4.58/**3.71**/3.53 | 6.11/**2.07**/2.13 | 7.64/**1.32**/1.35 |
| 5 (a) | | 6.11/**5.00**/2.44 | 6.72/**4.22**/2.02 | 7.64/**3.34**/1.57 | 9.17/**2.34**/1.10 | 12.23/**1.18**/0.64 | 15.28/**0.73**/0.42 |

standoff (fixed per step): s0 0.71, s1 0.68, s2 1.01, s3 1.13, s4 1.63, s5 1.81 · distance: s0/2/4 ≈0.20–0.22, s1 0.59, s3 0.56, s5 0.43. **All 36 runs feasible (scaling T_step up is always feasible).**

## Q1 — CAUSE: dock error → 0 (no nonzero floor), all 6 steps
Every step's dock **decreases monotonically** with T_step and is **still decreasing at ×2.5**
(docks 0.51–1.53 mm, all < baseline). No plateau. ⇒ **time/precision tradeoff** — a longer swing
tracks the terminal approach more precisely. Not a tracking-bias floor. **T_step should be DERIVED for
adequate dock precision** (more time buys margin); more time WILL close the gap on any step.

Corroboration via Ḣ_s: as T_step grows the realized \|Ḣ_s\|pk **falls** (slower swing ⇒ less
momentum rate), e.g. step 5 2.44→0.42. So the precision gain is bought by relaxing the momentum
demand.

## Q2 — UNIFORMITY: NO single universal law; per-step idiosyncrasy
- **The formula already imposes a "natural" normalization and it does NOT collapse the curves.**
  `T_step = distance/v_max` ⇒ `T_step/distance = 1/v_max = 14.2 s/m` for every step at ×1.0 (and
  =14.2·scale at any scale). So all 6 steps sit at the **same normalised time** `T_step/distance` at
  each scale — yet their docks **differ by 0.95–2.06 mm** (per-scale spread). ⇒ **no collapse** under
  `T_step/distance`.
- **Baseline dock correlates with nothing physical:** corr(dock, standoff) = **−0.11**,
  corr(dock, distance) = −0.21, corr(dock, T_step) = −0.21. The hypothesis "higher standoff → worse
  dock at the formula T_step" is **REJECTED** — step 4 (standoff 1.63) docks **best** (4.05), step 5
  (1.81) **worst** (5.00); step 1 (0.68) good (4.41), step 0 (0.71) bad (4.94).
- **A swing-arm split reduces the within-group spread at large scale** (×2.5: swing-A steps 1,3,5
  spread 0.22 mm, swing-B 0,2,4 spread 0.22 mm, A ~0.85 mm better) — BUT this is **confounded with
  absolute T_step** (swing-A steps have the long distances ⇒ 2–3× longer T_step at every scale). A
  same-absolute-T check breaks it: step 0 (swing-B) at T≈6.9 s docks **1.53**, step 5 (swing-A) at
  T≈6.7 s docks **4.22** — swing-B better at equal T. So neither swing arm nor absolute T_step
  collapses the 6 curves.
- **Ḣ_s constraint is NOT the discriminator:** it is saturated (\|Ḣ_s\|=5.00) only on the short steps
  (2, 4; step 0 near at 4.82), i.e. those are genuinely momentum-limited at the formula T_step — yet
  they are not the worst dockers. Step 5 (worst dock, 5.00) has Ḣ_s **headroom** (2.44) — its tightness
  is a **tracking** residual, not the momentum cap.

**Verdict:** each step has its **own** dock(T_step) curve; **no single physically-motivated
normalisation** (standoff, distance, T_step/distance, absolute T_step, or swing arm) collapses all six
onto one master curve. The law is **per-step idiosyncratic**, all sharing the same *qualitative* shape
(monotone → 0). The formula's uniform `T_step/distance` does **not** buy uniform dock precision.

## Implication for the STEP5-MARGIN approach
- Step-5's tightness is **not** a standoff law (standoff↔dock corr −0.11) — it is a **specific baseline
  outlier** (worst at ×1.0 despite Ḣ_s headroom; reverts into the swing-A cluster as T_step grows).
- Since dock → 0 with T_step for every step and correlates with nothing external, a **standoff-keyed**
  factor (STEP5-MARGIN) is **not** the principled lever — it happened to lift step 5 but is not a law.
  A defensible fix is either **per-step** T_step set for a target dock precision, or a **uniform T_step
  margin / minimum-swing-time floor** applied to all steps (each step's dock then improves), rather
  than a standoff formula. Precision is a T_step tradeoff, decoupled from standoff and from the Ḣ_s cap.

## Deliverable
Full (step×scale) table above + `tstep_sweep.json`; Q1 → 0 (no floor, all steps); Q2 no single-law
collapse (per-step idiosyncrasy; baseline dock uncorrelated with standoff/distance/T_step; Ḣ_s
saturated only on short steps). NO canonical change; default scale/gain remain off. Raw runs
(`figC_sw_*`) gitignored.
