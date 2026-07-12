# Phase STEP5-MARGIN — restore step-5 dock margin via a calibrated T_step adjustment

After PREPLAN-FIX (`e0407a0`), step-5 docks at 4.9999 mm (0.0001 mm margin). Goal: lengthen ONLY
step-5's T_step so it docks with ≥0.1 mm margin, no other step affected. Branch `j2/ds-active-rework`.

## STEP 0 — T_step control (mechanism)
`T_step` is set **per-step** by `sim_loop.py:1890`: `T_step_guess = max(0.5, distance/v_max)` with
`v_max = min|h_max|/(m·lever)`, `lever=1.0` nominal — echoed by the pre-planner (no search). The
existing `t_ss_margin` (`config.py:26`) is only a **timeout buffer** (`t_ss_deadline = ... + T_step +
t_ss_margin`), not the swing duration, so it's not the lever. No per-step override existed.

**Mechanism added (default OFF):** a standoff-KNEE dock-margin factor —
`if |r_com_0| > knee: T_step *= (1 + gain)` (`config.py preplanner_tstep_standoff_{gain,knee}`,
`sim_loop.py:1891`, CLI `--preplanner-tstep-standoff-{gain,knee}`). Keyed to standoff (principled: the
extended high-standoff swing tracks less precisely, needs more time) with the **knee set ABOVE the
other steps' standoff** so only the highest-standoff step changes and all others stay bit-identical.

**Why not a global standoff factor:** a first attempt (`T_step *= 1+gain·|r_com_0|` on *all* steps,
gain=0.03) improved step-5 (4.9999→4.5954) **but regressed step 4 (4.76→4.97)** via cross-step
coupling. Rejected — the knee isolates step 5 (steps 0-4 bit-identical) and avoids this.

## STEP 1 — calibration (per-step standoffs; knee=1.7 isolates step 5)
| step | stance | \|r_com_0\| | T_step | dock (e0407a0) |
|---|---|---|---|---|
| 0-4 | b/a/b/a/b | 0.71–1.63 | — | 4.94/4.41/4.90/4.44/4.76 |
| **5** | a | **1.812** | 6.112 | **4.9999** |

`knee = 1.7` (between step-4's 1.63 and step-5's 1.81) → only step 5 triggers. Measured `d(gain)`
(isolated, all steps 0-4 bit-identical in every run):

| gain | step-5 T_step | step-5 dock | margin |
|---|---|---|---|
| 0 | 6.112 | 4.9999 | 0.0001 |
| 0.022 | 6.247 | 4.9288 | 0.0712 |
| 0.08 | 6.601 | 4.2854 | 0.7146 |

**The relationship is CONVEX** (local slope ≈ −3.2 mm/gain near 0, ≈ −11 over [0.022, 0.08]). A linear
extrapolation from the single gain=0.08 point (avg slope −8.9) **under-estimates** the gain needed.

## STEP 2 — single application (gain=0.022) + revalidation
| metric | PRE (e0407a0) | STEP5FIX (gain=0.022) |
|---|---|---|
| T_step/step | [2.775, 8.399, 2.868, 7.98, 3.056, 6.112] | [2.775, 8.399, 2.868, 7.98, 3.056, **6.247**] |
| docks [mm] | [4.94,4.41,4.90,4.44,4.76,**5.00**] | [4.94,4.41,4.90,4.44,4.76,**4.93**] |
| step-5 precise / margin | 4.9999 / **0.0001** | 4.9288 / **0.0712** |
| realized \|Ḣ_s\| peak | [3.358,5.0,5.0] | [3.358,5.0,5.0] |
| h_w peak | [0.71,2.364,4.884] | [0.71,2.312,4.884] |
| θ_s peak / settled | 0.594 / 0.107 | 0.589 / 0.106 |

- **Isolation confirmed:** steps 0-4 T_step and docks **bit-identical**; realized Ḣ_s, h_w, θ_s
  essentially identical. **No other step affected — no regression.** Only step-5 T_step changed
  (6.112 → 6.247).
- **BUT the target is MISSED:** step-5 margin = **0.0712 mm < 0.1 mm** (dock 4.9288 > 4.9). The
  gain=0.022 (my linear estimate for ~4.8 mm) under-shot because of the convex nonlinearity.

## STOP (per instruction — do NOT re-tune to force it)
Per *"If step-5 still <0.1 mm margin … STOP and report — do NOT re-tune"*: I stopped. **The mechanism
is validated and clean** (isolates step 5, no regression); only the **gain value needs finalizing**,
and I now have 3 points to compute it properly instead of guessing:
- **gain ≈ 0.034 → step-5 ~4.80 mm (≈0.20 mm margin)** — squarely comparable to the other steps
  (interpolated between the 0.022 and 0.08 measurements). *(Not yet run — would be one more application.)*
- **gain = 0.08 → step-5 4.2854 mm (0.71 mm margin)** — already measured, isolated, no regression, but
  *more* margin than any other step (over-corrected).

## Awaiting Idriss (STOP-GATE)
Pick the step-5 gain (knee=1.7 fixed):
1. **gain ≈ 0.034** → ~4.8 mm, 0.2 mm margin (comparable to the pack) — I run it once and verify; or
2. **gain = 0.08** → 4.29 mm, 0.71 mm margin (already measured, safe but over-margined) — accept as-is; or
3. reassess the approach.

The mechanism (default off) is committed; canonical behavior is unchanged until a gain is set. Raw
runs (`figC_step5{cal,iso,fix}`) are gitignored.
