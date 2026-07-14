# COPRIORITY Weight Sweep — Complete Reference

**Branch** `j2/ds-active-rework` · **measurement only, `crawlbot/` untouched, NO canonical commit** · pushed, never merged.

Consolidated record of the interactive SS-docking-QP weight-tuning campaign (9 runs). This is the standalone
reference; the blow-by-blow is in `PHASE_COPRIORITY_1000.md` (Addenda 1–8). All numbers are read from the 9
per-run summary JSONs (`results/j2_adjconv/copri*result.json`); the tidy machine-readable table is
`results/j2_adjconv/copri_sweep_complete.csv` (built by `scripts/diag_copri_sweep_export.py`).

**Stack** (canonical SS `_two_task`, `wholebody_qp.py:678–722`): momentum P2, torso-pose P2, swing-EE P2,
posture P3, wrench-track P4, torque-min P5, accel-reg P6, hw-slack P1. Solver qpOASES, `method='weighted'`,
**`weight_ratio = 1`** ⇒ the nominal priority integers are **inert** — the α **magnitudes** *are* the hierarchy.
Fixed across all runs: posture 20, wrench 1, accel-reg 1 (regularizer floor), ε (Tikhonov) 1e-6, envelope
τ_w_max 5. Dock gate: d < 5 mm **AND** ori < 5° **AND** ‖J_c·v‖ < 0.05.

---

## 1. Complete weight vectors (all 9 runs)

| # | Add | torso | EE | momentum | hw-slack | posture | torque | wrench | accel-reg | ε | span |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A1 | 1000 | 1000 | 1000 | 10000 | 20 | 1 | 1 | 1 | 1e-6 | 1e4 |
| 2 | A1-hunt | 1000 | 1000 | **2000** | 10000 | 20 | 1 | 1 | 1 | 1e-6 | 1e4 |
| 3 | A1-hunt | **2000** | 1000 | 1000 | 10000 | 20 | 1 | 1 | 1 | 1e-6 | 1e4 |
| 4 | A1-hunt | 2000 | 1000 | **400** | 10000 | 20 | 1 | 1 | 1 | 1e-6 | 1e4 |
| 5 | A4-hunt | 2000 | 1000 | 400 | **800** | 20 | 1 | 1 | 1 | 1e-6 | 800 |
| 6 | **A5** | 2000 | 1000 | 400 | 800 | 20 | **5** | 1 | 1 | 1e-6 | 800 |
| 7 | A6 | **300** | **300** | 500 | 1000 | 20 | 5 | 1 | 1 | 1e-6 | 1000 |
| 8 | A7 | 300 | **1000** | 500 | 1000 | 20 | 5 | 1 | 1 | 1e-6 | 1000 |
| 9 | A8 | 300 | 1000 | **400** | 1000 | 20 | 5 | 1 | 1 | 1e-6 | 1000 |

Bold marks the single knob changed from the run above (the campaign is a one-variable-at-a-time chain).

---

## 2. Complete results (all 9 runs)

| # | Add | feasible | at-weld docks [mm] | worst / margin | Ḣ_s pk | sat | e_com | θ_s pk / settled | κ_SS | h_w pk | qp / nmpc fail |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A1 | 0/6 | — (timeout step 0, min d 6.87) | — | 2.31 | —¹ | 0.103¹ | 0.167 / 0.068¹ | 1.00e4 | 2.31 | 0 / 10 |
| 2 | A1-hunt | 0/6 | — (step 0, 6.87) | — | 2.33 | —¹ | 0.103¹ | 0.167 / 0.068¹ | 1.00e4 | 2.30 | 0 / 10 |
| 3 | A1-hunt | 0/6 | — (step 0, 6.87) | — | 2.51 | —¹ | 0.100¹ | 0.168 / 0.068¹ | 1.00e4 | 2.46 | 0 / 10 |
| 4 | A1-hunt | 0/6 | — (step 0, 6.87) | — | 2.50 | —¹ | 0.100¹ | 0.168 / 0.068¹ | 1.00e4 | 2.47 | 0 / 10 |
| 5 | A4-hunt | 0/6 | — (step 0, 6.87) | — | 2.50 | —¹ | 0.100¹ | 0.168 / 0.068¹ | 6.78e3 | 2.47 | 0 / 10 |
| 6 | **A5** | **6/6** | 2.56 4.59 4.89 4.39 2.49 4.49 | **4.89 / 0.11** | 2.50 | GONE | 0.137 | 0.432 / 0.424 | 7.61e3 | 4.12 | 0 / 1232 |
| 7 | A6 | 2/6 | 4.59 4.59 — (timeout step 2, min d 8.50) | — | 2.38 | GONE | 0.154 | 0.472 / 0.176 | **2.11e3** | 3.31 | 0 / 529 |
| 8 | A7 | 4/6 | 3.50 4.73 4.92 4.97 — (timeout step 4, 5.02) | — | 3.61 | PARTIAL | 0.164 | 0.423 / 0.083 | 6.33e3 | 3.41 | 0 / 1099 |
| 9 | A8 | 4/6 | 3.52 4.75 4.93 4.40 — (timeout step 4, 5.02) | — | 3.60 | PARTIAL | 0.164 | 0.424 / 0.084 | 6.31e3 | 3.41 | 0 / 1100 |

¹ Runs 1–5 abort at **step 0** (torque = 1); their saturation / e_com / θ_s-settled are **at-abort** (step-0
only, one SS segment), **not** meaningful traversal values. Ḣ_s and h_w likewise reflect step 0 alone.
`nmpc_fail` is dominated by the intentional DS-phase NMPC bypass (`sim_loop.py:1114`, "not run ≠ failure"),
pre-existing; `qp_fail = 0` everywhere. Reference: canonical κ ≈ 3.6e6, canonical e_com 0.095.

---

## 3. What each run isolated (the lever chain)

- **Runs 1–5 — every knob fails identically at torque = 1.** 1:1:1 base, momentum↑ (2000), torso↑ (2000),
  momentum↓ (400), hw-slack↓ (800): **all time out at step 0 with the same min d = 6.87 mm.** Raising the
  regularizer floor 0.01 → 1 (for conditioning, κ 1e4) dropped torque:floor to **1:1**, so the joint-torque
  regularizer lost dominance over accel-reg and redundancy resolution degraded. Neither torso, momentum, nor
  hw-slack moves the dock while torque:floor = 1:1.
- **Run 6 — torque 1 → 5 docks 6/6.** The single weight that flips timeout → dock is the lowest-tier
  **`alpha_torque`**. **Necessary gate: torque ≳ 5× the regularizer floor.**
- **Run 7 — EE 1000 → 300 breaks it grossly** (2/6, step 2 stalls at **8.50 mm**). EE is the **gross reach
  lever**; below ~1000 the swing can't close. (Matches Phase DOCK-CAUSE: dock = 100 % WBC EE-tracking residual.)
- **Run 8 — EE restored to 1000 recovers most of it** (4/6; step 2 goes 8.50 mm → 4.92 mm dock). Failure
  collapses to a **0.02 mm near-miss at step 4** (5.02 mm).
- **Run 9 — momentum 500 → 400 changes nothing** (still 4/6, step 4 still 5.02 mm, Ḣ_s 3.61 → 3.60).
  **Momentum weight is inert in this regime.** With momentum ruled out and EE fixed at 1000, the residual dock
  lever is **torso** (run 6 torso 2000 = 6/6 vs run 9 torso 300 = 4/6, everything else effectively equal).

---

## 4. The dock-lever hierarchy (what actually controls docking)

1. **torque ≳ 5× accel-reg floor** — *necessary*. Below it, step-0 timeout regardless of everything else. [5→6]
2. **EE ≳ 1000** — *gross reach*. EE 300 stalls at 8.5 mm. [7→8]
3. **torso** — *fine dock lever at EE 1000*. 6/6 floor sits in **(300, 2000]**: torso 2000 docks 6/6, torso 300
   misses step 4 by 0.02 mm. [6 / 9]
4. **momentum (400↔2000)** and **hw-slack (800↔10000)** — **INERT** for docking (they move κ and SS-saturation,
   not the dock). [1–5, 8–9]
5. **ε (1e-6)** — inert: λ_min(H_LS) = 1 ≫ ε, so H is unchanged (REG-DIAG).

---

## 5. Mechanism (defensible, from the data)

- **Dock is EE-residual-limited** (DOCK-CAUSE): the swing-EE task at P2 with no null-space and no terminal
  tightening leaves a residual that *is* the dock error ⇒ EE weight is the primary lever (runs 7↔8).
- **torso — transient vs settled.** torso 300 gives a *better settled* attitude (θ_s 0.084) than torso 2000
  (0.424) yet **docks worse**. A strong torso task damps the **base/momentum transient during the swing** (holds
  Ḣ_s ≤ 2.5, e_com 0.137), which is what lets the EE close the last mm. A weak torso still levels out at settle
  but the base wanders mid-swing (Ḣ_s 3.6, e_com 0.164) — and the dock gate is set by the **swing transient**,
  not the settled pose. Torso buys dock precision invisibly through the transient.
- **Momentum weight ⊥ saturation here.** mom 400 ↔ 500 leaves realized Ḣ_s at 3.60–3.61 (runs 8↔9) — the
  partial saturation at torso 300 is driven by the **low torso**, not the momentum weight. (Contrast the USERW2
  regime at torso 2000 / momentum 5000, where momentum *was* the saturation lever — the lever depends on the
  operating point.)

---

## 6. Standing-best recipe (run 6 / Addendum 5) — the only feasible + well-conditioned vector

`torso 2000 · EE 1000 · momentum 400 · hw-slack 800 · posture 20 · torque 5 · wrench 1 · accel-reg 1 · ε 1e-6`

**Docks 6/6** (at-weld worst **4.89 mm**, margin 0.11), **κ_SS 7.6e3 — 530× below canonical 3.6e6**,
SS-saturation gone, θ_s settled 0.42, e_com 0.137, h_w 4.12 (< ±5), qp_fail 0. Reproduces the committed userw2
run to 0.01 mm (ε confirmed inert).

---

## 7. Attribution-error log (honest)

The lever was **asserted wrong four times** before disciplined one-variable elimination found it:
torso, momentum, and hw-slack were each confidently blamed during the hunt (runs 2–5, all refuted — identical
6.87 mm timeout), and momentum was blamed again in Addendum 7 (refuted by Addendum 8). Only single-variable
runs settled it. **Lesson banked: test each weight, don't assert.**

---

## 8. Open / next test

Bisect the torso floor at EE 1000: **torso 1000 / EE 1000 / momentum 400 / hw-slack 800 · torque 5 · ε 1e-6.**
If 6/6 → floor ≤ 1000 (a lighter recipe than Add-5's 2000) and push lower; if it misses → floor in (1000, 2000].
Everything else is pinned inert.

---

## Files

| artifact | path |
|---|---|
| complete sweep CSV (this campaign, one row / run) | `results/j2_adjconv/copri_sweep_complete.csv` |
| CSV builder | `scripts/diag_copri_sweep_export.py` |
| per-run summary JSONs (9) | `results/j2_adjconv/copri*result.json` |
| blow-by-blow (Addenda 1–8) | `results/j2_adjconv/PHASE_COPRIORITY_1000.md` |
| runner (weight monkeypatch + κ capture) | `scripts/diag_copriority1000_run.py` |
| per-tick tidy CSV — *docking* userw2 (≈ Add-5) | `results/j2_adjconv/userw2_timeseries.csv`, `userw2_fulldiag.csv` |

Raw per-tick sim_logs for the copri runs live in gitignored `results/figC_copri*/`; a tidy per-tick CSV for any
single config can be regenerated on request via `scripts/export_figure_data.py`.
