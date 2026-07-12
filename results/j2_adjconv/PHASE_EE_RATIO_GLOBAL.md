# Phase EE-RATIO-GLOBAL — the global uniform torso:EE ratio is INSUFFICIENT to close the dock target

**Branch** `j2/ds-active-rework` · measurement sweep, NO canonical commit · pushed, never merged.
Data: `results/j2_adjconv/ratio_sweep.json`; scripts `scripts/diag_ratio_sweep_{run,extract}.py`
(each config = a full canonical C run via existing CLI flags `--alpha-torso-pose`/`--ss-alpha-ee`; no
`crawlbot/` change). Baseline (8:1) reused from `results/figC_sw_s5_x1`. θ_s = ‖struct_euler_deg‖
(validated: baseline peak 0.5936 / settled 0.1071 = canonical 0.594/0.107).

**Target:** all 6 SS-dock closest-approach < 5 mm with margin ≥ 0.2 mm (worst ≲ 4.8 mm), WITHOUT degrading
θ_s (peak < ~0.6°) or realized |Ḣ_s| (canonical: steps 2,4 saturate at 5.0). **Result: NO global ratio
meets it. STOP per the phase's own stop condition.**

---

## Sweep table — EE=3000 & momentum=5000 held (route A) / torso=24000 held (route B)

### Viable configs (traversal completes all 6 docks)
| config | torso:EE | 6 SS-docks [mm] (s0…s5) | worst | margin | θ_s pk | θ_s settled | torso_ori resid pk [°] | h_w pk | 6/6 |
|---|---|---|---|---|---|---|---|---|---|
| **baseline** (t24000/ee3000) | 8.0 | 4.940 4.405 4.904 4.436 4.045 5.000 | 5.000 | 0.000 | 0.594 | 0.107 | 0.509 | 5.44 | ✓ |
| A t12000 | 4.0 | 4.984 4.231 4.979 4.468 4.035 4.640 | 4.984 | 0.016 | **0.618** | 0.101 | **0.892** | 5.13 | ✓ |
| **B ee6000** | 4.0 | 4.835 4.211 **4.904** 4.074 4.750 4.860 | **4.904** | **0.097** | 0.600 | 0.091 | 0.539 | 5.41 | ✓ |
| B ee9000 | 2.67 | 4.916 4.136 4.974 4.318 4.787 4.702 | 4.974 | 0.026 | 0.606 | 0.092 | 0.503 | 5.42 | ✓ |

### Failed configs (route A below 4:1 — step-0 TIMEOUT, `[stop_on_failed_step]` aborts)
| config | torso:EE | step-0 min d | outcome |
|---|---|---|---|
| A t8000 | 2.67 | 5.3–7.0 mm | **TIMEOUT step 0, traversal aborted** (0 docks) |
| A t6000 | 2.0 | 5.5–7 mm | TIMEOUT step 0, aborted |
| A t4000 | 1.33 | 5.8–7 mm | TIMEOUT step 0, aborted |
| A t3000 | 1.0 | 6.0–7 mm | TIMEOUT step 0, aborted |

⚠ The failed configs' θ_s ≈ 0.16 and realized Ḣ_s ≈ 0 (steps 1–5) are **early-abort artifacts** (the
traversal ran only a partial step 0), NOT improvements — do not read them as low-collateral.

### Realized |Ḣ_s|pk per step (momentum-management KPI; must not shift materially)
| config | s0 | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|
| baseline 8:1 | 4.82 | 1.92 | **5.00** | 1.94 | **5.00** | 2.44 |
| A t12000 4:1 | 4.29 | 1.92 | **5.00** | 1.94 | **5.00** | 2.54 |
| B ee6000 4:1 | 4.84 | 1.93 | **5.00** | 1.95 | **5.00** | 2.34 |
| B ee9000 2.67:1 | 4.85 | 1.93 | **5.00** | 1.96 | **5.00** | 2.30 |

Across all **viable** configs the saturating steps (2,4) stay pinned at 5.00 and no step exceeds 5.0 —
**the momentum cap / management result survives** (shifts ≤ 0.5 on the non-saturating steps).

---

## Findings

**1. No global ratio hits the target.** Best worst-dock among viable configs = **route B ee6000, 4.904 mm
(margin 0.097 mm)** — better than baseline (5.000) but short of the ≲4.8 mm / ≥0.2 mm margin target. Nothing
reaches 4.8 mm.

**2. Route A (lower torso-pose) is NON-VIABLE.** Even the mildest change (4:1, torso 12000) gives worst
4.984 (margin 0.016, barely docks), **degrades θ_s to 0.618 (>0.6)**, and inflates the torso-orientation
residual +75 % (0.509→0.892). Below 4:1 (torso ≤ 8000) the torso is too loose to hold attitude, **step 0
TIMES OUT (min d ≈ 7 mm) and the traversal aborts** — lowering torso paradoxically **breaks** docking. The
dock-vs-ratio curve is **non-monotone**: 8:1→4:1 helps a hair, then collapses.

**3. Route B (raise EE) is the better route but plateaus short.** ee6000 (4:1): all 6 dock, worst
5.000→**4.904** (a genuine +0.096 mm), **θ_s 0.600 ≈ canonical**, Ḣ_s ≈ canonical, torso residual ≈
canonical (0.539 vs 0.509). Zero collateral — but raising EE **further** (ee9000, 2.67:1) makes worst
**worse** (4.974). Route B optimum ≈ EE 6000, worst ~4.90 mm — still short of 4.8.

**4. The binding step (step 2, swing-b @ anchor 4) is INSENSITIVE to the ratio.**
step-2 dock = 4.904 (baseline 8:1) = **4.904** (ee6000 4:1) → 4.974 (ee9000) → 4.979 (t12000 4:1). Pinned
at ~4.90 mm regardless of the global lever, and only *worsens* off the baseline. This is the **swing-b
short-reach floor** identified in DOCK-CAUSE (corr(WBC residual, reach) = −0.68; swing-b steps 0/2/4 have
the shortest reach and the highest residual). The global ratio moves the long-reach swing-a steps (1,3:
4.41/4.44 → 4.21/4.07) and the terminal step 5 (5.000→4.860), but **cannot move the swing-b floor** — so
it cannot lower the *worst* dock below ~4.9 mm.

**5. Route comparison (the phase's question): raising EE ≪ lowering torso in θ_s collateral.** At matched
ratio 4:1 — route A θ_s **0.618** (degraded), torso residual **0.892**; route B θ_s **0.600** (≈canonical),
torso residual **0.539**. Raising EE only changes the EE:everything ratios and leaves the momentum:torso
balance intact; lowering torso directly slackens the attitude hold (θ_s + torso residual up, then dock
collapse). **Route B is strictly better collateral-wise and is the only viable direction.**

---

## Tradeoff curve (worst-dock & θ_s-peak vs torso:EE ratio)
```
ratio  route  torso:EE   worst-dock   θ_s-pk    status
 8.0   base   24000:3000   5.000       0.594     baseline (worst at gate)
 4.0   A      12000:3000   4.984       0.618     θ_s>0.6, margin 0.016
 4.0   B      24000:6000   4.904       0.600     best viable; margin 0.097 (<0.2)
 2.67  B      24000:9000   4.974       0.606     worse than ee6000
 2.67  A       8000:3000    —          (abort)   step-0 TIMEOUT
 ≤2.0  A      ≤6000:3000    —          (abort)   step-0 TIMEOUT
```
The worst-dock **bottoms at ~4.90 mm (route B, EE≈6000) and rises on either side** — it never reaches 4.8.

---

## Verdict — global route insufficient; STOP for cross-check

**No uniform global torso:EE ratio brings all 6 SS-docks < 4.8 mm with ≥0.2 mm margin without collateral.**
- The worst-docking step (step 2, swing-b) floors at ~4.9 mm **independent of the ratio** — the global
  lever cannot touch it (it is a swing-b short-reach / Jacobian-geometry floor, DOCK-CAUSE).
- Lowering torso (route A) degrades θ_s at 4:1 and **breaks docking** below it (non-monotone collapse).
- Raising EE (route B) is safe and collateral-free but **plateaus at ~4.90 mm worst** (margin 0.097) and
  worsens if pushed further.

**Best available global option (if a partial, zero-collateral gain is acceptable):** route B **EE = 6000
(torso:EE = 4:1)** — worst 5.000→4.904, θ_s 0.600 ≈ canonical, Ḣ_s and torso residual ≈ canonical. It does
**NOT** meet the ≲4.8 mm / ≥0.2 mm target. Reaching that on the swing-b steps requires a **targeted
(per-step or terminal) EE intervention** — which this phase deprioritized as patchwork. Per the phase's
stop condition, **STOP and reconsider the approach.**

---

## Deliverable (STOP-GATE)
- **Sweep table** (6 docks + θ_s + realized Ḣ_s ×6 + torso residual + h_w + feasibility per ratio): above +
  `ratio_sweep.json`.
- **Recommended minimal-intervention ratio:** none meets target; the best safe global option is **route B
  4:1 (EE 6000)** at **+0.096 mm** worst-dock for **Δθ_s ≈ +0.006° (0.594→0.600), ΔḢ_s ≈ 0** — but it
  falls **0.1 mm short** of the target margin.
- **Lowering-torso vs raising-EE:** raising EE (B) is strictly better — less θ_s collateral (0.600 vs
  0.618) and less torso residual (0.539 vs 0.892) at matched 4:1, and route A collapses (dock TIMEOUT)
  below 4:1 while route B stays feasible.
- **Insufficiency root cause:** step-2 (swing-b) dock is **ratio-invariant at ~4.90 mm** — the global lever
  cannot move the worst step.

NO canonical commit. `crawlbot/` untouched. Raw runs (`figC_ratio_*`) gitignored. **STOP for cross-check.**
