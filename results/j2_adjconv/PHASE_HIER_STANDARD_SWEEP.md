# Phase HIER-STANDARD-SWEEP — torso:EE toward co-priority, momentum FIXED 5000

**Branch** `j2/ds-active-rework` · measurement sweep, NO canonical commit · pushed, never merged.
Data: `results/j2_adjconv/hier_sweep.json`; scripts `scripts/diag_hier_{run_one,extract}.py`.
Canonical stack (posture 20, torque 1, wrench 0.01, accel-reg 0.01, ε 1e-6); **`ss_alpha_mom`
held at 5000 in every run** (preserves swing-saturation + WBC-realizes-NMPC contract). One axis at a
time. **Dock = AT-WELD** (`dock_events` weld instant, per the FULL-DIAG-EXPORT fly-by lesson), NOT
min-over-swing. Reused canonical-stack runs where present (EE 3000/6000/9000, from EE-RATIO-GLOBAL);
3 new EE + 3 new torso runs with κ capture.

**Answer up front:** co-priority is reachable **by raising EE** (torso stays 24000, EE→24000 = 1:1 →
6/6 dock, saturation intact). It is **NOT reachable by lowering torso** — the torso floor (~12000, step-0
timeout below) is EE-independent, so the humanoid-style "demote the torso" only reaches **2:1**. VISPA's
docked stance-arm requires the torso to stay a high-authority task; you can make torso/EE numerically
co-priority only by lifting EE to the torso's *high* level (co-HIGH), not by dropping the torso to low.

## SWEEP 1 — raise EE (torso 24000, momentum 5000 fixed)
| config | torso:EE | at-weld docks s0…s5 [mm] | worst | margin | SS-sat (s2/s4) | θ_s pk / settled | e_com pk | κ_SS |
|---|---|---|---|---|---|---|---|---|
| ee3000 (canon) | 8.0:1 | 4.94 4.41 4.90 4.44 4.76 5.00 | 5.000 | 0.000 | **5.00/5.00** ✓ | 0.594 / 0.107 | 0.0953 | ~3.6e6\* |
| ee4500 | 5.3:1 | 4.79 4.29 4.88 4.20 4.89 4.98 | 4.980 | 0.020 | 5.00/5.00 ✓ | 0.596 / 0.090 | 0.0953 | 4.6e6 |
| **ee6000** | **4.0:1** | 4.84 4.21 4.90 4.07 4.77 4.86 | **4.900** | **0.100** | 5.00/5.00 ✓ | 0.600 / 0.091 | 0.0953 | ~5e6\* |
| ee9000 | 2.67:1 | 4.92 4.14 4.97 4.32 4.79 4.70 | 4.970 | 0.030 | 5.00/5.00 ✓ | 0.606 / 0.092 | 0.0953 | ~7e6\* |
| ee12000 | 2.0:1 | 4.85 4.11 4.96 4.30 4.88 4.68 | 4.960 | 0.040 | 5.00/5.00 ✓ | 0.607 / 0.093 | 0.0953 | 9.0e6 |
| ee24000 | **1.0:1** | 4.76 4.08 4.97 4.28 4.94 4.68 | 4.970 | 0.030 | 5.00/5.00 ✓ | 0.617 / 0.107 | 0.0953 | 1.65e7 |

## SWEEP 2 — lower torso (EE 6000 fixed, momentum 5000 fixed)
| config | torso:EE | at-weld docks s0…s5 [mm] | worst | margin | SS-sat (s2/s4) | θ_s pk / settled | e_com pk | κ_SS |
|---|---|---|---|---|---|---|---|---|
| t24000 (=ee6000) | 4.0:1 | 4.84 4.21 4.90 4.07 4.77 4.86 | 4.900 | 0.100 | 5.00/5.00 ✓ | 0.600 / 0.091 | 0.0953 | ~5e6\* |
| t18000 | 3.0:1 | 4.89 3.93 4.97 4.19 4.63 4.88 | 4.970 | 0.030 | 5.00/5.00 ✓ | 0.621 / 0.100 | 0.0951 | 4.97e6 |
| t12000 | 2.0:1 | 4.99 4.08 4.98 4.29 4.71 4.83 | 4.990 | 0.010 | 5.00/5.00 ✓ | 0.628 / 0.107 | 0.0948 | 4.56e6 |
| **t8000** | **1.33:1** | — TIMEOUT (step-0, min d 7.0 mm, abort) — | — | — | **LOST** (abort) | 0.164† | 0.092† | 3.95e6 |

\* κ not separately captured for reused runs (ee3000/6000/9000, t24000); value is the QP-COND
span-driven estimate (torso 24000 sets λ_max). Captured points bracket it. † t8000 numbers are
early-abort artifacts (only a partial step 0 ran) — NOT a low-collateral operating point.

## Findings
1. **SS-swing saturation PRESERVED across the entire viable sweep** (every 6/6 config: SS Ḣ_s s2=5.00,
   s4=5.00). Momentum held at 5000 ⇒ the swing saturates the τ_w cap regardless of torso:EE — **thesis
   intact** at all ratios down to 1:1. Only the aborted t8000 loses it (artifact).
2. **e_com CONSTANT (0.0948–0.0953)** across all viable configs — momentum-held preserves the CoM
   tracking / WBC-realizes-NMPC contract exactly; the torso:EE ratio does not touch it (momentum owns CoM).
3. **Co-priority reachability is route-dependent:**
   - **Raise EE → 1:1 WORKS** (ee24000): 6/6, sat ✓, θ_s 0.617, e_com 0.0953 — but κ climbs to **1.65e7**
     (the EE Jacobian adds to λ_max). This is *co-HIGH* (torso keeps its 24000 authority).
   - **Lower torso → floor at 2:1** (t12000 works, margin 0.010; t8000 = 1.33:1 **TIMES OUT**). The torso
     floor (~12000) is **EE-independent** — the extra EE authority (6000 vs the EE-RATIO-GLOBAL 3000)
     does NOT rescue it (same floor). Lowering κ falls (5e6→3.95e6), but docking breaks first.
4. **Dock worst plateaus ~4.9–4.99** across the whole viable sweep; best is **4:1 (EE 6000): worst 4.900,
   margin 0.100**. With momentum FIXED at 5000, torso:EE alone cannot push the dock below ~4.9 mm
   (consistent with EE-RATIO-GLOBAL, and with USERW2 showing the dock lever is the *momentum* weight).
5. **θ_s rises modestly toward co-priority** (0.594 → 0.628 at the extremes), all viable ≤0.63 — controlled
   (torso authority preserved on the raise-EE route; slightly degraded on the lower-torso route).
6. **κ diverges by route:** raise-EE **worsens** κ (3.6e6→1.65e7); lower-torso **improves** κ
   (5e6→3.95e6) but hits the dock-timeout floor first.

## Recommended ratio
**4:1 (EE = 6000, torso 24000, momentum 5000)** is the operating sweet spot: **best at-weld dock (worst
4.900, margin 0.100)**, θ_s 0.600 (at target), SS-saturation ✓, e_com tight (0.0953), κ ~5e6 (well
below the co-priority-EE 1.65e7). It is a modest, well-conditioned move from canonical 8:1 toward
co-priority that *improves* the dock margin (0.000→0.100) with negligible collateral — the
literature-direction, thesis-intact, best-conditioned point.
- If **true 1:1 co-priority** is required: `ee24000` reaches it (thesis ✓, docks ✓) at κ 1.65e7 / θ_s 0.617.
- The **humanoid-direction** (lower torso) closest is **2:1** (t12000/ee12000); **1:1 by lowering torso is
  unreachable** (torso floor).

## Verdict — is co-priority reachable?
**Partially, and it is a reportable VISPA-specific deviation.** torso:EE = **2:1 is reachable both ways**
keeping saturation + docks. **1:1 is reachable only by raising EE** to the torso's high weight (co-HIGH),
**never by lowering the torso** — VISPA's docked stance arm needs torso ≥ ~12000 to hold attitude, so the
standard humanoid move ("demote torso to low priority") is blocked at 2:1. **VISPA's docked torso does force
a higher torso authority than the humanoid norm**: it can be co-priority *in magnitude* only by keeping the
torso high, not by lowering it — a real, defensible finding for the paper.

## Deliverable (STOP-GATE)
Two sweep tables above (at-weld docks ×6 + SS-sat + θ_s + e_com + κ + feasibility, full weight vector:
torso/hw-slack 10000/mom 5000/EE/posture 20/torque 1/wrench 0.01/accel-reg 0.01/ε 1e-6); recommended ratio
**4:1 (EE 6000)**; torso floor **between 8000 and 12000** (t8000 times out); co-priority **reachable by
raising EE (co-HIGH), not by lowering torso**. NO canonical commit. `crawlbot/` untouched. Raw runs
(`figC_hier_*`) gitignored. **STOP for cross-check.**
