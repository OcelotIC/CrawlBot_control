# INTERNAL — DS inter-step settle chatter: ROOT-CAUSE DIAGNOSIS (no fix)

**Diagnosis only — NO fix applied, NO `crawlbot/` change.** Branch `j2/ds-active-rework` (pushed, never
merged). Reproducer `Misc/scripts/diag_chatter.sh` + `Misc/scripts/audit_chatter.py`. The active set is reconstructed
**offline** from the logged `lambda_qp` (the exact envelope box `|M_exact·λ| ≤ τ_w_max` has a fixed RHS, so
no logging hook was needed — flagged: zero crawlbot touch).

## VERDICT — **H2 (structural QP degeneracy), NOT H1 (c_curr).**

Freezing `c_curr` (test A) leaves the run **byte-identical** (worst |Δ| = 0.000 across the whole sim_log) →
`c_curr` is **irrelevant** to the chatter. The chattering constraint is the **exact envelope box**
`|M_exact·λ| ≤ 5` (a **fixed** RHS, no `c_curr`); `c_curr` enters only the *hw* box, which never binds in the
settle (inter-step h_w is small). The chatter is a **bounded period-2 active-set limit cycle**: in the arm-a
settle configs the exact origin-referenced `Ḣ_s` *wants* ≈7.27 N·m (orbital term ≈6) — above the ±5 envelope —
so the box binds on the y,z axes, and the settle QP's velocity-dissipation cost leaves a **flat direction in
the net-contact-force sign**, giving **two equal-cost saturating vertices (A ≈ −B)** that the active-set
solver alternates between every tick.

## Per-segment chatter metrics — baseline vs c_curr-frozen (test A)

Inter-step settles (canonical 5-step, exact box + AOCS-on). Segments labelled by the **dock that immediately
precedes** the settle (the just-docked arm/anchor):

| settle | preceding dock | n ticks | flip-frac | at±5 | exact ‖Ḣ_s‖ med | proxy ‖Ḣ_s‖ med | orbital med | ‖Δλ‖ med | Δ-trend | vtx dist ‖A−B‖ |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | (initial, at rest) | 10 | 0.89* | 0.00 | 0.000 | 0.000 | 0.000 | 0.00 | — | 0.00 |
| 1 | **arm-b** anchor3 | 101 | 0.030 | 0.10 | 1.151 | 0.250 | 0.768 | 0.06 | 0.03 | 0.05 |
| 2 | **arm-a** anchor3 | 50 | **0.939** | **0.86** | **7.274** | 1.319 | **5.989** | **13.78** | 1.44 | **11.57** |
| 3 | **arm-b** anchor4 | 100 | 0.031 | 0.10 | 1.516 | 0.216 | 1.220 | 0.05 | 0.05 | 0.06 |
| 4 | **arm-a** anchor4 | 50 | **0.980** | **0.96** | **7.112** | 1.286 | **5.966** | **13.04** | 1.35 | **11.70** |

`*` settle-0 flip-frac is sign-noise of a ≈0 wrench (at rest, exact ‖·‖=0) — not chatter.
**c_curr-frozen (test A): every value identical** to baseline (byte-identical run) — omitted for space; see
`chatter_analysis.log`. The chattering segments are exactly the two that **follow the arm-a docks** (the
brief's "arm-a docks, anchor 3 then 4"); the settle-index is +1 vs the dock-event index (a logging
convention), but by time they are the post-arm-a-dock settles.

## Active-set trace on a chattering segment (settle 2, post-arm-a-dock, steady ticks 20–31)

```
even tick: Ḣ_s = [−1.5, −5.00, +5.00]  binding: y−, z+   Σf = [+6.2, +0.7, +2.6]
odd  tick: Ḣ_s = [+2.3, +5.00, −5.00]  binding: y+, z−   Σf = [−6.8, +0.5, −2.9]
vertex A (even) λ ≈ [+2.3,+0.3,+1.0, −0.5,−1.7,+1.8, +4.0,+0.2,+1.7, −0.9,−2.4,+2.5]
vertex B (odd ) λ ≈ [−2.6,+0.1,−1.0, +0.7,+1.7,−1.9, −4.5,+0.4,−1.7, +1.1,+2.4,−2.7]
‖A−B‖ = 13.85   intra-A std = 0.21   intra-B std = 0.19
```
The **y and z envelope-box rows flip sign every tick** (vertex A binds {y−,z+}, vertex B binds {y+,z−}); the
**net contact force Σf itself flips sign** (A ≈ −B). `‖A−B‖ = 13.85 ≫ intra-vertex std 0.2` ⇒ **two fixed
vertices**, i.e. a textbook active-set alternation, not a drifting solution.

## Test B — wrench-rate (limit cycle vs instability)

`‖λ_k − λ_{k-1}‖` median ≈ **13.8 / 13.0 N·m** on the chatterers (the wheels are commanded to slam the wrench
 tick-to-tick), with **Δ-trend (2nd-half / 1st-half mean) ≈ 1.3–1.4 ≈ constant** and the two vertices fixed
(intra-std 0.2 ≪ ‖A−B‖ 13.85). ⇒ **constant-amplitude bounded limit cycle**, NOT a growing dynamic
instability. (Clean segments: ‖Δλ‖ med ≈ 0.05 — three orders smaller.)

## H1 vs H2 — discrimination

| test | result | implication |
|---|---|---|
| **Freeze c_curr (A)** | baseline ≡ frozen, **byte-identical (|Δ|=0)**; chatterer flip-frac 0.959 → 0.959 | **H1 refuted** — c_curr does not trigger it |
| envelope-box RHS | `τ_w_max = 5`, **fixed** (no c_curr); c_curr only in the non-binding hw box | confirms c_curr cannot be the trigger |
| two-vertex / amplitude (B) | two fixed vertices A≈−B, constant amplitude | **bounded active-set limit cycle** (degeneracy), not instability |

⇒ **H2 confirmed.** The trigger is the exact envelope box binding in the arm-a configs combined with a flat
cost direction → two equal-cost vertices → period-2 cycling.

## Why arm-a only (the configuration asymmetry)

The discriminator is the **orbital / lever term `r_com × Σf`** (the part of the exact `Ḣ_s` the proxy omits):

| settle | orbital ‖·‖ med | proxy ‖Ḣ_s‖ med | **exact ‖Ḣ_s‖ med** | envelope binds? | chatters? |
|---|---|---|---|---|---|
| arm-b (1,3) | 0.77 / 1.22 | 0.25 / 0.22 | **1.15 / 1.52 (< 5)** | no | **no** (flip 0.03) |
| arm-a (2,4) | **5.99 / 5.97** | 1.32 / 1.29 | **7.27 / 7.11 (> 5)** | **yes** | **yes** (flip 0.94 / 0.98) |

In the arm-a settle configs (robot CoM `r_com ≈ [0.75, −0.58, −0.35]` m and the net dissipation force
`Σf ≈ ±[6, 0.6, 2.7]` N), the **orbital lever term `r_com×Σf` ≈ 6 N·m** drives the exact origin-referenced
`Ḣ_s` to ≈7.27 N·m — **above the ±5 envelope** — so the box binds and the degeneracy cycles. The arm-a docks
are the **short ~50-tick settles** (fast plateau); the arm-b docks are the **long ~100-tick settles** with a
small orbital (<1.5) so the exact `Ḣ_s` stays under 5 and the box **never binds** → no degeneracy → clean.
(Note: the **proxy** `Ḣ_s` is ~1.3 in *both* — the binding, chattering quantity is entirely the orbital term,
not the proxy/centroidal part, confirming the exact box is what triggers it.)

## Task 5 (sampling-rate control) — not run, ruled out analytically

A higher QP rate was **not** run (the brief gated it on "time permits"). It is unnecessary: aliasing is ruled
out by the **two-fixed-vertex** signature (‖A−B‖ ≫ intra-std) + constant amplitude — an active-set vertex
alternation is a property of **each QP solve** (the solver picks the alternate equal-cost vertex each call),
so it reproduces at any `dt` as the *same* period-2 cycle at a higher tick density; sampling rate is not the
cause and a higher rate would not remove it.

## Recommended targeted fix (for the next brief)

H2 ⇒ break the cost degeneracy so the settle QP picks a **unique** vertex on the binding-envelope manifold:
- **Wrench-rate / proximal regularization** (minimal, targeted): add a small `‖λ_k − λ_{k-1}‖²` (or Tikhonov
  `‖λ‖²`) term to the settle-QP cost so the flat direction is resolved toward the previous tick's wrench → no
  sign alternation. This is the direct fix for the active-set cycling and only touches the settle QP cost.
- **NOT** freezing `c_curr` (proven irrelevant, H1 refuted) and **NOT** a sampling-rate change (ruled out).
- Separate, related point (out of scope here): in the arm-a configs the exact `Ḣ_s` genuinely *wants* >5 N·m
  (orbital ≈6), so even a unique vertex sits **at** the ±5 envelope — the demonstration envelope is *marginal*
  in those configs. The rate penalty removes the chatter; whether the wheels should run at the ±5 cap there
  is an envelope-margin question for later. (The chatter is currently filtered by the structure inertia —
  `h_w`, `θ_s` stay smooth — so this is a wheel-command/QP-conditioning issue, not a tracking failure.)

## Reproduce
```
bash Misc/scripts/diag_chatter.sh   # baseline + c_curr-frozen (exact box, n=5) + offline analysis
MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/audit_chatter.py \
  base=results/chatter_base cfrozen=results/chatter_cfrozen
```
Supporting: `Misc/runs/j2_chatter/chatter_analysis.log`. Raw run dirs reproducible from the driver, not
committed. **No fix applied. No `crawlbot/` change. No merge, no PR.**
