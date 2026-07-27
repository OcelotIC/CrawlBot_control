# C3.1 + C3.5 — AOCS decomposition report and docking table

**Review-Closure Bloc 2.** Post-processing only, no runs. Source: the C2
`--solver-diag` export of the canonical (92 columns + meta), which carries the
AOCS decomposition (C2.3) and the per-dock 6-D twist (C2.1).

Commit worked on: parent **`739e14f`** (C3.2). Env exact-matches
`gate/environment.lock`. Artifacts: `c3_1_c3_5.json`, `c3_1_c3_5.py`.

Both sections use the **traversal window** (up to the last single-support tick,
1876 of 2077) as the primary convention, with the full log reported alongside —
the C4 convention, adopted because the 20 s terminal settle moves peaks
materially and contributes zero saturation.

---

## C3.1 — AOCS decomposition

### Per-component peak and RMS, per axis, over the traversal [N·m]

| term | peak x | peak y | peak z | RMS x | RMS y | RMS z |
|---|---:|---:|---:|---:|---:|---:|
| `tau_ff` | 2.50000 | 2.50000 | **2.65373** | 0.35927 | 0.75513 | **1.22425** |
| `tau_att_p` | 0.00232 | 0.00465 | 0.00640 | 0.00070 | 0.00228 | 0.00283 |
| `tau_rate_d` | 0.04104 | 0.04929 | 0.08931 | 0.01109 | 0.01567 | 0.03251 |
| `tau_accel_d` | 0.59947 | **1.38235** | 0.21720 | 0.03070 | 0.04732 | 0.01350 |
| `tau_antiwindup` | **0** | **0** | **0** | **0** | **0** | **0** |
| `tau_w_preclip` | 2.50766 | **3.89899** | 2.73846 | 0.35707 | 0.76311 | 1.22040 |

Identity `Σ terms = tau_w_preclip` holds to **9.88e-07**, the CSV's
6-significant-figure formatting floor.

### Feedforward vs feedback share

| | value |
|---|---:|
| feedforward RMS | **0.85597** N·m |
| feedback RMS (att-P + rate-D + accel-D) | **0.03894** N·m |
| ratio | **22.0 : 1** |
| feedforward share of the RMS budget | **95.65 %** |

**The AOCS is a feedforward controller with a small trim.** The attitude-P term
peaks at **6.4 mN·m** — 0.24 % of the feedforward peak — which is what
`K_θ = 1.0 N·m/rad` against `θ_s ≤ 0.0094 rad` must give. The second-order
damping term is the largest feedback contribution and it is concentrated on
**y** (peak 1.382 N·m), an axis on which the feedforward is *not* the largest.

**Anti-windup is identically zero: max |·| = 0.0e+00, nonzero on 0 of 1876
ticks.** Not "small" — exactly zero, on every tick and axis, because
`|h_w|_∞ = 4.1019 < 5` throughout. C1.4 inferred this from the h_w peak; it is
now a direct per-tick measurement.

**One structural observation.** `tau_ff` peaks at *exactly* 2.50000 on x and y.
That is not a coincidence and not a clip: in DS the feedforward is the
contact-wrench couple `−Σ(r_Ci×f_i + τ_i)` built from `λ_qp`, and the QP's
envelope box constrains exactly that quantity to ±2.5. So the DS feedforward
inherits the QP bound. Only z exceeds it (2.654), from the SS
finite-difference branch, which no box constrains.

### Saturation at the ±2.5 cap — every convention

| window | any-axis per tick | axis-sample | per-axis [x, y, z] | peak demand |
|---|---:|---:|---|---:|
| **traversal** | **5.064 %** | 1.759 % | 0.16 / 1.12 / **4.00** % | 3.899 |
| full log | **4.574 %** | 1.589 % | 0.14 / 1.01 / 3.61 % | 3.899 |
| — SS | 3.937 % | — | — | 2.739 |
| — DS_interstep | **5.482 %** | — | — | **3.899** |
| — DS_terminal | 0.000 % | — | — | 0.705 |

**⚠ The paper's cited 4.1 % reproduces under no convention.** The nearest is
any-axis-per-tick over the *full log*, 4.574 %; the traversal-window figure is
5.064 % and the axis-sample figures are 1.6–1.8 %. The cited
"368 / 51 448 plant clamps" is at a third cadence entirely — 51 448 is neither
the tick count (2077), nor the axis-sample count (6231), nor ticks × 10
sub-steps (20 770) — so its denominator is not recorded anywhere in the
repository. **The figure needs restating from these channels with its
convention named.**

Saturation is concentrated on **z** (4.0 % of traversal ticks) and is **more
frequent in the inter-step settles than in the swings** (5.48 % vs 3.94 %),
with the terminal settle never saturating. C4 confirmed the same ordering
across three independent configurations.

---

## C3.5 — Docking table

Per-dock 6-D relative twist at capture, structure frame, with the thresholds in
force (`eps_pos = 5 mm`, `eps_ori = 5°`, `eps_twist = 0.05` on all six).

| step | arm | t [s] | d [mm] | ori [°] | ‖twist‖ | linear | angular | lin % | d margin [mm] | twist margin |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | b | 3.01 | 4.020 | 0.140 | 0.020058 | 0.014826 | 0.013510 | **73.9** | 0.980 | 0.029942 |
| 1 | a | 14.72 | 4.890 | 0.040 | 0.007735 | 0.007288 | 0.002590 | 94.2 | 0.110 | 0.042265 |
| 2 | b | 28.50 | **4.990** | 0.020 | 0.005781 | 0.005360 | 0.002165 | 92.7 | **0.010** | 0.044219 |
| 3 | a | 40.06 | 4.970 | 0.050 | 0.007170 | 0.006725 | 0.002487 | 93.8 | 0.030 | 0.042830 |
| 4 | b | 53.38 | 4.950 | 0.010 | 0.004990 | 0.004647 | 0.001818 | 93.1 | 0.050 | 0.045010 |
| 5 | a | 64.54 | 4.620 | 0.040 | 0.007020 | 0.006554 | 0.002516 | 93.4 | 0.380 | 0.042980 |

**Capture is linear-dominated** — 92.7–94.2 % of the twist norm is
translational on five of six docks. Step 0 is the outlier at 73.9 %, with an
angular component 5× the others; it is the only step released from the initial
IK pose rather than from a settled post-dock state.

**Position is the binding criterion, not twist.** Every accepted dock clears the
twist gate by 60–90 % of the bound (margins 0.030–0.045 against 0.05) while
position margins are 0.010–0.980 mm against 5 mm. The capture is decided by
distance; the twist term is slack *at the docks that happened*.

### The refused approaches, and what they revise

| step | t [s] | d [mm] | ori [°] | ‖twist‖ | over `eps_twist` by |
|---:|---:|---:|---:|---:|---:|
| 2 | 21.90 | **4.334** | 0.105 | 0.060522 | **+21.0 %** |
| 2 | 22.00 | 4.941 | 0.234 | 0.057386 | +14.8 % |
| 4 | 46.58 | 3.409 | 0.157 | 0.057616 | +15.2 % |
| 4 | 46.68 | **3.326** | 0.230 | 0.050607 | **+1.2 %** |

**This refines two earlier claims, in opposite directions.**

**(i) `eps_twist = 0.05` is more defensible than C1.6 implied.** The accepted
twists cluster at 0.005–0.020 and the refused ones at 0.051–0.061, with **an
empty factor-2.5 gap between 0.020 and 0.051**. The threshold sits in that gap.
Any value in roughly (0.021, 0.050) would produce *identical* results — the
bound is robust over a 2.4× range, not slicing through a dense distribution.
The system either arrives quiescent or arrives moving; 0.05 separates the two
populations. That is a better characterisation than "untuned", even though the
source comment is right that it was never swept.

**(ii) But the traversal-time cost is far more threshold-sensitive than the
dock precision.** The two effects need separating:

- **Dock precision.** To accept step 2's 4.334 mm approach you would need
  `eps_twist ≥ 0.0606`, a **+21 %** change. So C1.6's finding stands: the worst
  reported dock (4.990 mm, margin 0.01 mm) is a gate artifact, and undoing it
  takes a substantial threshold change.
- **Traversal time.** Step 4's second refusal misses by **1.2 %**. A threshold
  of 0.051 would have accepted it, recovering **7.0 s of the 8.5 s** C4
  attributes to capture refusals — while leaving the worst dock unchanged at
  4.990 mm. So ~82 % of the timing penalty hangs on a 1.2 % threshold margin,
  and the dock-precision penalty does not.

Quoting a cost-of-constraint duration without `eps_twist` beside it therefore
reports a number that a 1.2 % change to an unswept constant would move by 82 %.

---

## STOP

C3.1 and C3.5 complete. Two items carried to the corrections memo: the
unreproducible 4.1 % clip fraction (C3.1) and the threshold-sensitivity split
above (C3.5).

**C3.3 is now unblocked and its premise has changed.** The brief gated it on
C3.2(e) showing whether the conservation residual could account for the 0.048°
θ_y plateau. It cannot: C3.2(e) established the residual is injected at six
discrete weld events and then does not move — 0.0000e+00 drift over 879 s — so
there is no time-accumulating rotational floor of the ~0.04 %/900 s kind the
brief hypothesised. The plateau is therefore **not** explained away, and the
standing hypothesis (bias in the y error path) is live. That is a different
kind of investigation from the rest of C3 and is left as its own decision.
