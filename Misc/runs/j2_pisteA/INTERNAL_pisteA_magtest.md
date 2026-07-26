# INTERNAL — Piste A magnitude test: does the budget help CoM tracking at larger translation?

**Follow-up to the mag-0.05 Piste A sweep — characterize, NO success threshold.** Raw numbers; Piste A's
framing is for Idriss + reviewing Claude. Branch `j2/ds-active-rework` (pushed, never merged). No `crawlbot/`
change — reuses the committed Piste A knobs.

**DECISIVE VERDICT — NO. Opening the budget (β>0) does NOT reduce the CoM lag at mag 0.20** (com_max
0.1943→0.1939, **−0.2 %**; com_fin flat 0.0099→0.0098), and it **saturates at β=0.5** — even β=2.0 (W=7.78 W)
is inert. The reason is decisive: **the positive work the QP actually wants stays tiny — max +0.0041 W —
*unchanged* from mag 0.05**, while the budget is 1.9–7.8 W. The budget is **never the limiter at any
magnitude**. ⇒ **the CoM lag is KINEMATICS/REFERENCE-limited, not passivity-limited.** Piste A provides the
**safe, envelope-coupled, bounded-positive-work GUARANTEE** (budget never violated, envelope never exceeded)
— **not a tracking improvement.** This reframes what Piste A contributes.

---

## The decisive measurement — mag 0.20 β-sweep (exact box, dt_ds 2.5, n=2)

| β | CoM err max / final [m] | **needed work** max dqⱼᵀτ_q [W] | W_budget max [W] | passivity p_frac | positive-work frac | budget viol. | ‖Ḣ_s‖∞ (cap 5)/over | manip before→after | docks/to |
|---|---|---|---|---|---|---|---|---|---|
| 0 (strict) | 0.1943 / 0.0099 | −0.000 | 0.000 | 0.61 | 0.00 | 0 | 0.816 / 0 | 0.2541→0.2571 | 2/0 |
| 0.5 | 0.1939 / 0.0098 | **+0.0041** | 1.946 | 0.53 | 0.12 | 0 | 0.661 / 0 | 0.2541→0.2570 | 2/0 |
| 1.0 | 0.1939 / 0.0098 | **+0.0041** | 3.892 | 0.53 | 0.12 | 0 | 0.661 / 0 | 0.2541→0.2570 | 2/0 |
| 2.0 | 0.1939 / 0.0098 | **+0.0041** | 7.784 | 0.53 | 0.12 | 0 | 0.661 / 0 | 0.2541→0.2570 | 2/0 |

- **Tracking (the decisive metric): unchanged by β.** com_max 0.1943→0.1939 (−0.2 %), com_fin flat. The lag
  ≈ the full 0.20 m offset (ramp lag + the NMPC-horizon lead, reference queried t+0.8 s ahead) — a structural
  reference/kinematics effect β does not touch.
- **Needed work ≈ +0.0041 W — and it does NOT grow with magnitude** (mag 0.05 it was +0.0024 W; mag 0.20 it
  is +0.0041 W — both ≪ the 1.9–7.8 W budget). The DWELL is long (~1.5–2 s for the 20 cm move), so even a
  large CoM translation is quasi-static (~0.1 m/s, low KE) ⇒ negligible net positive work. **The budget is
  ~500–2000× over-provisioned at every magnitude** ⇒ never limiting ⇒ β saturates almost immediately (β=0.5).
- **Budget IS exercised but tiny:** positive work occurs in 0.12 of moving ticks (vs 0.00 strict),
  passivity binding eases 0.61→0.53 (not to 0 — at this magnitude many ticks are still dissipative). The
  budget enables bounded positive work; it just isn't what gates the CoM.
- **SAFETY holds at 4× magnitude:** budget **never violated** (0 ticks), envelope **never exceeded**
  (‖Ḣ_s‖∞ ≤ 0.82 ≪ 5, 0 over-cap; β>0 even *lowers* realized usage 0.816→0.661), feasible (qpf=0, nin=0,
  2/2 dock).
- **Manipulability ("rapprocher"): +1.2 % (0.2541→0.2570), β-independent** — same as α saw under strict
  passivity. The budget does not change the conditioning gain.

## Cross-magnitude summary (the budget never buys tracking)

| mag | β=0 com_max→ β>0 com_max | com_fin β=0→β>0 | needed work max [W] | passivity p_frac β=0→β>0 | β-saturation |
|---|---|---|---|---|---|
| 0.05 | 0.0484 → 0.0469 (−3 %) | 0.0029 → 0.0075 (worse) | +0.0024 | 0.79 → 0.00 | β=0.25 |
| 0.10 | 0.0968 → 0.0938 (−3 %) | 0.0058 → 0.0150 (worse) | (≈0.003) | 0.62 → 0.00 | β=1.0 |
| 0.20 | 0.1943 → 0.1939 (−0.2 %) | 0.0099 → 0.0098 (flat) | +0.0041 | 0.61 → 0.53 | β=0.5 |

At every magnitude: **com_max barely moves (≤3 %), com_fin flat-or-worse, the needed work stays ~0.003–0.004 W
(never grows), and β saturates almost immediately.** The hypothesis (larger translation ⇒ more needed work ⇒
budget becomes limiting ⇒ helps tracking) is **refuted** — the needed work does not scale up, so the budget
is never the limiter.

## Residual & C1–C5 (mag 0.20)

- **Residual** (traversal-final): β=0 **0.001746**, β>0 **0.001754** (+0.5 %, flat). β barely moves it.
- **C1 docking: FAIL** both β (worst 4.95/4.96 mm, margin 0.05 < baseline 0.06 — the marginal kinematic
  dock-floor trip, ~β-independent). All 5 dock, 0 timeouts.
- **C3 envelope: PASS** (‖Ḣ_s‖∞_SS = 5.00, the SS AOCS binding — unchanged by the DS budget).
- **C5 h_w∞: FAIL** — β=0 **4.881**, β=2.0 **4.891** (the budget adds +0.01 Nms, negligible). The C5 FAIL is
  the **exact-box (LOT B)** effect, **NOT** the budget. **Larger magnitude does NOT push h_w further** — at
  mag 0.20 h_w ≈ 4.88 vs mag 0.05's 4.93 (slightly *lower*); the exact-box h_w cost (~4.88–4.93, ~98 % of the
  5 Nms cap) is roughly magnitude-independent, driven by the SS envelope binding.
- **C6 OFF: BIT-IDENTICAL** both runs.

---

## Decisive output & flags

**At mag 0.20, opening the budget does NOT reduce the CoM lag (NO).** Combined with mag 0.05 and 0.10:
**the CoM lag is kinematics/reference-limited, not passivity-limited, at all tested magnitudes.** Piste A
delivers the **safe bounded-positive-work guarantee** — provably never violating the envelope, enabling
bounded positive work, β an interpretable knob — **but it is not a CoM-tracking enabler.**

**Divergences vs prior audit facts:**
1. **Refutes the α inference** that the moving-CoM conflict being "passivity-dominated" (passivity binds
   61–100 %) means relieving passivity (Piste A) would buy tracking. Across mag 0.05/0.10/0.20, relieving
   passivity does **not** reduce the lag — passivity *binds* but was never the tracking bottleneck. **Reframe
   Piste A: a safety/feasibility guarantee (bounded positive work within the envelope), not a tracking
   mechanism.**
2. **The "needed work scales with magnitude" hypothesis (this brief's premise) is refuted:** needed work
   stays ~0.003–0.004 W from mag 0.05 to 0.20 (the move is quasi-static over the long DWELL), so the budget
   is over-provisioned at every magnitude.
3. **Consistent with the dock-floor audit:** the dock stays ~4.5–5.0 mm (kinematic); β/magnitude shift it
   only ~0.05 mm at the margin (C1 marginal FAIL). And with the envelope/Piste-A audits: the envelope has
   huge DS headroom (‖Ḣ_s‖∞ ≤ 0.82 ≪ 5) — confirming the conflict was never envelope-limited either.
4. **LOT-B (exact box) C5 cost is real and ~magnitude-independent** (h_w ~4.88–4.93, C5 FAIL), from the SS
   envelope binding — not amplified by larger translation.

## Reproduce
```
bash Misc/scripts/run_pisteA_mag_sweep.sh   # mag 0.20 β{0,0.5,1.0,2.0} + mag 0.10 + C1-C5(n=5)
```
Supporting: `magtest_beta_mag0.20.log`, `magtest_beta_mag0.10.log`, `magtest_residual.log`,
`magtest_gate_C1-C5.log` (this dir). No regression run — no `crawlbot/` change (characterization only;
flags committed `a603c82`).

**STOP after the report.** No merge, no PR. Piste A's framing (safety guarantee vs tracking) is decided on
these numbers.
