# PHASE M0 — PAPER_CORRECTIONS_MEMO item 7, measured at 100 Hz

**Branch** `refactor/sim-loop-split`. Instrumentation `f75f316` (logging only,
proven inert). Measurement: `scripts/diag_m0_aocs_hifreq.py --settle` then
`--analyse`; artifacts `results/m0_aocs_hifreq/m0_summary.json`,
`m0_analysis.json` (the 37 MB per-step trace is regenerable, not committed).

## 1. The open item

`PAPER_CORRECTIONS_MEMO.md` item 7 (`d37ad02`): on the instrumented 900 s
settle, commanded wheel torque integrates to **+2.884 N·m·s** while the logged
wheel momentum changes by **+0.084 N·m·s** (z) — a 34× gap. Cowork's hypothesis:
the 10 Hz log samples only QP sub-step `qs=9`, missing the opposite-sign `qs=0`
AOCS kick (attitude.md §3).

## 2. Run

The canonical kwargs with `settle_seconds=900` (as `c3_3_run_settle900.py`),
`cfg.log_hifreq_all=True`: one record per plant step, 90 000 sub-steps over the
trailing DS hold `[64.54, 964.54] s`. No wheel command saturates (pre-clip max
|τ_w| 2.146 N·m).

## 3. Result — three factors, the gap closes

| (z axis, 900 s hold) | N·m·s |
|---|---:|
| ∫τ_w of the `qs=9` sub-steps (what the 10 Hz log integrates) | +3.357 |
| ∫τ_w over all 100 Hz sub-steps (applied) | **+2.668** |
| ∫τ_w of the `qs=0` sub-steps | −2.701 (the kick, opposite sign) |
| Δh_w of the channel, end − start | +1.302 |
| Δh_w of the channel as slope × duration (linear fit) | +0.060 |

**Per-step regression** over 96 442 consecutive pairs of plant steps:

```
Δh_w = k · τ_w·dt     k_x = 0.49998   k_y = 0.49993   k_z = 0.49990   (corr ≥ 0.99991)
```

| factor | value | cause |
|---|---:|---|
| metric | **21.8×** | Δh_w taken as slope × T of a linear fit on a trajectory that rises then plateaus (0.060 vs 1.302) |
| sampling | **1.26×** | the 10 Hz log records sub-step `qs=9` only; the `qs=0` kick is invisible to it (3.357 vs 2.668) — **Cowork's hypothesis, confirmed, but a minor factor** |
| inertia | **2.00×** | the h_w channel is `rwa_I_w·ω_w` with `rwa_I_w = 0.01 kg·m²`, while each MJCF wheel has spin inertia 0.01 **plus** joint `armature="0.01"`: effective 0.02 |
| residual | 1.024 | 2.668 applied vs 2.605 = 2 × 1.302; consistent with the wheel joint damping `1e-4` |

Our run does not reproduce the memo's absolute numbers (3.357 vs 2.884 logged;
the `c3_3` run used that branch's C2.x-instrumented code), but the three factors
multiply to the order of its 34× gap (21.8 × 1.26 × 2.0 ≈ 55).

## 4. What this means — for Idriss, not changed here

1. **The wheel-momentum channel is half the wheel's momentum.** `h_w = 0.01·ω_w`
   everywhere — sensor, log, NMPC conservation box, QP box, AOCS desaturation —
   while the plant integrates wheel torque into a 0.02 kg·m² rotor. If the
   armature is meant as rotor inertia (gear 1), the physical stored momentum is
   **2 × the channel**: the canonical `h_w` peak 4.10 N·m·s (82 % of the ±5 box)
   would be ~8.2 N·m·s physically. If the armature is only a numerical stabiliser
   of the MJCF, the model is still inconsistent with the plant it drives. Either
   way the storage-margin claim (paper §V-C, C4 "82 % of the box") rests on this
   channel and needs a decision.
2. **Open, not measured here:** whether `subtree_angmom` (the Fig. 3
   conservation quantity) includes the armature's momentum — MuJoCo's subtree
   momentum is body-based, armature is not a body.
3. **The logged τ_w statistics are biased** (sampling factor): the paper's τ_w
   figures come from `qs=9`. Uniform 100 Hz statistics are now available.
4. The memo's H1 test (10 Hz vs 5 Hz) could not see the kick, both cadences
   sampling `qs=9`; its "armature eliminated" conclusion should be re-read against
   §3 — the per-step slope is exactly 0.5.

No code or parameter changed: before submission the paper changes are text only
(decision of 2026-09-29).
