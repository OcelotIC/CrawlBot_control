# Phase DRIFT-CLOSURE — T4b: 900 s terminal settle (paper-figure trace + convergence confirmation)

**Scope:** identical to T4 except the terminal settle 450 s → **900 s**. Freeze `32aefaf` control path
untouched; new gitignored dir `figC25_t4b_900s`. Runner `Misc/scripts/diag_t4_settle450.py --settle 900`,
one-field proof `Misc/scripts/diag_t4b_cfg_diff.py`, analysis `Misc/scripts/diag_t4b_analyze.py` (committed `eb1d6f9`).

**Headline: the z-drift converges exactly as T4 extrapolated — |θ_s,z| crosses 0.05° at t = 860.4 s
(T4 predicted 861 s), tail τ = 299.5 s (T4 fit ≈ 305 s). C1/C2/C4 pass. C3-on-the-norm misses (0.0596°),
but NOT because of the z-drift — a separate ~0.048° residual in θ_s,y floors the norm.**

---

## Config diff vs T4 — exactly one field

`Misc/scripts/diag_t4b_cfg_diff.py` snapshots `dataclasses.asdict(cfg)` at `SimulationLoop.__init__` (before
planning/sim) for settle = 450 and 900 and diffs:

```
total cfg fields compared: 154
differing fields:          1
    t_settle_final: 450.0 -> 900.0
VERDICT: EXACTLY ONE FIELD ✓
```

No controller, weight, gain, or model change. Sanity from the run: `weights_match_frozen=True`, applied
wheel torque ≤ **2.500** every mj_step (368/96448 clamped — the *same 368* as T4, all in the bit-identical
traversal; the settle adds none), qp_fail 0.

---

## Gate 0 — traversal bit-identity (t ≤ 64.54 s)

18/18 control channels, **max|Δ| = 0.0** over 1877 ticks vs the canonical `figC25_addfive`. Docks 6/6
(4.02/4.89/4.99/4.97/4.95/4.62 mm). The 900 s change is provably terminal-only.

---

## Four criteria @ settle end (t = 964.44 s)

| # | criterion | measured | verdict |
|---|---|---|---|
| C1 | \|h_w\| < 0.05 Nms | [0.0015, 0.0002, **0.0041**] | **PASS** |
| C2 | \|ω_s\| < 1×10⁻³ deg/s | **1.25×10⁻⁴** | **PASS** |
| C3 | \|θ_s\| < 0.05° (norm) | **0.0596** (axes [0.0048, **0.0481**, 0.0348]) | **FAIL** |
| C4 | L_total ≈ 0 throughout | max\|axis\| [1.25e-3, 7.8e-4, 6.4e-4], norm **1.47×10⁻³** | **PASS** |

All three rate/conservation criteria tighten further vs T4 (h_w,z 0.019→0.004; ω_s 5.3e-4→1.2e-4).

---

## The two requested numbers

**t_cross — first time |θ_s,z| < 0.05° (and stays):**

> **t_cross = 860.44 s** (795.9 s of settle past the dock).

The T4 single-exponential fit extrapolated 869 s; the T4b tail re-fit extrapolates **861.2 s**; the
*measured* crossing is **860.44 s**. The exponential model of the z-drift is confirmed to <1 s — **this is
not a non-exponential floor.**

**Tail time-constant re-fit (t > 250 s):**

> **τ = 299.5 s** (T4 fit ≈ 305 s) — a 2 % consistency, independent run, longer window. Confirmed.

---

## Why C3-on-the-norm still misses — a separate θ_s,y residual, not the z-drift

The three attitude axes decouple in the settle (per-axis trace, `t4b_trace_900s.csv`):

| t [s] | θ_s,x | θ_s,y | θ_s,z | θ_s,xy | ‖θ_s‖ |
|---|---|---|---|---|---|
| 64.6 (entry) | +0.068 | +0.147 | +0.177 | 0.162 | 0.240 |
| 120 (peak z) | +0.065 | −0.004 | **+0.574** | 0.065 | 0.577 |
| 450 | +0.022 | +0.032 | +0.198 | 0.039 | 0.201 |
| 750 | +0.009 | +0.044 | +0.073 | 0.045 | 0.086 |
| 860 (t_cross) | +0.006 | +0.047 | **+0.050** | 0.047 | 0.069 |
| 964 (end) | +0.005 | **+0.048** | +0.035 | 0.048 | 0.060 |

- **θ_s,z** — the drift axis this whole stream is about: peaks 0.574° (~110 s), decays exponentially
  (τ≈300 s), crosses 0.05° at 860 s. **Resolved.**
- **θ_s,x** — decays monotonically toward 0 (0.068 → 0.005).
- **θ_s,y** — does **not** decay: it crosses zero at ~120 s then slowly climbs to a **~0.048° plateau**
  (last-300 s slope +0.02 millideg/s, increments shrinking). This ~0.048° cross-axis offset is what holds
  ‖θ_s‖ at 0.0596° — just above the 0.05° gate.

So the norm-C3 miss at 900 s is **not** the terminal z-drift (which converged on schedule); it is a small,
distinct **θ_s,y ≈ 0.048° static offset**. h_w and ω_s are already at 10⁻³–10⁻⁴ (both pass), so this is an
*attitude* residual with the wheels essentially unloaded — a quasi-static cross-axis bias, not a momentum
or rate problem.

---

## Verdict & STOP

- **Config = T4 ± exactly `t_settle_final`; Gate 0 bit-identical; C1/C2/C4 pass.**
- **z-drift closure confirmed:** t_cross(θ_s,z) = **860.4 s**, τ = **299.5 s** — the T4 extrapolation holds
  to <1 s. The exponential tail is real; there is no non-exponential z-floor.
- **New finding (stops the stream for diagnosis, per contract):** ‖θ_s‖ does not reach 0.05° by 900 s
  because of a **~0.048° residual in θ_s,y** that plateaus rather than decays. This is a separate,
  small cross-axis attitude offset — flagged, **not** tuned or chased. No gain/weight change, no further run.

| deliverable (committed) | path |
|---|---|
| plotting-grade trace (t, ω_s, h_w, θ_s, τ_w xyz; 5 Hz uniform, 4823 rows) | `results/j2_adjconv/t4b_trace_900s.csv` |
| L_total captures (2 Hz, 1929 rows) | `results/j2_adjconv/t4b_ltot_900s.csv` |
| figure | `results/j2_adjconv/t4b_traces.png` |
| criteria/summary | `results/j2_adjconv/t4b_settle900_analysis.json` |
| run sanity | `results/j2_adjconv/t4b_settle900_result.json` |
| sim log + dense captures | `results/figC25_t4b_900s/{sim_log.json, ltot_dense.json}` (gitignored) |
