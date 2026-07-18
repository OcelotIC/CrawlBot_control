# Phase DRIFT-CLOSURE — SUB-PHASE B (T4): extended 450 s terminal settle

**Scope:** re-run of the frozen 2.5 **managed** canonical with the trailing DS settle extended
20 s → 450 s (`settle_seconds` → `cfg.t_settle_final`, `sim_loop.py:2482`, the terminal-only
"end of gait" branch). Freeze `32aefaf` control path untouched; new gitignored output dir
`figC25_t4_450s` (canonical `figC25_addfive` not touched). Runner `scripts/diag_t4_settle450.py`,
analysis `scripts/diag_t4_analyze.py` (both committed `0fbbe58`).

**Result: 3 of 4 criteria PASS + traversal bit-identical; C3 (θ_s < 0.05°) FAILS — θ_s is still
decaying at 450 s (0.16°), a slow-asymptote miss, not a static floor.** Per the phase contract I
**report + STOP, no tuning, no retry.**

---

## Gate 0 — traversal identity (t ≤ 64.54 s)

Extending the settle must not perturb the traversal. Comparing all control-state channels tick-for-tick
against the canonical `figC25_addfive` over the 1877 pre-settle ticks:

| channel | max\|Δ\| | channel | max\|Δ\| |
|---|---|---|---|
| t, struct_pos, struct_quat | **0.0** | struct_euler_deg, omega_s | **0.0** |
| hw_physical, tau_w | **0.0** | lambda_qp, lambda_ref | **0.0** |
| q_ee, e_com, r_com, v_com | **0.0** | e_torso_pos/ori, p_torso | **0.0** |
| d_grip_swing, step_idx | **0.0** | — | — |

**BIT-IDENTICAL ✓** (18/18 channels, all 1877 ticks). This simultaneously proves (a) `settle_seconds`
is terminal-only, (b) build determinism holds on this container vs the Jul-14 canonical, and (c) the
intervening logging-only commits (torso-export `b37b528`, T2 exporter) are control-neutral. At-weld
docks reproduce **6/6: 4.02 / 4.89 / 4.99 / 4.97 / 4.95 / 4.62 mm**. Sanity: `weights_match_frozen=True`,
applied wheel torque ≤ **2.500** every mj_step (368/51448 steps at the plant cap). Feasibility: the
extended 450 s settle (4500 DS_terminal ticks) is **clean — 0 qp_fail, 0 nmpc_fail**; the run's total
`nmpc_fail = 1368` is **byte-identical to the canonical** and lies entirely in the DS_interstep traversal
ticks (the NMPC is bypassed in inter-step DS — a benign logging convention, not solve failures; the
traversal is bit-identical so the count matches exactly). `qp_fail = 0` throughout.

---

## The four pass criteria (settle end, t = 514.44 s)

| # | criterion | measured | verdict |
|---|---|---|---|
| **C1** | \|h_w\| < 0.05 Nms, all axes | [0.00231, 0.00118, **0.01947**] | **PASS** |
| **C2** | \|ω_s\| < 1×10⁻³ deg/s | norm **5.35×10⁻⁴** (axes [5.8e-5, 5.2e-5, 5.3e-4]) | **PASS** |
| **C3** | \|θ_s\| < 0.05° | norm **0.1649** (axes [0.018, 0.036, **0.160**]) | **FAIL** |
| **C4** | L_total ≈ 0 throughout | max\|L\|_axis [1.25e-3, 5.6e-4, 5.5e-4], norm **1.47×10⁻³** (450 settle captures) | **PASS** |

Wheel **re-load transient — visible** (part of the C3 expectation that *did* hold): peak \|h_w\|_axis over
the settle = [0.043, 0.509, **1.255**] Nms @ t≈66.5 s (the z-wheel holds the −1.28 arrest load), then all
three axes unload to ≈0 by ~120 s and stay < 0.02 for the remaining 400 s.

---

## Why C3 misses — slow asymptote, not a floor

θ_s,z **is** converging monotonically; 450 s is simply too short at the tightened cap.

| t [s] | θ_s,z [deg] | \|ω_s,z\| [deg/s] | h_w,z [Nms] |
|---|---|---|---|
| 64.7 (entry) | +0.181 | 3.9×10⁻² | −1.227 |
| ~110 (peak θ) | **+0.590** | ~1×10⁻³ | ≈0 (crossing) |
| 200 | +0.448 | 1.5×10⁻³ | +0.055 |
| 300 | +0.323 | 1.1×10⁻³ | +0.040 |
| 400 | +0.233 | 7.7×10⁻⁴ | +0.028 |
| 514 (end) | **+0.160** | 5.3×10⁻⁴ | +0.019 |

Mechanism (see `t4_traces.png`): at entry the wheels carry −1.28 Nms of z-momentum; over the first ~60 s
they arrest the residual spin (ω_s,z → ~0 by ~105 s), during which the structure **over-rotates** to a
θ_s,z peak of **0.59°**. h_w,z then crosses zero and holds a small **+0.02–0.05 Nms** residual that
counter-rotates the structure back — but that tiny torque against the ~2.18×10³ kg·m² composite inertia
gives a long recovery. A single-exponential fit to the tail (t > 250 s) yields a **time constant ≈ 305 s**;
the last-100 s slope is **−0.62 millideg/s** (0.222° → 0.160°). Extrapolated, \|θ_s,z\| reaches 0.05° at
**≈ 800 s of settle past the dock** — roughly **1.8× the 450 s tested**.

So the rate states (ω_s, h_w) and conservation (L_total) all close well inside 450 s; the **accumulated
attitude offset** bleeds off ~2.5× slower and is the sole criterion that misses. The 0.54° "plateau" seen
at the 20 s canonical cutoff is confirmed as the truncation point of this decay (θ_s at entry+20 s =
0.540°, matching the canonical), **not** a steady state.

---

## Verdict & artifacts

- **Traversal bit-identical (Gate 0 ✓); C1 ✓ C2 ✓ C4 ✓; C3 ✗** (θ_s = 0.165° at 450 s, decaying,
  τ≈305 s, ~800 s needed for 0.05°).
- The result is **honest and physically closed**: momentum is conserved, the wheels unload, the spin
  stops; only the slow attitude-offset recovery exceeds the 450 s window at τ_w,max = 2.5.
- **STOP** per the phase contract — no gain/weight tuning, no longer-settle retry. Awaiting direction.

| artifact | path |
|---|---|
| trace figure | `results/j2_adjconv/t4_traces.png` |
| trace CSV (downsampled) | `results/j2_adjconv/t4_traces.csv` |
| criteria summary | `results/j2_adjconv/t4_settle450_analysis.json` |
| run sanity (weights/cap/clamp) | `results/j2_adjconv/t4_settle450_result.json` |
| sim log + dense L_total captures | `results/figC25_t4_450s/{sim_log.json, ltot_dense.json}` (gitignored) |
| runner / analysis | `scripts/diag_t4_settle450.py`, `scripts/diag_t4_analyze.py` (committed `0fbbe58`) |
