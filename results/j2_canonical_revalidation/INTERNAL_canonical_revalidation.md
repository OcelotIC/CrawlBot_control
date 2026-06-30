# INTERNAL — canonical-config re-validation (J2, corrects the wrong-config run)

Branch `j2/ds-active-rework`. This re-runs the ENTIRE chatter-closure validation on the **confirmed canonical
config**, after the prior validation (PR #27) was found to have been run on a **wrong (simplified) config** that
silently dropped the SS-stack tuning. Every number below is read back from a **committed** artifact (paths given),
not from memory. Where canonical disagrees with the wrong-config number, both are shown and the disagreement is
flagged. **No `crawlbot/` code changed** — this is a re-measurement + two new analysis scripts.

## The error being corrected
The PR-#27 "locked-config" run used `--qp-envelope-exact --aocs_mode legacy_pid_numerical` + the two chatter
flags **only**. It DROPPED the figure config's SS-stack tuning:
`--ss-two-task --ss-alpha-mom 5000 --alpha-torso-pose 24000 --ss-kp-torso 3 --ss-kd-torso 2.5 --K_omega 50`.
The committed `runA_meta.json` even carried `aocs_mode: legacy_pid_numerical` with **no** `ss_*`/`alpha_torso_pose`
keys — the tell. The whole closure validation (POINT 1/A, C1–C5 gate, closure curves, the figure CSV) inherited
that wrong config.

## Canonical config (CONFIRMED = ground truth)
`scripts/run_figdata.sh` `COMMON` (blob `1cbd69d`, the figure-config authority) **+** the two derived chatter
flags. Verified flag-by-flag against `run_figdata.sh`; reproduced verbatim in `runA_meta.json:run_config`:
```
--ss-two-task --ss-alpha-mom 5000 --alpha-torso-pose 24000 --ss-kp-torso 3 --ss-kd-torso 2.5
--aocs_mode legacy_pid_numerical --K_omega 50 --n-steps 5 --qp-envelope-exact
--interstep-settle-alpha-wrench 3 --interstep-settle-epsilon-v 5e-3
```

## Artifacts (single source of truth)
| artifact | path | shape |
|---|---|---|
| paper figure CSV | `results/j2_figdata/runA_traversal.csv` | 905 ticks × 39 cols |
| meta (config + docks) | `results/j2_figdata/runA_meta.json` | 5 dock events, 11 segments |
| raw sim (the fix run) | `results/figA_canon/sim_log.json` | 905 ticks |
| chatter baselines | `results/figA_canon_e0` (α=0), `figA_canon_e1` (α=1), `figA_canon_evb` (ε_v=0) | — |
| closure curves | `results/j2_closure_curves/canonical_curves.csv` | 905 ticks |
| drivers | `scripts/run_canonical_validation.sh`, `scripts/analyze_canonical_metrics.py` | new |

---

## STEP 2 — gates on canonical (all reproduced from the committed artifacts)

| crit | metric | canonical | verdict | wrong-config (PR #27) |
|---|---|---|---|---|
| **C1** | docks [mm] | **[4.94, 4.45, 4.94, 4.65, 4.84]** (b3/a3/b4/a4/b5) | **PASS** 5/5 ≤5 (worst margin **0.06 mm**) | [1.86, 4.79, 4.90, 4.87, 4.67] (margin 0.10) |
| **C2** | torso-pos SS peak / rms | **23.0 / 9.3 mm** (median 7.8) | (tracking, looser) | 12.7 / 4.1 mm |
| **C3** | ‖Ḣ_s‖∞_SS (planned) | **5.000**, 0 ticks>5; full-cycle 5.000, **0/905** over | **PASS** ≤5 | 5.00, 0 over |
| **C4** | struct attitude θ_s peak | **0.578°** (norm) / 0.47° (∞) | **PASS** <5° | ~0.607° |
| **C4′** | torso-ori tracking SS peak | **0.523°** (init offset 5.157°, config-indep.) | **PASS** <5° | 2.42° |
| **C5** | **h_w∞ (hw≡hw_physical)** | **4.930 Nms** (peak hw_z = −4.930) | **FAIL > 4.5** (98.6% of ±5 HW) | 3.86 (PASS) |
| **C6** | default-off byte-identical | config-independent (knobs default 0, `qp_envelope_exact=False`); deep \|Δ\|=0 already confirmed; no code change since | **PASS (unchanged)** | same |

**Headline:** the only gate that FLIPS is **C5**. The wrong (simple) config reported `h_w∞ = 3.86` (PASS). On the
true canonical config the SS-stack momentum weighting (`ss-alpha-mom 5000` + `alpha-torso-pose 24000`) drives more
aggressive torso motion → more reaction-wheel storage → **`h_w∞ = 4.930 Nms`, OVER the 4.5 soft gate** (though still
within the ±5 N·m·s hardware budget, at 98.6%). The wrong-config validation **hid** this. (`hw` and `hw_physical`
are byte-identical on this run, so the verdict is signal-independent.)

### Chatter (POINT 1) — fix transfers to canonical
`scripts/point1_analyze.py` on the three canonical runs (arm-a settles = the chatterers):

| run | arm-a Σf-flip | arm-a ‖Σf‖med [N] | all_clean |
|---|---|---|---|
| e0 (α=0, baseline) | **0.90, 0.98** | 6.90, 6.88 | **False — chatters** |
| e1 (α=1, threshold) | 0.69, 0.61 | 5.30, 3.17 | False — partial |
| **figA_canon (α=3, fix)** | **0.06, 0.03** | 0.93, 0.33 | **True — clean** |

ε=3 (derived from κ_max≈1.3e4 / solver resolution × margin) is necessary and sufficient on the canonical SS stack,
not only the simple config. Envelope full-cycle: max\|Ḣ_s\| = 5.000, **0/905 ticks > 5**.

### Settle exit (POINT A) — ε_v transfers
`figA_canon` (ε_v=5 mm/s) vs `figA_canon_evb` (ε_v=0 → falls back to 1 mm/s default):

| settle | ε_v=5 mm/s ticks | ε_v→1 mm/s ticks | dock [mm] (identical) | clean |
|---|---|---|---|---|
| step1/b3 | 129 | 171 | 0.0122 | YES |
| step2/a3 | 50 (plateau) | 50 (plateau) | 0.0088 | YES |
| step3/b4 | 132 | 205 | 0.0065 | YES |
| step4/a4 | 118 | 204 | 0.0092 | YES |

The dock-derived ε_v=5 mm/s shortens each settle by ~40–85 ticks (~0.4–0.85 s) with **identical** dock distances and
**both remain chatter-clean** (arm-a flip 0.06/0.03 vs 0.06/0.02). Closure-curve settle-exit ticks: [170, 304, 472, 670].

### τ_w (closure curves)
Full-run max τ_w = 5.000 N·m (saturates, never sustained-exceeds). Hand-off transient: max tick-to-tick
\|Δτ_w\| in DS = **7.330 N·m·tick⁻¹** (isolated SS→DS phase-transition spikes, not period-2 chatter — consistent
with the PR's "isolated hand-off spikes, not slamming"). Source: `results/j2_closure_curves/canonical_curves.csv`.

### Conservation / terminal
- terminal (last tick t=50.94 s): **‖h_w‖ 0.338 N·m·s** (slope −0.034/s, still falling) · **‖θ_s‖ 0.0864°** (slope −0.0067/s, falling) — system is still relaxing at the fixed 20 s terminal window (C1-open, unchanged).
- Ltot at snapshots: max **2.96e-3 N·m·s** (conserved). Continuous ‖L_com‖ peak **0.782 N·m·s**, median 0.040 (numerical leakage during swings, D1-open).

---

## STEP 3 — config-dependent reconfirmation (canonical vs wrong-config)

| signal | wrong-config | **canonical** | direction | transfers? |
|---|---|---|---|---|
| chatter fix (ε=3 clean / α=0,1 not) | clean | **clean** (0.06/0.03) | same | **YES** (re-verified) |
| settle ε_v (5 mm/s faster, same dock) | holds | **holds** | same | **YES** (re-verified) |
| C3 envelope ≤5, 0 over | 5.0/0 | **5.0/0** | same | YES (exact-box, structural) |
| C5 h_w∞ ≤4.5 | 3.86 PASS | **4.930 FAIL** | **WORSE — gate flips** | **NO — config-dependent** |
| C2 torso-pos SS peak | 12.7 mm | **23.0 mm** | worse | config-dependent |
| CoM ‖e_com‖ med/max | 76.6 / 189.9 mm | **47.1 / 95.1 mm** | better | config-dependent |
| torso drift vs t=0 (struct) peak/final | 4.118° / 1.803° | **0.721° / 0.721°** | much smaller | config-dependent |
| torso drift world peak/final | 4.518° / 1.801° | **0.992° / 0.747°** | much smaller | config-dependent |
| D1 ‖L_com‖ peak | 1.507 | **0.782** | smaller | config-dependent |

**Torso CASE B is structural and UNCHANGED** (`sim_loop.py:1411/1478/1676` SS `R_torso_fixed=R_t0=current`;
`:2099/:2479-2499` DS `set_hold(…,R_now)`). What changes is the **magnitude**: on canonical the cumulative drift is
**0.721° (structure-frame, peak=final)** — far below the wrong-config 1.803°/4.118°. So the per-stance-relative hold
remains a limitation (not a global hold), but the accumulation is much smaller and effectively flat over the 5 steps.

---

## Conclusion
1. **The chatter fix is real and config-robust** — ε=3 and ε_v=5 mm/s both transfer to the canonical config
   (α=0 chatters, α=1 partial, α=3 clean; ε_v shortens settles with identical docks). C3 envelope clean (0/905 over).
2. **C5 FAILS on canonical (h_w∞ = 4.930 > 4.5)** — a genuine, config-dependent finding the wrong-config run hid.
   Still within the ±5 N·m·s hardware budget (98.6%), but over the 4.5 (90%) soft gate. This must be surfaced.
3. **Torso drift (0.72°) and CoM (47/95 mm) are SMALLER (better) than reported**; **torso-pos SS tracking (23 mm)
   is LARGER (worse)**. The VISPA open-items numbers are replaced accordingly.
4. The committed `runA` artifacts (CSV + meta) now carry the canonical config and docks.

The corrected open list (C1 terminal-settle, C2 CoM, C3 torso 6-DoF orientation, D1 leakage, **+ C5 soft-gate
exceedance**) is in `VISPA_OPEN_ITEMS_2026-06.md`. **No merge, no PR opened by tooling.**
