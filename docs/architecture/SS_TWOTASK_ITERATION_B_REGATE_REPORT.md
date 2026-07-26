# Two-task Iteration B — AOCS verification, C2 tightening under legacy_pid_numerical, re-gate

Single targeted tuning iteration under the paper's AOCS (`legacy_pid_numerical`), preceded by a
read-only AOCS-law verification, followed by a 5-step re-gate with the corrected C5. Tooling:
`scripts/run_iterB_screen.sh`, `screen_iterB.py`, `run_iterB_gate.sh`, `gate_phase3.py`
(corrected C5 + dir override), `plot_phase3.py`.

---

## ✅ GATE VERDICT: **CLEAN SIX-CRITERIA PASS** — recommended working point **5k : 24k, Kp3**

`ss_alpha_mom=5000, alpha_torso_pose=24000, ss_Kp_torso=3 (ss_Kd_torso=2.5), ss_alpha_ee=3000,
ss_alpha_posture=20`, under `--aocs_mode legacy_pid_numerical --K_omega 50`. All six criteria pass
cleanly (C2 a clean pass, not justified-equivalence), vs the `5bca42c` baseline:

| # | criterion | limit | 24k Kp3 (recommended) | 24k Kp4 (alt) | prior 20k Kp3 | baseline |
|---|-----------|-------|-----------------------|---------------|---------------|----------|
| 1 | docking 5/5, <5 mm | 5 mm gate | ✅ [4.94,4.51,4.93,4.62,4.89] | ✅ [4.77,4.45,4.82,4.95,4.73] | ✅ | [1.86,4.94,4.96,4.77,5.0] |
| 2 | torso pos peak | ≤17.6 mm | ✅ **16.5** | ✅ 13.4 | ❌ 18.8 | 17.6 |
| 2 | torso ori RMS | ≤0.68° | ✅ 0.092 | ✅ 0.070 | ✅ 0.106 | 0.68 |
| 3 | envelope ‖Ḣ_s‖∞ | ≤5 N·m | ✅ 5.0 (z binds) | ✅ 5.0 | ✅ | 5.0 |
| 4 | θ_s peak / final | ≤1.9 / 1.65° | ✅ **0.59 / 0.15** | ✅ 0.59 / 0.11 | ✅ 1.58/1.31 | 1.88 / 1.62 |
| 5 | h_w peak ∞-norm | **≤4.5** N·m·s | ✅ **4.405** (0.095 margin) | ✅ 4.462 (0.038) | ✅ 4.37 | 3.38 |
| 6 | flag-OFF Δ=0; test | byte-identical; 8/8 | ✅ Δ=0; 8/8 | ✅ Δ=0; 8/8 | ✅ | — |

**24k Kp3 is recommended over 24k Kp4**: both pass, but Kp3 keeps more C5 headroom (0.095 vs
0.038 N·m·s) and lower B15 (12.1% vs 13.1%) while still passing C2 cleanly. It tightens C2 the
"right" way — via **weight** (the C2 dial): raising α_torso_pose 20k→24k took the torso pos peak
18.8→**16.5 mm** (clean pass) while raising h_w only +0.035 (4.37→4.405). This is exactly the
Phase-2.2 decoupling (weight = arrival dial, Kp = h_w dial): C2 was fixed with minimal C5 cost by
moving the weight knob, not the gain knob. C4 attitude stays ~3× better than baseline (0.59° vs
1.88°) — the advantage was NOT traded away.

**Per the agreed plan, this working point becomes the candidate for paper regeneration. The
regeneration track (paper-vs-code audit, §VII regen, VI-C/VI-D + Fig.1 rewrite, F-ABL) remains
GATED and is the review session's call — NOT begun here.** Config-default promotion is deferred to
that review (it must be decided together with the AOCS-default alignment — see §1/§5).

---

## 0. AOCS control-law verification (READ-ONLY) — PD, **no integral**; §VI-E correct

`legacy_pid_numerical` (`crawlbot/aocs/force_estimator.py:compute_aocs_command_legacy_pid_numerical`,
lines 514-595), dispatched at `sim_loop.py:3037`:
```
τ_w = ff_term + K_hw·hw_error + (K_θ·θ_s + K_ω·ω_s + K_d·ω̇_s_num)        [:594, :577]
  ff_term  = −Ḣ_s_est − r_com×m·v̇_com_est        (feedforward)            [:585-588]
  hw_error = clip(hw,hw_min,hw_max) − hw           (desaturation, soft-clamp) [:575]
  ω̇_s_num  = (ω_s − ω_s_prev)/dt                   (numerical derivative)     [:576]
```
Map to §VI-E.1 `τ_w = τ_ff − Kθ·eθ − Kω·ωs − Kd·ω̇s − Khw·hw` (every VI-E.1 term present; none
extra; signs per the code's Newton-Euler derivation, docstring :545-548):

| VI-E.1 term | code term | role |
|---|---|---|
| τ_ff | ff_term (−Ḣ_s_est − orbital) | feedforward |
| Kθ·eθ | K_θ·θ_s | **P** on attitude |
| Kω·ωs | K_ω·ω_s | rate damping (1st deriv) |
| Kd·ω̇s | K_d·ω̇_s_num | accel damping (2nd deriv, numerical) |
| Khw·hw | K_hw·hw_error | desaturation |

**No integral term.** The function is stateless w.r.t. attitude error — there is no ∫eθ dt
accumulator anywhere in the AOCS path (verified by search). The law is **PD (+ accel) + feedforward
+ desaturation**, matching §VI-E.1 exactly. **"pid" in the mode name is a misnomer** ("numerical" =
the numerical ω̇ estimate). **§VI-E is correct as written (PD); the project invariant "AOCS is PD +
feedforward + desaturation — never PID" is upheld.** Only the code *identifier*
(`legacy_pid_numerical`) is misleading — recommend renaming/aliasing to "pd" for the regenerated
codebase. This bears on the paper but does NOT affect the tuning (the law is fixed either way).

## 1. Step 1 — single-step weight×gain screen (step 0, legacy_pid_numerical)

| α_torso_pose | Kp | C2 pos peak [mm] | ori RMS [°] | h_w peak | θ_s peak [°] | dock [mm] | arrival % |
|---|---|---|---|---|---|---|---|
| 16k | 3 | 13.71 | 0.024 | 2.33 | 0.160 | 4.98 | 9.69 |
| 20k | 3 | 11.16 | 0.023 | 2.35 | 0.159 | 4.91 | 8.15 (prior WP) |
| **24k** | **3** | **9.44** | 0.023 | 2.36 | 0.159 | 4.94 | 7.08 |
| 16k | 4 | 10.88 | 0.020 | 2.38 | 0.160 | 4.87 | 7.77 |
| 20k | 4 | 8.87 | 0.019 | 2.39 | 0.159 | 4.90 | 6.60 |
| 24k | 4 | 7.55 | 0.019 | 2.39 | 0.158 | 4.77 | 5.60 |

C2 tightens monotonically with weight and Kp; h_w is undifferentiated on the mild step 0 (all
~2.3-2.4 ≪ 4.5) — so the screen narrows on C2 (and confirms all dock), and the **5-step decides C5**
(only the binding swings 2,4 stress the wheels). Candidates carried forward: the two max-weight
points 24k Kp3 (weight-driven C2, low-h_w Kp) and 24k Kp4 (tightest C2, C5-ceiling bracket).

## 2. Step 2 — 5-step re-gate (full table above). Two-regime + B15 (24k Kp3):

- **Two-regime torso arrival** (characterisation): non-binding swings 0.4%, binding swings mean
  5.1% (step 0/2/4 — the constraint signature, expected; non-binding 1,3 reach the planned pose).
- **B15 (100 Hz QP-rate τ_w-sat):** aggregate **12.1%**, per-step {0: 5.8%, 1: 0%, 2: 40.0%,
  3: 0%, 4: 47.5%} — concentrated in the binding swings (2,4), not pervasive. (24k Kp4: agg 13.1%.)
- **Envelope:** realized ‖Ḣ_s‖∞_SS = 5.0 (z binds, ≤5); planned constraint unchanged.

## 3. Mandatory checks

- **Bit-identical-OFF re-confirmed at this iteration:** the OFF path is **untouched** — no
  `crawlbot/` change in iteration B (scripts + docs only). The flag-OFF 5-step is byte-identical to
  the Phase-1 baseline dcda974 (Δ=0, `Misc/runs/phase3_off/OFF_RESULT.txt`), and **test_reworked_qp
  8/8** (re-run this session). C6 holds for both candidates.
- τ_w and h_w logged at 100 Hz in SS; h_w reported in per-axis ∞-norm (peak 4.405 N·m·s for 24k
  Kp3). 2-norm of the same peak instant ≈ 4.43 (z-dominated; the ∞-norm is the binding measure).
- Run metadata (HEAD, clean/dirty, full flags) in `Misc/runs/p3b_gate_w24000_kp3/PHASE3_METADATA.txt`.

## 4. Plots (`Misc/runs/p3b_gate_w24000_kp3/phase3_plots/`, baseline grey/dashed, SS shaded)

`1_torso_tracking` (C2: ori ≪ baseline, pos peak 16.5 ≤17.6), `2_swing_ee`, `3_attitude`
(C4 0.59° ≪ 1.88°), `4_Hdot_tauw` (C3 + τ_w@100Hz, in-swing), `5_hw` (C5 z-peak 4.405 ≤4.5 N·m·s),
`6_dock_margins` (C1 5/5), `7_two_regime_arrival` (0.4% vs 5.1%).

## 5. Outcome and what it triggers (reported, NOT executed)

A clean six-criteria pass was achieved. Per the agreed plan:
- **24k Kp3** is the candidate working point for paper regeneration.
- The regeneration track is **GATED on the review's go** and is **not begun** (no audit, no §VII
  regen, no VI-C/VI-D/Fig.1 rewrite, no F-ABL).
- **Config-default promotion deferred to review:** the working point is validated as the *pair*
  (24k Kp3 + legacy_pid_numerical). The config default AOCS is currently `legacy_corrected`;
  promoting the SS weights without aligning the AOCS default would be a half-promotion of an
  un-gated combination. The promotion + AOCS-default decision belongs to the regeneration go (no
  architecture change made here). The working point is reproducible via the documented flags.
- §VI-E correction is NOT needed (it is correct as PD); only the code identifier "pid" should be
  renamed in the regenerated codebase (§0).
