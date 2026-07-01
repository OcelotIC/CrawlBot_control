# Phase QP-cond — QP conditioning + command-smoothness on canonical C (h_max=5)

Pure measurement (DIAGNOSTICS ONLY, ZERO `crawlbot/` change). Canonical config unchanged
(α_torso=24000, α_mom=5000, w_hw_slack=1e4). C's raw sim_log is ephemeral, so it was rerun
instrumented — the run reproduces committed C **bit-for-bit** (docks [4.94,4.41,4.88,4.42,4.76,4.92]).
Data: `qp_cond_raw.json` (584 cond-samples = every 10th of 5832 QP solves; 528 NMPC solves),
`qp_cond_summary.json`, `scripts/diag_qp_conditioning.py`, `scripts/analyze_qp_conditioning.py`.

**Bottom line (measurement only; the causality / reweight call is cross-check + the h_max=6
counterfactual, a separate round):** the whole-body QP is **moderately** conditioned (cond(H) ~1–4 ×10⁶),
solves **cleanly** (584/584 "Successful return.", 0 failures, no iteration-limit / near-active
issues), and does **not** spike at support switches. The α=24000 torso block is **not** the sole
driver (removing it worsens the tail). The fed-back momentum signal Ḣ_s tracks the smooth plan with
**no** high-frequency contamination; the k+1 model–plant residual is small (~0.2 Nms) and only weakly
correlated (ρ=0.147) with τ_q roughness. The one clear discontinuity is τ_q at support switches (81×),
i.e. a contact-change jump, not continuous chatter.

## TASK A — QP cost-Hessian conditioning
`H = Σ Aᵢᵀ Wᵢ Aᵢ + reg·I` (hierarchical_qp.py:271, weighted mode, weight_ratio=1, reg=1e-6).

| cond(H) | median | p90 | max | n |
|---|---|---|---|---|
| **ALL** | 3.51 ×10⁶ | 4.00 ×10⁶ | 4.05 ×10⁶ | 584 |
| **SS** | 3.85 ×10⁶ | 4.02 ×10⁶ | 4.05 ×10⁶ | 328 |
| **DS** (settle) | 1.00 ×10⁶ | 1.00 ×10⁶ | 1.00 ×10⁶ | 256 |

- **Magnitude:** ~10⁶ — moderate, not catastrophic (nowhere near 10¹²⁺ where the QP would be numerically
  unusable). DS is pinned at ≈1.00 ×10⁶ (regularization-floor dominated in the reduced settle stack).
- **Torso block (α=24000):** removing the max-weight (torso) block moves the **median** cond
  3.51 → 2.20 ×10⁶ (1.6× lower) **but blows up the tail** (p90 208 ×10⁶, max 1.0 ×10⁹). So the torso task
  *regularizes* the tail (constrains otherwise-rank-deficient directions); the ~10⁶ conditioning is a
  property of the **full stack**, not simply "driven by α=24000."
- **Dominant task** (by weight and by block spectral-norm): **SS → torso_pose** (328/328);
  **DS → hw_slack** (256/256, torso task off in settle mode).
- **Solver status:** `{'Successful return.': 584}` — **0 not-success**, no `RET_MAX_NWSR_REACHED`, no
  near-active-slack degeneracy. The QP solves cleanly at this conditioning.
- **Support-switch spike?** cond(H) at DS↔SS switch samples = 2.31 ×10⁶ vs overall 3.51 ×10⁶ → **0.66×**
  (lower, not a spike — switch samples include the ~1×10⁶ DS side). **No conditioning spike at switches.**

## TASK B — realized-command smoothness + feedback contamination
### B1 — τ_q (arm-joint torque) roughness, successive-tick |Δτ|
| phase | RMS(Δτ) | total variation / joint | max\|Δτ\| |
|---|---|---|---|
| SS | 0.203 | 23.9 | 2.70 |
| DS | 0.094 | 16.5 | 2.03 |

SS is ~2× rougher than DS. **At support switches, |Δτ|_max median = 0.730 vs typical-tick median 0.009 —
an 81× jump.** This is a genuine command discontinuity at the DS↔SS contact change (not continuous
chatter within a phase).

### B2 — PLANNED Ḣ_s (NMPC) vs REALIZED Ḣ_s (wheel τ_w), SS ticks, per axis [x,y,z]
- **RMS gap:** [1.15, 1.08, 1.84] Nm (a ~1–1.8 Nm offset on the ±5 Nm scale).
- **HF content (successive-diff RMS):** planned [0.154, 0.251, 0.578], realized [0.118, 0.262, 0.508],
  **ratio [0.8, 1.0, 0.9]** — the realized signal is **not** rougher than the plan. So Ḣ_s carries **no
  high-frequency content absent from the smooth plan**; the gap is a low-frequency tracking offset, not
  chatter. (τ_w is the AOCS wheel command, clamped ±5, decoupled from the QP's arm τ_q.)

### B3 — k+1 feedback residual (NMPC predicted t+dt vs measured state at the next NMPC tick)
| component | RMS | max |
|---|---|---|
| \|Δr_com\| | 0.631 mm | 3.61 mm |
| \|Δv_com\| | 2.76 mm/s | 10.3 mm/s |
| \|ΔL_com\| | 0.198 Nms | 0.713 Nms |

Small model–plant mismatch (0.6 mm on the 5 mm dock scale; 0.2 Nms on the 5 Nms scale). **Correlation of
the |ΔL| residual with τ_q roughness: Pearson ρ = 0.147** — weak; the residual is not strongly driven by
τ_q roughness. (n=528 NMPC solves.)

## What this does and does not show
- **Does NOT support** "the QP is severely ill-conditioned and injects HF chatter into the replanned
  state": cond(H) is ~10⁶ (moderate), all solves succeed, Ḣ_s has no HF contamination, the k+1 residual
  is small and only weakly tied to τ_q roughness.
- **Does show:** a real τ_q discontinuity at support switches (81×, a contact-change jump) and a
  low-frequency ~1.5 Nm planned-vs-realized Ḣ_s tracking gap.
- The α=24000 torso weight is not a clear villain here (it stabilizes the cond tail). Whether reweighting
  would help — and whether conditioning worsens at higher speed — needs the **h_max=6 counterfactual**
  (separate round) and is the cross-check's call, not this measurement's.

No config change, no reweighting, no smoothing term, no paper text. Canonical h_max=5 results are
unaffected. Raw sim dump (`figC_qpcond/`) gitignored (regenerable). Task 3 remains gated.
