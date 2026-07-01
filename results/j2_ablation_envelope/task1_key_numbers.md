# Task 1 — rate-cap ablation key numbers (envelope ON vs OFF)

Single-variable test: **U = C with `cfg.tau_w_max` raised from 5 to 1e6** (NMPC + pre-planner + QP wheel
**rate** cap removed). Storage box (`h_max_tight`), QP soft box (`hw_max`), AOCS realized-command clamp
(`aocs_tau_w_max`=5) and the physical MJCF `ctrlrange` ±5 are all **unchanged**, so this isolates the rate
constraint's effect on actuator demand.

- C: `results/j2_canonical_revalidation/runfix_traversal.csv`
- U: `results/j2_ablation_envelope/runU_rateoff_traversal.csv`
- rate cap τ_w,max = 5.0 Nm

| metric | C — envelope ON | U — rate cap OFF |
|---|---|---|
| ticks | 1080 | 1112 |
| **PLANNED \|Ḣ_s\| SS peak / axis (Nm)** | x 3.485  y 5.0  z 5.0 | x 3.373  y 4.483  **z 6.27** |
| **planned Ḣ_s SS ticks > 5 Nm** | 0/328 (0.0%) | **41/336 (12.2%)** |
| per-swing > 5 fraction | s0:0% s1:0% s2:0% s3:0% s4:0% s5:0% | s0:22% s1:0% s2:21% s3:0% s4:24% s5:27% |
| realized \|τ_w\|∞ peak (Nm) | 5.0 | 5.0 |
| realized τ_w saturation | 3.7% (40/1080) | **6.5% (72/1112)** |
| θ_s peak-norm (deg) | 0.5913 | 0.5367 |
| θ_s final-norm (deg) | 0.1078 | **0.2785** |
| h_w peak-∞ (Nms) | 4.8855 | 4.4841 |
| docks (mm, n=6) | [4.94, 4.41, 4.88, 4.42, 4.76, 4.92] | [4.94, 4.06, 4.78, 4.47, 4.79, 4.7] |

**Verdict (primary / actuator demand):** the pre-planner's PLANNED wheel-rate demand reaches
**6.27 Nm on z** with the cap OFF — **1.27 Nm (25%) above** the 5 Nm cap — and
**12.2% of SS ticks** ask for more than 5 Nm. With the envelope ON the same demand is
held at exactly **5.00/5.00 Nm (y/z pinned at the cap)** with **0** SS ticks over. The constraint is **binding**, not
slack — the plan genuinely wants more than the wheel can deliver, so the hard rate constraint actively
reshapes the trajectory. Paper claim **validated**.

**Realized actuator:** both runs realize the SAME peak 5.0 Nm — because the AOCS realized-command
clamp (aocs_tau_w_max=5) + physical MJCF ±5 still clip the wheel. Removing the *software* rate cap does not
remove the limit; it pushes enforcement onto the hard rail, raising time-at-saturation from 3.7% to
6.5% of ticks (plan-vs-plant mismatch).

**Secondary / attitude:** θ_s peak is comparable (0.5367° vs 0.5913°), but the rate-off run's
**final** attitude error is 2.6× worse (0.2785° vs 0.1078°).

**h_w:** bounded in BOTH (4.4841 vs 4.8855 Nms) — the storage box is still ON in U, so the rate-cap
ablation does not (and should not) blow up storage. The storage counterfactual is a separate test (Task 3, gated).
