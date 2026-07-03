# Ablation summary — with (C) vs without rate management (U)

Constants (plot reference lines): **rate cap = 5.0 N·m**, **storage box = ±5.0 N·m·s**.
Sources: C = `results/j2_canonical_revalidation/runfix_traversal.csv` @ `5ab2c91`; U = `results/j2_ablation_envelope/runU_rateoff_traversal.csv` @ `be76c9c`;
cross-check `results/j2_ablation_envelope/task1_key_numbers.json` @ `be76c9c`. Every value below is computed from the C/U CSV columns and
cross-checked against task1_key_numbers.json (all matches: True).

| metric | C (with mgmt) | U (rate-off) | source |
|---|---|---|---|
| planned \|Ḣ_s\| SS peak / axis (N·m) | x 3.485 / y 5.0 / z 5.0 | x 3.373 / y 4.483 / z 6.27 | CSV SS/planned ticks |
| planned \|Ḣ_s\| SS ticks > 5 | 0/328 (0.0%) | 41/336 (12.2%) | CSV |
| τ_w saturation (% ticks) | 3.7% (40/1080) | 6.5% (72/1112) | CSV τ_w, sat = max\|τ_w,i\|≥5−1e-3 |
| τ_w peak norm / ∞ (N·m) | 7.17 / 5.0 | 8.059 / 5.0 | CSV |
| h_w peak / axis (N·m·s) | [0.7107, 2.3562, 4.8855] | [0.7134, 2.8313, 4.4841] | CSV (≤5, no overshoot) |
| h_w peak-∞ (N·m·s) | 4.8855 | 4.4841 | CSV |
| θ_s peak geodesic (deg) | 0.5913 | 0.5367 | CSV (NOT a with/without benefit) |
| **θ_s settled geodesic (deg)** | **0.1078** | **0.2785** | CSV (honest discriminator) |
| docks (mm, 6/6 @ 5 mm gate) | [4.94, 4.41, 4.88, 4.42, 4.76, 4.92] | [4.94, 4.06, 4.78, 4.47, 4.79, 4.7] | task1_key_numbers.json |

### Per-step planned \|Ḣ_s\| > 5 (% of that step's SS ticks)
| step | C | U |
|---|---|---|
| 0 | 0.0% | 21.9% |
| 1 | 0.0% | 0.0% |
| 2 | 0.0% | 21.2% |
| 3 | 0.0% | 0.0% |
| 4 | 0.0% | 23.5% |
| 5 | 0.0% | 26.8% |

### Per-step h_w peak / axis (N·m·s) — which steps approach the ±5 box (z-axis; peaks on the SHORT steps 2 & 4, step 4 nearest at 4.885)
| step | C [x,y,z] | U [x,y,z] |
|---|---|---|
| 0 | [0.5804, 1.5857, 2.3586] | [0.5806, 1.5864, 2.3595] |
| 1 | [0.1484, 1.9566, 2.9426] | [0.1456, 1.9541, 2.9892] |
| 2 | [0.6583, 2.1155, 4.004] | [0.688, 2.1349, 3.7332] |
| 3 | [0.175, 2.1962, 3.0085] | [0.1744, 2.205, 3.1658] |
| 4 | [0.7107, 2.2931, 4.8855] | [0.7134, 2.3377, 4.4841] |
| 5 | [0.2223, 2.3562, 3.0272] | [0.1988, 2.8313, 3.7043] |

**Honesty notes:** h_w approaches but never exceeds ±5 (peak C 4.885 @ step 4 / U 4.484 @ step 4, both on
the z-axis of the SHORT steps 2 & 4); no overshoot.
θ_s **peak** is comparable/slightly lower for U — NOT a management benefit; the honest discriminators
are actuator demand (planned |Ḣ_s| > 5 and τ_w saturation, both worse for U) and **settled θ_s**
(U 0.278 vs C 0.108, worse for U).
