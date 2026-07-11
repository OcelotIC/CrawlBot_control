# Phase ABL-HDOT-2 · STEP 0 (STOP-GATE 1) — raw logs ARE the committed runs

Before extracting `λ_qp`, prove the gitignored raw logs are bit-identical to the committed
canonical runs. Method: for each committed CSV column with a direct sim_log source (the SAME
mapping `export_figure_data.py:127-267` used to build the CSV), compare the raw log array to the
committed CSV value **tick-for-tick** and report the max abs deviation. Match to CSV write
precision ⇒ same run. Script: `scripts/diag_ablhdot2_identity.py`. READ-ONLY; no `crawlbot/`
change, no new sim. (Idriss lifted the no-gitignored-logs rule for this extraction only.)

## Sources compared
| run | gitignored log | committed CSV |
|---|---|---|
| **C** | `results/figC_qpcond/sim_log.json` (1080 ticks) | `runfix_traversal.csv` @`5ab2c91` (1080) |
| **U** | `results/figU_rateoff/sim_log.json` (1112 ticks) | `runU_rateoff_traversal.csv` @`be76c9c` (1112) |

Column→key mapping used (from the export): `tauw_*`←`tau_w`, `hw_*`←`hw_physical`,
`theta_s_*`←`struct_euler_deg`, `rcom_*`←`r_com`, `torso_pos_*`←`p_torso`,
`torso_ori_err_deg`←`e_torso_ori`, `swing_dist_m`←`d_grip_swing`, `step_index`←`step_idx`,
`phase`←`phase_per_tick(sl)`.

## Identity table — max |deviation| per quantity (tick-for-tick)
| quantity | C max\|dev\| | U max\|dev\| | verdict |
|---|---|---|---|
| tick count | 1080 = 1080 | 1112 = 1112 | MATCH |
| tauw_x / y / z | 4.92e-7 / 4.96e-7 / 4.98e-7 | 4.87e-7 / 4.98e-7 / 5.00e-7 | ok |
| hw_x / y / z | 4.98e-8 / 4.99e-7 / 5.00e-7 | 4.99e-8 / 4.96e-7 / 4.98e-7 | ok |
| theta_s_x / y / z | 5.00e-7 / 5.00e-7 / 4.99e-7 | 5.00e-7 / 4.99e-7 / 5.00e-7 | ok |
| rcom_x / y / z | 5.00e-7 / 5.00e-7 / 5.00e-7 | 5.00e-7 / 5.00e-7 / 5.00e-7 | ok |
| torso_pos_x / y / z | 4.99e-7 / 4.99e-7 / 5.00e-7 | 5.00e-7 / 5.00e-7 / 5.00e-7 | ok |
| torso_ori_err_deg | 5.00e-7 | 5.00e-7 | ok |
| swing_dist_m | 4.99e-7 | 5.00e-7 | ok |
| step_index (int) | 0 mismatched ticks | 0 | ok |
| phase (string) | 0 mismatched ticks | 0 | ok |

**All deviations ≈ 5×10⁻⁷ = the CSV write-rounding floor** (`%.6f`→±5e-7 abs, `%.6e`→±rel 5e-7),
not a dynamics difference (which would be ≥1e-3). `step_index` and `phase` match exactly.

## Verdict
**BOTH RUNS IDENTITY-CONFIRMED.** The gitignored logs are the committed canonical runs. Their
`λ_qp` is therefore the realized contact wrench of the committed C/U traversals — safe to extract
and commit (Step 1) and reduce into the continuous realized Ḣ_s (Step 2). **STOP-GATE 1 — awaiting
Idriss GO before the `λ_qp` data commit.**
