# Ablation data manifest — with (C) vs without rate management (U)

Every exported column and summary number maps here to {source file, commit, raw column vs computed +
formula}, so the reviewing side can verify figure-vs-CSV with no ambiguity. **No sim run, no synthesis** —
all values are read/derived from committed artifacts. Generator: `scripts/export_ablation_data.py`.

## Sources (committed)
| id | file | commit | meaning |
|---|---|---|---|
| **C** | `results/j2_canonical_revalidation/runfix_traversal.csv` | `5ab2c91` | with management (canonical: rate cap 5 N·m + storage box ±5 N·m·s ON) |
| **U** | `results/j2_ablation_envelope/runU_rateoff_traversal.csv` | `be76c9c` | without rate management (rate cap OFF, `tau_w_max=1e6`; storage box + AOCS clamp kept) |
| task1 | `results/j2_ablation_envelope/task1_key_numbers.json` | `be76c9c` | committed headline numbers (cross-check) |
| meta C/U | `runfix_meta.json` `5ab2c91` / `runU_rateoff_meta.json` `be76c9c` | dock_events (per-step dock distances) |

## Constants (plot reference lines)
- **rate cap = 5 N·m** (`tau_w_max`; the planned \|Ḣ_s\| envelope) — source: task1_key_numbers.json `tau_w_max_Nm`, and `export_figure_data.py:303`.
- **storage box = ±5 N·m·s** (`h_max`; the h_w envelope).

## Value convention
All floats emitted as `%.6e` (7 significant figures). Raw columns re-emit the parsed source value (identical
to the source to 7 sig figs; full precision in the source CSV). An **empty cell** means the signal is not
defined at that tick (see Ḣ_s split below). Booleans/ints are plain.

## Ḣ_s provenance (why planned & realized are phase-disjoint) — READ THIS
The source CSVs carry a SINGLE tagged `Hdot_s_{xyz}_Nm` column with a `Hdot_s_source` tag
(`export_figure_data.py:316-324`): it is the exact origin-referenced Ḣ_s = Σ_j(r_Cj×f_j + τ_j) computed from
the **PLANNED** wrench (`lambda_ref`) on **SS** ticks and from the **REALIZED** settle wrench (`lambda_qp`) on
**DS** ticks. (The source export intentionally dropped a planned reconstruction during DS — it spuriously rode
7.5 N·m in the terminal DS — so planned Ḣ_s is **SS-only** and realized Ḣ_s in this column is **DS-only**.)
We split that one column faithfully into `Hdot_s_planned_*` (populated on SS, empty on DS) and
`Hdot_s_realized_*` (populated on DS, empty on SS). **The realized reaction-torque rate at EVERY tick is the
wheel command `tau_w` (see the tauw_* columns)** — that is the SS-phase realized counterpart to the planned Ḣ_s.
We do NOT synthesize a realized-SS or planned-DS Ḣ_s.

## DELIVERABLE 1 — time-series columns (`ablation_C_timeseries.csv`, `ablation_U_timeseries.csv`)
| column | raw/computed | source column(s) | formula / note |
|---|---|---|---|
| `t_s` | raw | `t_s` | tick time [s] |
| `phase` | computed | `phase` | `"DS"` if `phase` starts with `DS`, else `"SS"` |
| `phase_raw` | raw | `phase` | `DS_interstep` / `DS_terminal` / `SS` |
| `step_index` | raw | `step_index` | −1 = pre-step; 0..5 = locomotion steps |
| `Hdot_s_planned_{x,y,z}_Nm` | computed (split) | `Hdot_s_{x,y,z}_Nm` | = source value where `Hdot_s_source=='planned'` (SS); empty otherwise. Planned reaction-torque rate. [N·m] |
| `Hdot_s_realized_{x,y,z}_Nm` | computed (split) | `Hdot_s_{x,y,z}_Nm` | = source value where `Hdot_s_source=='realized'` (DS); empty otherwise. Realized settle-wrench rate. **DS-only** — use `tauw_*` for realized rate during SS. [N·m] |
| `hw_{x,y,z}_Nms` | raw | `hw_{x,y,z}_Nms` | reaction-wheel stored momentum [N·m·s] |
| `tauw_{x,y,z}_Nm` | raw | `tauw_{x,y,z}_Nm` | realized wheel torque = realized reaction-torque rate; AOCS-clamped ±5 per axis [N·m] |
| `tauw_norm_Nm` | computed | `tauw_{x,y,z}_Nm` | `sqrt(τ_wx²+τ_wy²+τ_wz²)` [N·m] |
| `tauw_saturated` | computed | `tauw_{x,y,z}_Nm` | `1` if `max_i|τ_w,i| ≥ 5 − 1e-3`, else `0` (per-axis AOCS clamp) |
| `theta_s_{x,y,z}_deg` | raw | `theta_s_{x,y,z}_deg` | structure attitude error, per axis [deg] |
| `theta_s_geodesic_deg` | computed | `theta_s_{x,y,z}_deg` | `sqrt(θx²+θy²+θz²)` (rotation-vector magnitude) [deg] |

Dropped source columns (available in the source CSV, not needed for this figure): `tick`,
`Hdot_s_proxy_*` (robot-CoM-lever proxy, orbital omitted), `Ltot_*`, `rcom_*`, `rref_*`, `torso_pos_*`,
`torso_pos_ref_*`, `torso_ori_err_deg`, `swing_dist_m`, `swing_active`, `swing_arm`.

## DELIVERABLE 2 — summary (`ablation_summary.json`, `ablation_summary.md`)
All computed from the C/U CSV columns above; each cross-checked against `task1_key_numbers.json` (@ `be76c9c`)
— **all six checks match for both runs** (`crosscheck_csv_vs_task1` in the JSON).
| summary number | derivation (over the run's ticks) | task1 cross-check key |
|---|---|---|
| planned \|Ḣ_s\| SS peak / axis | `max` of `|Hdot_s_planned_axis|` over SS ticks | `planned_Hdot_s_SS_peak_axis_Nm` |
| planned SS ticks > 5 (count/total/pct) | fraction of SS ticks with `max_axis|Hdot_s_planned| > 5` | `planned_Hdot_s_SS_over_cap` |
| per-step planned > 5 (pct) | same, grouped by `step_index` | `planned_Hdot_s_per_swing_over_cap_pct` |
| τ_w saturation (pct, count) | fraction of ALL ticks with `tauw_saturated==1` | `realized_tauw_saturation` |
| τ_w peak norm / peak-∞ | `max tauw_norm` / `max max_axis|tauw|` | `realized_tauw_peak_inf_Nm` (∞) |
| h_w peak / axis, peak-∞ | `max |hw_axis|` over all ticks | `hw_peak_inf_Nms` (∞) |
| h_w peak / step / axis | `max |hw_axis|` grouped by `step_index` | — (per-step, CSV only) |
| θ_s peak geodesic | `max theta_s_geodesic` | `theta_s_peak_norm_deg` |
| θ_s settled geodesic | `theta_s_geodesic` at the last tick | `theta_s_final_norm_deg` |
| docks (6, mm) | verbatim | `docks_mm` (also `*_meta.json` dock_events) |

## Honesty notes (carried in the summary)
- **h_w never exceeds ±5** (peak C 4.885 / U 4.484 N·m·s, both z-axis of the SHORT steps 2 & 4). It
  *approaches* the box; it does not overshoot. Exported as-is.
- **θ_s peak is NOT a with/without benefit** (U 0.537 ≤ C 0.591 deg). The honest discriminators are
  **actuator demand** (planned \|Ḣ_s\| > 5: 12.2% of U SS ticks vs 0% for C; τ_w saturation 6.5% vs 3.7%) and
  **settled θ_s** (U 0.278 vs C 0.108 deg — worse without management). Both exported plainly.

## Outputs (this commit)
`ablation_data/ablation_C_timeseries.csv` (1080 ticks), `ablation_data/ablation_U_timeseries.csv` (1112 ticks),
`ablation_data/ablation_summary.json`, `ablation_data/ablation_summary.md`, this `manifest.md`.
Generator (committed): `scripts/export_ablation_data.py`.
