# M7 post-abort DS diagnostic decomposition — summary

Four runs per `docs/architecture/M7_DS_DIAGNOSTIC_EXPERIMENTS.md`.
Each run toggles at most one `diag_*_on_abort` flag; all other
parameters identical to the v21 baseline. `τ_q` saturation threshold
is `0.99 · tau_max = 19.8 Nm`; `τ_w` saturation threshold is
`0.99 · tau_w_max = 4.95 Nm`. No interpretation included.

| run | SS peak ori [°] | q_ref jump [°] | DS entry ori [°] | DS peak ori [°] | max|τ_q|_DS [Nm] | τ_q sat frac DS | τ_w sat frac DS | dock? |
|---|---|---|---|---|---|---|---|---|
| R1_baseline | 0.5334 | 3.4162 | 3.3251 | 177.9665 | 20.000 | 0.785 | 0.985 | no |
| R2_freeze_ref | 0.5334 | 0.1990 | 0.0480 | 147.2024 | 20.000 | 0.735 | 0.990 | no |
| R3_single_contact | 0.5334 | 3.4162 | 3.3651 | 18.0524 | 20.000 | 1.000 | 1.000 | no |
| R4_no_passivity | 0.5334 | 3.4162 | 3.3251 | 179.7464 | 20.000 | 0.505 | 0.980 | no |

## Invariants

All SS-side invariants hold: SS peak ori within 0.01° of baseline across
all four runs; `ee_pos_peak_mm_SS`, `ss_end_torso_ori_deg`,
`preplanner_T_steps[0]`, and abort `d_mm` / `ori_deg` are bit-for-bit
identical to R1_baseline for R2, R3, R4.

## Baseline vs archive (pipeline-drift observation, no interpretation)

R1_baseline `torso_ori_peak_deg_DS` = 177.97° on this machine today;
the archived v21 value in `results/archive_rediagnostic.md` is 45.47°.
SS-side metrics match the archive (SS peak ori 0.5334°, abort d_mm 40.84,
abort ori_deg 6.97). R2/R3/R4 deltas are reported relative to R1_baseline
as re-run today, not relative to the archive.
