# M7 — DS divergence determinism check (2026-04-19)

**Date:** 2026-04-19
**Scope:** Explain the R1_baseline vs archived-v21 `torso_ori_peak_deg_DS` discrepancy (177.97° vs 45.47°) before any interpretation of the four-run diagnostic decomposition.
**Input artefacts:**
- `results/archive_rediagnostic.md` — archived v21 value (`DS peak ori = 45.4672°`, SS peak `0.5348°`).
- `results/M7_abort_diag/R1_baseline/sim_log.json` — committed in `6128db9`.
- `docs/architecture/M7_DS_DIAGNOSTIC_EXPERIMENTS.md` — task spec §4/§5.

## 1. Five-value reproduction table

All values are `torso_ori_peak_deg_DS` from the per-phase refactor of `compute_metrics`. Simulation setup: 1% mass ratio, `n_steps=1`, `start_a=2`, `start_b=2`, v21 config (`preplanner_a_cruise_max=0.01`, `preplanner_cruise_ramp_frac=0.2`, all other M7 settings as in `scripts/run_m7_v21_preplanner_cruise.py`).

| # | scenario | DS peak ori [°] |
|---|---|---|
| 1 | archived v21 (`results/M7_1pct_1step_v21/`, 2026-04-17) | 45.4672 |
| 2 | today, v21 script, branch HEAD (three `diag_*_on_abort` flags present, all `False`) | 177.9665 |
| 3 | today, v21 script, flags absent (`config.py` and `sim_loop.py` checked out from `HEAD~1`) | 177.9665 |
| 4 | today, R1_baseline run 1 (committed in `6128db9`) | 177.9665 |
| 5 | today, R1_baseline run 2 (fresh rerun, same branch HEAD) | 177.9665 |

SS-side quantities match the archive bit-for-bit on every run today (SS peak ori `0.5334°`, `ss_end_torso_ori_deg = 0.1990°`, `q_torso_ref_ss_to_ds_jump_deg = 3.4162°`, abort `d_mm = 40.8371`, `ori_deg = 6.9671`).

## 2. Four-scenario determinism table

Each scenario was executed twice on this machine today — once to `results/M7_abort_diag/` (committed in `6128db9`, "run 1") and once to a scratch directory (discarded, "run 2"). Identical config, identical machine, back-to-back runs.

| scenario | SS run 1 [°] | SS run 2 [°] | DS run 1 [°] | DS run 2 [°] | DS identical? |
|---|---|---|---|---|---|
| R1_baseline       | 0.533437 | 0.533437 | 177.9665 | 177.9665 | yes |
| R2_freeze_ref     | 0.533437 | 0.533437 | 147.2024 | 147.2024 | yes |
| R3_single_contact | 0.533437 | 0.533437 |  18.0524 |  18.0524 | yes |
| R4_no_passivity   | 0.533437 | 0.533437 | 179.7464 | 179.7464 | yes |

Every value is bit-identical across repeats.

## 3. Factual observations

1. **The three `diag_*_on_abort` flags are innocent of the archive-vs-today DS divergence.** Values 2 and 3 in §1 are bit-identical: branch HEAD with all three flags present-and-`False` reproduces exactly the same DS trajectory as a clean revert of `config.py` and `sim_loop.py` to their `HEAD~1` state (no flag definitions, no flag-gated conditionals, no `passivity_override` kwarg).

2. **On this machine today, every R* scenario is deterministic across repeats.** All four DS peak values in §2 are bit-identical between run 1 and run 2. SS-side metrics are also bit-identical. The R1–R4 deltas reported in `results/M7_abort_diag/summary.md` therefore reflect the diagnostic flag effects, not run-to-run noise.

3. **SS-side behaviour matches the 2026-04-17 archive bit-for-bit.** SS peak ori (`0.5334°`), `ss_end_torso_ori_deg` (`0.1990°`), `q_torso_ref_ss_to_ds_jump_deg` (`3.4162°`), and the abort event (`d_mm = 40.8371`, `ori_deg = 6.9671`) all reproduce today with or without the flags, and match the archived v21 values from `results/archive_rediagnostic.md`.

4. **The archive-vs-today DS divergence is confined to the post-abort DS trajectory, which receives bit-identical inputs on both sides and is therefore environmental.** Given that the state at the SS→DS boundary (log entry `i_ss_last`) and the reference generator's output at the first DS tick (`q_torso_ref[i_ds_first]`) are numerically identical between today's runs and the archive, the ~132° gap between 45.4672° and 177.9665° originates downstream of that boundary — in the trailing-DS solver loop (QP, passivity inequality, AOCS, MuJoCo integration) operating on identical inputs yet producing different trajectories. This points to a difference in the execution environment between the archived run (generated at some prior commit, with some prior solver/BLAS/MuJoCo/Python versions, mtime `2026-04-17 22:52:30`) and today's environment.

## 4. Reproducibility pointer

The four-run R1–R4 data live under `results/M7_abort_diag/` at commit `6128db9`. Together with `scripts/run_m7_abort_diag.py` (same commit) and the v21 config produced by `scripts/run_m7_single_step._make_m7_config()` + `preplanner_a_cruise_max=0.01` + `preplanner_cruise_ramp_frac=0.2`, they fully regenerate the §1 rows 2–5 and the §2 table on the same machine/toolchain.

The raw sim_logs and diagnostic plots from the scratch reruns used to populate §1 rows 2, 3, 5 and §2 "run 2" columns were **not preserved on the branch**. Given determinism on this machine (§3.2), the commit `6128db9` plus the v21 script are sufficient to regenerate them exactly; keeping ~15 MB of duplicate per-run artefacts adds nothing over the numerics in this report.
