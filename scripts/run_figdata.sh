#!/usr/bin/env bash
# Figure-data export driver (J2). Runs the two committed-flag configurations and
# exports tidy CSV + meta JSON for off-repo figure building. NO crawlbot/ change.
#   RUN A: canonical 5-step, LATEST FEATURES — exact box + AOCS-on + c_curr-on
#          (aocs_active_in_interstep & interstep_hw_refresh are the committed
#          defaults; --qp-envelope-exact turns on the exact envelope physics).
#   RUN B: DWELL with moving CoM — ds_mobile_com_magnitude 0.20, dt_ds 2.5, same
#          latest features (exposes the DS-mobile CoM lag dormant in run A).
set -u
cd "$(dirname "$0")/.."
SP=/tmp/claude-0/-home-user-CrawlBot-control/51ef2e0d-fbab-59f4-a937-ec828e0d7eda/scratchpad
COMMON="--ss-two-task --ss-alpha-mom 5000 --alpha-torso-pose 24000 \
  --ss-kp-torso 3 --ss-kd-torso 2.5 --aocs_mode legacy_pid_numerical --K_omega 50 \
  --n-steps 5 --qp-envelope-exact"

echo "=== RUN A (latest features) ==="; t0=$SECONDS
FRAMES_PER_STEP=0 MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_cooperative_arms.py \
  $COMMON --out-dir figA > "$SP/figA.log" 2>&1
echo "  runA diag rc=$? dt=$((SECONDS-t0))s"

echo "=== RUN B (DWELL moving-CoM mag 0.20, dt_ds 2.5) ==="; t0=$SECONDS
FRAMES_PER_STEP=0 MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_cooperative_arms.py \
  $COMMON --ds-mobile-com-magnitude 0.20 --dt-ds 2.5 --out-dir figB > "$SP/figB.log" 2>&1
echo "  runB diag rc=$? dt=$((SECONDS-t0))s"

echo "=== POSTPROC A + B (produces postproc_F3F4.csv with the C3 Hdot_s) ==="
SSMOM_RUN_DIR=figA MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/postprocess_results_figs.py > "$SP/figA_pp.log" 2>&1; echo "  ppA rc=$?"
SSMOM_RUN_DIR=figB MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/postprocess_results_figs.py > "$SP/figB_pp.log" 2>&1; echo "  ppB rc=$?"

echo "=== EXPORT A ==="
MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/export_figure_data.py \
  --run-dir results/figA --out results/j2_figdata/runA \
  --label "latest-features canonical 5-step (exact box + AOCS-on + c_curr-on)" \
  --config '{"qp_envelope_exact": true, "aocs_active_in_interstep": true, "interstep_hw_refresh": true, "aocs_mode": "legacy_pid_numerical", "n_steps": 5}'

echo "=== EXPORT B ==="
MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/export_figure_data.py \
  --run-dir results/figB --out results/j2_figdata/runB \
  --label "DWELL moving-CoM mag 0.20 dt_ds 2.5 (latest features)" \
  --config '{"qp_envelope_exact": true, "aocs_active_in_interstep": true, "interstep_hw_refresh": true, "aocs_mode": "legacy_pid_numerical", "ds_mobile_com_magnitude": 0.20, "dt_ds": 2.5, "n_steps": 5}'

echo "=== figdata export done ==="
