#!/usr/bin/env bash
# α (J2 #2) CoM-mobile DS sweep. Strict passivity (no Piste A). Translate the
# CoM toward the next anchor during the DWELL (dt_ds lengthened to trigger it).
# Binding/tracking sweep on n-steps 2 (one inter-step DWELL, fast); C1-C5 on a
# few n-steps 5 runs. RAW numbers; audit_alpha.py + gate_phase3.py analyse.
set -u
cd "$(dirname "$0")/.."
SP=/tmp/claude-0/-home-user-CrawlBot-control/51ef2e0d-fbab-59f4-a937-ec828e0d7eda/scratchpad
COMMON="--ss-two-task --ss-alpha-mom 5000 --alpha-torso-pose 24000 \
  --ss-kp-torso 3 --ss-kd-torso 2.5 --aocs_mode legacy_pid_numerical --K_omega 50"

run () {  local name="$1"; shift
  echo "=== [$name] ($*) ==="; local t0=$SECONDS
  FRAMES_PER_STEP=0 MUJOCO_GL=disabled PYTHONPATH=. python3 \
    scripts/diag_cooperative_arms.py $COMMON "$@" --out-dir "$name" \
    > "$SP/alpha_${name}.log" 2>&1
  echo "   exit=$? dt=$((SECONDS-t0))s"
}

# Binding/tracking sweep (n-steps 2, dt_ds 2.5): magnitude gentle->aggressive.
for m in 0.0 0.02 0.05 0.10 0.20; do
  run "alpha_m${m}" --n-steps 2 --dt-ds 2.5 --ds-mobile-com-magnitude "$m"
done
# Speed axis (n-steps 2, m=0.05): shorter DWELL = faster translation.
run alpha_dtds1.5_m0.05 --n-steps 2 --dt-ds 1.5 --ds-mobile-com-magnitude 0.05

echo "=== binding / tracking / manip analysis ==="
MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_alpha.py \
  m0.0=results/alpha_m0.0 m0.02=results/alpha_m0.02 \
  m0.05=results/alpha_m0.05 m0.10=results/alpha_m0.10 m0.20=results/alpha_m0.20 \
  dt1.5_m0.05=results/alpha_dtds1.5_m0.05 \
  2>&1 | tee "$SP/alpha_binding.log"

# C1-C5 on full 5-step traversals (baseline hold + one aggressive magnitude).
for m in 0.0 0.10; do
  run "alpha5_m${m}" --n-steps 5 --dt-ds 2.5 --ds-mobile-com-magnitude "$m"
  SSMOM_RUN_DIR=alpha5_m${m} MUJOCO_GL=disabled PYTHONPATH=. python3 \
    scripts/postprocess_results_figs.py > "$SP/alpha5_m${m}_pp.log" 2>&1
done
echo "=== C1-C5 (5-step) ===" | tee "$SP/alpha_gate.log"
for m in 0.0 0.10; do
  echo "--- alpha5_m${m} ---"
  SSMOM_WP_DIR=alpha5_m${m} MUJOCO_GL=disabled PYTHONPATH=. python3 \
    scripts/gate_phase3.py 2>&1 | grep -aE 'C[1-6]|dock' || echo "  (gate failed)"
done | tee -a "$SP/alpha_gate.log"
echo "=== alpha sweep done ==="
