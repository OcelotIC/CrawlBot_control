#!/usr/bin/env bash
# Iteration B — single-step weight×gain screen under the paper's AOCS
# (legacy_pid_numerical, K_omega=50). Grid around 5k:20k Kp3:
#   torso-pose weight {16k,20k,24k} × Kp {3,4} (Kd = 5·Kp/6).
# Screen = cheap NARROWING on step 0 (a binding-type step); the 5-step gate decides.
# mom fixed 5000; ee/posture/wrench at defaults. Sequential (shared-MJCF race).
set -u
cd "$(dirname "$0")/.."
SP=/tmp/claude-0/-home-user-CrawlBot-control/51ef2e0d-fbab-59f4-a937-ec828e0d7eda/scratchpad
AOCS="--aocs_mode legacy_pid_numerical --K_omega 50"
run () {  # W KP KD
  local W=$1 KP=$2 KD=$3 name="p3b_w${1}_kp${2}"
  echo "=== [$name] tp=$W Kp=$KP Kd=$KD (AOCS=legacy_pid_numerical) ==="
  local t0=$SECONDS
  FRAMES_PER_STEP=0 MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_cooperative_arms.py \
    --ss-two-task --ss-alpha-mom 5000 --alpha-torso-pose "$W" \
    --ss-kp-torso "$KP" --ss-kd-torso "$KD" $AOCS --n-steps 1 --out-dir "$name" \
    > "$SP/${name}.log" 2>&1
  echo "    diag exit=$? dt=$((SECONDS-t0))s"
  SSMOM_RUN_DIR=$name MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/postprocess_results_figs.py \
    > "$SP/${name}_pp.log" 2>&1
  echo "    postproc exit=$? F3F4=$(test -f results/$name/postproc_F3F4.csv && echo Y || echo N)"
}
run 16000 3 2.5
run 20000 3 2.5
run 24000 3 2.5
run 16000 4 3.33
run 20000 4 3.33
run 24000 4 3.33
echo "=== screen complete ==="