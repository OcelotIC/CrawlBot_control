#!/usr/bin/env bash
# Phase 2.2 SS two-task weight×gain tuning sweep (canonical single step).
# Two levers: torso-pose WEIGHT (priority) and torso GAINS Kp/Kd (aggressiveness).
# mom weight fixed at the validated 5000 except the TP-dom softening check.
# Kd scaled with Kp (Kd = 5*Kp/6) to preserve damping character.
# Runs are SEQUENTIAL: each diag invocation mutates+restores the shared MJCF.
set -u
cd "$(dirname "$0")/.."
COMMON="--ss-two-task --n-steps 1"
run () {  # name mom tp kp kd
  local name=$1 mom=$2 tp=$3 kp=$4 kd=$5
  echo "=== [$name] mom=$mom tp=$tp Kp=$kp Kd=$kd ==="
  local t0=$SECONDS
  FRAMES_PER_STEP=0 MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_cooperative_arms.py \
    $COMMON --ss-alpha-mom "$mom" --alpha-torso-pose "$tp" \
    --ss-kp-torso "$kp" --ss-kd-torso "$kd" --out-dir "$name" \
    > "/tmp/claude-0/-home-user-CrawlBot-control/51ef2e0d-fbab-59f4-a937-ec828e0d7eda/scratchpad/${name}.log" 2>&1
  echo "    exit=$? dt=$((SECONDS-t0))s"
}
#    name            mom    tp     kp   kd
# Weight axis (Kp6 default):
run p22_w8k_kp6      5000   8000   6.0  5.0
run p22_w12k_kp6     5000   12000  6.0  5.0
run p22_w20k_kp6     5000   20000  6.0  5.0
# Gain axis at torso-pose=12k and 20k (Kp 6->4->3, Kd scaled):
run p22_w12k_kp4     5000   12000  4.0  3.33
run p22_w20k_kp4     5000   20000  4.0  3.33
run p22_w12k_kp3     5000   12000  3.0  2.5
run p22_w20k_kp3     5000   20000  3.0  2.5
# TP-dom softening check: can the 23.8% sat be killed by a softer gain?
run p22_TPdom_kp3    500    30000  3.0  2.5
echo "=== sweep complete ==="