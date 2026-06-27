#!/usr/bin/env bash
# Iteration-B 5-step re-gate for one candidate (legacy_pid_numerical, K_omega=50).
# usage: run_iterB_gate.sh <weight> <Kp> <Kd>
# Corrected C5 (≤4.5 N·m·s) applied by gate_phase3.py. OFF (phase3_off) reused for
# C6 — the OFF path is untouched (no crawlbot/ change; test_reworked_qp 8/8).
set -u
cd "$(dirname "$0")/.."
SP=/tmp/claude-0/-home-user-CrawlBot-control/51ef2e0d-fbab-59f4-a937-ec828e0d7eda/scratchpad
W=$1; KP=$2; KD=$3; name="p3b_gate_w${W}_kp${KP}"
echo "=== [$name] 5-step gate: tp=$W Kp=$KP Kd=$KD (AOCS=legacy_pid_numerical) ==="
t0=$SECONDS
FRAMES_PER_STEP=0 MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_cooperative_arms.py \
  --ss-two-task --ss-alpha-mom 5000 --alpha-torso-pose "$W" \
  --ss-kp-torso "$KP" --ss-kd-torso "$KD" --aocs_mode legacy_pid_numerical --K_omega 50 \
  --n-steps 5 --out-dir "$name" > "$SP/${name}.log" 2>&1
echo "  diag exit=$? dt=$((SECONDS-t0))s"
{ echo "Iteration-B 5-step gate candidate"; echo "HEAD: $(git rev-parse HEAD)";
  echo "tree: $(test -z "$(git status --porcelain)" && echo clean || echo dirty)";
  echo "flags: --ss-two-task --ss-alpha-mom 5000 --alpha-torso-pose $W --ss-kp-torso $KP --ss-kd-torso $KD --aocs_mode legacy_pid_numerical --K_omega 50 --n-steps 5";
} > results/$name/PHASE3_METADATA.txt
SSMOM_RUN_DIR=$name MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/postprocess_results_figs.py \
  > "$SP/${name}_pp.log" 2>&1
echo "  postproc exit=$? F3F4=$(test -f results/$name/postproc_F3F4.csv && echo Y || echo N)"
echo "=== gate evaluation (corrected C5≤4.5) ==="
SSMOM_WP_DIR=$name MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/gate_phase3.py 2>&1 \
  | grep -aE 'C[1-6]|NON-binding|BINDING|B15'
echo "=== done [$name] ==="