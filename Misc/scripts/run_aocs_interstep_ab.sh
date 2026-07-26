#!/usr/bin/env bash
# J2 step 4a — AOCS re-activation in the inter-step DS loop: A/B characterization.
# AOCS-on (new default) vs AOCS-off (--no-aocs-in-interstep = the legacy
# hardcoded-zero wheels). Canonical 5-step working point (identical flags to
# Misc/runs/fixA_gate). Then postproc + the six-criteria gate for each. RAW
# numbers; no success threshold — the h_w/C5 trade and default-on are decided
# by Idriss + reviewing Claude.
set -u
cd "$(dirname "$0")/.."
SP=/tmp/claude-0/-home-user-CrawlBot-control/51ef2e0d-fbab-59f4-a937-ec828e0d7eda/scratchpad
COMMON="--ss-two-task --ss-alpha-mom 5000 --alpha-torso-pose 24000 \
  --ss-kp-torso 3 --ss-kd-torso 2.5 --aocs_mode legacy_pid_numerical --K_omega 50 \
  --n-steps 5"

run () {  local name="$1"; shift
  echo "=== [$name] ($*) ==="; local t0=$SECONDS
  FRAMES_PER_STEP=0 MUJOCO_GL=disabled PYTHONPATH=. python3 \
    scripts/diag_cooperative_arms.py $COMMON "$@" --out-dir "$name" \
    > "$SP/$name.log" 2>&1
  echo "   diag exit=$? dt=$((SECONDS-t0))s"
  SSMOM_RUN_DIR=$name MUJOCO_GL=disabled PYTHONPATH=. python3 \
    Misc/scripts/postprocess_results_figs.py > "$SP/${name}_pp.log" 2>&1
  echo "   postproc exit=$?"
}

run aocs_on                            # AOCS re-activated (new default)
run aocs_off --no-aocs-in-interstep    # legacy hardcoded-zero wheels

echo "=== inter-step characterization (θ_s drift / τ_w / settle / h_w) ==="
MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/audit_aocs_interstep.py \
  on=results/aocs_on off=results/aocs_off 2>&1 | tee "$SP/aocs_ab.log"

echo "=== six-criteria gate, each ==="
for name in aocs_on aocs_off; do
  echo "--- $name ---"
  SSMOM_WP_DIR=$name MUJOCO_GL=disabled PYTHONPATH=. python3 \
    Misc/scripts/gate_phase3.py 2>&1 | grep -aE 'C[1-6]|dock' || echo "  (gate failed)"
done | tee "$SP/aocs_gate.log"
echo "=== AOCS inter-step A/B done ==="
