#!/usr/bin/env bash
# J2 chatter DIAGNOSIS (no fix). Two canonical 5-step runs, both exact box +
# AOCS-on (the chattering config). Toggle test A = freeze c_curr (the inter-step
# QP hw_current) at DS entry via --no-interstep-hw-refresh. The chatter active
# set is reconstructed OFFLINE from the logged lambda_qp (the exact envelope box
# |M_exact·λ|≤5 has a fixed RHS) — no crawlbot/ change. postproc gives the
# per-tick stance indices + planned Hdot_s.
set -u
cd "$(dirname "$0")/.."
SP=/tmp/claude-0/-home-user-CrawlBot-control/51ef2e0d-fbab-59f4-a937-ec828e0d7eda/scratchpad
COMMON="--ss-two-task --ss-alpha-mom 5000 --alpha-torso-pose 24000 \
  --ss-kp-torso 3 --ss-kd-torso 2.5 --aocs_mode legacy_pid_numerical --K_omega 50 \
  --n-steps 5 --qp-envelope-exact"

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

run chatter_base                            # exact box, AOCS-on, c_curr refreshed (baseline)
run chatter_cfrozen --no-interstep-hw-refresh   # test A: c_curr frozen at DS entry

echo "=== chatter analysis (active set / flip / wrench-rate / config) ==="
MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/audit_chatter.py \
  base=results/chatter_base cfrozen=results/chatter_cfrozen 2>&1 | tee "$SP/chatter_analysis.log"
echo "=== chatter diagnosis done ==="
