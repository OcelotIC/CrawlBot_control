#!/usr/bin/env bash
# J2 c_curr characterization. Per-tick inter-step QP hw_current refresh.
#  2A (proxy box, AOCS-on): refresh OFF vs ON — the margin c_curr recovers.
#  2B (exact box, AOCS-on, refresh ON): the decisive C5 verdict (exact+AOCS-on
#     without c_curr was C5=4.949 FAIL).
# Canonical 5-step working point (= results/fixA_gate flags). AOCS-on throughout
# (c_curr only matters because the wheels move). RAW; no success threshold.
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
    scripts/postprocess_results_figs.py > "$SP/${name}_pp.log" 2>&1
  echo "   postproc exit=$?"
}

# 2A — proxy box (default), AOCS-on, refresh OFF vs ON.
run ccurr_proxy_off --no-interstep-hw-refresh
run ccurr_proxy_on
# 2B — exact box, AOCS-on, refresh ON (the target). (refresh-OFF exact = the
# prior aocs_on_exact C5=4.949, in results/j2_aocs_interstep/aocs_on_exact_gate.log)
run ccurr_exact_on  --qp-envelope-exact

echo "=== c_curr analysis (h_w split / staleness / settle) ==="
MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_ccurr.py \
  proxy_off=results/ccurr_proxy_off proxy_on=results/ccurr_proxy_on \
  exact_on=results/ccurr_exact_on 2>&1 | tee "$SP/ccurr_analysis.log"

echo "=== residual ==="
MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_fixC_residual.py \
  results/ccurr_proxy_off results/ccurr_proxy_on results/ccurr_exact_on \
  2>&1 | grep -aE "RUN:|traversal-final" | tee "$SP/ccurr_residual.log"

echo "=== six-criteria gate, each ==="
for name in ccurr_proxy_off ccurr_proxy_on ccurr_exact_on; do
  echo "--- $name ---"
  SSMOM_WP_DIR=$name MUJOCO_GL=disabled PYTHONPATH=. python3 \
    scripts/gate_phase3.py 2>&1 | grep -aE 'C[1-6]|dock|qpf|nin' || echo "  (gate failed)"
done | tee "$SP/ccurr_gate.log"
echo "=== c_curr characterization done ==="
