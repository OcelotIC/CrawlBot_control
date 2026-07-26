#!/usr/bin/env bash
# POINT 1 (CC BRIEF — close the chatter fix): inter-config robustness of the
# settle-only Tikhonov ε. Hold AOCS = legacy_pid_numerical (the FIGURE config,
# run-A) fixed; vary the CONFIGURATION (anchor reach/geometry + run-B DWELL) and
# re-locate the clean-threshold ε at each. Sequential (no CPU contention).
#
# Clean = flip-frac < 0.1 AND at-±5 = 0 (no Σf sign-flip) in the arm-a settles.
set -u
cd /home/user/CrawlBot_control
export MUJOCO_GL=disabled PYTHONPATH=.
DIAG=scripts/diag_cooperative_arms.py
BASE="--qp-envelope-exact --aocs_mode legacy_pid_numerical"

# config_label : extra flags (beyond BASE)
declare -A CFG
CFG[c0]=""                                            # canonical figure config (dx 0.8)
CFG[dx10]="--anchor_dx 1.0"                           # larger reach + different stance geometry
CFG[dx06]="--anchor_dx 0.6"                           # different (tighter) stance geometry
CFG[runB]="--ds-mobile-com-magnitude 0.20 --dt-ds 2.5" # run-B DWELL moving-CoM

EPS_LIST="0 1 3"     # 0 = baseline (chatter present); 1 = expect partial; 3 = expect clean

for cfg in c0 dx10 dx06 runB; do
  for e in $EPS_LIST; do
    out="p1_${cfg}_e${e}"
    echo "=== [$(date +%H:%M:%S)] RUN $out : $BASE ${CFG[$cfg]} --interstep-settle-alpha-wrench $e ==="
    python3 $DIAG $BASE ${CFG[$cfg]} --interstep-settle-alpha-wrench "$e" --out-dir "$out" \
        > "/home/user/CrawlBot_control/results/${out}_run.log" 2>&1
    rc=$?
    # quick docking summary from the run log
    docks=$(grep -c "DOCKED\|dock" "/home/user/CrawlBot_control/results/${out}_run.log" 2>/dev/null || echo "?")
    echo "    -> rc=$rc  (log: results/${out}_run.log)"
  done
done
echo "=== [$(date +%H:%M:%S)] POINT-1 SWEEP COMPLETE ==="
