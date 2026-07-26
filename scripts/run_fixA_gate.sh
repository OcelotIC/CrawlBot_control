#!/usr/bin/env bash
# Dock-leak Part 3 — Fix A (full-DOF impact map) re-gate. SINGLE PASS.
# Identical canonical invocation (PHASE3_METADATA flags) to Misc/runs/fixA_gate/
# (a NEW dir — the frozen canonical Misc/runs/p3b_gate_w24000_kp3/ is NOT touched),
# so the ONLY difference vs canonical is the impact-map code. Then postproc + the
# canonical six-criteria gate (vs the dcda974 baseline, corrected C5≤4.5).
set -u
cd "$(dirname "$0")/.."
SP=/tmp/claude-0/-home-user-CrawlBot-control/51ef2e0d-fbab-59f4-a937-ec828e0d7eda/scratchpad
name=fixA_gate
echo "=== [$name] Fix-A 5-step gate (full-DOF impact map; canonical flags) ==="
t0=$SECONDS
FRAMES_PER_STEP=0 MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_cooperative_arms.py \
  --ss-two-task --ss-alpha-mom 5000 --alpha-torso-pose 24000 \
  --ss-kp-torso 3 --ss-kd-torso 2.5 --aocs_mode legacy_pid_numerical --K_omega 50 \
  --n-steps 5 --out-dir "$name" > "$SP/${name}.log" 2>&1
echo "  diag exit=$? dt=$((SECONDS-t0))s"
{ echo "Fix-A (full-DOF impact map) 5-step gate"; echo "HEAD: $(git rev-parse HEAD)";
  echo "branch: $(git rev-parse --abbrev-ref HEAD)";
  echo "tree: $(test -z "$(git status --porcelain)" && echo clean || echo dirty)";
  echo "flags: --ss-two-task --ss-alpha-mom 5000 --alpha-torso-pose 24000 --ss-kp-torso 3 --ss-kd-torso 2.5 --aocs_mode legacy_pid_numerical --K_omega 50 --n-steps 5";
  echo "change vs canonical: ONLY the inelastic-impact block (Fix A full-DOF map)";
} > results/$name/PHASE3_METADATA.txt
echo "=== per-dock Impact(fullDOF) prints (3.1 parity vs A.1) ==="
grep -a "Impact(fullDOF)" "$SP/${name}.log" || echo "  (no impact prints found)"
echo "=== postproc ==="
SSMOM_RUN_DIR=$name MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/postprocess_results_figs.py \
  > "$SP/${name}_pp.log" 2>&1
echo "  postproc exit=$? F3F4=$(test -f results/$name/postproc_F3F4.csv && echo Y || echo N)"
echo "=== gate (corrected C5≤4.5) — Fix A vs dcda974 baseline ==="
SSMOM_WP_DIR=$name MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/gate_phase3.py 2>&1 \
  | grep -aE 'C[1-6]|NON-binding|BINDING|B15|dock'
echo "=== done [$name] ==="
