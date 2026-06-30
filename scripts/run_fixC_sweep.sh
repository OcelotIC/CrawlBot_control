#!/usr/bin/env bash
# Fix C (J2 #1) characterization sweep. Runs the canonical 5-step traversal
# (fixA_gate working-point flags) for: one LEGACY-linear-gate A/B reference
# + a sweep of ε_twist for the 6-D weld-relative-twist gate. Each run -> its
# own results/ dir. Then postproc + the six-criteria gate per run. NO verdict
# here — raw numbers only; audit_fixC_residual.py + gate_phase3.py do the rest.
set -u
cd "$(dirname "$0")/.."
SP=/tmp/claude-0/-home-user-CrawlBot-control/51ef2e0d-fbab-59f4-a937-ec828e0d7eda/scratchpad
COMMON="--ss-two-task --ss-alpha-mom 5000 --alpha-torso-pose 24000 \
  --ss-kp-torso 3 --ss-kd-torso 2.5 --aocs_mode legacy_pid_numerical \
  --K_omega 50 --n-steps 5"

run () {  # $1 = out-dir name, $2... = extra flags
  local name="$1"; shift
  echo "=== [$name] run ($*) ==="
  local t0=$SECONDS
  FRAMES_PER_STEP=0 MUJOCO_GL=disabled PYTHONPATH=. python3 \
    scripts/diag_cooperative_arms.py $COMMON "$@" --out-dir "$name" \
    > "$SP/fixC_${name}.log" 2>&1
  echo "   diag exit=$? dt=$((SECONDS-t0))s  docks=$(grep -ac 'DOCK ' "$SP/fixC_${name}.log")"
  SSMOM_RUN_DIR=$name MUJOCO_GL=disabled PYTHONPATH=. python3 \
    scripts/postprocess_results_figs.py > "$SP/fixC_${name}_pp.log" 2>&1
  echo "   postproc exit=$?"
}

# A/B reference: legacy linear gate (reproduces ae0673e dock behaviour).
run fixC_linear --dock-gate-linear

# 6-D twist gate sweep. Values bracket tight->loose; refine from the
# linear-run near-dock ‖Jc·v⁻‖ if needed.
for eps in 0.02 0.05 0.10 0.20; do
  run "fixC_e${eps}" --dock-twist-max "$eps"
done

echo "=== residual + twist/pose split + firing (all runs) ==="
MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_fixC_residual.py \
  results/fixC_linear results/fixC_e0.02 results/fixC_e0.05 \
  results/fixC_e0.10 results/fixC_e0.20 2>&1 | tee "$SP/fixC_residual.log"

echo "=== C1-C5 per run (vs dcda974 baseline, corrected C5≤4.5) ==="
for name in fixC_linear fixC_e0.02 fixC_e0.05 fixC_e0.10 fixC_e0.20; do
  echo "--- $name ---"
  SSMOM_WP_DIR=$name MUJOCO_GL=disabled PYTHONPATH=. python3 \
    scripts/gate_phase3.py 2>&1 | grep -aE 'C[1-6]|dock' || echo "  (gate failed)"
done | tee "$SP/fixC_gate.log"
echo "=== sweep done ==="
