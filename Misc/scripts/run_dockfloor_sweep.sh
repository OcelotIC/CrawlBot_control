#!/usr/bin/env bash
# Dock-floor passivity audit. Dock-only (no CoM translation, ds_mobile=0,
# dt_ds default). Vary passivity in the SS dock-close hold window: off
# (current) / strict-ON / budgeted-ON. Measure achievable dock distance,
# positive work, twist, residual, C1-C5. RAW; audit_dockfloor.py +
# audit_fixC_residual.py + gate_phase3.py analyse.
set -u
cd "$(dirname "$0")/.."
SP=/tmp/claude-0/-home-user-CrawlBot-control/51ef2e0d-fbab-59f4-a937-ec828e0d7eda/scratchpad
COMMON="--ss-two-task --ss-alpha-mom 5000 --alpha-torso-pose 24000 \
  --ss-kp-torso 3 --ss-kd-torso 2.5 --aocs_mode legacy_pid_numerical --K_omega 50 \
  --n-steps 5 --log-dock-work"

run () {  local name="$1"; shift
  echo "=== [$name] ($*) ==="; local t0=$SECONDS
  FRAMES_PER_STEP=0 MUJOCO_GL=disabled PYTHONPATH=. python3 \
    scripts/diag_cooperative_arms.py $COMMON "$@" --out-dir "$name" \
    > "$SP/df_${name}.log" 2>&1
  echo "   exit=$? dt=$((SECONDS-t0))s"
  SSMOM_RUN_DIR=$name MUJOCO_GL=disabled PYTHONPATH=. python3 \
    Misc/scripts/postprocess_results_figs.py > "$SP/df_${name}_pp.log" 2>&1
}

run df_off                                             # passivity OFF in hold (current floor)
run df_strict   --dock-hold-passivity-on               # passivity ON, RHS<=0 (strict)
run df_W0.5     --dock-hold-passivity-on --passivity-w-budget 0.5
run df_W2.0     --dock-hold-passivity-on --passivity-w-budget 2.0
run df_W10      --dock-hold-passivity-on --passivity-w-budget 10.0

echo "=== dock distance / positive work / twist ==="
MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/audit_dockfloor.py \
  off=results/df_off strict=results/df_strict W0.5=results/df_W0.5 \
  W2.0=results/df_W2.0 W10=results/df_W10 2>&1 | tee "$SP/df_dock.log"

echo "=== residual (subtree_angmom) per level ==="
MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/audit_fixC_residual.py \
  results/df_off results/df_strict results/df_W0.5 results/df_W2.0 results/df_W10 \
  2>&1 | grep -aE "RUN:|traversal-final|docks fired" | tee "$SP/df_residual.log"

echo "=== C1-C5 per level ==="
for name in df_off df_strict df_W0.5 df_W2.0 df_W10; do
  echo "--- $name ---"
  SSMOM_WP_DIR=$name MUJOCO_GL=disabled PYTHONPATH=. python3 \
    Misc/scripts/gate_phase3.py 2>&1 | grep -aE 'C[1-6]|dock' || echo "  (gate failed)"
done | tee "$SP/df_gate.log"
echo "=== dock-floor sweep done ==="
