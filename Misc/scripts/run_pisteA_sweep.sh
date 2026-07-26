#!/usr/bin/env bash
# Piste A (J2 #3) characterization. Moving-CoM scenario (mag 0.05, dt_ds 2.5).
# 2A: β-sweep {0,0.25,0.5,1.0} with the exact Ḣ_s box (LOT B on). 2B: box
# isolated — β=0, exact box OFF (proxy) vs ON. n-steps 5 ⇒ binding/tracking
# (from ds_mobile_trace across DWELLs) AND C1-C5 from one run set. RAW.
set -u
cd "$(dirname "$0")/.."
SP=/tmp/claude-0/-home-user-CrawlBot-control/51ef2e0d-fbab-59f4-a937-ec828e0d7eda/scratchpad
COMMON="--ss-two-task --ss-alpha-mom 5000 --alpha-torso-pose 24000 \
  --ss-kp-torso 3 --ss-kd-torso 2.5 --aocs_mode legacy_pid_numerical --K_omega 50 \
  --n-steps 5 --dt-ds 2.5 --ds-mobile-com-magnitude 0.05"

run () {  local name="$1"; shift
  echo "=== [$name] ($*) ==="; local t0=$SECONDS
  FRAMES_PER_STEP=0 MUJOCO_GL=disabled PYTHONPATH=. python3 \
    scripts/diag_cooperative_arms.py $COMMON "$@" --out-dir "$name" \
    > "$SP/pa_${name}.log" 2>&1
  echo "   exit=$? dt=$((SECONDS-t0))s"
  SSMOM_RUN_DIR=$name MUJOCO_GL=disabled PYTHONPATH=. python3 \
    Misc/scripts/postprocess_results_figs.py > "$SP/pa_${name}_pp.log" 2>&1
}

# 2A: β-sweep with the exact box ON (LOT B). β=0 here is the shared baseline.
run pa_b0_exact    --ds-passivity-beta 0.0  --qp-envelope-exact
run pa_b0.25_exact --ds-passivity-beta 0.25 --qp-envelope-exact
run pa_b0.5_exact  --ds-passivity-beta 0.5  --qp-envelope-exact
run pa_b1.0_exact  --ds-passivity-beta 1.0  --qp-envelope-exact
# 2B: box isolated — β=0, proxy box (exact OFF) vs the β=0_exact above.
run pa_b0_proxy    --ds-passivity-beta 0.0

echo "=== 2A β-sweep: tracking / binding / budget-bounded / feasibility ==="
MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/audit_pisteA.py \
  b0=results/pa_b0_exact b0.25=results/pa_b0.25_exact \
  b0.5=results/pa_b0.5_exact b1.0=results/pa_b1.0_exact \
  2>&1 | tee "$SP/pa_beta.log"

echo "=== 2B box isolation (β=0): proxy vs exact ==="
MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/audit_pisteA.py \
  proxy=results/pa_b0_proxy exact=results/pa_b0_exact 2>&1 | tee "$SP/pa_box.log"

echo "=== residual per run ==="
MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/audit_fixC_residual.py \
  results/pa_b0_exact results/pa_b0.25_exact results/pa_b0.5_exact \
  results/pa_b1.0_exact results/pa_b0_proxy 2>&1 \
  | grep -aE "RUN:|traversal-final|docks fired" | tee "$SP/pa_residual.log"

echo "=== C1-C5 per run ==="
for name in pa_b0_exact pa_b0.25_exact pa_b0.5_exact pa_b1.0_exact pa_b0_proxy; do
  echo "--- $name ---"
  SSMOM_WP_DIR=$name MUJOCO_GL=disabled PYTHONPATH=. python3 \
    Misc/scripts/gate_phase3.py 2>&1 | grep -aE 'C[1-6]|dock' || echo "  (gate failed)"
done | tee "$SP/pa_gate.log"
echo "=== piste A sweep done ==="
