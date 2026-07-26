#!/usr/bin/env bash
# Piste A magnitude test (J2 #3 follow-up). Re-run the β-sweep at LARGER CoM
# translation to decide whether the CoM lag is passivity/work-limited (budget
# helps) or kinematics/reference-limited (budget doesn't). Exact box ON,
# dt_ds=2.5 (DWELL). mag 0.20 primary (β {0,0.5,1.0,2.0}); mag 0.10 midpoint.
# n=2 for binding/tracking, n=5 for C1-C5. RAW; no success threshold.
set -u
cd "$(dirname "$0")/.."
SP=/tmp/claude-0/-home-user-CrawlBot-control/51ef2e0d-fbab-59f4-a937-ec828e0d7eda/scratchpad
COMMON="--ss-two-task --ss-alpha-mom 5000 --alpha-torso-pose 24000 \
  --ss-kp-torso 3 --ss-kd-torso 2.5 --aocs_mode legacy_pid_numerical --K_omega 50 \
  --dt-ds 2.5 --qp-envelope-exact"

run () {  local name="$1"; shift
  echo "=== [$name] ($*) ==="; local t0=$SECONDS
  FRAMES_PER_STEP=0 MUJOCO_GL=disabled PYTHONPATH=. python3 \
    scripts/diag_cooperative_arms.py $COMMON "$@" --out-dir "$name" \
    > "$SP/$name.log" 2>&1
  echo "   exit=$? dt=$((SECONDS-t0))s"
}

# mag 0.20 β-sweep (n=2) — the decisive tracking/binding/needed-work test.
for b in 0.0 0.5 1.0 2.0; do
  run "pam20_b${b}" --n-steps 2 --ds-mobile-com-magnitude 0.20 --ds-passivity-beta "$b"
done
# mag 0.10 midpoint (n=2).
for b in 0.0 1.0; do
  run "pam10_b${b}" --n-steps 2 --ds-mobile-com-magnitude 0.10 --ds-passivity-beta "$b"
done
# mag 0.20 C1-C5 (n=5, β extremes) + postproc.
for b in 0.0 2.0; do
  run "pam20n5_b${b}" --n-steps 5 --ds-mobile-com-magnitude 0.20 --ds-passivity-beta "$b"
  SSMOM_RUN_DIR=pam20n5_b${b} MUJOCO_GL=disabled PYTHONPATH=. python3 \
    Misc/scripts/postprocess_results_figs.py > "$SP/pam20n5_b${b}_pp.log" 2>&1
done

echo "=== mag 0.20 β-sweep: tracking / needed-work / binding / safety / manip ==="
MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/audit_pisteA.py \
  b0=results/pam20_b0.0 b0.5=results/pam20_b0.5 b1.0=results/pam20_b1.0 b2.0=results/pam20_b2.0 \
  2>&1 | tee "$SP/pam20_beta.log"

echo "=== mag 0.10 midpoint ==="
MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/audit_pisteA.py \
  b0=results/pam10_b0.0 b1.0=results/pam10_b1.0 2>&1 | tee "$SP/pam10_beta.log"

echo "=== residual (mag 0.20 sweep) ==="
MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/audit_fixC_residual.py \
  results/pam20_b0.0 results/pam20_b0.5 results/pam20_b1.0 results/pam20_b2.0 \
  2>&1 | grep -aE "RUN:|traversal-final" | tee "$SP/pam20_residual.log"

echo "=== C1-C5 (mag 0.20, n=5, β 0 vs 2.0) ==="
for name in pam20n5_b0.0 pam20n5_b2.0; do
  echo "--- $name ---"
  SSMOM_WP_DIR=$name MUJOCO_GL=disabled PYTHONPATH=. python3 \
    Misc/scripts/gate_phase3.py 2>&1 | grep -aE 'C[1-6]|dock' || echo "  (gate failed)"
done | tee "$SP/pam20_gate.log"
echo "=== piste A magnitude sweep done ==="
