#!/usr/bin/env bash
# Fix C (J2 #1) REFINED characterization sweep. The first sweep (ε_twist ∈
# [0.02,0.20]) showed the dock is POSE-gated: ‖Jc·v⁻‖ at firing is ~0.004-0.006,
# below that whole range, so the residual was invariant to ε_twist. This sweep
# probes the binding regimes:
#   (A) TIGHT ε_twist around the ~0.006 floor (weld_radius=0.005) — where the
#       twist gate starts to bind (delay-into-hold vs timeout vs residual move),
#   (B) ε_pos sweep (weld_radius, twist loose at 0.05) — the gap-couple knob.
set -u
cd "$(dirname "$0")/.."
SP=/tmp/claude-0/-home-user-CrawlBot-control/51ef2e0d-fbab-59f4-a937-ec828e0d7eda/scratchpad
COMMON="--ss-two-task --ss-alpha-mom 5000 --alpha-torso-pose 24000 \
  --ss-kp-torso 3 --ss-kd-torso 2.5 --aocs_mode legacy_pid_numerical \
  --K_omega 50 --n-steps 5"

run () {  local name="$1"; shift
  echo "=== [$name] run ($*) ==="
  local t0=$SECONDS
  FRAMES_PER_STEP=0 MUJOCO_GL=disabled PYTHONPATH=. python3 \
    scripts/diag_cooperative_arms.py $COMMON "$@" --out-dir "$name" \
    > "$SP/fixC_${name}.log" 2>&1
  echo "   diag exit=$? dt=$((SECONDS-t0))s"
}

# (A) TIGHT ε_twist (weld_radius=0.005 default)
for eps in 0.004 0.005 0.006 0.007; do
  run "fixC_t${eps}" --dock-twist-max "$eps"
done
# (B) ε_pos sweep (twist loose so it never binds → isolate the pose knob)
for wr in 0.002 0.003 0.004; do
  run "fixC_p${wr}" --dock-twist-max 0.05 --weld-radius "$wr"
done

echo "=== residual + twist/pose split + firing (refined runs) ==="
MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_fixC_residual.py \
  results/fixC_t0.004 results/fixC_t0.005 results/fixC_t0.006 results/fixC_t0.007 \
  results/fixC_p0.002 results/fixC_p0.003 results/fixC_p0.004 \
  2>&1 | tee "$SP/fixC_residual2.log"

echo "=== C1-C5 per refined run ==="
for name in fixC_t0.004 fixC_t0.005 fixC_t0.006 fixC_t0.007 \
            fixC_p0.002 fixC_p0.003 fixC_p0.004; do
  echo "--- $name ---"
  SSMOM_RUN_DIR=$name MUJOCO_GL=disabled PYTHONPATH=. python3 \
    scripts/postprocess_results_figs.py > "$SP/fixC_${name}_pp.log" 2>&1
  SSMOM_WP_DIR=$name MUJOCO_GL=disabled PYTHONPATH=. python3 \
    scripts/gate_phase3.py 2>&1 | grep -aE 'C[1-6]|dock' || echo "  (gate failed)"
done | tee "$SP/fixC_gate2.log"
echo "=== refined sweep done ==="
