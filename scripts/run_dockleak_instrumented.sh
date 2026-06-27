#!/usr/bin/env bash
# Dock-leak diagnostic — Part 1: ONE instrumented re-run of the FROZEN canonical
# working point (5k:24k Kp3, legacy_pid_numerical, K_omega 50, 5 steps), with the
# per-tick subtree_angmom[0] / n_weld instrument compiled in (diag/dock-leak branch).
#
# Writes to results/dockleak_instrumented/ — the canonical results/p3b_gate_w24000_kp3/
# is FROZEN and NOT touched. Flags are byte-for-byte the canonical PHASE3_METADATA.txt
# set, so the shared log fields must reproduce the canonical run bit-identically; that
# equality is the proof the instrument does not perturb the physics.
set -u
cd "$(dirname "$0")/.."
SP=/tmp/claude-0/-home-user-CrawlBot-control/51ef2e0d-fbab-59f4-a937-ec828e0d7eda/scratchpad
OUT=dockleak_instrumented
FLAGS="--ss-two-task --ss-alpha-mom 5000 --alpha-torso-pose 24000 --ss-kp-torso 3 --ss-kd-torso 2.5 --aocs_mode legacy_pid_numerical --K_omega 50 --n-steps 5"

echo "=== [dockleak] instrumented canonical re-run (flags matched to p3b_gate_w24000_kp3) ==="
t0=$SECONDS
FRAMES_PER_STEP=0 MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_cooperative_arms.py \
  $FLAGS --out-dir "$OUT" > "$SP/dockleak_run.log" 2>&1
echo "  diag exit=$? dt=$((SECONDS-t0))s"
{ echo "Dock-leak instrumented re-run (Part 1)";
  echo "HEAD: $(git rev-parse HEAD)";
  echo "branch: $(git rev-parse --abbrev-ref HEAD)";
  echo "tree: $(test -z "$(git status --porcelain)" && echo clean || echo dirty)";
  echo "flags: $FLAGS";
  echo "instrument: per-tick H_sys=subtree_angmom[0] (world), n_weld=eq_active.sum()";
  echo "canonical (frozen) reference: results/p3b_gate_w24000_kp3/";
} > "results/$OUT/DOCKLEAK_METADATA.txt"
echo "  metadata written to results/$OUT/DOCKLEAK_METADATA.txt"
echo "=== dockleak instrumented run complete ==="
