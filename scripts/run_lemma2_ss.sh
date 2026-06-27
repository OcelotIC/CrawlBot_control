#!/usr/bin/env bash
# J1 closing — Lemma 2 SS test: instrumented re-run of the Fix-A canonical
# (main = ae0673e) with the per-QP-substep Lemma-2 readout (HR_Os + plant weld
# wrench). Same canonical flags as results/fixA_gate (the 6/6 gate); writes to a
# NEW dir results/lemma2_ss/ (canonical untouched). log_hifreq_ss is on (diag
# sets it), so the SS-only instrument fires. READ-ONLY instrument; the run's
# shared per-tick fields must be bit-identical to fixA_gate.
set -u
cd "$(dirname "$0")/.."
SP=/tmp/claude-0/-home-user-CrawlBot-control/51ef2e0d-fbab-59f4-a937-ec828e0d7eda/scratchpad
name=lemma2_ss
echo "=== [$name] instrumented Fix-A canonical re-run (Lemma-2 SS readout) ==="
t0=$SECONDS
FRAMES_PER_STEP=0 MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_cooperative_arms.py \
  --ss-two-task --ss-alpha-mom 5000 --alpha-torso-pose 24000 \
  --ss-kp-torso 3 --ss-kd-torso 2.5 --aocs_mode legacy_pid_numerical --K_omega 50 \
  --n-steps 5 --out-dir "$name" > "$SP/${name}.log" 2>&1
echo "  diag exit=$? dt=$((SECONDS-t0))s"
{ echo "Lemma-2 SS instrumented re-run"; echo "HEAD: $(git rev-parse HEAD)";
  echo "branch: $(git rev-parse --abbrev-ref HEAD)";
  echo "flags: same as fixA_gate (canonical Fix-A WP)";
  echo "instrument: per-QP-substep SS HR_Os (robot-subtree angmom about O_s, world) + plant weld wrench (qfrc_constraint via world relative-site Jacobian)";
} > "results/$name/LEMMA2_METADATA.txt"
echo "=== done [$name] ==="
