#!/usr/bin/env bash
# Phase 3 GATE runs, AOCS matched to the 5bca42c baseline (legacy_pid_numerical)
# for DS/AOCS parity — so the gate isolates the SS two-task change and criterion-6
# OFF is byte-identical to dcda974. (Diag default is legacy_corrected; the 5bca42c
# canonical baseline ran legacy_pid_numerical, which is the comparison the brief cites.)
# --out-dir overrides the aocs auto-name, so the committed baseline dir is untouched.
set -u
cd "$(dirname "$0")/.."
SP=/tmp/claude-0/-home-user-CrawlBot-control/51ef2e0d-fbab-59f4-a937-ec828e0d7eda/scratchpad
AOCS="--aocs_mode legacy_pid_numerical"

echo "=== [phase3_wp] working-point 5-step (AOCS=legacy_pid_numerical) ==="
t0=$SECONDS
FRAMES_PER_STEP=0 MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_cooperative_arms.py \
  --ss-two-task --ss-alpha-mom 5000 --alpha-torso-pose 20000 \
  --ss-kp-torso 3 --ss-kd-torso 2.5 $AOCS --n-steps 5 --out-dir phase3_wp \
  > "$SP/phase3_wp_run.log" 2>&1
echo "  diag exit=$? dt=$((SECONDS-t0))s"
{ echo "Phase-3 working-point run (AOCS-matched gate)"; echo "HEAD: $(git rev-parse HEAD)";
  echo "branch: $(git rev-parse --abbrev-ref HEAD)";
  echo "tree: $(test -z "$(git status --porcelain)" && echo clean || echo dirty)";
  echo "flags: --ss-two-task --ss-alpha-mom 5000 --alpha-torso-pose 20000 --ss-kp-torso 3 --ss-kd-torso 2.5 --aocs_mode legacy_pid_numerical --n-steps 5";
  echo "ee/posture/wrench: defaults 3000/20/1e-2";
  echo "AOCS: legacy_pid_numerical (matched to 5bca42c baseline for DS parity)";
} > Misc/runs/phase3_wp/PHASE3_METADATA.txt
SSMOM_RUN_DIR=phase3_wp MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/postprocess_results_figs.py \
  > "$SP/phase3_wp_postproc.log" 2>&1
echo "  WP postproc exit=$?  F3F4=$(test -f Misc/runs/phase3_wp/postproc_F3F4.csv && echo YES || echo NO)"

echo "=== [phase3_off] flag-OFF 5-step (AOCS=legacy_pid_numerical) — criterion 6 ==="
t0=$SECONDS
FRAMES_PER_STEP=0 MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_cooperative_arms.py \
  $AOCS --n-steps 5 --out-dir phase3_off > "$SP/phase3_off_run.log" 2>&1
echo "  diag exit=$? dt=$((SECONDS-t0))s"
SSMOM_RUN_DIR=phase3_off MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/postprocess_results_figs.py \
  > "$SP/phase3_off_postproc.log" 2>&1
# C6(a) re-baseline (Fix A merge): compare the flag-OFF run to the Fix-A-
# CORRECTED plant baseline, NOT the historical dcda974 plant. Fix A is an
# always-on plant correction, so dcda974-invariance is intentionally retired;
# C6(a) now checks ss_two_task feature-gating on the corrected plant.
echo "=== byte-identical: phase3_off vs ssmom_phase1_baseline_fixA (Fix-A plant) ==="
MUJOCO_GL=disabled PYTHONPATH=. python3 -c "
import json,numpy as np
a=json.load(open('Misc/runs/ssmom_phase1_baseline_fixA/sim_log.json'))
b=json.load(open('Misc/runs/phase3_off/sim_log.json'))
skip={'qp_time_ms','nmpc_time_ms','t_ss_hifreq','tau_w_ss_hifreq','hw_ss_hifreq'}
worst=0.0; nz=[]
for k in set(a)&set(b):
    if k in skip: continue
    try: xa=np.asarray(a[k],float); xb=np.asarray(b[k],float)
    except: continue
    if xa.shape!=xb.shape: nz.append((k,'SHAPE %s/%s'%(xa.shape,xb.shape))); continue
    if xa.size:
        d=np.abs(xa-xb); d=d[np.isfinite(d)]; m=float(d.max()) if d.size else 0.0
        worst=max(worst,m)
        if m>0: nz.append((k,round(m,6)))
verdict='BIT-IDENTICAL' if worst==0 else 'DIFFERS'
open('Misc/runs/phase3_off/OFF_RESULT.txt','w').write(
  'flag-OFF vs ssmom_phase1_baseline_fixA (Fix-A plant, AOCS=legacy_pid_numerical): %s ; worst|delta|=%.2e ; nonzero=%s'%(verdict,worst,nz or 'NONE'))
print('  ', open('Misc/runs/phase3_off/OFF_RESULT.txt').read())
"
echo "=== test_reworked_qp ==="
TQ=$(MUJOCO_GL=disabled PYTHONPATH=. python3 -m pytest tests/test_reworked_qp.py -q --tb=line 2>&1 | grep -aE '[0-9]+ passed|[0-9]+ failed')
echo "  $TQ"
echo "test_reworked_qp: $TQ" >> Misc/runs/phase3_off/OFF_RESULT.txt
echo "=== gate runs complete ==="