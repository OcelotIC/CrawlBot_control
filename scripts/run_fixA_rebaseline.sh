#!/usr/bin/env bash
# Dock-leak Fix-A merge — Phase 1: re-baseline C6(a) on the branch (NO merge).
# Establishes the Fix-A-corrected-plant flag-OFF reference and re-runs the gate
# with C6(a) re-pointed to it. Single pass — if any criterion moves, STOP/report.
#
#   1. ssmom_phase1_baseline_fixA = Fix-A plant, ss_two_task OFF (frozen ref)
#   2. phase3_off                 = a fresh Fix-A-plant OFF run (gate OFF run)
#   3. byte-identical (2) vs (1)  -> OFF_RESULT.txt  (expect Δ=0: determinism +
#      ss_two_task feature-gating intact on the corrected plant) + test 8/8
#   4. gate_phase3.py (BASE now = ssmom_phase1_baseline_fixA) on the Fix-A WP
#      results/fixA_gate/  -> six criteria
set -u
cd "$(dirname "$0")/.."
SP=/tmp/claude-0/-home-user-CrawlBot-control/51ef2e0d-fbab-59f4-a937-ec828e0d7eda/scratchpad
OFF_FLAGS="--aocs_mode legacy_pid_numerical --n-steps 5"   # ss_two_task OFF (default)

echo "=== 1. Fix-A-plant OFF reference -> ssmom_phase1_baseline_fixA ==="
t0=$SECONDS
FRAMES_PER_STEP=0 MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_cooperative_arms.py \
  $OFF_FLAGS --out-dir ssmom_phase1_baseline_fixA > "$SP/baseline_fixA.log" 2>&1
echo "  diag exit=$? dt=$((SECONDS-t0))s"
SSMOM_RUN_DIR=ssmom_phase1_baseline_fixA MUJOCO_GL=disabled PYTHONPATH=. \
  python3 scripts/postprocess_results_figs.py > "$SP/baseline_fixA_pp.log" 2>&1
echo "  ref postproc exit=$? metrics=$(test -f results/ssmom_phase1_baseline_fixA/postproc_metrics.json && echo Y || echo N)"

echo "=== 2. fresh Fix-A-plant OFF run -> phase3_off ==="
t0=$SECONDS
FRAMES_PER_STEP=0 MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_cooperative_arms.py \
  $OFF_FLAGS --out-dir phase3_off > "$SP/phase3_off.log" 2>&1
echo "  diag exit=$? dt=$((SECONDS-t0))s"
SSMOM_RUN_DIR=phase3_off MUJOCO_GL=disabled PYTHONPATH=. \
  python3 scripts/postprocess_results_figs.py > "$SP/phase3_off_pp.log" 2>&1

echo "=== 3. byte-identical: phase3_off vs ssmom_phase1_baseline_fixA (C6a) ==="
MUJOCO_GL=disabled PYTHONPATH=. python3 -c "
import json,numpy as np
a=json.load(open('results/ssmom_phase1_baseline_fixA/sim_log.json'))
b=json.load(open('results/phase3_off/sim_log.json'))
skip={'qp_time_ms','nmpc_time_ms','t_ss_hifreq','tau_w_ss_hifreq','hw_ss_hifreq','environment'}
worst=0.0; nz=[]
for k in set(a)&set(b):
    if k in skip: continue
    try: xa=np.asarray(a[k],float); xb=np.asarray(b[k],float)
    except: continue
    if xa.shape!=xb.shape: nz.append((k,'SHAPE')); continue
    if xa.size:
        d=np.abs(xa-xb); d=d[np.isfinite(d)]; m=float(d.max()) if d.size else 0.0
        worst=max(worst,m)
        if m>0: nz.append((k,round(float(m),6)))
verdict='BIT-IDENTICAL' if worst==0 else 'DIFFERS'
open('results/phase3_off/OFF_RESULT.txt','w').write(
  'flag-OFF vs ssmom_phase1_baseline_fixA (Fix-A plant, AOCS=legacy_pid_numerical): %s ; worst|delta|=%.2e ; nonzero=%s'%(verdict,worst,nz or 'NONE'))
print('  ', open('results/phase3_off/OFF_RESULT.txt').read())
"
echo "=== test_reworked_qp ==="
TQ=$(MUJOCO_GL=disabled PYTHONPATH=. python3 -m pytest tests/test_reworked_qp.py -q --tb=line 2>&1 | grep -aE '[0-9]+ passed|[0-9]+ failed')
echo "  $TQ"; echo "test_reworked_qp: $TQ" >> results/phase3_off/OFF_RESULT.txt

echo "=== 4. gate (re-baselined C6; BASE=ssmom_phase1_baseline_fixA) on fixA_gate WP ==="
SSMOM_WP_DIR=fixA_gate SSMOM_OFF_DIR=phase3_off MUJOCO_GL=disabled PYTHONPATH=. \
  python3 scripts/gate_phase3.py 2>&1 | grep -aE 'C[1-6]|per-step|NON-binding|BINDING|B15'
echo "=== done (Phase 1) ==="
