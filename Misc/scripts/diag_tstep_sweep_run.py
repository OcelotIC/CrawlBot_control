#!/usr/bin/env python3
"""Phase TSTEP-DIAG-ALL — run the per-step T_step sweep (measurement only).

For each step k in 0..5 and each scale in SCALES, run the canonical traversal
truncated to --n-steps (k+1) with step k's T_step scaled by `scale` (scale_step=k).
Truncating to k+1 steps makes step k the terminal step (no look-ahead ⇒ step-k
behaviour identical to the full run) and cheap for early steps. Steps 0..k-1 stay
at their canonical formula T_step. Concurrency 3 (4 cores). No canonical change:
default gain stays 0; the sweep only sets the diagnostic per-step scale.

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/diag_tstep_sweep_run.py
"""
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

CANON = [
    '--ss-two-task', '--ss-alpha-mom', '5000', '--alpha-torso-pose', '24000',
    '--ss-kp-torso', '3', '--ss-kd-torso', '2.5',
    '--aocs_mode', 'legacy_pid_numerical', '--K_omega', '50',
    '--qp-envelope-exact',
    '--interstep-settle-alpha-wrench', '3', '--interstep-settle-epsilon-v', '5e-3',
    '--tau_w_max', '5',
]
SCALES = [1.0, 1.1, 1.25, 1.5, 2.0, 2.5]
STEPS = [0, 1, 2, 3, 4, 5]
SCRATCH = ('/tmp/claude-0/-home-user-CrawlBot-control/'
           '51ef2e0d-fbab-59f4-a937-ec828e0d7eda/scratchpad')


def one(k, scale):
    tag = f'sw_s{k}_x{scale:g}'
    out = f'figC_{tag}'
    if os.path.exists(f'results/{out}/sim_log.json'):
        return (k, scale, 'cached')
    cmd = (['python3', 'scripts/diag_cooperative_arms.py'] + CANON
           + ['--n-steps', str(k + 1),
              '--preplanner-tstep-scale-step', str(k),
              '--preplanner-tstep-scale-factor', f'{scale:g}',
              '--out-dir', out])
    env = dict(os.environ, MUJOCO_GL='disabled', PYTHONPATH='.')
    with open(f'{SCRATCH}/{tag}.log', 'w') as lf:
        r = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    ok = os.path.exists(f'results/{out}/sim_log.json')
    return (k, scale, 'ok' if ok else f'FAIL rc={r.returncode}')


jobs = [(k, s) for k in STEPS for s in SCALES]
print(f'launching {len(jobs)} runs, concurrency 3', flush=True)
done = 0
with ThreadPoolExecutor(max_workers=3) as ex:
    futs = {ex.submit(one, k, s): (k, s) for k, s in jobs}
    for f in as_completed(futs):
        k, s, status = f.result()
        done += 1
        print(f'[{done}/{len(jobs)}] step{k} x{s:g}: {status}', flush=True)
open(f'{SCRATCH}/tstep_sweep_DONE', 'w').write('done\n')
print('ALL DONE', flush=True)
