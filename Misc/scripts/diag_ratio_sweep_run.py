#!/usr/bin/env python3
"""Phase EE-RATIO-GLOBAL — sweep the UNIFORM torso:EE weight ratio (measurement).

No code change: --alpha-torso-pose and --ss-alpha-ee are existing CLI flags, so
each config is a full canonical C run with different flag values (EE=3000 and
momentum=5000 held unless noted). Ratio held CONSTANT over the whole swing (global).

Route A (lower torso-pose, EE=3000):  torso in {12000,8000,6000,4000,3000}
    -> torso:EE = {4, 2.67, 2, 1.33, 1}:1
Route B (raise EE, torso=24000):      EE in {6000,9000}
    -> torso:EE = {4, 2.67}:1
Baseline (torso 24000 / EE 3000 = 8:1) is reused from results/figC_sw_s5_x1
(bit-identical canonical, scale-factor 1.0).

Concurrency 3 (4 cores). MJCF mutation content is identical across configs
(anchor_dx=0.8, mass_ratio=0.01) so concurrent subprocess runs are race-tolerant
(same as diag_tstep_sweep_run.py).

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/diag_ratio_sweep_run.py
"""
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

CANON_BASE = [
    '--ss-two-task', '--ss-alpha-mom', '5000',
    '--ss-kp-torso', '3', '--ss-kd-torso', '2.5',
    '--aocs_mode', 'legacy_pid_numerical', '--K_omega', '50',
    '--qp-envelope-exact',
    '--interstep-settle-alpha-wrench', '3', '--interstep-settle-epsilon-v', '5e-3',
    '--tau_w_max', '5', '--n-steps', '6',
]
# (label, torso_pose, ee)
CONFIGS = [
    ('ratio_t12000', 12000, 3000),
    ('ratio_t8000', 8000, 3000),
    ('ratio_t6000', 6000, 3000),
    ('ratio_t4000', 4000, 3000),
    ('ratio_t3000', 3000, 3000),
    ('ratio_ee6000', 24000, 6000),
    ('ratio_ee9000', 24000, 9000),
]
SCRATCH = ('/tmp/claude-0/-home-user-CrawlBot-control/'
           '51ef2e0d-fbab-59f4-a937-ec828e0d7eda/scratchpad')


def one(label, torso, ee):
    out = f'figC_{label}'
    if os.path.exists(f'results/{out}/sim_log.json'):
        return (label, 'cached')
    cmd = (['python3', 'scripts/diag_cooperative_arms.py'] + CANON_BASE
           + ['--alpha-torso-pose', str(torso), '--ss-alpha-ee', str(ee),
              '--out-dir', out])
    env = dict(os.environ, MUJOCO_GL='disabled', PYTHONPATH='.')
    with open(f'{SCRATCH}/{label}.log', 'w') as lf:
        r = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    ok = os.path.exists(f'results/{out}/sim_log.json')
    return (label, 'ok' if ok else f'FAIL rc={r.returncode}')


print(f'launching {len(CONFIGS)} ratio runs, concurrency 3', flush=True)
done = 0
with ThreadPoolExecutor(max_workers=3) as ex:
    futs = {ex.submit(one, lab, t, e): lab for (lab, t, e) in CONFIGS}
    for f in as_completed(futs):
        lab, status = f.result()
        done += 1
        print(f'[{done}/{len(CONFIGS)}] {lab}: {status}', flush=True)
open(f'{SCRATCH}/ratio_sweep_DONE', 'w').write('done\n')
print('ALL DONE', flush=True)
