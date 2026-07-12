#!/usr/bin/env python3
"""Phase REG-DIAG — launch the epsilon sweep (5 canonical C runs, concurrency 3).

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_regdiag_sweep.py
"""
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# (tag, eps)
CONFIGS = [
    ('1em4', 1e-4),
    ('1em6', 1e-6),   # baseline (shipped default)
    ('1em8', 1e-8),
    ('1em10', 1e-10),
    ('1em12', 1e-12),
]
SCRATCH = ('/tmp/claude-0/-home-user-CrawlBot-control/'
           '51ef2e0d-fbab-59f4-a937-ec828e0d7eda/scratchpad')


def one(tag, eps):
    if os.path.exists(f'results/figC_reg_{tag}/sim_log.json'):
        return (tag, 'cached')
    cmd = ['python3', 'scripts/diag_regdiag_run_one.py', repr(eps), tag]
    env = dict(os.environ, MUJOCO_GL='disabled', PYTHONPATH='.')
    with open(f'{SCRATCH}/reg_{tag}.log', 'w') as lf:
        r = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    ok = os.path.exists(f'results/figC_reg_{tag}/sim_log.json')
    return (tag, 'ok' if ok else f'FAIL rc={r.returncode}')


print(f'launching {len(CONFIGS)} epsilon runs, concurrency 3', flush=True)
done = 0
with ThreadPoolExecutor(max_workers=3) as ex:
    futs = {ex.submit(one, t, e): t for (t, e) in CONFIGS}
    for f in as_completed(futs):
        t, status = f.result()
        done += 1
        print(f'[{done}/{len(CONFIGS)}] {t}: {status}', flush=True)
open(f'{SCRATCH}/reg_sweep_DONE', 'w').write('done\n')
print('ALL DONE', flush=True)
