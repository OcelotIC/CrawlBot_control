#!/usr/bin/env python3
"""T4b — prove the config differs from T4 by EXACTLY ONE field (t_settle_final).

Patches SimulationLoop.__init__ to snapshot dataclasses.asdict(cfg) and abort
BEFORE sim.setup()/run() (fast, no sim). Calls dca.main with the byte-identical
T4 kwargs at settle_seconds=450 and settle_seconds=900, then diffs the two cfgs.

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/diag_t4b_cfg_diff.py
"""
import dataclasses
import numpy as np
import crawlbot.simulation.sim_loop as slmod

CFGS = {}


class _StopInit(Exception):
    pass


_orig_init = slmod.SimulationLoop.__init__
_tag = {'v': None}


def _patched_init(self, *a, **k):
    cfg = k.get('config', a[2] if len(a) > 2 else None)
    CFGS[_tag['v']] = dataclasses.asdict(cfg)
    raise _StopInit


slmod.SimulationLoop.__init__ = _patched_init

import scripts.diag_cooperative_arms as dca


def _run_cfg(settle):
    _tag['v'] = settle
    try:
        dca.main(
            legacy=False, alpha_torso_lin=0.0, anchor_dx=0.8, mass_ratio=0.01,
            aocs_mode='legacy_pid_numerical', settle_seconds=settle,
            K_theta=1.0, K_omega=50.0, tau_w_max=2.5,
            n_steps=6, ss_two_task=True, ss_alpha_mom=400.0,
            alpha_torso_pose=2000.0, ss_alpha_ee=1000.0, ss_alpha_posture=2e1,
            ss_alpha_wrench=1.0, ss_kp_torso=3.0, ss_kd_torso=2.5,
            qp_envelope_exact=True,
            interstep_settle_alpha_wrench=3.0, interstep_settle_epsilon_v=5e-3,
            out_dir_override='figC25_cfgdiff_scratch')
    except _StopInit:
        pass


def _equal(x, y):
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        return np.array_equal(np.asarray(x), np.asarray(y))
    if isinstance(x, float) and isinstance(y, float):
        return x == y or (np.isnan(x) and np.isnan(y))
    return x == y


_run_cfg(450.0)
_run_cfg(900.0)
c450, c900 = CFGS[450.0], CFGS[900.0]
keys = sorted(set(c450) | set(c900))
diffs = []
for kk in keys:
    if kk not in c450 or kk not in c900:
        diffs.append((kk, c450.get(kk, '<absent>'), c900.get(kk, '<absent>')))
    elif not _equal(c450[kk], c900[kk]):
        diffs.append((kk, c450[kk], c900[kk]))

print('=' * 70)
print('T4b CONFIG DIFF  (T4 settle=450  vs  T4b settle=900)')
print(f'  total cfg fields compared: {len(keys)}')
print(f'  differing fields: {len(diffs)}')
for kk, a, b in diffs:
    print(f'    {kk}: {a}  ->  {b}')
ok = (len(diffs) == 1 and diffs[0][0] == 't_settle_final'
      and diffs[0][1] == 450.0 and diffs[0][2] == 900.0)
print(f'  VERDICT: {"EXACTLY ONE FIELD (t_settle_final 450->900) ✓" if ok else "NOT one-field — SEE ABOVE ✗"}')
print('=' * 70)
