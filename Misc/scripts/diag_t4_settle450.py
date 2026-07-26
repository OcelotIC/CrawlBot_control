#!/usr/bin/env python3
"""Phase DRIFT-CLOSURE SUB-PHASE B (T4) — extended 450 s terminal settle.

Re-runs the FROZEN 2.5 canonical MANAGED (C) scenario with the trailing DS
settle extended from 20 s to 450 s (settle_seconds -> cfg.t_settle_final,
sim_loop.py:2482, the "Trailing DS (end of gait)" branch — terminal-only).
EVERYTHING else is byte-identical to Misc/scripts/diag_canonical2p5_run.py's C run:
same MJCF mutation, same dca.main kwargs, same capture patches. The canonical
freeze 32aefaf control path is untouched (this branch differs only by the
logging-only torso-export fix + the T2 exporter columns).

Output dir: figC25_t4_450s (NEW — the canonical figC25_addfive is NOT touched).

Adds ONE measurement-only patch beyond the canonical runner: a decimated
(t, qpos, qvel) capture on every mj_step (post-step, never read by control),
saved to results/figC25_t4_450s/ltot_dense.json, so L_total can be evaluated
via mj_subtreeVel at ~1 s resolution THROUGHOUT the 450 s settle (the sim only
snapshots dock_step5 + final in that window).

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/diag_t4_settle450.py
"""
import json
import numpy as np
import mujoco
import crawlbot.solvers.hierarchical_qp as hq
import crawlbot.solvers.wholebody_qp as wq
import crawlbot.solvers.nmpc_solver as ns
import argparse

# Parameterized (T4: defaults; T4b: --settle 900 --out figC25_t4b_900s ...).
# The ONLY control-relevant knob is --settle (-> cfg.t_settle_final); --out,
# --result-tag, --decimate are output-path / measurement-cadence only.
_ap = argparse.ArgumentParser()
_ap.add_argument('--settle', type=float, default=450.0)
_ap.add_argument('--out', default='figC25_t4_450s')
_ap.add_argument('--result-tag', default='t4_settle450')
_ap.add_argument('--decimate', type=float, default=1.0)
_args, _ = _ap.parse_known_args()

CAP = 2.5
HMAX = 5.0
EPS = 1e-6
SETTLE = _args.settle
OUT = _args.out
FROZEN = dict(alpha_torso_pose=2000.0, alpha_ee=1000.0, ss_alpha_mom=400.0,
              w_hw_slack=800.0, alpha_posture=20.0, alpha_torque=5.0,
              alpha_wrench=1.0, alpha_reg=1.0)
RESULT_JSON = f'results/j2_adjconv/{_args.result_tag}_result.json'
DENSE_JSON = f'results/{OUT}/ltot_dense.json'
DECIMATE_S = _args.decimate      # dense (qpos,qvel) capture cadence [s of sim time]

# ── capture-only patches (verbatim from diag_canonical2p5_run.py) ────
EFF_W = []
_orig_init = wq.WholeBodyQP.__init__
def _pinit(self, config=None):
    _orig_init(self, config)
    d = {k: float(getattr(self.config, k)) for k in FROZEN}
    d['weight_ratio'] = float(self.config.weight_ratio)
    d['tau_w_max_qp'] = float(self.config.tau_w_max)
    EFF_W.append(d)
wq.WholeBodyQP.__init__ = _pinit

KAP = []; _KCALL = [0]
_orig_w = hq.HierarchicalQP._solve_weighted
def _pw(self, tasks, x0):
    self.regularization = EPS
    i = _KCALL[0]; _KCALL[0] += 1
    if i % 20 == 0:
        try:
            H = np.zeros((self.n_vars, self.n_vars))
            for t in tasks:
                H += (t.A.T @ t.W) @ t.A
            w = np.linalg.eigvalsh(0.5 * (H + H.T))
            ne = 0 if self._C_eq is None else int(np.atleast_2d(self._C_eq).shape[0])
            KAP.append((ne, float(w[0]), float(w[-1])))
        except np.linalg.LinAlgError:
            pass
    return _orig_w(self, tasks, x0)
hq.HierarchicalQP._solve_weighted = _pw

# decimated (t, qpos, qvel) capture + wheel actuator-force tracking
DENSE = []                       # [(t, qpos.list, qvel.list)]
_next_cap = [0.0]
FR = {'n': 0, 'clamped': 0, 'maxf': 0.0}
_orig_step = mujoco.mj_step
def _pstep(m, d, *a, **k):
    _orig_step(m, d, *a, **k)
    if d.actuator_force.shape[0] >= 17:
        af = float(np.max(np.abs(d.actuator_force[14:17])))
        FR['n'] += 1
        FR['maxf'] = max(FR['maxf'], af)
        if af >= CAP - 1e-6:
            FR['clamped'] += 1
    if d.time >= _next_cap[0] - 1e-9:
        DENSE.append((float(d.time),
                      d.qpos.copy().tolist(), d.qvel.copy().tolist()))
        _next_cap[0] = d.time + DECIMATE_S
mujoco.mj_step = _pstep

import scripts.diag_cooperative_arms as dca
from scripts.diag_cooperative_arms import _mutate_mjcf, _mjcf_md5, MJCF
from scripts.export_figure_data import phase_per_tick, MJCF as MJ2

res = {'cap': CAP, 'h_max': HMAX, 'eps': EPS, 'settle_seconds': SETTLE,
       'out_dir': OUT, 'frozen_weights': FROZEN}

with open(MJCF) as f:
    _original = f.read()
_pre = _mjcf_md5(MJCF)
try:
    _mutate_mjcf(damping=0.0, armature=0.05, anchor_dx=0.8, mass_ratio=0.01)
    print(f'\n########## C (managed 2.5) settle={SETTLE}s -> {OUT} ##########',
          flush=True)
    try:
        dca.main(
            legacy=False, alpha_torso_lin=0.0, anchor_dx=0.8, mass_ratio=0.01,
            aocs_mode='legacy_pid_numerical', settle_seconds=SETTLE,
            K_theta=1.0, K_omega=50.0, tau_w_max=2.5,
            n_steps=6, ss_two_task=True, ss_alpha_mom=400.0,
            alpha_torso_pose=2000.0, ss_alpha_ee=1000.0, ss_alpha_posture=2e1,
            ss_alpha_wrench=1.0, ss_kp_torso=3.0, ss_kd_torso=2.5,
            qp_envelope_exact=True,
            interstep_settle_alpha_wrench=3.0, interstep_settle_epsilon_v=5e-3,
            out_dir_override=OUT)
    except Exception as e:
        print(f'[note] main() raised after sim: {type(e).__name__}: {str(e)[:160]}',
              flush=True)
finally:
    with open(MJCF, 'w') as f:
        f.write(_original)
    assert _mjcf_md5(MJCF) == _pre, 'MJCF restore failed'

# persist dense captures for offline L_total
json.dump({'decimate_s': DECIMATE_S, 'n': len(DENSE), 'captures': DENSE},
          open(DENSE_JSON, 'w'))
res['weights_effective'] = EFF_W[0] if EFF_W else None
res['weights_match_frozen'] = bool(EFF_W) and all(
    all(abs(d[k] - v) < 1e-9 for k, v in FROZEN.items()) for d in EFF_W)
res['applied_force_max'] = round(FR['maxf'], 6)
res['mjstep_total'] = FR['n']; res['mjstep_clamped'] = FR['clamped']
res['n_dense_captures'] = len(DENSE)
json.dump(res, open(RESULT_JSON, 'w'), indent=2)
print(f"\n[done] wrote results/{OUT}/sim_log.json + {DENSE_JSON} "
      f"({len(DENSE)} dense captures) + {RESULT_JSON}")
print(f"[sanity] weights_match_frozen={res['weights_match_frozen']} "
      f"applied_force_max={res['applied_force_max']} "
      f"clamped={FR['clamped']}/{FR['n']}", flush=True)
