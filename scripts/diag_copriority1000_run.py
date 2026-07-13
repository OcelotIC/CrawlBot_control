#!/usr/bin/env python3
"""Phase COPRIORITY-1000-FINAL — Idriss's co-priority 1:1:1 weight design (measurement).

Full weight vector forced via monkeypatch (unambiguous single source of truth):
  hw-slack 10000 | torso 1000 | momentum 1000 | EE 1000 | posture 20 |
  torque 1 | wrench 1 | accel-reg 1 | eps 1e-6.   Span = 10000/1 = 1e4.
Runs canonical C, then extracts feasibility / SS-saturation / at-weld docks / e_com /
theta_s / realized+planned Hdot_s / kappa_SS / h_w peak / solver status.

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_copriority1000_run.py
"""
import os
import json
import numpy as np
import crawlbot.solvers.hierarchical_qp as hq
import crawlbot.solvers.wholebody_qp as wq

EPS = 1e-6
WEIGHTS = dict(alpha_torso_pose=2000.0, w_hw_slack=800.0, ss_alpha_mom=400.0,
               alpha_ee=1000.0, alpha_posture=20.0, alpha_wrench=1.0,
               alpha_torque=1.0, alpha_reg=1.0)
OUT = 'figC_copri_hw800'
RESULT_JSON = 'results/j2_adjconv/copri_hw800_result.json'

_orig_init = wq.WholeBodyQP.__init__
def _pinit(self, config=None):
    _orig_init(self, config)
    for k, v in WEIGHTS.items():
        setattr(self.config, k, v)
wq.WholeBodyQP.__init__ = _pinit

CAP = []; CALLS = [0]
_orig_w = hq.HierarchicalQP._solve_weighted
def _asm(tasks, n):
    H = np.zeros((n, n))
    for t in tasks:
        H += (t.A.T @ t.W) @ t.A
    return 0.5 * (H + H.T)
def _pw(self, tasks, x0):
    self.regularization = EPS
    idx = CALLS[0]; CALLS[0] += 1
    if idx % 20 == 0:
        try:
            w = np.linalg.eigvalsh(_asm(tasks, self.n_vars))
            ne = 0 if self._C_eq is None else int(np.atleast_2d(self._C_eq).shape[0])
            CAP.append((idx, ne, float(w[0]), float(w[-1])))
        except np.linalg.LinAlgError:
            pass
    return _orig_w(self, tasks, x0)
hq.HierarchicalQP._solve_weighted = _pw

import scripts.diag_cooperative_arms as dca
from scripts.diag_cooperative_arms import _mutate_mjcf, _mjcf_md5, MJCF

with open(MJCF) as f:
    _original = f.read()
_pre = _mjcf_md5(MJCF)
try:
    _mutate_mjcf(damping=0.0, armature=0.05, anchor_dx=0.8, mass_ratio=0.01)
    try:
        dca.main(
            legacy=False, alpha_torso_lin=0.0, anchor_dx=0.8, mass_ratio=0.01,
            aocs_mode='legacy_pid_numerical', settle_seconds=20.0,
            K_theta=1.0, K_omega=50.0, tau_w_max=5.0,
            n_steps=6, ss_two_task=True, ss_alpha_mom=400.0,
            alpha_torso_pose=2000.0, ss_alpha_ee=1000.0, ss_alpha_posture=2e1,
            ss_alpha_wrench=1.0, ss_kp_torso=3.0, ss_kd_torso=2.5,
            qp_envelope_exact=True,
            interstep_settle_alpha_wrench=3.0, interstep_settle_epsilon_v=5e-3,
            out_dir_override=OUT)
    except Exception as e:
        print(f'[note] main() raised after sim: {type(e).__name__}: {str(e)[:160]}')
finally:
    with open(MJCF, 'w') as f:
        f.write(_original)
    assert _mjcf_md5(MJCF) == _pre, 'MJCF restore failed'

# ── extract ──
import mujoco
from scripts.export_figure_data import load_anchors_struct_frame, phase_per_tick, MJCF as MJ2
model = mujoco.MjModel.from_xml_path(MJ2)
STANCE_ARM = ['a', 'b', 'a', 'b', 'a', 'b']; STANCE_ANCHOR = [2, 3, 3, 4, 4, 5]
sl = json.load(open(f'results/{OUT}/sim_log.json'))
si = np.asarray(sl['step_idx']); ph = np.array(phase_per_tick(sl)); is_ss = np.array([p == 'SS' for p in ph])
lam = np.asarray(sl['lambda_qp'], float); lref = np.asarray(sl['lambda_ref'], float)
eul = np.asarray(sl['struct_euler_deg'], float); ecom = np.asarray(sl.get('e_com', []), float)
hwp = np.asarray(sl['hw_physical'], float)
dev = {e['step']: e for e in sl.get('dock_events', [])}
weld = [round(dev[k]['d_mm'], 3) if k in dev else None for k in range(6)]
n_dock = len(dev); all6 = (n_dock == 6)
aA, aB = load_anchors_struct_frame(model, np.asarray(sl['struct_pos'])[0], np.asarray(sl['struct_quat'])[0])
ssr = []; ssp = []
for k in range(6):
    anch = (aA if STANCE_ARM[k] == 'a' else aB)[STANCE_ANCHOR[k]]
    off = 0 if STANCE_ARM[k] == 'a' else 6
    m = (si == k) & is_ss
    hr = hp_ = 0.0
    for i in np.where(m)[0]:
        hr = max(hr, float(np.max(np.abs(np.cross(anch, lam[i, off:off+3]) + lam[i, off+3:off+6]))))
        if np.all(np.isfinite(lref[i])):
            hp_ = max(hp_, float(np.max(np.abs(np.cross(anch, lref[i, off:off+3]) + lref[i, off+3:off+6]))))
    ssr.append(round(hr, 3)); ssp.append(round(hp_, 3))
sat = 'INTACT' if (ssr[2] >= 4.9 and ssr[4] >= 4.9) else ('GONE' if max(ssr) < 3.2 else 'PARTIAL')
th = np.linalg.norm(eul, axis=1)
ecom_pk = float(np.max(ecom[is_ss])) if ecom.size and is_ss.any() else float('nan')
cap = np.array(CAP, float); kap = None
if len(cap):
    ss = cap[cap[:, 1] == 26]
    if len(ss):
        kap = float((np.median(ss[:, 3]) + EPS) / (np.median(ss[:, 2]) + EPS))
        lmin = float(np.median(ss[:, 2]))
worst = max([w for w in weld if w is not None], default=float('nan'))
res = dict(weights=WEIGHTS, eps=EPS, span=10000.0,
           feasible_6of6=bool(all6), n_dock=n_dock,
           at_weld_docks=weld, worst=round(worst, 3) if all6 else None,
           margin=round(5.0 - worst, 3) if all6 else None,
           ss_hdot_realized=ssr, ss_hdot_planned=ssp, saturation=sat,
           e_com_pk=round(ecom_pk, 5), theta_pk=round(float(th.max()), 4),
           theta_settled=round(float(th[-1]), 4), kappa_SS=kap,
           lambda_min_LS=round(lmin, 4) if len(cap) else None,
           hw_peak=round(float(np.max(np.linalg.norm(hwp, axis=1))), 4),
           qp_fail=int(np.sum(~np.asarray(sl['qp_ok']))),
           nmpc_fail=int(np.sum(~np.asarray(sl['nmpc_ok']))))
json.dump(res, open(RESULT_JSON, 'w'), indent=2)
print('\n================ COPRIORITY-1000-FINAL ================')
print('weights:', WEIGHTS, 'eps', EPS, 'span 1e4')
print(f'FEASIBLE 6/6: {all6}  (docks={n_dock})   qp_fail={res["qp_fail"]} nmpc_fail={res["nmpc_fail"]}')
print('at-weld docks:', weld, ' worst', res['worst'], 'margin', res['margin'])
print(f'SS saturation: {sat}')
print('SS Hdot realized:', ssr)
print('SS Hdot planned :', ssp)
print(f'e_com_pk={res["e_com_pk"]} (canon 0.095, userw2 0.137)  theta_s pk={res["theta_pk"]} settled={res["theta_settled"]}')
print(f'kappa_SS={kap:.3e}  lambda_min(H_LS)={res.get("lambda_min_LS")}  (canon kappa 3.6e6)' if kap else 'kappa: n/a')
print(f'hw_peak={res["hw_peak"]} Nms (box +-5)')
print('wrote results/j2_adjconv/copri1000_result.json')
