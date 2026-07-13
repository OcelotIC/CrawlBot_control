#!/usr/bin/env python3
"""Quick try — user-specified full WBC weight stack (measurement, no canonical change).

Forces the SS weighted stack to:
  torso-pose 1000, hw-slack 800, momentum 400, swing-EE 600, posture 20,
  torque-min 5, wrench-track 1, accel-reg 1, Tikhonov eps 1e-4.
(torso/mom/EE/posture/wrench are CLI-settable; hw-slack/torque/accel-reg are hardcoded
in _build_qp and eps is the HierarchicalQP default — so ALL are forced via monkeypatch
for an unambiguous single source of truth.) Runs canonical C (EE-baseline gains), full
6-step, then extracts docks / theta_s / realized Hdot_s / kappa / feasibility.

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_userweights_run.py
"""
import json
import numpy as np
import mujoco
import crawlbot.solvers.hierarchical_qp as hq
import crawlbot.solvers.wholebody_qp as wq

EPS = 1e-4
WEIGHTS = dict(alpha_torso_pose=1000.0, w_hw_slack=800.0, ss_alpha_mom=400.0,
               alpha_ee=600.0, alpha_posture=20.0, alpha_wrench=1.0,
               alpha_torque=5.0, alpha_reg=1.0)
OUT = 'figC_userw'

_orig_init = wq.WholeBodyQP.__init__
def _patched_init(self, config=None):
    _orig_init(self, config)
    for k, v in WEIGHTS.items():
        setattr(self.config, k, v)
wq.WholeBodyQP.__init__ = _patched_init

CAP = []; CALLS = [0]
_orig_w = hq.HierarchicalQP._solve_weighted
def _assemble_LS(tasks, n):
    H = np.zeros((n, n))
    for t in tasks:
        H += (t.A.T @ t.W) @ t.A
    return 0.5 * (H + H.T)
def _patched_w(self, sorted_tasks, x0):
    self.regularization = EPS
    idx = CALLS[0]; CALLS[0] += 1
    if idx % 20 == 0:
        n = self.n_vars
        try:
            w = np.linalg.eigvalsh(_assemble_LS(sorted_tasks, n))
            n_eq = 0 if self._C_eq is None else int(np.atleast_2d(self._C_eq).shape[0])
            CAP.append((idx, n_eq, float(w[0]), float(w[-1])))
        except np.linalg.LinAlgError:
            pass
    return _orig_w(self, sorted_tasks, x0)
hq.HierarchicalQP._solve_weighted = _patched_w

import scripts.diag_cooperative_arms as dca
from scripts.diag_cooperative_arms import _mutate_mjcf, _mjcf_md5, MJCF
from scripts.export_figure_data import load_anchors_struct_frame, phase_per_tick

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
            alpha_torso_pose=1000.0, ss_alpha_ee=600.0, ss_alpha_posture=2e1,
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

# ---- extract ----
model = mujoco.MjModel.from_xml_path(MJCF)
STANCE_ARM = ['a', 'b', 'a', 'b', 'a', 'b']; STANCE_ANCHOR = [2, 3, 3, 4, 4, 5]
BASE = [4.9401, 4.4054, 4.9039, 4.4358, 4.0453, 4.9999]
sl = json.load(open(f'results/{OUT}/sim_log.json'))
si = np.asarray(sl['step_idx']); ph = np.array(phase_per_tick(sl))
dg = np.asarray(sl['d_grip_swing'], float); lam = np.asarray(sl['lambda_qp'], float)
eul = np.asarray(sl['struct_euler_deg'], float)
docks = []
for k in range(6):
    m = (si == k) & (ph == 'SS') & np.isfinite(dg)
    docks.append(round(float(1000 * dg[m].min()), 4) if m.any() else float('nan'))
th = np.linalg.norm(eul, axis=1)
aA, aB = load_anchors_struct_frame(model, np.asarray(sl['struct_pos'])[0],
                                   np.asarray(sl['struct_quat'])[0])
hdot = []
for k in range(6):
    anch = (aA if STANCE_ARM[k] == 'a' else aB)[STANCE_ANCHOR[k]]
    off = 0 if STANCE_ARM[k] == 'a' else 6
    ss = (si == k) & (ph == 'SS'); hpk = 0.0
    for i in np.where(ss)[0]:
        hpk = max(hpk, float(np.max(np.abs(np.cross(anch, lam[i, off:off+3]) + lam[i, off+3:off+6]))))
    hdot.append(round(hpk, 3))
cap = np.array(CAP, float)
kap_ss = kap_ds = lam_min = float('nan')
if len(cap):
    ss = cap[cap[:, 1] == 26]; ds = cap[cap[:, 1] == 32]
    lam_min = float(cap[:, 2].min())
    if len(ss):
        kap_ss = float((ss[-1, 3] + EPS) / (ss[-1, 2] + EPS))
    if len(ds):
        kap_ds = float((ds[0, 3] + EPS) / (ds[0, 2] + EPS))
worst = float(np.nanmax(docks))
res = dict(weights=WEIGHTS, eps=EPS, docks=docks, worst=round(worst, 4),
           margin=round(5.0 - worst, 4), n_dock=len(sl.get('dock_events', [])),
           all6=bool(sum(1 for d in docks if d < 5.0) == 6 and not any(np.isnan(docks))),
           theta_s_peak=round(float(th.max()), 4), theta_s_settled=round(float(th[-1]), 4),
           hdot_s_pk=hdot, kappa_SS=kap_ss, kappa_DS=kap_ds, min_lam_min_LS=lam_min,
           dock_vs_canonical=[round(d - b, 4) for d, b in zip(docks, BASE)])
json.dump(res, open('results/j2_adjconv/userweights_result.json', 'w'), indent=2)

print('\n================ USER-WEIGHT TRY ================')
print(f'weights: {WEIGHTS}  eps={EPS}')
print(f'docks [mm]:   ' + '  '.join(f's{k}:{d:.4f}' for k, d in enumerate(docks)))
print(f'vs canonical: ' + '  '.join(f's{k}:{d:+.4f}' for k, d in enumerate(res["dock_vs_canonical"])))
print(f'worst={worst:.4f}mm  margin={res["margin"]:.4f}  all6={res["all6"]}  docks={res["n_dock"]}')
print(f'theta_s peak={res["theta_s_peak"]:.4f} settled={res["theta_s_settled"]:.4f} (canon 0.594/0.107)')
print(f'realized |Hdot_s|pk: ' + '  '.join(f's{k}:{v:.3f}' for k, v in enumerate(hdot)) + '  (canon s2/s4=5.00)')
print(f'kappa(H): SS={kap_ss:.3e} DS={kap_ds:.3e}  min lam_min(H_LS)={lam_min:.4e}  (canon kappa~3.6e6)')
