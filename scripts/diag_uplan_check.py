#!/usr/bin/env python3
"""Phase U-PLAN-CHECK — does the rate-OFF (U) NMPC plan EXCEED the ±5 envelope?

Falsifies/validates the NMPC-PLAN-SATURATION reframe ("C-plan saturates z=5.0 =
management thesis"). The reframe holds ONLY IF the U-plan (Ḣ_s constraint OFF)
plans MORE aggressive Ḣ_s (> 5) on the arm-a steps where C-plan pins z=5.0 — i.e.
the constraint is what caps C. If U-plan ALSO stays ≤5, C's saturation is just
problem dynamics touching 5, not the constraint binding → reframe FAILS.

U = same base as NMPC-PLAN-SATURATION (Add-5) with τ_w_max = 1e6 (rate cap OFF:
NMPC Ḣ_s constraint inactive, WQP box inactive — dca.main tau_w_max≠5 is only a
dir-suffix, no control switch). ss_alpha_mom = 400. Change ONLY τ_w_max vs C.

Per-axis ‖·‖∞, stance-only = exact full Ḣ_s (swing_ref_pk=0 check), same
extraction as NMPC-PLAN-SATURATION. Compares against committed C mom=400.

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_uplan_check.py
"""
import os
import json
import numpy as np
import crawlbot.solvers.hierarchical_qp as hq
import crawlbot.solvers.wholebody_qp as wq
import crawlbot.solvers.nmpc_solver as ns

EPS = 1e-6
TAU_W_MAX = 1e6          # rate cap OFF
MOM = 400.0
BASE = dict(alpha_torso_pose=2000.0, w_hw_slack=800.0, alpha_ee=1000.0,
            alpha_posture=20.0, alpha_wrench=1.0, alpha_torque=5.0, alpha_reg=1.0)
OUT = 'figU_plan_check_mom400_Tw1e6'
RESULT_JSON = 'results/j2_adjconv/uplan_check_result.json'
C_JSON = 'results/j2_adjconv/nmpc_plan_sat_result.json'  # committed C reference

_orig_init = wq.WholeBodyQP.__init__
def _pinit(self, config=None):
    _orig_init(self, config)
    for k, v in BASE.items():
        setattr(self.config, k, v)
    setattr(self.config, 'ss_alpha_mom', MOM)
wq.WholeBodyQP.__init__ = _pinit

_orig_w = hq.HierarchicalQP._solve_weighted
def _pw(self, tasks, x0):
    self.regularization = EPS
    return _orig_w(self, tasks, x0)
hq.HierarchicalQP._solve_weighted = _pw

NMPC_CAP = []
_orig_solve = ns.NMPCSolver.solve
def _psolve(self, *a, **k):
    out = _orig_solve(self, *a, **k)
    try:
        e = self.step_log[-1]
        NMPC_CAP.append({'label': str(e['label']), 'status': str(e['status'])})
    except Exception:
        pass
    return out
ns.NMPCSolver.solve = _psolve

import scripts.diag_cooperative_arms as dca
from scripts.diag_cooperative_arms import _mutate_mjcf, _mjcf_md5, MJCF
from scripts.export_figure_data import load_anchors_struct_frame, phase_per_tick, MJCF as MJ2
import mujoco

STANCE_ARM = ['a', 'b', 'a', 'b', 'a', 'b']
STANCE_ANCHOR = [2, 3, 3, 4, 4, 5]
model = mujoco.MjModel.from_xml_path(MJ2)

with open(MJCF) as f:
    _original = f.read()
_pre = _mjcf_md5(MJCF)
try:
    _mutate_mjcf(damping=0.0, armature=0.05, anchor_dx=0.8, mass_ratio=0.01)
    try:
        dca.main(
            legacy=False, alpha_torso_lin=0.0, anchor_dx=0.8, mass_ratio=0.01,
            aocs_mode='legacy_pid_numerical', settle_seconds=20.0,
            K_theta=1.0, K_omega=50.0, tau_w_max=TAU_W_MAX,
            n_steps=6, ss_two_task=True, ss_alpha_mom=MOM,
            alpha_torso_pose=2000.0, ss_alpha_ee=1000.0, ss_alpha_posture=2e1,
            ss_alpha_wrench=1.0, ss_kp_torso=3.0, ss_kd_torso=2.5,
            qp_envelope_exact=True,
            interstep_settle_alpha_wrench=3.0, interstep_settle_epsilon_v=5e-3,
            out_dir_override=OUT)
    except Exception as e:
        print(f'[note] main() raised after sim: {type(e).__name__}: {str(e)[:150]}')
finally:
    with open(MJCF, 'w') as f:
        f.write(_original)
    assert _mjcf_md5(MJCF) == _pre, 'MJCF restore failed'

# ── extract U per-axis planned + realized ──
sl = json.load(open(f'results/{OUT}/sim_log.json'))
si = np.asarray(sl['step_idx'])
ph = np.array(phase_per_tick(sl))
is_ss = ph == 'SS'
lam = np.asarray(sl['lambda_qp'], float)
lref = np.asarray(sl['lambda_ref'], float)
dev = {e['step']: e for e in sl.get('dock_events', [])}
aA, aB = load_anchors_struct_frame(
    model, np.asarray(sl['struct_pos'])[0], np.asarray(sl['struct_quat'])[0])

plan_ax = np.zeros(3); real_ax = np.zeros(3)
real_norm_pk = 0.0; swing_ref_pk = 0.0; n_ss = 0
step_plan_axis = [[0.0, 0.0, 0.0] for _ in range(6)]
step_real_axis = [[0.0, 0.0, 0.0] for _ in range(6)]
for k in range(6):
    anch = (aA if STANCE_ARM[k] == 'a' else aB)[STANCE_ANCHOR[k]]
    off = 0 if STANCE_ARM[k] == 'a' else 6
    soff = 6 - off
    m = (si == k) & is_ss
    for i in np.where(m)[0]:
        if not np.all(np.isfinite(lref[i])):
            continue
        n_ss += 1
        hp = np.cross(anch, lref[i, off:off+3]) + lref[i, off+3:off+6]
        hr = np.cross(anch, lam[i, off:off+3]) + lam[i, off+3:off+6]
        plan_ax = np.maximum(plan_ax, np.abs(hp))
        real_ax = np.maximum(real_ax, np.abs(hr))
        real_norm_pk = max(real_norm_pk, float(np.linalg.norm(hr)))
        for ax in range(3):
            step_plan_axis[k][ax] = max(step_plan_axis[k][ax], abs(float(hp[ax])))
            step_real_axis[k][ax] = max(step_real_axis[k][ax], abs(float(hr[ax])))
        swing_ref_pk = max(swing_ref_pk, float(np.max(np.abs(lref[i, soff:soff+6]))))

ss_cap = [c for c in NMPC_CAP if '/SS/' in c['label']]
hist = {}
for c in ss_cap:
    hist[c['status']] = hist.get(c['status'], 0) + 1

plan_maxax = float(np.max(plan_ax))
res = dict(
    label='U (rate-off, tau_w_max=1e6)', ss_alpha_mom=MOM, tau_w_max=TAU_W_MAX,
    base=BASE, eps=EPS, n_dock=len(dev), feasible_6of6=(len(dev) == 6),
    n_ss_ticks=int(n_ss),
    plan_axis_pk=[round(float(x), 3) for x in plan_ax],
    plan_maxaxis_pk=round(plan_maxax, 3),
    plan_exceeds_5=bool(plan_maxax > 5.0),
    plan_exceedance=round(plan_maxax - 5.0, 3),
    real_axis_pk=[round(float(x), 3) for x in real_ax],
    real_maxaxis_pk=round(float(np.max(real_ax)), 3),
    real_norm_pk=round(real_norm_pk, 3),
    step_plan_axis=[[round(x, 3) for x in s] for s in step_plan_axis],
    step_real_axis=[[round(x, 3) for x in s] for s in step_real_axis],
    swing_ref_pk=round(swing_ref_pk, 4),
    ipopt_ss_status=hist, n_ss_solve=len(ss_cap),
    qp_fail=int(np.sum(~np.asarray(sl['qp_ok']))),
    nmpc_fail=int(np.sum(~np.asarray(sl['nmpc_ok']))),
)
json.dump(res, open(RESULT_JSON, 'w'), indent=2)

# ── side-by-side vs committed C mom=400 ──
C = None
if os.path.exists(C_JSON):
    cj = json.load(open(C_JSON))
    C = next((r for r in cj['runs'] if r['ss_alpha_mom'] == 400.0), None)

print('\n================ U-PLAN-CHECK ================')
print(f'base {BASE}  eps {EPS}  ss_alpha_mom {MOM}  tau_w_max {TAU_W_MAX} (rate OFF)')
print(f'dock={res["n_dock"]}/6  qp_fail={res["qp_fail"]}  swing_ref_pk={res["swing_ref_pk"]}')
print(f'\nU-plan per-axis peak [x,y,z] = {res["plan_axis_pk"]}  maxaxis={res["plan_maxaxis_pk"]}')
print(f'U-plan EXCEEDS 5.0 : {res["plan_exceeds_5"]}   (by {res["plan_exceedance"]} Nm)')
print(f'U-realized per-axis peak [x,y,z] = {res["real_axis_pk"]}  maxaxis={res["real_maxaxis_pk"]}  ||.||2={res["real_norm_pk"]}')
print('\nper-step PLANNED z-axis peak  (arm a b a b a b):')
print(f'  U : {[s[2] for s in step_plan_axis]}')
if C is not None:
    print(f'  C : {C["step_plan_maxax"]}   (C max-axis; arm-a steps = z ~5.0, capped)')
    print('\n── per arm-a step: C-plan z (capped 5.0) vs U-plan z (rate-off) ──')
    for k in (0, 2, 4):
        print(f'  step {k} (arm a): C={C["step_plan_maxax"][k]:>5}  '
              f'U-z={step_plan_axis[k][2]:>6}  '
              f'{"U EXCEEDS" if step_plan_axis[k][2] > 5.0 else "U<=5"}')
print(f'\nVerbatim IPOPT (SS, n={res["n_ss_solve"]}): {res["ipopt_ss_status"]}')
print(f'wrote {RESULT_JSON}')
