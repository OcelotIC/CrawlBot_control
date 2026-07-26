#!/usr/bin/env python3
"""Phase NMPC-PLAN-SATURATION — is the planned Ḣ_s envelope-saturation
independent of the WQP momentum-task weight? (falsifiable, measurement).

HYPOTHESIS: the momentum management (|Ḣ_s,i| ≤ τ_w_max=5, PER-AXIS ‖·‖_∞) is
enforced at the NMPC (IPOPT) level, so the NMPC's PLANNED Ḣ_s(lambda_ref)
saturates the envelope (some axis ~5.0) regardless of the WQP momentum-task
weight ss_alpha_mom. ss_alpha_mom (WQP-only, absent from NMPC/preplanner) only
governs whether the REALIZED Ḣ_s follows the plan.

Ground truth in code:
  - NMPC constraint  centroidal_nmpc.py:279-282  H_dot_s=Σ(r_Cj×fj+τj), |·,i|≤tw
  - WQP box (same q)  wholebody_qp.py:564-580      |M_exact·λ| ≤ τ_w_max (per-axis)
  - planned = NMPC-EXACT Ḣ_s(lambda_ref)           sim_loop.py:3049
  - verbatim IPOPT    info.status=stats['return_status'] nmpc_solver.py:480

SWEEP ss_alpha_mom ∈ {400,1000,2000,5000}. Hold fixed: torso 2000, EE 1000,
hw-slack 800, posture 20, torque 5 (=5× accel-reg floor 1, feasibility gate),
wrench 1, accel-reg 1, ε 1e-6, τ_w_max 5. Base = Add-5 docking recipe.

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/diag_nmpc_plan_saturation.py
"""
import os
import json
import numpy as np
import crawlbot.solvers.hierarchical_qp as hq
import crawlbot.solvers.wholebody_qp as wq
import crawlbot.solvers.nmpc_solver as ns

EPS = 1e-6
TAU_W_MAX = 5.0
SAT = 4.95  # per-axis saturation threshold (≥ 4.95 ⇒ hitting the ±5 cap)
MOMS = [400.0, 1000.0, 2000.0, 5000.0]
BASE = dict(alpha_torso_pose=2000.0, w_hw_slack=800.0, alpha_ee=1000.0,
            alpha_posture=20.0, alpha_wrench=1.0, alpha_torque=5.0, alpha_reg=1.0)
RESULT_JSON = 'results/j2_adjconv/nmpc_plan_sat_result.json'

_CUR_MOM = [MOMS[0]]

# ── WQP weight patch (full fixed vector + swept ss_alpha_mom) ──
_orig_init = wq.WholeBodyQP.__init__
def _pinit(self, config=None):
    _orig_init(self, config)
    for k, v in BASE.items():
        setattr(self.config, k, v)
    setattr(self.config, 'ss_alpha_mom', _CUR_MOM[0])
wq.WholeBodyQP.__init__ = _pinit

# ── hierarchical regularization ε ──
_orig_w = hq.HierarchicalQP._solve_weighted
def _pw(self, tasks, x0):
    self.regularization = EPS
    return _orig_w(self, tasks, x0)
hq.HierarchicalQP._solve_weighted = _pw

# ── verbatim IPOPT capture (per NMPC solve, tagged by step/phase/t label) ──
NMPC_CAP = []
_orig_solve = ns.NMPCSolver.solve
def _psolve(self, *a, **k):
    out = _orig_solve(self, *a, **k)
    try:
        e = self.step_log[-1]
        NMPC_CAP.append({'label': str(e['label']), 'status': str(e['status']),
                         'iter': int(e['iter']), 'inf_pr': float(e['inf_pr'])})
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


def _run_one(mom):
    _CUR_MOM[0] = mom
    NMPC_CAP.clear()
    out = f'figN_mom{int(mom)}'
    try:
        dca.main(
            legacy=False, alpha_torso_lin=0.0, anchor_dx=0.8, mass_ratio=0.01,
            aocs_mode='legacy_pid_numerical', settle_seconds=20.0,
            K_theta=1.0, K_omega=50.0, tau_w_max=TAU_W_MAX,
            n_steps=6, ss_two_task=True, ss_alpha_mom=mom,
            alpha_torso_pose=2000.0, ss_alpha_ee=1000.0, ss_alpha_posture=2e1,
            ss_alpha_wrench=1.0, ss_kp_torso=3.0, ss_kd_torso=2.5,
            qp_envelope_exact=True,
            interstep_settle_alpha_wrench=3.0, interstep_settle_epsilon_v=5e-3,
            out_dir_override=out)
    except Exception as e:
        print(f'[note] main() raised after sim (mom={mom}): '
              f'{type(e).__name__}: {str(e)[:140]}')
    cap = [dict(c) for c in NMPC_CAP]  # snapshot before next run clears it
    return out, cap


def _extract(out, cap):
    sl = json.load(open(f'results/{out}/sim_log.json'))
    si = np.asarray(sl['step_idx'])
    ph = np.array(phase_per_tick(sl))
    is_ss = ph == 'SS'
    lam = np.asarray(sl['lambda_qp'], float)
    lref = np.asarray(sl['lambda_ref'], float)
    ecom = np.asarray(sl.get('e_com', []), float)
    dev = {e['step']: e for e in sl.get('dock_events', [])}
    n_dock = len(dev)

    aA, aB = load_anchors_struct_frame(
        model, np.asarray(sl['struct_pos'])[0], np.asarray(sl['struct_quat'])[0])

    plan_ax = np.zeros(3)   # per-axis planned peak |Ḣ_s,ax|
    real_ax = np.zeros(3)   # per-axis realized peak
    n_ss = n_plan_sat = n_real_sat = 0
    swing_ref_pk = 0.0      # sanity: max |swing-arm lambda_ref| (should be ~0)
    step_plan = [0.0] * 6   # per-step planned max-axis (cross-check vs copri_tq5)
    step_real = [0.0] * 6

    for k in range(6):
        anch = (aA if STANCE_ARM[k] == 'a' else aB)[STANCE_ANCHOR[k]]
        off = 0 if STANCE_ARM[k] == 'a' else 6
        soff = 6 - off      # swing block offset
        m = (si == k) & is_ss
        for i in np.where(m)[0]:
            if not np.all(np.isfinite(lref[i])):
                continue
            n_ss += 1
            hp = np.cross(anch, lref[i, off:off+3]) + lref[i, off+3:off+6]
            hr = np.cross(anch, lam[i, off:off+3]) + lam[i, off+3:off+6]
            plan_ax = np.maximum(plan_ax, np.abs(hp))
            real_ax = np.maximum(real_ax, np.abs(hr))
            step_plan[k] = max(step_plan[k], float(np.max(np.abs(hp))))
            step_real[k] = max(step_real[k], float(np.max(np.abs(hr))))
            if np.max(np.abs(hp)) >= SAT:
                n_plan_sat += 1
            if np.max(np.abs(hr)) >= SAT:
                n_real_sat += 1
            swing_ref_pk = max(swing_ref_pk, float(np.max(np.abs(lref[i, soff:soff+6]))))

    # verbatim IPOPT, SS solves only (label carries "/SS/")
    ss_cap = [c for c in cap if '/SS/' in c['label']]
    status_hist = {}
    for c in ss_cap:
        status_hist[c['status']] = status_hist.get(c['status'], 0) + 1
    n_ss_solve = len(ss_cap)
    n_ok = sum(v for kk, v in status_hist.items() if kk == 'Solve_Succeeded')

    e_com_pk = float(np.max(ecom[is_ss])) if ecom.size and is_ss.any() else float('nan')

    plan_maxax = float(np.max(plan_ax))
    real_maxax = float(np.max(real_ax))
    return dict(
        ss_alpha_mom=_CUR_MOM[0], n_dock=n_dock, feasible_6of6=(n_dock == 6),
        n_ss_ticks=int(n_ss),
        plan_axis_pk=[round(float(x), 3) for x in plan_ax],
        plan_maxaxis_pk=round(plan_maxax, 3),
        plan_saturates=bool(plan_maxax >= SAT),
        n_plan_sat=int(n_plan_sat),
        frac_plan_sat=round(n_plan_sat / n_ss, 3) if n_ss else None,
        real_axis_pk=[round(float(x), 3) for x in real_ax],
        real_maxaxis_pk=round(real_maxax, 3),
        real_saturates=bool(real_maxax >= SAT),
        n_real_sat=int(n_real_sat),
        frac_real_sat=round(n_real_sat / n_ss, 3) if n_ss else None,
        real_over_plan=round(real_maxax / plan_maxax, 3) if plan_maxax else None,
        step_plan_maxax=[round(x, 3) for x in step_plan],
        step_real_maxax=[round(x, 3) for x in step_real],
        swing_ref_pk=round(swing_ref_pk, 4),
        e_com_pk=round(e_com_pk, 5),
        ipopt_ss_status=status_hist, n_ss_solve=n_ss_solve,
        ipopt_ss_ok=n_ok,
        qp_fail=int(np.sum(~np.asarray(sl['qp_ok']))),
        nmpc_fail=int(np.sum(~np.asarray(sl['nmpc_ok']))),
    )


# ── run the sweep (single MJCF mutation, restored once) ──
with open(MJCF) as f:
    _original = f.read()
_pre = _mjcf_md5(MJCF)
results = []
try:
    _mutate_mjcf(damping=0.0, armature=0.05, anchor_dx=0.8, mass_ratio=0.01)
    for mom in MOMS:
        print(f'\n########## ss_alpha_mom = {mom} ##########', flush=True)
        out, cap = _run_one(mom)
        res = _extract(out, cap)
        results.append(res)
        json.dump(dict(base=BASE, eps=EPS, tau_w_max=TAU_W_MAX, sat_thresh=SAT,
                       moms=MOMS, runs=results),
                  open(RESULT_JSON, 'w'), indent=2)
        print(f'[mom {mom}] dock={res["n_dock"]}/6 '
              f'PLAN axis={res["plan_axis_pk"]} maxax={res["plan_maxaxis_pk"]} '
              f'sat={res["plan_saturates"]} ({res["n_plan_sat"]}/{res["n_ss_ticks"]}) | '
              f'REAL axis={res["real_axis_pk"]} maxax={res["real_maxaxis_pk"]} '
              f'sat={res["real_saturates"]} ({res["n_real_sat"]}/{res["n_ss_ticks"]}) '
              f'r/p={res["real_over_plan"]} | swing_ref_pk={res["swing_ref_pk"]} '
              f'e_com={res["e_com_pk"]} | IPOPT-SS {res["ipopt_ss_status"]}',
              flush=True)
finally:
    with open(MJCF, 'w') as f:
        f.write(_original)
    assert _mjcf_md5(MJCF) == _pre, 'MJCF restore failed'

print('\n================ NMPC-PLAN-SATURATION ================')
print(f'fixed base: {BASE}  eps {EPS}  tau_w_max {TAU_W_MAX}')
hdr = ('  mom  dock            PLAN x/y/z   pMax  pSat            '
       'REAL x/y/z   rMax  rSat    r/p  e_com')
print(hdr)
for r in results:
    print(f'{r["ss_alpha_mom"]:>5g} {r["n_dock"]}/6  '
          f'{str(r["plan_axis_pk"]):>22} {r["plan_maxaxis_pk"]:>6} '
          f'{str(r["plan_saturates"]):>5}  '
          f'{str(r["real_axis_pk"]):>22} {r["real_maxaxis_pk"]:>6} '
          f'{str(r["real_saturates"]):>5} {str(r["real_over_plan"]):>6} '
          f'{r["e_com_pk"]:>6}')
print(f'\nwrote {RESULT_JSON}')
