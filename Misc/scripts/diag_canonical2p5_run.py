#!/usr/bin/env python3
"""Phase CANONICAL-2p5 — C(2.5) + U on the frozen tau_w_max=2.5 canonical.

The FREEZE (committed, this phase) sets the live values:
  config.py      tau_w_max 2.5 | aocs_tau_w_max 2.5 | ss_alpha_ee 1e3 |
                 ss_alpha_wrench 1.0 | ss_alpha_mom 400 | alpha_torso_pose 2e3
  sim_loop.py    alpha_torque 5.0, alpha_reg 1.0 (QP construction literals)
  wholebody_qp   w_hw_slack 800
  MJCF           wheel motors ctrlrange +-2.5 (PHYSICAL plant cap, both C & U)
  (h_max +-5 UNCHANGED; eps 1e-6 = HierarchicalQP default; weight_ratio 1)

This runner VERIFIES (capture-only, no setting) the effective QP weights equal
the frozen vector, smoke-tests the MJCF plant clamp (ctrl=10 -> force 2.5),
then runs:
  U first (tau_w_max=1e6 via dca -> NMPC envelope OFF + QP box OFF + AOCS clip
           lifted; the MJCF ctrlrange is the ONLY bound = STEP 0 verification)
  C second (tau_w_max=2.5 -> management ON)  [skipped if STEP-0 gate fails]

NOTE dca.main:302-303 sets cfg.tau_w_max AND cfg.aocs_tau_w_max from ONE kwarg,
so U lifts both controller-side caps; the plant cap is the MJCF ctrlrange.
Applied wheel torque is measured from data.actuator_force[14:17] every mj_step
(the post-clamp actuator force), NOT from log.tau_w (the pre-plant command).

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/diag_canonical2p5_run.py
"""
import json
import numpy as np
import mujoco
import crawlbot.solvers.hierarchical_qp as hq
import crawlbot.solvers.wholebody_qp as wq
import crawlbot.solvers.nmpc_solver as ns

CAP = 2.5
HMAX = 5.0
EPS = 1e-6
FROZEN = dict(alpha_torso_pose=2000.0, alpha_ee=1000.0, ss_alpha_mom=400.0,
              w_hw_slack=800.0, alpha_posture=20.0, alpha_torque=5.0,
              alpha_wrench=1.0, alpha_reg=1.0)
RESULT_JSON = 'results/j2_adjconv/canonical2p5_result.json'

# ── capture-only patches ─────────────────────────────────────────────
EFF_W = []                       # effective QP weights (verification)
_orig_init = wq.WholeBodyQP.__init__
def _pinit(self, config=None):
    _orig_init(self, config)
    d = {k: float(getattr(self.config, k)) for k in FROZEN}
    d['weight_ratio'] = float(self.config.weight_ratio)
    d['tau_w_max_qp'] = float(self.config.tau_w_max)
    EFF_W.append(d)
wq.WholeBodyQP.__init__ = _pinit

KAP = []; _KCALL = [0]           # (n_eq, lmin, lmax) every 20 weighted solves
_orig_w = hq.HierarchicalQP._solve_weighted
def _pw(self, tasks, x0):
    self.regularization = EPS    # == frozen default; explicit for the record
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

NMPC_CAP = []                    # verbatim IPOPT per solve
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

FR = {'n': 0, 'clamped': 0, 'maxf': 0.0}   # wheel actuator force per mj_step
_orig_step = mujoco.mj_step
def _pstep(m, d, *a, **k):
    _orig_step(m, d, *a, **k)
    if d.actuator_force.shape[0] >= 17:
        af = float(np.max(np.abs(d.actuator_force[14:17])))
        FR['n'] += 1
        FR['maxf'] = max(FR['maxf'], af)
        if af >= CAP - 1e-6:
            FR['clamped'] += 1
mujoco.mj_step = _pstep

import scripts.diag_cooperative_arms as dca
from scripts.diag_cooperative_arms import _mutate_mjcf, _mjcf_md5, MJCF
from scripts.export_figure_data import load_anchors_struct_frame, phase_per_tick, MJCF as MJ2

STANCE_ARM = ['a', 'b', 'a', 'b', 'a', 'b']
STANCE_ANCHOR = [2, 3, 3, 4, 4, 5]
res = {'cap': CAP, 'h_max': HMAX, 'eps': EPS, 'frozen_weights': FROZEN}

# ── STEP 0a: static plant-clamp smoke test (canonical MJCF, orig step) ──
sm = mujoco.MjModel.from_xml_path(MJCF)
sd = mujoco.MjData(sm)
cr = sm.actuator_ctrlrange[14:17].tolist()
sd.ctrl[14:17] = 10.0            # demand 4x the cap
_orig_step(sm, sd)
smoke_force = [float(x) for x in sd.actuator_force[14:17]]
res['smoke'] = {'ctrlrange': cr, 'ctrl_demand': 10.0, 'applied_force': smoke_force,
                'pass': bool(max(abs(f) for f in smoke_force) <= CAP + 1e-9
                             and all(abs(lo + CAP) < 1e-9 and abs(hi - CAP) < 1e-9
                                     for lo, hi in cr))}
print(f"[smoke] ctrlrange={cr} ctrl=10 -> applied force={smoke_force} "
      f"PASS={res['smoke']['pass']}", flush=True)
json.dump(res, open(RESULT_JSON, 'w'), indent=2)
if not res['smoke']['pass']:
    print('STOP: MJCF plant clamp NOT enforced (smoke test failed).')
    raise SystemExit(1)


def _reset_caps():
    EFF_W.clear(); KAP.clear(); _KCALL[0] = 0; NMPC_CAP.clear()
    FR.update(n=0, clamped=0, maxf=0.0)


def _run(label, tau, out):
    _reset_caps()
    try:
        dca.main(
            legacy=False, alpha_torso_lin=0.0, anchor_dx=0.8, mass_ratio=0.01,
            aocs_mode='legacy_pid_numerical', settle_seconds=20.0,
            K_theta=1.0, K_omega=50.0, tau_w_max=tau,
            n_steps=6, ss_two_task=True, ss_alpha_mom=400.0,
            alpha_torso_pose=2000.0, ss_alpha_ee=1000.0, ss_alpha_posture=2e1,
            ss_alpha_wrench=1.0, ss_kp_torso=3.0, ss_kd_torso=2.5,
            qp_envelope_exact=True,
            interstep_settle_alpha_wrench=3.0, interstep_settle_epsilon_v=5e-3,
            out_dir_override=out)
    except Exception as e:
        print(f'[note] main() raised after sim ({label}): '
              f'{type(e).__name__}: {str(e)[:140]}', flush=True)
    return _extract(label, tau, out)


def _extract(label, tau, out):
    model = mujoco.MjModel.from_xml_path(MJ2)
    sl = json.load(open(f'results/{out}/sim_log.json'))
    si = np.asarray(sl['step_idx'])
    ph = np.array(phase_per_tick(sl))
    is_ss = ph == 'SS'
    lam = np.asarray(sl['lambda_qp'], float)
    lref = np.asarray(sl['lambda_ref'], float)
    eul = np.asarray(sl['struct_euler_deg'], float)
    hwp = np.asarray(sl['hw_physical'], float)
    spos = np.asarray(sl['struct_pos'], float)
    tauw = np.asarray(sl['tau_w'], float)
    ecom = np.asarray(sl.get('e_com', []), float)
    dev = {e['step']: e for e in sl.get('dock_events', [])}
    weld = [round(dev[k]['d_mm'], 3) if k in dev else None for k in range(6)]

    aA, aB = load_anchors_struct_frame(
        model, spos[0], np.asarray(sl['struct_quat'])[0])
    plan_ax = np.zeros(3); real_ax = np.zeros(3)
    n_ss = n_plan_at_cap = 0
    swing_ref_pk = 0.0
    step_plan = [0.0] * 6; step_real = [0.0] * 6
    for k in range(6):
        anch = (aA if STANCE_ARM[k] == 'a' else aB)[STANCE_ANCHOR[k]]
        off = 0 if STANCE_ARM[k] == 'a' else 6
        soff = 6 - off
        for i in np.where((si == k) & is_ss)[0]:
            if not np.all(np.isfinite(lref[i])):
                continue
            n_ss += 1
            hp = np.cross(anch, lref[i, off:off+3]) + lref[i, off+3:off+6]
            hr = np.cross(anch, lam[i, off:off+3]) + lam[i, off+3:off+6]
            plan_ax = np.maximum(plan_ax, np.abs(hp))
            real_ax = np.maximum(real_ax, np.abs(hr))
            step_plan[k] = max(step_plan[k], float(np.max(np.abs(hp))))
            step_real[k] = max(step_real[k], float(np.max(np.abs(hr))))
            if np.max(np.abs(hp)) >= CAP - 0.05:
                n_plan_at_cap += 1
            swing_ref_pk = max(swing_ref_pk,
                               float(np.max(np.abs(lref[i, soff:soff+6]))))

    th = np.linalg.norm(eul, axis=1)
    hw_ax = np.max(np.abs(hwp), axis=0)
    hw_norm = np.linalg.norm(hwp, axis=1)
    n_hw_over = int(np.sum(np.any(np.abs(hwp) > HMAX, axis=1)))
    drift_mm = float(np.max(np.linalg.norm(spos - spos[0], axis=1)) * 1000)
    cmd_max = float(np.max(np.abs(tauw))) if tauw.size else 0.0
    n_cmd_at_cap = int(np.sum(np.any(np.abs(tauw) >= CAP - 1e-3, axis=1)))
    ss_cap = [c for c in NMPC_CAP if '/SS/' in c['label']]
    hist = {}
    for c in ss_cap:
        hist[c['status']] = hist.get(c['status'], 0) + 1
    kap = lmin = None
    ka = np.array(KAP, float)
    if len(ka):
        ss = ka[ka[:, 0] == 26]
        if len(ss):
            kap = float((np.median(ss[:, 2]) + EPS) / (np.median(ss[:, 1]) + EPS))
            lmin = float(np.median(ss[:, 1]))
    wok = bool(EFF_W) and all(
        all(abs(d[k] - v) < 1e-9 for k, v in FROZEN.items()) for d in EFF_W)
    ecom_pk = float(np.max(ecom[is_ss])) if ecom.size and is_ss.any() else None

    return dict(
        label=label, tau_w_max_cmd=tau,
        weights_effective=EFF_W[0] if EFF_W else None,
        weights_match_frozen=wok,
        n_dock=len(dev), at_weld_docks=weld,
        worst=max([w for w in weld if w is not None], default=None) if len(dev) == 6 else None,
        plan_axis_pk=[round(float(x), 3) for x in plan_ax],
        real_axis_pk=[round(float(x), 3) for x in real_ax],
        step_plan_maxax=[round(x, 3) for x in step_plan],
        step_real_maxax=[round(x, 3) for x in step_real],
        n_ss_ticks=int(n_ss), n_plan_at_cap=int(n_plan_at_cap),
        swing_ref_pk=round(swing_ref_pk, 4),
        theta_pk=round(float(th.max()), 4), theta_settled=round(float(th[-1]), 4),
        hw_axis_pk=[round(float(x), 4) for x in hw_ax],
        hw_norm_pk=round(float(hw_norm.max()), 4), n_ticks_hw_over_hmax=n_hw_over,
        drift_mm=round(drift_mm, 2), e_com_pk=ecom_pk,
        tauw_cmd_max=round(cmd_max, 4), n_ticks_cmd_at_cap=n_cmd_at_cap,
        applied_force_max=round(FR['maxf'], 6),
        mjstep_total=FR['n'], mjstep_clamped=FR['clamped'],
        clamped_frac=round(FR['clamped'] / FR['n'], 4) if FR['n'] else None,
        kappa_SS=kap, lambda_min_LS=lmin,
        ipopt_ss_status=hist, n_ss_solve=len(ss_cap),
        qp_fail=int(np.sum(~np.asarray(sl['qp_ok']))),
        nmpc_fail=int(np.sum(~np.asarray(sl['nmpc_ok']))),
    )


with open(MJCF) as f:
    _original = f.read()
_pre = _mjcf_md5(MJCF)
try:
    _mutate_mjcf(damping=0.0, armature=0.05, anchor_dx=0.8, mass_ratio=0.01)

    print('\n########## U (management OFF, plant cap 2.5) ##########', flush=True)
    res['U'] = _run('U', 1e6, 'figU25_addfive')
    json.dump(res, open(RESULT_JSON, 'w'), indent=2)
    print(f"[U] dock={res['U']['n_dock']}/6 applied_force_max={res['U']['applied_force_max']} "
          f"clamped={res['U']['mjstep_clamped']}/{res['U']['mjstep_total']} "
          f"cmd_max={res['U']['tauw_cmd_max']} plan_ax={res['U']['plan_axis_pk']} "
          f"real_ax={res['U']['real_axis_pk']} theta={res['U']['theta_pk']}/{res['U']['theta_settled']} "
          f"hw_ax={res['U']['hw_axis_pk']} over_hmax={res['U']['n_ticks_hw_over_hmax']} "
          f"drift={res['U']['drift_mm']}mm", flush=True)

    if res['U']['applied_force_max'] > CAP + 1e-3:
        res['step0_gate'] = 'FAIL — applied wheel torque exceeds 2.5 in U; MJCF cap not enforced'
        json.dump(res, open(RESULT_JSON, 'w'), indent=2)
        print('STOP: STEP-0 gate FAILED — skipping C.', flush=True)
    else:
        res['step0_gate'] = 'PASS'
        print('\n########## C (management ON, cap 2.5 both) ##########', flush=True)
        res['C'] = _run('C', 2.5, 'figC25_addfive')
        json.dump(res, open(RESULT_JSON, 'w'), indent=2)
        print(f"[C] dock={res['C']['n_dock']}/6 at-weld={res['C']['at_weld_docks']} "
              f"applied_force_max={res['C']['applied_force_max']} "
              f"clamped={res['C']['mjstep_clamped']}/{res['C']['mjstep_total']} "
              f"plan_ax={res['C']['plan_axis_pk']} real_ax={res['C']['real_axis_pk']} "
              f"plan_at_cap={res['C']['n_plan_at_cap']}/{res['C']['n_ss_ticks']} "
              f"theta={res['C']['theta_pk']}/{res['C']['theta_settled']} "
              f"hw_ax={res['C']['hw_axis_pk']} drift={res['C']['drift_mm']}mm "
              f"kappa={res['C']['kappa_SS']}", flush=True)
finally:
    with open(MJCF, 'w') as f:
        f.write(_original)
    assert _mjcf_md5(MJCF) == _pre, 'MJCF restore failed'

print('\n================ CANONICAL-2p5 done ================')
print(f'wrote {RESULT_JSON}')
