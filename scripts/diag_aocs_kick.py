#!/usr/bin/env python3
"""AOCS ω̇_s-kick measurement — READ-ONLY instrumentation of the canonical run.

Question: the QPCarry (formerly _step's locals) re-initialises the AOCS
ω_s history to zero every NMPC tick, so on QP sub-step qs=0 the numerical
derivative is ω̇_s = (ω_s − 0)/dt. How large is the resulting K_d term, and
does it drive the wheel-torque saturation?

Method (no control change):
  * wrap force_estimator.compute_aocs_command_legacy_pid_numerical — the law
    the canonical AOCS calls on BOTH paths (NMPC-tick `command` and DS-settle
    `command_interstep`); record its inputs and output;
  * wrap MujocoPlant.step to record ω_s at every control tick just before the
    step (= the ω_s that tick's law read) — the TRUE previous-tick ω_s;
  * for every call, re-evaluate the SAME pure law (a) unclipped, (b) with
    omega_s_prev := the true previous-tick ω_s ("cf"), and (c) with the WHOLE
    carry from the true previous tick — ω_s, L_com, v_com ("cf2"). The reset
    also zeroes the FD feedforward on qs=0 (L_com_prev = v_com_prev = the
    current state, so L̇_est = v̇_est = 0); (c) separates the two effects.
    All side calls; their results are never applied.
  * the replay's sim_log.json is compared bit-for-bit with the host-local
    reference (gate/_run/local_ref/ref) to prove the instrumentation is inert.

Limitation: (b) is a ONE-STEP counterfactual — the torque that tick would
have commanded with a correct history, state held fixed. It is not a
closed-loop re-run.

    MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_aocs_kick.py
        -> results/aocs_kick/        canonical (legacy restart); inert vs ref
    MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_aocs_kick.py --carry-fix
        -> results/aocs_kick_fix/    CLOSED-LOOP run with
                                     cfg.aocs_carry_across_nmpc_ticks = True
Each writes {aocs_kick_summary.json, aocs_kick_ticks.csv, aocs_kick.png} and
the run's sim_log.json under <out>/run/ (feed it to gate/dock_check.py).
"""
import csv
import json
import os
import sys

os.environ.setdefault('MUJOCO_GL', 'disabled')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

import crawlbot.solvers.hierarchical_qp as hq
_ow = hq.HierarchicalQP._solve_weighted


def _pw(self, tasks, x0):
    self.regularization = 1e-6          # == replay_canonical.py
    return _ow(self, tasks, x0)


hq.HierarchicalQP._solve_weighted = _pw

import crawlbot.aocs.force_estimator as fe
import crawlbot.control.attitude as att
import crawlbot.control.controller as ctl
import crawlbot.simulation.plant as plant_mod

CARRY_FIX = '--carry-fix' in sys.argv
OUT = 'results/aocs_kick_fix' if CARRY_FIX else 'results/aocs_kick'
RUN_DIR = OUT[len('results/'):] + '/run'   # dca writes results/<this>
CAP = None                                 # set from the first call

law = fe.compute_aocs_command_legacy_pid_numerical
rec = []                                   # one row per law call
ctx = {'path': None, 'qs': None, 'tq': None, 'phase': None,
       'omega_tick_prev': None, 'omega_tick_cur': None, 'plant': None,
       'prev_L': None, 'prev_v': None}


def law_wrapped(**kw):
    global CAP
    out = law(**kw)
    CAP = kw['tau_w_max']
    unclipped = law(**dict(kw, tau_w_max=np.inf))
    true_prev = ctx['omega_tick_prev']
    if true_prev is not None:
        cf = law(**dict(kw, omega_s_prev=true_prev))
        cf_unclipped = law(**dict(kw, omega_s_prev=true_prev,
                                  tau_w_max=np.inf))
    else:
        cf = cf_unclipped = None
    # (c) full true carry: previous law call == previous control tick
    # (the law runs once per tick on both paths when the RWA is active).
    if true_prev is not None and ctx['prev_L'] is not None:
        full = dict(kw, omega_s_prev=true_prev, L_com_prev=ctx['prev_L'],
                    v_com_prev=ctx['prev_v'])
        cf2 = law(**full)
        cf2_unclipped = law(**dict(full, tau_w_max=np.inf))
    else:
        cf2 = cf2_unclipped = None
    ctx['prev_L'] = np.asarray(kw['L_com'], float).copy()
    ctx['prev_v'] = np.asarray(kw['v_com'], float).copy()
    kd_term = kw['K_d'] * (kw['omega_s'] - kw['omega_s_prev']) / kw['dt']
    kd_true = (kw['K_d'] * (kw['omega_s'] - true_prev) / kw['dt']
               if true_prev is not None else None)
    rec.append(dict(
        path=ctx['path'], qs=ctx['qs'], tq=ctx['tq'], phase=ctx['phase'],
        prev_is_zero=bool(np.all(kw['omega_s_prev'] == 0.0)),
        omega=np.asarray(kw['omega_s'], float).copy(),
        kd=kd_term, kd_true=kd_true,
        tau=np.asarray(out, float).copy(), tau_unclipped=unclipped,
        tau_cf=cf, tau_cf_unclipped=cf_unclipped,
        tau_cf2=cf2, tau_cf2_unclipped=cf2_unclipped,
        ff_used=kw.get('tau_struct_ff') is not None))
    return out


fe.compute_aocs_command_legacy_pid_numerical = law_wrapped

# path tags
_cmd, _inter = att.AttitudeController.command, att.AttitudeController.command_interstep


def cmd_wrapped(self, **kw):
    ctx['path'], ctx['phase'] = 'nmpc', kw['phase']
    return _cmd(self, **kw)


def inter_wrapped(self, *a, **kw):
    ctx['path'], ctx['phase'], ctx['qs'], ctx['tq'] = 'ds_settle', 'DS', None, None
    return _inter(self, *a, **kw)


att.AttitudeController.command = cmd_wrapped
att.AttitudeController.command_interstep = inter_wrapped

_track = ctl.WholeBodyController.track


def track_wrapped(self, carry, qs, tq, intent, refs, plan):
    ctx['qs'], ctx['tq'] = qs, tq
    return _track(self, carry, qs, tq, intent, refs, plan)


ctl.WholeBodyController.track = track_wrapped

# true per-tick ω_s history: ω read this tick == qvel[3:6] just before step
_step = plant_mod.MujocoPlant.step


def step_wrapped(self, lock_arm_joints=False):
    ctx['omega_tick_prev'] = self.data.qvel[3:6].copy()
    return _step(self, lock_arm_joints)


plant_mod.MujocoPlant.step = step_wrapped

if CARRY_FIX:
    import crawlbot.simulation.sim_loop as sl
    _init = sl.SimulationLoop.__init__

    def _init_fix(self, *a, **k):
        _init(self, *a, **k)
        self.cfg.aocs_carry_across_nmpc_ticks = True
    sl.SimulationLoop.__init__ = _init_fix

# ── canonical replay (same kwargs as gate/replay_canonical.py) ───────────
import scripts.diag_cooperative_arms as dca
from scripts.diag_cooperative_arms import _mutate_mjcf, _mjcf_md5, MJCF

# C-run kwargs — verbatim from gate/replay_canonical.py.
C_KWARGS = dict(
    legacy=False, alpha_torso_lin=0.0, anchor_dx=0.8, mass_ratio=0.01,
    aocs_mode='legacy_pid_numerical', settle_seconds=20.0,
    K_theta=1.0, K_omega=50.0, tau_w_max=2.5,
    n_steps=6, ss_two_task=True, ss_alpha_mom=400.0,
    alpha_torso_pose=2000.0, ss_alpha_ee=1000.0, ss_alpha_posture=2e1,
    ss_alpha_wrench=1.0, ss_kp_torso=3.0, ss_kd_torso=2.5,
    qp_envelope_exact=True,
    interstep_settle_alpha_wrench=3.0, interstep_settle_epsilon_v=5e-3,
    out_dir_override=RUN_DIR,
)

with open(MJCF) as f:
    _orig = f.read()
_pre = _mjcf_md5(MJCF)
try:
    _mutate_mjcf(damping=0.0, armature=0.05, anchor_dx=0.8, mass_ratio=0.01)
    dca.main(**C_KWARGS)
finally:
    with open(MJCF, 'w') as f:
        f.write(_orig)
    assert _mjcf_md5(MJCF) == _pre, 'MJCF restore failed'

# ── 1. inertness: instrumented sim_log vs host-local reference ───────────
sys.path.insert(0, ROOT)
from gate.local_ref import _walk  # noqa: E402
st = {'floats': 0, 'other': 0, 'fields': 0, 'excluded': set()}
ref_log = json.load(open('gate/_run/local_ref/ref/sim_log.json'))
new_log = json.load(open(f'results/{RUN_DIR}/sim_log.json'))
inert = _walk(ref_log, new_log, 'sim_log.json', st)
print(f'[{"fix: differs from ref as expected" if CARRY_FIX else "inert"}] '
      f'sim_log vs ref: {"IDENTICAL" if inert is None else inert} '
      f'({st["floats"]} floats)')

# ── 2. analysis ──────────────────────────────────────────────────────────
os.makedirs(OUT, exist_ok=True)
EPS = 1e-9


def sat(v):
    return bool(np.any(np.abs(v) >= CAP - EPS))


def group(rows):
    if not rows:
        return None
    kd = np.array([np.max(np.abs(r['kd'])) for r in rows])
    kd_t = np.array([np.max(np.abs(r['kd_true'])) for r in rows
                     if r['kd_true'] is not None])
    dtau = np.array([np.max(np.abs(r['tau'] - r['tau_cf'])) for r in rows
                     if r['tau_cf'] is not None])
    dtau2 = np.array([np.max(np.abs(r['tau'] - r['tau_cf2'])) for r in rows
                      if r['tau_cf2'] is not None])
    n_sat = sum(sat(r['tau']) for r in rows)
    n_sat_cf2 = sum(sat(r['tau_cf2']) for r in rows if r['tau_cf2'] is not None)
    n_sat_cf = sum(sat(r['tau_cf']) for r in rows if r['tau_cf'] is not None)
    return {
        'n': len(rows),
        'kd_inf_max': float(kd.max()), 'kd_inf_p50': float(np.median(kd)),
        'kd_inf_p95': float(np.percentile(kd, 95)),
        'kd_true_inf_max': float(kd_t.max()) if kd_t.size else None,
        'kd_true_inf_p50': float(np.median(kd_t)) if kd_t.size else None,
        'dtau_vs_true_history_inf_max': float(dtau.max()) if dtau.size else None,
        'dtau_vs_true_history_inf_p50': float(np.median(dtau)) if dtau.size else None,
        'dtau_vs_full_true_carry_inf_max': float(dtau2.max()) if dtau2.size else None,
        'dtau_vs_full_true_carry_inf_p50': float(np.median(dtau2)) if dtau2.size else None,
        'dtau_vs_full_true_carry_inf_p95': float(np.percentile(dtau2, 95)) if dtau2.size else None,
        'tau_inf_p50': float(np.median([np.max(np.abs(r['tau'])) for r in rows])),
        'tau_full_true_carry_inf_p50': (float(np.median([np.max(np.abs(r['tau_cf2'])) for r in rows if r['tau_cf2'] is not None])) if dtau2.size else None),
        'n_saturated': int(n_sat), 'n_saturated_true_history': int(n_sat_cf),
        'n_saturated_full_true_carry': int(n_sat_cf2),
        'n_wrench_ff': int(sum(r['ff_used'] for r in rows)),
        'omega_inf_max_mrad_s': float(max(np.max(np.abs(r['omega'])) for r in rows) * 1e3),
    }


nm = [r for r in rec if r['path'] == 'nmpc']
groups = {
    'nmpc_qs0': [r for r in nm if r['qs'] == 0],
    'nmpc_qs1_9': [r for r in nm if r['qs'] != 0],
    'ds_settle': [r for r in rec if r['path'] == 'ds_settle'],
}
summary = {k: group(v) for k, v in groups.items()}
summary['check_qs0_prev_is_zero'] = all(r['prev_is_zero'] for r in groups['nmpc_qs0'])
summary['check_qs_ge1_prev_nonzero'] = sum(r['prev_is_zero'] for r in groups['nmpc_qs1_9'])
for ph in ('SS', 'DS'):
    summary[f'nmpc_qs0_{ph}'] = group([r for r in groups['nmpc_qs0'] if r['phase'] == ph])
tot_sat = sum(sat(r['tau']) for r in rec)
summary['all_calls'] = {'n': len(rec), 'n_saturated': int(tot_sat)}
summary['tau_w_cap'] = CAP
summary['inert_vs_ref'] = inert is None
summary['aocs_carry_across_nmpc_ticks'] = CARRY_FIX
summary['note'] = ('*_true_history: one-step counterfactual, omega_s_prev := '
                   'true previous-tick omega_s only. *_full_true_carry: also '
                   'L_com_prev / v_com_prev := true previous tick. Same state '
                   'either way; not a closed-loop re-run.')
json.dump(summary, open(f'{OUT}/aocs_kick_summary.json', 'w'), indent=1)

with open(f'{OUT}/aocs_kick_ticks.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['path', 'phase', 'qs', 'tq', 'kd_inf', 'kd_true_inf',
                'tau_inf', 'tau_cf_inf', 'tau_unclipped_inf',
                'tau_cf_unclipped_inf', 'omega_inf'])
    for r in rec:
        w.writerow([r['path'], r['phase'], r['qs'], r['tq'],
                    float(np.max(np.abs(r['kd']))),
                    '' if r['kd_true'] is None else float(np.max(np.abs(r['kd_true']))),
                    float(np.max(np.abs(r['tau']))),
                    '' if r['tau_cf'] is None else float(np.max(np.abs(r['tau_cf']))),
                    float(np.max(np.abs(r['tau_unclipped']))),
                    '' if r['tau_cf_unclipped'] is None else float(np.max(np.abs(r['tau_cf_unclipped']))),
                    float(np.max(np.abs(r['omega'])))])

# figure: |K_d term| and |tau_w| over time, NMPC path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
for qs0, c, lab in ((False, '0.6', 'qs 1-9'), (True, 'C3', 'qs 0 (history = 0)')):
    rr = [r for r in nm if (r['qs'] == 0) == qs0]
    tq = [r['tq'] for r in rr]
    ax[0].plot(tq, [np.max(np.abs(r['kd'])) for r in rr], '.', ms=2, c=c, label=lab)
    ax[1].plot(tq, [np.max(np.abs(r['tau'])) for r in rr], '.', ms=2, c=c, label=lab)
rr = [r for r in groups['nmpc_qs0'] if r['kd_true'] is not None]
ax[0].plot([r['tq'] for r in rr], [np.max(np.abs(r['kd_true'])) for r in rr],
           '.', ms=2, c='C0', label='qs 0, true history (counterfactual)')
rr2 = [r for r in groups['nmpc_qs0'] if r['tau_cf2'] is not None]
ax[1].plot([r['tq'] for r in rr2], [np.max(np.abs(r['tau_cf2'])) for r in rr2],
           '.', ms=2, c='C0', label='qs 0, full true carry (ω, L, v) — counterfactual')
ax[1].axhline(CAP, ls='--', c='k', lw=0.8)
ax[0].set_yscale('log'); ax[0].set_ylabel('|K_d·Δω_s/dt|∞ [N·m]')
ax[1].set_ylabel('|τ_w|∞ [N·m]'); ax[1].set_xlabel('t [s]')
ax[0].legend(fontsize=7, markerscale=4); ax[1].legend(fontsize=7, markerscale=4)
ax[0].set_title('AOCS K_d term — NMPC-tracked ticks (C run'
                + (', aocs_carry_across_nmpc_ticks=True)' if CARRY_FIX else ', canonical)'))
fig.tight_layout(); fig.savefig(f'{OUT}/aocs_kick.png', dpi=130)
print(json.dumps(summary, indent=1))
