#!/usr/bin/env python3
"""Phase-1e — momentum-budget (h_max) vs locomotion-speed frontier. ZERO crawlbot/ change.

The sim couples storage budget to speed: T_step = distance / (min|h_max|/m/lever)
(sim_loop.py:1888), so larger h_max -> shorter T_step -> faster -> higher required rate
L_dot = (r_C - r) x f + tau, which the pre-planner caps at +/-tau_w_max=5
(coarse_preplanner.py:342-343). This maps the trade-off: for each h_max we recompute the
per-step T_step from the real heuristic and re-solve each of the 6 steps' pre-planner
(step boundary conditions captured from a canonical h_max=5 traversal), measuring
convergence, planned rate |L_dot| vs 5, and planned storage |h_w| vs the h_max box.

Level-1 (this script): PLANNED frontier + convergence + Task-B classification of the first
failure. Fresh opti per solve (matches the sim's per-step first-solve). Realized quantities
(closed-loop Hdot_s/h_w/docks/theta_s) come from full sims run separately for the h_max
that fully converge.

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_hmax_frontier.py
"""
import json
import os
import runpy
import sys

import numpy as np

import scripts.run_m7_single_step as r_single
from crawlbot.planning.coarse_preplanner import (
    CoarsePrePlanner, CoarsePrePlannerConfig)

SWEEP = [5.0, 6.0, 7.0, 8.0, 10.0, 12.0]
LEVER = 1.0
OUT = 'results/j2_ablation_envelope/hmax_frontier.json'

_orig_make = r_single._make_m7_config


def _mk(*a, **k):
    c = _orig_make(*a, **k)
    c.h_max_tight = np.full(3, 5.0)          # canonical, working: captures the 6-step BC
    return c


r_single._make_m7_config = _mk

# capture each step's solve() kwargs during the h_max=5 traversal
_caps = []
_cfg_holder = {}
_orig_solve = CoarsePrePlanner.solve


def _cap_solve(self, **kw):
    r = _orig_solve(self, **kw)
    _caps.append(dict(kw={k: (np.array(v, float) if k != 'T_step' else float(v))
                           for k, v in kw.items()},
                      T5=float(kw['T_step']), ok=bool(r.success)))
    _cfg_holder['cfg'] = self.config
    return r


CoarsePrePlanner.solve = _cap_solve

sys.argv = [
    'diag_cooperative_arms.py',
    '--ss-two-task', '--ss-alpha-mom', '5000', '--alpha-torso-pose', '24000',
    '--ss-kp-torso', '3', '--ss-kd-torso', '2.5',
    '--aocs_mode', 'legacy_pid_numerical', '--K_omega', '50', '--qp-envelope-exact',
    '--interstep-settle-alpha-wrench', '3', '--interstep-settle-epsilon-v', '5e-3',
    '--n-steps', '6', '--envelope-constraint', 'on', '--out-dir', '_frontier_cap',
]
print('[frontier] capturing 6-step boundary conditions from the canonical h_max=5 traversal ...')
try:
    runpy.run_module('scripts.diag_cooperative_arms', run_name='__main__')
except SystemExit:
    pass
CoarsePrePlanner.solve = _orig_solve          # stop capturing; grid uses fresh planners

cfg0 = _cfg_holder['cfg']
M = cfg0.M
m = float(cfg0.robot_mass)
TW = float(cfg0.tau_w_max)
steps = _caps[:6]
print(f'[frontier] captured {len(_caps)} solves; using first 6 as steps 0-5 '
      f'(m={m:.2f}, M={M}, tau_w_max={TW}, f_max={cfg0.f_max}, tau_max={cfg0.tau_max})')


def _default_guess(kw, T):
    x0 = np.concatenate([kw['r_com_0'].ravel(), kw['v_com_0'].ravel(), kw['L_com_0'].ravel()])
    rg = kw['r_com_goal'].ravel()
    X = np.zeros((9, M + 1))
    for k in range(M + 1):
        a = k / M
        X[0:3, k] = (1.0 - a) * x0[0:3] + a * rg
        X[3:6, k] = (rg - x0[0:3]) / T
    X[3:6, M] = 0.0
    return X, np.zeros((6, M))


def _metrics(kw, X, U, T):
    dt = T / M
    r = X[0:3, :]; v = X[3:6, :]; L = X[6:9, :]
    rC = kw['r_C_stance'].ravel(); c = kw['c_const'].ravel(); rg = kw['r_com_goal'].ravel()
    # planned rate L_dot = (rC - r_k) x f_k + tau_k  (coarse_preplanner:342, per interval)
    Ldot = np.array([np.cross(rC - r[:, k], U[0:3, k]) + U[3:6, k] for k in range(M)])
    hw = np.array([c - L[:, k] - np.cross(r[:, k], m * v[:, k]) for k in range(M + 1)])
    return dict(
        Ldot_peak_axis=[round(float(x), 3) for x in np.max(np.abs(Ldot), axis=0)],
        Ldot_peak=round(float(np.max(np.abs(Ldot))), 3),
        hw_peak_axis=[round(float(x), 3) for x in np.max(np.abs(hw), axis=0)],
        hw_peak=round(float(np.max(np.abs(hw))), 3),
        f_peak=round(float(np.max(np.abs(U[0:3, :]))), 3),
        term_r_err=round(float(np.linalg.norm(r[:, M] - rg)), 5),
        term_v=round(float(np.linalg.norm(v[:, M])), 5),
        term_L=round(float(np.linalg.norm(L[:, M])), 5))


def _solve_raw(kw, h_max, T, X_init, U_init, verbose=False):
    d = dict(cfg0.__dict__); d['ipopt_print_level'] = 5 if verbose else 0
    p = CoarsePrePlanner(CoarsePrePlannerConfig(**d)); p.build()
    o = p._opti; dt = T / M
    x0 = np.concatenate([kw['r_com_0'].ravel(), kw['v_com_0'].ravel(), kw['L_com_0'].ravel()])
    o.set_value(p._p_x0, x0); o.set_value(p._p_r_goal, kw['r_com_goal'].ravel())
    o.set_value(p._p_r_C, kw['r_C_stance'].ravel()); o.set_value(p._p_c_const, kw['c_const'].ravel())
    o.set_value(p._p_h_max, np.full(3, h_max)); o.set_value(p._p_dt, dt)
    o.set_initial(p._X, X_init); o.set_initial(p._U, U_init)
    ok = True; status = 'Solve_Succeeded'; sol = None
    try:
        sol = o.solve()
    except RuntimeError as e:
        ok = False; status = f'{type(e).__name__}: {e}'.splitlines()[0][:90]
    src = sol if sol is not None else o.debug
    try:
        X = np.asarray(src.value(p._X), float).reshape(9, M + 1)
        U = np.asarray(src.value(p._U), float).reshape(6, M)
        cost = float(src.value(o.f))
    except Exception:
        X, U = X_init.copy(), U_init.copy(); cost = float('inf')
    try:
        it = int(o.stats().get('iter_count', 0))
    except Exception:
        it = 0
    out = dict(success=ok, cost=round(cost, 3), iters=it, status=status, T_step=round(T, 4),
               X=X, U=U)
    out.update(_metrics(kw, X, U, T))
    return out


# ---- Level-1 grid: h_max x step ----
grid = {}
xstar = {}
for h in SWEEP:
    per = []
    for k, cap in enumerate(steps):
        kw = cap['kw']
        dist = float(np.linalg.norm(kw['r_com_goal'].ravel() - kw['r_com_0'].ravel()))
        T = max(0.5, dist / (h / m / LEVER))
        res = _solve_raw(kw, h, T, *_default_guess(kw, T))
        res['dist'] = round(dist, 4)
        if res['success']:
            xstar[(h, k)] = (res['X'].copy(), res['U'].copy())
        per.append({kk: vv for kk, vv in res.items() if kk not in ('X', 'U')})
    grid[h] = per

# ---- Task B: classify the first failure ----
taskB = None
fail = [(h, k) for h in SWEEP for k in range(len(steps)) if not grid[h][k]['success']]
if fail:
    h_f, k_f = min(fail, key=lambda hk: (hk[0], hk[1]))
    kw = steps[k_f]['kw']
    dist = float(np.linalg.norm(kw['r_com_goal'].ravel() - kw['r_com_0'].ravel()))
    T_f = max(0.5, dist / (h_f / m / LEVER))
    # nearest LOWER h_max that converged at this step -> its x* as warm start
    lower = [h for h in SWEEP if h < h_f and (h, k_f) in xstar]
    warm = xstar[(max(lower), k_f)] if lower else None
    res_warm = _solve_raw(kw, h_f, T_f, warm[0], warm[1]) if warm else None
    verbose = _solve_raw(kw, h_f, T_f, *_default_guess(kw, T_f), verbose=True)  # verbatim EXIT
    taskB = dict(
        h_max=h_f, step=k_f, T_step=round(T_f, 4), dist=round(dist, 4),
        warm_from_h=max(lower) if lower else None,
        warm_success=(res_warm['success'] if res_warm else None),
        warm_cost=(res_warm['cost'] if res_warm else None),
        failed_iterate=dict(
            Ldot_peak=grid[h_f][k_f]['Ldot_peak'], rate_cap=TW,
            f_peak=grid[h_f][k_f]['f_peak'], f_max=float(cfg0.f_max),
            hw_peak=grid[h_f][k_f]['hw_peak'], box=h_f,
            term_r_err=grid[h_f][k_f]['term_r_err'],
            term_v=grid[h_f][k_f]['term_v'], term_L=grid[h_f][k_f]['term_L']),
        classification=('(a) SOLVER — warm-start from nearest working h_max rescues it'
                        if (res_warm and res_warm['success'])
                        else '(b) INFEASIBLE at this speed — warm-start does not rescue; '
                             'see failed_iterate for the binding/violated constraint'))

# ---- summarize + write ----
summary = {'sweep': SWEEP, 'm': m, 'M': M, 'tau_w_max': TW,
           'f_max': float(cfg0.f_max), 'tau_max': float(cfg0.tau_max),
           'eps_v_terminal': float(cfg0.eps_v_terminal), 'eps_L_terminal': float(cfg0.eps_L_terminal),
           'per_step_dist_m': [grid[SWEEP[0]][k]['dist'] for k in range(len(steps))],
           'grid': {str(h): grid[h] for h in SWEEP}, 'taskB': taskB}
os.makedirs('results/j2_ablation_envelope', exist_ok=True)
# strip any residual arrays
json.dump(summary, open(OUT, 'w'), indent=2, default=lambda o: None)

print('\n' + '=' * 96)
print('PHASE-1E FRONTIER — planned pre-planner, per h_max (rate cap L_dot<=%.0f, box=h_max)' % TW)
print('per-step CoM distance [m]:', summary['per_step_dist_m'])
print(f'  {"h_max":>5} | {"steps ok":>8} | {"minT_step":>9} | {"planned speed":>13} | '
      f'{"Ldot pk":>8} | {"h_w pk":>7} | notes')
for h in SWEEP:
    per = grid[h]
    nok = sum(1 for r in per if r['success'])
    Ts = [r['T_step'] for r in per]
    ok_T = [r['T_step'] for r in per if r['success']]
    dtot = sum(r['dist'] for r in per)
    Ttot = sum(Ts)
    speed = dtot / Ttot if Ttot > 0 else 0.0
    ldpk = max((r['Ldot_peak'] for r in per if r['success']), default=float('nan'))
    hwpk = max((r['hw_peak'] for r in per if r['success']), default=float('nan'))
    note = 'ALL CONVERGE' if nok == len(per) else \
        'fail@steps ' + ','.join(str(k) for k, r in enumerate(per) if not r['success'])
    print(f'  {h:>5.0f} | {nok:>6}/{len(per)} | {min(Ts):>9.3f} | {speed:>11.4f} m/s | '
          f'{ldpk:>8.3f} | {hwpk:>7.3f} | {note}')

print('\nPer-step planned L_dot peak (vs cap %.0f) and h_w peak (vs box):' % TW)
for h in SWEEP:
    print(f'  h_max={h:>4.0f}: ' + '  '.join(
        f's{k}[T{r["T_step"]:.2f} ok{int(r["success"])} Ld{r["Ldot_peak"]:.2f} hw{r["hw_peak"]:.2f}]'
        for k, r in enumerate(grid[h])))

if taskB:
    print('\nTASK B — first failure: h_max=%g step=%d (T_step=%.3fs, dist=%.3fm)'
          % (taskB['h_max'], taskB['step'], taskB['T_step'], taskB['dist']))
    print(f'  warm-start from h_max={taskB["warm_from_h"]} x*: success={taskB["warm_success"]} '
          f'(cost={taskB["warm_cost"]})')
    fi = taskB['failed_iterate']
    print(f'  failed iterate: |L_dot|pk={fi["Ldot_peak"]} (cap {fi["rate_cap"]})  '
          f'|f|pk={fi["f_peak"]} (max {fi["f_max"]})  |h_w|pk={fi["hw_peak"]} (box {fi["box"]})')
    print(f'    terminal: r_err={fi["term_r_err"]}m  v={fi["term_v"]}  L={fi["term_L"]}  '
          f'(eps_v={summary["eps_v_terminal"]}, eps_L={summary["eps_L_terminal"]})')
    print('  =>', taskB['classification'])
print(f'\nwrote {OUT}')
print('=' * 96)
