#!/usr/bin/env python3
"""Phase 1h — isolate the TRUE cause of the storage-off / loose-box pre-planner failure.
Five controlled step-0 solves separating {storage box, T_step, rate cap}. ZERO crawlbot/
change (monkeypatch only). Fresh opti per solve, default guess (== the sim's guess),
verbose IPOPT (print_level=5) for a verbatim EXIT on each.

  R1  box=5    T=T5(~2.78)  rate=5      canonical control (expect SUCCESS)
  R2  box=100  T=T5(~2.78)  rate=5      loose box, T held  -> success => box value irrelevant
  R3  box=100  T=T_short     rate=5      the failing case  -> confirm FAIL
  R4  box=100  T=T_short     rate=INF    KEY: remove rate cap -> success => rate-vs-time infeasible
  R5  box=5    T=T_short     rate=5      short step WITH tight box -> fail => box not load-bearing

T5 = the canonical h_max=5 heuristic T_step (captured). T_short = the sim heuristic for a loose
box: max(0.5, dist/(min|h_max|/m/lever)) with h_max=100 (lever=1) -> hits the 0.5 s floor.
Rate cap varied via CoarsePrePlannerConfig.tau_w_max (build-time constraint, coarse_preplanner:343);
box via the solve-time p_h_max parameter; T_step via p_dt. Reports per solve: IPOPT status,
success, cost, iters, failed-iterate |h_w|/|L| peak, terminal residuals (r_err, |v_N|, |L_N|),
and planned rate peak |L_dot| = max|(r_C-r)xf + tau| (what the rate cap limits).

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_1h_factor_isolation.py
"""
import json
import runpy
import sys

import numpy as np

import scripts.run_m7_single_step as r_single
from crawlbot.planning.coarse_preplanner import (
    CoarsePrePlanner, CoarsePrePlannerConfig)

OUT = 'results/j2_ablation_envelope/factor_isolation_1h.json'
LEVER = 1.0

_orig_make = r_single._make_m7_config


def _mk(*a, **k):
    c = _orig_make(*a, **k)
    c.h_max_tight = np.full(3, 5.0)      # canonical: capture the real step-0 boundary conditions
    return c


r_single._make_m7_config = _mk

_done = {}
_orig_solve = CoarsePrePlanner.solve


def _default_guess(kw, T, M):
    x0 = np.concatenate([kw['r_com_0'].ravel(), kw['v_com_0'].ravel(), kw['L_com_0'].ravel()])
    rg = kw['r_com_goal'].ravel()
    X = np.zeros((9, M + 1))
    for k in range(M + 1):
        a = k / M
        X[0:3, k] = (1.0 - a) * x0[0:3] + a * rg
        X[3:6, k] = (rg - x0[0:3]) / T
    X[3:6, M] = 0.0
    return X, np.zeros((6, M))


def _solve(cfg, kw, box, T, rate, M, m, verbose=False):
    d = dict(cfg.__dict__)
    d['tau_w_max'] = float(rate)                 # rate cap (build-time constraint)
    d['ipopt_print_level'] = 5 if verbose else 0
    d['ipopt_max_iter'] = 1000
    p = CoarsePrePlanner(CoarsePrePlannerConfig(**d)); p.build()
    o = p._opti
    x0 = np.concatenate([kw['r_com_0'].ravel(), kw['v_com_0'].ravel(), kw['L_com_0'].ravel()])
    o.set_value(p._p_x0, x0)
    o.set_value(p._p_r_goal, kw['r_com_goal'].ravel())
    o.set_value(p._p_r_C, kw['r_C_stance'].ravel())
    o.set_value(p._p_c_const, kw['c_const'].ravel())
    o.set_value(p._p_h_max, np.full(3, box))     # storage box (solve-time param)
    o.set_value(p._p_dt, T / M)                  # T_step (solve-time param)
    Xi, Ui = _default_guess(kw, T, M)
    o.set_initial(p._X, Xi); o.set_initial(p._U, Ui)
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
        X, U = Xi.copy(), Ui.copy(); cost = float('inf')
    try:
        it = int(o.stats().get('iter_count', 0))
    except Exception:
        it = 0
    r = X[0:3]; v = X[3:6]; L = X[6:9]
    rC = kw['r_C_stance'].ravel(); c = kw['c_const'].ravel(); rg = kw['r_com_goal'].ravel()
    hw = np.array([c - L[:, k] - np.cross(r[:, k], m * v[:, k]) for k in range(M + 1)])
    Ldot = np.array([np.cross(rC - r[:, k], U[0:3, k]) + U[3:6, k] for k in range(M)])
    return dict(
        box=box, T_step=round(T, 4), rate_cap=rate, success=ok, cost=round(cost, 3), iters=it,
        status=status,
        hw_peak=round(float(np.max(np.abs(hw))), 4),
        L_peak=round(float(np.max(np.linalg.norm(L, axis=0))), 4),
        term_r_err_m=round(float(np.linalg.norm(r[:, M] - rg)), 6),
        term_v=round(float(np.linalg.norm(v[:, M])), 6),
        term_L=round(float(np.linalg.norm(L[:, M])), 6),
        Ldot_peak=round(float(np.max(np.abs(Ldot))), 4),
        f_peak=round(float(np.max(np.abs(U[0:3, :]))), 4))


def _patched(self, **kw):
    if _done:
        return _orig_solve(self, **kw)
    _done['x'] = True
    kw2 = {k: (np.array(v, float) if k != 'T_step' else float(v)) for k, v in kw.items()}
    cfg = self.config; M = cfg.M; m = float(cfg.robot_mass)
    T5 = float(kw['T_step'])
    dist = float(np.linalg.norm(kw2['r_com_goal'].ravel() - kw2['r_com_0'].ravel()))
    T_short = max(0.5, dist / (100.0 / m / LEVER))      # sim heuristic for a loose box=100
    eps_v = float(cfg.eps_v_terminal); eps_L = float(cfg.eps_L_terminal)
    print('\n' + '=' * 100)
    print(f'PHASE 1h FACTOR ISOLATION — step-0  (dist={dist:.3f}m, T5={T5:.3f}s, T_short={T_short:.3f}s, '
          f'M={M}, m={m:.2f}, eps_v={eps_v}, eps_L={eps_L}, f_max={cfg.f_max})')
    plan = [('R1', 5.0, T5, 5.0), ('R2', 100.0, T5, 5.0), ('R3', 100.0, T_short, 5.0),
            ('R4', 100.0, T_short, 1e6), ('R5', 5.0, T_short, 5.0)]
    res = {}
    for name, box, T, rate in plan:
        print('\n' + '-' * 100)
        print(f'--- {name}: box={box:g}  T_step={T:.3f}s  rate_cap={rate:g}  (verbose IPOPT) ---')
        res[name] = _solve(cfg, kw2, box, T, rate, M, m, verbose=True)
        print('-' * 100)
    import os
    os.makedirs('results/j2_ablation_envelope', exist_ok=True)
    json.dump(dict(dist=dist, T5=T5, T_short=T_short, M=M, m=m, eps_v=eps_v, eps_L=eps_L,
                   f_max=float(cfg.f_max), results=res), open(OUT, 'w'), indent=2)

    print('\n' + '=' * 100)
    print(f'  {"run":>3} | {"box":>5} {"T_step":>7} {"rate":>7} | {"OK":>5} | {"cost":>9} | '
          f'{"|h_w|pk":>8} {"|L|pk":>7} | {"r_err_m":>9} {"|v_N|":>8} {"|L_N|":>8} | {"|Ldot|pk":>9} {"|f|pk":>7}')
    for name, box, T, rate in plan:
        r = res[name]
        print(f'  {name:>3} | {box:>5g} {T:>7.3f} {rate:>7g} | {str(r["success"]):>5} | {r["cost"]:>9.3f} | '
              f'{r["hw_peak"]:>8.3f} {r["L_peak"]:>7.3f} | {r["term_r_err_m"]:>9.5f} {r["term_v"]:>8.5f} '
              f'{r["term_L"]:>8.5f} | {r["Ldot_peak"]:>9.3f} {r["f_peak"]:>7.3f}')
    print(f'\n  eps_v (terminal |v_N| soft bound) = {eps_v}   eps_L (terminal |L_N| soft bound) = {eps_L}')
    r4 = res['R4']
    if r4['success']:
        print(f'  R4 (rate cap OFF) SOLVED: terminal r_err={r4["term_r_err_m"]}m |v_N|={r4["term_v"]} '
              f'|L_N|={r4["term_L"]}; planned |Ldot|peak={r4["Ldot_peak"]} '
              f'-> {"EXCEEDS 5 (the plan the rate cap forbade)" if r4["Ldot_peak"] > 5 else "<=5"}')
    print(f'  wrote {OUT}')
    print('=' * 100)
    sys.exit(0)


CoarsePrePlanner.solve = _patched

sys.argv = [
    'diag_cooperative_arms.py',
    '--ss-two-task', '--ss-alpha-mom', '5000', '--alpha-torso-pose', '24000',
    '--ss-kp-torso', '3', '--ss-kd-torso', '2.5',
    '--aocs_mode', 'legacy_pid_numerical', '--K_omega', '50', '--qp-envelope-exact',
    '--interstep-settle-alpha-wrench', '3', '--interstep-settle-epsilon-v', '5e-3',
    '--n-steps', '1', '--envelope-constraint', 'on', '--out-dir', '_1h_factor',
]
runpy.run_module('scripts.diag_cooperative_arms', run_name='__main__')
