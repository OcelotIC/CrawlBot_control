#!/usr/bin/env python3
"""Phase-1d — root-cause the h_max=10 pre-planner conditioning anomaly. ZERO crawlbot/ change.

Phase-1c compared canonical (h_max=5) vs loose (h_max=10) via the REAL sim and found 5
converges / 10 fails. But the sim's T_step_guess (sim_loop.py:1888-1890) is INVERSELY
coupled to h_max:  v_max = min|h_max| / m / lever ,  T_step = max(0.5, distance / v_max).
So enlarging h_max 5->10 also HALVES the planned step time — a hidden 2nd variable. This
diagnostic separates the three factors at the step-0 problem:
  - box value h_max (5 vs 10) — parametric (opti p_h_max), does not move the optimum
    (C's h_w peak 4.886 is interior to both boxes)
  - T_step (T5 = sim's h_max=5 guess ; T10 = sim's h_max=10 guess ≈ T5/2)
  - initial guess (default linear vs warm x*_5, the converged h_max=5 primal)

Method: patch _make_m7_config -> h_max=5 (working); monkeypatch CoarsePrePlanner.solve so
that on the first (step-0) call it (a) runs the real h_max=5 solve -> x*_5, then (b)
re-solves a matrix of (h_max, T_step, guess) on FRESH planners (fresh opti, no retained
duals — matches the sim's step-0, which is the first solve on its opti). All params are set
exactly as solve() does (coarse_preplanner.py:471-489); only the guess/box/T_step vary.

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_landscape_probe.py
"""
import json
import runpy
import sys

import numpy as np

import scripts.run_m7_single_step as r_single
from crawlbot.planning.coarse_preplanner import (
    CoarsePrePlanner, CoarsePrePlannerConfig)

OUT = 'results/j2_ablation_envelope/landscape_probe.json'

_orig_make = r_single._make_m7_config


def _mk(*a, **k):
    c = _orig_make(*a, **k)
    c.h_max_tight = np.full(3, 5.0)      # working canonical box
    return c


r_single._make_m7_config = _mk

_done = {}
_orig_solve = CoarsePrePlanner.solve


def _default_guess(kw, T_step, M):
    """The pre-planner's own initial guess (coarse_preplanner.py:479-489)."""
    x0 = np.concatenate([np.ravel(kw['r_com_0']), np.ravel(kw['v_com_0']),
                         np.ravel(kw['L_com_0'])])
    rg = np.ravel(kw['r_com_goal'])
    X = np.zeros((9, M + 1))
    for k in range(M + 1):
        a = k / M
        X[0:3, k] = (1.0 - a) * x0[0:3] + a * rg
        X[3:6, k] = (rg - x0[0:3]) / T_step
    X[3:6, M] = 0.0
    return X, np.zeros((6, M))


def _hw(kw, X, m):
    c = np.ravel(kw['c_const'])
    return np.array([c - X[6:9, k] - np.cross(X[0:3, k], m * X[3:6, k])
                     for k in range(X.shape[1])])


def _solve_raw(cfg, kw, h_max_val, T_step, X_init, U_init, verbose=False):
    """Fresh planner (fresh opti); set params exactly as solve() does, chosen guess/box/T."""
    d = dict(cfg.__dict__)
    d['ipopt_print_level'] = 5 if verbose else 0
    p = CoarsePrePlanner(CoarsePrePlannerConfig(**d)); p.build()
    M = p.config.M; m = float(p.config.robot_mass); dt = T_step / M
    o = p._opti
    x0 = np.concatenate([np.ravel(kw['r_com_0']), np.ravel(kw['v_com_0']),
                         np.ravel(kw['L_com_0'])])
    o.set_value(p._p_x0, x0)
    o.set_value(p._p_r_goal, np.ravel(kw['r_com_goal']))
    o.set_value(p._p_r_C, np.ravel(kw['r_C_stance']))
    o.set_value(p._p_c_const, np.ravel(kw['c_const']))
    o.set_value(p._p_h_max, np.full(3, h_max_val))
    o.set_value(p._p_dt, dt)
    o.set_initial(p._X, X_init)
    o.set_initial(p._U, U_init)
    ok = True; status = 'Solve_Succeeded'; sol = None
    try:
        sol = o.solve()
    except RuntimeError as e:
        ok = False; status = f'{type(e).__name__}: {e}'.splitlines()[0][:110]
    src = sol if sol is not None else o.debug
    try:
        X = np.asarray(src.value(p._X), float).reshape(9, M + 1)
        cost = float(src.value(o.f))
    except Exception:
        X = X_init.copy(); cost = float('inf')
    try:
        it = int(o.stats().get('iter_count', 0))
    except Exception:
        it = 0
    hw = _hw(kw, X, m)
    peak_ax = np.max(np.abs(hw), axis=0)
    return dict(success=ok, cost=cost, iters=it,
                hw_peak=float(np.max(np.abs(hw))),
                hw_peak_axis=[round(float(v), 3) for v in peak_ax],
                L_peak=float(np.max(np.linalg.norm(X[6:9, :], axis=0))),
                status=status)


def _patched_solve(self, **kw):
    if _done:
        return _orig_solve(self, **kw)
    _done['x'] = True
    cfg = self.config; M = cfg.M; m = float(cfg.robot_mass)
    r5 = _orig_solve(self, **kw)               # real working h_max=5 solve -> x*_5
    T5 = float(kw['T_step'])
    dist = float(np.linalg.norm(np.ravel(kw['r_com_goal']) - np.ravel(kw['r_com_0'])))
    v10 = 10.0 / max(m, 1e-6) / 1.0            # sim T_step_guess with h_max=10 (lever=1)
    T10 = max(0.5, dist / v10) if v10 > 0 else 1.0
    Xd5, Ud = _default_guess(kw, T5, M)
    Xd10, _ = _default_guess(kw, T10, M)
    Xstar = np.vstack([r5.r_com.T, r5.v_com.T, r5.L_com.T])
    Ustar = np.vstack([r5.f_stance.T, r5.tau_stance.T])
    hw5 = _hw(kw, Xstar, m)

    matrix = {
        '1_box5_T5_default':     _solve_raw(cfg, kw, 5.0,  T5,  Xd5, Ud),
        '2_box10_T5_default':    _solve_raw(cfg, kw, 10.0, T5,  Xd5, Ud),   # KEY: box up, T fixed
        '3_box10_T10_default':   _solve_raw(cfg, kw, 10.0, T10, Xd10, Ud),  # reproduces sim fail
        '4_box10_T5_warmxstar':  _solve_raw(cfg, kw, 10.0, T5,  Xstar, Ustar),  # TASK A
        '5_box10_T10_warmxstar': _solve_raw(cfg, kw, 10.0, T10, Xstar, Ustar),  # warm sim-problem
    }
    taskC = []
    for s in (1, 2, 3):
        rng = np.random.default_rng(s)
        Xp = Xd5 * (1.0 + rng.uniform(-0.05, 0.05, Xd5.shape))   # U_init is 0 (default); perturb state guess
        taskC.append(dict(seed=s, **_solve_raw(cfg, kw, 5.0, T5, Xp, Ud)))

    out = dict(
        T5=round(T5, 4), T10=round(T10, 4), distance_m=round(dist, 4),
        x5_solve=dict(cost=float(r5.cost), hw_peak=float(np.max(np.abs(hw5))),
                      hw_peak_axis=[round(float(v), 3) for v in np.max(np.abs(hw5), axis=0)]),
        matrix=matrix, taskC=taskC)
    import os
    os.makedirs('results/j2_ablation_envelope', exist_ok=True)
    json.dump(out, open(OUT, 'w'), indent=2)

    print('\n' + '=' * 82)
    print('PHASE-1D LANDSCAPE PROBE — step-0 pre-planner  (T5=%.3fs  T10=%.3fs  dist=%.3fm)' % (T5, T10, dist))
    print('  real x*_5: cost=%.3f  h_w peak=%.3f  axis=%s' %
          (r5.cost, out['x5_solve']['hw_peak'], out['x5_solve']['hw_peak_axis']))
    print(f'  {"experiment":<24} | {"ok":>5} | {"cost":>10} | {"h_w pk":>7} | {"iters":>5} | status')
    for name, r in matrix.items():
        st = r['status'] if len(r['status']) < 40 else r['status'][:37] + '...'
        print(f'  {name:<24} | {str(r["success"]):>5} | {r["cost"]:>10.3f} | '
              f'{r["hw_peak"]:>7.3f} | {r["iters"]:>5} | {st}')
    print('  TASK C — perturb default guess +/-5% at (box5,T5):')
    for r in taskC:
        print(f'    seed {r["seed"]}: ok={r["success"]}  cost={r["cost"]:.3f}  '
              f'h_w peak={r["hw_peak"]:.3f}  iters={r["iters"]}')
    print(f'  -> wrote {OUT}')

    # verbatim IPOPT EXIT for the two KEY box=10 cells (default-guess, both T_step)
    print('\n' + '-' * 82)
    for name, hv, T, Xi in (('2_box10_T5_default', 10.0, T5, Xd5),
                            ('3_box10_T10_default', 10.0, T10, Xd10)):
        print(f'--- verbose IPOPT (print_level=5): {name} ---')
        _solve_raw(cfg, kw, hv, T, Xi, Ud, verbose=True)
        print('-' * 82)
    print('=' * 82)
    sys.exit(0)


CoarsePrePlanner.solve = _patched_solve

sys.argv = [
    'diag_cooperative_arms.py',
    '--ss-two-task', '--ss-alpha-mom', '5000', '--alpha-torso-pose', '24000',
    '--ss-kp-torso', '3', '--ss-kd-torso', '2.5',
    '--aocs_mode', 'legacy_pid_numerical', '--K_omega', '50', '--qp-envelope-exact',
    '--interstep-settle-alpha-wrench', '3', '--interstep-settle-epsilon-v', '5e-3',
    '--n-steps', '1', '--envelope-constraint', 'on', '--out-dir', '_landscape',
]
runpy.run_module('scripts.diag_cooperative_arms', run_name='__main__')
