#!/usr/bin/env python3
"""Piste A (J2 #3) characterization: read ds_mobile_trace from β-sweep runs and
report — RAW — CoM tracking, which constraint binds, the envelope-coupled
W_budget, whether positive joint work occurs AND is bounded by the budget, and
feasibility. Segment-aware (the MOVING-CoM DWELL segment is the conflict).

Usage:
  MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_pisteA.py \
      LABEL=results/<dir> [LABEL2=results/<dir2> ...]
"""
import json
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TAU_W_MAX = 5.0
PASS_BIND = 1e-4    # pass_resid within PASS_BIND of its RHS ⇒ (near-)binding
BUDGET_TOL = 1e-4   # pass_resid > W_budget + tol ⇒ budget VIOLATED


def _segments(tr):
    ts = np.array([r['t'] for r in tr])
    brk = [0] + [i for i in range(1, len(ts)) if ts[i] - ts[i - 1] > 0.5] + [len(ts)]
    return [(brk[k], brk[k + 1]) for k in range(len(brk) - 1)]


def analyse(label, run_dir):
    p = os.path.join(run_dir, 'sim_log.json')
    if not os.path.exists(p):
        print(f'  !! {p} missing'); return None
    sl = json.load(open(p))
    tr = sl.get('ds_mobile_trace', [])
    print('=' * 78)
    print(f'RUN [{label}]: {run_dir}  (trace ticks={len(tr)})')
    docks = len(sl.get('dock_events', []))
    tos = len([a for a in sl.get('aborted_steps', []) if a.get('reason') == 'dock_timeout'])
    if not tr:
        print('  (empty trace)'); print(f'  docks={docks} timeouts={tos}'); return None
    ce = np.array([r['com_err'] for r in tr])
    pr = np.array([r['pass_resid'] for r in tr])
    dqt = np.array([r.get('dq_tau', float('nan')) for r in tr])
    wb = np.array([r.get('W_budget', 0.0) for r in tr])
    hd = np.array([r['Hdot_inf'] for r in tr])
    mn = np.array([r.get('swing_manip', float('nan')) for r in tr])
    segs = _segments(tr)
    moving = None
    for a, b in segs:
        rng = ce[a:b].max() - ce[a:b].min()
        if rng <= 5e-3:
            continue   # hold segment
        ce_s, pr_s, dq_s, wb_s, hd_s, mn_s = ce[a:b], pr[a:b], dqt[a:b], wb[a:b], hd[a:b], mn[a:b]
        # passivity binding = pass_resid within PASS_BIND of its RHS (W_budget)
        pbf = float(np.mean(pr_s > wb_s - PASS_BIND))
        # positive work occurs / fraction
        posf = float(np.mean(dq_s > 1e-9))
        # budget VIOLATIONS (should be 0): pass_resid > W_budget + tol
        viol = int(np.sum(pr_s > wb_s + BUDGET_TOL))
        qpf = sum(1 for r in tr[a:b] if not r['qp_ok'])
        nin = sum(1 for r in tr[a:b] if r['nmpc_status'] == 2)
        env_over = int(np.sum(hd_s > TAU_W_MAX + 1e-6))
        print(f'  MOVING seg [{a}:{b}] ({b-a} ticks):')
        print(f'    CoM err   max={ce_s.max():.4f}  final={ce_s[-1]:.4f} m')
        print(f'    W_budget  max={wb_s.max():.4f}  mean={wb_s.mean():.4f} W')
        print(f'    passivity p_frac (at RHS) = {pbf:.2f}')
        print(f'    positive work dqⱼᵀτ_q>0   frac = {posf:.2f}  (max={dq_s.max():+.4f} W)')
        print(f'    budget VIOLATIONS (pass_resid>W+tol) = {viol}  '
              f'[max(pass_resid - W) = {float((pr_s - wb_s).max()):+.2e}]')
        print(f'    envelope ‖Ḣ_s‖∞ max={hd_s.max():.3f} (cap {TAU_W_MAX})  '
              f'over-cap ticks={env_over}')
        print(f'    feasibility qpf={qpf} nin={nin} ; swing manip {mn_s[0]:.3e}→{mn_s[-1]:.3e}')
        moving = dict(label=label, com_max=ce_s.max(), com_fin=ce_s[-1],
                      wb_max=wb_s.max(), wb_mean=wb_s.mean(), pbf=pbf, posf=posf,
                      viol=viol, hd_max=hd_s.max(), env_over=env_over,
                      qpf=qpf, nin=nin, docks=docks, tos=tos)
        break
    print(f'  docks={docks} timeouts={tos}')
    return moving


def main():
    if len(sys.argv) < 2:
        print(__doc__); return 2
    rows = []
    for arg in sys.argv[1:]:
        label, _, rd = arg.partition('=')
        if not rd:
            rd, label = label, os.path.basename(label)
        r = analyse(label, rd if os.path.isabs(rd) else os.path.join(ROOT, rd))
        if r:
            rows.append(r)
    if rows:
        print('=' * 78)
        print('β-SWEEP SUMMARY (raw — MOVING segment):')
        print(' label        com_max com_fin  W_max  p_frac posW>0 viol  Hd_max env>cap  dk/to')
        for r in rows:
            print(' %-12s %7.4f %7.4f %6.3f %6.2f %6.2f %4d %7.3f %6d  %d/%d'
                  % (r['label'], r['com_max'], r['com_fin'], r['wb_max'], r['pbf'],
                     r['posf'], r['viol'], r['hd_max'], r['env_over'], r['docks'], r['tos']))
        print('\n  (viol=0 ⇒ budget never violated; env>cap=0 ⇒ envelope never exceeded)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
