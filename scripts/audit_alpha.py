#!/usr/bin/env python3
"""α (J2 #2) characterization: read ds_mobile_trace from CoM-mobile DS runs and
report — RAW, no verdict — which constraint binds (passivity / envelope /
feasibility), CoM tracking, and swing-arm manipulability before/after.

Usage:
  MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_alpha.py \
      LABEL=results/<dir> [LABEL2=results/<dir2> ...]
(LABEL is just a display tag, e.g. m0.05 or dtds2.5_m0.10)
"""
import json
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TAU_W_MAX = 5.0
PASS_BIND = 1e-4   # pass_resid > -PASS_BIND ⇒ passivity (near-)binding
ENV_BIND = 0.90 * TAU_W_MAX


def analyse(label, run_dir):
    p = os.path.join(run_dir, 'sim_log.json')
    if not os.path.exists(p):
        print(f'  !! {p} missing'); return None
    sl = json.load(open(p))
    tr = sl.get('ds_mobile_trace', [])
    print('=' * 74)
    print(f'RUN [{label}]: {run_dir}   (ds_mobile_trace ticks = {len(tr)})')
    if not tr:
        print('  (empty trace — magnitude 0 / DWELL not triggered?)')
        # still report docks/timeouts
    docks = len(sl.get('dock_events', []))
    tos = len([a for a in sl.get('aborted_steps', [])
               if a.get('reason') == 'dock_timeout'])
    res = None
    if tr:
        ce = np.array([r['com_err'] for r in tr])
        pr = np.array([r['pass_resid'] for r in tr])
        hd = np.array([r['Hdot_inf'] for r in tr])
        mn = np.array([r.get('swing_manip', float('nan')) for r in tr])
        qpf = sum(1 for r in tr if not r['qp_ok'])
        nin = sum(1 for r in tr if r['nmpc_status'] == 2)
        pass_bind_frac = float(np.mean(pr > -PASS_BIND))
        env_bind_frac = float(np.mean(hd > ENV_BIND))
        print(f'  CoM tracking err [m]:   max={ce.max():.5f}  mean={ce.mean():.5f}  final={ce[-1]:.5f}')
        print(f'  passivity resid:        max={pr.max():.3e} (→0 binds)  '
              f'binding-frac={pass_bind_frac:.2f}')
        print(f'  envelope ‖Ḣ_s‖∞ [N·m]:  max={hd.max():.3f} (cap {TAU_W_MAX})  '
              f'≥90%%cap-frac={env_bind_frac:.2f}')
        print(f'  feasibility:            qp_fail_ticks={qpf}  nmpc_infeas_ticks={nin}')
        print(f'  swing manip:            first={mn[0]:.4e}  last={mn[-1]:.4e}  '
              f'Δ={"+" if mn[-1]>=mn[0] else ""}{(mn[-1]-mn[0]):.4e}')
        # which binds (heuristic ordering for the table; reviewer decides)
        binds = []
        if qpf or nin:
            binds.append('FEASIBILITY')
        if pass_bind_frac > 0.10:
            binds.append('passivity')
        if env_bind_frac > 0.10:
            binds.append('envelope')
        print(f'  >>> binds: {", ".join(binds) if binds else "none (slack)"}')
        res = dict(label=label, ticks=len(tr), com_max=ce.max(), com_final=ce[-1],
                   pass_max=pr.max(), pass_frac=pass_bind_frac,
                   hd_max=hd.max(), env_frac=env_bind_frac,
                   qpf=qpf, nin=nin, manip0=mn[0], manip1=mn[-1],
                   docks=docks, tos=tos)
    print(f'  docks fired={docks}  dock_timeouts={tos}')
    return res


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
        print('=' * 74)
        print('SWEEP SUMMARY (raw — reviewer decides the character):')
        print(' label         com_max  com_fin  pass→0  p_frac  Hd_max  e_frac  '
              'qpf nin  manip0→manip1   dk/to')
        for r in rows:
            print(' %-12s %7.4f %8.4f %7.1e %6.2f %7.3f %6.2f %4d%4d  '
                  '%.2e→%.2e %2d/%d'
                  % (r['label'], r['com_max'], r['com_final'], r['pass_max'],
                     r['pass_frac'], r['hd_max'], r['env_frac'], r['qpf'],
                     r['nin'], r['manip0'], r['manip1'], r['docks'], r['tos']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
