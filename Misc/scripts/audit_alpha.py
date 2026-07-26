#!/usr/bin/env python3
"""α (J2 #2) characterization: read ds_mobile_trace from CoM-mobile DS runs and
report — RAW, no verdict — which constraint binds (passivity / envelope /
feasibility), CoM tracking, and swing-arm manipulability before/after.

Usage:
  MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/audit_alpha.py \
      LABEL=results/<dir> [LABEL2=results/<dir2> ...]
(LABEL is just a display tag, e.g. m0.05 or dtds2.5_m0.10)
"""
import json
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TAU_W_MAX = 5.0
PASS_BIND = 1e-4   # pass_resid > -PASS_BIND ⇒ passivity (near-)binding
ENV_BIND = 0.90 * TAU_W_MAX


def _segments(tr):
    """Split the trace into contiguous DS segments (break on t-gaps > 0.5 s)."""
    ts = np.array([r['t'] for r in tr])
    brk = [0] + [i for i in range(1, len(ts)) if ts[i] - ts[i - 1] > 0.5] + [len(ts)]
    return [(brk[k], brk[k + 1]) for k in range(len(brk) - 1)]


def analyse(label, run_dir):
    p = os.path.join(run_dir, 'sim_log.json')
    if not os.path.exists(p):
        print(f'  !! {p} missing'); return None
    sl = json.load(open(p))
    tr = sl.get('ds_mobile_trace', [])
    print('=' * 74)
    print(f'RUN [{label}]: {run_dir}   (ds_mobile_trace ticks = {len(tr)})')
    docks = len(sl.get('dock_events', []))
    tos = len([a for a in sl.get('aborted_steps', [])
               if a.get('reason') == 'dock_timeout'])
    res = None
    if tr:
        ce = np.array([r['com_err'] for r in tr])
        pr = np.array([r['pass_resid'] for r in tr])
        hd = np.array([r['Hdot_inf'] for r in tr])
        mn = np.array([r.get('swing_manip', float('nan')) for r in tr])
        # Per-segment: a "moving" segment translates the CoM (com_err range
        # > 5 mm); "hold" segments are the trailing/settle DS. Report each;
        # the MOVING segment is the moving-CoM-under-passivity conflict.
        segs = _segments(tr)
        moving = None
        print(f'  segments ({len(segs)}): '
              + ', '.join(f'[{a}:{b}] t={tr[a]["t"]:.1f}-{tr[b-1]["t"]:.1f}'
                          for a, b in segs))
        for a, b in segs:
            ce_s, pr_s, hd_s, mn_s = ce[a:b], pr[a:b], hd[a:b], mn[a:b]
            kind = 'MOVING' if (ce_s.max() - ce_s.min()) > 5e-3 else 'hold  '
            pbf = float(np.mean(pr_s > -PASS_BIND))
            ebf = float(np.mean(hd_s > ENV_BIND))
            qpf = sum(1 for r in tr[a:b] if not r['qp_ok'])
            nin = sum(1 for r in tr[a:b] if r['nmpc_status'] == 2)
            print(f'    {kind} ce[max={ce_s.max():.4f} fin={ce_s[-1]:.4f}] '
                  f'pass→0={pr_s.max():.1e} p_frac={pbf:.2f} '
                  f'Ḣ_max={hd_s.max():.3f}(cap{TAU_W_MAX:.0f}) e_frac={ebf:.2f} '
                  f'qpf={qpf} nin={nin} manip={mn_s[0]:.3e}→{mn_s[-1]:.3e}')
            if kind == 'MOVING' and moving is None:
                binds = (['FEASIBILITY'] if (qpf or nin) else []) \
                    + (['passivity'] if pbf > 0.10 else []) \
                    + (['envelope'] if ebf > 0.10 else [])
                moving = dict(label=label, com_max=ce_s.max(), com_fin=ce_s[-1],
                              pass_frac=pbf, hd_max=hd_s.max(), env_frac=ebf,
                              qpf=qpf, nin=nin, manip0=mn_s[0], manip1=mn_s[-1],
                              binds='+'.join(binds) or 'slack',
                              docks=docks, tos=tos)
        if moving is None:   # no moving segment (e.g. m=0): use the whole trace
            moving = dict(label=label, com_max=ce.max(), com_fin=ce[-1],
                          pass_frac=float(np.mean(pr > -PASS_BIND)),
                          hd_max=hd.max(), env_frac=float(np.mean(hd > ENV_BIND)),
                          qpf=0, nin=0, manip0=mn[0], manip1=mn[-1],
                          binds='hold', docks=docks, tos=tos)
        print(f'  >>> MOVING-segment binds: {moving["binds"]}')
        res = moving
    else:
        print('  (empty trace — magnitude 0 / DWELL not triggered)')
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
