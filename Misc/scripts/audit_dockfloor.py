#!/usr/bin/env python3
"""Dock-floor passivity audit: read dock runs at varying hold-window passivity
and report — RAW — the achievable dock distance, whether the arm does positive
work, and the dock twist. Decides: is the ~4.5 mm floor passivity-limited or
kinematic?

Usage:
  MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/audit_dockfloor.py \
      LABEL=results/<dir> [LABEL2=results/<dir2> ...]
"""
import json
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CLOSE_MM = 10.0   # "dock-close window" = trace ticks with d < CLOSE_MM


def analyse(label, run_dir):
    p = os.path.join(run_dir, 'sim_log.json')
    if not os.path.exists(p):
        print(f'  !! {p} missing'); return None
    sl = json.load(open(p))
    de = sl.get('dock_events', [])
    ab = [a for a in sl.get('aborted_steps', []) if a.get('reason') == 'dock_timeout']
    wt = sl.get('dock_work_trace', [])
    d_mm = [e.get('d_mm') for e in de]
    tw = [e.get('twist', float('nan')) for e in de]
    print('=' * 74)
    print(f'RUN [{label}]: {run_dir}')
    print(f'  docks fired = {len(de)}  timeouts = {len(ab)}')
    print(f'  dock distance per step [mm] = {[round(x,3) for x in d_mm]}')
    print(f'  dock twist ‖Jc·v⁻‖ per step  = {[round(x,5) for x in tw]}')
    if ab:
        print(f'  TIMEOUTS: ' + ', '.join(
            f"step{a.get('step_idx')} min_d={a.get('d_mm'):.2f}mm" for a in ab))
    worst = max(d_mm) if d_mm else float('nan')
    # positive work in the dock-close window (d < CLOSE_MM)
    pos_frac = max_dqtau = float('nan')
    pass_on = None
    if wt:
        close = [r for r in wt if r['d_mm'] < CLOSE_MM]
        if close:
            dq = np.array([r['dq_tau'] for r in close])
            pos_frac = float(np.mean(dq > 1e-9))
            max_dqtau = float(dq.max())
            pass_on = bool(np.any([r['pass_active'] for r in close]))
            print(f'  close-window (d<{CLOSE_MM:.0f}mm, {len(close)} ticks): '
                  f'dqⱼᵀτ_q max={max_dqtau:+.4f} W  positive-frac={pos_frac:.2f}  '
                  f'passivity_active={pass_on}')
    print(f'  >>> worst dock distance = {worst:.3f} mm')
    return dict(label=label, d_mm=d_mm, worst=worst, twist=tw,
                docks=len(de), tos=len(ab),
                pos_frac=pos_frac, max_dqtau=max_dqtau, pass_on=pass_on)


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
        print('DOCK-FLOOR SUMMARY (raw):')
        print(' label             worst_d  per-step d [mm]                pos_work  pass_on  dk/to')
        for r in rows:
            print(' %-16s %7.3f  %-30s %8s  %-7s %d/%d'
                  % (r['label'], r['worst'],
                     str([round(x, 2) for x in r['d_mm']]),
                     ('—' if r['pos_frac'] != r['pos_frac'] else f"{r['pos_frac']:.2f}"),
                     str(r['pass_on']), r['docks'], r['tos']))
        wmin = min(r['worst'] for r in rows if r['worst'] == r['worst'])
        print(f'\n  closest worst-dock across all levels = {wmin:.3f} mm')
    return 0


if __name__ == '__main__':
    sys.exit(main())
