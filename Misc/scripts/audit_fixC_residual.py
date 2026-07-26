#!/usr/bin/env python3
"""Fix-C characterization: per-run dock residual + twist/pose split + gate firing.

Given one or more results dirs (each holding a sim_log.json from a 5-step
traversal), compute and print, RAW (no pass/fail):
  - the dock momentum residual = ‖subtree_angmom[0]‖ recomputed at each dock
    snapshot (the 0.0040 N·m·s quantity on ae0673e); final + max,
  - per-dock at-firing values: twist ‖Jc·v⁻‖, d [mm], ori [deg] (from
    dock_events) — the twist-vs-pose split inputs,
  - gate firing vs timeout per step (dock_events vs aborted_steps),
  - the near-dock ‖Jc·v⁻‖ trajectory (last K dock_gate_trace rows / step).

Usage:
  MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/audit_fixC_residual.py \
      results/<dir1> [results/<dir2> ...]
"""
import json
import os
import sys
import numpy as np
import mujoco

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MJCF = os.path.join(ROOT, 'models', 'VISPA_crawling_rwa3.xml')


def h_sys_at_snapshots(sim_log):
    """Return [(t, label, ‖subtree_angmom[0]‖)] for every snapshot."""
    m = mujoco.MjModel.from_xml_path(MJCF)
    d = mujoco.MjData(m)
    out = []
    for (t, q, v, lbl) in sim_log.get('snapshots', []):
        d.qpos[:] = np.array(q)
        d.qvel[:] = np.array(v)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)
        mujoco.mj_subtreeVel(m, d)
        out.append((float(t), lbl, float(np.linalg.norm(d.subtree_angmom[0]))))
    return out


def analyse(run_dir):
    path = os.path.join(run_dir, 'sim_log.json')
    if not os.path.exists(path):
        print(f'  !! {path} missing')
        return None
    sl = json.load(open(path))
    print('=' * 74)
    print(f'RUN: {run_dir}')

    # --- residual via H_sys at dock snapshots ---
    hs = h_sys_at_snapshots(sl)
    dock_hs = [(t, lbl, h) for (t, lbl, h) in hs if lbl.startswith('dock')]
    print('  H_sys (‖subtree_angmom[0]‖) at snapshots [N·m·s]:')
    for (t, lbl, h) in hs:
        print(f'      t={t:7.2f}  {lbl:20s}  {h:.6f}')
    final_res = dock_hs[-1][2] if dock_hs else float('nan')
    max_res = max((h for _, _, h in dock_hs), default=float('nan'))
    # Headline residual = the LAST snapshot of the traversal (the 'final'
    # post-settle H_sys; this is the 0.0040 N·m·s reference on ae0673e).
    traversal_final = hs[-1][2] if hs else float('nan')
    print(f'  --> residual: traversal-final={traversal_final:.6f}  '
          f'final-dock={final_res:.6f}  max-dock={max_res:.6f} N·m·s')

    # --- per-dock at-firing twist/pose + firing vs timeout ---
    events = sl.get('dock_events', [])
    aborts = [a for a in sl.get('aborted_steps', [])
              if a.get('reason') == 'dock_timeout']
    print(f'  docks fired = {len(events)} ; dock_timeouts = {len(aborts)}')
    print('  per-dock at firing:  step   t      d[mm]   ori[deg]  twist‖Jc·v⁻‖')
    for e in events:
        print(f'      dock  step{e.get("step")}  t={e.get("t"):6.2f}  '
              f'{e.get("d_mm"):6.2f}  {e.get("ori_deg"):7.2f}   '
              f'{e.get("twist", float("nan")):.6f}  [{e.get("method")}]')
    for a in aborts:
        print(f'      TIMEOUT step{a.get("step_idx")}  t={a.get("t"):6.2f}  '
              f'min_d={a.get("d_mm"):6.2f}mm  ori={a.get("ori_deg"):6.2f}°')

    # --- near-dock twist trajectory (last 8 gate evals per step) ---
    trace = sl.get('dock_gate_trace', [])
    if trace:
        by_step = {}
        for r in trace:
            by_step.setdefault(r.get('step'), []).append(r)
        print('  near-dock ‖Jc·v⁻‖ trajectory (last 8 gate evals / step):')
        for step in sorted(by_step):
            rows = by_step[step][-8:]
            seq = '  '.join(f'{r["twist"]:.4f}{"*" if r.get("fired") else ""}'
                            for r in rows)
            print(f'      step{step}: {seq}')
            print(f'              (d_mm: ' +
                  ' '.join(f'{r["d_mm"]:.2f}' for r in rows) + ')')
    else:
        print('  (no dock_gate_trace — legacy log)')

    return {
        'run_dir': run_dir,
        'traversal_final': traversal_final,
        'final_residual': final_res,
        'max_residual': max_res,
        'docks_fired': len(events),
        'timeouts': len(aborts),
        'per_dock': [(e.get('step'), e.get('t'), e.get('d_mm'),
                      e.get('ori_deg'), e.get('twist')) for e in events],
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    summary = []
    for rd in sys.argv[1:]:
        r = analyse(rd if os.path.isabs(rd) else os.path.join(ROOT, rd))
        if r:
            summary.append(r)
    if len(summary) > 1:
        print('=' * 74)
        print('SWEEP SUMMARY:  run                      trav_final  max_dock  '
              'docks  timeouts')
        for r in summary:
            print(f'  {os.path.basename(r["run_dir"]):26s} '
                  f'{r["traversal_final"]:.6f}  {r["max_residual"]:.6f}   '
                  f'{r["docks_fired"]}      {r["timeouts"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
