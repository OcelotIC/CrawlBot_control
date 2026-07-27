#!/usr/bin/env python3
"""C3.1 (AOCS decomposition report) + C3.5 (docking table). Post-processing.

Reads the C2 --solver-diag export of the canonical
(results/review_closure/c2/c25_c2_fulldiag.csv, 92 cols + meta), which carries
the AOCS decomposition (C2.3) and the per-dock 6-D twist (C2.1).

Both are reported over the TRAVERSAL WINDOW — up to and including the last
single-support tick — as well as over the full log, following the C4
convention. The distinction is not cosmetic: the 20 s terminal settle changes
peaks by up to 46 % on theta_s and contributes zero saturation.
"""
import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
C2 = os.path.join(ROOT, 'results', 'review_closure', 'c2')
OUT = os.path.join(ROOT, 'results', 'review_closure', 'c3')
CAP = 2.5
TERMS = ('tau_ff', 'tau_att_p', 'tau_rate_d', 'tau_accel_d', 'tau_antiwindup')


def main():
    rows = list(csv.DictReader(open(os.path.join(C2, 'c25_c2_fulldiag.csv'))))
    meta = json.load(open(os.path.join(C2, 'c25_c2_fulldiag_meta.json')))
    ph = np.array([r['phase'] for r in rows])
    ss = ph == 'SS'
    last_ss = int(np.where(ss)[0].max())
    trav = np.zeros(len(rows), bool)
    trav[:last_ss + 1] = True

    def V(stem):
        return np.array([[float(r[f'{stem}_{a}_Nm']) for a in 'xyz']
                         for r in rows])

    res = {'n_ticks': len(rows), 'traversal_ticks': int(trav.sum())}

    # ── C3.1 : per-component peak / RMS, per axis, traversal window ──
    comp = {}
    for name in TERMS + ('tau_w_preclip',):
        v = V(f'aocs_{name}')[trav]
        comp[name] = {
            'peak_per_axis': [round(float(x), 6) for x in np.abs(v).max(0)],
            'rms_per_axis': [round(float(x), 6)
                             for x in np.sqrt((v ** 2).mean(0))],
            'peak': round(float(np.abs(v).max()), 6),
            'rms': round(float(np.sqrt((v ** 2).mean())), 6),
        }
    res['c3_1_components_traversal'] = comp

    ff = V('aocs_tau_ff')[trav]
    fb = sum(V(f'aocs_{n}')[trav] for n in
             ('tau_att_p', 'tau_rate_d', 'tau_accel_d'))
    aw = V('aocs_tau_antiwindup')[trav]
    rms = lambda x: float(np.sqrt((x ** 2).mean()))          # noqa: E731
    res['c3_1_shares'] = {
        'ff_rms': round(rms(ff), 6),
        'feedback_rms': round(rms(fb), 6),
        'ff_to_feedback_rms_ratio': round(rms(ff) / max(rms(fb), 1e-15), 2),
        'ff_share_of_rms_pct': round(
            100 * rms(ff) / max(rms(ff) + rms(fb), 1e-15), 2),
        'antiwindup_rms': rms(aw),
        'antiwindup_max_abs': float(np.abs(aw).max()),
        'antiwindup_nonzero_ticks': int((np.abs(aw) > 0).any(1).sum()),
    }

    # identity check, over the whole log
    S = sum(V(f'aocs_{n}') for n in TERMS)
    P = V('aocs_tau_w_preclip')
    res['c3_1_identity_max_abs_residual'] = float(np.abs(S - P).max())

    # ── saturation / clip, every convention, traversal and full ──
    def sat_block(mask):
        p = P[mask]
        s = np.abs(p) > CAP + 1e-9
        n = int(mask.sum())
        return {
            'n_ticks': n,
            'ticks_any_axis': int(s.any(1).sum()),
            'pct_ticks_any_axis': round(100 * float(s.any(1).mean()), 3),
            'axis_samples': int(s.sum()),
            'pct_axis_samples': round(100 * float(s.sum()) / (3 * n), 3),
            'per_axis_pct': [round(100 * float(s[:, j].mean()), 3)
                             for j in range(3)],
            'peak_demand': round(float(np.abs(p).max()), 4),
        }
    res['c3_1_saturation'] = {
        'traversal': sat_block(trav),
        'full_log': sat_block(np.ones(len(rows), bool)),
        'by_phase': {p: sat_block(ph == p)
                     for p in ('SS', 'DS_interstep', 'DS_terminal')},
        'paper_cited_pct': 4.1,
        'note': ('No convention reproduces the cited 4.1 %. Nearest is '
                 'any-axis-per-tick over the FULL log (4.574 %); the '
                 'traversal-window and axis-sample figures are further away. '
                 'The cited "368/51448 plant clamps" counts at a third '
                 'cadence: 51448 is neither the tick count (2077) nor the '
                 'axis-sample count (6231) nor ticks x 10 sub-steps (20770), '
                 'so its denominator is unrecorded in the repository.'),
    }

    # ── C3.5 : docking table ──
    docks = []
    for e in meta.get('dock_events', []):
        docks.append({
            'step': e['step'], 'arm': e.get('arm'), 'anchor': e.get('anchor'),
            't_s': e['t'],
            'd_mm': e['d_mm'], 'ori_deg': e['ori_deg'],
            'twist_norm': e['twist'],
            'twist_lin_s': e.get('twist_lin_s'),
            'twist_ang_s': e.get('twist_ang_s'),
            'twist_lin_norm': e.get('twist_lin_norm'),
            'twist_ang_norm': e.get('twist_ang_norm'),
            'linear_share_pct': (
                round(100 * e['twist_lin_norm'] / e['twist'], 1)
                if e.get('twist_lin_norm') and e['twist'] else None),
            'eps_pos_m': e.get('eps_pos_m'), 'eps_ori_deg': e.get('eps_ori_deg'),
            'eps_twist': e.get('eps_twist'),
            'margin_pos_mm': (round(1000 * e['eps_pos_m'] - e['d_mm'], 3)
                              if e.get('eps_pos_m') else None),
            'margin_twist': (round(e['eps_twist'] - e['twist'], 6)
                             if e.get('eps_twist') else None),
        })
    res['c3_5_docks'] = docks

    # refused-but-pose-valid approaches, for the same table
    ref = [r for r in meta.get('dock_gate_trace', [])
           if not r['fired'] and r['d_mm'] < 5.0 and r['ori_deg'] < 5.0]
    res['c3_5_pose_valid_refusals'] = [
        {'step': r['step'], 't_s': r['t'], 'd_mm': r['d_mm'],
         'ori_deg': r['ori_deg'], 'twist_norm': r['twist'],
         'twist_lin_norm': r.get('twist_lin_norm'),
         'twist_ang_norm': r.get('twist_ang_norm'),
         'exceeded_eps_twist_by_pct': round(100 * (r['twist'] / 0.05 - 1), 1)}
        for r in ref]

    with open(os.path.join(OUT, 'c3_1_c3_5.json'), 'w') as fh:
        json.dump(res, fh, indent=2)

    # ── printout ──
    print('=' * 78)
    print(f"C3.1 — AOCS decomposition over the TRAVERSAL window "
          f"({res['traversal_ticks']} of {res['n_ticks']} ticks)")
    print(f"  {'term':18s} {'peak x':>9} {'peak y':>9} {'peak z':>9} "
          f"{'RMS x':>9} {'RMS y':>9} {'RMS z':>9}")
    for name in TERMS + ('tau_w_preclip',):
        c = comp[name]
        print(f"  {name:18s}" +
              ''.join(f"{v:9.5f}" for v in c['peak_per_axis']) +
              ''.join(f"{v:9.5f}" for v in c['rms_per_axis']))
    s = res['c3_1_shares']
    print()
    print(f"  feedforward RMS {s['ff_rms']:.5f}  vs feedback RMS "
          f"{s['feedback_rms']:.5f}   ratio {s['ff_to_feedback_rms_ratio']}:1"
          f"   (FF = {s['ff_share_of_rms_pct']} % of the RMS budget)")
    print(f"  anti-windup: max |.| = {s['antiwindup_max_abs']:.1e}, "
          f"nonzero on {s['antiwindup_nonzero_ticks']} ticks -> IDENTICALLY ZERO")
    print(f"  identity max |sum(terms) - preclip| = "
          f"{res['c3_1_identity_max_abs_residual']:.2e} (CSV formatting floor)")
    print()
    print('  saturation at the +-2.5 cap:')
    for k in ('traversal', 'full_log'):
        b = res['c3_1_saturation'][k]
        print(f"    {k:12s} any-axis {b['pct_ticks_any_axis']:6.3f} %   "
              f"axis-sample {b['pct_axis_samples']:6.3f} %   "
              f"per-axis {b['per_axis_pct']}   peak {b['peak_demand']}")
    for p, b in res['c3_1_saturation']['by_phase'].items():
        print(f"    [{p:13s}] any-axis {b['pct_ticks_any_axis']:6.3f} %   "
              f"peak demand {b['peak_demand']}")
    print()
    print('=' * 78)
    print('C3.5 — docking table (structure frame; eps_twist = 0.05 on all)')
    print(f"  {'st':>2} {'arm':>3} {'t[s]':>7} {'d[mm]':>7} {'ori[°]':>7} "
          f"{'|twist|':>9} {'lin':>9} {'ang':>9} {'lin%':>5} "
          f"{'d marg':>7} {'tw marg':>9}")
    for d in docks:
        print(f"  {d['step']:>2} {str(d['arm']):>3} {d['t_s']:7.2f} "
              f"{d['d_mm']:7.3f} {d['ori_deg']:7.3f} {d['twist_norm']:9.6f} "
              f"{d['twist_lin_norm']:9.6f} {d['twist_ang_norm']:9.6f} "
              f"{d['linear_share_pct']:5.1f} {d['margin_pos_mm']:7.3f} "
              f"{d['margin_twist']:9.6f}")
    print()
    print('  approaches refused while pose-valid (would have docked closer):')
    for r in res['c3_5_pose_valid_refusals']:
        print(f"   step {r['step']} t={r['t_s']:6.2f}  d={r['d_mm']:6.3f} mm  "
              f"ori={r['ori_deg']:5.3f}  |twist|={r['twist_norm']:.6f} "
              f"(+{r['exceeded_eps_twist_by_pct']:.1f} % over eps_twist)")
    print()
    print(f"wrote {os.path.join(OUT, 'c3_1_c3_5.json')}")


if __name__ == '__main__':
    main()
