#!/usr/bin/env python3
"""C1 evidence from the committed canonical artifacts (read-only).

Covers:
  C1.4  anti-windup term  K_hw*(sat_{+-5}(h_w) - h_w)  identically zero?
  C1.6  managed vs unmanaged traversal duration, and WHERE the extra time is
        spent: in the planned swing T_step (= the constrained plan) or after
        the swing finished, waiting for the dock gate (= phase-advance
        criteria).

Mechanism decomposition (from sim_loop.py:1783-1791):
    the dock gate is evaluated ONLY when
        (t - t_ss_start) > dock_check_delay (0.5 s)   AND
        (t - t_ss_start) >= swing_early_finish_fraction * T_step   (= T_step)
    so the FIRST tick with gate_eval == 1 marks t_ss_start + T_step, and
        T_step_inferred      = t(first gate_eval)  - t(SS start)
        post_swing_hold      = t(dock)             - t(first gate_eval)
    The inference is validated against the C run's logged
    `preplanner_T_steps` (results/gate_run_scratch/sim_log.json), where both
    quantities are available.

Reads   results/j2_adjconv/{c25,u25}_fulldiag.csv (+ _meta.json)
        results/gate_run_scratch/sim_log.json     (C only, T_step validation)
Writes  results/review_closure/c1/c1_log_analysis.json
"""
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
ADJ = os.path.join(ROOT, 'results', 'j2_adjconv')
OUT = os.path.join(ROOT, 'results', 'review_closure', 'c1')
SCRATCH = os.path.join(ROOT, 'results', 'gate_run_scratch', 'sim_log.json')


def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def fnum(row, key):
    v = row[key]
    return float(v) if v not in ('', 'None', 'nan') else float('nan')


def analyse(tag, rows, meta):
    HW_MAX = 5.0
    hw_axis_peak = 0.0
    windup_ticks = 0
    for r in rows:
        for ax in 'xyz':
            h = abs(fnum(r, f'hw_{ax}_Nms'))
            hw_axis_peak = max(hw_axis_peak, h)
            if h > HW_MAX:
                windup_ticks += 1

    # phase occupancy (phases are SS / DS_interstep / DS_terminal)
    occ = {}
    for r in rows:
        occ[r['phase']] = occ.get(r['phase'], 0) + 1

    # per-step SS window + gate timeline
    steps = {}
    for r in rows:
        if r['phase'] != 'SS':
            continue
        s = int(r['step_index'])
        t = fnum(r, 't_s')
        d = steps.setdefault(s, {'ss_t0': t, 'ss_t1': t,
                                 'first_gate_eval': None,
                                 'dock_t': None, 'n_gate_evals': 0,
                                 'd_min_mm': float('inf'),
                                 't_at_d_min': None,
                                 'gate_fail_pos': 0, 'gate_fail_ori': 0,
                                 'gate_fail_twist': 0})
        d['ss_t0'] = min(d['ss_t0'], t)
        d['ss_t1'] = max(d['ss_t1'], t)
        dg = fnum(r, 'd_grip_swing_mm')
        if dg == dg and dg < d['d_min_mm']:
            d['d_min_mm'] = dg
            d['t_at_d_min'] = t
        if r['gate_eval'] == '1':
            d['n_gate_evals'] += 1
            if d['first_gate_eval'] is None:
                d['first_gate_eval'] = t
            if r['docked_at_gateeval'] == '1' and d['dock_t'] is None:
                d['dock_t'] = t
            if r['docked_at_gateeval'] == '0':
                if r['pos_ok'] != '1':
                    d['gate_fail_pos'] += 1
                elif r['ori_ok'] != '1':
                    d['gate_fail_ori'] += 1
                else:
                    d['gate_fail_twist'] += 1

    for s, d in steps.items():
        d['ss_span_s'] = d['ss_t1'] - d['ss_t0']
        if d['first_gate_eval'] is not None:
            d['T_step_inferred_s'] = d['first_gate_eval'] - d['ss_t0']
            if d['dock_t'] is not None:
                d['post_swing_hold_s'] = d['dock_t'] - d['first_gate_eval']

    t0 = fnum(rows[0], 't_s')
    t1 = fnum(rows[-1], 't_s')
    docks = [{'step': e['step'], 'weld_t': e['weld_t'],
              'd_weld_mm': e['d_weld_mm'],
              'd_min_swing_mm': e['d_min_swing_mm'],
              'twist_weld': e['twist_weld']}
             for e in meta.get('at_weld_vs_min', [])]

    return {
        'tag': tag, 'n_ticks': len(rows),
        't_first': t0, 't_last': t1, 'traversal_span_s': t1 - t0,
        'phase_occupancy_ticks': occ,
        'hw_axis_peak_Nms': hw_axis_peak,
        'antiwindup_active_ticks': windup_ticks,
        'last_dock_t': docks[-1]['weld_t'] if docks else None,
        'docks': docks,
        'per_step': {str(k): v for k, v in sorted(steps.items())},
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for tag, stem in (('C_managed', 'c25'), ('U_unmanaged', 'u25')):
        rows = load(os.path.join(ADJ, f'{stem}_fulldiag.csv'))
        with open(os.path.join(ADJ, f'{stem}_fulldiag_meta.json')) as fh:
            meta = json.load(fh)
        res[tag] = analyse(tag, rows, meta)

    # validate the T_step inference against the C run's logged plan
    if os.path.exists(SCRATCH):
        with open(SCRATCH) as fh:
            log = json.load(fh)
        res['C_managed']['preplanner_T_steps_logged'] = log.get(
            'preplanner_T_steps')

    c, u = res['C_managed'], res['U_unmanaged']
    print('=' * 78)
    print('C1.4 — anti-windup  K_hw * (sat_5(h_w) - h_w)')
    for tag in ('C_managed', 'U_unmanaged'):
        r = res[tag]
        print(f"  {tag:14s}  |h_w|_inf peak = {r['hw_axis_peak_Nms']:.4f} Nms"
              f"   ticks with |h_w| > 5.0 : {r['antiwindup_active_ticks']}"
              f" / {r['n_ticks']}")
    print()
    print('=' * 78)
    print('C1.6 — traversal duration and where the difference is spent')
    print(f"  C managed   : span {c['traversal_span_s']:7.2f} s"
          f"   last dock {c['last_dock_t']:7.2f} s"
          f"   ticks {c['phase_occupancy_ticks']}")
    print(f"  U unmanaged : span {u['traversal_span_s']:7.2f} s"
          f"   last dock {u['last_dock_t']:7.2f} s"
          f"   ticks {u['phase_occupancy_ticks']}")
    print(f"  DELTA       : span {c['traversal_span_s']-u['traversal_span_s']:+7.2f} s"
          f"   last dock {c['last_dock_t']-u['last_dock_t']:+7.2f} s")
    print()
    tsl = c.get('preplanner_T_steps_logged')
    if tsl:
        print(f"  C logged preplanner_T_steps: "
              f"{[round(float(x), 2) for x in tsl]}")
    print()
    hdr = (f"  {'step':>4} | {'T_step (planned swing)':>26} | "
           f"{'post-swing hold':>24} | {'SS total':>18}")
    print(hdr)
    print(f"  {'':>4} | {'C':>8} {'U':>8} {'dT':>8} | "
          f"{'C':>7} {'U':>7} {'dHold':>8} | {'C':>5} {'U':>5} {'dSS':>6}")
    print('  ' + '-' * (len(hdr) - 2))
    tot = {'Tc': 0.0, 'Tu': 0.0, 'Hc': 0.0, 'Hu': 0.0, 'Sc': 0.0, 'Su': 0.0}
    for k in sorted(c['per_step'], key=int):
        if int(k) < 0:
            continue
        cs, us = c['per_step'][k], u['per_step'].get(k, {})
        Tc = cs.get('T_step_inferred_s', float('nan'))
        Tu = us.get('T_step_inferred_s', float('nan'))
        Hc = cs.get('post_swing_hold_s', float('nan'))
        Hu = us.get('post_swing_hold_s', float('nan'))
        Sc, Su = cs.get('ss_span_s', 0.0), us.get('ss_span_s', 0.0)
        for nm, v in (('Tc', Tc), ('Tu', Tu), ('Hc', Hc),
                      ('Hu', Hu), ('Sc', Sc), ('Su', Su)):
            if v == v:
                tot[nm] += v
        print(f"  {k:>4} | {Tc:8.2f} {Tu:8.2f} {Tc-Tu:+8.2f} | "
              f"{Hc:7.2f} {Hu:7.2f} {Hc-Hu:+8.2f} | "
              f"{Sc:5.1f} {Su:5.1f} {Sc-Su:+6.2f}")
    print('  ' + '-' * (len(hdr) - 2))
    print(f"  {'TOT':>4} | {tot['Tc']:8.2f} {tot['Tu']:8.2f} "
          f"{tot['Tc']-tot['Tu']:+8.2f} | "
          f"{tot['Hc']:7.2f} {tot['Hu']:7.2f} {tot['Hc']-tot['Hu']:+8.2f} | "
          f"{tot['Sc']:5.1f} {tot['Su']:5.1f} {tot['Sc']-tot['Su']:+6.2f}")
    print()
    print('  per-step gate-evaluation failures before the dock fired')
    print(f"  {'step':>4} | {'C n_evals':>9} {'pos':>5} {'ori':>5} {'twist':>6}"
          f" | {'U n_evals':>9} {'pos':>5} {'ori':>5} {'twist':>6}")
    for k in sorted(c['per_step'], key=int):
        if int(k) < 0:
            continue
        cs, us = c['per_step'][k], u['per_step'].get(k, {})
        print(f"  {k:>4} | {cs['n_gate_evals']:>9} {cs['gate_fail_pos']:>5} "
              f"{cs['gate_fail_ori']:>5} {cs['gate_fail_twist']:>6} | "
              f"{us.get('n_gate_evals', 0):>9} {us.get('gate_fail_pos', 0):>5} "
              f"{us.get('gate_fail_ori', 0):>5} "
              f"{us.get('gate_fail_twist', 0):>6}")

    with open(os.path.join(OUT, 'c1_log_analysis.json'), 'w') as fh:
        json.dump(res, fh, indent=2)
    print()
    print(f"wrote {os.path.join(OUT, 'c1_log_analysis.json')}")


if __name__ == '__main__':
    main()
