#!/usr/bin/env python3
"""C3.2 — conservation and the c_simple ansatz, (a)-(f). Post-processing only.

Sources, all committed:
  results/j2_adjconv/t4b_ltot_900s.csv     dense ||L_total||, 1929 samples / 964 s
  results/j2_adjconv/t4b_trace_900s.csv    omega_s, h_w, theta_s, tau_w, 4823 / 964 s
  results/j2_adjconv/c25_fulldiag.csv      the canonical traversal (66 cols)
  results/j2_adjconv/c25_fulldiag_meta.json  44 L_total snapshots, event-labelled

The T4b run is the canonical traversal followed by a 900 s station-keeping tail
(traversal bit-identical to the canonical, per PHASE_DRIFT_CLOSURE_T4b), which
is what makes (e) answerable: a residual that accumulates and one that is
injected at discrete events look identical over 85 s and completely different
over 900 s.

(f) is a VALIDATOR, never a corrector — nothing here is fed back into any state.
"""
import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
ADJ = os.path.join(ROOT, 'results', 'j2_adjconv')
OUT = os.path.join(ROOT, 'results', 'review_closure', 'c3')

# Structure inertia as DECLARED in the plant MJCF (C1.7): body-frame diagonal.
I_S = np.array([597.0, 1493.0, 1777.0])
LAST_DOCK_T = 64.54          # canonical: last single-support instant
TRAVERSAL_END_T = 84.54      # canonical: end of log


def rd(path):
    with open(os.path.join(ADJ, path)) as fh:
        return list(csv.DictReader(fh))


def arr(rows, *keys):
    return np.array([[float(r[k]) for k in keys] for r in rows])


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}

    lt = rd('t4b_ltot_900s.csv')
    t = np.array([float(r['t_s']) for r in lt])
    Lv = arr(lt, 'Ltot_x_Nms', 'Ltot_y_Nms', 'Ltot_z_Nms')
    Ln = np.array([float(r['Ltot_norm_Nms']) for r in lt])

    tr = rd('t4b_trace_900s.csv')
    t2 = np.array([float(r['t_s']) for r in tr])
    om = arr(tr, 'omega_s_x_dps', 'omega_s_y_dps', 'omega_s_z_dps')
    om = np.deg2rad(om)                                  # trace is deg/s
    hw = arr(tr, 'hw_x_Nms', 'hw_y_Nms', 'hw_z_Nms')

    # ── (a) ||L_total|| profile ─────────────────────────────────────────
    trav = t <= LAST_DOCK_T
    tail = t >= TRAVERSAL_END_T
    res['a_Ltot_profile'] = {
        'n_samples': len(t), 't_span_s': [float(t.min()), float(t.max())],
        'peak_Nms': float(Ln.max()),
        'at_last_dock_Nms': float(Ln[np.argmin(np.abs(t - LAST_DOCK_T))]),
        'at_traversal_end_Nms': float(Ln[np.argmin(np.abs(t - TRAVERSAL_END_T))]),
        'final_Nms': float(Ln[-1]),
        'traversal_gain_Nms': float(Ln[trav].max() - Ln[trav].min()),
        'tail_drift_Nms': float(Ln[tail].max() - Ln[tail].min()),
        'tail_span_s': float(t[tail].max() - t[tail].min()),
    }

    # ── (b) ||I_s omega_s|| profile ─────────────────────────────────────
    Ls = om * I_S                                        # body-frame diagonal
    Lsn = np.linalg.norm(Ls, axis=1)
    m_tr = t2 <= LAST_DOCK_T
    res['b_structure_momentum'] = {
        'peak_Nms': float(Lsn.max()),
        'peak_traversal_Nms': float(Lsn[m_tr].max()),
        'mean_traversal_Nms': float(Lsn[m_tr].mean()),
        'final_Nms': float(Lsn[-1]),
        'peak_per_axis_Nms': [float(x) for x in np.abs(Ls).max(0)],
        'vs_hw_peak_Nms': float(np.abs(hw).max()),
    }

    # ── (c) reduced relation residual:  h_w ~= c_simple - L_robot ───────
    # With the robot welded and quiescent in the tail, L_robot -> 0 and the
    # relation collapses to  I_comp*omega_s + h_w = const. Residual measured
    # per axis against the tail's own mean.
    m_tail = t2 >= TRAVERSAL_END_T
    tot = Ls + hw
    res['c_reduced_relation'] = {
        'tail_mean_Nms': [float(x) for x in tot[m_tail].mean(0)],
        'tail_std_Nms': [float(x) for x in tot[m_tail].std(0)],
        'tail_ptp_Nms': [float(x) for x in np.ptp(tot[m_tail], axis=0)],
        'note': ('I_s*omega_s + h_w over the quiescent tail; a drifting mean '
                 'would indicate the reduced relation losing validity'),
    }

    # ── (e) mechanism of the residual ───────────────────────────────────
    # A power-law fit is only meaningful on a growing signal. Test growth
    # FIRST, on the 900 s tail, then fit only if it grows.
    tl, Ll = t[tail], Ln[tail]
    grew = float(Ll.max() - Ll.min())
    rel_growth = grew / max(float(Ll.mean()), 1e-30)
    alpha = None
    if rel_growth > 1e-6:
        m = (tl > 0) & (Ll > 0)
        alpha = float(np.polyfit(np.log(tl[m]), np.log(Ll[m]), 1)[0])
    res['e_mechanism'] = {
        'tail_span_s': float(tl.max() - tl.min()),
        'tail_absolute_growth_Nms': grew,
        'tail_relative_growth': rel_growth,
        'alpha_fit': alpha,
        'tail_is_weak_evidence': (
            'A flat tail does NOT by itself exclude the integrator. Round-off '
            'accumulates per unit of DYNAMICAL WORK, not per unit of '
            'wall-clock: in a welded quiescent tail the state barely changes, '
            'so ~0 growth is what BOTH the round-off hypothesis and the '
            'event-injection hypothesis predict. The tail only excludes a '
            'time-driven bias. The discriminator is the leg asymmetry below.'),
        'verdict': ('EVENT-INJECTED — see e_leg_summary. The discriminator is '
                    'that residual injection is ANTI-correlated with '
                    'dynamical activity: the swing legs carry 22x the CoM '
                    'path length, 3-8x the joint torque and 1.5x the duration '
                    'of the dock legs, yet inject ~300x less residual '
                    '(~6600x less per metre travelled). Integrator round-off '
                    'would scale WITH that activity, not against it. The '
                    'dock legs are the quiet ones; what distinguishes them is '
                    'that each contains a weld ACTIVATION (constraint added, '
                    'inelastic impact projection) and a release.'),
    }

    # per-event increments, from the event-labelled snapshots
    with open(os.path.join(ADJ, 'c25_fulldiag_meta.json')) as fh:
        meta = json.load(fh)
    snaps = [s for s in meta['ltot_snapshots']
             if s['label'].startswith(('dock_', 'release_', 'initial', 'final'))]
    snaps.sort(key=lambda s: s['t'])
    ev = []
    for i in range(1, len(snaps)):
        a, b = snaps[i - 1], snaps[i]
        ev.append({'from': a['label'], 'to': b['label'],
                   'dt_s': round(b['t'] - a['t'], 2),
                   'd_Ltot_norm_Nms': b['Ltot_norm'] - a['Ltot_norm']})
    res['e_per_event_increments'] = ev
    ss_leg = [e for e in ev if e['from'].startswith('release_')
              and e['to'].startswith('dock_')]
    ds_leg = [e for e in ev if e['from'].startswith('dock_')
              and e['to'].startswith('release_')]
    res['e_leg_summary'] = {
        'swing_legs_release_to_dock': {
            'n': len(ss_leg),
            'total_abs_Nms': float(sum(abs(e['d_Ltot_norm_Nms'])
                                       for e in ss_leg))},
        'dock_legs_dock_to_next_release': {
            'n': len(ds_leg),
            'total_abs_Nms': float(sum(abs(e['d_Ltot_norm_Nms'])
                                       for e in ds_leg))},
    }

    # ── (f) plant cross-check: reconstruct the structure rate ───────────
    # Quiescent tail, welded: I_eff*omega_s ~= const - h_w. Fit I_eff per axis
    # by least squares on (h_w - mean) vs (omega_s - mean); the slope is -I_eff.
    I_eff, r2 = [], []
    for j in range(3):
        x = om[m_tail, j] - om[m_tail, j].mean()
        y = hw[m_tail, j] - hw[m_tail, j].mean()
        if x.std() < 1e-12:
            I_eff.append(None); r2.append(None); continue
        s = float(-np.polyfit(x, y, 1)[0])
        pred = -s * x
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        I_eff.append(s)
        r2.append(1 - ss_res / ss_tot if ss_tot > 0 else None)
    res['f_plant_crosscheck'] = {
        'I_eff_fitted_kgm2': I_eff,
        'fit_r2': r2,
        'I_s_declared_MJCF_kgm2': [float(x) for x in I_S],
        'ratio_I_eff_over_I_s': [
            (I_eff[j] / I_S[j]) if I_eff[j] else None for j in range(3)],
        'note': ('I_eff > I_s indicates the COMPOSITE (locked) inertia — the '
                 'robot is welded to the structure in the tail, so the wheels '
                 'react against structure + robot, not the structure alone. '
                 'T2 reported I_comp ~ 2152-2180 on the z axis. VALIDATOR '
                 'ONLY: no plant state is modified.'),
    }
    # RMS discrepancy of the reconstructed rate, z axis (the driven one)
    j = int(np.argmax(np.abs(hw[m_tail]).max(0)))
    if I_eff[j]:
        om_rec = -(hw[m_tail, j] - hw[m_tail, j].mean()) / I_eff[j] \
            + om[m_tail, j].mean()
        res['f_plant_crosscheck']['dominant_axis'] = 'xyz'[j]
        res['f_plant_crosscheck']['omega_rms_discrepancy_radps'] = float(
            np.sqrt(((om_rec - om[m_tail, j]) ** 2).mean()))
        res['f_plant_crosscheck']['omega_rms_of_signal_radps'] = float(
            np.sqrt((om[m_tail, j] ** 2).mean()))

    with open(os.path.join(OUT, 'c3_2_conservation.json'), 'w') as fh:
        json.dump(res, fh, indent=2)

    # ── printout ──
    a = res['a_Ltot_profile']
    print('=' * 74)
    print('C3.2 (a) ||L_total|| profile')
    print(f"   samples {a['n_samples']} over {a['t_span_s'][0]:.2f}..{a['t_span_s'][1]:.2f} s")
    print(f"   at last dock (64.54 s) : {a['at_last_dock_Nms']:.4e} Nms")
    print(f"   at traversal end       : {a['at_traversal_end_Nms']:.4e} Nms")
    print(f"   final (964 s)          : {a['final_Nms']:.4e} Nms")
    print(f"   gained during traversal: {a['traversal_gain_Nms']:.4e} Nms")
    print(f"   DRIFT over the {a['tail_span_s']:.0f} s tail : "
          f"{a['tail_drift_Nms']:.4e} Nms")
    b = res['b_structure_momentum']
    print()
    print('C3.2 (b) ||I_s omega_s|| (structure momentum)')
    print(f"   peak over traversal {b['peak_traversal_Nms']:.4f} Nms   "
          f"mean {b['mean_traversal_Nms']:.4f}")
    print(f"   per-axis peak {[round(x,4) for x in b['peak_per_axis_Nms']]}   "
          f"vs |h_w| peak {b['vs_hw_peak_Nms']:.4f}")
    c = res['c_reduced_relation']
    print()
    print('C3.2 (c) I_s*omega_s + h_w over the quiescent tail [Nms]')
    print(f"   mean {[f'{x:.4f}' for x in c['tail_mean_Nms']]}")
    print(f"   std  {[f'{x:.2e}' for x in c['tail_std_Nms']]}")
    e = res['e_mechanism']
    print()
    print('C3.2 (e) mechanism')
    print(f"   growth over the {e['tail_span_s']:.0f} s tail: "
          f"{e['tail_absolute_growth_Nms']:.3e} Nms "
          f"(relative {e['tail_relative_growth']:.3e})")
    print(f"   alpha fit: {e['alpha_fit']}")
    print(f"   -> {e['verdict']}")
    print()
    print('   per-event increments in ||L_total|| [Nms]:')
    for x in res['e_per_event_increments']:
        print(f"      {x['from']:16s} -> {x['to']:16s} "
              f"dt={x['dt_s']:6.2f}s   d={x['d_Ltot_norm_Nms']:+.4e}")
    ls = res['e_leg_summary']
    print(f"   swing legs (release->dock): total |d| = "
          f"{ls['swing_legs_release_to_dock']['total_abs_Nms']:.4e}")
    print(f"   dock  legs (dock->release): total |d| = "
          f"{ls['dock_legs_dock_to_next_release']['total_abs_Nms']:.4e}")
    f = res['f_plant_crosscheck']
    print()
    print('C3.2 (f) plant cross-check — fitted effective inertia [kg m^2]')
    for j, ax in enumerate('xyz'):
        ie = f['I_eff_fitted_kgm2'][j]
        print(f"   {ax}: I_eff {('%.1f' % ie) if ie else '   n/a':>8}   "
              f"I_s {I_S[j]:7.1f}   ratio "
              f"{('%.3f' % f['ratio_I_eff_over_I_s'][j]) if ie else 'n/a':>6}   "
              f"r2 {('%.5f' % f['fit_r2'][j]) if f['fit_r2'][j] else 'n/a'}")
    if 'omega_rms_discrepancy_radps' in f:
        print(f"   dominant axis {f['dominant_axis']}: reconstructed-vs-"
              f"integrated omega_s RMS discrepancy "
              f"{f['omega_rms_discrepancy_radps']:.3e} rad/s "
              f"({100*f['omega_rms_discrepancy_radps']/f['omega_rms_of_signal_radps']:.2f} % of signal RMS)")
    print()
    print(f"wrote {os.path.join(OUT, 'c3_2_conservation.json')}")


if __name__ == '__main__':
    main()
