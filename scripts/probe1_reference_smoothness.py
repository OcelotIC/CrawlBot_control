#!/usr/bin/env python3
"""Probe 1 — reference-channel smoothness: NMPC momentum vs δ-mapping.

A CMM-interface QP task would track the NMPC's NATIVE momentum output
(v_com_ref, L_com_ref) and the CoM position r_com_ref. The current
architecture instead tracks p_torso_ref = (m_t/m_b)·r_com_ref −
δ(q_current)/m_b (post-F-SAT) — a position-level inversion that
recomputes δ at the live 100 Hz config on top of the 10 Hz NMPC plan.

Premise to confirm: the momentum/CoM channel the CMM task would track is
SMOOTH, while the δ-mapping channel the QP tracks today is JITTERY at the
same ticks (it needs the F-SAT rate-clamp on 49.6% of SS ticks). If true,
swapping to a CMM task removes the jitter source rather than relocating
it. Falsifier: if r_com_ref/v_com_ref are themselves jittery, or if
p_torso_ref is no rougher than r_com_ref, the premise fails.

All quantities are LOGGED — pure analysis, no model needed.

Reads:  Misc/runs/diag_cooperative_arms/sim_log.json
Writes: Misc/runs/diag_attribution/probe1_reference_smoothness/{metrics.json, probe1.png}
"""
import os
import sys
import json

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG = os.path.join(_root, 'Misc', 'runs', 'diag_cooperative_arms', 'sim_log.json')
OUT = os.path.join(_root, 'Misc', 'runs', 'diag_attribution',
                   'probe1_reference_smoothness')


def incr_stats(x, mask):
    """Per-tick |Δ| stats of a (N,k) signal over a boolean tick mask.
    Increments computed on the full series, then masked (so we measure
    tick-to-tick jumps, not phase-boundary jumps)."""
    d = np.linalg.norm(np.diff(x, axis=0), axis=1)        # (N-1,)
    m = mask[1:]
    dd = d[m]
    return {
        'median_mm': float(np.median(dd) * 1000),
        'p95_mm': float(np.percentile(dd, 95) * 1000),
        'max_mm': float(dd.max() * 1000),
        'rough_ratio_max_over_median': float(dd.max() / max(np.median(dd), 1e-12)),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    log = json.load(open(LOG))
    t = np.array(log['t'])
    ph = np.array(log['phase'])
    step = np.array(log['step_idx'])
    r_com_ref = np.array(log['r_com_ref'])      # NMPC CoM position plan
    p_torso_ref = np.array(log['p_torso_ref'])  # δ-mapped torso ref (post-F-SAT)
    v_com_ref = np.array(log['v_com_ref'])      # NMPC CoM velocity plan
    L_com_ref = np.array(log['L_com_ref'])      # NMPC angular-momentum plan
    lam = np.array(log['lambda_qp'])
    f1n = np.linalg.norm(lam[:, 0:3], axis=1)

    ss_all = (ph == 'SS')
    ss2 = (step == 2) & (ph == 'SS')

    metrics = {}
    for tag, mask in [('all_SS', ss_all), ('step2_SS', ss2)]:
        metrics[tag] = {
            'r_com_ref (NMPC CoM, smooth?)': incr_stats(r_com_ref, mask),
            'p_torso_ref (δ-mapping, jittery?)': incr_stats(p_torso_ref, mask),
            'v_com_ref (NMPC vel, m/s units)': incr_stats(v_com_ref, mask),
            'L_com_ref (NMPC ang-mom, Nms units)': incr_stats(L_com_ref, mask),
        }
        # excess roughness: how much rougher is the mapping channel than
        # the NMPC CoM plan it is derived from?
        rc = metrics[tag]['r_com_ref (NMPC CoM, smooth?)']
        pt = metrics[tag]['p_torso_ref (δ-mapping, jittery?)']
        metrics[tag]['mapping_excess_roughness'] = {
            'p_torso_ref_max / r_com_ref_max': float(
                pt['max_mm'] / max(rc['max_mm'], 1e-9)),
            'p_torso_ref_p95 / r_com_ref_p95': float(
                pt['p95_mm'] / max(rc['p95_mm'], 1e-9)),
        }

    # coincidence: does the step-2 f1 peak land on a p_torso_ref jerk?
    idx2 = np.where(ss2)[0]
    jpk = int(idx2[int(np.argmax(f1n[idx2]))])
    d_pt = np.r_[0, np.linalg.norm(np.diff(p_torso_ref, axis=0), axis=1)]
    d_rc = np.r_[0, np.linalg.norm(np.diff(r_com_ref, axis=0), axis=1)]
    metrics['coincidence_at_step2_f1_peak'] = {
        't': float(t[jpk]),
        'f1_N': float(f1n[jpk]),
        'p_torso_ref_incr_mm': float(d_pt[jpk] * 1000),
        'p_torso_ref_incr_percentile_in_step2SS': float(
            100 * (d_pt[idx2] <= d_pt[jpk]).mean()),
        'r_com_ref_incr_mm': float(d_rc[jpk] * 1000),
    }

    with open(os.path.join(OUT, 'metrics.json'), 'w') as fh:
        json.dump(metrics, fh, indent=2)

    # ---- plot ----
    tt = t[idx2]
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(tt, d_rc[idx2] * 1000, label='Δ r_com_ref (NMPC CoM plan)', lw=1.8)
    ax[0].plot(tt, d_pt[idx2] * 1000, label='Δ p_torso_ref (δ-mapping, post-F-SAT)', lw=1.8)
    ax[0].axvline(t[jpk], color='k', ls=':', alpha=0.6)
    ax[0].set_ylabel('per-tick |Δ| [mm]')
    ax[0].set_title('Step-2 SS: NMPC CoM-plan vs δ-mapped torso ref — '
                    'per-tick increment (jitter)')
    ax[0].legend(fontsize=8)

    ax[1].plot(tt, f1n[idx2], color='C3', lw=1.8, label='|f1| QP contact force')
    ax[1].axvline(t[jpk], color='k', ls=':', alpha=0.6,
                  label=f'f1 peak (t={t[jpk]:.2f})')
    ax[1].set_ylabel('|f1| [N]')
    ax[1].set_xlabel('t [s]')
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'probe1.png'), dpi=110)

    # ---- stdout ----
    print('=' * 70)
    print(' Probe 1 — reference-channel smoothness (per-tick |Δ|)')
    print('=' * 70)
    for tag in ('all_SS', 'step2_SS'):
        print(f"\n[{tag}]")
        for ch, s in metrics[tag].items():
            if ch == 'mapping_excess_roughness':
                continue
            print(f"  {ch:<40} median={s['median_mm']:7.3f}  "
                  f"p95={s['p95_mm']:7.3f}  max={s['max_mm']:8.3f}  "
                  f"max/med={s['rough_ratio_max_over_median']:6.1f}  [mm or m/s·1e3]")
        ex = metrics[tag]['mapping_excess_roughness']
        print(f"  → p_torso_ref rougher than r_com_ref:  "
              f"max {ex['p_torso_ref_max / r_com_ref_max']:.1f}× , "
              f"p95 {ex['p_torso_ref_p95 / r_com_ref_p95']:.1f}×")
    c = metrics['coincidence_at_step2_f1_peak']
    print(f"\n[f1 peak coincidence]  t={c['t']:.2f}  |f1|={c['f1_N']:.1f} N")
    print(f"  Δp_torso_ref = {c['p_torso_ref_incr_mm']:.2f} mm "
          f"({c['p_torso_ref_incr_percentile_in_step2SS']:.0f}th pct of step-2 SS)  "
          f"| Δr_com_ref = {c['r_com_ref_incr_mm']:.3f} mm")
    print(f"\n  saved: {os.path.relpath(OUT, _root)}/{{metrics.json, probe1.png}}")


if __name__ == '__main__':
    main()
