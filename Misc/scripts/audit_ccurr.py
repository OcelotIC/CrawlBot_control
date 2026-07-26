#!/usr/bin/env python3
"""J2 c_curr characterization — per-tick refresh of the inter-step QP hw_current.

Reads N run dirs (label=dir ...) and reports, per run:
  - h_w∞ (per-axis ∞-norm, the C5 metric) whole-traversal AND split SS /
    inter-step DS / DWELL-or-terminal DS  — tells whether C5 pressure is the
    SS exact-box binding or the inter-step (c_curr's domain).
  - STALENESS: max intra-settle drift of the live wheel momentum from its
    settle-entry value, max|hw_phys(t) − hw_phys(settle_start)| — this is how
    stale the entry-frozen hw_current was getting (what c_curr removes).
  - settle convergence: per-settle n_steps / exit / T_end (should be ~unchanged).
RAW; no pass/fail. C1-C6 / dock / residual come from gate_phase3 separately.

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/audit_ccurr.py \
        off=results/ccurr_proxy_off on=results/ccurr_proxy_on
"""
import json
import os
import sys
import numpy as np


def load(d):
    with open(os.path.join(d, 'sim_log.json')) as f:
        return json.load(f)


def interstep_mask(sl):
    t = np.asarray(sl['t'], float); ph = np.asarray(sl['phase'])
    m = np.zeros(len(t), bool)
    for s in (sl.get('inter_step_settles', []) or []):
        m |= (ph == 'DS') & (t > float(s['t_start']) + 1e-6) & (t <= float(s['t_end']) + 1e-6)
    return m


def hw_inf(hwp, mask=None):
    """Per-axis ∞-norm (max |hw_i|) over the (masked) ticks — the C5 metric."""
    a = hwp if mask is None else hwp[mask]
    if a.shape[0] == 0:
        return float('nan')
    return float(np.max(np.abs(a)))


def analyze(tag, sl):
    t = np.asarray(sl['t'], float); ph = np.asarray(sl['phase'])
    hwp = np.asarray(sl['hw_physical'], float)
    m_is = interstep_mask(sl)
    m_ss = (ph == 'SS')
    m_dsT = (ph == 'DS') & ~m_is          # terminal/DWELL DS (_step)
    print(f"\n========== {tag} ==========")
    print(f"  h_w∞ (per-axis, C5 metric):  whole={hw_inf(hwp):.4f}  "
          f"SS={hw_inf(hwp, m_ss):.4f}  inter-step={hw_inf(hwp, m_is):.4f}  "
          f"DWELL/termDS={hw_inf(hwp, m_dsT):.4f}  Nms")

    # staleness: per-settle max drift of live hw from the settle-entry value
    idx = np.where(m_is)[0]
    worst_stale = 0.0
    if len(idx):
        groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        print("  staleness (live hw − settle-entry hw), per inter-step settle:")
        for g in groups:
            entry = hwp[g[0]]
            drift = np.linalg.norm(hwp[g] - entry, axis=1)  # 2-norm of the per-tick offset
            d = float(drift.max())
            worst_stale = max(worst_stale, d)
            print(f"    settle n={len(g):3d}: max‖hw_live−hw_entry‖={d:.4f} Nms "
                  f"(entry‖hw‖={np.linalg.norm(entry):.3f} → max‖hw‖={np.linalg.norm(hwp[g],axis=1).max():.3f})")
    print(f"  WORST intra-settle staleness = {worst_stale:.4f} Nms  "
          f"(= how stale the entry-frozen hw_current got; c_curr removes this)")

    # settle convergence
    print("  settles (n / exit / T_start→T_end):")
    for s in (sl.get('inter_step_settles', []) or []):
        print(f"    step {s['step_idx']}: n={s['n_steps']:4d} {s['exit_reason']:11s} "
              f"{s['T_start']:.2e}→{s['T_end']:.2e}")
    return {'hw_whole': hw_inf(hwp), 'hw_ss': hw_inf(hwp, m_ss),
            'hw_is': hw_inf(hwp, m_is), 'worst_stale': worst_stale}


def main(argv):
    runs = {}
    order = []
    for a in argv:
        if '=' in a:
            k, v = a.split('=', 1); runs[k] = v; order.append(k)
    if not runs:
        print("usage: audit_ccurr.py off=<dir> on=<dir> [exact_on=<dir> ...]"); return 2
    print('=' * 72)
    print('J2 c_curr — per-tick inter-step QP hw_current refresh: A/B')
    print('=' * 72)
    res = {k: analyze(k.upper(), load(runs[k])) for k in order}
    print('\n' + '=' * 72)
    print('SUMMARY  (h_w∞ per-axis = the C5 metric; cap 4.5, physical 5.0)')
    print('=' * 72)
    print(f"  {'run':16s} {'hw∞ whole':>10s} {'hw∞ SS':>9s} {'hw∞ inter':>10s} {'worst-stale':>12s}")
    for k in order:
        r = res[k]
        print(f"  {k:16s} {r['hw_whole']:10.4f} {r['hw_ss']:9.4f} {r['hw_is']:10.4f} "
              f"{r['worst_stale']:12.4f}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
