"""COM-GAIN-AUDIT Phase 2b: CoM tracking performance.

Answers two questions the Phase 0-2 audit left open:

  Q1. What is the closed-loop CoM tracking performance of the canonical run,
      measured on the exported reference (`r_com_ref_*`, `e_com_m`) rather
      than on the per-tick residual the QP task differences?

  Q2. Is the rank-one gain defect visible in that tracking performance, or is
      the QP CoM task's authority bounded by something else?

Reads only committed artifacts. Writes no canonical file.
"""
import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULLDIAG = os.path.join(ROOT, 'results/j2_adjconv/c25_fulldiag.csv')
AUDIT = os.path.join(ROOT, 'results/j2_adjconv/com_gain_audit_ticks.csv')


def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def col(rows, name, cast=float):
    out = []
    for r in rows:
        v = r.get(name, '')
        try:
            out.append(cast(v))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.asarray(out)


def stats(label, a, unit='mm', scale=1e3):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        print(f'  {label:<34} (no finite samples)')
        return
    print(f'  {label:<34} n={a.size:<5d} '
          f'median={np.median(a) * scale:9.3f} '
          f'p95={np.percentile(a, 95) * scale:9.3f} '
          f'max={a.max() * scale:9.3f} {unit}')


def main():
    fd = load(FULLDIAG)
    print(f'fulldiag rows: {len(fd)}')

    phase = np.array([r.get('phase', '') for r in fd])
    print(f'phase labels: {sorted(set(phase))}')

    r = np.stack([col(fd, f'r_com_{k}_m') for k in 'xyz'], axis=1)
    ref = np.stack([col(fd, f'r_com_ref_{k}_m') for k in 'xyz'], axis=1)
    e_exported = col(fd, 'e_com_m')

    e_vec = ref - r
    e_norm = np.linalg.norm(e_vec, axis=1)

    # ---- Q1: closed-loop tracking, exported reference -------------------
    print('\n=== Q1  closed-loop CoM tracking (exported reference) ===')
    stats('e_com_m column, ALL', e_exported)
    stats('|ref - r| recomputed, ALL', e_norm)
    for ph in sorted(set(phase)):
        m = phase == ph
        stats(f'|ref - r|, phase={ph}', e_norm[m])

    # The exported column and the recomputed norm must agree, else the
    # export convention is doing something to one of them.
    both = np.isfinite(e_exported) & np.isfinite(e_norm)
    if both.any():
        d = np.abs(e_exported[both] - e_norm[both])
        print(f'\n  export self-consistency |e_com_m - |ref-r||: '
              f'max={d.max() * 1e3:.6f} mm  '
              f'(disagreements >1um: {(d > 1e-6).sum()}/{both.sum()})')

    # ---- per-axis structure --------------------------------------------
    print('\n=== per-axis signed CoM error (SS only) ===')
    ss = phase == 'SS'
    if ss.any():
        for i, k in enumerate('xyz'):
            a = e_vec[ss, i]
            a = a[np.isfinite(a)]
            print(f'  e_com_{k}: median={np.median(a) * 1e3:+9.3f}  '
                  f'mean={a.mean() * 1e3:+9.3f}  '
                  f'min={a.min() * 1e3:+9.3f}  max={a.max() * 1e3:+9.3f} mm')
        # How much of the error is in the rank-one-invisible subspace?
        # The buggy law only sees sum(e); the component orthogonal to
        # [1,1,1] is structurally invisible to it.
        u = np.ones(3) / np.sqrt(3.0)
        ev = e_vec[ss]
        ev = ev[np.isfinite(ev).all(axis=1)]
        par = ev @ u
        perp = np.linalg.norm(ev - np.outer(par, u), axis=1)
        nz = np.linalg.norm(ev, axis=1) > 1e-12
        frac = perp[nz] / np.linalg.norm(ev[nz], axis=1)
        print(f'\n  invisible (perp to [1,1,1]) fraction of e_com in SS:')
        print(f'    median={np.median(frac):.4f}  mean={frac.mean():.4f}  '
              f'max={frac.max():.4f}   n={frac.size}')

    # ---- Q2: what the QP task actually saw -----------------------------
    if not os.path.exists(AUDIT):
        print('\n(audit CSV absent - skipping Q2)')
        return
    ad = load(AUDIT)
    print(f'\n=== Q2  what the QP CoM task saw (audit replay, {len(ad)} rows) ===')
    aphase = np.array([r['phase'] for r in ad])
    er = col(ad, 'er_norm')
    dfrac = col(ad, 'delivered_frac')
    resid = col(ad, 'resid_norm')
    ades = col(ad, 'ades_norm')
    perp_frac = col(ad, 'perp_frac')

    for ph in sorted(set(aphase)):
        m = aphase == ph
        stats(f'|e_r| seen by QP task, {ph}', er[m])

    print('\n  task delivery (dimensionless):')
    for ph in sorted(set(aphase)):
        m = aphase == ph
        d = dfrac[m][np.isfinite(dfrac[m])]
        print(f'    delivered_frac {ph}: median={np.median(d):.4f}  '
              f'p95={np.percentile(d, 95):.4f}  n={d.size}')
    stats('  |a_com_des| commanded', ades, unit='mm/s^2')
    stats('  |residual| left by stack', resid, unit='mm/s^2')

    pf = perp_frac[np.isfinite(perp_frac)]
    print(f'\n  perp_frac of e_r (audit): median={np.median(pf):.4f} '
          f'max={pf.max():.4f}')

    # The decisive comparison: the task's own error vs the closed-loop error.
    print('\n=== decisive comparison ===')
    ss_er = er[aphase == 'SS']
    ss_er = ss_er[np.isfinite(ss_er)]
    ss_ecom = e_norm[phase == 'SS']
    ss_ecom = ss_ecom[np.isfinite(ss_ecom)]
    if ss_er.size and ss_ecom.size:
        print(f'  |e_r| the QP differences (SS):  max={ss_er.max() * 1e3:9.3f} mm')
        print(f'  |e_com| closed loop     (SS):  max={ss_ecom.max() * 1e3:9.3f} mm')
        print(f'  ratio: {ss_ecom.max() / max(ss_er.max(), 1e-15):.1f}x')

    out = {
        'e_com_all_max_mm': float(np.nanmax(e_norm) * 1e3),
        'e_com_ss_max_mm': float(ss_ecom.max() * 1e3) if ss_ecom.size else None,
        'e_com_ss_median_mm': float(np.median(ss_ecom) * 1e3) if ss_ecom.size else None,
        'er_ss_max_mm': float(ss_er.max() * 1e3) if ss_er.size else None,
        'invisible_frac_median': float(np.median(frac)) if ss.any() else None,
    }
    dest = os.path.join(ROOT, 'results/j2_adjconv/com_tracking_perf.json')
    with open(dest, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f'\nwrote {dest}')


if __name__ == '__main__':
    main()
