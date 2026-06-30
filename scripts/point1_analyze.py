#!/usr/bin/env python3
"""POINT 1 (close the chatter fix): inter-config ε-robustness analyzer.

Works from sim_log.json ALONE (no postproc_F3F4.csv): the chatter signature is
the net contact force Σf = λ[0:3]+λ[6:9] flipping sign every tick (≡ the exact
Ḣ_s hitting ±5 — when Σf→0 the box is not binding and Ḣ_s→0). Per inter-step
settle: Σf sign-flip frac, ‖Σf‖ med, joint-vel flip frac, settle exit, dock hold.
'clean' = Σf-flip < 0.10 AND ‖Σf‖med < 1.0 N in the arm-a (chattering) settles.

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/point1_analyze.py results/p1_*
"""
import json
import os
import sys
import numpy as np


def settle_segments(sl):
    t = np.asarray(sl['t'], float)
    ph = np.asarray(sl['phase'])
    docks = sorted(sl.get('dock_events', []), key=lambda d: float(d['t']))
    out = []
    for s in (sl.get('inter_step_settles', []) or []):
        lo, hi = float(s['t_start']), float(s['t_end'])
        idx = np.where((ph == 'DS') & (t > lo + 1e-6) & (t <= hi + 1e-6))[0]
        if len(idx) < 3:
            continue
        prev = [d for d in docks if float(d['t']) <= lo + 0.05]
        arm = prev[-1].get('arm', '?') if prev else 'init'
        anc = prev[-1].get('anchor', -1) if prev else -1
        out.append((int(s['step_idx']), f'{arm}{anc if anc >= 0 else ""}', idx, s))
    return out


def analyze(run_dir):
    p = os.path.join(run_dir, 'sim_log.json')
    if not os.path.exists(p):
        return None
    sl = json.load(open(p))
    lq = np.asarray(sl['lambda_qp'], float)
    qa = np.asarray(sl['qvel_joints_a'], float)
    qb = np.asarray(sl['qvel_joints_b'], float)
    dgs = np.asarray(sl['d_grip_stance'], float)
    rc = np.asarray(sl['r_com'], float)
    de = sl.get('dock_events', [])
    ab = sl.get('aborted_steps', [])
    rmax = float(np.max(np.linalg.norm(rc, axis=1)))
    segs = []
    for (step, arm, idx, s) in settle_segments(sl):
        Sf = lq[idx, 0:3] + lq[idx, 6:9]              # net contact force Σf
        # flip-frac: per-axis sign flip tick-to-tick, max axis (the chatter axis)
        ff = max(np.mean(np.sign(Sf[1:, a]) != np.sign(Sf[:-1, a])) for a in range(3))
        sfmed = float(np.median(np.linalg.norm(Sf, axis=1)))
        comp = np.concatenate([qa[idx], qb[idx]], axis=1)
        dom = int(np.argmax(np.abs(comp).mean(0)))
        jflip = float(np.mean(comp[1:, dom] * comp[:-1, dom] < 0))
        jmean = float(np.linalg.norm(comp, axis=1).mean())
        clean = (ff < 0.10) and (sfmed < 1.0)
        segs.append(dict(step=step, arm=arm, n=len(idx), sf_flip=float(ff),
                         sf_med=sfmed, j_flip=jflip, j_mean=jmean,
                         T_start=s['T_start'], T_end=s['T_end'],
                         exit=s['exit_reason'], dock_mm=float(dgs[idx].max() * 1000),
                         clean=clean, r_com=float(np.median(np.linalg.norm(rc[idx], axis=1)))))
    return dict(run=os.path.basename(run_dir), ndock=len(de), rmax=rmax,
                abort=[a.get('reason') for a in ab],
                docks=[round(d['d_mm'], 1) for d in de], segs=segs)


def main(argv):
    rows = [r for r in (analyze(d) for d in argv if os.path.isdir(d)) if r]
    rows.sort(key=lambda r: r['run'])
    for r in rows:
        print(f"\n=== {r['run']:14s}  docks={r['ndock']}/5 {r['docks']}  "
              f"max|r_com|={r['rmax']:.2f}m  abort={r['abort']}")
        if not r['segs']:
            print("   (no measurable inter-step settles)")
            continue
        print(f"   {'settle':10s} {'n':>4s} {'r_com':>6s} {'Σf-flip':>8s} {'‖Σf‖med':>8s} "
              f"{'jflip':>6s} {'jmean':>7s} {'T_end':>9s} {'exit':>10s} {'dock_mm':>7s} clean")
        for s in r['segs']:
            print(f"   step{s['step']}/{s['arm']:4s} {s['n']:4d} {s['r_com']:6.2f} "
                  f"{s['sf_flip']:8.3f} {s['sf_med']:8.2f} {s['j_flip']:6.2f} "
                  f"{s['j_mean']:7.4f} {s['T_end']:9.1e} {s['exit']:>10s} {s['dock_mm']:7.4f} "
                  f"{'YES' if s['clean'] else 'no'}")
    # robustness summary: arm-a settles only (the chatterers)
    print(f"\n{'='*70}\nARM-A SETTLE CLEANLINESS (the chatterers) per run")
    for r in rows:
        arma = [s for s in r['segs'] if s['arm'].startswith('a')]
        if arma:
            allclean = all(s['clean'] for s in arma)
            flips = ','.join('%.2f' % s['sf_flip'] for s in arma)
            print(f"  {r['run']:14s}: arm-a settles={len(arma)}  "
                  f"flip=[{flips}]  all_clean={allclean}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
