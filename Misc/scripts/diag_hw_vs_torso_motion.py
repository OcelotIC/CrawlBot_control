#!/usr/bin/env python3
"""READ-ONLY diagnostic: is the baseline's lower h_w a consequence of poorer torso
positioning (less torso motion -> less reaction -> less stored wheel momentum)?

Per SS swing, for baseline (dcda974) and two-task (phase3_wp), correlate torso MOTION
amplitude (integrated displacement, peak speed/accel, angular path) with the swing's
h_w peak (inf-norm). If h_w tracks torso motion on a common trend across both
controllers, the baseline's low h_w is explained by its lesser motion (hypothesis
CONFIRMED). Otherwise quantify the unexplained fraction. No sim, no edit beyond the
figure + note. Focus: binding swings 2 & 4 (where the C5 miss occurs).
"""
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from crawlbot.core.robot_interface import RobotInterface

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
RES = os.path.join(ROOT, 'results')
BASE = 'ssmom_phase1_baseline_main_dcda974'
WP = 'phase3_wp'
OUT = os.path.join(RES, WP, 'phase3_plots')
BINDING = {2, 4}

_robot = RobotInterface(os.path.join(ROOT, 'models', 'VISPA_crawling_fixed.urdf'),
                        gravity='zero')
_zv = np.zeros(_robot.model.nv)


def fk_torso(q):
    return _robot.update(np.array(q, float), _zv).oMf_torso.translation.copy()


def _rel_torso(sl):
    """Torso position in the STRUCTURE frame (removes the bulk crawl along the
    truss) — the reaction-relevant relative motion. struct_quat is (w,x,y,z)."""
    import pinocchio as pin
    p = np.array(sl['p_torso']); ps = np.array(sl['struct_pos'])
    q = np.array(sl['struct_quat'])
    rel = np.empty_like(p)
    for i in range(len(p)):
        R = pin.Quaternion(q[i, 0], q[i, 1], q[i, 2], q[i, 3]).toRotationMatrix()
        rel[i] = R.T @ (p[i] - ps[i])
    return rel


def per_swing(run):
    sl = json.load(open(os.path.join(RES, run, 'sim_log.json')))
    rows = list(csv.DictReader(open(os.path.join(RES, run, 'postproc_F3F4.csv'))))
    qd = json.load(open(os.path.join(RES, run, 'step_q_end.json')))
    t = np.array(sl['t']); p = np.array(sl['p_torso']); v = np.array(sl['v_torso'])
    om = np.array(sl['omega_torso']); hw = np.array(sl['hw'])
    prel = _rel_torso(sl)
    step_idx = np.array([int(r['step_idx']) for r in rows])
    mask = np.array([r['mask_ss'] == '1' for r in rows])
    ths = np.array([float(r['theta_s_deg']) for r in rows])
    dt = float(np.median(np.diff(t)))
    out = []
    for s in sorted(set(step_idx[step_idx >= 0])):
        sel = mask & (step_idx == s)
        if sel.sum() < 3:
            continue
        ps, vs, oms, hws = p[sel], v[sel], om[sel], hw[sel]
        prs = prel[sel]
        path_mm = float(np.sum(np.linalg.norm(np.diff(ps, axis=0), axis=1)) * 1000)
        netdisp_mm = float(np.linalg.norm(ps[-1] - ps[0]) * 1000)
        # structure-relative (crawl-removed) torso motion — reaction-relevant
        path_rel_mm = float(np.sum(np.linalg.norm(np.diff(prs, axis=0), axis=1)) * 1000)
        netdisp_rel_mm = float(np.linalg.norm(prs[-1] - prs[0]) * 1000)
        peak_speed_rel = float(np.max(np.linalg.norm(np.gradient(prs, dt, axis=0), axis=1)))
        ang_path_deg = float(np.degrees(np.sum(np.linalg.norm(np.diff(oms, axis=0), axis=1)) * dt)) \
            if False else float(np.degrees(np.sum(np.linalg.norm(oms, axis=1)) * dt))
        peak_speed = float(np.max(np.linalg.norm(vs, axis=1)))
        peak_omega = float(np.max(np.linalg.norm(oms, axis=1)))
        acc = np.gradient(vs, dt, axis=0)
        peak_acc = float(np.max(np.linalg.norm(acc, axis=1)))
        hw_peak = float(np.max(np.abs(hws)))
        theta_peak = float(np.max(np.abs(ths[sel])))
        qe = next(e for e in qd if e['step_idx'] == s)
        p_t1 = fk_torso(qe['q_end'])
        arrival_mm = float(np.linalg.norm(ps[-1] - p_t1) * 1000)
        out.append(dict(step=int(s), path_mm=path_mm, netdisp_mm=netdisp_mm,
                        path_rel_mm=path_rel_mm, netdisp_rel_mm=netdisp_rel_mm,
                        peak_speed_rel=peak_speed_rel,
                        ang_path_deg=ang_path_deg, peak_speed=peak_speed,
                        peak_omega=peak_omega, peak_acc=peak_acc, hw_peak=hw_peak,
                        theta_s_peak=theta_peak,
                        arrival_mm=arrival_mm, binding=int(s) in BINDING))
    return out


def main():
    b = per_swing(BASE)
    w = per_swing(WP)
    print(f'{"run":>9} {"step":>4} {"bind":>4} {"pathWORLD":>9} {"pathREL":>8} '
          f'{"netdspREL":>9} {"angpath°":>9} {"hw_pk":>6} {"arrival_mm":>10}')
    for lab, rows in [('baseline', b), ('two-task', w)]:
        for r in rows:
            print(f'{lab:>9} {r["step"]:>4} {"Y" if r["binding"] else "-":>4} '
                  f'{r["path_mm"]:>9.1f} {r["path_rel_mm"]:>8.1f} {r["netdisp_rel_mm"]:>9.1f} '
                  f'{r["ang_path_deg"]:>9.2f} {r["hw_peak"]:>6.2f} {r["arrival_mm"]:>10.1f}')

    # ---- correlation of h_w_peak vs each motion measure (pooled) ----
    allr = [('baseline', r) for r in b] + [('two-task', r) for r in w]
    hwv = np.array([r['hw_peak'] for _, r in allr])
    print('\n## h_w_peak vs motion measure — pooled correlation (R) across all 10 swings')
    best = None
    for key in ['path_mm', 'netdisp_mm', 'path_rel_mm', 'netdisp_rel_mm', 'peak_speed_rel',
                'ang_path_deg', 'peak_speed', 'peak_omega', 'peak_acc']:
        x = np.array([r[key] for _, r in allr])
        R = float(np.corrcoef(x, hwv)[0, 1])
        print(f'  {key:>12}: R={R:+.3f}  R²={R*R:.3f}')
        if best is None or abs(R) > abs(best[1]):
            best = (key, R)
    mkey = best[0]
    print(f'  -> strongest motion correlate: {mkey} (R={best[1]:+.3f})')

    # ---- quantify the explained fraction on the C5-limiting swings ----
    # Fit h_w = a*motion + b on the BASELINE swings; predict two-task h_w from its motion.
    xb = np.array([r[mkey] for r in b]); yb = np.array([r['hw_peak'] for r in b])
    a, c = np.polyfit(xb, yb, 1)
    print(f'\n## baseline trend  h_w = {a:.4g}*{mkey} + {c:.4g}')
    print('   two-task swings: predicted (from baseline trend) vs actual h_w')
    unexpl = {}
    for r in w:
        pred = a * r[mkey] + c
        # same-step baseline h_w (the C5 reference is per-swing peak)
        rb = next((x for x in b if x['step'] == r['step']), None)
        base_hw = rb['hw_peak'] if rb else float('nan')
        excess = r['hw_peak'] - base_hw
        resid = r['hw_peak'] - pred
        frac_unexpl = (resid / excess) if abs(excess) > 1e-6 else float('nan')
        unexpl[r['step']] = (r['hw_peak'], pred, base_hw, excess, resid, frac_unexpl)
        tag = 'BIND' if r['binding'] else 'free'
        print(f'   step {r["step"]} ({tag}): actual={r["hw_peak"]:.2f} pred={pred:.2f} '
              f'base={base_hw:.2f}  excess(tt-base)={excess:+.2f}  resid(tt-pred)={resid:+.2f}  '
              f'unexplained_frac={frac_unexpl:+.2f}')
    # global peak swing (drives C5): step with max two-task hw
    s_lim = max(w, key=lambda r: r['hw_peak'])['step']
    print(f'\n   C5-limiting swing = step {s_lim} (two-task global h_w peak)')

    # ---- the actual mechanism: h_w <-> theta_s (attitude) disturbance-budget tradeoff ----
    steps = sorted(set(r['step'] for r in b) & set(r['step'] for r in w))
    hwa = np.array([r['hw_peak'] for _, r in allr]); tha = np.array([r['theta_s_peak'] for _, r in allr])
    print('\n## "Elsewhere": h_w vs platform attitude θ_s — the disturbance-budget tradeoff')
    print(f'  pooled R(h_w_peak, θ_s_peak) over 10 swings = {np.corrcoef(hwa, tha)[0, 1]:+.3f}')
    print('  matched-swing two-task vs baseline (does MORE h_w buy LESS attitude tilt?):')
    tradeoff = 0
    for s in steps:
        rb = next(r for r in b if r['step'] == s); rw = next(r for r in w if r['step'] == s)
        td = rw['hw_peak'] > rb['hw_peak'] and rw['theta_s_peak'] < rb['theta_s_peak']
        tradeoff += td
        print(f"   step {s}: h_w {rb['hw_peak']:.2f}->{rw['hw_peak']:.2f} ({rw['hw_peak']-rb['hw_peak']:+.2f})"
              f"  θ_s {rb['theta_s_peak']:.2f}->{rw['theta_s_peak']:.2f} ({rw['theta_s_peak']-rb['theta_s_peak']:+.2f})"
              f"  {'TRADEOFF (↑h_w,↓θ_s)' if td else ''}")
    print(f"  -> tradeoff holds on {tradeoff}/{len(steps)} swings")

    # ================= FIGURE =================
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    # left: h_w vs strongest motion correlate, with baseline trend line
    xs = np.linspace(min(min(xb), min(r[mkey] for r in w)) * 0.9,
                     max(max(xb), max(r[mkey] for r in w)) * 1.05, 50)
    ax[0].plot(xs, a * xs + c, '0.6', ls='--', lw=1.0,
               label=f'baseline trend (R²={best[1]**2:.2f})')
    for r in b:
        ax[0].scatter(r[mkey], r['hw_peak'], s=110, c='0.5', marker='s',
                      edgecolors='k', zorder=3)
        ax[0].annotate(f"b{r['step']}", (r[mkey], r['hw_peak']), fontsize=7,
                       xytext=(3, 3), textcoords='offset points')
    for r in w:
        col = 'tab:red' if r['binding'] else 'tab:green'
        ax[0].scatter(r[mkey], r['hw_peak'], s=120, c=col, marker='o',
                      edgecolors='k', zorder=4)
        ax[0].annotate(f"w{r['step']}", (r[mkey], r['hw_peak']), fontsize=7,
                       xytext=(3, 3), textcoords='offset points')
    ax[0].axhline(5.0, color='r', ls=':', lw=0.8, label='±5 N·m·s hw limit')
    ax[0].set_xlabel(f'torso motion: {mkey}'); ax[0].set_ylabel('h_w peak ∞-norm [N·m·s]')
    ax[0].set_title('h_w peak vs torso motion per swing\n(grey □ baseline, ● two-task: red=binding green=free)')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    # right: integrated torso path length per step, baseline vs two-task (the "moves less" claim)
    steps = sorted(set(r['step'] for r in b) & set(r['step'] for r in w))
    wbar = 0.38
    ax[1].bar([s - wbar / 2 for s in steps], [next(r for r in b if r['step'] == s)['path_mm'] for s in steps],
              wbar, color='0.5', label='baseline')
    ax[1].bar([s + wbar / 2 for s in steps], [next(r for r in w if r['step'] == s)['path_mm'] for s in steps],
              wbar, color='tab:blue', label='two-task')
    ax[1].set_xticks(steps); ax[1].set_xlabel('step'); ax[1].set_ylabel('torso path length over swing [mm]')
    ax[1].set_title('torso motion per swing\n(baseline moves the torso MORE, not less)')
    ax[1].legend(fontsize=8)
    # right-most: the ACTUAL mechanism — h_w vs platform attitude θ_s (disturbance budget)
    for r in b:
        ax[2].scatter(r['theta_s_peak'], r['hw_peak'], s=110, c='0.5', marker='s',
                      edgecolors='k', zorder=3)
        ax[2].annotate(f"b{r['step']}", (r['theta_s_peak'], r['hw_peak']), fontsize=7,
                       xytext=(3, 3), textcoords='offset points')
    for r in w:
        col = 'tab:red' if r['binding'] else 'tab:green'
        ax[2].scatter(r['theta_s_peak'], r['hw_peak'], s=120, c=col, marker='o',
                      edgecolors='k', zorder=4)
        ax[2].annotate(f"w{r['step']}", (r['theta_s_peak'], r['hw_peak']), fontsize=7,
                       xytext=(3, 3), textcoords='offset points')
    for s in steps:
        rb = next(r for r in b if r['step'] == s); rw = next(r for r in w if r['step'] == s)
        ax[2].annotate('', xy=(rw['theta_s_peak'], rw['hw_peak']),
                       xytext=(rb['theta_s_peak'], rb['hw_peak']),
                       arrowprops=dict(arrowstyle='->', color='0.7', lw=0.8))
    ax[2].set_xlabel('platform attitude |θ_s| peak on swing [deg]')
    ax[2].set_ylabel('h_w peak ∞-norm [N·m·s]')
    ax[2].set_title('the real mechanism: h_w ↔ θ_s tradeoff\n(two-task buys low θ_s with more h_w; arrows base→two-task)')
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fp = os.path.join(OUT, '8_hw_vs_torso_motion.png')
    fig.savefig(fp, dpi=120); plt.close(fig)
    print(f'\nfigure -> {fp}')

    # persist machine-readable
    json.dump(dict(baseline=b, two_task=w, motion_correlate=mkey,
                   pooled_R={k: float(np.corrcoef([r[k] for _, r in allr], hwv)[0, 1])
                             for k in ['path_mm', 'netdisp_mm', 'path_rel_mm', 'netdisp_rel_mm',
                                       'peak_speed_rel', 'ang_path_deg', 'peak_speed', 'peak_omega', 'peak_acc']},
                   baseline_trend=dict(slope=a, intercept=c),
                   unexplained=unexpl, c5_limiting_step=s_lim),
              open(os.path.join(RES, WP, 'hw_vs_torso_motion.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
