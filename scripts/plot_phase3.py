#!/usr/bin/env python3
"""Phase 3 plot set (memo §6 + brief): 7 paper-style figures, baseline overlaid
grey/dashed, candidate (two-task working point) in colour, SS phases shaded,
constraint limits as dashed horizontals. Every gate criterion readable off >=1 plot.
Reads phase3_wp + baseline F3F4 CSV + sim_log + gate_phase3_results.json. Read-only.
"""
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results')
BASE = 'ssmom_phase1_baseline_main_dcda974'
WP = 'phase3_wp'
OUT = os.path.join(RES, WP, 'phase3_plots')
TAU_W_MAX = 5.0
HW_MAX = 5.0
GATE_MM = 5.0
GREY = '0.55'


def load_csv(d):
    rows = list(csv.DictReader(open(os.path.join(RES, d, 'postproc_F3F4.csv'))))
    cols = {}
    for k in rows[0]:
        try:
            cols[k] = np.array([float(r[k]) if r[k] not in ('', 'nan') else np.nan
                                for r in rows])
        except ValueError:
            cols[k] = np.array([r[k] for r in rows])
    return cols


def ss_intervals(c):
    """Contiguous [t0,t1] intervals where mask_ss==1, with step_idx."""
    m = c['mask_ss'] > 0.5
    t = c['t']
    iv = []
    i = 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            iv.append((t[i], t[j - 1], int(c['step_idx'][i])))
            i = j
        else:
            i += 1
    return iv


def shade_ss(ax, iv, label_once=True):
    for k, (t0, t1, si) in enumerate(iv):
        ax.axvspan(t0, t1, color='0.85', alpha=0.5, zorder=0,
                   label='SS phase' if (label_once and k == 0) else None)


def main():
    os.makedirs(OUT, exist_ok=True)
    cw = load_csv(WP)
    cb = load_csv(BASE)
    slw = json.load(open(os.path.join(RES, WP, 'sim_log.json')))
    slb = json.load(open(os.path.join(RES, BASE, 'sim_log.json')))
    gate = json.load(open(os.path.join(RES, WP, 'gate_phase3_results.json')))
    iv = ss_intervals(cw)
    docks_t = [cw['t'][i] for i in range(1, len(cw['t']))
               if cw['mask_ss'][i - 1] > 0.5 and cw['mask_ss'][i] < 0.5]

    # ---- 1. torso tracking: orientation geodesic + position per axis ----
    fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    shade_ss(ax[0], iv)
    ax[0].plot(cb['t'], cb['e_ori_torso_deg'], GREY, ls='--', lw=1.1, label='baseline')
    ax[0].plot(cw['t'], cw['e_ori_torso_deg'], 'b-', lw=1.1, label='two-task WP')
    ax[0].set_ylabel('torso ori err [deg]')
    ax[0].set_title('Fig.1 — torso TRACKING (orientation geodesic + position per axis), SS shaded')
    ax[0].legend(fontsize=8)
    for a, ax_i in zip('xyz', ax[1:]):
        shade_ss(ax_i, iv)
        ax_i.plot(cb['t'], cb[f'e_pos_torso_{a}'] * 1000, GREY, ls='--', lw=1.1)
        ax_i.plot(cw['t'], cw[f'e_pos_torso_{a}'] * 1000, 'b-', lw=1.1)
        ax_i.set_ylabel(f'torso pos err {a} [mm]')
    ax[-1].set_xlabel('t [s]')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, '1_torso_tracking.png'), dpi=120); plt.close(fig)

    # ---- 2. swing-EE orientation + distance to anchor ----
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    shade_ss(ax[0], iv)
    ax[0].plot(cb['t'], cb['d_ee_to_target'] * 1000, GREY, ls='--', lw=1.1, label='baseline')
    ax[0].plot(cw['t'], cw['d_ee_to_target'] * 1000, 'b-', lw=1.1, label='two-task WP')
    ax[0].axhline(GATE_MM, color='r', ls=':', lw=1.0, label='5 mm dock gate')
    ax[0].set_ylabel('swing-EE dist to anchor [mm]'); ax[0].set_ylim(0, 60)
    ax[0].set_title('Fig.2 — swing-EE distance + orientation to anchor (docking approach)')
    ax[0].legend(fontsize=8)
    shade_ss(ax[1], iv)
    ax[1].plot(cb['t'], cb['e_ori_ee_deg'], GREY, ls='--', lw=1.1)
    ax[1].plot(cw['t'], cw['e_ori_ee_deg'], 'b-', lw=1.1)
    ax[1].set_ylabel('swing-EE ori err [deg]'); ax[1].set_xlabel('t [s]'); ax[1].set_ylim(0, 5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, '2_swing_ee.png'), dpi=120); plt.close(fig)

    # ---- 3. platform attitude |theta_s|, docks marked ----
    fig, ax = plt.subplots(figsize=(11, 4.6))
    shade_ss(ax, iv)
    ax.plot(cb['t'], np.abs(cb['theta_s_deg']), GREY, ls='--', lw=1.2, label='baseline')
    ax.plot(cw['t'], np.abs(cw['theta_s_deg']), 'b-', lw=1.2, label='two-task WP')
    for k, td in enumerate(docks_t):
        ax.axvline(td, color='g', ls=':', lw=0.9, label='dock' if k == 0 else None)
    ax.axhline(1.9, color='r', ls=':', lw=1.0, label='~1.9° gate (peak)')
    ax.set_ylabel('|θ_s| platform attitude [deg]'); ax.set_xlabel('t [s]')
    ax.set_title('Fig.3 — platform attitude |θ_s| over traversal (dock instants marked)')
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, '3_attitude.png'), dpi=120); plt.close(fig)

    # ---- 4. Hdot_s + tau_w per axis, +-5, SS shaded, tau_w at 100Hz ----
    t_hf = np.array(slw.get('t_ss_hifreq', []))
    tw_hf = np.array(slw.get('tau_w_ss_hifreq', [])).reshape(-1, 3) if len(t_hf) else None
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    shade_ss(ax[0], iv)
    for a, col in zip('xyz', ('tab:red', 'tab:green', 'tab:blue')):
        ax[0].plot(cb['t'], cb[f'Hdot_s_{a}'], GREY, ls='--', lw=0.9)
        ax[0].plot(cw['t'], cw[f'Hdot_s_{a}'], color=col, lw=1.0, label=f'Ḣ_s {a}')
    ax[0].axhline(TAU_W_MAX, color='k', ls=':', lw=1.0); ax[0].axhline(-TAU_W_MAX, color='k', ls=':', lw=1.0)
    ax[0].set_ylabel('Ḣ_s [N·m]'); ax[0].set_ylim(-7, 7); ax[0].legend(fontsize=7, ncol=3)
    ax[0].set_title('Fig.4 — Ḣ_s (top) & commanded τ_w @100Hz (bottom), per axis, ±5 N·m (baseline grey)')
    shade_ss(ax[1], iv)
    for a, col, j in zip('xyz', ('tab:red', 'tab:green', 'tab:blue'), range(3)):
        ax[1].plot(cb['t'], cb[f'tau_w_{a}'], GREY, ls='--', lw=0.9)
        if tw_hf is not None:
            ax[1].plot(t_hf, tw_hf[:, j], color=col, lw=0.7, label=f'τ_w {a} (100Hz)')
    ax[1].axhline(TAU_W_MAX, color='k', ls=':', lw=1.0); ax[1].axhline(-TAU_W_MAX, color='k', ls=':', lw=1.0)
    ax[1].set_ylabel('τ_w [N·m]'); ax[1].set_ylim(-6.5, 6.5); ax[1].set_xlabel('t [s]'); ax[1].legend(fontsize=7, ncol=3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, '4_Hdot_tauw.png'), dpi=120); plt.close(fig)

    # ---- 5. h_w per axis (inf-norm) vs +-5 N·m·s ----
    tb = np.array(slb['t']); tw_b = np.array(slb['hw'])
    tt = np.array(slw['t']); hw_w = np.array(slw['hw'])
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for j, (a, col) in enumerate(zip('xyz', ('tab:red', 'tab:green', 'tab:blue'))):
        ax.plot(tb, tw_b[:, j], GREY, ls='--', lw=0.9)
        ax.plot(tt, hw_w[:, j], color=col, lw=1.0, label=f'h_w {a}')
    ax.axhline(HW_MAX, color='r', ls=':', lw=1.0, label='±5 N·m·s')
    ax.axhline(-HW_MAX, color='r', ls=':', lw=1.0)
    ax.set_ylabel('stored wheel momentum h_w [N·m·s]'); ax.set_xlabel('t [s]')
    ax.set_title('Fig.5 — h_w per axis vs ±5 N·m·s (baseline grey/dashed)')
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, '5_hw.png'), dpi=120); plt.close(fig)

    # ---- 6. per-step docking-margin bar chart ----
    c1 = gate['c1']
    steps = list(range(len(c1['wp_d'])))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    w = 0.38
    ax.bar([s - w / 2 for s in steps], c1['base_d'], w, color=GREY, label='baseline d')
    ax.bar([s + w / 2 for s in steps], c1['wp_d'], w, color='tab:blue', label='two-task WP d')
    ax.axhline(GATE_MM, color='r', ls='--', lw=1.2, label='5 mm dock gate')
    ax.set_xticks(steps); ax.set_xlabel('step'); ax.set_ylabel('dock distance d [mm]')
    ax.set_title('Fig.6 — per-step docking distance (Phase-3 vs baseline vs 5 mm gate)')
    for s in steps:
        ax.text(s + w / 2, c1['wp_d'][s] + 0.05, f"{c1['wp_d'][s]:.2f}", ha='center', fontsize=7)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, '6_dock_margins.png'), dpi=120); plt.close(fig)

    # ---- 7. two-regime torso arrival ----
    reg = gate['two_regime']
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for r in reg:
        col = 'tab:red' if r['binding'] else 'tab:green'
        ax.bar(r['step'], r['arrival_pct'], 0.6, color=col,
               edgecolor='k', linewidth=0.5)
        ax.text(r['step'], r['arrival_pct'] + 0.15,
                f"{r['arrival_pct']:.1f}%\n(bind {r['bind_frac']:.0f}%,\nτ_w-sat {r['tau_w_sat_100hz']:.0f}%)",
                ha='center', fontsize=7)
    nb = np.mean([r['arrival_pct'] for r in reg if not r['binding']])
    bd = np.mean([r['arrival_pct'] for r in reg if r['binding']])
    ax.axhline(nb, color='tab:green', ls=':', lw=1.2, label=f'NON-binding mean {nb:.1f}%')
    ax.axhline(bd, color='tab:red', ls=':', lw=1.2, label=f'BINDING mean {bd:.1f}%')
    from matplotlib.patches import Patch
    leg = [Patch(fc='tab:green', label='non-binding swing'),
           Patch(fc='tab:red', label='envelope-binding swing')]
    ax.legend(handles=leg + ax.get_legend_handles_labels()[0], fontsize=8)
    ax.set_xlabel('step'); ax.set_ylabel('torso arrival vs p_t1 [% of step travel]')
    ax.set_title('Fig.7 — two-regime torso arrival (binding vs non-binding; characterisation, NOT a gate)')
    ax.set_xticks([r['step'] for r in reg])
    fig.tight_layout(); fig.savefig(os.path.join(OUT, '7_two_regime_arrival.png'), dpi=120); plt.close(fig)

    print(f'7 plots -> {OUT}/')


if __name__ == '__main__':
    main()
