#!/usr/bin/env python3
"""Ablation figures (§VII) — envelope rate-cap ON (C) vs OFF (U).

Produces 4 comparison PDFs from the two committed/exported tidy CSVs:
  C = results/j2_canonical_revalidation/runfix_traversal.csv   (envelope ON, canonical @5ab2c91)
  U = results/j2_ablation_envelope/runU_rateoff_traversal.csv  (rate cap tau_w_max=1e6, everything else canonical)

  figA_hw.pdf       storage momentum h_w (3 axes, +/-5 box)      — both bounded (storage box still ON in U)
  figB_Hdot_s.pdf   PLANNED wheel-rate demand Hdot_s (3 axes)    — KEY: U exceeds the 5 Nm cap; C pinned at 5
  figC_theta_s.pdf  attitude error theta_s (3 axes + norm)       — secondary axis
  figD_tauw.pdf     REALIZED wheel torque tau_w (3 axes, clamp)  — saturation fraction

Frozen convention (implemented INLINE — vispa_plot_style.py is not in the tree):
  DS band = blue, SS band = orange; x = red, y = green, z = blue; Liberation Sans; pdf.fonttype = 42.
Each figure is 2 rows (C top, U bottom) with per-run DS/SS bands and shared y-limits (the two runs
have different durations, so x-axes are independent — NOT shared).

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/fig_ablation_rateoff.py
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42          # TrueType (editable text), frozen convention
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'Liberation Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

C_CSV = 'results/j2_canonical_revalidation/runfix_traversal.csv'
U_CSV = 'results/j2_ablation_envelope/runU_rateoff_traversal.csv'
OUT = 'results/j2_ablation_envelope/figs'
os.makedirs(OUT, exist_ok=True)

TAU_W_MAX = 5.0        # rate cap / AOCS clamp (Nm)  — spec §5.1
H_MAX = 5.0            # storage box (Nms)           — spec §4.6
AXC = {'x': '#d62728', 'y': '#2ca02c', 'z': '#1f77b4'}   # x=red, y=green, z=blue
DS_BAND = '#1f77b4'    # blue
SS_BAND = '#ff7f0e'    # orange


def load(path):
    r = list(csv.DictReader(open(path)))
    def col(n, f=float):
        return np.array([f(x[n]) if x[n] != '' else np.nan for x in r])
    d = dict(
        t=col('t_s'), phase=np.array([x['phase'] for x in r]),
        src=np.array([x['Hdot_s_source'] for x in r]),
        step=col('step_index', int),
        H=np.stack([col('Hdot_s_x_Nm'), col('Hdot_s_y_Nm'), col('Hdot_s_z_Nm')], 1),
        hw=np.stack([col('hw_x_Nms'), col('hw_y_Nms'), col('hw_z_Nms')], 1),
        th=np.stack([col('theta_s_x_deg'), col('theta_s_y_deg'), col('theta_s_z_deg')], 1),
        tw=np.stack([col('tauw_x_Nm'), col('tauw_y_Nm'), col('tauw_z_Nm')], 1),
    )
    return d


def phase_segments(t, phase):
    """Contiguous [t0, t1, class] runs; class in {'DS','SS'} (DS_* collapsed to DS)."""
    cls = np.where(np.char.startswith(phase.astype(str), 'DS'), 'DS', 'SS')
    segs, i = [], 0
    n = len(t)
    while i < n:
        j = i
        while j + 1 < n and cls[j + 1] == cls[i]:
            j += 1
        t1 = t[j + 1] if j + 1 < n else t[j] + (t[j] - t[j - 1] if j > 0 else 0.09)
        segs.append((t[i], t1, cls[i]))
        i = j + 1
    return segs


def shade(ax, segs):
    for t0, t1, c in segs:
        ax.axvspan(t0, t1, color=DS_BAND if c == 'DS' else SS_BAND, alpha=0.07, lw=0, zorder=0)


def axis_lines(ax, d, sig, mask=None):
    for k, a in enumerate('xyz'):
        y = d[sig][:, k].copy()
        if mask is not None:
            y = np.where(mask, y, np.nan)
        ax.plot(d['t'], y, color=AXC[a], lw=1.1, label=a, zorder=3)


BAND_LEGEND = [Patch(fc=DS_BAND, alpha=0.18, label='DS'),
               Patch(fc=SS_BAND, alpha=0.18, label='SS')]
AXIS_LEGEND = [Line2D([0], [0], color=AXC['x'], lw=1.6, label='x'),
               Line2D([0], [0], color=AXC['y'], lw=1.6, label='y'),
               Line2D([0], [0], color=AXC['z'], lw=1.6, label='z')]


def two_row(sig, ylabel, title, fname, hlines=(), mask_src=None, ylim=None, annot=None):
    C, U = load(C_CSV), load(U_CSV)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0))
    for ax, d, tag in [(axes[0], C, 'C  (envelope rate-cap ON)'),
                       (axes[1], U, 'U  (rate cap OFF, tau_w_max=1e6)')]:
        shade(ax, phase_segments(d['t'], d['phase']))
        m = (d['src'] == mask_src) if mask_src else None
        axis_lines(ax, d, sig, mask=m)
        for h in hlines:
            ax.axhline(h, color='0.35', ls=':', lw=0.9, zorder=2)
        ax.set_ylabel(ylabel)
        ax.set_title(tag, fontsize=9, loc='left')
        ax.grid(True, alpha=0.18, lw=0.5)
        if ylim:
            ax.set_ylim(*ylim)
        ax.margins(x=0.005)
        if annot:
            annot(ax, d)
    axes[1].set_xlabel('time  [s]')
    axes[0].legend(handles=AXIS_LEGEND + BAND_LEGEND, ncol=5, fontsize=7.5,
                   loc='upper right', framealpha=0.9)
    fig.suptitle(title, fontsize=11, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(OUT, fname)
    fig.savefig(p)
    prev = os.environ.get('PREVIEW_DIR')
    if prev:
        pp = os.path.join(prev, fname.replace('.pdf', '.png'))
        fig.savefig(pp, dpi=110)
        print(f'  preview {pp}')
    plt.close(fig)
    print(f'  wrote {p}')


# ---- Fig A: storage momentum h_w (both bounded — storage box ON in both) ----
two_row('hw', r'$h_w$  [Nms]',
        'Fig A — reaction-wheel stored momentum $h_w$  (storage box $\\pm5$ Nms ON in both)',
        'figA_hw.pdf', hlines=(+H_MAX, -H_MAX), ylim=(-5.6, 5.6))

# ---- Fig B: PLANNED wheel-rate demand Hdot_s (KEY) ----
def _annot_B(ax, d):
    m = d['src'] == 'planned'
    if m.any():
        inf = np.nanmax(np.abs(d['H'][m]), axis=1)
        pk = np.nanmax(inf); over = int((inf > TAU_W_MAX + 1e-6).sum()); tot = int(m.sum())
        ax.text(0.012, 0.04, f'planned $\\|\\dot H_s\\|_\\infty$ peak = {pk:.2f} Nm   '
                f'SS ticks $>5$: {over}/{tot} ({100*over/tot:.1f}%)',
                transform=ax.transAxes, fontsize=7.6, va='bottom',
                bbox=dict(fc='white', ec='0.7', alpha=0.85, pad=1.5))
two_row('H', r'planned $\dot H_s$  [Nm]',
        'Fig B — PLANNED reaction-wheel rate demand $\\dot H_s$  (rate cap $\\pm5$ Nm)',
        'figB_Hdot_s.pdf', hlines=(+TAU_W_MAX, -TAU_W_MAX),
        mask_src='planned', ylim=(-7.0, 7.0), annot=_annot_B)

# ---- Fig C: attitude error theta_s (secondary) ----
def _annot_C(ax, d):
    thn = np.linalg.norm(d['th'], axis=1)
    ax.plot(d['t'], thn, color='0.25', lw=1.4, ls='--', zorder=4, label='|θ|')
    ax.text(0.012, 0.92, f'$\\|\\theta_s\\|$ peak = {np.nanmax(thn):.3f}°   final = {thn[-1]:.3f}°',
            transform=ax.transAxes, fontsize=7.6, va='top',
            bbox=dict(fc='white', ec='0.7', alpha=0.85, pad=1.5))
two_row('th', r'$\theta_s$  [deg]',
        'Fig C — structure attitude error $\\theta_s$  (secondary metric)',
        'figC_theta_s.pdf', ylim=(-0.62, 0.62), annot=_annot_C)

# ---- Fig D: REALIZED wheel torque tau_w (saturation at the +/-5 clamp) ----
def _annot_D(ax, d):
    inf = np.nanmax(np.abs(d['tw']), axis=1)
    sat = inf >= TAU_W_MAX - 1e-3
    ax.text(0.012, 0.04, f'realized $\\|\\tau_w\\|_\\infty$ peak = {np.nanmax(inf):.2f} Nm   '
            f'saturated {100*sat.mean():.1f}% of ticks ({int(sat.sum())}/{len(sat)})',
            transform=ax.transAxes, fontsize=7.6, va='bottom',
            bbox=dict(fc='white', ec='0.7', alpha=0.85, pad=1.5))
two_row('tw', r'realized $\tau_w$  [Nm]',
        'Fig D — REALIZED reaction-wheel torque $\\tau_w$  (AOCS/actuator clamp $\\pm5$ Nm)',
        'figD_tauw.pdf', hlines=(+TAU_W_MAX, -TAU_W_MAX), ylim=(-5.8, 5.8), annot=_annot_D)

print('done ->', OUT)
