"""SS centroidal-momentum task — overlay plot set (memo §6).

Generates the four paper-style figures for a candidate run overlaid on a
baseline reference, per the §6 overlay convention:
  - baseline (reference) in grey / dashed; candidate in colour; SAME axes
    (superposed, never side-by-side-only);
  - SS phases shaded; per-axis curves labelled x/y/z;
  - constraint limits drawn as dashed horizontals where applicable.

Each run is read from its committed/derived ``postproc_F3F4.csv`` (produced by
Misc/scripts/postprocess_results_figs.py). Figures are written as PNG into --out-dir.

Phase-1.1 default invocation (main-HEAD baseline vs the 5bca42c reference):

    PYTHONPATH=. python3 Misc/scripts/plot_ssmom_phase1.py \
        --ref-dir  Misc/runs/diag_cooperative_arms_legacy_pid_numerical \
        --ref-label 5bca42c \
        --cand-dir Misc/runs/ssmom_phase1_baseline_main_dcda974 \
        --cand-label "main HEAD (dcda974)"

Reusable for Phase 2/3 by pointing --cand-dir at a variant run.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TAU_W_MAX = 5.0  # ±N·m wheel-torque / ‖Ḣ_s‖ envelope

_FLOAT_COLS = [
    't', 'theta_s_deg', 'Hdot_s_x', 'Hdot_s_y', 'Hdot_s_z',
    'tau_w_x', 'tau_w_y', 'tau_w_z', 'e_ori_torso_deg',
    'e_pos_torso_x', 'e_pos_torso_y', 'e_pos_torso_z',
    'd_ee_to_target', 'e_ori_ee_deg', 'mask_ss',
]

AXC = {'x': 'tab:red', 'y': 'tab:green', 'z': 'tab:blue'}


def load(run_dir: str) -> dict:
    path = Path(run_dir) / 'postproc_F3F4.csv'
    if not path.exists():
        raise FileNotFoundError(f'missing postproc CSV: {path}')
    with open(path) as f:
        rows = list(csv.DictReader(f))
    data: dict = {}
    for name in _FLOAT_COLS:
        vals = []
        for r in rows:
            v = r.get(name, '')
            vals.append(float(v) if v not in ('', 'nan', 'NaN') else np.nan)
        data[name] = np.asarray(vals)
    data['phase'] = np.asarray([r['phase'] for r in rows], dtype=object)
    return data


def shade_ss(ax, t, mask_ss, label_once=True):
    """Shade contiguous SS spans (mask_ss>0.5)."""
    m = mask_ss > 0.5
    labelled = not label_once
    i = 0
    n = len(t)
    while i < n:
        if m[i]:
            j = i
            while j + 1 < n and m[j + 1]:
                j += 1
            ax.axvspan(t[i], t[j], color='0.88', lw=0, zorder=0,
                       label=(None if labelled else 'SS phase'))
            labelled = True
            i = j + 1
        else:
            i += 1


def _ref_cand_lines(ax, t_ref, y_ref, t_cand, y_cand, ref_label, cand_label,
                    color='tab:orange'):
    ax.plot(t_ref, y_ref, color='0.45', ls='--', lw=1.6, zorder=3,
            label=f'{ref_label}')
    ax.plot(t_cand, y_cand, color=color, ls='-', lw=1.0, zorder=4,
            label=f'{cand_label}')


def _per_axis(ax, ref, cand, prefix, ref_label, cand_label):
    for a in ('x', 'y', 'z'):
        col = f'{prefix}_{a}'
        ax.plot(ref['t'], ref[col], color='0.55', ls='--', lw=1.6, zorder=3,
                label=(f'{ref_label}' if a == 'x' else None))
        ax.plot(cand['t'], cand[col], color=AXC[a], ls='-', lw=1.0, zorder=4,
                label=f'{cand_label} {a}')


def fig_torso(ref, cand, ref_label, cand_label, out: Path):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    shade_ss(a1, ref['t'], ref['mask_ss'])
    _ref_cand_lines(a1, ref['t'], ref['e_ori_torso_deg'],
                    cand['t'], cand['e_ori_torso_deg'], ref_label, cand_label)
    a1.set_ylabel('torso ori err [deg]\n(SS: vs planner; DS: hold-dev)')
    a1.set_title('Figure 1 — Torso tracking (orientation + position)')
    a1.legend(loc='upper left', fontsize=8, ncol=3)
    a1.grid(alpha=0.3)

    shade_ss(a2, ref['t'], ref['mask_ss'], label_once=False)
    for a in ('x', 'y', 'z'):
        col = f'e_pos_torso_{a}'
        a2.plot(ref['t'], ref[col] * 1e3, color='0.55', ls='--', lw=1.6,
                zorder=3, label=(ref_label if a == 'x' else None))
        a2.plot(cand['t'], cand[col] * 1e3, color=AXC[a], ls='-', lw=1.0,
                zorder=4, label=f'{cand_label} {a}')
    a2.set_ylabel('torso pos err [mm]')
    a2.set_xlabel('t [s]')
    a2.legend(loc='upper left', fontsize=8, ncol=4)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'f1_torso_tracking.png', dpi=130)
    plt.close(fig)


def fig_swing_ee(ref, cand, ref_label, cand_label, out: Path):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    shade_ss(a1, ref['t'], ref['mask_ss'])
    _ref_cand_lines(a1, ref['t'], ref['d_ee_to_target'] * 1e3,
                    cand['t'], cand['d_ee_to_target'] * 1e3,
                    ref_label, cand_label)
    a1.axhline(5.0, color='k', ls=':', lw=1.0, label='5 mm dock gate')
    a1.set_ylabel('swing-EE distance\nto target [mm]')
    a1.set_yscale('log')
    a1.set_title('Figure 2 — Swing end-effector docking (distance + orientation)')
    a1.legend(loc='upper right', fontsize=8)
    a1.grid(alpha=0.3, which='both')

    shade_ss(a2, ref['t'], ref['mask_ss'], label_once=False)
    _ref_cand_lines(a2, ref['t'], ref['e_ori_ee_deg'],
                    cand['t'], cand['e_ori_ee_deg'], ref_label, cand_label)
    a2.set_ylabel('swing-EE ori err [deg]')
    a2.set_xlabel('t [s]')
    a2.legend(loc='upper right', fontsize=8)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'f2_swing_ee.png', dpi=130)
    plt.close(fig)


def fig_attitude(ref, cand, ref_label, cand_label, out: Path):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    shade_ss(ax, ref['t'], ref['mask_ss'])
    _ref_cand_lines(ax, ref['t'], ref['theta_s_deg'],
                    cand['t'], cand['theta_s_deg'], ref_label, cand_label)
    ax.set_ylabel(r'platform attitude $|\theta_s|$ [deg]')
    ax.set_xlabel('t [s]')
    ax.set_title(r'Figure 3 — Platform attitude $|\theta_s|$ vs initial')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'f3_attitude.png', dpi=130)
    plt.close(fig)


def fig_hdot_tau(ref, cand, ref_label, cand_label, out: Path):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    shade_ss(a1, ref['t'], ref['mask_ss'])
    _per_axis(a1, ref, cand, 'Hdot_s', ref_label, cand_label)
    for s in (+TAU_W_MAX, -TAU_W_MAX):
        a1.axhline(s, color='k', ls=':', lw=1.0)
    a1.set_ylabel(r'$\dot H_s$ per axis [N·m]')
    a1.set_title(r'Figure 4 — NMPC-planned $\dot H_s$ and AOCS $\tau_w$ '
                 r'(per axis, $\pm5$ N·m envelope)')
    a1.legend(loc='upper right', fontsize=7, ncol=4)
    a1.grid(alpha=0.3)

    shade_ss(a2, ref['t'], ref['mask_ss'], label_once=False)
    _per_axis(a2, ref, cand, 'tau_w', ref_label, cand_label)
    for s in (+TAU_W_MAX, -TAU_W_MAX):
        a2.axhline(s, color='k', ls=':', lw=1.0)
    a2.set_ylabel(r'$\tau_w$ per axis [N·m]')
    a2.set_xlabel('t [s]')
    a2.legend(loc='upper right', fontsize=7, ncol=4)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'f4_hdot_tau.png', dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref-dir', required=True)
    ap.add_argument('--cand-dir', required=True)
    ap.add_argument('--ref-label', default='baseline')
    ap.add_argument('--cand-label', default='candidate')
    ap.add_argument('--out-dir', default=None,
                    help='default: <cand-dir>/phase1_plots')
    args = ap.parse_args()

    ref = load(args.ref_dir)
    cand = load(args.cand_dir)
    out = Path(args.out_dir) if args.out_dir else Path(args.cand_dir) / 'phase1_plots'
    out.mkdir(parents=True, exist_ok=True)

    fig_torso(ref, cand, args.ref_label, args.cand_label, out)
    fig_swing_ee(ref, cand, args.ref_label, args.cand_label, out)
    fig_attitude(ref, cand, args.ref_label, args.cand_label, out)
    fig_hdot_tau(ref, cand, args.ref_label, args.cand_label, out)
    print(f'[plots] wrote 4 figures to {out}')


if __name__ == '__main__':
    main()
