#!/usr/bin/env python3
"""J1 Lemma-2 SS figure: Ḣ_R/Os vs plant weld wrench about O_s, and the m_i term."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), '..')
INSTR = os.path.join(ROOT, 'results', 'lemma2_ss')
OUT = os.path.join(INSTR, 'lemma2_plots')
os.makedirs(OUT, exist_ok=True)


def segments(t, gap):
    segs, i = [], 0
    while i < len(t):
        j = i + 1
        while j < len(t) and (t[j] - t[j - 1]) < gap:
            j += 1
        segs.append((i, j - 1)); i = j
    return segs


def main():
    sl = json.load(open(os.path.join(INSTR, 'sim_log.json')))
    t = np.array(sl['t_l2_hf']); H = np.array(sl['HR_Os_hf'])
    wf = np.array(sl['weld_f_hf']); wm = np.array(sl['weld_m_hf'])
    pg = np.array(sl['p_grip_hf']); rOs = np.array(sl['r_Os_hf'])
    lever = pg - rOs
    tau_full = wm + np.cross(lever, wf)
    tau_noM = np.cross(lever, wf)
    dt = np.median(np.diff(t[:50]))
    segs = [s for s in segments(t, 5 * dt) if s[1] - s[0] >= 8]
    segs = sorted(segs, key=lambda s: -np.nanmean(np.linalg.norm(tau_full[s[0]:s[1]+1], axis=1)))
    a, b = segs[0]; sl_ = slice(a, b + 1); ts = t[sl_]
    Hdot = np.gradient(H[sl_], ts, axis=0)
    g = slice(2, len(ts) - 2)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    cols = ['C0', 'C1', 'C2']
    for i, ax in enumerate('xyz'):
        ax0.plot(ts[g], Hdot[g][:, i], '-', color=cols[i], lw=1.6, label=f'Ḣ_R/Os,{ax}')
        ax0.plot(ts[g], tau_full[sl_][g][:, i], '--', color=cols[i], lw=1.1,
                 label=f'w_weld/Os,{ax}')
    ax0.set_xlabel('time [s]'); ax0.set_ylabel('[N·m]')
    ax0.set_title(f'Lemma 2 in SS (swing t={ts[0]:.1f}–{ts[-1]:.1f}s): '
                  f'Ḣ_R/Os (solid) = plant weld wrench about O_s (dashed)\n'
                  f'ε/‖Ḣ‖ ≈ 1%  (MuJoCo-direct, inertial, m_i included)')
    ax0.legend(fontsize=7, ncol=3, loc='upper right'); ax0.grid(alpha=0.3)

    # residual norm: with vs without m_i, across the 3 highest-rate swings
    off = 0
    for k, (sa, sb) in enumerate(segs[:3]):
        s_ = slice(sa, sb + 1); tt = t[s_]
        Hd = np.gradient(H[s_], tt, axis=0)
        gg = slice(2, len(tt) - 2)
        Hn = np.linalg.norm(Hd[gg], axis=1)
        e_full = np.linalg.norm(Hd[gg] - tau_full[s_][gg], axis=1) / (Hn + 1e-12)
        e_noM = np.linalg.norm(Hd[gg] - tau_noM[s_][gg], axis=1) / (Hn + 1e-12)
        x = off + np.arange(len(e_full))
        ax1.plot(x, e_full, '-', color='C2', lw=1.0,
                 label='with m_i' if k == 0 else None)
        ax1.plot(x, e_noM, '-', color='C3', lw=1.0, alpha=0.7,
                 label='without m_i' if k == 0 else None)
        off += len(e_full) + 10
    ax1.axhline(1.0, color='k', ls=':', lw=0.6)
    ax1.set_ylabel('ε / ‖Ḣ_R/Os‖'); ax1.set_xlabel('SS samples (3 highest-rate swings, concatenated)')
    ax1.set_title('The 6-DOF weld moment m_i is required\n'
                  'with m_i: ε≈1%   ;   without m_i: ε≈100%')
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3); ax1.set_ylim(0, 2.2)

    fig.tight_layout()
    p = os.path.join(OUT, 'lemma2_reduction_identity.png')
    fig.savefig(p, dpi=130); print('wrote', p)


if __name__ == '__main__':
    main()
