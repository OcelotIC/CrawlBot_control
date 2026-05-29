"""Diagnostic: |Ḣ_s| (structure-disturbance moment) vs the 5 Nm budget.

Ḣ_s = Σ_j [r_Cj × f_j + τ_j]  — the exact moment the contact wrenches
apply about the structure CoM (origin in structure frame). This is the
quantity the AOCS wheels must counter (per spec §5.1, |Ḣ_s,i| ≤ τ_w_max).

The NMPC currently enforces only |L̇_com,i| ≤ τ_w_max where
L̇_com = Σ [(r_Cj - r_com) × f_j + τ_j] uses the lever from the *robot*
CoM. The two differ by r_com × m·a_com — non-negligible at the −0.35m
standoff. The exact constraint is present in code (`centroidal_nmpc.py:276`)
but disabled (`tau_struct_max = np.inf`).

Reports per SS step:
  - stance anchor + lever |r_C|
  - median / p95 / peak |Ḣ_s|_per-axis
  - fraction of ticks above the 5 Nm budget
  - comparison vs |L̇_com| (the current proxy)

Plots:
  - time series of |Ḣ_s|_per-axis and |L̇_com|_per-axis vs 5 Nm
  - divergence |Ḣ_s − L̇_com| (how much the proxy mis-states the demand)
  - stance lever |r_C| (geometric throttling baseline)

Usage:
  PYTHONPATH=. MUJOCO_GL=osmesa python3 scripts/diag_hdot_struct.py [subdir]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_subdir = sys.argv[1] if len(sys.argv) > 1 else 'diag_cooperative_arms'
LOG = os.path.join(_root, 'results', _subdir, 'sim_log.json')
OUT = os.path.join(_root, 'results', _subdir, 'hdot_struct.png')

TAU_W_MAX = 5.0  # Nm — per-axis wheel torque limit


def _anchor_pos(idx: int, arm: str) -> np.ndarray:
    """Canonical structure-frame anchor position (dx=0.8, dy=0.3, z=0.025)."""
    x = (idx - 3.5) * 0.8
    y = 0.3 if arm == 'a' else -0.3
    return np.array([x, y, 0.025])


# Stance anchor per SS step (the arm NOT swinging; even k -> swing b, odd -> swing a).
# Starting pair (2a, 2b). After step k SS: anchor advances by 1 on the swing arm.
STANCE_ANCHOR = {
    0: ('a', 2),   # swing b 2→3; stance a at 2a
    1: ('b', 3),   # swing a 2→3; stance b at 3b
    2: ('a', 3),   # swing b 3→4; stance a at 3a
    3: ('b', 4),   # swing a 3→4; stance b at 4b
    4: ('a', 4),   # swing b 4→5; stance a at 4a
}


def main():
    sl = json.load(open(LOG))
    t = np.array(sl['t'])
    ph = np.array(sl['phase'], dtype=object)
    sidx = np.array(sl['step_idx'])
    lqp = np.array(sl['lambda_qp'])             # (N, 12)
    r_com = np.array(sl['r_com'])               # (N, 3)
    dock_t = [e['t'] for e in sl['dock_events']]

    # lambda_qp convention: contact_1 = arm A, contact_2 = arm B
    # slot 0:6 = (f_a, tau_a), slot 6:12 = (f_b, tau_b)
    f_a = lqp[:, 0:3]; tau_a = lqp[:, 3:6]
    f_b = lqp[:, 6:9]; tau_b = lqp[:, 9:12]

    N = len(t)
    Hdot_s = np.zeros((N, 3))
    Ldot_com = np.zeros((N, 3))
    rC_stance = np.full((N, 3), np.nan)

    for i in range(N):
        s = int(sidx[i])
        phase = ph[i]
        if phase == 'SS' and s in STANCE_ANCHOR:
            arm, idx = STANCE_ANCHOR[s]
            rC = _anchor_pos(idx, arm)
            f, tau = (f_a[i], tau_a[i]) if arm == 'a' else (f_b[i], tau_b[i])
            Hdot_s[i] = np.cross(rC, f) + tau
            Ldot_com[i] = np.cross(rC - r_com[i], f) + tau
            rC_stance[i] = rC
        else:
            # DS: both contacts active. Use the anchor pair currently welded.
            # Inferred from how many docks have fired before t[i] (start: (2a, 2b)).
            ndocks = sum(1 for dt in dock_t if dt < t[i])
            # ndocks=0 -> initial (2a,2b); ndocks=k after k docks
            # After step k dock, anchor pair = ('a', a_idx_after_step_k), ('b', b_idx)
            # Schedule: even step advances b; odd step advances a (matches STANCE table).
            a_idx = 2 + sum(1 for k in range(ndocks) if k % 2 == 1)
            b_idx = 2 + sum(1 for k in range(ndocks) if k % 2 == 0)
            for arm, idx, f, tau in [('a', a_idx, f_a[i], tau_a[i]),
                                     ('b', b_idx, f_b[i], tau_b[i])]:
                rC = _anchor_pos(idx, arm)
                Hdot_s[i] += np.cross(rC, f) + tau
                Ldot_com[i] += np.cross(rC - r_com[i], f) + tau

    Hdot_s_axis = np.abs(Hdot_s).max(axis=1)
    Ldot_com_axis = np.abs(Ldot_com).max(axis=1)

    print('=== Group Ḣ_s: structure-disturbance moment vs τ_w_max=5 Nm ===\n')
    print('Per SS step (constraint matters most here):')
    print(f"  {'step':>4} {'stance':>6} {'|r_C|[m]':>9} "
          f"{'budget_f⊥[N]':>13} {'med|Ḣ_s|':>9} {'p95':>6} {'peak':>6} "
          f"{'%>5Nm':>7} {'|L̇|peak':>9}")
    for s in range(5):
        m = (sidx == s) & (ph == 'SS')
        if not m.any():
            continue
        h_ax = Hdot_s_axis[m]
        l_ax = Ldot_com_axis[m]
        arm, idx = STANCE_ANCHOR[s]
        rC = _anchor_pos(idx, arm)
        rC_mag = np.linalg.norm(rC)
        f_budget = TAU_W_MAX / rC_mag  # |f⊥| ≤ τ_w / |r_C|
        pct_over = float(np.mean(h_ax > TAU_W_MAX) * 100)
        print(f"  {s:>4} {arm+str(idx):>6} {rC_mag:>9.3f} "
              f"{f_budget:>13.1f} "
              f"{np.median(h_ax):>9.2f} {np.percentile(h_ax, 95):>6.2f} "
              f"{h_ax.max():>6.2f} {pct_over:>6.1f}% {l_ax.max():>9.2f}")

    print()
    print(f"  All-tick peak |Ḣ_s|_per-axis = {Hdot_s_axis.max():.2f} Nm "
          f"(vs τ_w_max=5)")
    print(f"  All-tick peak |L̇_com|_per-axis = {Ldot_com_axis.max():.2f} Nm "
          f"(current NMPC proxy)")
    div = np.linalg.norm(Hdot_s - Ldot_com, axis=1)
    print(f"  Max |Ḣ_s − L̇_com| over run = {div.max():.2f} Nm "
          f"(divergence = r_com × m·a_com term)")

    # --- figure ---
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ss = ph == 'SS'

    def deco(a):
        ymin, ymax = a.get_ylim()
        a.fill_between(t, ymin, ymax, where=ss, color='orange', alpha=0.06)
        for dt in dock_t:
            a.axvline(dt, color='k', ls=':', lw=0.6, alpha=0.5)
        a.grid(alpha=0.3)

    ax[0].plot(t, Hdot_s_axis, 'C3', lw=1.0,
               label='|Ḣ_s|_per-axis (exact wheel demand)')
    ax[0].plot(t, Ldot_com_axis, 'C0', lw=0.9, alpha=0.7,
               label='|L̇_com|_per-axis (current NMPC proxy)')
    ax[0].axhline(TAU_W_MAX, color='r', ls='--', lw=0.9, label='τ_w_max=5 Nm')
    ax[0].set_ylabel('moment [Nm]')
    ax[0].legend(fontsize=8, loc='upper right')
    ax[0].set_title('Structure-disturbance moment vs wheel-torque budget')
    ax[0].set_yscale('symlog', linthresh=1.0)

    ax[1].plot(t, div, 'C2', lw=1.0)
    ax[1].set_ylabel('|Ḣ_s − L̇_com| [Nm]')
    ax[1].set_title('Divergence — how much the L̇_com proxy mis-states wheel demand')

    rC_norm = np.linalg.norm(rC_stance, axis=1)
    ax[2].plot(t, rC_norm, 'C4', lw=1.2)
    ax[2].set_ylabel('|r_C| stance [m]')
    ax[2].set_title('Stance contact lever from structure CoM')
    ax[2].set_xlabel('t [s]')

    for a in ax:
        deco(a)
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    plt.close(fig)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
