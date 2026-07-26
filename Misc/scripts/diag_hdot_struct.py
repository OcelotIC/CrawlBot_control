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
  PYTHONPATH=. MUJOCO_GL=osmesa python3 Misc/scripts/diag_hdot_struct.py [subdir]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_subdir = sys.argv[1] if len(sys.argv) > 1 else 'diag_cooperative_arms'
LOG = os.path.join(_root, 'results', _subdir, 'sim_log.json')
OUT = os.path.join(_root, 'results', _subdir, 'hdot_struct.png')

TAU_W_MAX = 5.0  # Nm — per-axis wheel torque limit


def _anchor_pos(idx: int, arm: str) -> np.ndarray:
    """Canonical structure-frame anchor position (dx=0.8, dy=0.3, z=0.025).

    idx is the **0-indexed array index** matching `anchors_a[idx]` in
    `ContactScheduler` — NOT the MJCF anchor name (which is 1-indexed).
    MJCF site `anchor_{idx+1}{arm}` sits at x = ((idx+1) - 3.5) * 0.8
    = (idx - 2.5) * 0.8.
    """
    x = (idx - 2.5) * 0.8
    y = 0.3 if arm == 'a' else -0.3
    return np.array([x, y, 0.025])


# Stance anchor per SS step. stance index = anchors_a/b[k] index that has
# NOT swung yet at the start of this step. Starting (start_a=2, start_b=2)
# per diag_cooperative_arms.py:291. Even step => swing b (stance A);
# odd step => swing a (stance B). Indices are 0-based array indices.
STANCE_ANCHOR = {
    0: ('a', 2),   # step 0 SS: stance A at anchors_a[2] (x=-0.4)
    1: ('b', 3),   # step 1 SS: stance B at anchors_b[3] (x=+0.4)  (b advanced 2→3 in step 0)
    2: ('a', 3),   # step 2 SS: stance A at anchors_a[3] (x=+0.4)  (a advanced 2→3 in step 1)
    3: ('b', 4),   # step 3 SS: stance B at anchors_b[4] (x=+1.2)
    4: ('a', 4),   # step 4 SS: stance A at anchors_a[4] (x=+1.2)
}


def main():
    sl = json.load(open(LOG))
    t = np.array(sl['t'])
    ph = np.array(sl['phase'], dtype=object)
    sidx = np.array(sl['step_idx'])
    lqp = np.array(sl['lambda_qp'])             # (N, 12)
    lref = np.array(sl['lambda_ref'])           # (N, 12) NMPC plan
    r_com = np.array(sl['r_com'])               # (N, 3)
    dock_t = [e['t'] for e in sl['dock_events']]

    # lambda_* convention: contact_1 = arm A, contact_2 = arm B
    # slot 0:6 = (f_a, tau_a), slot 6:12 = (f_b, tau_b)
    f_a = lqp[:, 0:3]; tau_a = lqp[:, 3:6]
    f_b = lqp[:, 6:9]; tau_b = lqp[:, 9:12]
    f_a_p = lref[:, 0:3]; tau_a_p = lref[:, 3:6]
    f_b_p = lref[:, 6:9]; tau_b_p = lref[:, 9:12]

    N = len(t)
    Hdot_s = np.zeros((N, 3))           # QP output
    Hdot_s_plan = np.zeros((N, 3))      # NMPC plan
    Ldot_com = np.zeros((N, 3))
    rC_stance = np.full((N, 3), np.nan)

    for i in range(N):
        s = int(sidx[i])
        phase = ph[i]
        if phase == 'SS' and s in STANCE_ANCHOR:
            arm, idx = STANCE_ANCHOR[s]
            rC = _anchor_pos(idx, arm)
            if arm == 'a':
                f, tau = f_a[i], tau_a[i]
                fp, taup = f_a_p[i], tau_a_p[i]
            else:
                f, tau = f_b[i], tau_b[i]
                fp, taup = f_b_p[i], tau_b_p[i]
            Hdot_s[i] = np.cross(rC, f) + tau
            Hdot_s_plan[i] = np.cross(rC, fp) + taup
            Ldot_com[i] = np.cross(rC - r_com[i], f) + tau
            rC_stance[i] = rC
        else:
            # DS: both contacts active. Use the anchor pair currently welded.
            ndocks = sum(1 for dt in dock_t if dt < t[i])
            a_idx = 2 + sum(1 for k in range(ndocks) if k % 2 == 1)
            b_idx = 2 + sum(1 for k in range(ndocks) if k % 2 == 0)
            for arm, idx, f, tau, fp, taup in [
                    ('a', a_idx, f_a[i], tau_a[i], f_a_p[i], tau_a_p[i]),
                    ('b', b_idx, f_b[i], tau_b[i], f_b_p[i], tau_b_p[i])]:
                rC = _anchor_pos(idx, arm)
                Hdot_s[i] += np.cross(rC, f) + tau
                Hdot_s_plan[i] += np.cross(rC, fp) + taup
                Ldot_com[i] += np.cross(rC - r_com[i], f) + tau

    Hdot_s_axis = np.abs(Hdot_s).max(axis=1)
    Hdot_s_plan_axis = np.abs(Hdot_s_plan).max(axis=1)
    Ldot_com_axis = np.abs(Ldot_com).max(axis=1)

    print('=== Group Ḣ_s: structure-disturbance moment vs τ_w_max=5 Nm ===\n')
    print('Per SS step — comparing NMPC plan (lambda_ref) vs QP output (lambda_qp):')
    print(f"  {'step':>4} {'stance':>6} {'|r_C|[m]':>9} "
          f"{'plan_peak':>10} {'plan_p95':>9} "
          f"{'qp_peak':>8} {'qp_p95':>7} {'qp_%>5':>7}")
    for s in range(5):
        m = (sidx == s) & (ph == 'SS')
        if not m.any():
            continue
        h_ax = Hdot_s_axis[m]
        h_pl = Hdot_s_plan_axis[m]
        arm, idx = STANCE_ANCHOR[s]
        rC_mag = np.linalg.norm(_anchor_pos(idx, arm))
        pct_over = float(np.mean(h_ax > TAU_W_MAX) * 100)
        print(f"  {s:>4} {arm+str(idx):>6} {rC_mag:>9.3f} "
              f"{h_pl.max():>10.2f} {np.percentile(h_pl, 95):>9.2f} "
              f"{h_ax.max():>8.2f} {np.percentile(h_ax, 95):>7.2f} "
              f"{pct_over:>6.1f}%")

    print()
    binding = Hdot_s_plan_axis.max() <= TAU_W_MAX + 1e-3
    print(f"  All-tick peak |Ḣ_s|_per-axis (NMPC plan)  = "
          f"{Hdot_s_plan_axis.max():.2f} Nm  "
          f"({'≤ 5 Nm — constraint respected ✓' if binding else '> 5 Nm — IPOPT relaxed / constraint not in NLP'})")
    print(f"  All-tick peak |Ḣ_s|_per-axis (QP output)   = "
          f"{Hdot_s_axis.max():.2f} Nm  (downstream / mapping)")
    print(f"  All-tick peak |L̇_com|_per-axis (former proxy) = "
          f"{Ldot_com_axis.max():.2f} Nm")
    div = np.linalg.norm(Hdot_s - Ldot_com, axis=1)
    print(f"  Max |Ḣ_s − L̇_com| over run = {div.max():.2f} Nm "
          f"(former-proxy divergence)")

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
               label='|Ḣ_s| QP output (downstream)')
    ax[0].plot(t, Hdot_s_plan_axis, 'C1', lw=1.1,
               label='|Ḣ_s| NMPC plan (active constraint)')
    ax[0].plot(t, Ldot_com_axis, 'C0', lw=0.8, alpha=0.5,
               label='|L̇_com| (former proxy)')
    ax[0].axhline(TAU_W_MAX, color='r', ls='--', lw=0.9, label='τ_w_max=5 Nm')
    ax[0].set_ylabel('moment [Nm]')
    ax[0].legend(fontsize=8, loc='upper right')
    ax[0].set_title('Wheel-torque rate cap: NMPC plan |Ḣ_s| (the active constraint) vs QP-output downstream')
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
