"""Group D diagnostic: actuator + solver health.

Questions:
  - The step-2 ~62N contact-force spike: when, which contact, and is it
    a QP-vs-NMPC mismatch (QP demanding more wrench than the NMPC
    planned)? (B localized a tau_w saturation to the same step-2 window.)
  - Per-joint torque vs +-tau_max (20 Nm): any saturation?
  - NMPC solver health: solve rate < 50ms, infeasibility rate, iters.

Plots: |f| per contact (QP vs NMPC ref); per-joint tau vs +-20; NMPC
solve time vs 50ms + status. Phase shading + dock lines.

Usage:
  PYTHONPATH=. MUJOCO_GL=osmesa python3 scripts/diag_actuator_solver.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(_root, 'results', 'diag_cooperative_arms', 'sim_log.json')
OUT = os.path.join(_root, 'results', 'diag_cooperative_arms', 'actuator_solver.png')
TAU_MAX = 20.0


def main():
    sl = json.load(open(LOG))
    t = np.array(sl['t']); ph = np.array(sl['phase'], dtype=object)
    sidx = np.array(sl['step_idx'])
    lqp = np.array(sl['lambda_qp'])               # (N,12) 2 contacts x 6D
    lref = np.array(sl['lambda_ref'])             # (N,12) NMPC planned
    tau = np.array(sl['tau'])                     # (N,14)
    nt = np.array(sl['nmpc_time_ms']) if sl.get('nmpc_time_ms') else None
    nok = np.array(sl['nmpc_ok']) if sl.get('nmpc_ok') else None
    dock_t = [e['t'] for e in sl['dock_events']]

    # contact force magnitudes (per contact)
    f1_qp = np.linalg.norm(lqp[:, 0:3], axis=1)
    f2_qp = np.linalg.norm(lqp[:, 6:9], axis=1)
    f1_ref = np.linalg.norm(lref[:, 0:3], axis=1)
    f2_ref = np.linalg.norm(lref[:, 6:9], axis=1)
    f_qp_max = np.maximum(f1_qp, f2_qp)
    f_ref_max = np.maximum(f1_ref, f2_ref)

    tau_abs = np.abs(tau)
    tau_peak_joint = tau_abs.max(axis=0)          # (14,)

    print('=== Group D: actuator + solver health ===')
    kspk = int(f_qp_max.argmax())
    print(f'  peak QP contact force |f| = {f_qp_max[kspk]:.1f} N at t={t[kspk]:.2f} '
          f'(step {sidx[kspk]}, {ph[kspk]})')
    print(f'    NMPC-planned |f| at that tick = {f_ref_max[kspk]:.1f} N '
          f'-> QP/NMPC ratio = {f_qp_max[kspk]/max(f_ref_max[kspk],1e-6):.1f}x')
    print(f'  contact force: QP peak {f_qp_max.max():.1f}N vs NMPC-ref peak {f_ref_max.max():.1f}N')
    print()
    print(f'  per-joint |tau| peak (Nm), limit {TAU_MAX}:')
    print('    ' + ' '.join(f'{v:.1f}' for v in tau_peak_joint))
    print(f'    worst joint = {tau_peak_joint.max():.1f} Nm '
          f'({tau_peak_joint.max()/TAU_MAX*100:.0f}% of limit); '
          f'ticks any joint >=0.99*lim: {(tau_abs >= 0.99*TAU_MAX).any(axis=1).sum()}')
    print()
    if nt is not None:
        rate = float((nt < 50.0).mean())
        print(f'  NMPC solve time: peak {nt.max():.1f}ms, mean {nt.mean():.1f}ms; '
              f'fraction <50ms = {rate*100:.1f}% (thresh 95%)')
    if nok is not None:
        print(f'  NMPC ok rate = {float(nok.mean())*100:.1f}% '
              f'(infeasible {100*(1-float(nok.mean())):.1f}%, thresh <2%)')

    # --- figure ---
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ss = ph == 'SS'
    def deco(a):
        a.fill_between(t, *a.get_ylim(), where=ss, color='orange', alpha=0.06)
        for dt in dock_t:
            a.axvline(dt, color='k', ls=':', lw=0.6, alpha=0.5)
        a.grid(alpha=0.3)

    ax[0].plot(t, f_qp_max, 'C3', lw=1.0, label='|f| QP (worst contact)')
    ax[0].plot(t, f_ref_max, 'C0', lw=0.9, alpha=0.7, label='|f| NMPC plan')
    ax[0].annotate(f'{f_qp_max[kspk]:.0f}N', xy=(t[kspk], f_qp_max[kspk]),
                   xytext=(t[kspk]+1, f_qp_max[kspk]), fontsize=8,
                   arrowprops=dict(arrowstyle='->', lw=0.6))
    ax[0].set_ylabel('contact |f| [N]'); ax[0].legend(fontsize=8)
    ax[0].set_title('Contact force: QP solution vs NMPC plan')

    for j in range(tau.shape[1]):
        ax[1].plot(t, tau[:, j], lw=0.6)
    ax[1].axhline(TAU_MAX, color='r', ls='--', lw=0.8); ax[1].axhline(-TAU_MAX, color='r', ls='--', lw=0.8)
    ax[1].set_ylabel('joint tau [Nm]\n(+-20 limit)')
    ax[1].set_title(f'Per-joint torque (worst peak {tau_peak_joint.max():.1f} Nm)')

    if nt is not None:
        ax[2].plot(t, nt, 'C2', lw=0.8)
        ax[2].axhline(50.0, color='r', ls='--', lw=0.8, label='50 ms')
        ax[2].set_ylabel('NMPC solve [ms]'); ax[2].legend(fontsize=8)
    ax[2].set_xlabel('t [s]')
    for a in ax:
        deco(a)
    fig.tight_layout(); fig.savefig(OUT, dpi=120); plt.close(fig)
    print('\nwrote', OUT)


if __name__ == '__main__':
    main()
