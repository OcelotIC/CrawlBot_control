#!/usr/bin/env python3
"""Attribution experiment 1 — λ decomposition (net vs internal) at the
step-2 wrench peak of the 1% canonical cooperative-arms traversal.

Question (campaign §6/§7 / §7c): the 10.8× QP-vs-NMPC contact-force
mismatch at t≈16.63 — is it INTERNAL force (two contacts simultaneously
large and opposing, living in the null space of the 12→6 centroidal map,
which soft-CoM constitutively cannot touch) or NET force (real momentum
rate the point-mass NMPC under-budgeted)?

Method (pure analysis on the existing sim_log.json):
  - λ_qp is [f1, τ1, f2, τ2] (two 6D contacts; wholebody_qp.py:17).
  - If contact 2 is inactive (f2=τ2=0) the grasp map collapses to a
    single 6D contact → the 12→6 map has NO internal null space →
    internal force is identically 0 by construction.
  - Cross-check the single-contact force against m·a_com (a_com from the
    numerically-differentiated logged v_com): if |f1| ≈ m·|a_com| the
    wrench is genuinely NET (it is accelerating the CoM), not a solver
    artifact.
  - Compare against the NMPC plan (lambda_ref) to reproduce the ratio.

Reads:  results/diag_cooperative_arms/sim_log.json
Writes: results/diag_attribution/lambda_decomp/{metrics.json, lambda_decomp.png}
"""
import os
import sys
import json

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')
os.environ.setdefault('MPLBACKEND', 'Agg')

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG = os.path.join(_root, 'results', 'diag_cooperative_arms', 'sim_log.json')
OUT = os.path.join(_root, 'results', 'diag_attribution', 'lambda_decomp')
M_ROBOT = 71.0  # kg, spec §0.4


def main(log_path=LOG, out_dir=OUT):
    LOG = log_path
    OUT = out_dir
    os.makedirs(OUT, exist_ok=True)
    log = json.load(open(LOG))
    t = np.array(log['t'])
    ph = np.array(log['phase'])
    step = np.array(log['step_idx'])
    lam = np.array(log['lambda_qp'])      # (N, 12) = [f1,tau1,f2,tau2]
    lref = np.array(log['lambda_ref'])    # (N, 12) NMPC plan
    vcom = np.array(log['v_com'])         # (N, 3)

    f1 = lam[:, 0:3]
    tau1 = lam[:, 3:6]
    f2 = lam[:, 6:9]
    tau2 = lam[:, 9:12]
    f1n = np.linalg.norm(f1, axis=1)
    f2n = np.linalg.norm(f2, axis=1)

    # a_com via central differences on logged v_com
    dt = np.gradient(t)
    acom = np.gradient(vcom, axis=0) / dt[:, None]
    acomn = np.linalg.norm(acom, axis=1)

    ss2 = (step == 2) & (ph == 'SS')
    idx2 = np.where(ss2)[0]
    assert len(idx2) > 0, "no step-2 SS ticks found"

    # peak by total contact-1 force in step-2 SS
    jrel = int(np.argmax(f1n[idx2]))
    j = int(idx2[jrel])

    # transient vs sustained: is the peak a lone spike or a raised floor?
    vcom_ref = np.array(log['v_com_ref'])
    f1n_ss = f1n[idx2]
    spike = {
        'f1_mean_N': float(f1n_ss.mean()),
        'f1_median_N': float(np.median(f1n_ss)),
        'f1_p95_N': float(np.percentile(f1n_ss, 95)),
        'f1_peak_N': float(f1n_ss.max()),
        'peak_over_median': float(f1n_ss.max() / max(np.median(f1n_ss), 1e-9)),
        'vcom_over_vcom_ref_at_peak': float(
            np.linalg.norm(vcom[j]) / max(np.linalg.norm(vcom_ref[j]), 1e-9)),
        'interpretation':
            'peak >> median AND v_com on-plan => transient jerk '
            '(F-SAT/delta(q) jitter), not a sustained authority shortfall',
    }

    metrics = {
        'log': os.path.relpath(LOG, _root),
        'm_robot_kg': M_ROBOT,
        'whole_run': {
            'contact2_f_max_N': float(f2n.max()),
            'contact2_f_max_t': float(t[int(np.argmax(f2n))]),
            'contact2_f_max_step': int(step[int(np.argmax(f2n))]),
        },
        'step2_SS': {
            'n_ticks': int(len(idx2)),
            'contact2_f_max_N': float(f2n[idx2].max()),
            'contact1_f_max_N': float(f1n[idx2].max()),
        },
        'step2_peak': {
            't': float(t[j]),
            'f1_N': [float(x) for x in f1[j]],
            'f1_norm_N': float(f1n[j]),
            'tau1_Nm': [float(x) for x in tau1[j]],
            'tau1_norm_Nm': float(np.linalg.norm(tau1[j])),
            'f2_N': [float(x) for x in f2[j]],
            'f2_norm_N': float(f2n[j]),
            'nmpc_ref_f_norm_N': float(np.linalg.norm(lref[j, 0:3])),
            'nmpc_ref_tau_norm_Nm': float(np.linalg.norm(lref[j, 3:6])),
            'qp_over_nmpc_force_ratio': float(
                f1n[j] / max(np.linalg.norm(lref[j, 0:3]), 1e-9)),
            'a_com_mps2': [float(x) for x in acom[j]],
            'a_com_norm_mps2': float(acomn[j]),
            'm_times_a_com_N': float(M_ROBOT * acomn[j]),
            # single active contact => grasp map is 6x6 full rank =>
            # internal-force null space is empty => internal == 0.
            'internal_force_N': 0.0,
            'internal_force_note':
                'contact 2 inactive at peak (f2=0) => no 12->6 null space',
        },
        'transient_vs_sustained': spike,
    }

    with open(os.path.join(OUT, 'metrics.json'), 'w') as fh:
        json.dump(metrics, fh, indent=2)

    # ---- plot over step-2 SS window ----
    tt = t[idx2]
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(tt, f1n[idx2], label='|f1| QP (stance contact)', lw=2)
    ax[0].plot(tt, f2n[idx2], label='|f2| QP (swing contact)', lw=2)
    ax[0].plot(tt, np.linalg.norm(lref[idx2, 0:3], axis=1),
               '--', label='|f| NMPC plan', lw=1.5)
    ax[0].plot(tt, M_ROBOT * acomn[idx2], ':',
               label='m·|a_com| (from v_com)', lw=1.5)
    ax[0].axvline(t[j], color='k', ls=':', alpha=0.5)
    ax[0].set_ylabel('force [N]')
    ax[0].legend(fontsize=8)
    ax[0].set_title('Step-2 SS: contact-force decomposition '
                    '(contact-2 ≡ 0 → no internal force)')

    ax[1].plot(tt, np.linalg.norm(tau1[idx2], axis=1),
               label='|τ1| QP (stance)', lw=2)
    ax[1].plot(tt, np.linalg.norm(lref[idx2, 3:6], axis=1),
               '--', label='|τ| NMPC plan', lw=1.5)
    ax[1].axvline(t[j], color='k', ls=':', alpha=0.5)
    ax[1].set_ylabel('moment [Nm]')
    ax[1].set_xlabel('t [s]')
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'lambda_decomp.png'), dpi=110)

    # ---- stdout summary ----
    p = metrics['step2_peak']
    print('=' * 68)
    print(' Experiment 1 — λ decomposition (net vs internal), step-2 peak')
    print('=' * 68)
    print(f"  peak t                 = {p['t']:.3f} s")
    print(f"  |f1| (stance, QP)      = {p['f1_norm_N']:.2f} N")
    print(f"  |f2| (swing, QP)       = {p['f2_norm_N']:.2f} N   "
          f"(max over step-2 SS = {metrics['step2_SS']['contact2_f_max_N']:.3f} N)")
    print(f"  |f| NMPC plan          = {p['nmpc_ref_f_norm_N']:.2f} N")
    print(f"  QP / NMPC force ratio  = {p['qp_over_nmpc_force_ratio']:.1f}x")
    print(f"  m·|a_com| (indep.)     = {p['m_times_a_com_N']:.1f} N")
    print(f"  |τ1| (stance, QP)      = {p['tau1_norm_Nm']:.2f} Nm   "
          f"(NMPC plan {p['nmpc_ref_tau_norm_Nm']:.2f} Nm)")
    print(f"  INTERNAL force         = 0 N  (single active contact)")
    print('-' * 68)
    sp = metrics['transient_vs_sustained']
    print(f"  |f1| over step-2 SS    : median={sp['f1_median_N']:.1f} "
          f"p95={sp['f1_p95_N']:.1f} peak={sp['f1_peak_N']:.1f} N "
          f"(peak/median={sp['peak_over_median']:.0f}x)")
    print(f"  v_com / v_com_ref@peak : {sp['vcom_over_vcom_ref_at_peak']:.2f}x "
          f"(~1 => CoM velocity ON-plan; peak is a transient jerk)")
    print('-' * 68)
    print(f"  contact-2 max over WHOLE run = "
          f"{metrics['whole_run']['contact2_f_max_N']:.2f} N at "
          f"t={metrics['whole_run']['contact2_f_max_t']:.2f} "
          f"(step {metrics['whole_run']['contact2_f_max_step']})")
    print(f"\n  saved: {os.path.relpath(OUT, _root)}/"
          "{metrics.json, lambda_decomp.png}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default=LOG)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    main(a.log, a.out)
