#!/usr/bin/env python3
"""Diagnostic: run SS with the QP in pure-PD mode.

Sets three runtime diagnostic hooks on SimulationLoop:
  _diag_pure_pd = True    → strip a_com_ff, a_torso_ff, λ_ref, L_com_ref
  (keep r_b_ref / v_b_ref from mapping as the PD target)

Then runs the M6 baseline config for 1 step at 1% mass ratio and
reports the torso tracking / joint saturation / kinetic energy
telemetry. The user's protocol:

  - If the torso still oscillates → PD gains are unstable for this
    plant. Needs Kp_torso reduction.
  - If the torso tracks cleanly → the NMPC feedforward (λ_ref or
    a_b_ff) is the energy source. Print λ_ref at the first SS call
    to check whether it's a stale DS value being carried into SS.

Output:
  stdout: side-by-side metric table (baseline vs pure-PD)
  plot:   Misc/runs/M6_platform_diag/pure_pd_vs_baseline.png
"""
import os
import sys

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import json
from io import StringIO
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from crawlbot.simulation.sim_loop import SimulationLoop
from crawlbot.simulation.config import SimConfig
from crawlbot.core.state_conversions import mujoco_to_pinocchio

URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')
MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')
OUT_DIR = 'Misc/runs/M6_platform_diag'


def _make_cfg():
    return SimConfig(
        use_m2_stack=True,
        alpha_com_soft=5.0, alpha_passivity=1.0,
        enforce_hw_conservation=True,
        h_max_tight=np.full(3, 5.0),
        w_L_nmpc=1.0, kappa_terminal=1.0,
        aocs_mode='legacy_corrected',
        aocs_use_legacy_corrected=True,
        use_coarse_preplanner=True,
    )


def run(tag: str, pure_pd: bool, freeze_ref: bool = False):
    old = sys.stdout
    sys.stdout = StringIO()
    try:
        sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=_make_cfg())
        sim.setup(n_steps=1, start_a=2, start_b=2)
        sim._diag_pure_pd = pure_pd
        sim._diag_freeze_ref = freeze_ref
        log = sim.run(verbose=False)
    finally:
        sys.stdout = old
    return sim, log


def summarize(tag, log):
    tau = np.asarray(log.tau)
    e_tp = np.asarray(log.e_torso_pos)
    e_ee = np.asarray(log.e_ee_pos)
    T_kin = np.asarray(log.T_kinetic)
    p_tor = np.asarray(log.p_torso)
    p_ref = np.asarray(log.p_torso_ref)
    nmpc_ok = np.asarray(log.nmpc_ok)
    d_swing = np.asarray(log.d_grip_swing)
    eul = np.asarray(log.struct_euler_deg)
    eul0 = eul[0].copy()
    peak_rot = float(np.max(np.abs(eul - eul0)))

    any_sat = np.any(np.abs(tau) > 19.9, axis=1)
    n_sat = int(any_sat.sum())
    n_tot = len(any_sat)

    return {
        'tag': tag,
        'peak_torso_err_mm': float(e_tp.max() * 1000),
        'mean_torso_err_mm': float(e_tp.mean() * 1000),
        'peak_ee_err_mm': float(e_ee.max() * 1000),
        'peak_T_kin_J': float(T_kin.max()),
        'mean_T_kin_J': float(T_kin.mean()),
        'peak_struct_rot_deg': peak_rot,
        'sat_frac': float(any_sat.mean()),
        'n_sat': n_sat,
        'n_tot': n_tot,
        'nmpc_success_frac': float(nmpc_ok.mean()),
        'min_d_swing_mm': float(d_swing.min() * 1000),
        't': np.asarray(log.t),
        'p_tor': p_tor, 'p_ref': p_ref,
        'T_kin': T_kin, 'tau': tau, 'e_tp': e_tp,
    }


def print_lambda_ref_at_first_ss(sim, log):
    """Print the NMPC's λ_ref at the first QP step of SS phase
    (baseline run only — not the pure-PD run, which zeros it).
    """
    phase = np.asarray(log.phase)
    t = np.asarray(log.t)
    lam_norm = np.asarray(log.lambda_ref_norm)
    lam = np.asarray(log.lambda_ref) if hasattr(log, 'lambda_ref') else None

    # Find first SS step
    ss_idx = np.where(phase == 'SS')[0]
    if len(ss_idx) == 0:
        print("  (no SS step found)")
        return
    i0 = int(ss_idx[0])
    t0 = float(t[i0])
    print(f"  first SS step: idx={i0}, t={t0:.3f}s")
    print(f"  |λ_ref| norm = {lam_norm[i0]:.4f}")
    if lam is not None:
        print(f"  λ_ref vector = {lam[i0]}")
        print(f"    contact1 f = {lam[i0][0:3]}")
        print(f"    contact1 τ = {lam[i0][3:6]}")
        print(f"    contact2 f = {lam[i0][6:9]}")
        print(f"    contact2 τ = {lam[i0][9:12]}")

    # Also print the previous step (last DS) for comparison
    if i0 > 0:
        i_prev = i0 - 1
        print(f"  previous (DS) step: idx={i_prev}, t={float(t[i_prev]):.3f}s")
        print(f"  |λ_ref| at prev  = {lam_norm[i_prev]:.4f}")
        if lam is not None:
            print(f"  λ_ref at prev    = {lam[i_prev]}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 78)
    print("  Pure-PD diagnostic (1% single step)")
    print("=" * 78)

    # Baseline run (full feedforward)
    print("\nRunning BASELINE (full feedforward)...")
    sim_b, log_b = run("baseline", pure_pd=False)
    sum_b = summarize("baseline", log_b)

    # Pure-PD run (references still move; only feedforward stripped)
    print("Running PURE-PD (a_b_ff=0, λ_ref=0, L_com_ref=0)...")
    sim_p, log_p = run("pure_pd", pure_pd=True)
    sum_p = summarize("pure_pd", log_p)

    # Pure-PD + FROZEN reference (r_b_ref held at first sample).
    # This probes the PD loop's stability around a STATIC target,
    # decoupling the "can PD track a moving reference" question from
    # "is the PD loop itself stable".
    print("Running PURE-PD + FROZEN REF (r_b_ref held static)...")
    sim_f, log_f = run("pure_pd_frozen", pure_pd=True, freeze_ref=True)
    sum_f = summarize("pure_pd_frozen", log_f)

    # Side-by-side summary
    print()
    print("=" * 78)
    print("  RESULTS")
    print("=" * 78)
    rows = [
        ('peak torso err [mm]',  'peak_torso_err_mm'),
        ('mean torso err [mm]',  'mean_torso_err_mm'),
        ('peak EE err [mm]',     'peak_ee_err_mm'),
        ('peak T_kin [J]',       'peak_T_kin_J'),
        ('mean T_kin [J]',       'mean_T_kin_J'),
        ('peak |Δplatform| [°]', 'peak_struct_rot_deg'),
        ('joint sat. fraction',  'sat_frac'),
        ('NMPC success frac.',   'nmpc_success_frac'),
        ('min d_swing [mm]',     'min_d_swing_mm'),
    ]
    print(f"{'metric':>28} {'baseline':>14} {'pure_pd':>14} {'pure_pd+fz':>14}")
    print("-" * 92)
    for label, k in rows:
        vb = sum_b[k]; vp = sum_p[k]; vf = sum_f[k]
        if k == 'sat_frac' or k == 'nmpc_success_frac':
            print(f"{label:>28} {vb*100:13.1f}% {vp*100:13.1f}% {vf*100:13.1f}%")
        elif 'kin' in k or 'rot' in k:
            print(f"{label:>28} {vb:14.3f} {vp:14.3f} {vf:14.3f}")
        else:
            print(f"{label:>28} {vb:14.1f} {vp:14.1f} {vf:14.1f}")

    # λ_ref trace (from baseline, since pure_pd zeroes it)
    print()
    print("-" * 78)
    print("  λ_ref at first SS call (BASELINE run)")
    print("-" * 78)
    print_lambda_ref_at_first_ss(sim_b, log_b)

    # Plot comparison (3 modes)
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for i, ax in enumerate(axes[:, 0]):
        ax.plot(sum_b['t'], sum_b['p_ref'][:, i] * 1000, 'b-', label='ref (moving)',   lw=1.5)
        ax.plot(sum_b['t'], sum_b['p_tor'][:, i] * 1000, 'r-', label='baseline',       lw=1.5)
        ax.plot(sum_p['t'], sum_p['p_tor'][:, i] * 1000, 'g-', label='pure_pd',        lw=1.5)
        ax.plot(sum_f['t'], sum_f['p_tor'][:, i] * 1000, 'm-', label='pure_pd+frozen', lw=1.5)
        ax.set_ylabel(f"{'xyz'[i]} [mm]")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc='best', fontsize=7)
    axes[0, 0].set_title('p_torso trajectory')
    axes[-1, 0].set_xlabel('time [s]')

    # Right column: kinetic energy + joint torques + torso error
    ax = axes[0, 1]
    ax.semilogy(sum_b['t'], np.maximum(sum_b['T_kin'], 1e-8), 'r-', label='baseline',     lw=1.3)
    ax.semilogy(sum_p['t'], np.maximum(sum_p['T_kin'], 1e-8), 'g-', label='pure_pd',      lw=1.3)
    ax.semilogy(sum_f['t'], np.maximum(sum_f['T_kin'], 1e-8), 'm-', label='pure_pd+frozen', lw=1.3)
    ax.set_ylabel('T_kin [J] (log)')
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which='both')
    ax.set_title('Arm kinetic energy')

    ax = axes[1, 1]
    ax.plot(sum_b['t'], np.max(np.abs(sum_b['tau']), axis=1), 'r-', label='baseline',     lw=1.0)
    ax.plot(sum_p['t'], np.max(np.abs(sum_p['tau']), axis=1), 'g-', label='pure_pd',      lw=1.0)
    ax.plot(sum_f['t'], np.max(np.abs(sum_f['tau']), axis=1), 'm-', label='pure_pd+frozen', lw=1.0)
    ax.axhline(20, color='k', ls=':', alpha=0.5, label='τ_max')
    ax.set_ylabel('max |τ_j| [Nm]')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[2, 1]
    ax.plot(sum_b['t'], sum_b['e_tp'] * 1000, 'r-', label='baseline', lw=1.3)
    ax.plot(sum_p['t'], sum_p['e_tp'] * 1000, 'g-', label='pure_pd',  lw=1.3)
    ax.plot(sum_f['t'], sum_f['e_tp'] * 1000, 'm-', label='pure_pd+frozen', lw=1.3)
    ax.set_ylabel('|p_tor - p_ref| [mm]')
    ax.set_xlabel('time [s]')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle("Pure-PD vs baseline (M6 1% single step)")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, 'pure_pd_vs_baseline.png')
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\n  saved: {out_path}")

    print()
    print("=" * 78)
    print("  INTERPRETATION")
    print("=" * 78)
    r_Tb = sum_b['peak_T_kin_J']
    r_Tp = sum_p['peak_T_kin_J']
    r_Tf = sum_f['peak_T_kin_J']
    print(f"  peak T_kin: baseline={r_Tb:.2f}  pure_pd={r_Tp:.2f}  pure_pd+frozen={r_Tf:.2f}")
    if r_Tf < 0.5:
        print("  ✓ With a FROZEN reference the PD loop converges (peak T_kin < 0.5 J).")
        print("    → The PD gains are stable around a STATIC target.")
        print("    → The instability comes from REFERENCE MOTION:")
        if r_Tp > r_Tb:
            print("      + Pure-PD (moving ref, no feedforward) is WORSE than")
            print("        baseline → the feedforward is essential for reference")
            print("        tracking and removing it creates lag that the PD")
            print("        amplifies. This is NOT a stale-λ_ref problem.")
        else:
            print("      + Pure-PD is better than baseline → the feedforward IS")
            print("        the energy source. Investigate λ_ref / a_b_ff.")
        print("    → Next: look at the reference trajectory itself. Is the")
        print("      torso planner / mapping asking for motion that exceeds")
        print("      the actuator bandwidth?")
    elif r_Tf > 0.5 * r_Tb:
        print(f"  ✗ Even with a FROZEN reference, the PD loop blows up.")
        print(f"    → PD gains are unstable. Kp_torso={_make_cfg().ss_Kp_torso}, "
              f"Kd_torso={_make_cfg().ss_Kd_torso}.")
        print("    → Next: sweep Kp_torso downward until frozen-ref T_kin < 0.5 J.")
    else:
        print(f"  ~ Frozen-ref partially improved but still energetic.")
        print("    → Both PD gains AND reference tracking contribute.")


if __name__ == "__main__":
    main()
