"""Step-2-only QP isolation test: T15 (3,4) anchor pair from clean (3,3) state.

The T15 closed-loop runs steps 0/1 first, accumulating ~13 mm of
swing-EE drift by the time step 2 starts. This script ISOLATES step 2
by:

  - Initialising the system at start_a=3, start_b=3 (both arms welded
    at anchor 3 — the configuration the controller would have *if*
    steps 0/1 had docked perfectly).
  - Running n_steps=1: a single SS phase that swings arm-b from
    anchor_b[3] to anchor_b[4].

Two variants are exercised back-to-back:

  variant A: bypass_on_FK   — current default (cfg.mapping_bypass_in_ss
              = True under FK mode, linear torso ref frozen). Expected:
              swing arm physically can't reach (per CLOSING_REPORT.md
              §4.7's geometric infeasibility analysis).

  variant B: bypass_off_FK  — FK ref forwarded directly to QP. Tests
              whether the QP can track the FK refs CLEANLY when not
              fighting drift accumulated from prior steps.

Outputs (per variant):

  results/M7_step2_isolation/{A_bypass_on, B_bypass_off}/
    sim_log.json                  full per-tick trace
    physics_trace.pkl             live q at 50 Hz
    metrics.csv                   summary
    fig{1..10}_*.png              standard diagnostic figures
    fig_qp_isolation_*.png        custom figures for this test:
      - swing-EE position tracking (3D + per-axis)
      - torso position tracking (3D + per-axis)
      - tracking error norms over time (e_torso_pos, e_torso_ori,
        e_ee_pos, e_ee_ori)
      - constraint residuals: |stance EE - anchor| in world frame,
        weld torque magnitude, joint torque peak per tick
      - actuator load: tau_q vs limit (20 Nm), tau_w vs limit (5 Nm),
        hw vs limit (5 Nms)

  results/M7_step2_isolation/STEP2_QP_ISOLATION_REPORT.md
    Side-by-side comparison + headline metrics + verdict.

Run:
  PYTHONPATH=. MUJOCO_GL=disabled python3 \\
      scripts/run_m7_step2_isolation_qp_test.py
"""
from __future__ import annotations

import json
import os
import sys
import pickle
from typing import Optional

import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import scripts.run_m7_single_step as r_single
from crawlbot.simulation.config import SimConfig

OUT_ROOT = os.path.join(_root, 'results', 'M7_step2_isolation_long_hold')


def _make_step2_isolation_config(bypass_on: bool) -> SimConfig:
    """Construct the cfg for the step-2 isolation. Mirrors the T15-FK
    runner (run_m7_v22_1pct_3step_t15_fk.py) save for the bypass flag.
    """
    cfg = r_single._make_m7_config()
    # T15 baseline settings (byte-identical to the merged T15 runner).
    cfg.preplanner_a_cruise_max = 0.01
    cfg.preplanner_cruise_ramp_frac = 0.2
    cfg.mapping_bypass_in_ss = bypass_on   # variant knob
    cfg.swing_early_finish_fraction = 0.80
    cfg.aocs_off_in_ds = True
    # IK + FK refs (same as T15-FK runner).
    cfg.use_trajectory_aware_ik = True
    cfg.trajectory_ik_qstart_tolerance = 0.05
    cfg.trajectory_ik_n_samples = 5
    cfg.trajectory_ik_w_min_threshold = 1e-3
    cfg.reference_source = 'joint_space_fk'
    cfg.geodesic_n_tau = 21
    cfg.geodesic_n_iter = 120
    cfg.geodesic_tol = 1e-5
    # 5 extra seconds of grace beyond T_step. The default t_ss_margin
    # is 1.0 s. The QP isolation v1 showed d_swing decaying
    # exponentially through the early-finish hold (35 mm → 20 mm in
    # 2 s); ~3 more seconds at the same rate should reach <5 mm.
    cfg.t_ss_margin = 5.0
    return cfg


# --------------------------------------------------------------------
# Custom diagnostic figures
# --------------------------------------------------------------------


def _per_step_window(log_dict, step_idx: int = 0):
    """Return mask + time slice for the (single) step's SS phase."""
    ts = np.asarray(log_dict['t'])
    sx = np.asarray(log_dict['step_idx'])
    ph = np.asarray(log_dict['phase'])
    mask = (sx == step_idx) & (ph == 'SS')
    return mask, ts[mask]


def _plot_qp_isolation(log_dict, out_dir: str, label: str):
    """Generate the 4-panel custom figure suite for the QP isolation test."""
    mask, t = _per_step_window(log_dict, step_idx=0)
    if mask.sum() == 0:
        print(f"  [{label}] no step-0 SS samples; skipping plots")
        return
    t0 = t[0]
    t_rel = t - t0

    p_ee = np.asarray(log_dict['p_ee'])[mask]
    p_ee_ref = np.asarray(log_dict['p_ee_ref'])[mask]
    p_torso = np.asarray(log_dict['p_torso'])[mask]
    p_torso_ref = np.asarray(log_dict['p_torso_ref'])[mask]
    e_torso_pos = np.asarray(log_dict['e_torso_pos'])[mask] * 1000  # mm
    e_torso_ori = np.asarray(log_dict['e_torso_ori'])[mask]         # deg
    e_ee_pos = np.asarray(log_dict['e_ee_pos'])[mask] * 1000        # mm
    e_ee_ori = np.asarray(log_dict['e_ee_ori'])[mask]               # deg
    d_grip_swing = np.asarray(log_dict['d_grip_swing'])[mask] * 1000   # mm
    d_grip_stance = np.asarray(log_dict['d_grip_stance'])[mask] * 1000 # mm
    tau_max_joint = np.asarray(log_dict['tau_max_joint'])[mask]
    tau_w = np.asarray(log_dict['tau_w'])[mask]
    hw = np.asarray(log_dict['hw'])[mask]

    # Figure 1: tracking — position and rotation
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].plot(t_rel, p_torso_ref[:, 0], '--', color='C0', label='ref x')
    axes[0, 0].plot(t_rel, p_torso[:, 0], '-', color='C0', label='actual x')
    axes[0, 0].plot(t_rel, p_torso_ref[:, 1], '--', color='C1', label='ref y')
    axes[0, 0].plot(t_rel, p_torso[:, 1], '-', color='C1', label='actual y')
    axes[0, 0].plot(t_rel, p_torso_ref[:, 2], '--', color='C2', label='ref z')
    axes[0, 0].plot(t_rel, p_torso[:, 2], '-', color='C2', label='actual z')
    axes[0, 0].set_title('Torso position: reference vs actual')
    axes[0, 0].set_ylabel('m')
    axes[0, 0].set_xlabel('t [s] (SS-relative)')
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(t_rel, p_ee_ref[:, 0], '--', color='C0', label='ref x')
    axes[0, 1].plot(t_rel, p_ee[:, 0], '-', color='C0', label='actual x')
    axes[0, 1].plot(t_rel, p_ee_ref[:, 1], '--', color='C1', label='ref y')
    axes[0, 1].plot(t_rel, p_ee[:, 1], '-', color='C1', label='actual y')
    axes[0, 1].plot(t_rel, p_ee_ref[:, 2], '--', color='C2', label='ref z')
    axes[0, 1].plot(t_rel, p_ee[:, 2], '-', color='C2', label='actual z')
    axes[0, 1].set_title('Swing-EE position: reference vs actual')
    axes[0, 1].set_ylabel('m')
    axes[0, 1].set_xlabel('t [s] (SS-relative)')
    axes[0, 1].legend(fontsize=8, ncol=2)
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(t_rel, e_torso_pos, '-', color='C0', label='||e_torso_pos|| [mm]')
    axes[1, 0].plot(t_rel, e_ee_pos,    '-', color='C1', label='||e_ee_pos||    [mm]')
    axes[1, 0].plot(t_rel, d_grip_swing, '--', color='C3', label='d_grip_swing  [mm]')
    axes[1, 0].plot(t_rel, d_grip_stance, ':', color='C2', label='d_grip_stance [mm]')
    axes[1, 0].axhline(5, color='red', ls='--', lw=1, label='dock 5 mm')
    axes[1, 0].set_title('Tracking errors + gripper distances')
    axes[1, 0].set_ylabel('mm')
    axes[1, 0].set_xlabel('t [s] (SS-relative)')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_ylim(0.1, 1500)

    axes[1, 1].plot(t_rel, e_torso_ori, '-', color='C0', label='e_torso_ori [°]')
    axes[1, 1].plot(t_rel, e_ee_ori,    '-', color='C1', label='e_ee_ori    [°]')
    axes[1, 1].axhline(5, color='red', ls='--', lw=1, label='dock 5°')
    axes[1, 1].set_title('Orientation tracking errors')
    axes[1, 1].set_ylabel('deg')
    axes[1, 1].set_xlabel('t [s] (SS-relative)')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)
    plt.suptitle(f'QP isolation step-2 [{label}]: tracking', fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_qp_isolation_tracking.png'), dpi=140)
    plt.close(fig)

    # Figure 2: actuator load — joint torque, AOCS, momentum
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(t_rel, tau_max_joint, '-', color='C0', label='peak |tau_q| [Nm]')
    axes[0].axhline(20, color='red', ls='--', lw=1, label='limit 20 Nm')
    axes[0].set_title('Joint torque (peak per tick)')
    axes[0].set_ylabel('Nm')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    tau_w_norm = np.linalg.norm(tau_w, axis=1)
    axes[1].plot(t_rel, tau_w[:, 0], '-', color='C0', alpha=0.6, label='tau_w_x')
    axes[1].plot(t_rel, tau_w[:, 1], '-', color='C1', alpha=0.6, label='tau_w_y')
    axes[1].plot(t_rel, tau_w[:, 2], '-', color='C2', alpha=0.6, label='tau_w_z')
    axes[1].plot(t_rel, tau_w_norm,  '-', color='black',           label='||tau_w||')
    axes[1].axhline(5, color='red', ls='--', lw=1, label='limit 5 Nm')
    axes[1].axhline(-5, color='red', ls='--', lw=1)
    axes[1].set_title('AOCS reaction-wheel torque')
    axes[1].set_ylabel('Nm')
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].grid(alpha=0.3)

    hw_norm = np.linalg.norm(hw, axis=1)
    axes[2].plot(t_rel, hw[:, 0], '-', color='C0', alpha=0.6, label='hw_x')
    axes[2].plot(t_rel, hw[:, 1], '-', color='C1', alpha=0.6, label='hw_y')
    axes[2].plot(t_rel, hw[:, 2], '-', color='C2', alpha=0.6, label='hw_z')
    axes[2].plot(t_rel, hw_norm,  '-', color='black',           label='||hw||')
    axes[2].axhline(5, color='red', ls='--', lw=1, label='limit 5 Nms')
    axes[2].axhline(-5, color='red', ls='--', lw=1)
    axes[2].set_title('AOCS reaction-wheel momentum')
    axes[2].set_ylabel('Nms')
    axes[2].set_xlabel('t [s] (SS-relative)')
    axes[2].legend(fontsize=8, ncol=2)
    axes[2].grid(alpha=0.3)
    plt.suptitle(f'QP isolation step-2 [{label}]: actuator load', fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_qp_isolation_actuators.png'), dpi=140)
    plt.close(fig)

    print(f"  [{label}] custom figs → {out_dir}/fig_qp_isolation_{{tracking,actuators}}.png")


def _summarise(log_dict, label: str):
    """Compute headline metrics for the report."""
    mask, t = _per_step_window(log_dict, step_idx=0)
    if mask.sum() == 0:
        return None
    metrics = {
        'label': label,
        't_ss_start': float(t[0]),
        't_ss_end': float(t[-1]),
        'T_phase_s': float(t[-1] - t[0]),
        'd_grip_swing_min_mm': float(np.asarray(log_dict['d_grip_swing'])[mask].min()) * 1000,
        'd_grip_swing_final_mm': float(np.asarray(log_dict['d_grip_swing'])[mask][-1]) * 1000,
        'e_torso_pos_peak_mm': float(np.asarray(log_dict['e_torso_pos'])[mask].max()) * 1000,
        'e_torso_ori_peak_deg': float(np.asarray(log_dict['e_torso_ori'])[mask].max()),
        'e_ee_pos_peak_mm': float(np.asarray(log_dict['e_ee_pos'])[mask].max()) * 1000,
        'e_ee_ori_peak_deg': float(np.asarray(log_dict['e_ee_ori'])[mask].max()),
        'tau_max_joint_peak_Nm': float(np.asarray(log_dict['tau_max_joint'])[mask].max()),
        'tau_w_peak_Nm': float(np.linalg.norm(np.asarray(log_dict['tau_w'])[mask], axis=1).max()),
        'hw_peak_Nms': float(np.linalg.norm(np.asarray(log_dict['hw'])[mask], axis=1).max()),
        'nmpc_infeas_count': sum(
            1 for s in np.asarray(log_dict['nmpc_status_str'])[mask]
            if 'Infeasible' in str(s)),
    }
    docks = log_dict.get('dock_events', []) or []
    aborts = log_dict.get('aborted_steps', []) or []
    metrics['docked'] = bool(docks)
    if docks:
        metrics['dock_d_mm'] = float(docks[0]['d_mm'])
        metrics['dock_ori_deg'] = float(docks[0].get('ori_deg', float('nan')))
    if aborts:
        metrics['abort_reason'] = aborts[0].get('reason', 'unknown')
    return metrics


def _verdict_for(metrics: dict) -> str:
    if metrics['docked']:
        return f"DOCKED at d={metrics['dock_d_mm']:.2f} mm, ori={metrics.get('dock_ori_deg', float('nan')):.2f}°"
    return f"FAILED at d_min={metrics['d_grip_swing_min_mm']:.1f} mm ({metrics.get('abort_reason', 'unknown')})"


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    variants = [
        ('long_hold', True, 't_ss_margin_5.0'),
    ]
    results = {}
    for sub, bypass_on, label in variants:
        out_dir = os.path.join(OUT_ROOT, sub)
        os.makedirs(out_dir, exist_ok=True)
        cfg = _make_step2_isolation_config(bypass_on=bypass_on)
        print(f"\n{'='*72}")
        print(f"  variant {sub}: t_ss_margin={cfg.t_ss_margin}s, "
              f"start=(3, 3), n_steps=1, swing arm goes 3 → 4")
        print(f"{'='*72}")
        sim, log, metrics = r_single.run_case(
            f'STEP2-ISOLATION variant={sub} '
            f'(reference_source=joint_space_fk, t_ss_margin={cfg.t_ss_margin}s)',
            out_dir, n_steps=1, config=cfg, start_a=3, start_b=3,
        )
        log_dict = json.load(open(os.path.join(out_dir, 'sim_log.json')))
        _plot_qp_isolation(log_dict, out_dir, sub)
        m = _summarise(log_dict, sub)
        results[sub] = m
        with open(os.path.join(out_dir, 'qp_isolation_metrics.json'), 'w') as fp:
            json.dump(m, fp, indent=2)

    # Single-variant report
    rep_path = os.path.join(OUT_ROOT, 'STEP2_LONG_HOLD_REPORT.md')
    with open(rep_path, 'w') as fp:
        fp.write("# Step-2 isolation with extended SS margin — report\n\n")
        fp.write(f"**Branch:** `claude/fk-bypass-aware-tuning`\n\n")
        fp.write(f"**Test scenario:** `start_a=3, start_b=3, n_steps=1, "
                 f"t_ss_margin=5.0s` — clean (3,3) start, swing arm goes "
                 f"to anchor 4, with **5 extra seconds of grace beyond "
                 f"T_step** to let the QP's exponential approach converge.\n\n")
        fp.write("## Headline metric\n\n")
        m = results['long_hold']
        fp.write(f"- d_grip_swing_min: **{m['d_grip_swing_min_mm']:.2f} mm**\n")
        fp.write(f"- d_grip_swing_final: {m['d_grip_swing_final_mm']:.2f} mm\n")
        fp.write(f"- T_phase: {m['T_phase_s']:.2f} s\n")
        fp.write(f"- Verdict: **{_verdict_for(m)}**\n\n")
        fp.write("## Full metrics\n\n")
        fp.write("| metric | value |\n|---|---|\n")
        for k in ['docked', 'd_grip_swing_min_mm', 'd_grip_swing_final_mm',
                  'e_torso_pos_peak_mm', 'e_torso_ori_peak_deg',
                  'e_ee_pos_peak_mm',  'e_ee_ori_peak_deg',
                  'tau_max_joint_peak_Nm', 'tau_w_peak_Nm', 'hw_peak_Nms',
                  'nmpc_infeas_count', 'T_phase_s']:
            v = m.get(k, '—')
            if isinstance(v, float):
                v = f"{v:.3f}"
            fp.write(f"| {k} | {v} |\n")
        fp.write(f"\n## Figures\n\n")
        fp.write(f"- `long_hold/fig_qp_isolation_tracking.png` — torso/EE refs vs actual\n")
        fp.write(f"- `long_hold/fig_qp_isolation_actuators.png` — joint, AOCS torque, AOCS momentum\n")
        fp.write(f"- `long_hold/fig{{1..10}}_*.png` — full diagnostic suite\n")
    print(f"\nReport: {rep_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
