"""Q2b — Trackability bound at T15 step 2 specifically.

Two sweeps, both running the full 3-step T15 scenario but
intercepting `dock_configuration_fixed_rotation` ONLY on its 3rd
call (step 2). Steps 0 and 1 run with the natural IK.

Sweep A (along-axis): q_perturbed = q_start + α(q_natural - q_start)
  for α ∈ {0.5, 0.75, 1.0, 1.25, 1.5}. Tests whether the
  trackability finding from the single-step Q2 also holds at step 2.

Sweep B (orthogonal lateral-y): q_perturbed = q_natural with
  q_perturbed[1] += β. Tests the empirically-observed failure
  direction — the Phase-7 mid-waypoint's q_mid was at torso
  y=-0.44m vs natural near 0.
  β ∈ {-0.45, -0.30, -0.15, 0.0, +0.15, +0.30, +0.45} m.

Run per-perturbation as a fresh sim. Each run takes ~5-10 min.
Total runtime ~1-2 hours for both sweeps.

Outputs:
  Misc/runs/q1_q2/q2b_step2_results.json
  Misc/runs/q1_q2/q2b_along_axis.png
  Misc/runs/q1_q2/q2b_orthogonal_y.png
"""
from __future__ import annotations

import os
import json
import sys
import time
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import Misc.scripts.run_m7_v22_1pct_3step_t15 as t15_mod  # noqa: E402
from crawlbot.core import ik as ik_module  # noqa: E402
import crawlbot.simulation.sim_loop as sim_loop_mod  # noqa: E402

OUTDIR = os.path.join(_root, 'diagnostic')
RUN_OUT_BASE = os.path.join(_root, 'Misc', 'runs', 'diagnostic_q2b')

_orig_fixed_rot_ik = ik_module.dock_configuration_fixed_rotation


class StepTwoIkPatch:
    """Monkey-patch that intercepts ONLY the 3rd call (step 2)."""
    def __init__(self, kind: str, value: float):
        self.kind = kind     # 'alpha' or 'beta_y'
        self.value = value
        self.n_calls = 0
        self.recorded = {}

    def __call__(self, model, anchor_a, anchor_b, R_torso_fixed,
                 torso_pos=None, q_init=None, max_iter=2000, tol=1e-8):
        self.n_calls += 1
        q_natural, err, w_yosh, w_sigma_min = _orig_fixed_rot_ik(
            model, anchor_a, anchor_b, R_torso_fixed,
            torso_pos=torso_pos, q_init=q_init,
            max_iter=max_iter, tol=tol,
        )
        if self.n_calls != 3:
            return q_natural, err, w_yosh, w_sigma_min

        # Step 2: perturb.
        q_start_local = (np.asarray(q_init, dtype=float).copy()
                         if q_init is not None else q_natural.copy())
        q_perturbed = q_natural.copy()

        if self.kind == 'alpha':
            alpha = self.value
            # Torso position
            q_perturbed[:3] = (q_start_local[:3]
                               + alpha * (q_natural[:3] - q_start_local[:3]))
            # Torso quat via SO(3) extrap
            qS = q_start_local[3:7]; qN = q_natural[3:7]
            R_S = pin.Quaternion(qS[3], qS[0], qS[1], qS[2]).toRotationMatrix()
            R_N = pin.Quaternion(qN[3], qN[0], qN[1], qN[2]).toRotationMatrix()
            omega = pin.log3(R_S.T @ R_N)
            R_P = R_S @ pin.exp3(alpha * omega)
            qP = pin.Quaternion(R_P)
            q_perturbed[3] = qP.x; q_perturbed[4] = qP.y
            q_perturbed[5] = qP.z; q_perturbed[6] = qP.w
            # Arm joints
            q_perturbed[7:] = (q_start_local[7:]
                               + alpha * (q_natural[7:] - q_start_local[7:]))
            self.recorded['alpha'] = float(alpha)
            self.recorded['torso_pos_natural'] = q_natural[:3].tolist()
            self.recorded['torso_pos_perturbed'] = q_perturbed[:3].tolist()
            self.recorded['torso_pos_dist_natural_m'] = float(
                np.linalg.norm(q_natural[:3] - q_start_local[:3]))
            self.recorded['torso_pos_dist_perturbed_m'] = float(
                np.linalg.norm(q_perturbed[:3] - q_start_local[:3]))

        elif self.kind == 'beta_y':
            beta = self.value
            q_perturbed[:] = q_natural[:]
            q_perturbed[1] = q_natural[1] + beta
            self.recorded['beta_y'] = float(beta)
            self.recorded['torso_pos_natural'] = q_natural[:3].tolist()
            self.recorded['torso_pos_perturbed'] = q_perturbed[:3].tolist()

        return q_perturbed, err, w_yosh, w_sigma_min


def _run_one(kind: str, value: float, label: str, out_dir: str):
    patch = StepTwoIkPatch(kind, value)
    ik_module.dock_configuration_fixed_rotation = patch
    sim_loop_mod.dock_configuration_fixed_rotation = patch
    try:
        t0 = time.perf_counter()
        # Use the T15 baseline runner. It mutates MJCF, runs 3-step,
        # restores MJCF. We let it do its thing — only step 2's IK
        # output is replaced.
        os.makedirs(out_dir, exist_ok=True)
        # Need to override the OUT path used by the t15 module.
        orig_out = t15_mod.OUT
        t15_mod.OUT = out_dir
        try:
            t15_mod.main()
        finally:
            t15_mod.OUT = orig_out
        elapsed = time.perf_counter() - t0
    finally:
        ik_module.dock_configuration_fixed_rotation = _orig_fixed_rot_ik
        sim_loop_mod.dock_configuration_fixed_rotation = _orig_fixed_rot_ik

    # Extract step-2 outcome.
    sim_log = json.load(open(os.path.join(out_dir, 'sim_log.json')))
    step2_dock = None
    for ev in sim_log.get('dock_events', []):
        if ev['step'] == 2:
            step2_dock = ev
    step2_abort = None
    for ab in sim_log.get('aborted_steps', []):
        if ab['step_idx'] == 2:
            step2_abort = ab

    # SS-2 peaks
    t_arr = np.array(sim_log['t'])
    e_torso_pos = np.array(sim_log['e_torso_pos'])
    e_ee_pos = np.array(sim_log['e_ee_pos'])
    phase = np.array(sim_log['phase'])
    step_idx = np.array(sim_log['step_idx'])
    ss2_mask = (phase == 'SS') & (step_idx == 2)
    ss2_e_torso_peak = float(np.max(np.abs(e_torso_pos[ss2_mask])) * 1000) if ss2_mask.any() else float('nan')
    ss2_e_swing_peak = float(np.max(np.abs(e_ee_pos[ss2_mask])) * 1000) if ss2_mask.any() else float('nan')

    nmpc = Counter(sim_log.get('nmpc_status_str', []))
    n_total = sum(nmpc.values())

    out = {
        'kind': kind, 'value': float(value), 'label': label,
        'recorded': patch.recorded,
        'step2_docked': step2_dock is not None,
        'step2_dock_event': step2_dock,
        'step2_abort_event': step2_abort,
        'ss2_e_torso_peak_mm': ss2_e_torso_peak,
        'ss2_e_swing_peak_mm': ss2_e_swing_peak,
        'nmpc_total': int(n_total),
        'nmpc_infeas': int(nmpc.get('Infeasible_Problem_Detected', 0)),
        'elapsed_s': float(elapsed),
    }
    return out


def main():
    results = []

    # Sweep A: along-axis at step 2
    alphas = [0.5, 0.75, 1.0, 1.25, 1.5]
    for alpha in alphas:
        label = f"A_alpha_{alpha:.2f}"
        out_dir = os.path.join(RUN_OUT_BASE, label)
        print(f"\n{'='*70}\n  Sweep A (along-axis): α={alpha:.2f}\n{'='*70}")
        try:
            r = _run_one('alpha', alpha, label, out_dir)
            results.append(r)
            print(f"  step2: docked={r['step2_docked']}  "
                  f"e_swing_peak={r['ss2_e_swing_peak_mm']:.1f}mm  "
                  f"({r['elapsed_s']:.1f}s)")
        except Exception as e:
            print(f"  Run FAILED: {type(e).__name__}: {e}")
            results.append({'kind': 'alpha', 'value': float(alpha),
                            'label': label, 'error': str(e)})
        # Save incrementally so we don't lose data on crash.
        json.dump(results, open(os.path.join(OUTDIR, 'q2b_step2_results.json'),
                                 'w'), indent=2,
                  default=lambda o: list(o) if hasattr(o, '__iter__') else str(o))

    # Sweep B: orthogonal lateral-y at step 2
    betas = [-0.45, -0.30, -0.15, 0.0, +0.15, +0.30, +0.45]
    for beta in betas:
        label = f"B_betay_{beta:+.2f}"
        out_dir = os.path.join(RUN_OUT_BASE, label)
        print(f"\n{'='*70}\n  Sweep B (orthogonal y): β={beta:+.2f} m\n{'='*70}")
        try:
            r = _run_one('beta_y', beta, label, out_dir)
            results.append(r)
            print(f"  step2: docked={r['step2_docked']}  "
                  f"e_swing_peak={r['ss2_e_swing_peak_mm']:.1f}mm  "
                  f"({r['elapsed_s']:.1f}s)")
        except Exception as e:
            print(f"  Run FAILED: {type(e).__name__}: {e}")
            results.append({'kind': 'beta_y', 'value': float(beta),
                            'label': label, 'error': str(e)})
        json.dump(results, open(os.path.join(OUTDIR, 'q2b_step2_results.json'),
                                 'w'), indent=2,
                  default=lambda o: list(o) if hasattr(o, '__iter__') else str(o))

    # Plots
    valid_a = [r for r in results if r.get('kind') == 'alpha' and 'error' not in r]
    valid_b = [r for r in results if r.get('kind') == 'beta_y' and 'error' not in r]

    if valid_a:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        xs = [r['value'] for r in valid_a]
        ys_swing = [r['ss2_e_swing_peak_mm'] for r in valid_a]
        ys_dock = [
            (r['step2_dock_event']['d_mm'] if r['step2_docked']
             else r['step2_abort_event']['d_mm']) for r in valid_a]
        docked = [r['step2_docked'] for r in valid_a]
        axes[0].plot(xs, ys_swing, 'o-', linewidth=2)
        for x, y, ok in zip(xs, ys_swing, docked):
            axes[0].plot(x, y, 'o' if ok else 'x', markersize=10,
                         color='green' if ok else 'red', mew=2)
        axes[0].axvline(1.0, ls=':', color='gray')
        axes[0].set_xlabel(r'$\alpha$ (along-axis at step 2)')
        axes[0].set_ylabel('Step-2 SS swing-EE peak [mm]')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Step-2 swing peak')
        axes[1].plot(xs, ys_dock, 'o-', linewidth=2)
        for x, y, ok in zip(xs, ys_dock, docked):
            axes[1].plot(x, y, 'o' if ok else 'x', markersize=10,
                         color='green' if ok else 'red', mew=2)
        axes[1].axvline(1.0, ls=':', color='gray')
        axes[1].axhline(5.0, ls='--', color='red', alpha=0.5,
                        label='5 mm dock gate')
        axes[1].set_xlabel(r'$\alpha$')
        axes[1].set_ylabel('Step-2 d at dock or abort [mm]')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='upper right', fontsize=9)
        axes[1].set_title('Step-2 dock distance (green o=DOCKED, red x=ABORT)')
        fig.suptitle('Q2b sweep A — along-axis perturbation at T15 step 2')
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, 'q2b_along_axis.png'), dpi=130)
        plt.close(fig)

    if valid_b:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        xs = [r['value'] for r in valid_b]
        ys_swing = [r['ss2_e_swing_peak_mm'] for r in valid_b]
        ys_dock = [
            (r['step2_dock_event']['d_mm'] if r['step2_docked']
             else r['step2_abort_event']['d_mm']) for r in valid_b]
        docked = [r['step2_docked'] for r in valid_b]
        axes[0].plot(xs, ys_swing, 'o-', linewidth=2, color='#d62728')
        for x, y, ok in zip(xs, ys_swing, docked):
            axes[0].plot(x, y, 'o' if ok else 'x', markersize=10,
                         color='green' if ok else 'red', mew=2)
        axes[0].axvline(0.0, ls=':', color='gray')
        axes[0].axvline(-0.44, ls='-.', color='orange', alpha=0.6,
                        label='step-1 mid-waypoint y offset')
        axes[0].set_xlabel(r'$\beta_y$ at step 2 [m]')
        axes[0].set_ylabel('Step-2 SS swing-EE peak [mm]')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Step-2 swing peak')
        axes[0].legend(loc='upper right', fontsize=9)
        axes[1].plot(xs, ys_dock, 'o-', linewidth=2, color='#d62728')
        for x, y, ok in zip(xs, ys_dock, docked):
            axes[1].plot(x, y, 'o' if ok else 'x', markersize=10,
                         color='green' if ok else 'red', mew=2)
        axes[1].axvline(0.0, ls=':', color='gray')
        axes[1].axvline(-0.44, ls='-.', color='orange', alpha=0.6)
        axes[1].axhline(5.0, ls='--', color='red', alpha=0.5,
                        label='5 mm dock gate')
        axes[1].set_xlabel(r'$\beta_y$ [m]')
        axes[1].set_ylabel('Step-2 d at dock or abort [mm]')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('Step-2 dock distance (green o=DOCKED, red x=ABORT)')
        axes[1].legend(loc='upper right', fontsize=9)
        fig.suptitle('Q2b sweep B — orthogonal y perturbation at T15 step 2')
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, 'q2b_orthogonal_y.png'), dpi=130)
        plt.close(fig)

    print(f"\nWrote {OUTDIR}/q2b_step2_results.json (n={len(results)})")
    if valid_a:
        print(f"      {OUTDIR}/q2b_along_axis.png")
    if valid_b:
        print(f"      {OUTDIR}/q2b_orthogonal_y.png")


if __name__ == '__main__':
    main()
