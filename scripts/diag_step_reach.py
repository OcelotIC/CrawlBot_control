"""Per-step arm-extension and σ_min diagnostic.

Loads step_q_end.json + ik_trace.json from a results dir (produced
by diag_cooperative_arms.py). For each step, samples the planned
quintic SE3 trajectory from q_start to q_end at K interior τ values
(via crawlbot.core.ik._interpolate_q_quintic — same sampler as IK 3)
and computes per arm:

  - Extension ratio:  ||tool - shoulder|| / R_max
  - σ_min(Ĵ):         arm-internal Jacobian min-singular-value
                      (via crawlbot.core.ik._sigma_min_pair)

R_max is the theoretical straight-line shoulder→tool reach derived
from the URDF link offsets along the 7-DOF chain (Joint_1 → tool):
0.5 (elbow swivel) + 0.65 + 0.1 + 0.325 = 1.575 m. This is an upper
bound on shoulder→tool distance at any configuration (triangle
inequality), so extension_ratio ∈ [0, 1] with 1 = workspace boundary.

Outputs (in the given results dir):
  step_reach.csv  — long-form (step_idx, anchor_pair, tau, ext_a,
                    ext_b, sig_a, sig_b)
  step_reach.png  — extension vs σ_min, per-arm panels, colored by
                    step; horizontal/vertical guide lines at
                    σ_min = 1e-4 and extension = 0.85

Usage:
  PYTHONPATH=. MUJOCO_GL=osmesa python3 scripts/diag_step_reach.py \
      Misc/runs/diag_cooperative_arms/
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import pinocchio as pin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from crawlbot.core.ik import (  # noqa: E402
    _get_tool_frames, _arm_v_slice, _interpolate_q_quintic,
)

URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')

R_MAX = 1.575  # m — sum of link offsets along Joint_1 → tool chain
K = 5          # quintic interior samples per step (matches IK 3 default)


def _extension_and_sigma(model, data, q, fid_a, fid_b,
                         jid_sh_a, jid_sh_b, sl_a, sl_b):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pin.computeJointJacobians(model, data, q)
    p_sh_a = data.oMi[jid_sh_a].translation
    p_sh_b = data.oMi[jid_sh_b].translation
    p_t_a = data.oMf[fid_a].translation
    p_t_b = data.oMf[fid_b].translation
    ext_a = float(np.linalg.norm(p_t_a - p_sh_a) / R_MAX)
    ext_b = float(np.linalg.norm(p_t_b - p_sh_b) / R_MAX)
    Ja = pin.getFrameJacobian(model, data, fid_a, pin.LOCAL)[:, sl_a]
    Jb = pin.getFrameJacobian(model, data, fid_b, pin.LOCAL)[:, sl_b]
    sa = float(np.linalg.svd(Ja, compute_uv=False)[-1])
    sb = float(np.linalg.svd(Jb, compute_uv=False)[-1])
    return ext_a, ext_b, sa, sb


def main(results_dir: str):
    q_log_path = os.path.join(results_dir, 'step_q_end.json')
    ik_trace_path = os.path.join(results_dir, 'ik_trace.json')
    with open(q_log_path) as f:
        q_log = json.load(f)
    with open(ik_trace_path) as f:
        ik_trace = json.load(f)

    model = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
    data = model.createData()
    fid_a, fid_b = _get_tool_frames(model)
    sl_a = _arm_v_slice(model, fid_a)
    sl_b = _arm_v_slice(model, fid_b)
    jid_sh_a = model.getJointId("Joint_1_a")
    jid_sh_b = model.getJointId("Joint_1_b")

    rows = []
    for i, entry in enumerate(q_log):
        q_start = np.array(entry['q_start'])
        q_end = np.array(entry['q_end'])
        step_idx = entry['step_idx']
        anchor_pair = ik_trace[i].get('step_ab_end') if i < len(ik_trace) else None
        for k in range(K + 1):
            tau = k / K
            if tau == 0.0:
                q_k = q_start
            else:
                q_k = _interpolate_q_quintic(model, q_start, q_end, tau)
            ext_a, ext_b, sa, sb = _extension_and_sigma(
                model, data, q_k, fid_a, fid_b,
                jid_sh_a, jid_sh_b, sl_a, sl_b)
            rows.append({
                'step_idx': step_idx,
                'anchor_pair': str(anchor_pair) if anchor_pair else '?',
                'tau': tau,
                'ext_a': ext_a,
                'ext_b': ext_b,
                'sig_a': sa,
                'sig_b': sb,
            })

    csv_path = os.path.join(results_dir, 'step_reach.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'step_idx', 'anchor_pair', 'tau',
            'ext_a', 'ext_b', 'sig_a', 'sig_b'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    n_steps = max(r['step_idx'] for r in rows) + 1
    cmap_arr = plt.colormaps.get_cmap('viridis')(np.linspace(0, 0.9, n_steps))
    for arm_idx, (arm_label, ext_key, sig_key) in enumerate([
            ('Arm A', 'ext_a', 'sig_a'),
            ('Arm B', 'ext_b', 'sig_b')]):
        ax = axes[arm_idx]
        for step in range(n_steps):
            step_rows = [r for r in rows if r['step_idx'] == step]
            xs = [r[ext_key] for r in step_rows]
            ys = [r[sig_key] for r in step_rows]
            ax.plot(xs, ys, '-o', color=cmap_arr[step],
                    label=f'step {step}', markersize=5)
        ax.set_xlabel('Extension ratio  ||tool-shoulder|| / R_max')
        if arm_idx == 0:
            ax.set_ylabel(r'$\sigma_{\min}(\hat J)$  [arm-internal]')
        ax.set_title(arm_label)
        ax.set_yscale('log')
        ax.axhline(1e-4, color='red', linestyle='--', alpha=0.5,
                   label=r'$\sigma_{\min}=10^{-4}$')
        ax.axvline(0.85, color='orange', linestyle='--', alpha=0.5,
                   label='ext=0.85')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8, ncol=2, loc='lower left')
    fig.suptitle(f'Per-step arm reach vs σ_min  (K={K} samples)\n{results_dir}')
    fig.tight_layout()
    png_path = os.path.join(results_dir, 'step_reach.png')
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {png_path}")

    print('\nPer-step summary (peak extension and min σ_min across τ samples):')
    print(f"{'step':>4s} {'pair':>10s} "
          f"{'peak_ext_a':>11s} {'peak_ext_b':>11s} "
          f"{'min_sig_a':>11s} {'min_sig_b':>11s}")
    for step in range(n_steps):
        step_rows = [r for r in rows if r['step_idx'] == step]
        pair = step_rows[0]['anchor_pair'] if step_rows else '?'
        pe_a = max(r['ext_a'] for r in step_rows)
        pe_b = max(r['ext_b'] for r in step_rows)
        ms_a = min(r['sig_a'] for r in step_rows)
        ms_b = min(r['sig_b'] for r in step_rows)
        print(f"{step:>4d} {pair:>10s} "
              f"{pe_a:>11.3f} {pe_b:>11.3f} "
              f"{ms_a:>11.2e} {ms_b:>11.2e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir',
                        help='Path to results dir containing '
                             'step_q_end.json + ik_trace.json')
    args = parser.parse_args()
    main(args.results_dir)
