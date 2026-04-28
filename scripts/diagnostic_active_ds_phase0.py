"""Phase 0 pre-flight: q_target_DS feasibility on the double-stance manifold.

Per docs/architecture/active_ds_torso_advance.md §6 Phase 0, the
gate before any sim_loop edits land. For each T15 step k (0, 1, 2):

  1. Determine the CURRENT stance anchor pair (anchor_a[i_a],
     anchor_b[i_b]) — the pair both arms are welded to during DS_k
     before the upcoming SS_k.
  2. Compute q_DS_start ≈ q_dock_init for the current pair (clean
     post-dock state).
  3. Compute q_SS_end via manipulability_config_trajectory for the
     upcoming SS_k anchor target.
  4. For β ∈ {0.3, 0.5, 0.7, 0.9}:
       q_target_DS_unconstrained = pin.interpolate(q_DS_start,
                                                    q_SS_end, β)
       q_target_DS_proj = solve_ik(q_target_DS_unconstrained,
                                   targets={fid_a: anchor_a[i_a],
                                            fid_b: anchor_b[i_b]})
     Report: 2-task IK convergence, stance residuals (both arms),
     world-frame torso advance, joint-space distance.

Pass criterion (gate): for β = 0.7, the 2-task IK converges to
both stance residuals ≤ 50 μm on all 3 T15 steps.

Pass → proceed with Phase 1 (double_stance_planner.py).
Fail → lower β until 2-task IK converges; if even β=0.3 fails,
the architecture must be reconsidered.

Outputs:
  results/diagnostic/active_ds_phase0/
    step{0,1,2}_data.json              per-β raw data
    all_steps_torso_advance.png        torso advance vs β
    summary.txt                        per-step verdict + gate

Run:
  PYTHONPATH=. MUJOCO_GL=disabled python3 \\
      scripts/diagnostic_active_ds_phase0.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.environ.setdefault('MUJOCO_GL', 'disabled')

URDF = os.path.join(_ROOT, 'models', 'VISPA_crawling_fixed.urdf')
MJCF = os.path.join(_ROOT, 'models', 'VISPA_crawling_rwa3.xml')
OUTDIR = os.path.join(_ROOT, 'results', 'diagnostic', 'active_ds_phase0')

# Pass criterion per active_ds_torso_advance.md §6 Phase 0.
GATE_BETA = 0.7
GATE_STANCE_TOL_M = 5e-5  # 50 μm
GATE_BETAS = [0.3, 0.5, 0.7, 0.9]


# T15 traversal anchor sequence (start (2, 2), 3 steps):
#   step 0: swing 'b', stance pair stays at (a=2, b=2) during DS_0
#           → SS_0 ends at (a=2, b=3)
#   step 1: swing 'a', stance pair (a=2, b=3) during DS_1
#           → SS_1 ends at (a=3, b=3)
#   step 2: swing 'b', stance pair (a=3, b=3) during DS_2
#           → SS_2 ends at (a=3, b=4)
T15_STEPS = [
    {'idx': 0, 'swing': 'b', 'ds_pair': (2, 2), 'ss_target_pair': (2, 3)},
    {'idx': 1, 'swing': 'a', 'ds_pair': (2, 3), 'ss_target_pair': (3, 3)},
    {'idx': 2, 'swing': 'b', 'ds_pair': (3, 3), 'ss_target_pair': (3, 4)},
]


def _load_robot_and_anchors():
    from crawlbot.core.robot_interface import RobotInterface
    from crawlbot.core.ik import _get_tool_frames
    import mujoco
    from crawlbot.planning.contact_scheduler import read_anchors_from_mujoco

    robot = RobotInterface(URDF, tau_max=20.0, gravity='zero')
    fid_a, fid_b = _get_tool_frames(robot.model)
    fid_torso = robot.frame_torso

    mj_model = mujoco.MjModel.from_xml_path(MJCF)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    a_world, b_world = read_anchors_from_mujoco(mj_model, mj_data)

    # Transform anchors to initial-structure-local (= Pinocchio world).
    p_s0 = mj_data.qpos[0:3].copy()
    qw, qx, qy, qz = mj_data.qpos[3:7]
    R_s0 = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
    a_local = [R_s0.T @ (a - p_s0) for a in a_world]
    b_local = [R_s0.T @ (b - p_s0) for b in b_world]

    return {
        'robot': robot, 'fid_a': fid_a, 'fid_b': fid_b, 'fid_torso': fid_torso,
        'a_local': a_local, 'b_local': b_local,
    }


def _q_for_pair(rb, i_a, i_b):
    """q at the (a@i_a, b@i_b) docked pair via dock_configuration."""
    from crawlbot.core.ik import dock_configuration
    se3_a = pin.SE3(np.eye(3), rb['a_local'][i_a])
    se3_b = pin.SE3(np.eye(3), rb['b_local'][i_b])
    q = dock_configuration(rb['robot'].model, se3_a, se3_b)
    return q


def _compute_ss_end(rb, q_ds_start, ss_target_pair):
    """q_SS_end via manipulability_config_trajectory with the new pair."""
    from crawlbot.core.ik import manipulability_config_trajectory
    i_a, i_b = ss_target_pair
    q_end, w_w, w_e = manipulability_config_trajectory(
        rb['robot'].model,
        rb['a_local'][i_a],
        rb['b_local'][i_b],
        q_start=q_ds_start,
        n_samples=5,
    )
    return q_end, w_e


def _project_to_double_stance(rb, q_seed, ds_pair):
    """2-task IK pinning both stance arms at their CURRENT anchors.
    Uses solve_ik from crawlbot.core.ik."""
    from crawlbot.core.ik import solve_ik
    i_a, i_b = ds_pair
    targets = {
        rb['fid_a']: pin.SE3(np.eye(3), rb['a_local'][i_a]),
        rb['fid_b']: pin.SE3(np.eye(3), rb['b_local'][i_b]),
    }
    q_proj, err = solve_ik(rb['robot'].model, q_seed, targets,
                           max_iter=500)
    return q_proj, err


def _stance_residuals(model, q, fid_a, fid_b, p_anchor_a, p_anchor_b):
    """Return (||FK[fid_a] - anchor_a||, ||FK[fid_b] - anchor_b||)."""
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    da = float(np.linalg.norm(data.oMf[fid_a].translation - p_anchor_a))
    db = float(np.linalg.norm(data.oMf[fid_b].translation - p_anchor_b))
    return da, db


def _torso_pos(model, q, fid_torso):
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[fid_torso].translation.copy()


def analyse_step(rb, step_info):
    idx = step_info['idx']
    ds_pair = step_info['ds_pair']
    ss_target_pair = step_info['ss_target_pair']
    print(f"\n=== Step {idx}: DS pair {ds_pair}, SS target pair {ss_target_pair} ===")

    # 1. q_DS_start ≈ docked state at the current stance pair.
    q_ds_start = _q_for_pair(rb, ds_pair[0], ds_pair[1])
    p_torso_start = _torso_pos(rb['robot'].model, q_ds_start, rb['fid_torso'])
    print(f"  q_DS_start: torso world pos = {p_torso_start}")

    # 2. q_SS_end via IK for the upcoming step.
    print(f"  Running manipulability_config_trajectory for SS_{idx}...")
    q_ss_end, w_e = _compute_ss_end(rb, q_ds_start, ss_target_pair)
    if q_ss_end is None:
        print(f"  IK rejected (w_end below threshold). Skipping step.")
        return None
    p_torso_end = _torso_pos(rb['robot'].model, q_ss_end, rb['fid_torso'])
    p_advance = p_torso_end - p_torso_start
    print(f"  q_SS_end: torso world pos = {p_torso_end}, w_e = {w_e:.3e}")
    print(f"  Total body translation needed (SS): {np.linalg.norm(p_advance)*1000:.1f} mm")

    # 3. Sweep β.
    p_anchor_a_pin = rb['a_local'][ds_pair[0]]
    p_anchor_b_pin = rb['b_local'][ds_pair[1]]
    rows = []
    for beta in GATE_BETAS:
        q_unconstrained = pin.interpolate(rb['robot'].model, q_ds_start,
                                          q_ss_end, float(beta))
        q_proj, err = _project_to_double_stance(rb, q_unconstrained, ds_pair)
        ra, rb_dev = _stance_residuals(rb['robot'].model, q_proj,
                                       rb['fid_a'], rb['fid_b'],
                                       p_anchor_a_pin, p_anchor_b_pin)
        p_torso_proj = _torso_pos(rb['robot'].model, q_proj, rb['fid_torso'])
        torso_advance_world = float(np.linalg.norm(p_torso_proj - p_torso_start))
        torso_advance_frac = torso_advance_world / max(np.linalg.norm(p_advance), 1e-9)
        # Joint-space distance from q_DS_start
        dq_norm = float(np.linalg.norm(
            pin.difference(rb['robot'].model, q_ds_start, q_proj)))
        gate_pass = (ra < GATE_STANCE_TOL_M) and (rb_dev < GATE_STANCE_TOL_M)
        rows.append({
            'beta': float(beta),
            'ik_residual': float(err),
            'stance_a_residual_m': ra,
            'stance_b_residual_m': rb_dev,
            'torso_advance_world_mm': torso_advance_world * 1000,
            'torso_advance_fraction_of_ss_total': float(torso_advance_frac),
            'dq_joint_space_norm_rad': dq_norm,
            'p_torso_proj': p_torso_proj.tolist(),
            'gate_pass': gate_pass,
        })
        print(f"    β={beta:.2f}:  IK err = {err:.2e},  "
              f"stance_a = {ra*1e6:7.1f} μm,  stance_b = {rb_dev*1e6:7.1f} μm,  "
              f"torso advance = {torso_advance_world*1000:6.1f} mm "
              f"({torso_advance_frac*100:4.1f}% of SS-total)  "
              f"{'PASS' if gate_pass else 'FAIL'}")

    return {
        'step_idx': idx,
        'ds_pair': list(ds_pair),
        'ss_target_pair': list(ss_target_pair),
        'q_DS_start': q_ds_start.tolist(),
        'q_SS_end': q_ss_end.tolist(),
        'p_torso_start': p_torso_start.tolist(),
        'p_torso_end': p_torso_end.tolist(),
        'ss_total_torso_advance_mm': float(np.linalg.norm(p_advance) * 1000),
        'beta_sweep': rows,
    }


def _plot_torso_advance(per_step, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ['C0', 'C1', 'C2']
    for d, c in zip(per_step, colors):
        if d is None:
            continue
        betas = [r['beta'] for r in d['beta_sweep']]
        adv_mm = [r['torso_advance_world_mm'] for r in d['beta_sweep']]
        adv_frac = [r['torso_advance_fraction_of_ss_total'] * 100
                    for r in d['beta_sweep']]
        ds_p, ss_t = tuple(d['ds_pair']), tuple(d['ss_target_pair'])
        label = f"step {d['step_idx']} (DS pair {ds_p} → SS {ss_t})"
        axes[0].plot(betas, adv_mm, '-o', color=c, label=label)
        axes[1].plot(betas, adv_frac, '-o', color=c, label=label)
    axes[0].set_title('DS-end torso advance (world frame)')
    axes[0].set_xlabel('β')
    axes[0].set_ylabel('mm')
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].axhline(70, color='red', ls='--', lw=1, label='target 70%')
    axes[1].set_title('DS-end torso advance / SS-total torso advance')
    axes[1].set_xlabel('β')
    axes[1].set_ylabel('%')
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    print(f"  -> {out_path}")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rb = _load_robot_and_anchors()
    print(f"Model: nq={rb['robot'].model.nq}, nv={rb['robot'].model.nv}")
    print(f"  fid_a={rb['fid_a']}, fid_b={rb['fid_b']}, "
          f"fid_torso={rb['fid_torso']}")

    per_step = []
    for step_info in T15_STEPS:
        d = analyse_step(rb, step_info)
        per_step.append(d)
        if d is not None:
            with open(os.path.join(OUTDIR, f"step{d['step_idx']}_data.json"), 'w') as fp:
                json.dump(d, fp, indent=2)

    _plot_torso_advance(per_step, os.path.join(OUTDIR,
                                               'all_steps_torso_advance.png'))

    # Gate verdict — at β = GATE_BETA, all 3 steps must pass.
    summary = []
    summary.append("Phase-0 pre-flight: q_target_DS feasibility on M_double")
    summary.append("=" * 72)
    summary.append(f"  Gate: β = {GATE_BETA}, both stance residuals ≤ "
                   f"{GATE_STANCE_TOL_M*1e6:.0f} μm")
    summary.append("")
    overall_pass = True
    for d in per_step:
        if d is None:
            summary.append(f"  step ?: SKIP (IK failed)")
            overall_pass = False
            continue
        row = next(r for r in d['beta_sweep'] if r['beta'] == GATE_BETA)
        ok = row['gate_pass']
        overall_pass &= ok
        summary.append(
            f"  step {d['step_idx']} (DS pair {tuple(d['ds_pair'])}, "
            f"SS target {tuple(d['ss_target_pair'])}): "
            f"stance_a = {row['stance_a_residual_m']*1e6:6.1f} μm, "
            f"stance_b = {row['stance_b_residual_m']*1e6:6.1f} μm, "
            f"torso advance = {row['torso_advance_world_mm']:6.1f} mm "
            f"[{'PASS' if ok else 'FAIL'}]")
    summary.append("")
    if overall_pass:
        summary.append(f"VERDICT: gate PASS at β = {GATE_BETA}.")
        summary.append(f"         Proceed with Phase 1 implementation.")
    else:
        # Find largest passing β.
        max_beta_passing = {}
        for d in per_step:
            if d is None:
                continue
            for row in d['beta_sweep']:
                if row['gate_pass']:
                    max_beta_passing[d['step_idx']] = max(
                        max_beta_passing.get(d['step_idx'], 0.0), row['beta'])
        summary.append(f"VERDICT: gate FAIL at β = {GATE_BETA}.")
        summary.append(f"         Max passing β per step: {max_beta_passing}")
        summary.append(f"         Re-tune β before Phase 1, or reconsider.")
    summary_text = "\n".join(summary)
    print()
    print(summary_text)
    with open(os.path.join(OUTDIR, 'summary.txt'), 'w') as fp:
        fp.write(summary_text + "\n")
    return 0 if overall_pass else 1


if __name__ == '__main__':
    sys.exit(main())
