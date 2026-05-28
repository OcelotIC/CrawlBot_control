#!/usr/bin/env python3
"""Probe 2 — constrained-dynamics conditioning along the trajectory.

Probe 1 showed the tracked reference is SMOOTH at the f1 peak (Δp_torso_ref
= 3.5 mm, 35th pct) and the 62 N is a single-tick spike with v_com still
on-plan. That rules out "QP chases a jerky reference" and points at a
momentary constrained-dynamics ill-conditioning (spec §3 constrained
dynamic singularity: stance arm near full extension → poor transmission
of torso/CoM motion to the platform → QP needs a large contact force /
loses EE null-space freedom).

Question for the CMM-interface decision: does a centroidal-momentum QP
task hit the SAME §3 wall the loop-free mapping did? The CMM-CoM task
lives in the contact-constrained space, so its controllability is
σ_min( J_com · N_c ), N_c = null(stance-weld Jacobian). The upstream
driver is the stance-arm kinematic conditioning σ_min(J_stance 6×7).

We evaluate at the logged snapshots (qpos/qvel at phase boundaries +
mid-SS frames). The step-2 f1 peak (t=16.63) is bracketed by
frame_step2_2 (15.93) and frame_step2_3 (16.83). Loop-free died at step
0, so step-0 frames/dock are the comparison.

Falsifiers:
 - if σ_min dips sharply at the step-2 peak bracket AND at step-0 →
   §3 conditioning is implicated even in the working run; a CMM task
   would inherit it → CMM needs §6 mitigation.
 - if σ_min is healthy throughout → the spike is NOT a §3 singularity;
   a CMM task would be well-conditioned (and the spike is some other
   transient — EE-task / numerical).

Reads:  results/diag_cooperative_arms/sim_log.json (snapshots)
        models/VISPA_crawling_fixed.urdf
Writes: results/diag_attribution/probe2_cmm_conditioning/{metrics.json, probe2.png}
"""
import os
import sys
import json

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from crawlbot.core.robot_interface import RobotInterface
from crawlbot.core.state_conversions import mujoco_to_pinocchio

LOG = os.path.join(_root, 'results', 'diag_cooperative_arms', 'sim_log.json')
URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')
OUT = os.path.join(_root, 'results', 'diag_attribution',
                   'probe2_cmm_conditioning')

# swing arm per step (from the log): 0=b,1=a,2=b,3=a,4=b → stance is the other
SWING = {0: 'b', 1: 'a', 2: 'b', 3: 'a', 4: 'b'}


def step_of_label(lbl):
    for s in range(5):
        if f'step{s}' in lbl:
            return s
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    robot = RobotInterface(URDF, gravity='zero')
    log = json.load(open(LOG))
    snaps = log['snapshots']

    rows = []
    for (tt, qpos, qvel, lbl) in snaps:
        s = step_of_label(lbl)
        if s is None:
            continue
        stance = 'a' if SWING[s] == 'b' else 'b'
        q, v = mujoco_to_pinocchio(np.array(qpos), np.array(qvel))
        rs = robot.update(q, v)
        J_com = rs.J_com                       # (3, nv)
        J_c = rs.J_tool_a if stance == 'a' else rs.J_tool_b  # (6, nv) stance weld
        st_slice = (robot.arm_a_v_slice if stance == 'a'
                    else robot.arm_b_v_slice)

        # stance-arm kinematic conditioning (tool 6D vs stance 7 joints)
        J_stance_arm = J_c[:, st_slice]        # (6, 7)
        sv_arm = np.linalg.svd(J_stance_arm, compute_uv=False)
        # constrained-CoM controllability: CoM accel reachable within
        # the weld-constrained accel space N_c = null(J_c)
        N_c = np.eye(robot.model.nv) - np.linalg.pinv(J_c, rcond=1e-10) @ J_c
        J_com_c = J_com @ N_c                   # (3, nv)
        sv_comc = np.linalg.svd(J_com_c, compute_uv=False)

        rows.append({
            't': float(tt), 'step': s, 'label': lbl, 'stance': stance,
            'sigma_min_stance_arm': float(sv_arm.min()),
            'sigma_min_stance_arm_over_max': float(sv_arm.min() / sv_arm.max()),
            'sigma_min_com_constrained': float(sv_comc[sv_comc > 1e-12].min()
                                               if np.any(sv_comc > 1e-12) else 0.0),
            'sigma_max_com_constrained': float(sv_comc.max()),
        })

    with open(os.path.join(OUT, 'metrics.json'), 'w') as fh:
        json.dump(rows, fh, indent=2)

    # ---- stdout table ----
    print('=' * 92)
    print(' Probe 2 — constrained-dynamics conditioning at snapshots')
    print('=' * 92)
    print(f"{'t':>6} {'step':>4} {'label':<16} {'stance':>6} "
          f"{'σmin(arm)':>10} {'σmin/σmax':>10} {'σmin(CoM|c)':>12} {'σmax(CoM|c)':>12}")
    print('-' * 92)
    for r in rows:
        mark = ''
        if r['step'] == 2 and ('frame_step2_2' in r['label']
                               or 'frame_step2_3' in r['label']):
            mark = '  <-- brackets f1 peak (t=16.63)'
        if r['step'] == 0:
            mark = mark or '  (loop-free died here)'
        print(f"{r['t']:>6.2f} {r['step']:>4} {r['label']:<16} {r['stance']:>6} "
              f"{r['sigma_min_stance_arm']:>10.4f} "
              f"{r['sigma_min_stance_arm_over_max']:>10.4f} "
              f"{r['sigma_min_com_constrained']:>12.4f} "
              f"{r['sigma_max_com_constrained']:>12.4f}{mark}")

    # ---- summary: is the step-2 peak bracket / step-0 worse than the rest? ----
    def grp(pred):
        a = [r['sigma_min_stance_arm'] for r in rows if pred(r)]
        b = [r['sigma_min_com_constrained'] for r in rows if pred(r)]
        return (np.mean(a) if a else np.nan, np.mean(b) if b else np.nan)
    peak_arm, peak_com = grp(lambda r: 'frame_step2_2' in r['label']
                             or 'frame_step2_3' in r['label'])
    s0_arm, s0_com = grp(lambda r: r['step'] == 0)
    all_arm, all_com = grp(lambda r: True)
    print('-' * 92)
    print(f"  mean σmin(stance arm):  step2-peak-bracket={peak_arm:.4f}  "
          f"step0={s0_arm:.4f}  all={all_arm:.4f}")
    print(f"  mean σmin(CoM|constrained): step2-peak-bracket={peak_com:.4f}  "
          f"step0={s0_com:.4f}  all={all_com:.4f}")

    # ---- plot σ_min trajectories vs time ----
    ts = np.array([r['t'] for r in rows])
    arm = np.array([r['sigma_min_stance_arm'] for r in rows])
    comc = np.array([r['sigma_min_com_constrained'] for r in rows])
    order = np.argsort(ts)
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(ts[order], arm[order], 'o-', lw=1.5)
    ax[0].axvline(16.63, color='r', ls=':', label='f1 peak t=16.63')
    ax[0].set_ylabel('σ_min(stance arm 6×7)')
    ax[0].set_title('Constrained-dynamics conditioning along traversal '
                    '(low σ_min ⇒ §3 singularity risk)')
    ax[0].legend(fontsize=8)
    ax[1].plot(ts[order], comc[order], 'o-', lw=1.5, color='C1')
    ax[1].axvline(16.63, color='r', ls=':')
    ax[1].set_ylabel('σ_min(CoM | constrained)')
    ax[1].set_xlabel('t [s]')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'probe2.png'), dpi=110)
    print(f"\n  saved: {os.path.relpath(OUT, _root)}/{{metrics.json, probe2.png}}")


if __name__ == '__main__':
    main()
