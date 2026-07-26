#!/usr/bin/env python3
"""Tick-by-tick closure-curve dump for the LOCKED run (CC closure brief).

Focused validation curves for STOP-GATES C & D (NOT the full figure export):
  Signal 1 — chatter dead:  exact Ḣ_s (per axis) + commanded τ_w (per axis), full cycle.
  Signal 2 — residual bounded:  ‖dq_joint‖ + T_kin per tick within every inter-step settle.
  Signal 3 — conservation:  ‖L_com‖ + ‖hw‖ per tick, full traversal.

Ḣ_s is exact origin-referenced Σ_j(r_Cj×f_j+τ_j), reconstructed in the structure frame from the
per-tick stance anchors (postproc_F3F4.csv) — PLANNED (lambda_ref) on SS/_step ticks, REALIZED
(lambda_qp) on inter-step DS ticks (lambda_ref is absent there), tagged by Hdot_s_source.

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/dump_closure_curves.py <run_dir> <out.csv>
"""
import csv
import json
import os
import sys
import numpy as np
import mujoco
import pinocchio as pin

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MJCF = os.path.join(ROOT, 'models', 'VISPA_crawling_rwa3.xml')


def anchors_struct_frame(sp0, sq0):
    m = mujoco.MjModel.from_xml_path(MJCF)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    aw, bw = [], []
    for i in range(1, 20):
        sa = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f'anchor_{i}a')
        sb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f'anchor_{i}b')
        if sa < 0 or sb < 0:
            break
        aw.append(d.site_xpos[sa].copy())
        bw.append(d.site_xpos[sb].copy())
    qw, qx, qy, qz = sq0
    R0 = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
    p0 = np.asarray(sp0, float)
    return ([R0.T @ (a - p0) for a in aw], [R0.T @ (b - p0) for b in bw])


def hdot_exact(rA, rB, L):
    return np.cross(rA, L[0:3]) + L[3:6] + np.cross(rB, L[6:9]) + L[9:12]


def main(run_dir, out_csv):
    sl = json.load(open(os.path.join(run_dir, 'sim_log.json')))
    pp = list(csv.DictReader(open(os.path.join(run_dir, 'postproc_F3F4.csv'))))
    t = np.asarray(sl['t'], float)
    ph = np.asarray(sl['phase'])
    step = np.asarray(sl['step_idx'], int)
    lq = np.asarray(sl['lambda_qp'], float)
    lr = np.asarray(sl['lambda_ref'], float)
    tauw = np.asarray(sl['tau_w'], float)
    L = np.asarray(sl['L_com'], float)
    hw = np.asarray(sl['hw'], float)
    qa = np.asarray(sl['qvel_joints_a'], float)
    qb = np.asarray(sl['qvel_joints_b'], float)
    Tkin = np.asarray(sl['T_kinetic'], float) if 'T_kinetic' in sl else None
    sa_i = np.array([int(r['stance_a_idx']) for r in pp])
    sb_i = np.array([int(r['stance_b_idx']) for r in pp])
    n = min(len(t), len(sa_i))
    anchors_a, anchors_b = anchors_struct_frame(
        np.asarray(sl['struct_pos'], float)[0], np.asarray(sl['struct_quat'], float)[0])

    # settle-exit ticks: last logged tick within each inter_step_settle window
    exit_ticks = set()
    for s in (sl.get('inter_step_settles', []) or []):
        if s['n_steps'] < 12:
            continue
        idx = np.where((ph == 'DS') & (t > s['t_start'] + 1e-6)
                       & (t <= s['t_end'] + 1e-6))[0]
        if len(idx):
            exit_ticks.add(int(idx[-1]))

    rows = []
    for k in range(n):
        is_ds = (ph[k] == 'DS')
        src = 'realized' if is_ds else 'planned'
        lam = lq[k] if is_ds else lr[k]
        H = hdot_exact(anchors_a[sa_i[k]], anchors_b[sb_i[k]], lam)
        dqj = float(np.linalg.norm(np.concatenate([qa[k], qb[k]])))
        rows.append(dict(
            tick=k, t_s=round(float(t[k]), 4), phase=str(ph[k]), step_index=int(step[k]),
            Hdot_s_x=round(float(H[0]), 5), Hdot_s_y=round(float(H[1]), 5),
            Hdot_s_z=round(float(H[2]), 5), Hdot_s_source=src,
            tau_w_x=round(float(tauw[k, 0]), 5), tau_w_y=round(float(tauw[k, 1]), 5),
            tau_w_z=round(float(tauw[k, 2]), 5),
            dq_joint_norm=round(dqj, 6),
            T_kin=(round(float(Tkin[k]), 9) if Tkin is not None else ''),
            L_com_norm=round(float(np.linalg.norm(L[k])), 6),
            hw_norm=round(float(np.linalg.norm(hw[k])), 6),
            settle_exit=int(k in exit_ticks)))
    cols = ['tick', 't_s', 'phase', 'step_index', 'Hdot_s_x', 'Hdot_s_y', 'Hdot_s_z',
            'Hdot_s_source', 'tau_w_x', 'tau_w_y', 'tau_w_z', 'dq_joint_norm', 'T_kin',
            'L_com_norm', 'hw_norm', 'settle_exit']
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {out_csv}: {len(rows)} ticks')
    # quick console sanity: peak |Hdot_s| in inter-step DS vs SS, τ_w tick-to-tick max-jump in DS
    ds = np.array([r['phase'] == 'DS' for r in rows])
    Hn = np.array([[r['Hdot_s_x'], r['Hdot_s_y'], r['Hdot_s_z']] for r in rows])
    tw = np.array([[r['tau_w_x'], r['tau_w_y'], r['tau_w_z']] for r in rows])
    dtw = np.linalg.norm(np.diff(tw, axis=0), axis=1)
    print(f'  inter-step DS: max|Hdot_s|={np.abs(Hn[ds]).max():.3f}  '
          f'max tick-to-tick |Δτ_w| (DS) ={dtw[ds[1:]].max():.3f}')
    print(f'  settle-exit ticks: {sorted(exit_ticks)}')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
