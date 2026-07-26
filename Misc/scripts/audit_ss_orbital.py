#!/usr/bin/env python3
"""J2 Task 3 — SS orbital scaling law (offline, for the paper).

Characterizes how the planned orbital term ‖r_com×Σf‖ (the part of the exact
Ḣ_s the proxy omits) grows as the robot crawls outward (r_com increasing), per
SS phase. Fits orbital vs r_com, reports the Σf trend, and extrapolates the
crawl distance at which the orbital term alone hits the per-axis envelope
τ_w_max=5. Uses the PLANNED wrench (lambda_ref) — real SS motion.

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/audit_ss_orbital.py results/<run>
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
TAU_W_MAX = 5.0


def anchors(struct_pos0, struct_quat0):
    m = mujoco.MjModel.from_xml_path(MJCF); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    aw, bw = [], []
    for i in range(1, 20):
        sa = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f'anchor_{i}a')
        sb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f'anchor_{i}b')
        if sa < 0 or sb < 0:
            break
        aw.append(d.site_xpos[sa].copy()); bw.append(d.site_xpos[sb].copy())
    R0 = pin.Quaternion(*struct_quat0).toRotationMatrix(); p0 = np.asarray(struct_pos0, float)
    return [R0.T @ (a - p0) for a in aw], [R0.T @ (b - p0) for b in bw]


def main(run_dir):
    sl = json.load(open(os.path.join(run_dir, 'sim_log.json')))
    pp = list(csv.DictReader(open(os.path.join(run_dir, 'postproc_F3F4.csv'))))
    lr = np.asarray(sl['lambda_ref'], float)
    r_com = np.asarray(sl['r_com'], float)
    ph = np.asarray(sl['phase']); sidx = np.asarray(sl['step_idx'], int)
    sa = np.array([int(r['stance_a_idx']) for r in pp])
    sb = np.array([int(r['stance_b_idx']) for r in pp])
    A, B = anchors(np.asarray(sl['struct_pos'], float)[0], np.asarray(sl['struct_quat'], float)[0])

    print('=' * 78)
    print(f'SS ORBITAL SCALING — {run_dir}  (planned wrench lambda_ref)')
    print('=' * 78)
    print(f"  {'SS step':8s} {'|r_com|mean':>11s} {'‖Σf‖med':>8s} {'orbital‖·‖med':>13s} "
          f"{'proxy‖Ḣ‖med':>12s} {'exact‖Ḣ‖med':>12s} {'orb/|rcom|':>10s}")
    rc_list, orb_list, sf_list = [], [], []
    for step in range(0, 5):
        m = (ph == 'SS') & (sidx == step)
        idx = np.where(m)[0]
        if len(idx) < 3:
            continue
        fsum = lr[idx, 0:3] + lr[idx, 6:9]                 # Σf (planned, stance contact)
        orb = np.cross(r_com[idx], fsum)                    # orbital r_com×Σf
        H = np.array([np.cross(A[sa[k]], lr[k, 0:3]) + lr[k, 3:6]
                      + np.cross(B[sb[k]], lr[k, 6:9]) + lr[k, 9:12] for k in idx])
        proxy = H - orb
        rc = np.median(np.linalg.norm(r_com[idx], axis=1))
        sf = np.median(np.linalg.norm(fsum, axis=1))
        ob = np.median(np.linalg.norm(orb, axis=1))
        rc_list.append(rc); orb_list.append(ob); sf_list.append(sf)
        print(f"  step{step:<4d} {rc:11.3f} {sf:8.3f} {ob:13.3f} "
              f"{np.median(np.linalg.norm(proxy,axis=1)):12.3f} "
              f"{np.median(np.linalg.norm(H,axis=1)):12.3f} {ob/rc if rc>0 else 0:10.3f}")

    rc_a = np.array(rc_list); orb_a = np.array(orb_list); sf_a = np.array(sf_list)
    print('\n  SCALING LAW (orbital ‖r_com×Σf‖ vs |r_com| across SS phases):')
    if len(rc_a) >= 2:
        # linear fit through origin (orbital ≈ k·r_com) and free fit.
        k0 = float((rc_a @ orb_a) / (rc_a @ rc_a))           # through-origin slope
        A_fit = np.vstack([rc_a, np.ones_like(rc_a)]).T
        slope, icpt = np.linalg.lstsq(A_fit, orb_a, rcond=None)[0]
        print(f"    through-origin slope k = {k0:.3f} N·m per m  (orbital ≈ k·|r_com|; k≈mean|Σf|={sf_a.mean():.2f} N)")
        print(f"    free fit: orbital ≈ {slope:.3f}·|r_com| + {icpt:.3f}")
        print(f"    Σf trend: |Σf| over SS steps = {np.round(sf_a,2)} N "
              f"({'decreasing' if sf_a[-1] < sf_a[0] else 'increasing/flat'} as r_com grows "
              f"{rc_a[0]:.2f}→{rc_a[-1]:.2f} m)")
        # extrapolation: r_com at which orbital alone = τ_w_max, at the mean Σf.
        rc_cap = TAU_W_MAX / k0 if k0 > 0 else float('inf')
        print(f"    EXTRAPOLATION: orbital hits ±{TAU_W_MAX} N·m at |r_com| ≈ {rc_cap:.2f} m "
              f"(for the representative swing |Σf|≈{sf_a.mean():.2f} N). "
              f"Beyond that crawl distance the orbital lever alone saturates the wheel envelope.")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else 'results/chat_e0'))
