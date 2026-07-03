#!/usr/bin/env python3
"""Phase 1i Task A — trace the configuration-dependent arm contact-force capacity.
READ-ONLY static kinematics (no sim): loads the committed URDF + the 6 committed docked
configs, computes the stance-arm contact Jacobian at each, and maps the arm joint-torque
limit (MJCF ctrlrange) through it to the admissible contact force.

Static contact-force relation: to hold contact force f at the stance tool, the stance arm
joints carry tau = J_lin^T f, so |(J_lin^T f)_i| <= tau_max per joint. Hence:
  - along a unit direction d:      f_max(d) = tau_max / max_i |(J_lin^T d)_i|
  - worst-case (min over all d):   f_worst  = tau_max / max_i ||row_i(J_lin)||_2
  - manipulability:                singular values of J_lin (sigma_min -> singularity).

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_1i_arm_capacity.py
"""
import json

import numpy as np
import pinocchio as pin

from crawlbot.core.robot_interface import RobotInterface

URDF = 'models/VISPA_crawling_fixed.urdf'
QEND = 'results/figC_qpcond/step_q_end.json'
SIM = 'results/figC_qpcond/sim_log.json'
TAU_PHYS = 50.0   # MJCF ctrlrange +/-50 Nm per arm joint (models/VISPA_crawling_rwa3.xml:304-318)
TAU_SW = 20.0     # SimConfig software joint cap (CLAUDE.md tau_max) — tighter operational limit

ri = RobotInterface(URDF, gravity='zero')
model, data = ri.model, ri.data
fid = {'a': ri.frame_tool_a, 'b': ri.frame_tool_b}
vsl = {'a': ri.arm_a_v_slice, 'b': ri.arm_b_v_slice}
print(f'model nq={model.nq} nv={model.nv}; arm_a_v={vsl["a"]} arm_b_v={vsl["b"]}')

qs = [np.array(e['q_end'], float) for e in json.load(open(QEND))]
sl = json.load(open(SIM))
tp = np.array(sl['p_torso'], float); step = np.array(sl['step_idx'])
d_step = []
for s in range(6):
    m = step == s
    seg = tp[m]; disp = seg[-1] - seg[0]
    d_step.append(disp / max(np.linalg.norm(disp), 1e-12))
stance = ['a', 'b', 'a', 'b', 'a', 'b']   # swing = b,a,b,a,b,a (log)


def cap(Jlin, tau):
    """force-capacity numbers for a 3x7 linear Jacobian at torque limit tau."""
    S = np.linalg.svd(Jlin, compute_uv=False)
    return dict(sigma=S.round(4).tolist(), cond=round(float(S.max() / max(S.min(), 1e-12)), 1),
                f_worst=round(float(tau / max(np.linalg.norm(Jlin, axis=1))), 1),  # min over all dirs
                f_isotropic_min=round(float(tau / S.max()), 1))                     # 2-norm ellipsoid min


rows = []
print('\n' + '=' * 100)
print('PHASE 1i TASK A — stance-arm contact-force capacity at the 6 docked configs')
print(f'  tau_max: MJCF ctrlrange = {TAU_PHYS} Nm (physical);  SimConfig software = {TAU_SW} Nm')
print(f'  {"step":>4} {"stance":>6} | {"along-step dir":>22} | {"f_along@50":>10} {"f_along@20":>10} | '
      f'{"f_worst@50":>10} | {"sigma_min":>9} {"cond":>7}')
for k in range(6):
    q = qs[k]; st = stance[k]; d = d_step[k]
    J = pin.computeFrameJacobian(model, data, q, fid[st], pin.LOCAL_WORLD_ALIGNED)  # 6 x nv
    Jarm = J[:, vsl[st]]              # 6 x 7 (stance arm columns)
    Jlin = np.asarray(Jarm[0:3, :])   # 3 x 7 linear part (force maps through this)
    jTd = Jlin.T @ d                  # 7-vec; |tau_i| for unit force along d
    f_along_50 = TAU_PHYS / max(np.max(np.abs(jTd)), 1e-12)
    f_along_20 = TAU_SW / max(np.max(np.abs(jTd)), 1e-12)
    c = cap(Jlin, TAU_PHYS)
    rows.append(dict(step=k, stance=st, dir=[round(x, 3) for x in d],
                     f_along_50=round(float(f_along_50), 1), f_along_20=round(float(f_along_20), 1),
                     f_worst_50=c['f_worst'], f_isotropic_min_50=c['f_isotropic_min'],
                     sigma=c['sigma'], sigma_min=round(c['sigma'][-1] if isinstance(c['sigma'], list) else 0, 4),
                     cond=c['cond']))
    print(f'  {k:>4} {st:>6} | {str([round(x,2) for x in d]):>22} | {f_along_50:>10.1f} {f_along_20:>10.1f} | '
          f'{c["f_worst"]:>10.1f} | {min(c["sigma"]):>9.4f} {c["cond"]:>7.1f}')

fa50 = [r['f_along_50'] for r in rows]; fw50 = [r['f_worst_50'] for r in rows]
print('\n  ACROSS 6 STEPS (physical tau=50 Nm):')
print(f'    f_along (locomotion dir): min={min(fa50):.1f}  max={max(fa50):.1f}  N   (spread {max(fa50)-min(fa50):.1f})')
print(f'    f_worst (weakest dir)   : min={min(fw50):.1f}  max={max(fw50):.1f}  N')
print(f'    -> conservative binding (min f_along across steps) = {min(fa50):.1f} N @ tau=50  '
      f'({min(r["f_along_20"] for r in rows):.1f} N @ tau=20 software)')
print(f'    HOTDOCK interface ceiling = 3000 N (given).  min(arm, HOTDOCK) = {min(min(fa50),3000):.1f} N -> ARM binds')
json.dump(dict(tau_phys=TAU_PHYS, tau_sw=TAU_SW, rows=rows,
               min_f_along_50=round(min(fa50), 1), min_f_along_20=round(min(r['f_along_20'] for r in rows), 1),
               min_f_worst_50=round(min(fw50), 1)),
          open('results/j2_ablation_envelope/arm_capacity_1i.json', 'w'), indent=2)
print('  wrote results/j2_ablation_envelope/arm_capacity_1i.json')
print('=' * 100)
