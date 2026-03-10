#!/usr/bin/env python3
"""Tests R4 : r_contact_A/B live en frame monde pour le NMPC.

Valide que les positions d'ancre passées au CentroidalNMPC sont lues
en temps réel depuis MuJoCo (world frame) plutôt que figées à t=0.

Usage:
    python test_r4_fixes.py --urdf models/VISPA_crawling_fixed.urdf \
                            --mjcf models/VISPA_crawling.xml
"""
import sys, os, argparse, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

_root = os.path.join(os.path.dirname(__file__), '..')
parser = argparse.ArgumentParser()
parser.add_argument('--urdf', default=os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf'))
parser.add_argument('--mjcf', default=os.path.join(_root, 'models', 'VISPA_crawling.xml'))
args = parser.parse_args()

import mujoco, pinocchio as pin
from simulation_loop import SimulationLoop, mujoco_to_pinocchio
from contact_scheduler import read_anchors_from_mujoco

PASS = '\033[92m PASS \033[0m'; FAIL = '\033[91m FAIL \033[0m'
n_pass = n_fail = 0

def check(name, cond, detail=''):
    global n_pass, n_fail
    if cond: n_pass += 1; print(f'  [{PASS}] {name}')
    else:    n_fail += 1; print(f'  [{FAIL}] {name}  → {detail}')

sim = SimulationLoop(mjcf_path=args.mjcf, urdf_path=args.urdf)
sim.setup(n_steps=2, start_a=2, start_b=2)
ss_gp    = sim.plan.phases[1]
t_ss0    = sim.plan.t_start[1]
stance_a = ss_gp.anchor_a_idx; stance_b = ss_gp.anchor_b_idx

print('\n' + '='*60)
print('  TEST I — Cohérence live vs nominal à t=0')
print('='*60)
mj_a0, mj_b0 = read_anchors_from_mujoco(sim.mj_model, sim.mj_data)
cc_nom = sim.sched.contact_config_at(t_ss0 + 0.1)
err_A = np.linalg.norm(mj_a0[stance_a][:3] - cc_nom.r_contact_A)
print(f'  r_cA nominal : {cc_nom.r_contact_A}')
print(f'  r_cA live    : {mj_a0[stance_a][:3]}')
print(f'  Offset A     : {err_A*1000:.1f} mm  (grid vs cinématique réelle)')
check('Live et nominal cohérents — offset < 5 cm (même ancre)', err_A < 0.05,
      f'{err_A*100:.1f} cm')

print('\n' + '='*60)
print('  TEST J — Dérive structure : r_contact_live suit exactement')
print('='*60)
DRIFT = np.array([0.10, 0.05, 0.0])
r_cA_before = mj_a0[stance_a][:3].copy()
sim.mj_data.qpos[0:3]  += DRIFT
sim.mj_data.qpos[7:10] += DRIFT
mujoco.mj_forward(sim.mj_model, sim.mj_data)
mj_a1, _ = read_anchors_from_mujoco(sim.mj_model, sim.mj_data)
r_cA_after = mj_a1[stance_a][:3]
delta_A = r_cA_after - r_cA_before
print(f'  Dérive injectée   : {DRIFT*100} cm')
print(f'  Delta r_cA_live   : {delta_A*100} cm')
check('r_cA_live suit la dérive en x (< 1 mm)',
      abs(delta_A[0] - DRIFT[0]) < 1e-3)
check('r_cA_live suit la dérive en y (< 1 mm)',
      abs(delta_A[1] - DRIFT[1]) < 1e-3)

print('\n' + '='*60)
print('  TEST J2 — Bras de levier invariant à dérive solidaire')
print('='*60)
pq0, _ = mujoco_to_pinocchio(np.concatenate([
    sim.mj_data.qpos[0:3] - DRIFT, sim.mj_data.qpos[3:7],
    sim.mj_data.qpos[7:10] - DRIFT, sim.mj_data.qpos[10:26]
]), np.zeros(24))
rs0 = sim.robot.update(pq0, np.zeros(18))
lever_before = r_cA_before - rs0.r_com
pq1, pv1 = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
rs1 = sim.robot.update(pq1, pv1)
lever_live = r_cA_after          - rs1.r_com
lever_nom  = cc_nom.r_contact_A  - rs1.r_com
err_live = np.linalg.norm(lever_live - lever_before)
err_nom  = np.linalg.norm(lever_nom  - lever_before)
print(f'  err bras de levier live  : {err_live*1000:.3f} mm')
print(f'  err bras de levier nom   : {err_nom*1000:.1f} mm  (erreur évitée par R4)')
check('Bras de levier live invariant (< 1 mm)', err_live < 1e-3, f'{err_live*1000:.3f} mm')
check('Bras de levier nominal biaisé > 2 cm sans le fix', err_nom > 0.02,
      f'{err_nom*100:.1f} cm')

print('\n' + '='*60)
print('  TEST K — Régression R1–R3')
print('='*60)
import subprocess, os
result = subprocess.run(
    ['python3', 'test_r3_fixes.py', '--urdf', args.urdf, '--mjcf', args.mjcf],
    capture_output=True, text=True,
    cwd=os.path.dirname(os.path.abspath(__file__)))
final = [l for l in result.stdout.split('\n') if 'RÉSULTATS' in l]
print(f'  {final[0].strip() if final else "no summary"}')
check('R3 regression : 11 PASS 0 FAIL', result.returncode == 0)

print('\n' + '='*60)
print(f'  RÉSULTATS R4 : {n_pass} PASS, {n_fail} FAIL')
print('='*60)
sys.exit(0 if n_fail == 0 else 1)
