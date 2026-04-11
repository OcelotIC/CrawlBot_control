#!/usr/bin/env python3
"""Tests structure-frame: anchors constant, lever arm invariant to drift.

After the structure-frame refactoring, contact anchor positions are constant
in the structure body frame and are never re-read from MuJoCo during the
control loop. This test validates:

  TEST I:  Scheduler anchors match initial MuJoCo sites (after frame transform)
  TEST J:  Lever arm (r_Cj - r_com) is invariant to structure drift
  TEST K:  Regression — run test_r3_fixes

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
from crawlbot.simulation.sim_loop import SimulationLoop
from crawlbot.core.state_conversions import mujoco_to_pinocchio
from crawlbot.planning.contact_scheduler import read_anchors_from_mujoco

PASS = '\033[92m PASS \033[0m'; FAIL = '\033[91m FAIL \033[0m'
n_pass = n_fail = 0

def check(name, cond, detail=''):
    global n_pass, n_fail
    if cond: n_pass += 1; print(f'  [{PASS}] {name}')
    else:    n_fail += 1; print(f'  [{FAIL}] {name}  -> {detail}')

sim = SimulationLoop(mjcf_path=args.mjcf, urdf_path=args.urdf)
sim.setup(n_steps=2, start_a=2, start_b=2)
ss_gp    = sim.plan.phases[1]
t_ss0    = sim.plan.t_start[1]
stance_a = ss_gp.anchor_a_idx; stance_b = ss_gp.anchor_b_idx

print('\n' + '='*60)
print('  TEST I - Scheduler anchors consistent with MuJoCo sites')
print('='*60)

# Read live world-frame sites and convert to struct frame for comparison
mj_a_world, _ = read_anchors_from_mujoco(sim.mj_model, sim.mj_data)
p_s = sim.mj_data.qpos[0:3].copy()
w, x, y, z = sim.mj_data.qpos[3:7]
R_s = pin.Quaternion(w, x, y, z).toRotationMatrix()
mj_a_local = R_s.T @ (mj_a_world[stance_a] - p_s)

sched_a = sim.sched.anchors_a[stance_a]
err_A = np.linalg.norm(sched_a - mj_a_local)
print(f'  r_cA scheduler (struct frame) : {sched_a}')
print(f'  r_cA from MuJoCo (converted)  : {mj_a_local}')
print(f'  Error                         : {err_A*1000:.3f} mm')
check('Scheduler anchor matches MuJoCo site (< 0.01 mm)', err_A < 1e-5,
      f'{err_A*1000:.3f} mm')

print('\n' + '='*60)
print('  TEST J - Lever arm invariant under structure drift')
print('='*60)

# Compute lever arm before drift
pq0, pv0 = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
rs0 = sim.robot.update(pq0, pv0)
r_cA = sim.sched.anchors_a[stance_a]
lever_before = r_cA - rs0.r_com
print(f'  lever arm before drift: {lever_before}')

# Inject drift: +10 cm x, +5 cm y
DRIFT = np.array([0.10, 0.05, 0.0])
sim.mj_data.qpos[0:3] += DRIFT
# Also shift torso by same amount (rigid body motion, no relative change)
off_q = 3 if sim.has_rwa else 0
sim.mj_data.qpos[7 + off_q: 10 + off_q] += DRIFT
mujoco.mj_forward(sim.mj_model, sim.mj_data)

# Recompute in structure frame — should give same lever arm
pq1, pv1 = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
rs1 = sim.robot.update(pq1, pv1)
lever_after = r_cA - rs1.r_com  # r_cA is constant (struct frame)
err_lever = np.linalg.norm(lever_after - lever_before)
print(f'  lever arm after drift:  {lever_after}')
print(f'  error:                  {err_lever*1000:.3f} mm')
check('Lever arm invariant to rigid drift (< 0.01 mm)', err_lever < 1e-5,
      f'{err_lever*1000:.3f} mm')

# Verify that r_com in struct frame is the same (rigid drift = no relative motion)
err_rcom = np.linalg.norm(rs1.r_com - rs0.r_com)
print(f'  r_com struct before: {rs0.r_com}')
print(f'  r_com struct after:  {rs1.r_com}')
check('r_com invariant in struct frame (< 0.01 mm)', err_rcom < 1e-5,
      f'{err_rcom*1000:.3f} mm')

# Undo drift
sim.mj_data.qpos[0:3] -= DRIFT
sim.mj_data.qpos[7 + off_q: 10 + off_q] -= DRIFT
mujoco.mj_forward(sim.mj_model, sim.mj_data)

print('\n' + '='*60)
print('  TEST K - Regression R3')
print('='*60)
import subprocess
env = os.environ.copy()
env['PYTHONPATH'] = _root
env['MUJOCO_GL'] = 'disabled'
result = subprocess.run(
    ['python3', 'test_r3_fixes.py',
     '--urdf', os.path.abspath(args.urdf),
     '--mjcf', os.path.abspath(args.mjcf)],
    capture_output=True, text=True, env=env,
    cwd=os.path.dirname(os.path.abspath(__file__)))
final = [l for l in result.stdout.split('\n') if 'RESULTATS' in l]
print(f'  {final[0].strip() if final else "no summary"}')
check('R3 regression passes', result.returncode == 0)

print('\n' + '='*60)
print(f'  RESULTATS R4 : {n_pass} PASS, {n_fail} FAIL')
print('='*60)
sys.exit(0 if n_fail == 0 else 1)
