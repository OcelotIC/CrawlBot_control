#!/usr/bin/env python3
"""
test_r3_fixes.py — Validation of structure-frame refactoring
=============================================================================

After the structure-frame refactoring, all controller quantities (torso pose,
CoM, L_com, anchor positions, trajectories) are expressed in the structure
body frame. This test validates:

  TEST A: mujoco_to_pinocchio returns torso pose in structure frame
  TEST B: Anchors in scheduler are structure-local constants
  TEST C: TorsoPlanner works in structure frame (no p_struct args)
  TEST D: Quintic continuity (v=0 at phase boundaries)

Usage:
    python test_r3_fixes.py --urdf <path_to_urdf> --mjcf <path_to_mjcf>

Exit code 0 = all checks pass.
"""

import sys
import os
import argparse
import numpy as np

np.set_printoptions(precision=5, suppress=True)

PASS = '\033[92m PASS \033[0m'
FAIL = '\033[91m FAIL \033[0m'
n_pass = n_fail = 0


def check(name, condition, detail=''):
    global n_pass, n_fail
    if condition:
        n_pass += 1
        print(f'  [{PASS}] {name}')
    else:
        n_fail += 1
        print(f'  [{FAIL}] {name}  ->  {detail}')


parser = argparse.ArgumentParser()
parser.add_argument('--urdf', default='models/VISPA_crawling_fixed.urdf')
parser.add_argument('--mjcf', default='models/VISPA_crawling.xml')
args = parser.parse_args()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import pinocchio as pin
from simulation_loop import SimulationLoop, SimConfig, mujoco_to_pinocchio

# ---------------------------------------------------------------------------
print('\n' + '='*65)
print('  SETUP - Initialisation SimulationLoop (n_steps=2)')
print('='*65)

sim = SimulationLoop(mjcf_path=args.mjcf, urdf_path=args.urdf)
sim.setup(n_steps=2, start_a=2, start_b=2)

plan = sim.plan
ss_gp = plan.phases[1]       # first SS phase
t_ss0 = plan.t_start[1]
t_ss1 = plan.t_end[1]
swing = ss_gp.swing_arm
stance_a = ss_gp.anchor_a_idx
stance_b = ss_gp.anchor_b_idx
target = ss_gp.swing_to_idx

print(f'  Step info : swing={swing}, stance=({stance_a}a,{stance_b}b), '
      f'target={target}{swing}')
print(f'  SS timing : [{t_ss0:.2f}, {t_ss1:.2f}] s')

# ---------------------------------------------------------------------------
print('\n' + '='*65)
print('  TEST A - mujoco_to_pinocchio returns structure-frame torso')
print('='*65)

# Read structure pose
p_s = sim.mj_data.qpos[0:3].copy()
w, x, y, z = sim.mj_data.qpos[3:7]
R_s = pin.Quaternion(w, x, y, z).toRotationMatrix()

# Read torso pose in world
rwa = sim.has_rwa
off_q = 3 if rwa else 0
p_t_world = sim.mj_data.qpos[7 + off_q: 10 + off_q].copy()
qw_t, qx_t, qy_t, qz_t = sim.mj_data.qpos[10 + off_q: 14 + off_q]
R_t_world = pin.Quaternion(qw_t, qx_t, qy_t, qz_t).toRotationMatrix()

# Expected: torso in structure frame
p_t_expected = R_s.T @ (p_t_world - p_s)
R_t_expected = R_s.T @ R_t_world

# Get from mujoco_to_pinocchio
pq, pv = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
p_t_from_func = pq[0:3]
x_q, y_q, z_q, w_q = pq[3:7]
R_t_from_func = pin.Quaternion(x_q, y_q, z_q, w_q).toRotationMatrix()

err_pos = np.linalg.norm(p_t_from_func - p_t_expected)
err_rot = np.linalg.norm(R_t_from_func - R_t_expected)
print(f'  p_torso struct (expected): {p_t_expected}')
print(f'  p_torso struct (func):     {p_t_from_func}')
check('Position transform correct (< 1e-10)', err_pos < 1e-10,
      f'err={err_pos:.2e}')
check('Rotation transform correct (< 1e-10)', err_rot < 1e-10,
      f'err={err_rot:.2e}')

# ---------------------------------------------------------------------------
print('\n' + '='*65)
print('  TEST B - Anchors are structure-local constants')
print('='*65)

# The scheduler anchors should be in structure frame
anchor_a2 = sim.sched.anchors_a[2]
anchor_b2 = sim.sched.anchors_b[2]
print(f'  anchor_a[2] = {anchor_a2}')
print(f'  anchor_b[2] = {anchor_b2}')

# Verify they are NOT in world frame (world has structure offset)
# The structure starts at some position (typically [0, 0, -1.8])
check('anchor z not at world-frame level (not ~ -1.8)',
      abs(anchor_a2[2] - (-1.8)) > 0.1,
      f'z={anchor_a2[2]:.3f}')

# Verify y offsets are +-0.3 (structure local grid)
check('anchor_a y > 0 (positive side)', anchor_a2[1] > 0.2,
      f'y={anchor_a2[1]:.3f}')
check('anchor_b y < 0 (negative side)', anchor_b2[1] < -0.2,
      f'y={anchor_b2[1]:.3f}')

# Anchors should be constant even if structure drifts
anchors_before = [a.copy() for a in sim.sched.anchors_a]
# Inject artificial drift
sim.mj_data.qpos[0] += 0.1
mujoco.mj_forward(sim.mj_model, sim.mj_data)
anchors_after = [a.copy() for a in sim.sched.anchors_a]
max_change = max(np.linalg.norm(a - b) for a, b in
                 zip(anchors_before, anchors_after))
check('Anchors unchanged after structure drift', max_change < 1e-15,
      f'max_change={max_change:.2e}')
# Undo drift
sim.mj_data.qpos[0] -= 0.1
mujoco.mj_forward(sim.mj_model, sim.mj_data)

# ---------------------------------------------------------------------------
print('\n' + '='*65)
print('  TEST C - TorsoPlanner in structure frame (Fix 1 analog)')
print('='*65)

q_ret = sim._setup_torso_for_step(
    t_ss0, t_ss1, swing, stance_a, stance_b, swing, target)

# Read live torso state in structure frame
pq_live, _ = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
rs_live = sim.robot.update(pq_live, np.zeros(18))
p_live = rs_live.oMf_torso.translation.copy()

# TorsoPlanner reference (no struct args needed)
ref_hold = sim.torso_planner.reference_at(t_ss0 - 0.01)
ref_start = sim.torso_planner.reference_at(t_ss0 + 0.01)

err_hold = np.linalg.norm(ref_hold.p - p_live)
err_start = np.linalg.norm(ref_start.p - p_live)

check('set_hold.p == torso_live struct frame (|err| < 0.1 mm)',
      err_hold < 1e-4, f'|err|={err_hold*1000:.3f} mm')
check('add_phase.p_start == torso_live struct frame (|err| < 0.1 mm)',
      err_start < 1e-4, f'|err|={err_start*1000:.3f} mm')
check('q_start returned = live config',
      np.allclose(q_ret[:7], pq_live[:7], atol=1e-10),
      f'||dq||={np.linalg.norm(q_ret[:7]-pq_live[:7]):.2e}')

print(f'\n  p_torso_live (struct) = {p_live}')
print(f'  p_hold               = {ref_hold.p}')

# ---------------------------------------------------------------------------
print('\n' + '='*65)
print('  TEST D - Quintic continuity: v(tau=0) = v(tau=1) = 0')
print('='*65)

t_traj_start = t_ss0 + sim.cfg.torso_delay * sim.cfg.t_swing
eps = 1e-4
ref_begin = sim.torso_planner.reference_at(t_traj_start + eps)
ref_end_c = sim.torso_planner.reference_at(t_ss1 - eps)

check('v_lin(tau~0) ~ 0 (quintic start)',
      np.linalg.norm(ref_begin.v[:3]) < 0.01,
      f'||v||={np.linalg.norm(ref_begin.v[:3]):.4f} m/s')
check('v_lin(tau~1) ~ 0 (quintic end)',
      np.linalg.norm(ref_end_c.v[:3]) < 0.01,
      f'||v||={np.linalg.norm(ref_end_c.v[:3]):.4f} m/s')
check('v_ang(tau~0) ~ 0',
      np.linalg.norm(ref_begin.v[3:]) < 0.01,
      f'||w||={np.linalg.norm(ref_begin.v[3:]):.4f} rad/s')
check('v_ang(tau~1) ~ 0',
      np.linalg.norm(ref_end_c.v[3:]) < 0.01,
      f'||w||={np.linalg.norm(ref_end_c.v[3:]):.4f} rad/s')

# ---------------------------------------------------------------------------
print('\n' + '='*65)
print(f'  RESULTATS : {n_pass} PASS, {n_fail} FAIL')
print('='*65)
sys.exit(0 if n_fail == 0 else 1)
