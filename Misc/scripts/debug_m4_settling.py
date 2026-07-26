#!/usr/bin/env python3
"""Diagnose why the initial state has |L_com|=0.73 Nms and v_com!=0
after 500 settle steps with both welds active.

Prints the state every 50 settle steps to see how it evolves.
"""
import os
import sys

_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import numpy as np
import mujoco
import pinocchio as pin

from crawlbot.simulation.sim_loop import SimulationLoop
from crawlbot.simulation.config import SimConfig
from crawlbot.core.state_conversions import mujoco_to_pinocchio

URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')
MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')

np.set_printoptions(precision=5, suppress=True, linewidth=100)

cfg = SimConfig(n_settle_steps=0)  # Skip the normal settling
sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=cfg)
sim.setup(n_steps=1, start_a=2, start_b=2)

# sim.setup() has already placed the robot in the dock config and
# activated both welds. qvel is now zero. We re-zero it for a clean
# starting point and step manually.
sim.mj_data.qvel[:] = 0.0
mujoco.mj_forward(sim.mj_model, sim.mj_data)

print("=" * 85)
print("  Initial state AFTER activating welds (no stepping)")
print("=" * 85)
pq, pv = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
rs = sim.robot.update(pq, pv)
print(f"  struct pos (qpos[0:3]):  {sim.mj_data.qpos[0:3]}")
print(f"  struct quat (qpos[3:7]): {sim.mj_data.qpos[3:7]}")
print(f"  struct lin vel (qvel[0:3]): {sim.mj_data.qvel[0:3]}")
print(f"  struct ang vel (qvel[3:6]): {sim.mj_data.qvel[3:6]}")
print(f"  r_com (Pinocchio):  {rs.r_com}")
print(f"  v_com (Pinocchio):  {rs.v_com}")
print(f"  L_com (Pinocchio):  {rs.L_com}  |L|={np.linalg.norm(rs.L_com):.4f}")
print(f"  robot v (qvel norm): {np.linalg.norm(sim.mj_data.qvel):.5f}")
print()

# Check constraint violation
print("=" * 85)
print("  Constraint violation at t=0")
print("=" * 85)
ncon = sim.mj_data.ncon
print(f"  ncon        : {ncon}")
print(f"  nefc (total): {sim.mj_data.nefc}")
# efc_force: constraint forces
if sim.mj_data.nefc > 0:
    print(f"  |efc_force| : {np.linalg.norm(sim.mj_data.efc_force):.4f}")

# Now step 500 times and print every 50
print()
print("=" * 85)
print("  State evolution over 500 settle steps (no control)")
print("=" * 85)
print(f"{'step':>5}  {'|qvel|':>10}  {'|v_com|':>10}  {'|L_com|':>10}  "
      f"{'|struct_v|':>10}  {'|struct_ω|':>10}")
print("-" * 85)
for k in range(501):
    if k % 50 == 0:
        pq, pv = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
        rs = sim.robot.update(pq, pv)
        print(f"{k:>5d}  "
              f"{np.linalg.norm(sim.mj_data.qvel):>10.5f}  "
              f"{np.linalg.norm(rs.v_com):>10.5f}  "
              f"{np.linalg.norm(rs.L_com):>10.5f}  "
              f"{np.linalg.norm(sim.mj_data.qvel[0:3]):>10.5f}  "
              f"{np.linalg.norm(sim.mj_data.qvel[3:6]):>10.5f}")
    if k < 500:
        mujoco.mj_step(sim.mj_model, sim.mj_data)

print()
print("=" * 85)
print("  Final state components")
print("=" * 85)
pq, pv = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
rs = sim.robot.update(pq, pv)
print(f"  struct pos (qpos[0:3]):  {sim.mj_data.qpos[0:3]}")
print(f"  struct lin vel (qvel[0:3]): {sim.mj_data.qvel[0:3]}")
print(f"  struct ang vel (qvel[3:6]): {sim.mj_data.qvel[3:6]}")
print(f"  r_com:  {rs.r_com}")
print(f"  v_com:  {rs.v_com}")
print(f"  L_com:  {rs.L_com}")
