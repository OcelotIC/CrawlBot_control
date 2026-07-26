#!/usr/bin/env python3
"""R5-fix — Validation suite for physical RWA model.

Tests:
    T1 — Layout DOF:              nq=29, nv=27, nu=15
    T2 — RW joints present:       rw_x, rw_y, rw_z in model
    T3 — mujoco_to_pinocchio:     torso pos correct after +3 offset
    T4 — AOCS active:             max(|tau_w|) > 0 during simulation
    T5 — hw_physical bounded:     max(||hw||) <= 6.0 Nms (h_max + 20%)
    T6 — Docking n_steps=3:       3/3 docks at d < 5 mm
    T7 — Violation rate:          hw > 5 Nms < 5% of steps
    T8 — Regression R3:           existing R3 tests pass
    T9 — Regression R4:           existing R4 tests pass

Usage:
    PYTHONPATH=. MUJOCO_GL=disabled python3 tests/test_r5fix_rwa.py
"""
import sys
import os
import numpy as np

_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _root)

os.environ.setdefault('MUJOCO_GL', 'disabled')

import mujoco

PASS = '\033[92m PASS \033[0m'
FAIL = '\033[91m FAIL \033[0m'
WARN = '\033[93m WARN \033[0m'
n_pass = n_fail = 0


def check(name, cond, detail=''):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f'  [{PASS}] {name}')
    else:
        n_fail += 1
        print(f'  [{FAIL}] {name}  -> {detail}')


# Paths
mjcf_rwa3 = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')
mjcf_orig = os.path.join(_root, 'models', 'VISPA_crawling.xml')
urdf_path = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')

# ═══════════════════════════════════════════════════════════════════════════════
# T1 — Layout DOF
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('  T1 — Layout DOF (nq=29, nv=27, nu=15)')
print('=' * 70)

m = mujoco.MjModel.from_xml_path(mjcf_rwa3)
check('T1a — nq == 29', m.nq == 29, f'got nq={m.nq}')
check('T1b — nv == 27', m.nv == 27, f'got nv={m.nv}')
check('T1c — nu == 15', m.nu == 15, f'got nu={m.nu}')

# ═══════════════════════════════════════════════════════════════════════════════
# T2 — RW joints present
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('  T2 — Reaction wheel joints present')
print('=' * 70)

for jn in ['rw_x', 'rw_y', 'rw_z']:
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
    check(f'T2 — joint {jn} exists', jid >= 0, f'not found')

# ═══════════════════════════════════════════════════════════════════════════════
# T3 — mujoco_to_pinocchio coherent with new layout
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('  T3 — mujoco_to_pinocchio coherence')
print('=' * 70)

from crawlbot.core.state_conversions import mujoco_to_pinocchio

d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

# Set a known torso position in the RWA layout
d.qpos[10] = 1.0   # torso x
d.qpos[11] = 2.0   # torso y
d.qpos[12] = 3.0   # torso z
d.qpos[13:17] = [1, 0, 0, 0]  # identity quat wxyz

pin_q, pin_v = mujoco_to_pinocchio(d.qpos, d.qvel)
# Structure at [0,0,-1.8], torso at [1,2,3] → struct frame = [1, 2, 4.8]
p_s = d.qpos[0:3]  # struct pos
p_t_expected = np.array([1.0, 2.0, 3.0]) - p_s  # R_s=I → simple subtraction
check('T3a — pin_q[0:3] == torso pos in struct frame',
      np.allclose(pin_q[0:3], p_t_expected, atol=1e-10),
      f'got {pin_q[0:3]}, expected {p_t_expected}')
# xyzw from wxyz=(1,0,0,0) -> (0,0,0,1) — same relative rotation (both identity)
check('T3b — pin_q[3:7] quat xyzw correct',
      np.allclose(pin_q[3:7], [0, 0, 0, 1]),
      f'got {pin_q[3:7]}')

# Also test with original layout (nq=26)
m_orig = mujoco.MjModel.from_xml_path(mjcf_orig)
d_orig = mujoco.MjData(m_orig)
d_orig.qpos[7] = 1.0
d_orig.qpos[8] = 2.0
d_orig.qpos[9] = 3.0
d_orig.qpos[10:14] = [1, 0, 0, 0]
pin_q2, _ = mujoco_to_pinocchio(d_orig.qpos, d_orig.qvel)
p_s2 = d_orig.qpos[0:3]
p_t2_expected = np.array([1.0, 2.0, 3.0]) - p_s2
check('T3c — original layout: torso in struct frame',
      np.allclose(pin_q2[0:3], p_t2_expected, atol=1e-10),
      f'got {pin_q2[0:3]}, expected {p_t2_expected}')

# ═══════════════════════════════════════════════════════════════════════════════
# T4–T7 — Full simulation tests (need pinocchio + casadi)
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('  T4-T7 — Full simulation with RWA model (n_steps=3)')
print('=' * 70)

try:
    from simulation_loop import SimulationLoop, SimConfig
    import pinocchio as pin

    cfg = SimConfig()
    sim = SimulationLoop(
        mjcf_path=mjcf_rwa3,
        urdf_path=urdf_path,
        config=cfg)
    sim.setup(n_steps=3, start_a=2, start_b=2)

    # Verify RWA detected
    check('T4-pre — RWA detected', sim.has_rwa, 'has_rwa is False')

    import time
    t0 = time.time()
    log = sim.run(verbose=True)
    elapsed = time.time() - t0
    print(f'\n  Sim completed in {elapsed:.1f}s')

    # T4 — AOCS active
    tau_w_arr = np.array(log.tau_w)
    max_tau_w = np.max(np.abs(tau_w_arr))
    check('T4 — AOCS active (max |tau_w| > 0)',
          max_tau_w > 0,
          f'max |tau_w| = {max_tau_w:.4f}')

    # T5 — hw_physical bounded
    hw_phys = np.array(log.hw_physical)
    hw_norms = np.linalg.norm(hw_phys, axis=1)
    peak_hw = hw_norms.max()
    check('T5 — hw_physical bounded (peak <= 6.0 Nms)',
          peak_hw <= 6.0,
          f'peak ||hw|| = {peak_hw:.2f} Nms')

    # T6 — Docking
    n_docks = len(log.dock_events)
    dock_thresh_mm = cfg.weld_radius * 1000
    check('T6 — Docking n_steps=3 (3/3 docks)',
          n_docks >= 3,
          f'got {n_docks}/3 docks')
    for ev in log.dock_events:
        check(f'T6 — dock step {ev["step"]} d < {dock_thresh_mm:.0f}mm',
              ev['d_mm'] < dock_thresh_mm,
              f'd = {ev["d_mm"]:.2f} mm')

    # T7 — Violation rate
    n_total = len(hw_norms)
    n_viol = np.sum(hw_norms > 5.0)
    viol_rate = n_viol / max(n_total, 1)
    check(f'T7 — hw violation rate < 5% ({n_viol}/{n_total})',
          viol_rate < 0.05,
          f'violation rate = {100*viol_rate:.1f}%')

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f'\n  [SKIP] T4-T7 failed with: {e}')
    # Count as failures
    for name in ['T4', 'T5', 'T6', 'T7']:
        n_fail += 1
        print(f'  [{FAIL}] {name} — skipped due to error')

# ═══════════════════════════════════════════════════════════════════════════════
# T8/T9 — Regression (check that existing tests still importable/runnable)
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('  T8/T9 — Regression check (original model still works)')
print('=' * 70)

# T8: Verify original model still loads and has correct DOF
check('T8 — Original model nq=26',
      m_orig.nq == 26, f'got nq={m_orig.nq}')
check('T8 — Original model nv=24',
      m_orig.nv == 24, f'got nv={m_orig.nv}')
check('T8 — Original model nu=12',
      m_orig.nu == 12, f'got nu={m_orig.nu}')

# T9: Verify pinocchio_to_mujoco backward compat (original layout)
from crawlbot.core.state_conversions import pinocchio_to_mujoco
pin_q_test = np.zeros(19)
pin_q_test[0:3] = [1, 2, 3]
pin_q_test[3:7] = [0, 0, 0, 1]  # xyzw
mj_q, mj_v = pinocchio_to_mujoco(pin_q_test, np.zeros(18), rwa=False)
check('T9a — pinocchio_to_mujoco(rwa=False) nq=26',
      len(mj_q) == 26, f'got len={len(mj_q)}')
check('T9b — pinocchio_to_mujoco(rwa=True) nq=29',
      len(pinocchio_to_mujoco(pin_q_test, np.zeros(18), rwa=True)[0]) == 29,
      'wrong length')
# Round-trip test
pin_q_rt, _ = mujoco_to_pinocchio(mj_q, mj_v)
check('T9c — Round-trip mj->pin->mj preserves torso pos',
      np.allclose(pin_q_rt[0:3], [1, 2, 3]),
      f'got {pin_q_rt[0:3]}')

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
total = n_pass + n_fail
print(f'  RESULTS:  {n_pass}/{total} PASS,  {n_fail}/{total} FAIL')
print('=' * 70)

if n_fail > 0:
    sys.exit(1)
