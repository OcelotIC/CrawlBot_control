#!/usr/bin/env python3
"""Experiment: does a stiffer weld solref eliminate the settling plateau?

Current MJCF: solref="0.003 1"  (3 ms time constant, critically damped)
The plateau at T ≈ 5e-8 J during stage 2 settling corresponds to residual
oscillations around the soft weld equilibrium. We test several stiffness
levels by creating a temporary MJCF with the modified solref.
"""
import os
import sys
import tempfile
import re

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import numpy as np

MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')
URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')


def _make_tmp_mjcf(tc, damp=1.0):
    """Write a temporary MJCF with all weld `solref` replaced."""
    with open(MJCF, 'r') as f:
        content = f.read()
    new_solref = f'solref="{tc} {damp}"'
    # Replace any weld's solref attribute
    content = re.sub(
        r'(<weld[^>]*)solref="[^"]*"',
        r'\1' + new_solref,
        content)
    # If any weld doesn't have a solref, insert one — not needed here
    # because all welds in the canonical MJCF have it.
    fd, tmp_path = tempfile.mkstemp(suffix='.xml',
                                     dir=os.path.dirname(MJCF))
    with os.fdopen(fd, 'w') as f:
        f.write(content)
    return tmp_path


def _run_case(tc, damp=1.0):
    tmp = _make_tmp_mjcf(tc, damp)
    try:
        # Import inside the function so the sim_loop re-imports fresh
        from crawlbot.simulation.sim_loop import SimulationLoop
        from crawlbot.simulation.config import SimConfig
        from crawlbot.core.state_conversions import mujoco_to_pinocchio

        cfg = SimConfig(use_m2_stack=True, enforce_hw_conservation=True,
                        aocs_use_legacy_corrected=True)
        sim = SimulationLoop(mjcf_path=tmp, urdf_path=URDF, config=cfg)
        # Silence the normal setup() print by redirecting stdout
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            sim.setup(n_steps=1, start_a=2, start_b=2)
        finally:
            sys.stdout = old_stdout

        s = sim._settling_log
        pq, pv = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
        rs = sim.robot.update(pq, pv)
        return {
            'tc': tc,
            'stage1': s['stage1_steps'],
            'stage2': s['stage2_steps'],
            'T_start': s['T_start'],
            'T_end': s['T_end'],
            'exit': s['exit_reason'],
            'vcom_mm': np.linalg.norm(rs.v_com) * 1000,
            'Lcom_mNms': np.linalg.norm(rs.L_com) * 1000,
        }
    finally:
        os.unlink(tmp)


print("\n" + "=" * 110)
print("  Weld stiffness experiment — does a harder weld reduce the settling T_kin floor?")
print("=" * 110)
print(f"{'tc [s]':>10} {'stage1':>8} {'stage2':>8} "
      f"{'T_end [J]':>14} {'|v_com| [mm/s]':>16} "
      f"{'|L_com| [mNms]':>16} {'exit':>12}")
print("-" * 110)

for tc in [0.003, 0.001, 0.0005, 0.0002, 0.00005]:
    try:
        r = _run_case(tc)
        print(f"{r['tc']:>10.5f} {r['stage1']:>8d} {r['stage2']:>8d} "
              f"{r['T_end']:>14.3e} {r['vcom_mm']:>16.4f} "
              f"{r['Lcom_mNms']:>16.4f} {r['exit']:>12s}")
    except Exception as e:
        print(f"{tc:>10.5f}  FAILED: {type(e).__name__}: {e}")
