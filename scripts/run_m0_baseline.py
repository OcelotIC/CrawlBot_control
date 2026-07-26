#!/usr/bin/env python3
"""Run a 3-step simulation and produce the M0 baseline diagnostic report.

Usage:
    PYTHONPATH=. MUJOCO_GL=disabled python3 scripts/run_m0_baseline.py
"""
import os
import sys

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

from crawlbot.simulation.sim_loop import SimulationLoop
from crawlbot.simulation.config import SimConfig
from crawlbot.diagnostics import run_diagnostics

OUTPUT_DIR = os.path.join(_root, 'Misc', 'runs', 'M0_baseline')
URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')
MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')

cfg = SimConfig()

print("=" * 70)
print("  M0 Baseline: 3-step simulation with diagnostics")
print("=" * 70)

sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=cfg)
sim.setup(n_steps=3, start_a=2, start_b=2)
log = sim.run(verbose=True)

# Save raw log
os.makedirs(OUTPUT_DIR, exist_ok=True)
log.save(os.path.join(OUTPUT_DIR, 'sim_log.json'))

# Run full diagnostics
metrics = run_diagnostics(
    log, OUTPUT_DIR, cfg=cfg,
    model=sim.mj_model, data=sim.mj_data)

print(f"\nOutputs saved to: {OUTPUT_DIR}")
