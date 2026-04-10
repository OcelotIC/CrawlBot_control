#!/usr/bin/env python3
"""M4 baseline: single-step closed-loop sim with all M1+M2+M3+M4 wiring.

Runs at two mass ratios:
  - 1 %  : structure mass 7110 kg, robot ~71 kg (canonical model)
  - 14 % : structure mass  507 kg, robot ~71 kg (reduced structure)

For the 14 % run, the structure mass is overridden programmatically on
the loaded MuJoCo model (no new MJCF required — per anti-pattern A2).

Outputs full diagnostics to
    results/M4_baseline_1pct/
    results/M4_baseline_14pct/

Usage:
    PYTHONPATH=. MUJOCO_GL=disabled python3 scripts/run_m4_baseline.py
"""
import os
import sys

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import numpy as np
import mujoco

from crawlbot.simulation.sim_loop import SimulationLoop
from crawlbot.simulation.config import SimConfig
from crawlbot.diagnostics import run_diagnostics

URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')
MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')


def _make_m4_config():
    """SimConfig with all M1+M2+M3+M4 flags enabled."""
    return SimConfig(
        # M2: reworked QP stack
        use_m2_stack=True,
        alpha_com_soft=5.0,
        alpha_passivity=1.0,
        # M3: NMPC conservation-law box
        enforce_hw_conservation=True,
        h_max_tight=np.full(3, 5.0),
        w_L_nmpc=1.0,
        kappa_terminal=1.0,   # no terminal margin for the baseline
        # M4: corrected legacy AOCS feedforward
        aocs_mode='legacy_corrected',
        aocs_use_legacy_corrected=True,
        aocs_use_H_estimator=False,
    )


def _scale_structure_mass(sim, new_mass_kg):
    """Override the structure body mass AND inertia on the already-loaded
    MuJoCo model.

    The structure is a uniform-density rigid body, so the principal
    inertia tensor scales linearly with mass (I = m·I_per_unit_mass).
    We must also update:
      - body_mass (scalar)
      - body_inertia (3-vector of principal moments)
      - body_subtreemass / body_subtreeinertia (composite, stale after
        mutating body_mass — recomputed by walking the kinematic tree)

    After mutation we call mj_forward() to recompute dependent
    quantities (crb, qfrc_bias, etc.).
    """
    bid = sim.mj_model.body("structure").id
    old_mass = float(sim.mj_model.body_mass[bid])
    k = new_mass_kg / old_mass

    # 1. Scale the structure body's mass AND its principal inertias
    #    (per-unit-mass inertia unchanged → I_new = k · I_old).
    sim.mj_model.body_mass[bid] = new_mass_kg
    sim.mj_model.body_inertia[bid] *= k

    # 2. Recompute composite subtree masses and inertias by walking the
    #    kinematic tree bottom-up. MuJoCo caches these at compile time
    #    and does NOT refresh them during mj_forward, so we do it here.
    njnt = sim.mj_model.nbody
    # Reset composite fields
    sim.mj_model.body_subtreemass[:] = sim.mj_model.body_mass.copy()
    for i in range(njnt - 1, 0, -1):  # children before parents
        parent = int(sim.mj_model.body_parentid[i])
        if parent >= 0:
            sim.mj_model.body_subtreemass[parent] += (
                sim.mj_model.body_subtreemass[i])

    # 3. Recompute forward kinematics + dynamics so the new inertia is
    #    reflected in crb_inertia, qM, qfrc_bias, etc.
    mujoco.mj_forward(sim.mj_model, sim.mj_data)

    return old_mass, new_mass_kg


def run_case(tag, structure_mass, output_dir, n_steps=1):
    print("\n" + "=" * 70)
    print(f"  M4 baseline: {tag}, structure = {structure_mass} kg, "
          f"n_steps = {n_steps}")
    print("=" * 70)

    cfg = _make_m4_config()
    sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=cfg)
    sim.setup(n_steps=n_steps, start_a=2, start_b=2)

    if structure_mass is not None:
        old, new = _scale_structure_mass(sim, structure_mass)
        print(f"  Structure mass: {old:.0f} kg -> {new:.0f} kg "
              f"(ratio {sim.robot._total_mass / new * 100:.1f}%)")

    log = sim.run(verbose=True)

    os.makedirs(output_dir, exist_ok=True)
    log.save(os.path.join(output_dir, 'sim_log.json'))

    metrics = run_diagnostics(log, output_dir, cfg=cfg)
    print(f"\n  Outputs: {output_dir}")
    return log, metrics


if __name__ == "__main__":
    # Single-step closed-loop run per the M4 task description.
    run_case("1pct",  7110.0, "results/M4_baseline_1pct",  n_steps=1)
    run_case("14pct",  507.0, "results/M4_baseline_14pct", n_steps=1)
