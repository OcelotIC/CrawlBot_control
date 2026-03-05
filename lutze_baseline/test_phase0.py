"""Phase 0 smoke test — verify centroidal layer against Pinocchio.

Runs without MuJoCo: just loads the URDF, sets a neutral config, and checks
that centroidal_model, contact_adjoint, and momentum_map produce plausible
outputs with correct shapes.
"""

import argparse
import os
import sys
import numpy as np
import pinocchio as pin

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lutze_baseline.centroidal_model import compute_centroidal_state
from lutze_baseline.contact_adjoint import (
    compute_contact_adjoint, compute_dual_contact_adjoints, skew,
)
from lutze_baseline.momentum_map import compute_momentum_map


def run_test(urdf_path):
    print("=== Phase 0 Smoke Test ===\n")

    # Load model
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    q = pin.neutral(model)
    v = np.zeros(model.nv)

    print(f"Robot loaded: {model.nv} dof_v, mass={pin.computeTotalMass(model):.3f} kg")

    # --- centroidal_model ---
    print("\n[centroidal_model]")
    cs = compute_centroidal_state(model, data, q, v)
    print(f"  r_com     = {cs.r_com}")
    print(f"  v_com     = {cs.v_com}")
    print(f"  L_com     = {cs.L_com}")
    print(f"  I_locked  = diag({np.diag(cs.I_locked)})")
    print(f"  mass      = {cs.mass:.3f}")

    assert cs.r_com.shape == (3,), f"r_com shape {cs.r_com.shape}"
    assert cs.I_locked.shape == (3, 3), f"I_locked shape {cs.I_locked.shape}"
    assert cs.mass > 0, "Mass must be positive"
    assert np.allclose(cs.v_com, 0), "Zero velocity at rest"
    assert np.allclose(cs.L_com, 0, atol=1e-10), "Zero momentum at rest"
    print("  [OK] All shapes and values plausible.")

    # --- contact_adjoint ---
    print("\n[contact_adjoint]")
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    Ad_a, Ad_b = compute_dual_contact_adjoints(model, data, q,
                                                active_a=True, active_b=True)
    print(f"  Ad_a shape = {Ad_a.shape}")
    print(f"  Ad_b shape = {Ad_b.shape}")

    fid_a = model.getFrameId('tool_a')
    fid_b = model.getFrameId('tool_b')
    print(f"  tool_a pos = {data.oMf[fid_a].translation}")
    print(f"  tool_b pos = {data.oMf[fid_b].translation}")

    assert Ad_a.shape == (6, 6), f"Ad_a shape {Ad_a.shape}"
    assert Ad_b.shape == (6, 6), f"Ad_b shape {Ad_b.shape}"

    # Unit force at tool_a → check world wrench is nonzero
    # Convention: [f(3), tau(3)] — force Fx is at index 0
    F_unit = np.array([1, 0, 0, 0, 0, 0])  # unit Fx at contact
    F_world = Ad_a @ F_unit
    print(f"  Unit Fx at tool_a → world wrench = {F_world}")
    assert np.linalg.norm(F_world) > 0, "Force should be nonzero"

    # Inactive contact should return None
    Ad_a2, Ad_b2 = compute_dual_contact_adjoints(model, data, q,
                                                  active_a=True, active_b=False)
    assert Ad_b2 is None, "Inactive contact should return None"
    print("  [OK] Adjoint matrices correct.")

    # --- momentum_map ---
    print("\n[momentum_map]")
    r_a = data.oMf[fid_a].translation.copy()
    r_b = data.oMf[fid_b].translation.copy()

    M_dual = compute_momentum_map(cs.r_com, r_a, r_b)
    M_single = compute_momentum_map(cs.r_com, r_a, None)
    M_none = compute_momentum_map(cs.r_com, None, None)

    print(f"  M_dual   shape = {M_dual.shape}")
    print(f"  M_single shape = {M_single.shape}")
    print(f"  M_none   shape = {M_none.shape}")

    assert M_dual.shape == (3, 12), f"M_dual shape {M_dual.shape}"
    assert M_single.shape == (3, 6), f"M_single shape {M_single.shape}"
    assert M_none.shape == (3, 0), f"M_none shape {M_none.shape}"

    # Unit torque at contact A should pass through to L_dot
    # Convention: [f(3), tau(3)] — torque tau_x is at index 3
    Fc_unit_torque = np.array([0, 0, 0, 1, 0, 0])
    L_dot = M_single @ Fc_unit_torque
    print(f"  Unit tau_x at A → L_dot = {L_dot}")
    assert np.allclose(L_dot, [1, 0, 0]), "Pure torque should pass through"

    # Unit force at contact A should create lever-arm torque
    # Convention: [f(3), tau(3)] — force f_x is at index 0
    Fc_unit_force = np.array([1, 0, 0, 0, 0, 0])
    L_dot_f = M_single @ Fc_unit_force
    lever = r_a - cs.r_com
    expected = skew(lever) @ np.array([1, 0, 0])
    print(f"  Unit f_x at A → L_dot = {L_dot_f}, expected = {expected}")
    assert np.allclose(L_dot_f, expected, atol=1e-10), "Lever arm cross product"
    print("  [OK] Momentum map correct.")

    print("\n=== Phase 0: ALL TESTS PASSED ===")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf', default='models/VISPA_crawling_fixed.urdf')
    args = parser.parse_args()
    run_test(args.urdf)
