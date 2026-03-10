"""Phase 0 smoke test — verify centroidal layer against Pinocchio.

Runs without MuJoCo: just loads the URDF, sets a neutral config, and checks
that centroidal_model, contact_adjoint, and momentum_map produce correctly
shaped and plausible outputs.

Usage:
    python -m lutze_baseline.test_phase0 --urdf models/VISPA_crawling_fixed.urdf
"""

import argparse
import sys
import os
import numpy as np

# Add parent directory so we can import robot_interface
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_interface import RobotInterface, NQ, NV
from lutze_baseline.centroidal_model import compute_centroidal_state
from lutze_baseline.contact_adjoint import (
    compute_contact_adjoint, compute_dual_contact_adjoints,
)
from lutze_baseline.momentum_map import compute_momentum_map


def run_test(urdf_path: str):
    print("=== Phase 0 Smoke Test ===\n")

    # 1. Load robot
    robot = RobotInterface(urdf_path, gravity='zero', torso_mass=40.0)
    q0 = robot.neutral_configuration()
    v0 = np.zeros(NV)
    rs = robot.update(q0, v0)
    print(f"Robot loaded: {NQ} dof_q, {NV} dof_v, "
          f"mass={rs.total_mass:.1f} kg")

    # 2. Centroidal model
    cs = compute_centroidal_state(robot, rs)
    print(f"\n[centroidal_model]")
    print(f"  r_com     = {cs.r_com}")
    print(f"  v_com     = {cs.v_com}")
    print(f"  L_com     = {cs.L_com}")
    print(f"  I_locked  = diag({np.diag(cs.I_locked).round(3)})")
    print(f"  mass      = {cs.mass:.1f} kg")

    assert cs.r_com.shape == (3,), f"r_com shape {cs.r_com.shape}"
    assert cs.I_locked.shape == (3, 3), f"I_locked shape {cs.I_locked.shape}"
    assert cs.mass > 0, f"mass={cs.mass}"
    # At neutral config with v=0, L_com should be ~zero
    assert np.linalg.norm(cs.L_com) < 1e-6, f"|L_com|={np.linalg.norm(cs.L_com)}"
    print("  [OK] All shapes and values plausible.")

    # 3. Contact adjoint
    Ad_a, Ad_b = compute_dual_contact_adjoints(rs, active_a=True, active_b=True)
    print(f"\n[contact_adjoint]")
    print(f"  Ad_a shape = {Ad_a.shape}")
    print(f"  Ad_b shape = {Ad_b.shape}")
    print(f"  tool_a pos = {rs.oMf_tool_a.translation.round(4)}")
    print(f"  tool_b pos = {rs.oMf_tool_b.translation.round(4)}")

    assert Ad_a.shape == (6, 6), f"Ad_a shape {Ad_a.shape}"
    assert Ad_b.shape == (6, 6), f"Ad_b shape {Ad_b.shape}"

    # Verify: a unit force at contact should produce a torque at the origin
    F_test = np.array([0, 0, 0, 1, 0, 0])  # unit force in x
    W_a = Ad_a @ F_test
    print(f"  Unit Fx at tool_a → world wrench = {W_a.round(4)}")
    # Force should be preserved (rotated), torque = p × f
    assert np.linalg.norm(W_a[3:]) > 0, "Force should be nonzero"

    # Test None for inactive contact
    Ad_a2, Ad_b2 = compute_dual_contact_adjoints(rs, active_a=True, active_b=False)
    assert Ad_b2 is None, "Inactive contact should return None"
    print("  [OK] Adjoint matrices correct.")

    # 4. Momentum map
    r_ca = rs.oMf_tool_a.translation
    r_cb = rs.oMf_tool_b.translation
    M_dual = compute_momentum_map(cs.r_com, r_ca, r_cb)
    M_single = compute_momentum_map(cs.r_com, r_ca, None)
    M_none = compute_momentum_map(cs.r_com, None, None)

    print(f"\n[momentum_map]")
    print(f"  M_dual   shape = {M_dual.shape}")
    print(f"  M_single shape = {M_single.shape}")
    print(f"  M_none   shape = {M_none.shape}")

    assert M_dual.shape == (3, 12), f"M_dual shape {M_dual.shape}"
    assert M_single.shape == (3, 6), f"M_single shape {M_single.shape}"
    assert M_none.shape == (3, 0), f"M_none shape {M_none.shape}"

    # Test: pure torque at contact A → L_dot = tau
    Fc_torque = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # tau_x at A
    L_dot = M_dual @ Fc_torque
    print(f"  Unit tau_x at A → L_dot = {L_dot.round(4)}")
    assert abs(L_dot[0] - 1.0) < 1e-10, "Pure torque should pass through"

    # Test: force at contact → L_dot = lever × force
    Fc_force = np.zeros(12)
    Fc_force[3] = 1.0  # unit f_x at contact A
    L_dot = M_dual @ Fc_force
    lever = r_ca - cs.r_com
    expected_L_dot = np.cross(lever, np.array([1, 0, 0]))
    print(f"  Unit f_x at A → L_dot = {L_dot.round(4)}, "
          f"expected = {expected_L_dot.round(4)}")
    assert np.allclose(L_dot, expected_L_dot, atol=1e-10), "Lever arm cross product"
    print("  [OK] Momentum map correct.")

    print(f"\n=== Phase 0: ALL TESTS PASSED ===")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf', default='models/VISPA_crawling_fixed.urdf')
    args = parser.parse_args()
    run_test(args.urdf)
