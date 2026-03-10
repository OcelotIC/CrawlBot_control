"""Phase 1 smoke test — verify Lutze controller pipeline.

Tests the full chain: feedforward → QP → joint torques → swing controller.
No MuJoCo required — uses Pinocchio only at neutral configuration.

Usage:
    python -m lutze_baseline.test_phase1 --urdf models/VISPA_crawling_fixed.urdf
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_interface import RobotInterface, NV
from lutze_baseline.centroidal_model import compute_centroidal_state
from lutze_baseline.contact_adjoint import compute_dual_contact_adjoints
from lutze_baseline.momentum_map import compute_momentum_map
from lutze_baseline.lutze_feedforward import compute_feedforward, LutzeFeedforwardConfig
from lutze_baseline.lutze_qp import LutzeQP, LutzeQPConfig
from lutze_baseline.lutze_joint_torques import compute_joint_torques
from lutze_baseline.lutze_swing_controller import compute_swing_torques, SwingImpedanceConfig


def run_test(urdf_path: str):
    print("=== Phase 1 Smoke Test: Full Controller Pipeline ===\n")

    # --- Setup ---
    robot = RobotInterface(urdf_path, gravity='zero', torso_mass=40.0)
    q0 = robot.neutral_configuration()
    v0 = np.zeros(NV)
    rs = robot.update(q0, v0)
    cs = compute_centroidal_state(robot, rs)

    # --- 1. Feedforward ---
    r_ref = cs.r_com + np.array([0.1, 0.0, 0.0])  # 10 cm ahead
    v_ref = np.array([0.01, 0.0, 0.0])             # slow forward
    cfg_ff = LutzeFeedforwardConfig()

    F_d_r, F_d_b = compute_feedforward(
        cs.r_com, cs.v_com, r_ref, v_ref,
        struct_quat_wxyz=np.array([1.0, 0, 0, 0]),
        struct_omega=np.zeros(3),
        cfg=cfg_ff,
    )
    print("[feedforward]")
    print(f"  F_d_r = {F_d_r.round(3)}")
    print(f"  F_d_b = {F_d_b.round(3)}")
    assert F_d_r.shape == (6,), f"F_d_r shape {F_d_r.shape}"
    assert np.linalg.norm(F_d_r[3:]) > 0, "Should have nonzero tracking force"
    assert np.allclose(F_d_r[:3], 0), "No angular tracking for CoM"
    print("  [OK]")

    # --- 2. QP (dual contact) ---
    Ad_a, Ad_b = compute_dual_contact_adjoints(rs, active_a=True, active_b=True)
    r_ca = rs.oMf_tool_a.translation
    r_cb = rs.oMf_tool_b.translation
    M_lam = compute_momentum_map(cs.r_com, r_ca, r_cb)

    cfg_qp = LutzeQPConfig(f_max=25.0, tau_max=8.0)
    qp = LutzeQP(cfg_qp)

    Fc_a, Fc_b, info = qp.solve(Ad_a, Ad_b, M_lam, F_d_r, F_d_b)
    print(f"\n[QP - dual contact]")
    print(f"  status = {info['status']}")
    print(f"  Fc_a   = {Fc_a.round(3)}")
    print(f"  Fc_b   = {Fc_b.round(3)}")
    print(f"  |Fc_a| = {np.linalg.norm(Fc_a):.3f}")
    print(f"  |Fc_b| = {np.linalg.norm(Fc_b):.3f}")

    # Check momentum rate
    L_dot = M_lam @ np.concatenate([Fc_a, Fc_b])
    print(f"  L_dot  = {L_dot.round(4)} Nm")
    print(f"  |L_dot|= {np.linalg.norm(L_dot):.4f} Nm")

    assert Fc_a.shape == (6,), f"Fc_a shape {Fc_a.shape}"
    assert info['status'] in ('optimal', 'fallback')
    # Bounds check
    assert np.all(np.abs(Fc_a[:3]) <= cfg_qp.tau_max + 1e-6)
    assert np.all(np.abs(Fc_a[3:]) <= cfg_qp.f_max + 1e-6)
    print("  [OK]")

    # --- 3. QP (single contact) ---
    M_lam_single = compute_momentum_map(cs.r_com, r_ca, None)
    Fc_a_s, Fc_b_s, info_s = qp.solve(Ad_a, None, M_lam_single, F_d_r, F_d_b)
    print(f"\n[QP - single contact (A only)]")
    print(f"  status = {info_s['status']}")
    print(f"  Fc_a   = {Fc_a_s.round(3)}")
    print(f"  Fc_b   = {Fc_b_s.round(3)} (should be zeros)")
    assert np.allclose(Fc_b_s, 0), "Inactive contact should have zero wrench"
    print("  [OK]")

    # --- 4. Joint torques ---
    tau = compute_joint_torques(
        Fc_a, Fc_b, rs.J_tool_a, rs.J_tool_b,
        active_a=True, active_b=True, tau_max=10.0,
    )
    print(f"\n[joint_torques]")
    print(f"  tau = {tau.round(3)}")
    print(f"  |tau| = {np.linalg.norm(tau):.3f} Nm")
    print(f"  max |tau_i| = {np.max(np.abs(tau)):.3f} Nm")
    assert tau.shape == (12,), f"tau shape {tau.shape}"
    assert np.all(np.abs(tau) <= 10.0 + 1e-6), "Torque limit violated"
    print("  [OK]")

    # --- 5. Swing controller ---
    p_ee = rs.oMf_tool_b.translation
    v_ee = (rs.J_tool_b @ v0)[3:]  # linear velocity
    p_ee_ref = p_ee + np.array([0.05, 0.0, -0.03])  # 5cm ahead, 3cm down
    v_ee_ref = np.zeros(3)

    tau_swing = compute_swing_torques(
        p_ee, v_ee, p_ee_ref, v_ee_ref,
        J_ee=rs.J_tool_b, tau_max=10.0,
    )
    print(f"\n[swing_controller]")
    print(f"  tau_swing = {tau_swing.round(3)}")
    print(f"  max |tau_swing_i| = {np.max(np.abs(tau_swing)):.3f} Nm")
    assert tau_swing.shape == (12,), f"tau_swing shape {tau_swing.shape}"
    assert np.all(np.abs(tau_swing) <= 10.0 + 1e-6), "Torque limit violated"
    print("  [OK]")

    # --- 6. Combined torques ---
    tau_total = tau + tau_swing
    tau_total = np.clip(tau_total, -10.0, 10.0)
    print(f"\n[combined]")
    print(f"  tau_total = {tau_total.round(3)}")
    print(f"  max |tau_i| = {np.max(np.abs(tau_total)):.3f} Nm")

    print(f"\n=== Phase 1: ALL TESTS PASSED ===")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf', default='models/VISPA_crawling_fixed.urdf')
    args = parser.parse_args()
    run_test(args.urdf)
