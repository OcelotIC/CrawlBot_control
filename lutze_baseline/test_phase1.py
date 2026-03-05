"""Phase 1 smoke test — verify Lutze controller pipeline.

Tests the full chain: feedforward → QP → joint torques → swing controller.
No MuJoCo required — uses Pinocchio only at neutral configuration.

Usage:
    python -m lutze_baseline.test_phase1 --urdf models/VISPA_crawling_fixed.urdf
"""

import argparse
import os
import sys
import numpy as np
import pinocchio as pin

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lutze_baseline.centroidal_model import compute_centroidal_state
from lutze_baseline.contact_adjoint import compute_dual_contact_adjoints
from lutze_baseline.momentum_map import compute_momentum_map
from lutze_baseline.lutze_feedforward import compute_feedforward, LutzeFeedforwardConfig
from lutze_baseline.lutze_qp import LutzeQP, LutzeQPConfig
from lutze_baseline.lutze_joint_torques import compute_joint_torques, get_contact_jacobians
from lutze_baseline.lutze_swing_controller import compute_swing_torques, SwingImpedanceConfig


def run_test(urdf_path):
    print("=== Phase 1 Smoke Test: Full Controller Pipeline ===\n")

    # Load
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    q = pin.neutral(model)
    v = np.zeros(model.nv)

    # Centroidal state
    cs = compute_centroidal_state(model, data, q, v)

    # Contact placements
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    fid_a = model.getFrameId('tool_a')
    fid_b = model.getFrameId('tool_b')
    r_a = data.oMf[fid_a].translation.copy()
    r_b = data.oMf[fid_b].translation.copy()

    # --- Feedforward ---
    print("[feedforward]")
    r_ref = cs.r_com + np.array([0.1, 0.0, 0.0])  # 10cm ahead
    v_ref = np.array([0.01, 0.0, 0.0])
    cfg_ff = LutzeFeedforwardConfig()
    F_d_r, F_d_b = compute_feedforward(cs.r_com, cs.v_com, r_ref, v_ref, cfg=cfg_ff)

    assert F_d_r.shape == (6,), f"F_d_r shape {F_d_r.shape}"
    assert np.linalg.norm(F_d_r[3:]) > 0, "Should have nonzero tracking force"
    assert np.allclose(F_d_r[:3], 0), "No angular tracking for CoM"
    print(f"  F_d_r = {F_d_r}")
    print(f"  F_d_b = {F_d_b}")

    # --- QP (dual contact) ---
    print("\n[QP - dual contact]")
    Ad_a, Ad_b = compute_dual_contact_adjoints(model, data, q)
    M_lambda = compute_momentum_map(cs.r_com, r_a, r_b)

    qp_cfg = LutzeQPConfig(
        F_max=50.0, tau_max=50.0,  # VISPA-scale limits
    )
    qp = LutzeQP(qp_cfg)
    Fc_a, Fc_b, info = qp.solve(Ad_a, Ad_b, M_lambda, F_d_r, F_d_b)

    print(f"  status = {info['status']}")
    print(f"  Fc_a   = {Fc_a}")
    print(f"  Fc_b   = {Fc_b}")
    print(f"  |Fc_a| = {np.linalg.norm(Fc_a):.4f}")
    print(f"  |Fc_b| = {np.linalg.norm(Fc_b):.4f}")
    print(f"  L_dot  = {info['L_dot']}")
    print(f"  |L_dot|= {np.linalg.norm(info['L_dot']):.6f}")

    assert Fc_a.shape == (6,), f"Fc_a shape {Fc_a.shape}"
    assert info['status'] in ('optimal', 'fallback')

    # --- QP (single contact — A only) ---
    print("\n[QP - single contact (A only)]")
    M_single = compute_momentum_map(cs.r_com, r_a, None)
    Fc_a2, Fc_b2, info2 = qp.solve(Ad_a, None, M_single, F_d_r, F_d_b)
    print(f"  Fc_b = {Fc_b2} (should be zeros)")
    assert np.allclose(Fc_b2, 0), "Inactive contact should have zero wrench"

    # --- Joint torques ---
    print("\n[joint_torques]")
    J_a, J_b = get_contact_jacobians(model, data, q)
    tau = compute_joint_torques(Fc_a, Fc_b, J_a, J_b, tau_max=50.0)
    print(f"  tau = {tau}")
    print(f"  max |tau_i| = {np.max(np.abs(tau)):.4f}")
    assert tau.shape == (model.nv,)
    assert np.max(np.abs(tau)) <= 50.0 + 1e-6, "Torque limit violated"

    # --- Swing controller ---
    print("\n[swing_controller]")
    p_ee = r_b.copy()
    v_ee = np.zeros(3)
    p_ref = r_b + np.array([0.05, 0.0, 0.02])
    v_ref_sw = np.zeros(3)
    cfg_sw = SwingImpedanceConfig()
    tau_swing = compute_swing_torques(p_ee, v_ee, p_ref, v_ref_sw, J_b, cfg_sw, tau_max=50.0)
    print(f"  tau_swing = {tau_swing}")
    print(f"  max |tau_swing_i| = {np.max(np.abs(tau_swing)):.4f}")
    assert tau_swing.shape == (model.nv,), f"tau_swing shape {tau_swing.shape}"

    # --- Combined ---
    print("\n[combined]")
    tau_total = tau + tau_swing
    print(f"  tau_total = {tau_total}")

    print("\n=== Phase 1: ALL TESTS PASSED ===")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf', default='models/VISPA_crawling_fixed.urdf')
    args = parser.parse_args()
    run_test(args.urdf)
