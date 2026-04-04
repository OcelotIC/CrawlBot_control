"""
Tests for NMPC/QP consistency and solver properties.

Validates CentroidalNMPC convergence, constraint enforcement,
warm-start behaviour, and infeasibility reporting.
"""

import numpy as np
import numpy.testing as npt
import pytest

from crawlbot.solvers.centroidal_nmpc import CentroidalNMPC, CentroidalNMPCConfig
from crawlbot.solvers.contact_phase import (
    ContactConfig,
    ContactPhase,
    compute_momentum_map,
    skew,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nominal_solve_inputs(nominal_robot_state, single_contact_config):
    """Build nominal NMPC inputs: everything at rest, reference = current."""
    r_com = nominal_robot_state.r_com.copy()
    v_com = np.zeros(3)
    L_com = np.zeros(3)
    hw = np.zeros(3)
    r_com_ref = r_com.copy()
    v_com_ref = np.zeros(3)
    return r_com, v_com, L_com, hw, r_com_ref, v_com_ref, single_contact_config


# ---------------------------------------------------------------------------
# CentroidalNMPC convergence tests
# ---------------------------------------------------------------------------

class TestNMPCConvergence:

    def test_nmpc_converges_from_nominal(
        self, centroidal_nmpc, nominal_robot_state, single_contact_config
    ):
        """NMPC should converge when initial state is at the reference."""
        r_com, v_com, L_com, hw, r_ref, v_ref, cc = _nominal_solve_inputs(
            nominal_robot_state, single_contact_config
        )
        _, _, _, _, _, info = centroidal_nmpc.solve(
            r_com, v_com, L_com, hw, r_ref, v_ref, cc, warm_start=False,
        )
        assert info.success, f"NMPC failed from nominal state: {info.status}"

    def test_nmpc_convergence_from_perturbation(
        self, centroidal_nmpc, nominal_robot_state, single_contact_config
    ):
        """NMPC should converge when initial state is perturbed by ~10%."""
        r_com, v_com, L_com, hw, r_ref, v_ref, cc = _nominal_solve_inputs(
            nominal_robot_state, single_contact_config
        )
        # Perturb initial state
        r_com += 0.01
        v_com += 0.05

        _, _, _, _, _, info = centroidal_nmpc.solve(
            r_com, v_com, L_com, hw, r_ref, v_ref, cc, warm_start=False,
        )
        assert info.success, f"NMPC failed from perturbed state: {info.status}"


# ---------------------------------------------------------------------------
# Constraint enforcement
# ---------------------------------------------------------------------------

class TestNMPCConstraints:

    def test_nmpc_hw_stays_in_envelope(
        self, centroidal_nmpc, nominal_robot_state, single_contact_config
    ):
        """Wheel momentum hw must remain within hw_min/max (with safety margin)."""
        r_com, v_com, L_com, hw, r_ref, v_ref, cc = _nominal_solve_inputs(
            nominal_robot_state, single_contact_config
        )
        # Perturb to generate nontrivial trajectory
        v_com += 0.05

        x_opt, _, info = centroidal_nmpc.get_full_trajectory(
            r_com, v_com, L_com, hw, r_ref, v_ref, cc,
        )
        assert info.success, f"NMPC failed: {info.status}"

        cfg = centroidal_nmpc.config
        hw_traj = x_opt[9:12, :]  # (3, N+1)

        # Safety-margin-adjusted bounds (as applied internally)
        hw_min_safe = (1 + cfg.safety_margin) * cfg.hw_min
        hw_max_safe = (1 - cfg.safety_margin) * cfg.hw_max

        assert np.all(hw_traj >= hw_min_safe.reshape(3, 1) - 1e-6), (
            f"hw below safe lower bound: min={hw_traj.min(axis=1)}, bound={hw_min_safe}"
        )
        assert np.all(hw_traj <= hw_max_safe.reshape(3, 1) + 1e-6), (
            f"hw above safe upper bound: max={hw_traj.max(axis=1)}, bound={hw_max_safe}"
        )

    def test_nmpc_Ldot_constraint_documented(
        self, centroidal_nmpc, nominal_robot_state, single_contact_config
    ):
        """Document that L_dot constraint covers spin only, not orbital.

        The NMPC constrains |L_dot_com| (the spin angular momentum rate)
        but the total torque demand on wheels is L_dot + orbital, where
        orbital = (r_com - r_mid) x sum(f_j). The orbital term can be
        large and is handled by the hw box constraint, not the L_dot
        path constraint.
        """
        r_com, v_com, L_com, hw, r_ref, v_ref, cc = _nominal_solve_inputs(
            nominal_robot_state, single_contact_config
        )
        v_com += 0.05

        x_opt, u_opt, info = centroidal_nmpc.get_full_trajectory(
            r_com, v_com, L_com, hw, r_ref, v_ref, cc,
        )
        assert info.success

        dt = centroidal_nmpc.config.dt
        N = centroidal_nmpc.config.N

        # Extract L_com trajectory and compute finite-difference L_dot
        L_traj = x_opt[6:9, :]  # (3, N+1)
        L_dot = np.diff(L_traj, axis=1) / dt  # (3, N)

        # L_dot should be bounded (spin component is constrained)
        assert np.all(np.isfinite(L_dot)), "L_dot contains non-finite values"

        # Compute orbital term for documentation
        r_mid = (cc.r_contact_A + cc.r_contact_B) / 2
        orbital_norms = []
        for k in range(N):
            r_com_k = x_opt[0:3, k]
            f_total = u_opt[0:3, k] + u_opt[6:9, k]
            orbital = np.cross(r_com_k - r_mid, f_total)
            orbital_norms.append(np.linalg.norm(orbital))

        # Document: orbital term can be comparable to or larger than spin L_dot
        max_Ldot = np.max(np.linalg.norm(L_dot, axis=0))
        max_orbital = np.max(orbital_norms)
        # This is informational -- no strict assertion on ratio, just verify
        # the computation succeeds and both terms are finite.
        assert np.isfinite(max_Ldot), "L_dot norm is not finite"
        assert np.isfinite(max_orbital), "orbital norm is not finite"


# ---------------------------------------------------------------------------
# Warm-start and infeasibility
# ---------------------------------------------------------------------------

class TestNMPCSolverBehaviour:

    def test_nmpc_cold_vs_warm_start(
        self, centroidal_nmpc, nominal_robot_state, single_contact_config
    ):
        """Warm-started solve should succeed and produce similar cost."""
        r_com, v_com, L_com, hw, r_ref, v_ref, cc = _nominal_solve_inputs(
            nominal_robot_state, single_contact_config
        )
        v_com += 0.02

        # Cold solve
        _, _, _, _, _, info_cold = centroidal_nmpc.solve(
            r_com, v_com, L_com, hw, r_ref, v_ref, cc, warm_start=False,
        )
        assert info_cold.success, "Cold solve failed"

        # Warm solve (same inputs -- previous solution is cached)
        _, _, _, _, _, info_warm = centroidal_nmpc.solve(
            r_com, v_com, L_com, hw, r_ref, v_ref, cc, warm_start=True,
        )
        assert info_warm.success, "Warm solve failed"

        # Costs should be close (same problem, warm start converges faster)
        npt.assert_allclose(
            info_warm.cost, info_cold.cost, rtol=0.1,
            err_msg="Warm-start cost diverges significantly from cold-start",
        )

    def test_nmpc_infeasible_input_reports_failure(
        self, centroidal_nmpc, nominal_robot_state, single_contact_config
    ):
        """NMPC should report failure when hw is far outside envelope."""
        r_com, v_com, L_com, _, r_ref, v_ref, cc = _nominal_solve_inputs(
            nominal_robot_state, single_contact_config
        )
        hw_infeasible = np.array([100.0, 100.0, 100.0])

        _, _, _, _, _, info = centroidal_nmpc.solve(
            r_com, v_com, L_com, hw_infeasible, r_ref, v_ref, cc,
            warm_start=False,
        )
        assert not info.success, (
            "NMPC should fail with hw=[100,100,100] far outside "
            f"envelope [{centroidal_nmpc.config.hw_min}, {centroidal_nmpc.config.hw_max}]"
        )


# ---------------------------------------------------------------------------
# QP / contact-phase utilities
# ---------------------------------------------------------------------------

class TestQPContactUtilities:

    def test_qp_feasibility_single_support(self, single_contact_config):
        """ContactConfig.from_phase gives correct nc and active_contacts for single-A."""
        cc = single_contact_config
        assert cc.nc == 1
        assert cc.active_contacts == (True, False)

        # compute_momentum_map returns correct shapes for single support
        r_com = np.array([0.0, 0.0, 0.0])
        M = compute_momentum_map(r_com, cc)
        assert M.shape == (3, 12)

        # Inactive contact B: columns 6:12 should be zero
        npt.assert_allclose(M[:, 6:12], 0.0,
                            err_msg="Inactive contact B columns must be zero")

    def test_qp_feasibility_double_support(self, double_contact_config):
        """ContactConfig for double support: nc=2, both active."""
        cc = double_contact_config
        assert cc.nc == 2
        assert cc.active_contacts == (True, True)

        r_com = np.zeros(3)
        M = compute_momentum_map(r_com, cc)
        assert M.shape == (3, 12)

        # Both blocks should be nonzero (contacts at +-0.3 in y)
        assert np.any(M[:, 0:6] != 0), "Contact A block should be nonzero"
        assert np.any(M[:, 6:12] != 0), "Contact B block should be nonzero"


# ---------------------------------------------------------------------------
# Feedforward acceleration
# ---------------------------------------------------------------------------

class TestFeedforwardAcceleration:

    def test_compute_feedforward_acceleration(self, centroidal_nmpc):
        """compute_feedforward_acceleration returns (f1 + f2) / m."""
        m = centroidal_nmpc.config.robot_mass
        lambda_ref = np.zeros(12)
        lambda_ref[0:3] = [1.0, 2.0, 3.0]   # f1
        lambda_ref[6:9] = [4.0, 5.0, 6.0]   # f2

        a_ff = centroidal_nmpc.compute_feedforward_acceleration(lambda_ref)
        expected = np.array([5.0, 7.0, 9.0]) / m
        npt.assert_allclose(a_ff, expected, atol=1e-12,
                            err_msg="Feedforward acceleration = (f1+f2)/m")
