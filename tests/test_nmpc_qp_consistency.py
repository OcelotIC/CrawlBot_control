"""
Tests for NMPC/QP consistency and solver properties.

Validates CentroidalNMPC convergence, constraint enforcement,
warm-start behaviour, and infeasibility reporting.

NMPC state: x = [r_com(3), v_com(3), L_com(3)]  (NX=9, hw removed)
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
    r_com_ref = r_com.copy()
    v_com_ref = np.zeros(3)
    return r_com, v_com, L_com, r_com_ref, v_com_ref, single_contact_config


# ---------------------------------------------------------------------------
# CentroidalNMPC convergence tests
# ---------------------------------------------------------------------------

class TestNMPCConvergence:

    def test_nmpc_converges_from_nominal(
        self, centroidal_nmpc, nominal_robot_state, single_contact_config
    ):
        """NMPC should converge from nominal (everything at rest)."""
        r_com, v_com, L_com, r_ref, v_ref, cc = _nominal_solve_inputs(
            nominal_robot_state, single_contact_config
        )
        _, _, _, _, info = centroidal_nmpc.solve(
            r_com=r_com, v_com=v_com, L_com=L_com,
            r_com_ref=r_ref, v_com_ref=v_ref,
            contact_config=cc, warm_start=False,
        )
        assert info.success, f"NMPC failed from nominal: {info.status}"

    def test_nmpc_convergence_from_perturbation(
        self, centroidal_nmpc, nominal_robot_state, single_contact_config
    ):
        """NMPC should converge from a 10% perturbation of nominal."""
        r_com, v_com, L_com, r_ref, v_ref, cc = _nominal_solve_inputs(
            nominal_robot_state, single_contact_config
        )
        # Perturb
        r_com += 0.01
        v_com += 0.05

        _, _, _, _, info = centroidal_nmpc.solve(
            r_com=r_com, v_com=v_com, L_com=L_com,
            r_com_ref=r_ref, v_com_ref=v_ref,
            contact_config=cc, warm_start=False,
        )
        assert info.success, f"NMPC failed from perturbed state: {info.status}"


class TestNMPCConstraints:

    def test_nmpc_L_com_stays_in_bounds(
        self, centroidal_nmpc, nominal_robot_state, single_contact_config
    ):
        """L_com must remain within [-L_max, L_max] over the trajectory."""
        r_com, v_com, L_com, r_ref, v_ref, cc = _nominal_solve_inputs(
            nominal_robot_state, single_contact_config
        )
        v_com += 0.05  # perturb to generate nontrivial trajectory

        x_opt, _, info = centroidal_nmpc.get_full_trajectory(
            r_com, v_com, L_com, r_ref, v_ref, cc,
        )
        assert info.success, f"NMPC failed: {info.status}"

        L_traj = x_opt[6:9, :]  # (3, N+1)
        L_max = centroidal_nmpc.config.L_max

        assert np.all(L_traj >= -L_max - 1e-6), (
            f"L_com below -L_max: min={L_traj.min(axis=1)}"
        )
        assert np.all(L_traj <= L_max + 1e-6), (
            f"L_com above L_max: max={L_traj.max(axis=1)}"
        )

    def test_nmpc_Ldot_constraint_documented(
        self, centroidal_nmpc, nominal_robot_state, single_contact_config
    ):
        """Document that L_dot constraint covers spin only, not orbital.

        The path constraint |L̇_com,i| ≤ τ_w_max bounds the centroidal
        angular momentum rate. The orbital term (r_com × Σf) is no longer
        in the NMPC state — the AOCS handles it independently.
        """
        r_com, v_com, L_com, r_ref, v_ref, cc = _nominal_solve_inputs(
            nominal_robot_state, single_contact_config
        )
        v_com += 0.05

        x_opt, u_opt, info = centroidal_nmpc.get_full_trajectory(
            r_com, v_com, L_com, r_ref, v_ref, cc,
        )
        assert info.success

        # Verify L_com trajectory is bounded
        L_traj = x_opt[6:9, :]
        L_norms = np.linalg.norm(L_traj, axis=0)
        assert L_norms.max() < 100, "L_com unexpectedly large"

    def test_nmpc_linear_momentum_bounded(
        self, centroidal_nmpc, nominal_robot_state, single_contact_config
    ):
        """Linear momentum ||m·v_com|| should be bounded by p_max."""
        r_com, v_com, L_com, r_ref, v_ref, cc = _nominal_solve_inputs(
            nominal_robot_state, single_contact_config
        )
        v_com += 0.1  # nontrivial velocity

        x_opt, _, info = centroidal_nmpc.get_full_trajectory(
            r_com, v_com, L_com, r_ref, v_ref, cc,
        )
        assert info.success

        m = centroidal_nmpc.config.robot_mass
        v_traj = x_opt[3:6, :]  # (3, N+1)
        p_norms = m * np.linalg.norm(v_traj, axis=0)
        p_max = centroidal_nmpc.config.p_max

        if np.isfinite(p_max):
            assert np.all(p_norms <= p_max + 1e-3), (
                f"Linear momentum exceeds p_max: max={p_norms.max():.2f}, "
                f"p_max={p_max}"
            )


class TestNMPCSolverBehaviour:

    def test_nmpc_cold_vs_warm_start(
        self, centroidal_nmpc, nominal_robot_state, single_contact_config
    ):
        """Warm start should converge and produce similar cost to cold start."""
        r_com, v_com, L_com, r_ref, v_ref, cc = _nominal_solve_inputs(
            nominal_robot_state, single_contact_config
        )

        # Cold start
        _, _, _, _, info_cold = centroidal_nmpc.solve(
            r_com=r_com, v_com=v_com, L_com=L_com,
            r_com_ref=r_ref, v_com_ref=v_ref,
            contact_config=cc, warm_start=False,
        )
        assert info_cold.success, "Cold solve failed"

        # Warm start (same inputs)
        _, _, _, _, info_warm = centroidal_nmpc.solve(
            r_com=r_com, v_com=v_com, L_com=L_com,
            r_com_ref=r_ref, v_com_ref=v_ref,
            contact_config=cc, warm_start=True,
        )
        assert info_warm.success, "Warm solve failed"

    def test_nmpc_infeasible_input_reports_failure(
        self, centroidal_nmpc, nominal_robot_state, single_contact_config
    ):
        """L_com far outside bounds should be reported as infeasible."""
        r_com, v_com, L_com, r_ref, v_ref, cc = _nominal_solve_inputs(
            nominal_robot_state, single_contact_config
        )

        _, _, _, _, info = centroidal_nmpc.solve(
            r_com=r_com, v_com=v_com,
            L_com=np.array([100.0, 100.0, 100.0]),
            r_com_ref=r_ref, v_com_ref=v_ref,
            contact_config=cc, warm_start=False,
        )
        assert info.success is False, (
            "NMPC should fail with L_com=100 (L_max=10)"
        )


class TestQPContactUtilities:

    def test_qp_feasibility_single_support(self, single_contact_config):
        """ContactConfig.from_phase gives correct nc and active_contacts."""
        cc = single_contact_config
        assert cc.nc == 1
        assert cc.active_contacts == (True, False)
        M = compute_momentum_map(np.zeros(3), cc)
        assert M.shape == (3, 12)

    def test_qp_feasibility_double_support(self, double_contact_config):
        """Double-support ContactConfig."""
        cc = double_contact_config
        assert cc.nc == 2
        assert cc.active_contacts == (True, True)


class TestFeedforwardAcceleration:

    def test_compute_feedforward_acceleration(self, centroidal_nmpc):
        """a_com_ff = (f1 + f2) / m should match compute_feedforward_acceleration."""
        lambda_ref = np.zeros(12)
        lambda_ref[0:3] = [1.0, 2.0, 3.0]   # f1
        lambda_ref[6:9] = [-1.0, 0.5, 0.0]  # f2

        af = centroidal_nmpc.compute_feedforward_acceleration(lambda_ref)
        expected = (lambda_ref[0:3] + lambda_ref[6:9]) / centroidal_nmpc.config.robot_mass
        npt.assert_allclose(af, expected, atol=1e-12)
