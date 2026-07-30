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


# ---------------------------------------------------------------------------
# F1 (NMPC_AUDIT): per-knot reference parameters
# ---------------------------------------------------------------------------

class TestPerStageReferences:
    """The NLP must be able to track a trajectory, not only a setpoint.

    Before F1 the NLP carried ONE parameter block shared by every stage, so any
    reference in `p` was necessarily constant over the horizon. `sim_loop`
    compensated by sampling the reference at the horizon END, which is what tied
    `nmpc_N` to the reference lead — changing the horizon changed the target, so
    no clean horizon ablation was possible.

    These tests pin the three properties the switch has to have: the block count
    expands without changing the problem size, a broadcast reference reproduces
    the legacy problem exactly (so the refactor is provably inert), and a
    varying reference actually changes the solution (so the blocks are not
    silently ignored).
    """

    @staticmethod
    def _make(per_stage, N=6):
        cfg = CentroidalNMPCConfig(
            robot_mass=71.056, N=N, dt=0.1,
            f_max=300.0, tau_max=8.0, L_max=10.0, tau_w_max=2.5, p_max=50.0,
            per_stage_refs=per_stage)
        nmpc = CentroidalNMPC(cfg)
        nmpc.build()
        return nmpc

    def test_block_count_and_problem_size(self):
        """N+1 parameter blocks; decision vars and constraints unchanged."""
        N = 6
        legacy, staged = self._make(False, N), self._make(True, N)
        assert legacy._nmpc.n_param_blocks == 1
        assert staged._nmpc.n_param_blocks == N + 1
        NP = CentroidalNMPC.NP
        assert legacy._nmpc._np_total == CentroidalNMPC.NX + NP
        assert staged._nmpc._np_total == CentroidalNMPC.NX + (N + 1) * NP
        # Only the parameterization changes — not the NLP's size.
        assert len(legacy._nmpc._lbw) == len(staged._nmpc._lbw)
        assert len(legacy._nmpc._lbg) == len(staged._nmpc._lbg)

    def test_broadcast_reference_reproduces_legacy(
            self, nominal_robot_state, single_contact_config):
        """A single reference under per_stage must equal the legacy solution."""
        r, v, L, r_ref, v_ref, cc = _nominal_solve_inputs(
            nominal_robot_state, single_contact_config)
        r_ref = r_ref + np.array([0.05, 0.0, 0.0])
        kw = dict(r_com=r, v_com=v, L_com=L, r_com_ref=r_ref, v_com_ref=v_ref,
                  contact_config=cc, warm_start=False)
        a = self._make(False).solve(**kw)
        b = self._make(True).solve(**kw)
        assert a[4].success and b[4].success
        npt.assert_allclose(b[0], a[0], atol=1e-9)   # r_com_plan
        npt.assert_allclose(b[3], a[3], atol=1e-6)   # lambda_0
        assert abs(a[4].cost - b[4].cost) <= 1e-7 * max(abs(a[4].cost), 1.0)

    def test_varying_reference_changes_the_solution(
            self, nominal_robot_state, single_contact_config):
        """Otherwise the per-stage blocks would be decorative."""
        N = 6
        r, v, L, r_ref, v_ref, cc = _nominal_solve_inputs(
            nominal_robot_state, single_contact_config)
        nmpc = self._make(True, N)
        kw = dict(r_com=r, v_com=v, L_com=L, contact_config=cc,
                  warm_start=False)
        flat = nmpc.solve(r_com_ref=r_ref, v_com_ref=v_ref, **kw)
        ramp = np.stack([r_ref + np.array([0.03, 0.0, 0.0]) * k
                         for k in range(N + 1)])
        varied = nmpc.solve(r_com_ref=ramp,
                            v_com_ref=np.tile(v_ref, (N + 1, 1)), **kw)
        assert flat[4].success and varied[4].success
        assert np.max(np.abs(varied[0] - flat[0])) > 1e-6, (
            'a ramped reference produced the same plan as a constant one — '
            'the per-stage parameter blocks are not reaching the NLP')

    def test_per_knot_reference_rejected_by_legacy_nlp(
            self, nominal_robot_state, single_contact_config):
        """Silently dropping the extra knots would be the dangerous failure."""
        N = 6
        r, v, L, r_ref, v_ref, cc = _nominal_solve_inputs(
            nominal_robot_state, single_contact_config)
        with pytest.raises(ValueError):
            self._make(False, N).solve(
                r_com=r, v_com=v, L_com=L,
                r_com_ref=np.tile(r_ref, (N + 1, 1)), v_com_ref=v_ref,
                contact_config=cc, warm_start=False)

    def test_wrong_knot_count_rejected(
            self, nominal_robot_state, single_contact_config):
        """N rows instead of N+1 must raise, not broadcast or truncate."""
        N = 6
        r, v, L, r_ref, v_ref, cc = _nominal_solve_inputs(
            nominal_robot_state, single_contact_config)
        with pytest.raises(ValueError):
            self._make(True, N).solve(
                r_com=r, v_com=v, L_com=L,
                r_com_ref=np.tile(r_ref, (N, 1)), v_com_ref=v_ref,
                contact_config=cc, warm_start=False)
