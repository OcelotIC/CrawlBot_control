"""Targeted tests for code-level blocking points (liabilities) in CrawlBot_control.

Each test group exposes a specific class of latent defect: silent failures,
hardcoded tolerances, unchecked divisions, missing validation, stale warm-starts,
and inconsistent configs.  Where a gap is found (no runtime error on bad input),
the docstring documents the finding so it can be tracked as tech-debt.

Run:
    PYTHONPATH=. MUJOCO_GL=disabled pytest tests/test_liabilities.py -v
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Imports from crawlbot
# ---------------------------------------------------------------------------
from crawlbot.solvers.centroidal_nmpc import CentroidalNMPC, CentroidalNMPCConfig
from crawlbot.solvers.nmpc_solver import NMPCSolver, NMPCSolveInfo
from crawlbot.solvers.contact_phase import ContactConfig, ContactPhase
from crawlbot.core.state_conversions import quat_wxyz_to_euler_deg
from crawlbot.core.robot_interface import RobotInterface
from crawlbot.simulation.config import SimConfig

# Path constant (shared with conftest)
from pathlib import Path
URDF_PATH = str(Path(__file__).resolve().parent.parent / "models" / "VISPA_crawling_fixed.urdf")


# ===================================================================== #
#  Helpers                                                                #
# ===================================================================== #

def _make_double_contact():
    """Double-support contact config at a simple symmetric configuration."""
    return ContactConfig.from_phase(
        ContactPhase.DOUBLE,
        r_contact_A=np.array([0.0, 0.3, 0.0]),
        r_contact_B=np.array([0.0, -0.3, 0.0]),
    )


def _make_feasible_nmpc():
    """Build a CentroidalNMPC with finite constraint bounds (tau_w_max, L_max)
    so that IPOPT does not encounter inf - inf = NaN in the constraint
    evaluation."""
    cfg = CentroidalNMPCConfig(
        N=8, dt=0.1,
        robot_mass=71.0,
        f_max=25.0, tau_max=8.0,
        hw_min=-5.0 * np.ones(3),
        hw_max=5.0 * np.ones(3),
        tau_w_max=5.0,
        L_max=10.0,
        solver_opts={'ipopt.print_level': 0, 'print_time': 0,
                     'ipopt.max_iter': 200},
    )
    nmpc = CentroidalNMPC(cfg)
    nmpc.build()
    return nmpc


def _feasible_nmpc_inputs():
    """Return a dict of clearly feasible inputs for CentroidalNMPC.solve().

    Uses a small positive z-offset for r_com so the CoM is not exactly
    coplanar with the contact points (avoids degenerate geometry).
    """
    r_com = np.array([0.0, 0.0, 0.05])
    return dict(
        r_com=r_com,
        v_com=np.zeros(3),
        L_com=np.zeros(3),
        hw_current=np.zeros(3),
        r_com_ref=r_com.copy(),
        v_com_ref=np.zeros(3),
        contact_config=_make_double_contact(),
        warm_start=False,
    )


# ===================================================================== #
#  L1 -- Silent exception swallowing                                      #
# ===================================================================== #

class TestL1SilentExceptionSwallowing:
    """sim_loop.py:529-532, 595-597 silently fall back to zeros on solver
    failure.  We cannot easily mock SimulationLoop internals, so we verify
    that the underlying solvers properly report failure via info.success."""

    def test_nmpc_infeasible_hw_reports_failure(self):
        """hw_current far outside [hw_min, hw_max] should make the problem
        infeasible.  Verify info.success == False."""
        nmpc = _make_feasible_nmpc()
        cc = _make_double_contact()

        # hw_min/hw_max are +/-5 Nms; push hw_current to +/-100
        _, _, _, _, _, info = nmpc.solve(
            r_com=np.array([0.0, 0.0, 0.05]),
            v_com=np.zeros(3),
            L_com=np.zeros(3),
            hw_current=np.array([100.0, 100.0, 100.0]),
            r_com_ref=np.array([0.0, 0.0, 0.05]),
            v_com_ref=np.zeros(3),
            contact_config=cc,
            warm_start=False,
        )
        assert info.success is False, (
            "NMPC should report failure when hw_current is far outside "
            "the momentum envelope, but info.success was True."
        )

    def test_nmpc_nan_input_does_not_silently_succeed(self):
        """Passing NaN in the initial state must not silently return
        info.success == True."""
        nmpc = _make_feasible_nmpc()
        cc = _make_double_contact()

        r_com_nan = np.array([np.nan, 0.0, 0.0])
        _, _, _, _, _, info = nmpc.solve(
            r_com=r_com_nan,
            v_com=np.zeros(3),
            L_com=np.zeros(3),
            hw_current=np.zeros(3),
            r_com_ref=np.zeros(3),
            v_com_ref=np.zeros(3),
            contact_config=cc,
            warm_start=False,
        )
        # At minimum, the solver should not claim success
        assert info.success is False, (
            "NMPC returned success=True with NaN input.  "
            "This is the silent-swallowing path that sim_loop masks."
        )


# ===================================================================== #
#  L2 -- Hardcoded numerical tolerances in IK                             #
# ===================================================================== #

class TestL2HardcodedIKTolerances:
    """crawlbot/core/ik.py uses hardcoded regularization (1e-4 for arm
    Jacobian, 1e-3 for base Jacobian).  Verify robustness near singularity
    and basic convergence for well-conditioned cases."""

    def test_ik_near_singular_no_nan(self, robot_interface):
        """Create a near-singular IK problem (target almost at full arm
        extension).  The returned q must have no NaN/Inf."""
        import pinocchio as pin
        from crawlbot.core.ik import solve_ik, FRAME_TOOL_A, FRAME_TOOL_B

        model = robot_interface.model
        q0 = pin.neutral(model)

        # Place targets unreachably far to stress the solver
        far_target_a = pin.SE3(np.eye(3), np.array([2.0, 0.5, 0.0]))
        far_target_b = pin.SE3(np.eye(3), np.array([2.0, -0.5, 0.0]))
        targets = {FRAME_TOOL_A: far_target_a, FRAME_TOOL_B: far_target_b}

        q, err = solve_ik(model, q0, targets, max_iter=200, tol=1e-8)

        assert np.all(np.isfinite(q)), (
            f"solve_ik returned NaN/Inf in q: {q}"
        )

    def test_ik_well_conditioned_converges(self, robot_interface):
        """For a well-conditioned target (near neutral pose frames),
        IK should converge to low error."""
        import pinocchio as pin
        from crawlbot.core.ik import solve_ik, FRAME_TOOL_A, FRAME_TOOL_B

        model = robot_interface.model
        q0 = pin.neutral(model)
        data = model.createData()

        # Use the frame placements at neutral as targets -- trivially reachable
        pin.forwardKinematics(model, data, q0)
        pin.updateFramePlacements(model, data)
        target_a = data.oMf[FRAME_TOOL_A].copy()
        target_b = data.oMf[FRAME_TOOL_B].copy()

        targets = {FRAME_TOOL_A: target_a, FRAME_TOOL_B: target_b}
        q, err = solve_ik(model, q0, targets, max_iter=500, tol=1e-8)

        assert err < 1e-4, f"IK did not converge for well-conditioned problem: err={err:.2e}"
        np.testing.assert_allclose(
            q[7:19], q0[7:19], atol=0.1,
            err_msg="Joint angles changed significantly for identity target",
        )

    def test_dock_configuration_well_conditioned(self, robot_interface):
        """dock_configuration should succeed for reachable anchor poses."""
        import pinocchio as pin
        from crawlbot.core.ik import dock_configuration

        model = robot_interface.model
        q0 = pin.neutral(model)
        data = model.createData()
        pin.forwardKinematics(model, data, q0)
        pin.updateFramePlacements(model, data)

        from crawlbot.core.ik import FRAME_TOOL_A, FRAME_TOOL_B
        anchor_a = data.oMf[FRAME_TOOL_A].copy()
        anchor_b = data.oMf[FRAME_TOOL_B].copy()

        q = dock_configuration(model, anchor_a, anchor_b)
        assert np.all(np.isfinite(q)), "dock_configuration returned NaN/Inf"


# ===================================================================== #
#  L3 -- Unchecked division edge cases                                    #
# ===================================================================== #

class TestL3UncheckedDivision:
    """robot_interface.py:115 uses max(old_I.mass, 1e-6) -- safe but the
    resulting inertia scaling may be wrong when torso_mass=0.
    state_conversions.py:quat_wxyz_to_euler_deg should handle gimbal lock."""

    def test_robot_interface_zero_torso_mass_no_crash(self):
        """RobotInterface with torso_mass=0.0 should not crash.

        Liability: ratio = 0/max(mass, 1e-6) = 0 -> all inertia zeroed.
        This is physically nonsensical but should not raise."""
        ri = RobotInterface(URDF_PATH, tau_max=20.0, gravity='zero',
                            torso_mass=0.0)
        assert ri.model is not None
        # Verify computation still works at neutral
        import pinocchio as pin
        q = pin.neutral(ri.model)
        v = np.zeros(ri.model.nv)
        state = ri.update(q, v)
        assert np.all(np.isfinite(state.r_com)), "r_com contains NaN/Inf with zero torso mass"

    @pytest.mark.parametrize("qw,qx,qy,qz,description", [
        # Pure 90-degree pitch -> gimbal lock
        (np.sqrt(2) / 2, 0.0, np.sqrt(2) / 2, 0.0, "pitch +90 deg (gimbal lock)"),
        # Pure -90-degree pitch
        (np.sqrt(2) / 2, 0.0, -np.sqrt(2) / 2, 0.0, "pitch -90 deg (gimbal lock)"),
        # Identity quaternion
        (1.0, 0.0, 0.0, 0.0, "identity"),
        # 180-degree rotation about z
        (0.0, 0.0, 0.0, 1.0, "180 deg yaw"),
        # Near-zero quaternion (unnormalized edge case)
        (1e-10, 1e-10, 1e-10, 1e-10, "near-zero quaternion"),
    ])
    def test_quat_to_euler_no_nan(self, qw, qx, qy, qz, description):
        """quat_wxyz_to_euler_deg must never produce NaN, even at gimbal
        lock or with degenerate quaternions."""
        result = quat_wxyz_to_euler_deg(qw, qx, qy, qz)
        assert result.shape == (3,), f"Expected shape (3,), got {result.shape}"
        assert np.all(np.isfinite(result)), (
            f"NaN/Inf in euler output for {description}: {result}"
        )

    def test_quat_to_euler_gimbal_lock_values(self):
        """At exact +90 deg pitch, arcsin(sinp) should be clipped to +/-1
        and pitch should be exactly +/-90 degrees."""
        # Pure pitch +90: qw=cos(45), qy=sin(45)
        c = np.cos(np.pi / 4)
        s = np.sin(np.pi / 4)
        result = quat_wxyz_to_euler_deg(c, 0.0, s, 0.0)
        np.testing.assert_allclose(
            result[1], 90.0, atol=1e-10,
            err_msg="Pitch should be exactly 90 deg at gimbal lock"
        )


# ===================================================================== #
#  L4 -- Missing solver input validation                                  #
# ===================================================================== #

class TestL4MissingSolverValidation:
    """NMPCSolver.solve() and CentroidalNMPC.solve() accept inputs without
    validating shapes.  Document what happens with wrong-sized inputs."""

    def test_nmpc_solver_wrong_x0_size(self):
        """NMPCSolver.solve() with wrong-sized x0 should raise or clearly
        fail, not silently produce garbage.

        GAP: CasADi may raise a dimension error at the parameter vector
        concatenation, but this depends on the CasADi version."""
        import casadi as ca

        nmpc = NMPCSolver(nx=4, nu=2, N=5, dt=0.1)
        nmpc.set_continuous_dynamics(lambda x, u, p: -x)
        nmpc.set_stage_cost(lambda x, u, p: ca.dot(x, x) + ca.dot(u, u))
        nmpc.build({'ipopt.print_level': 0, 'print_time': 0,
                    'ipopt.max_iter': 50})

        # Correct size would be (4,), try (5,)
        x0_wrong = np.zeros(5)
        with pytest.raises(Exception):
            # Should raise due to parameter dimension mismatch
            nmpc.solve(x0_wrong, warm_start=False)

    def test_centroidal_nmpc_wrong_rcom_size(self):
        """CentroidalNMPC.solve() with wrong-length r_com should raise.

        r_com is expected to be shape (3,); passing shape (2,) should
        fail at np.concatenate or CasADi parameter dimension check."""
        nmpc = _make_feasible_nmpc()
        cc = _make_double_contact()

        with pytest.raises((ValueError, Exception)):
            nmpc.solve(
                r_com=np.zeros(2),  # Wrong shape!
                v_com=np.zeros(3),
                L_com=np.zeros(3),
                hw_current=np.zeros(3),
                r_com_ref=np.zeros(3),
                v_com_ref=np.zeros(3),
                contact_config=cc,
                warm_start=False,
            )

    def test_nmpc_solver_empty_params_when_expected(self):
        """When NMPCSolver expects parameters but none are provided,
        the solve should fail at the parameter vector construction.

        GAP DOCUMENTED: NMPCSolver does not validate params size before
        calling CasADi.  CasADi catches it at the low level."""
        import casadi as ca

        nmpc = NMPCSolver(nx=2, nu=1, N=3, dt=0.1)
        nmpc.set_parameters(4)  # Expect 4 parameters
        nmpc.set_continuous_dynamics(lambda x, u, p: -x)
        nmpc.set_stage_cost(lambda x, u, p: ca.dot(x, x) + ca.dot(u, u))
        nmpc.build({'ipopt.print_level': 0, 'print_time': 0,
                    'ipopt.max_iter': 50})

        x0 = np.zeros(2)
        # Don't provide params when 4 are expected
        with pytest.raises(Exception):
            nmpc.solve(x0, warm_start=False)


# ===================================================================== #
#  L5 -- Stale warm-start after partial failure                           #
# ===================================================================== #

class TestL5StaleWarmStart:
    """nmpc_solver.py:457-460 stores warm-start data even on solver failure.
    Verify that a failed solve does not corrupt subsequent feasible solves."""

    def test_warm_start_survives_failed_solve(self):
        """Sequence: feasible solve -> infeasible solve -> feasible solve.
        The third solve should still succeed despite the intermediate
        failure having stored its (bad) warm-start data.

        Uses finite tau_w_max and L_max to avoid inf-inf=NaN in constraint
        evaluation, which would mask the actual warm-start corruption issue."""
        nmpc = _make_feasible_nmpc()
        cc = _make_double_contact()

        # Step 1: Feasible solve
        result1 = nmpc.solve(**_feasible_nmpc_inputs())
        info1 = result1[-1]
        assert info1.success, (
            f"First feasible solve should succeed, got status={info1.status}"
        )

        # Step 2: Infeasible solve (hw far outside bounds)
        result2 = nmpc.solve(
            r_com=np.array([0.0, 0.0, 0.05]),
            v_com=np.zeros(3),
            L_com=np.zeros(3),
            hw_current=np.array([100.0, 100.0, 100.0]),
            r_com_ref=np.array([0.0, 0.0, 0.05]),
            v_com_ref=np.zeros(3),
            contact_config=cc,
            warm_start=True,
        )
        info2 = result2[-1]
        # This should fail (infeasible)
        assert info2.success is False, (
            "Infeasible solve unexpectedly succeeded -- "
            "hw_current=100 is far outside [hw_min, hw_max]=[-5,5]"
        )

        # Step 3: Feasible solve again -- should still work
        result3 = nmpc.solve(**_feasible_nmpc_inputs())
        info3 = result3[-1]
        assert info3.success, (
            "Third feasible solve failed after an intermediate infeasible solve. "
            "This suggests the warm-start data from the failed solve corrupted "
            "the solver state (nmpc_solver.py:457-460 stores warm-start "
            "unconditionally, even on failure)."
        )


# ===================================================================== #
#  L6 -- Config consistency                                               #
# ===================================================================== #

class TestL6ConfigConsistency:
    """SimConfig has dt_nmpc and dt_qp whose ratio must be integer for
    correct sub-stepping, and hw_min/hw_max should satisfy hw_min < hw_max.
    Currently there is NO validation on either."""

    def test_non_integer_dt_ratio_allowed(self):
        """GAP: SimConfig(dt_nmpc=0.1, dt_qp=0.03) creates a non-integer
        ratio (10/3 = 3.333...).  This would cause incorrect sub-stepping
        in SimulationLoop but is silently accepted by the config."""
        cfg = SimConfig(dt_nmpc=0.1, dt_qp=0.03)
        ratio = cfg.dt_nmpc / cfg.dt_qp
        assert cfg.dt_nmpc == 0.1
        assert cfg.dt_qp == 0.03
        # Document the gap: ratio is not integer
        assert not float(ratio).is_integer(), (
            "Expected non-integer ratio to be accepted (no validation). "
            "If this fails, validation was added -- update this test."
        )

    def test_inverted_hw_bounds_allowed(self):
        """GAP: SimConfig allows hw_min > hw_max.  This would make all
        NMPC problems infeasible but is not caught at config creation.

        If this test starts failing, it means validation was added."""
        cfg = SimConfig(
            hw_min=np.full(3, 10.0),
            hw_max=np.full(3, -10.0),
        )
        assert np.all(cfg.hw_min > cfg.hw_max), (
            "Expected inverted bounds to be accepted (no validation)."
        )

    def test_valid_config_sanity(self):
        """Baseline: default SimConfig should have sensible values."""
        cfg = SimConfig()
        ratio = cfg.dt_nmpc / cfg.dt_qp
        assert float(ratio).is_integer(), (
            f"Default dt_nmpc/dt_qp ratio should be integer, got {ratio}"
        )
        np.testing.assert_array_less(
            cfg.hw_min, cfg.hw_max,
            err_msg="Default hw_min should be < hw_max"
        )

    def test_zero_dt_qp_raises_or_creates(self):
        """GAP: dt_qp=0 would cause ZeroDivisionError in SimulationLoop's
        ratio computation, but the config itself does not prevent it."""
        # Should create without error (no validation)
        cfg = SimConfig(dt_qp=0.0)
        assert cfg.dt_qp == 0.0
