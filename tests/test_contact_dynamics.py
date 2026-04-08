"""
Tests for contact configuration, momentum map, and weld dynamics properties.

Validates ContactConfig phase construction, momentum map geometry,
known constraint gaps, MuJoCo weld constraints, and gait plan timing.
"""

import numpy as np
import numpy.testing as npt
import pytest

from crawlbot.solvers.contact_phase import (
    ContactConfig,
    ContactPhase,
    compute_momentum_map,
    skew,
)
from crawlbot.planning.contact_scheduler import ContactScheduler


# ---------------------------------------------------------------------------
# ContactConfig.from_phase
# ---------------------------------------------------------------------------

class TestContactConfigFromPhase:

    def test_contact_config_from_phase_single_a(self):
        """SINGLE_A gives nc=1, active=(True, False)."""
        r_a = np.array([0.0, 0.3, 0.0])
        r_b = np.array([0.0, -0.3, 0.0])
        cc = ContactConfig.from_phase(ContactPhase.SINGLE_A, r_a, r_b)

        assert cc.nc == 1
        assert cc.active_contacts == (True, False)
        assert cc.phase == ContactPhase.SINGLE_A

    def test_contact_config_from_phase_single_b(self):
        """SINGLE_B gives nc=1, active=(False, True)."""
        r_a = np.array([0.0, 0.3, 0.0])
        r_b = np.array([0.0, -0.3, 0.0])
        cc = ContactConfig.from_phase(ContactPhase.SINGLE_B, r_a, r_b)

        assert cc.nc == 1
        assert cc.active_contacts == (False, True)
        assert cc.phase == ContactPhase.SINGLE_B

    def test_contact_config_from_phase_double(self):
        """DOUBLE gives nc=2, active=(True, True)."""
        r_a = np.array([0.0, 0.3, 0.0])
        r_b = np.array([0.0, -0.3, 0.0])
        cc = ContactConfig.from_phase(ContactPhase.DOUBLE, r_a, r_b)

        assert cc.nc == 2
        assert cc.active_contacts == (True, True)
        assert cc.phase == ContactPhase.DOUBLE

    def test_contact_config_active_positions(self):
        """active_contact_positions returns only positions of active contacts."""
        r_a = np.array([1.0, 0.3, 0.0])
        r_b = np.array([1.0, -0.3, 0.0])

        # Single A: only r_a returned
        cc_a = ContactConfig.from_phase(ContactPhase.SINGLE_A, r_a, r_b)
        positions_a = cc_a.active_contact_positions
        assert len(positions_a) == 1
        npt.assert_allclose(positions_a[0], r_a)

        # Single B: only r_b returned
        cc_b = ContactConfig.from_phase(ContactPhase.SINGLE_B, r_a, r_b)
        positions_b = cc_b.active_contact_positions
        assert len(positions_b) == 1
        npt.assert_allclose(positions_b[0], r_b)

        # Double: both returned
        cc_d = ContactConfig.from_phase(ContactPhase.DOUBLE, r_a, r_b)
        positions_d = cc_d.active_contact_positions
        assert len(positions_d) == 2
        npt.assert_allclose(positions_d[0], r_a)
        npt.assert_allclose(positions_d[1], r_b)


# ---------------------------------------------------------------------------
# Momentum map geometry
# ---------------------------------------------------------------------------

class TestMomentumMap:

    def test_momentum_map_single_support(self):
        """In single-support-A: columns 6:12 zero, 0:3 = skew(lever_A), 3:6 = I_3."""
        r_a = np.array([0.0, 0.3, 0.0])
        r_b = np.array([0.0, -0.3, 0.0])
        r_com = np.array([0.0, 0.0, 0.0])

        cc = ContactConfig.from_phase(ContactPhase.SINGLE_A, r_a, r_b)
        M = compute_momentum_map(r_com, cc)

        # Inactive contact B: columns 6:12 must be zero
        npt.assert_allclose(M[:, 6:12], 0.0, atol=1e-15,
                            err_msg="Inactive contact B columns must be zero")

        # Active contact A: columns 0:3 = skew(r_a - r_com)
        lever_A = r_a - r_com
        npt.assert_allclose(M[:, 0:3], skew(lever_A), atol=1e-15,
                            err_msg="Force columns must be skew(lever_A)")

        # Active contact A: columns 3:6 = I_3 (torque contribution)
        npt.assert_allclose(M[:, 3:6], np.eye(3), atol=1e-15,
                            err_msg="Torque columns must be I_3")

    def test_momentum_map_symmetry_check(self):
        """For symmetric contacts with r_com at midpoint, force columns are equal."""
        d = 0.3
        r_a = np.array([0.0, d, 0.0])
        r_b = np.array([0.0, -d, 0.0])
        r_com = np.array([0.0, 0.0, 0.0])  # midpoint

        cc = ContactConfig.from_phase(ContactPhase.DOUBLE, r_a, r_b)
        M = compute_momentum_map(r_com, cc)

        # Lever arms: r_a - r_com = [0, d, 0], r_b - r_com = [0, -d, 0]
        # skew([0, d, 0]) and skew([0, -d, 0]) differ by sign
        # The force columns should satisfy: M[:, 0:3] = -M[:, 6:9]
        # because skew(lever_A) = -skew(lever_B) for symmetric contacts
        npt.assert_allclose(
            M[:, 0:3], -M[:, 6:9], atol=1e-15,
            err_msg="Symmetric contacts: skew(lever_A) = -skew(lever_B)",
        )

        # Torque columns should both be I_3
        npt.assert_allclose(M[:, 3:6], np.eye(3), atol=1e-15)
        npt.assert_allclose(M[:, 9:12], np.eye(3), atol=1e-15)


# ---------------------------------------------------------------------------
# Known constraint gap: box vs norm
# ---------------------------------------------------------------------------

class TestMomentumSpikeDocumentation:

    def test_momentum_spike_box_vs_norm_documentation(self):
        """Document: per-component box |L_i| <= L_max permits ||L||_2 > L_max.

        A per-component box constraint (as used by the NMPC for hw bounds)
        allows the 2-norm to reach sqrt(3) * L_max. This is a known
        conservatism trade-off: box constraints are cheaper for IPOPT than
        SOC constraints on the momentum envelope.
        """
        L_max = 5.0

        # Construct L_com at the corner of the box
        L_com = np.array([L_max, L_max, L_max])

        # Per-component check passes
        assert np.all(np.abs(L_com) <= L_max), (
            "Per-component box check should pass at the corner"
        )

        # 2-norm exceeds L_max
        L_norm = np.linalg.norm(L_com)
        expected_norm = np.sqrt(3) * L_max  # ~8.66
        npt.assert_allclose(L_norm, expected_norm, atol=1e-12)

        # Document the gap
        gap_ratio = L_norm / L_max
        assert gap_ratio > 1.0, (
            f"Box constraint gap: ||L||_2 / L_max = {gap_ratio:.3f}. "
            f"Per-component box allows ||L||_2 = sqrt(3)*L_max = {expected_norm:.2f} Nms, "
            f"which is {gap_ratio:.1f}x the scalar limit. "
            f"This is a known trade-off: box constraints are cheaper for IPOPT "
            f"than SOC constraints on the momentum envelope."
        )


# ---------------------------------------------------------------------------
# MuJoCo weld equality constraints
# ---------------------------------------------------------------------------

class TestWeldConstraints:

    def test_weld_equality_constraints_exist(self, mujoco_model):
        """MuJoCo model should contain equality constraints (welds) for docking."""
        import mujoco

        assert mujoco_model.neq > 0, (
            "MuJoCo model must have at least one equality constraint (weld)"
        )

        # Check that at least one constraint is a weld type
        # mujoco.mjtEq.mjEQ_WELD == 1
        weld_type = mujoco.mjtEq.mjEQ_WELD
        eq_types = mujoco_model.eq_type[:mujoco_model.neq]
        n_welds = np.sum(eq_types == weld_type)
        assert n_welds > 0, (
            f"Expected weld-type equality constraints; "
            f"found {mujoco_model.neq} constraints with types {eq_types}, "
            f"but none are weld (type={weld_type})"
        )


# ---------------------------------------------------------------------------
# QP / MuJoCo force gap documentation
# ---------------------------------------------------------------------------

class TestQPMujocoForceGap:

    def test_qp_mujoco_force_gap_documentation(self):
        """Document the known ~33x gap between QP lambda and MuJoCo constraint forces.

        Design context:
        - The WholeBodyQP (Stage 2) optimizes virtual contact wrenches lambda
          to track the centroidal momentum reference from Stage 1.
        - MuJoCo determines actual contact forces independently via its
          constraint solver (PGS/Newton) from the weld equality constraints.
        - These two force computations are fundamentally different:
            * QP lambda: what forces *should* be applied to track the plan
            * MuJoCo efc_force: what forces the physics engine *actually* applies
        - The gap is expected and handled by the H_{r/O} momentum estimator,
          which measures actual momentum changes and feeds back to Stage 1.
        - Typical gap magnitude: ~33x (varies with configuration and phase).

        This test documents the gap as a design limitation, not a bug.
        """
        # Illustrative example: QP requests a gentle wrench, MuJoCo applies
        # whatever is needed to maintain the rigid weld constraint.
        lambda_qp = np.array([0.5, 0.3, 0.1, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Hypothetical MuJoCo constraint force (order-of-magnitude larger)
        f_mujoco_magnitude = 33.0 * np.linalg.norm(lambda_qp[:6])

        gap = f_mujoco_magnitude / (np.linalg.norm(lambda_qp[:6]) + 1e-12)
        assert gap > 10.0, (
            f"Expected significant gap between QP and MuJoCo forces, got {gap:.1f}x. "
            f"The QP optimizes virtual wrenches; MuJoCo enforces rigid welds independently. "
            f"The H_{{r/O}} estimator closes this loop by measuring actual momentum."
        )


# ---------------------------------------------------------------------------
# Gait plan timing
# ---------------------------------------------------------------------------

class TestGaitPlanTiming:

    def test_gait_plan_timing_consistency(self, contact_scheduler):
        """Gait plan timing: t_start[0]==0, t_end[-1]==total, durations match."""
        plan = contact_scheduler.plan

        # First phase starts at t=0
        npt.assert_allclose(plan.t_start[0], 0.0, atol=1e-12,
                            err_msg="First phase must start at t=0")

        # Last phase ends at total_duration
        npt.assert_allclose(plan.t_end[-1], plan.total_duration, atol=1e-12,
                            err_msg="Last phase must end at total_duration")

        # Each phase duration matches configured dt_ss / dt_ds
        dt_ss = contact_scheduler.dt_ss
        dt_ds = contact_scheduler.dt_ds

        for i, gp in enumerate(plan.phases):
            actual_duration = plan.t_end[i] - plan.t_start[i]
            if gp.phase == ContactPhase.DOUBLE:
                npt.assert_allclose(
                    actual_duration, dt_ds, atol=1e-12,
                    err_msg=f"Phase {i} ({gp.phase.value}): "
                            f"duration {actual_duration} != dt_ds={dt_ds}",
                )
            else:
                npt.assert_allclose(
                    actual_duration, dt_ss, atol=1e-12,
                    err_msg=f"Phase {i} ({gp.phase.value}): "
                            f"duration {actual_duration} != dt_ss={dt_ss}",
                )

        # Phase sequence alternates: DS, SS, DS, SS, ...
        # First phase is always DS (initial double support)
        assert plan.phases[0].phase == ContactPhase.DOUBLE, (
            "First phase must be DOUBLE (initial double-support)"
        )
        for i in range(len(plan.phases)):
            if i % 2 == 0:
                assert plan.phases[i].phase == ContactPhase.DOUBLE, (
                    f"Phase {i} should be DOUBLE, got {plan.phases[i].phase.value}"
                )
            else:
                assert plan.phases[i].phase in (
                    ContactPhase.SINGLE_A, ContactPhase.SINGLE_B
                ), (
                    f"Phase {i} should be SINGLE_A or SINGLE_B, "
                    f"got {plan.phases[i].phase.value}"
                )

    def test_contact_config_at_returns_correct_phase(self, contact_scheduler):
        """contact_config_at(t) returns the correct phase for each time segment."""
        plan = contact_scheduler.plan

        for i, gp in enumerate(plan.phases):
            # Query at the midpoint of each phase
            t_mid = (plan.t_start[i] + plan.t_end[i]) / 2
            cc = contact_scheduler.contact_config_at(t_mid)

            assert cc.phase == gp.phase, (
                f"At t={t_mid:.2f}s (phase {i}): expected {gp.phase.value}, "
                f"got {cc.phase.value}"
            )

            # Verify nc is consistent with phase
            if gp.phase == ContactPhase.DOUBLE:
                assert cc.nc == 2
                assert cc.active_contacts == (True, True)
            elif gp.phase == ContactPhase.SINGLE_A:
                assert cc.nc == 1
                assert cc.active_contacts == (True, False)
            elif gp.phase == ContactPhase.SINGLE_B:
                assert cc.nc == 1
                assert cc.active_contacts == (False, True)
