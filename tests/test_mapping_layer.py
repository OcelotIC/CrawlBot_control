"""M1 — CoM-to-torso mapping layer tests.

Covers the T3 Jacobian equivalence test from the HANDOFF doc:

    J_b_pos = (m_total/m_b) * J_com - (1/m_b) * sum_{i != torso} m_i * J_i

This identity is the mathematical justification for letting the whole-body
QP track torso position (rigid-body task) rather than CoM position
(centroidal task). It is an algebraic equivalence, so the error should be
at roundoff level.
"""

import numpy as np
import pinocchio as pin
import pytest

from crawlbot.core.robot_interface import RobotInterface
from crawlbot.core.com_to_torso_mapping import CoMToTorsoMapping, TORSO_JOINT_IDX


URDF = 'models/VISPA_crawling_fixed.urdf'


@pytest.fixture(scope='module')
def mapping():
    ri = RobotInterface(URDF, gravity='zero')
    return CoMToTorsoMapping(ri)


def _random_config(model, rng):
    """Sample a random configuration with valid free-flyer quaternion."""
    q = pin.randomConfiguration(
        model,
        np.full(model.nq, -np.pi),
        np.full(model.nq, np.pi),
    )
    # Override free-flyer position/orientation with a random but valid SE3.
    q[0:3] = rng.uniform(-1.0, 1.0, size=3)
    quat = rng.normal(size=4)
    quat /= np.linalg.norm(quat)
    q[3:7] = quat  # xyzw convention in Pinocchio
    return q


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_masses_consistent(self, mapping):
        assert mapping.m_total > 0
        assert mapping.m_b > 0
        assert np.isclose(mapping.m_b + float(np.sum(mapping.m_others)),
                          mapping.m_total, atol=1e-6)

    def test_ratio_greater_than_one(self, mapping):
        # Full robot heavier than torso alone
        assert mapping.ratio > 1.0

    def test_non_torso_joints_exclude_torso(self, mapping):
        assert TORSO_JOINT_IDX not in mapping.non_torso_joints
        assert len(mapping.non_torso_joints) == len(mapping.m_others)


# ---------------------------------------------------------------------------
# delta(q)
# ---------------------------------------------------------------------------

class TestDelta:
    def test_delta_at_neutral_nonzero(self, mapping):
        q = pin.neutral(mapping.model)
        delta = mapping.compute_delta(q)
        assert delta.shape == (3,)
        # At neutral, arms extend out so at least one component is nonzero
        assert np.linalg.norm(delta) > 1e-6

    def test_delta_shifts_with_base(self, mapping):
        """If we translate the base by (dx,dy,dz), delta shifts by
        (m_total - m_b) * (dx,dy,dz)."""
        q0 = pin.neutral(mapping.model)
        shift = np.array([0.3, -0.1, 0.2])
        q1 = q0.copy()
        q1[:3] += shift
        delta0 = mapping.compute_delta(q0)
        delta1 = mapping.compute_delta(q1)
        expected = (mapping.m_total - mapping.m_b) * shift
        assert np.allclose(delta1 - delta0, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# T3 — Jacobian equivalence (the pass criterion from HANDOFF)
# ---------------------------------------------------------------------------

class TestT3JacobianEquivalence:
    def test_identity_holds_at_neutral(self, mapping):
        q = pin.neutral(mapping.model)
        data = mapping.model.createData()
        pin.forwardKinematics(mapping.model, data, q)
        pin.computeJointJacobians(mapping.model, data, q)

        J_b_direct = mapping.body_com_jacobian(data, TORSO_JOINT_IDX)
        J_b_mapped = mapping.torso_pos_jacobian_from_com(q)

        err = float(np.max(np.abs(J_b_direct - J_b_mapped)))
        assert err < 1e-10, f"Identity error = {err:.2e} exceeds 1e-10"

    @pytest.mark.parametrize('seed', list(range(10)))
    def test_identity_at_random_configs(self, mapping, seed):
        """Verify the identity at 10 random configurations."""
        rng = np.random.default_rng(seed)
        q = _random_config(mapping.model, rng)

        data = mapping.model.createData()
        pin.forwardKinematics(mapping.model, data, q)
        pin.computeJointJacobians(mapping.model, data, q)

        J_b_direct = mapping.body_com_jacobian(data, TORSO_JOINT_IDX)
        J_b_mapped = mapping.torso_pos_jacobian_from_com(q)

        err = float(np.max(np.abs(J_b_direct - J_b_mapped)))
        assert err < 1e-10, \
            f"seed={seed}: identity error = {err:.2e} exceeds 1e-10"

    def test_identity_is_centroidal_decomposition(self, mapping):
        """Cross-check: (m_b * J_b + sum_i m_i * J_i) / m_total == J_com."""
        rng = np.random.default_rng(42)
        q = _random_config(mapping.model, rng)

        data = mapping.model.createData()
        pin.forwardKinematics(mapping.model, data, q)
        pin.computeJointJacobians(mapping.model, data, q)
        pin.centerOfMass(mapping.model, data, q)
        J_com = pin.jacobianCenterOfMass(mapping.model, data, q)

        J_b = mapping.body_com_jacobian(data, TORSO_JOINT_IDX)
        J_sum_others = np.zeros_like(J_com)
        for i, m_i in zip(mapping.non_torso_joints, mapping.m_others):
            J_sum_others += m_i * mapping.body_com_jacobian(data, i)

        J_com_reconstructed = (mapping.m_b * J_b + J_sum_others) / mapping.m_total
        err = float(np.max(np.abs(J_com - J_com_reconstructed)))
        assert err < 1e-10, f"Centroidal decomposition error = {err:.2e}"


# ---------------------------------------------------------------------------
# Forward mapping: compute() signature and round-trip consistency
# ---------------------------------------------------------------------------

class TestForwardMapping:
    def test_compute_returns_four_arrays(self, mapping):
        q = pin.neutral(mapping.model)
        dq = np.zeros(mapping.model.nv)
        r_com_ref = np.array([0.5, 0.0, -0.3])
        v_com_ref = np.array([0.1, 0.0, 0.0])
        a_com_ff = np.array([0.0, 0.0, 0.01])

        r_b, v_b, a_b, delta = mapping.compute(
            r_com_ref, v_com_ref, a_com_ff, q, dq)

        assert r_b.shape == (3,)
        assert v_b.shape == (3,)
        assert a_b.shape == (3,)
        assert delta.shape == (3,)

    def test_velocity_mapping_linear(self, mapping):
        """v_b_ref = ratio * v_com_ref  (drop delta_dot in v0)."""
        q = pin.neutral(mapping.model)
        v_com_ref = np.array([0.2, -0.1, 0.3])
        _, v_b, _, _ = mapping.compute(
            np.zeros(3), v_com_ref, np.zeros(3), q)
        assert np.allclose(v_b, mapping.ratio * v_com_ref, atol=1e-12)

    def test_accel_mapping_linear(self, mapping):
        q = pin.neutral(mapping.model)
        a_com = np.array([0.05, 0.02, -0.01])
        _, _, a_b, _ = mapping.compute(
            np.zeros(3), np.zeros(3), a_com, q)
        assert np.allclose(a_b, mapping.ratio * a_com, atol=1e-12)

    @pytest.mark.parametrize('seed', list(range(10)))
    def test_self_consistency_at_current_state(self, mapping, seed):
        """Consistency: when r_com_ref = r_com(q_current), the mapping
        must return r_b_ref = r_b(q_current).

        Proof:
            ratio * r_com(q) - delta(q)/m_b
              = (m_total/m_b) * (m_b*r_b + delta)/m_total - delta/m_b
              = (m_b*r_b + delta - delta)/m_b
              = r_b(q)
        """
        rng = np.random.default_rng(seed)
        q = _random_config(mapping.model, rng)
        dq = np.zeros(mapping.model.nv)

        # Compute r_com(q) and r_b(q) from Pinocchio directly
        data = mapping.model.createData()
        pin.forwardKinematics(mapping.model, data, q)
        pin.centerOfMass(mapping.model, data, q)
        r_com_current = data.com[0].copy()
        r_b_current = data.oMi[TORSO_JOINT_IDX].act(
            mapping.model.inertias[TORSO_JOINT_IDX].lever)

        # Apply the mapping with r_com_ref = r_com_current
        r_b_ref, _, _, _ = mapping.compute(
            r_com_current, np.zeros(3), np.zeros(3), q, dq)

        err = float(np.linalg.norm(r_b_ref - r_b_current))
        assert err < 1e-10, \
            f"seed={seed}: self-consistency error = {err:.2e}"
