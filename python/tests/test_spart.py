"""Category 2: Unit Tests for SPART Kinematics & Dynamics.

Tests for URDF parser, forward kinematics, differential kinematics,
velocities, inertia, GIM, CIM, Jacobian, and forward/inverse dynamics.
"""

import numpy as np
import pytest

from python.spart.robot_model import urdf2robot, connectivity_map
from python.spart.kinematics import kinematics
from python.spart.diff_kinematics import diff_kinematics
from python.spart.velocities import velocities
from python.spart.inertia import inertia_projection, mass_composite_body
from python.spart.gim import generalized_inertia_matrix
from python.spart.cim import convective_inertia_matrix
from python.spart.jacobian import geometric_jacobian
from python.spart.dynamics import forward_dynamics, inverse_dynamics
from python.spart.accelerations import accelerations


# ---------------------------------------------------------------------------
# 2.1 URDF Parser
# ---------------------------------------------------------------------------

class TestURDFParser:

    def test_n_links_joints(self, robot):
        assert robot['n_links_joints'] == 12

    def test_n_q(self, robot):
        assert robot['n_q'] == 12

    def test_base_link_mass(self, robot):
        assert robot['base_link']['mass'] > 0

    def test_base_link_inertia_symmetric(self, robot):
        I = robot['base_link']['inertia']
        assert np.allclose(I, I.T)

    def test_all_joints_revolute(self, robot):
        for j in robot['joints']:
            assert j['type'] == 1, f"Joint {j['id']} is not revolute"

    def test_joint_axes_unit(self, robot):
        for j in robot['joints']:
            assert abs(np.linalg.norm(j['axis']) - 1.0) < 1e-10

    def test_connectivity_map_shapes(self, robot):
        n = robot['n_links_joints']
        assert robot['con']['branch'].shape == (n, n)
        assert robot['con']['child'].shape == (n, n)
        assert robot['con']['child_base'].shape == (n,)

    def test_connectivity_binary(self, robot):
        assert set(np.unique(robot['con']['branch'])).issubset({0, 1})
        assert set(np.unique(robot['con']['child'])).issubset({0, 1})
        assert set(np.unique(robot['con']['child_base'])).issubset({0, 1})

    def test_branch_diagonal_one(self, robot):
        """Every link is on its own branch."""
        n = robot['n_links_joints']
        for i in range(n):
            assert robot['con']['branch'][i, i] == 1

    def test_dual_arm_topology(self, robot):
        """Two arms rooted at base: child_base should have 2 entries."""
        assert np.sum(robot['con']['child_base']) == 2

    def test_link_masses_positive(self, robot):
        for link in robot['links']:
            assert link['mass'] > 0

    def test_link_inertia_symmetric(self, robot):
        for link in robot['links']:
            I = link['inertia']
            assert np.allclose(I, I.T), f"Link {link['id']} inertia not symmetric"


# ---------------------------------------------------------------------------
# 2.2 Forward Kinematics
# ---------------------------------------------------------------------------

class TestKinematics:

    def test_zero_config_runs(self, robot):
        R0 = np.eye(3)
        r0 = np.zeros(3)
        qm = np.zeros(robot['n_q'])
        RJ, RL, rJ, rL, e, g = kinematics(R0, r0, qm, robot)
        assert rL.shape == (3, robot['n_links_joints'])
        assert rJ.shape == (3, robot['n_links_joints'])

    def test_g_consistency(self, robot, default_qm):
        """g[:, i] = rL[:, i] - rJ[:, parent_joint_of_link_i]."""
        R0 = np.eye(3)
        r0 = np.zeros(3)
        RJ, RL, rJ, rL, e, g = kinematics(R0, r0, default_qm, robot)

        for i in range(robot['n_links_joints']):
            pj = robot['links'][i]['parent_joint'] - 1
            expected = rL[:, i] - rJ[:, pj]
            assert np.allclose(g[:, i], expected, atol=1e-12), \
                f"g consistency failed for link {i}"

    def test_rotation_matrices_proper(self, robot, default_qm):
        R0 = np.eye(3)
        r0 = np.zeros(3)
        RJ, RL, rJ, rL, e, g = kinematics(R0, r0, default_qm, robot)

        for i in range(robot['n_links_joints']):
            assert np.allclose(RL[i] @ RL[i].T, np.eye(3), atol=1e-10)
            assert abs(np.linalg.det(RL[i]) - 1.0) < 1e-10
            assert np.allclose(RJ[i] @ RJ[i].T, np.eye(3), atol=1e-10)

    def test_base_translation(self, robot):
        """Translating base moves all link positions."""
        R0 = np.eye(3)
        r0_a = np.zeros(3)
        r0_b = np.array([5.0, 3.0, 1.0])
        qm = np.zeros(robot['n_q'])

        _, _, _, rL_a, _, _ = kinematics(R0, r0_a, qm, robot)
        _, _, _, rL_b, _, _ = kinematics(R0, r0_b, qm, robot)

        for i in range(robot['n_links_joints']):
            diff = rL_b[:, i] - rL_a[:, i]
            assert np.allclose(diff, r0_b, atol=1e-10)


# ---------------------------------------------------------------------------
# 2.3 Differential Kinematics
# ---------------------------------------------------------------------------

class TestDiffKinematics:

    def test_Bij_zero_off_branch(self, robot, default_qm):
        R0 = np.eye(3)
        r0 = np.zeros(3)
        _, _, _, rL, e, g = kinematics(R0, r0, default_qm, robot)
        Bij, Bi0, P0, pm = diff_kinematics(R0, r0, rL, e, g, robot)

        branch = robot['con']['branch']
        n = robot['n_links_joints']
        for i in range(n):
            for j in range(n):
                if branch[i, j] == 0:
                    assert np.allclose(Bij[i][j], np.zeros((6, 6))), \
                        f"Bij[{i}][{j}] should be zero (off-branch)"

    def test_Bi0_structure(self, robot, default_qm):
        """Bi0 top-left = I, bottom-left = skew(r0 - rL_i)."""
        R0 = np.eye(3)
        r0 = np.zeros(3)
        _, _, _, rL, e, g = kinematics(R0, r0, default_qm, robot)
        Bij, Bi0, P0, pm = diff_kinematics(R0, r0, rL, e, g, robot)

        from python.utils.spatial import skew_sym
        for i in range(robot['n_links_joints']):
            assert np.allclose(Bi0[i][:3, :3], np.eye(3))
            assert np.allclose(Bi0[i][:3, 3:], np.zeros((3, 3)))
            expected_skew = skew_sym(r0 - rL[:, i])
            assert np.allclose(Bi0[i][3:, :3], expected_skew, atol=1e-12)

    def test_P0_structure(self, robot, default_qm):
        R0 = np.eye(3)
        r0 = np.zeros(3)
        _, _, _, rL, e, g = kinematics(R0, r0, default_qm, robot)
        _, _, P0, _ = diff_kinematics(R0, r0, rL, e, g, robot)

        assert np.allclose(P0[:3, :3], R0)
        assert np.allclose(P0[3:, 3:], np.eye(3))
        assert np.allclose(P0[:3, 3:], np.zeros((3, 3)))
        assert np.allclose(P0[3:, :3], np.zeros((3, 3)))

    def test_pm_revolute_structure(self, robot, default_qm):
        """For revolute joints: pm = [e; e × g]."""
        R0 = np.eye(3)
        r0 = np.zeros(3)
        _, _, _, rL, e, g = kinematics(R0, r0, default_qm, robot)
        _, _, _, pm = diff_kinematics(R0, r0, rL, e, g, robot)

        for i in range(robot['n_links_joints']):
            if robot['joints'][i]['type'] == 1:
                expected = np.concatenate([e[:, i], np.cross(e[:, i], g[:, i])])
                assert np.allclose(pm[:, i], expected, atol=1e-12)


# ---------------------------------------------------------------------------
# 2.4 Velocities
# ---------------------------------------------------------------------------

class TestVelocities:

    def test_zero_velocity(self, robot, default_qm):
        R0 = np.eye(3)
        r0 = np.zeros(3)
        _, _, _, rL, e, g = kinematics(R0, r0, default_qm, robot)
        Bij, Bi0, P0, pm = diff_kinematics(R0, r0, rL, e, g, robot)

        u0 = np.zeros(6)
        um = np.zeros(robot['n_q'])
        t0, tL = velocities(Bij, Bi0, P0, pm, u0, um, robot)

        assert np.allclose(t0, np.zeros(6))
        assert np.allclose(tL, np.zeros((6, robot['n_links_joints'])))

    def test_base_only_motion(self, robot, default_qm):
        R0 = np.eye(3)
        r0 = np.zeros(3)
        _, _, _, rL, e, g = kinematics(R0, r0, default_qm, robot)
        Bij, Bi0, P0, pm = diff_kinematics(R0, r0, rL, e, g, robot)

        u0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # pure x translation
        um = np.zeros(robot['n_q'])
        t0, tL = velocities(Bij, Bi0, P0, pm, u0, um, robot)

        assert np.allclose(t0, P0 @ u0)
        # All links should have the same linear velocity (rigid base motion)
        for i in range(robot['n_links_joints']):
            assert abs(tL[3, i] - t0[3]) < 1e-10, \
                f"Link {i} x-velocity differs from base"


# ---------------------------------------------------------------------------
# 2.5 Inertia Projection
# ---------------------------------------------------------------------------

class TestInertiaProjection:

    def test_symmetry(self, robot, default_qm):
        R0 = np.eye(3)
        RJ, RL, rJ, rL, e, g = kinematics(R0, np.zeros(3), default_qm, robot)
        I0, Im = inertia_projection(R0, RL, robot)

        assert np.allclose(I0, I0.T)
        for i in range(robot['n_links_joints']):
            assert np.allclose(Im[i], Im[i].T)

    def test_positive_definite(self, robot, default_qm):
        R0 = np.eye(3)
        _, RL, _, _, _, _ = kinematics(R0, np.zeros(3), default_qm, robot)
        I0, Im = inertia_projection(R0, RL, robot)

        assert np.all(np.linalg.eigvalsh(I0) > 0)
        for i in range(robot['n_links_joints']):
            eigs = np.linalg.eigvalsh(Im[i])
            assert np.all(eigs >= -1e-12), f"Link {i} inertia not PSD"

    def test_identity_rotation(self, robot):
        """R0=I => I0 == robot base_link inertia (unchanged)."""
        R0 = np.eye(3)
        qm = np.zeros(robot['n_q'])
        _, RL, _, _, _, _ = kinematics(R0, np.zeros(3), qm, robot)
        I0, _ = inertia_projection(R0, RL, robot)
        assert np.allclose(I0, robot['base_link']['inertia'])


# ---------------------------------------------------------------------------
# 2.6 Mass Composite Body
# ---------------------------------------------------------------------------

class TestMCB:

    def test_symmetry(self, robot, default_qm):
        R0 = np.eye(3)
        r0 = np.zeros(3)
        _, RL, _, rL, e, g = kinematics(R0, r0, default_qm, robot)
        Bij, Bi0, _, _ = diff_kinematics(R0, r0, rL, e, g, robot)
        I0, Im = inertia_projection(R0, RL, robot)
        M0_tilde, Mm_tilde = mass_composite_body(I0, Im, Bij, Bi0, robot)

        assert np.allclose(M0_tilde, M0_tilde.T)
        for i in range(robot['n_links_joints']):
            assert np.allclose(Mm_tilde[i], Mm_tilde[i].T, atol=1e-10)

    def test_positive_semidefinite(self, robot, default_qm):
        R0 = np.eye(3)
        r0 = np.zeros(3)
        _, RL, _, rL, e, g = kinematics(R0, r0, default_qm, robot)
        Bij, Bi0, _, _ = diff_kinematics(R0, r0, rL, e, g, robot)
        I0, Im = inertia_projection(R0, RL, robot)
        M0_tilde, Mm_tilde = mass_composite_body(I0, Im, Bij, Bi0, robot)

        eigs = np.linalg.eigvalsh(M0_tilde)
        assert np.all(eigs >= -1e-10), f"M0_tilde not PSD, min eig = {eigs.min()}"

    def test_leaf_link_mcb(self, robot, default_qm):
        """Leaf links (no children): Mm_tilde = [Im, 0; 0, m*I]."""
        R0 = np.eye(3)
        r0 = np.zeros(3)
        _, RL, _, rL, e, g = kinematics(R0, r0, default_qm, robot)
        Bij, Bi0, _, _ = diff_kinematics(R0, r0, rL, e, g, robot)
        I0, Im = inertia_projection(R0, RL, robot)
        _, Mm_tilde = mass_composite_body(I0, Im, Bij, Bi0, robot)

        child = robot['con']['child']
        for i in range(robot['n_links_joints']):
            if np.sum(child[:, i]) == 0:  # leaf
                expected = np.block([
                    [Im[i], np.zeros((3, 3))],
                    [np.zeros((3, 3)), robot['links'][i]['mass'] * np.eye(3)],
                ])
                assert np.allclose(Mm_tilde[i], expected, atol=1e-10), \
                    f"Leaf link {i} MCB mismatch"


# ---------------------------------------------------------------------------
# 2.7 Generalized Inertia Matrix
# ---------------------------------------------------------------------------

class TestGIM:

    def _compute_gim(self, robot, qm):
        R0 = np.eye(3)
        r0 = np.zeros(3)
        _, RL, _, rL, e, g = kinematics(R0, r0, qm, robot)
        Bij, Bi0, P0, pm = diff_kinematics(R0, r0, rL, e, g, robot)
        I0, Im = inertia_projection(R0, RL, robot)
        M0_tilde, Mm_tilde = mass_composite_body(I0, Im, Bij, Bi0, robot)
        return generalized_inertia_matrix(M0_tilde, Mm_tilde, Bij, Bi0, P0, pm, robot)

    def test_dimensions(self, robot, default_qm):
        H0, H0m, Hm = self._compute_gim(robot, default_qm)
        assert H0.shape == (6, 6)
        assert H0m.shape == (6, robot['n_q'])
        assert Hm.shape == (robot['n_q'], robot['n_q'])

    def test_full_gim_symmetric(self, robot, default_qm):
        H0, H0m, Hm = self._compute_gim(robot, default_qm)
        H = np.block([[H0, H0m], [H0m.T, Hm]])
        assert np.allclose(H, H.T, atol=1e-10)

    def test_full_gim_positive_definite(self, robot, default_qm):
        H0, H0m, Hm = self._compute_gim(robot, default_qm)
        H = np.block([[H0, H0m], [H0m.T, Hm]])
        eigs = np.linalg.eigvalsh(H)
        assert np.all(eigs > 0), f"H not PD, min eig = {eigs.min()}"

    def test_varies_with_config(self, robot):
        """GIM should change when joint configuration changes."""
        H0a, _, Hma = self._compute_gim(robot, np.zeros(robot['n_q']))
        H0b, _, Hmb = self._compute_gim(robot, np.ones(robot['n_q']) * 0.5)
        assert not np.allclose(H0a, H0b), "H0 should vary with config"
        assert not np.allclose(Hma, Hmb), "Hm should vary with config"


# ---------------------------------------------------------------------------
# 2.8 Convective Inertia Matrix
# ---------------------------------------------------------------------------

class TestCIM:

    def test_zero_at_zero_velocity(self, robot, default_qm):
        R0 = np.eye(3)
        r0 = np.zeros(3)
        _, RL, _, rL, e, g = kinematics(R0, r0, default_qm, robot)
        Bij, Bi0, P0, pm = diff_kinematics(R0, r0, rL, e, g, robot)

        u0 = np.zeros(6)
        um = np.zeros(robot['n_q'])
        t0, tL = velocities(Bij, Bi0, P0, pm, u0, um, robot)

        I0, Im = inertia_projection(R0, RL, robot)
        M0_tilde, Mm_tilde = mass_composite_body(I0, Im, Bij, Bi0, robot)
        C0, C0m, Cm0, Cm = convective_inertia_matrix(
            t0, tL, I0, Im, M0_tilde, Mm_tilde, Bij, Bi0, P0, pm, robot
        )

        C = np.block([[C0, C0m], [Cm0, Cm]])
        assert np.allclose(C, 0, atol=1e-14)

    def test_dimensions(self, robot, default_qm):
        R0 = np.eye(3)
        r0 = np.zeros(3)
        _, RL, _, rL, e, g = kinematics(R0, r0, default_qm, robot)
        Bij, Bi0, P0, pm = diff_kinematics(R0, r0, rL, e, g, robot)

        u0 = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        um = np.ones(robot['n_q']) * 0.1
        t0, tL = velocities(Bij, Bi0, P0, pm, u0, um, robot)

        I0, Im = inertia_projection(R0, RL, robot)
        M0_tilde, Mm_tilde = mass_composite_body(I0, Im, Bij, Bi0, robot)
        C0, C0m, Cm0, Cm = convective_inertia_matrix(
            t0, tL, I0, Im, M0_tilde, Mm_tilde, Bij, Bi0, P0, pm, robot
        )

        assert C0.shape == (6, 6)
        assert C0m.shape == (6, robot['n_q'])
        assert Cm0.shape == (robot['n_q'], 6)
        assert Cm.shape == (robot['n_q'], robot['n_q'])


# ---------------------------------------------------------------------------
# 2.9 Geometric Jacobian
# ---------------------------------------------------------------------------

class TestJacobian:

    def test_velocity_consistency(self, robot, default_qm):
        """J0 @ u0 + Jm @ um should match twist from Velocities."""
        R0 = np.eye(3)
        r0 = np.zeros(3)
        _, RL, _, rL, e, g = kinematics(R0, r0, default_qm, robot)
        Bij, Bi0, P0, pm = diff_kinematics(R0, r0, rL, e, g, robot)

        u0 = np.array([0.1, 0.05, -0.02, 0.3, -0.1, 0.05])
        um = np.random.default_rng(42).standard_normal(robot['n_q']) * 0.1

        t0, tL = velocities(Bij, Bi0, P0, pm, u0, um, robot)

        # Check Jacobian at end-effector (last link)
        n = robot['n_links_joints']
        rp = rL[:, -1]
        J0, Jm = geometric_jacobian(rp, r0, rL, P0, pm, n, robot)
        tp_jac = J0 @ u0 + Jm @ um
        tp_vel = tL[:, -1]  # Twist at last link CoM should approximate

        # They won't be exactly equal because Jacobian is at rp = rL[:,-1]
        # and Velocities computes twist at link CoM which IS rL[:,-1]
        # so they should match for the last link
        assert np.allclose(tp_jac, tp_vel, atol=1e-8), \
            f"Jacobian twist differs from Velocities twist:\n{tp_jac}\nvs\n{tp_vel}"

    def test_zero_columns_off_chain(self, robot, default_qm):
        """Joints not on kinematic chain to link i produce zero Jm columns."""
        R0 = np.eye(3)
        r0 = np.zeros(3)
        _, _, _, rL, e, g = kinematics(R0, r0, default_qm, robot)
        _, _, P0, pm = diff_kinematics(R0, r0, rL, e, g, robot)

        branch = robot['con']['branch']
        # Check Jacobian for link 5 (end of arm 1, 0-indexed)
        i = 5  # 0-indexed
        rp = rL[:, i]
        J0, Jm = geometric_jacobian(rp, r0, rL, P0, pm, i + 1, robot)

        # Joints of arm 2 (indices 6-11) should not affect arm 1 tip
        for j in range(6, 12):
            q_id = robot['joints'][j]['q_id'] - 1
            assert np.allclose(Jm[:, q_id], 0, atol=1e-14), \
                f"Jm column {q_id} (joint {j}) should be zero for link {i}"


# ---------------------------------------------------------------------------
# 2.10 Forward / Inverse Dynamics
# ---------------------------------------------------------------------------

class TestDynamics:

    def _setup_dynamics(self, robot, qm, u0, um):
        R0 = np.eye(3)
        r0 = np.zeros(3)
        _, RL, _, rL, e, g = kinematics(R0, r0, qm, robot)
        Bij, Bi0, P0, pm = diff_kinematics(R0, r0, rL, e, g, robot)
        t0, tL = velocities(Bij, Bi0, P0, pm, u0, um, robot)
        I0, Im = inertia_projection(R0, RL, robot)
        return Bij, Bi0, P0, pm, t0, tL, I0, Im

    def test_fd_id_roundtrip(self, robot, default_qm):
        """Apply forces -> FD -> get accel -> ID -> recover forces."""
        n = robot['n_links_joints']
        n_q = robot['n_q']

        u0 = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        um = np.ones(n_q) * 0.05

        Bij, Bi0, P0, pm, t0, tL, I0, Im = self._setup_dynamics(
            robot, default_qm, u0, um
        )

        # Apply known forces
        tau0 = np.array([1.0, 0.5, -0.3, 2.0, 1.0, 0.5])
        taum = np.random.default_rng(42).standard_normal(n_q)
        wF0 = np.zeros(6)
        wFm = np.zeros((6, n))

        # Forward dynamics
        u0dot, umdot = forward_dynamics(
            tau0, taum, wF0, wFm, t0, tL, P0, pm, I0, Im,
            Bij, Bi0, u0, um, robot
        )

        # Compute accelerations (twist rates) from u0dot, umdot
        t0dot, tLdot = accelerations(t0, tL, P0, pm, Bi0, Bij,
                                      u0, um, u0dot, umdot, robot)

        # Inverse dynamics to recover forces
        tau0_rec, taum_rec = inverse_dynamics(
            wF0, wFm, t0, tL, t0dot, tLdot, P0, pm, I0, Im,
            Bij, Bi0, robot
        )

        assert np.allclose(tau0_rec, tau0, atol=1e-6), \
            f"tau0 roundtrip failed:\n{tau0_rec}\nvs\n{tau0}"
        assert np.allclose(taum_rec, taum, atol=1e-6), \
            f"taum roundtrip failed:\n{taum_rec}\nvs\n{taum}"

    def test_zero_force_zero_accel_at_rest(self, robot, default_qm):
        """Zero forces at rest => zero accelerations."""
        n = robot['n_links_joints']
        u0 = np.zeros(6)
        um = np.zeros(robot['n_q'])

        Bij, Bi0, P0, pm, t0, tL, I0, Im = self._setup_dynamics(
            robot, default_qm, u0, um
        )

        tau0 = np.zeros(6)
        taum = np.zeros(robot['n_q'])
        wF0 = np.zeros(6)
        wFm = np.zeros((6, n))

        u0dot, umdot = forward_dynamics(
            tau0, taum, wF0, wFm, t0, tL, P0, pm, I0, Im,
            Bij, Bi0, u0, um, robot
        )

        assert np.allclose(u0dot, 0, atol=1e-10)
        assert np.allclose(umdot, 0, atol=1e-10)
