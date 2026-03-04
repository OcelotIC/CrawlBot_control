"""Category 3: Integration Tests for Controller & Simulation.

Tests for feedforward wrench, QP controller, dynamics integration,
and full simulation loop.
"""

import warnings
import numpy as np
import pytest

from python.config.trajectories import setup_trajectories
from python.control.feedforward import compute_feedforward_wrench
from python.control.controller import LutzeQPController
from python.dynamics.system_state import compute_system_state
from python.simulation.integrator import integrate_dynamics
from python.spart.attitude import quat_angles321


# ---------------------------------------------------------------------------
# 3.1 Feedforward Wrench
# ---------------------------------------------------------------------------

class TestFeedforwardWrench:

    def _make_sys_at_target(self, robot, env, traj, t_idx):
        """Build a sys dict where robot is at the desired position."""
        state = {
            'q_base': traj['pos'][:, t_idx].copy(),
            'quat_base': np.array([0.0, 0.0, 0.0, 1.0]),
            'qm': np.deg2rad([15, -30, 60, 0, -30, 0, -15, -30, 60, 0, -30, 0]),
            'omega_base': np.zeros(3),
            'u_base': traj['vel'][:, t_idx].copy(),
            'um': np.zeros(robot['n_q']),
            'omega_satellite': np.zeros(3),
            'quat_satellite': np.array([0.0, 0.0, 0.0, 1.0]),
            'h_RW_stored': np.zeros(3),
        }
        return compute_system_state(state, robot, env)

    def test_zero_error_gives_small_wrench(self, robot, env):
        """When robot is at desired pos/vel, F_d_r should be near zero."""
        traj, _ = setup_trajectories(2)
        sys = self._make_sys_at_target(robot, env, traj, 0)
        gains = {
            'Kr': env['control']['Kr'],
            'Dr': env['control']['Dr'],
            'Kb': env['control']['Kb'],
            'Db': env['control']['Db'],
        }
        F_d_r, F_d_b = compute_feedforward_wrench(sys, traj, 0, gains)
        # At t=0: v_des = 0, pos = start => error ~ 0
        assert np.linalg.norm(F_d_r) < 1.0, f"F_d_r = {F_d_r}, should be ~0"

    def test_position_error_saturation(self, robot, env):
        """Large position errors should be clamped."""
        traj, _ = setup_trajectories(2)
        state = {
            'q_base': traj['pos'][:, 0] + np.array([100.0, 0.0, 0.0]),
            'quat_base': np.array([0.0, 0.0, 0.0, 1.0]),
            'qm': np.deg2rad([15, -30, 60, 0, -30, 0, -15, -30, 60, 0, -30, 0]),
            'omega_base': np.zeros(3),
            'u_base': np.zeros(3),
            'um': np.zeros(robot['n_q']),
            'omega_satellite': np.zeros(3),
            'quat_satellite': np.array([0.0, 0.0, 0.0, 1.0]),
            'h_RW_stored': np.zeros(3),
        }
        sys = compute_system_state(state, robot, env)
        gains = {
            'Kr': env['control']['Kr'],
            'Dr': env['control']['Dr'],
            'Kb': env['control']['Kb'],
            'Db': env['control']['Db'],
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            F_d_r, _ = compute_feedforward_wrench(sys, traj, 0, gains)
        # Wrench should be bounded (saturated)
        assert np.linalg.norm(F_d_r) <= 1001, \
            f"F_d_r norm = {np.linalg.norm(F_d_r)}, should be <= 1000"

    def test_base_wrench_identity_quaternion(self, robot, env):
        """Identity quaternion (no rotation) => F_d_b rotation part ~ 0."""
        traj, _ = setup_trajectories(2)
        state = {
            'q_base': traj['pos'][:, 0].copy(),
            'quat_base': np.array([0.0, 0.0, 0.0, 1.0]),
            'qm': np.deg2rad([15, -30, 60, 0, -30, 0, -15, -30, 60, 0, -30, 0]),
            'omega_base': np.zeros(3),
            'u_base': np.zeros(3),
            'um': np.zeros(robot['n_q']),
            'omega_satellite': np.zeros(3),
            'quat_satellite': np.array([0.0, 0.0, 0.0, 1.0]),
            'h_RW_stored': np.zeros(3),
        }
        sys = compute_system_state(state, robot, env)
        gains = {
            'Kr': env['control']['Kr'],
            'Dr': env['control']['Dr'],
            'Kb': env['control']['Kb'],
            'Db': env['control']['Db'],
        }
        _, F_d_b = compute_feedforward_wrench(sys, traj, 0, gains)
        # Identity quat => e_quat = 2*[0,0,0] = 0, omega = 0 => F_d_b = 0
        assert np.linalg.norm(F_d_b) < 1e-6, \
            f"F_d_b = {F_d_b}, should be ~0 for identity quat"

    def test_invalid_input_returns_zero(self, robot, env):
        """NaN in r0 should return zero wrenches."""
        traj, _ = setup_trajectories(2)
        sys = {
            'r0': np.array([np.nan, 0.0, 0.0]),
            't0': np.zeros(6),
            'quat_base': np.array([0.0, 0.0, 0.0, 1.0]),
        }
        gains = {
            'Kr': env['control']['Kr'],
            'Dr': env['control']['Dr'],
            'Kb': env['control']['Kb'],
            'Db': env['control']['Db'],
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            F_d_r, F_d_b = compute_feedforward_wrench(sys, traj, 0, gains)
        assert np.allclose(F_d_r, 0)
        assert np.allclose(F_d_b, 0)


# ---------------------------------------------------------------------------
# 3.2 QP Controller
# ---------------------------------------------------------------------------

class TestQPController:

    def _make_state_and_sys(self, robot, env, traj, t_idx):
        state = {
            'q_base': traj['pos'][:, t_idx].copy(),
            'quat_base': np.array([0.0, 0.0, 0.0, 1.0]),
            'qm': np.deg2rad([15, -30, 60, 0, -30, 0, -15, -30, 60, 0, -30, 0]),
            'omega_base': np.zeros(3),
            'u_base': np.zeros(3),
            'um': np.zeros(robot['n_q']),
            'omega_satellite': np.zeros(3),
            'quat_satellite': np.array([0.0, 0.0, 0.0, 1.0]),
            'h_RW_stored': np.zeros(3),
        }
        sys = compute_system_state(state, robot, env)
        return state, sys

    def test_non_optimized_returns_valid(self, robot, env, controller):
        traj, _ = setup_trajectories(2)
        _, sys = self._make_state_and_sys(robot, env, traj, 100)
        Fc, tau = controller.solve(sys, traj, 100, use_optimization=False)
        assert Fc.shape == (6,)
        assert tau.shape == (robot['n_q'],)
        assert not np.any(np.isnan(Fc))
        assert not np.any(np.isnan(tau))

    def test_optimized_returns_valid(self, robot, env, controller):
        traj, _ = setup_trajectories(2)
        _, sys = self._make_state_and_sys(robot, env, traj, 100)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Fc, tau = controller.solve(sys, traj, 100, use_optimization=True)
        assert Fc.shape == (6,)
        assert not np.any(np.isnan(Fc))

    def test_wrench_bounds(self, robot, env, controller):
        """Optimized wrench must respect SI limits."""
        traj, _ = setup_trajectories(2)
        _, sys = self._make_state_and_sys(robot, env, traj, 200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Fc, _ = controller.solve(sys, traj, 200, use_optimization=True)
        assert np.all(Fc[:3] >= -env['SI']['F_max'] - 1e-6)
        assert np.all(Fc[:3] <= env['SI']['F_max'] + 1e-6)
        assert np.all(Fc[3:] >= -env['SI']['tau_max'] - 1e-6)
        assert np.all(Fc[3:] <= env['SI']['tau_max'] + 1e-6)


# ---------------------------------------------------------------------------
# 3.3 Dynamics Integration
# ---------------------------------------------------------------------------

class TestDynamicsIntegration:

    def _single_step(self, robot, env, traj, controller, use_opt=True):
        state = {
            'q_base': traj['pos'][:, 0].copy(),
            'quat_base': np.array([0.0, 0.0, 0.0, 1.0]),
            'qm': np.deg2rad([15, -30, 60, 0, -30, 0, -15, -30, 60, 0, -30, 0]),
            'omega_base': np.zeros(3),
            'u_base': np.zeros(3),
            'um': np.zeros(robot['n_q']),
            'omega_satellite': np.zeros(3),
            'quat_satellite': np.array([0.0, 0.0, 0.0, 1.0]),
            'h_RW_stored': np.zeros(3),
        }
        sys = compute_system_state(state, robot, env)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Fc, tau = controller.solve(sys, traj, 0, use_opt)
        state_next = integrate_dynamics(state, tau, Fc, sys, robot, 0.01, env)
        return state, state_next, Fc

    def test_quaternion_normalization(self, robot, env, controller):
        traj, _ = setup_trajectories(2)
        _, state_next, _ = self._single_step(robot, env, traj, controller)
        assert abs(np.linalg.norm(state_next['quat_base']) - 1.0) < 1e-12
        assert abs(np.linalg.norm(state_next['quat_satellite']) - 1.0) < 1e-12

    def test_no_nan_after_step(self, robot, env, controller):
        traj, _ = setup_trajectories(2)
        _, state_next, _ = self._single_step(robot, env, traj, controller)
        for key, val in state_next.items():
            if isinstance(val, np.ndarray):
                assert not np.any(np.isnan(val)), f"NaN in state_next['{key}']"
                assert not np.any(np.isinf(val)), f"Inf in state_next['{key}']"

    def test_zero_force_minimal_change(self, robot, env):
        """Zero controller output => state changes minimally."""
        traj, _ = setup_trajectories(2)
        state = {
            'q_base': traj['pos'][:, 0].copy(),
            'quat_base': np.array([0.0, 0.0, 0.0, 1.0]),
            'qm': np.deg2rad([15, -30, 60, 0, -30, 0, -15, -30, 60, 0, -30, 0]),
            'omega_base': np.zeros(3),
            'u_base': np.zeros(3),
            'um': np.zeros(robot['n_q']),
            'omega_satellite': np.zeros(3),
            'quat_satellite': np.array([0.0, 0.0, 0.0, 1.0]),
            'h_RW_stored': np.zeros(3),
        }
        sys = compute_system_state(state, robot, env)
        Fc = np.zeros(6)
        tau = np.zeros(robot['n_q'])
        state_next = integrate_dynamics(state, tau, Fc, sys, robot, 0.01, env)

        # Position shouldn't change much (no velocity, no force)
        assert np.allclose(state_next['q_base'], state['q_base'], atol=1e-4)
        assert np.allclose(state_next['qm'], state['qm'], atol=1e-4)


# ---------------------------------------------------------------------------
# 3.4 Full Simulation Loop
# ---------------------------------------------------------------------------

class TestSimulationLoop:

    def test_short_run_no_crash(self, robot, env, controller):
        """10 timesteps for each experiment without crash."""
        for exp_id in [1, 2, 3]:
            traj, sim_params = setup_trajectories(exp_id)
            state = {
                'q_base': traj['pos'][:, 0].copy(),
                'quat_base': np.array([0.0, 0.0, 0.0, 1.0]),
                'qm': np.deg2rad([15, -30, 60, 0, -30, 0,
                                   -15, -30, 60, 0, -30, 0]),
                'omega_base': np.zeros(3),
                'u_base': np.zeros(3),
                'um': np.zeros(robot['n_q']),
                'omega_satellite': np.zeros(3),
                'quat_satellite': np.array([0.0, 0.0, 0.0, 1.0]),
                'h_RW_stored': np.zeros(3),
            }

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for k in range(10):
                    sys = compute_system_state(state, robot, env)
                    Fc, tau = controller.solve(sys, traj, k, True)
                    state = integrate_dynamics(state, tau, Fc, sys, robot,
                                               sim_params['dt'], env)

            # Verify no NaN
            for key, val in state.items():
                if isinstance(val, np.ndarray):
                    assert not np.any(np.isnan(val)), \
                        f"NaN in state['{key}'] for exp #{exp_id}"

    def test_reproducibility(self, robot, env, controller):
        """Two identical runs should produce identical results."""
        traj, sim_params = setup_trajectories(2)

        def run_n_steps(n):
            state = {
                'q_base': traj['pos'][:, 0].copy(),
                'quat_base': np.array([0.0, 0.0, 0.0, 1.0]),
                'qm': np.deg2rad([15, -30, 60, 0, -30, 0,
                                   -15, -30, 60, 0, -30, 0]),
                'omega_base': np.zeros(3),
                'u_base': np.zeros(3),
                'um': np.zeros(robot['n_q']),
                'omega_satellite': np.zeros(3),
                'quat_satellite': np.array([0.0, 0.0, 0.0, 1.0]),
                'h_RW_stored': np.zeros(3),
            }
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for k in range(n):
                    sys = compute_system_state(state, robot, env)
                    Fc, tau = controller.solve(sys, traj, k, True)
                    state = integrate_dynamics(state, tau, Fc, sys, robot,
                                               sim_params['dt'], env)
            return state

        s1 = run_n_steps(20)
        s2 = run_n_steps(20)

        for key in s1:
            if isinstance(s1[key], np.ndarray):
                assert np.allclose(s1[key], s2[key], atol=1e-14), \
                    f"state['{key}'] not reproducible"
