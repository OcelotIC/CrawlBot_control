"""Tests for the AOCS law kept by R1: legacy_pid_numerical.

The estimator (MomentumDisturbanceEstimator) and the other laws
(compute_aocs_command / H_est, legacy_corrected, legacy_pd_*, legacy_pid_model)
were retired with their tests; the orbital feedforward is covered by
test_aocs_orbital.py on the kept law.
"""

import numpy as np


# ---------------------------------------------------------------------------
# legacy_pid_* attitude-tracking sign convention
# ---------------------------------------------------------------------------

class TestAOCSCommandLegacyPID:
    """Sign convention for the new attitude-tracking modes.

    Same Newton-Euler derivation as the K_omega term: for θ_s > 0 to
    decrease, need negative angular acceleration ⇒ τ_w > -Ḣ_s ⇒ K_θ
    contribution must add POSITIVE.
    """

    def _make_inputs(self, theta_z=0.0):
        """Minimal AOCS inputs: at-rest robot, single nonzero θ_s_z."""
        z = np.zeros(3)
        return dict(
            L_com=z.copy(), L_com_prev=z.copy(),
            r_com=z.copy(), v_com=z.copy(), v_com_prev=z.copy(),
            omega_s=z.copy(),
            theta_s=np.array([0.0, 0.0, theta_z]),
            hw_current=z.copy(),
            dt=0.01, robot_mass=71.0,
            K_hw=0.0, K_omega=0.0, K_d=0.0, K_theta=10.0,
            tau_w_max=100.0,
        )

    def test_pid_numerical_theta_sign(self):
        """θ_s > 0 ⇒ τ_w > 0 (positive damping). Numerical mode."""
        from crawlbot.aocs.force_estimator import (
            compute_aocs_command_legacy_pid_numerical)
        inputs = self._make_inputs(theta_z=+0.05)
        inputs['omega_s_prev'] = inputs['omega_s'].copy()
        tau_w = compute_aocs_command_legacy_pid_numerical(**inputs)
        assert tau_w[2] > 0, f"τ_w_z={tau_w[2]} should be POSITIVE for θ_z>0"

        inputs = self._make_inputs(theta_z=-0.05)
        inputs['omega_s_prev'] = inputs['omega_s'].copy()
        tau_w = compute_aocs_command_legacy_pid_numerical(**inputs)
        assert tau_w[2] < 0, f"τ_w_z={tau_w[2]} should be NEGATIVE for θ_z<0"

    def test_pid_matches_explicit_formula_when_K_theta_zero(self):
        """K_theta=0: τ_w = −L̇ − r×m·v̇ + K_hw·hw_err + K_ω·ω + K_d·ω̇,
        clipped — the formula written out (replaces the comparison against
        the retired legacy_pd_numerical, R1)."""
        from crawlbot.aocs.force_estimator import (
            compute_aocs_command_legacy_pid_numerical)
        z = np.zeros(3)
        rng = np.random.default_rng(0)
        c = dict(
            L_com=rng.standard_normal(3), L_com_prev=rng.standard_normal(3),
            r_com=rng.standard_normal(3),
            v_com=rng.standard_normal(3), v_com_prev=rng.standard_normal(3),
            omega_s=rng.standard_normal(3) * 0.01,
            omega_s_prev=rng.standard_normal(3) * 0.01,
            hw_current=rng.standard_normal(3) * 0.5,
            dt=0.01, robot_mass=71.0,
            K_hw=2.0, K_omega=50.0, K_d=25.0,
            tau_w_max=100.0,
        )
        pid = compute_aocs_command_legacy_pid_numerical(
            theta_s=z.copy(), K_theta=0.0, **c)
        L_dot = (c['L_com'] - c['L_com_prev']) / c['dt']
        v_dot = (c['v_com'] - c['v_com_prev']) / c['dt']
        w_dot = (c['omega_s'] - c['omega_s_prev']) / c['dt']
        expected = (-L_dot - np.cross(c['r_com'], c['robot_mass'] * v_dot)
                    + c['K_hw'] * (0.0 - 0.0)   # hw inside ±inf box
                    + c['K_omega'] * c['omega_s'] + c['K_d'] * w_dot)
        np.testing.assert_allclose(pid, np.clip(expected, -100.0, 100.0),
                                   rtol=1e-12, atol=1e-12)
