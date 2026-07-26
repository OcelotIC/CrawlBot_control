"""M4 — Corrected AOCS feedforward tests.

The legacy AOCS formula in sim_loop.py was:

    τ_w = -L̇_com_est - K_hw · (h_w - clip(h_w, bounds))

It tracked the spin (centroidal) rate but missed the orbital rate:

    τ_w = -L̇_com_est - r_com × m · v̇_com_est - K_hw · (...)
                        ^^^^^^^^^^^^^^^^^^^^^
                        NEW — M4 correction

This file covers the algebraic properties of the corrected helper:
  1. With zero acceleration, the legacy and corrected formulae agree.
  2. With non-zero linear acceleration AND non-zero r_com, the orbital
     term is non-trivial and points in the expected direction.
  3. The clip (desaturation) term works as before.
  4. Output is clipped to [-tau_w_max, tau_w_max].

The closed-loop sim effect (platform rotation drop) is verified by the
M4 baseline runs in Misc/runs/M4_baseline_1pct/ and M4_baseline_14pct/.
"""

import numpy as np
import pytest

from crawlbot.aocs.force_estimator import (
    compute_aocs_command_legacy_corrected,
)


ROBOT_MASS = 71.0


def _legacy_uncorrected(L_com, L_com_prev, hw_current,
                        dt, K_hw, hw_min, hw_max, tau_w_max):
    """Reproduce the sim_loop legacy (no-orbital) formula for comparison.

    Formula after the 04/2026 desaturation sign fix:
        τ_w = -L̇_com_est + K_hw · hw_error,  hw_error := clip - current

    Note: the previous version of this helper had "-K_hw·hw_error" to
    match the pre-fix sim_loop inline legacy. Both the helper and the
    sim_loop formula were updated together so this test continues to
    verify the sim_loop↔force_estimator equivalence.
    """
    L_dot_est = (L_com - L_com_prev) / dt
    hw_error = np.clip(hw_current, hw_min, hw_max) - hw_current
    tau_w = -L_dot_est + K_hw * hw_error
    return np.clip(tau_w, -tau_w_max, tau_w_max)


class TestOrbitalTerm:
    def test_zero_accel_matches_legacy(self):
        """If v_com is constant across the step (dv_com=0), then the
        orbital correction is zero and the corrected formula must equal
        the legacy formula exactly (within clipping)."""
        v = np.array([0.1, 0.2, -0.05])
        L = np.array([0.5, 0.0, -0.3])
        r = np.array([0.8, -0.2, -0.5])
        hw = np.array([1.0, 0.0, -0.5])
        L_prev = L - np.array([0.1, 0.05, -0.02])  # non-trivial L_dot
        v_prev = v.copy()                           # zero accel

        kw = dict(dt=0.01, K_hw=2.0,
                  hw_min=np.full(3, -5.0), hw_max=np.full(3, 5.0),
                  tau_w_max=5.0)

        tw_corrected = compute_aocs_command_legacy_corrected(
            L_com=L, L_com_prev=L_prev, r_com=r, v_com=v, v_com_prev=v_prev,
            hw_current=hw, robot_mass=ROBOT_MASS, **kw)
        tw_legacy = _legacy_uncorrected(
            L_com=L, L_com_prev=L_prev, hw_current=hw, **kw)

        np.testing.assert_allclose(tw_corrected, tw_legacy, atol=1e-12)

    def test_nonzero_accel_adds_orbital(self):
        """Pure linear acceleration in +x with r_com in +x and -z
        should produce a y-axis orbital term (right-hand rule).

        Use a small accel magnitude so the result stays below the clip.
        """
        r = np.array([1.0, 0.0, -0.5])
        v_prev = np.zeros(3)
        v = np.array([0.002, 0.0, 0.0])  # small accel: dv/dt ≈ 0.2 m/s²
        L = np.zeros(3)
        L_prev = np.zeros(3)  # no centroidal rate
        hw = np.zeros(3)      # no desaturation error

        dt = 0.01
        kw = dict(dt=dt, K_hw=2.0,
                  hw_min=np.full(3, -5.0), hw_max=np.full(3, 5.0),
                  tau_w_max=1e6)  # effectively unclipped

        tw = compute_aocs_command_legacy_corrected(
            L_com=L, L_com_prev=L_prev, r_com=r, v_com=v, v_com_prev=v_prev,
            hw_current=hw, robot_mass=ROBOT_MASS, **kw)

        # Expected:
        #   dv_com = v - v_prev = (0.2, 0, 0) / dt = (20, 0, 0) m/s²
        #   orbital = r × (m · dv_com) = (1, 0, -0.5) × (71·20, 0, 0)
        #            = (1, 0, -0.5) × (1420, 0, 0)
        #            = (0·0 - (-0.5)·0, (-0.5)·1420 - 1·0, 1·0 - 0·1420)
        #            = (0, -710, 0)
        #   τ_w = -L_dot - orbital = -0 - (0, -710, 0) = (0, 710, 0)
        dv = (v - v_prev) / dt
        expected_orbital = np.cross(r, ROBOT_MASS * dv)
        expected_tw = -expected_orbital  # L_dot and hw_error are zero
        np.testing.assert_allclose(tw, expected_tw, atol=1e-10)

    def test_desaturation_term_alone(self):
        """With zero accelerations and zero centroidal rate but hw out
        of bounds, the output should be the desaturation feedback.

        After the 04/2026 sign fix:
            τ_w = -L_dot - orbital + K_hw·hw_error
        With h_w=+7 above the +5 box, hw_error = clip - current = -2,
        so the command is -4 Nm which (in MuJoCo's ctrl sign) applies
        +4 Nm of torque on the wheel...wait — ctrl[rw] = -4 applies
        -4 Nm on the wheel (decelerating it → h_w decreases → toward
        the box). Verified empirically: h_w = +7 → +4.97 after 1 s.
        """
        hw = np.array([7.0, 0.0, 0.0])  # above +5 bound on x
        hw_min = np.full(3, -5.0)
        hw_max = np.full(3, 5.0)
        K_hw = 2.0

        tw = compute_aocs_command_legacy_corrected(
            L_com=np.zeros(3), L_com_prev=np.zeros(3),
            r_com=np.zeros(3),  # no orbital term
            v_com=np.zeros(3), v_com_prev=np.zeros(3),
            hw_current=hw, dt=0.01, robot_mass=ROBOT_MASS,
            K_hw=K_hw, hw_min=hw_min, hw_max=hw_max, tau_w_max=100.0)
        # hw_error = clip(7, -5, 5) - 7 = -2
        # τ_w = -L_dot - orbital + K_hw * hw_error = 0 - 0 + 2*(-2) = -4
        np.testing.assert_allclose(tw, np.array([-4.0, 0.0, 0.0]), atol=1e-12)

    def test_clipping_to_tau_w_max(self):
        """Large orbital + centroidal rates should be clipped."""
        L = np.array([100.0, 0.0, 0.0])
        L_prev = np.zeros(3)  # huge L_dot
        tw = compute_aocs_command_legacy_corrected(
            L_com=L, L_com_prev=L_prev,
            r_com=np.zeros(3),
            v_com=np.zeros(3), v_com_prev=np.zeros(3),
            hw_current=np.zeros(3), dt=0.01, robot_mass=ROBOT_MASS,
            K_hw=2.0, hw_min=np.full(3, -5.0), hw_max=np.full(3, 5.0),
            tau_w_max=5.0)
        # Raw τ_w ~ -(100/0.01)·x = -10000·x, clipped to [-5, 5]
        np.testing.assert_allclose(tw, np.array([-5.0, 0.0, 0.0]))

    def test_orbital_product_rule_identity(self):
        """Verify d/dt(r × m·v) = r × m·dv̇ numerically (not v × m·v).

        This confirms the corrected formula matches the product-rule
        expansion of the orbital momentum derivative: the dr/dt × v
        term vanishes because dr/dt = v is parallel to v.
        """
        rng = np.random.default_rng(42)
        dt = 1e-6  # small dt → second-order error O(dt) → 0
        for _ in range(5):
            r0 = rng.uniform(-1, 1, 3)
            v0 = rng.uniform(-0.3, 0.3, 3)
            a = rng.uniform(-0.5, 0.5, 3)
            r1 = r0 + v0 * dt
            v1 = v0 + a * dt

            # Finite-difference of the full orbital momentum
            H_orb_0 = np.cross(r0, ROBOT_MASS * v0)
            H_orb_1 = np.cross(r1, ROBOT_MASS * v1)
            dH_fd = (H_orb_1 - H_orb_0) / dt

            # Product-rule approximation at t=0:
            # d/dt (r × m·v) = v × m·v + r × m·a = 0 + r × m·a
            dH_pr = np.cross(r0, ROBOT_MASS * a)

            # They must agree since v × m·v ≡ 0 exactly (dt² terms → 0).
            np.testing.assert_allclose(dH_fd, dH_pr, rtol=1e-5, atol=1e-5)
