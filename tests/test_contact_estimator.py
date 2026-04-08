"""
Test suite for the Generalized Momentum Observer (GMO) and ContactStateMachine.

Tests:
    T1 — Observer convergence: known weld constraint, r -> J_c^T f_ext
    T2 — Zero-force baseline: no contact, ||r|| ~ 0
    T3 — State machine transitions: synthetic inputs
    T4 — Debounce: transient spike does not trigger CONFIRMED
    T5 — Reset: after reset, r returns to zero

Usage:
    PYTHONPATH=. MUJOCO_GL=disabled pytest tests/test_contact_estimator.py -v
"""
import os
import sys
import numpy as np
import pytest

os.environ.setdefault('MUJOCO_GL', 'disabled')

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)

from crawlbot.estimation.contact_estimator import (
    GeneralizedMomentumObserver, ContactStateMachine,
    ContactObserverConfig, ContactState)


# ═══════════════════════════════════════════════════════════════════════════════
# T1 — Observer convergence with known external force
# ═══════════════════════════════════════════════════════════════════════════════

class TestObserverConvergence:
    """Verify that the GMO residual converges to a known tau_ext."""

    def test_converges_to_constant_external_force(self):
        """With constant M, C=0, tau=0, and p_dot = tau_ext, r should converge."""
        nv = 6
        cfg = ContactObserverConfig(K_O=50.0, dt=0.01, nv=nv)
        gmo = GeneralizedMomentumObserver(cfg)

        # Constant mass matrix (identity), zero Coriolis
        M = np.eye(nv)
        C = np.zeros((nv, nv))

        # Known external force
        tau_ext = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])

        # Simulate: v evolves as v_dot = tau_ext (since M=I, C=0, tau=0)
        v = np.zeros(nv)
        tau_applied = np.zeros(nv)

        gmo.reset(M, v)

        for k in range(500):  # 5 seconds at 100 Hz
            # Physics: v_k+1 = v_k + dt * tau_ext
            v = v + cfg.dt * tau_ext
            r = gmo.update(M, v, C, tau_applied)

        # Residual should converge to tau_ext
        np.testing.assert_allclose(r, tau_ext, atol=0.1,
            err_msg="GMO residual did not converge to tau_ext")

    def test_convergence_rate(self):
        """Time constant should be approximately 1/K_O."""
        nv = 3
        K_O = 80.0
        cfg = ContactObserverConfig(K_O=K_O, dt=0.01, nv=nv)
        gmo = GeneralizedMomentumObserver(cfg)

        M = np.eye(nv)
        C = np.zeros((nv, nv))
        tau_ext = np.array([10.0, 0.0, 0.0])

        v = np.zeros(nv)
        tau_applied = np.zeros(nv)
        gmo.reset(M, v)

        # After 1 time constant (1/K_O = 12.5ms = 1-2 steps at 100Hz),
        # r should reach ~63% of tau_ext.
        # After 5 time constants (~62.5ms = 6-7 steps), >99%.
        n_steps_5tau = int(5.0 / K_O / cfg.dt) + 1
        for k in range(n_steps_5tau):
            v = v + cfg.dt * tau_ext
            r = gmo.update(M, v, C, tau_applied)

        # Should be within 1% after 5 time constants
        np.testing.assert_allclose(r, tau_ext, rtol=0.05,
            err_msg="GMO did not converge within 5 time constants")


# ═══════════════════════════════════════════════════════════════════════════════
# T2 — Zero-force baseline
# ═══════════════════════════════════════════════════════════════════════════════

class TestZeroForce:
    """With no external forces, the residual should stay near zero."""

    def test_zero_residual_no_contact(self):
        """tau_ext = 0, tau_applied drives motion => r ~ 0."""
        nv = 6
        cfg = ContactObserverConfig(K_O=80.0, dt=0.01, nv=nv)
        gmo = GeneralizedMomentumObserver(cfg)

        M = np.eye(nv) * 5.0  # 5 kg
        C = np.zeros((nv, nv))
        tau_applied = np.array([1.0, -0.5, 0.2, 0.0, 0.0, 0.0])

        v = np.zeros(nv)
        gmo.reset(M, v)

        max_r = 0.0
        for k in range(200):
            # Physics: M v_dot = tau_applied => v_dot = M^{-1} tau
            v = v + cfg.dt * np.linalg.solve(M, tau_applied)
            r = gmo.update(M, v, C, tau_applied)
            max_r = max(max_r, np.linalg.norm(r))

        # Residual should stay small (numerical noise only)
        assert max_r < 1.0, f"Residual too large without external force: {max_r:.2f}"

    def test_zero_residual_with_coriolis(self):
        """Non-zero C(q,v) but no external force => r ~ 0."""
        nv = 4
        cfg = ContactObserverConfig(K_O=80.0, dt=0.01, nv=nv)
        gmo = GeneralizedMomentumObserver(cfg)

        M = np.eye(nv) * 2.0
        # Skew-symmetric C (physically plausible)
        C = np.array([[ 0.0,  0.1, -0.2,  0.0],
                      [-0.1,  0.0,  0.3,  0.0],
                      [ 0.2, -0.3,  0.0,  0.1],
                      [ 0.0,  0.0, -0.1,  0.0]])
        tau_applied = np.zeros(nv)

        v = np.array([0.1, -0.05, 0.02, 0.0])
        gmo.reset(M, v)

        for k in range(200):
            # Physics: M v_dot = -C v (Coriolis only)
            v_dot = np.linalg.solve(M, -C @ v)
            v = v + cfg.dt * v_dot
            r = gmo.update(M, v, C, tau_applied)

        assert np.linalg.norm(r) < 0.5, \
            f"Residual too large with Coriolis, no ext force: {np.linalg.norm(r):.2f}"


# ═══════════════════════════════════════════════════════════════════════════════
# T3 — State machine transitions
# ═══════════════════════════════════════════════════════════════════════════════

class TestStateMachine:
    """Verify correct state transitions."""

    def test_full_transition_sequence(self):
        cfg = ContactObserverConfig(
            F_threshold=5.0, d_proximity=0.020,
            d_contact=0.010, d_reset=0.030, debounce_count=3)
        sm = ContactStateMachine(cfg)

        # Start: NO_CONTACT
        assert sm.state == ContactState.NO_CONTACT
        assert not sm.is_docked

        # Far away: stays NO_CONTACT
        sm.update(0.0, 0.050)
        assert sm.state == ContactState.NO_CONTACT

        # Enter proximity zone: -> PROXIMITY
        sm.update(0.0, 0.015)
        assert sm.state == ContactState.PROXIMITY

        # Close but no force (sim mode): -> CONTACT
        sm.update(0.0, 0.008, force_mode=False)
        assert sm.state == ContactState.CONTACT

        # Debounce: 2 more cycles needed
        sm.update(0.0, 0.007, force_mode=False)
        assert sm.state == ContactState.CONTACT
        sm.update(0.0, 0.006, force_mode=False)
        assert sm.state == ContactState.CONTACT
        sm.update(0.0, 0.005, force_mode=False)
        assert sm.state == ContactState.CONFIRMED
        assert sm.is_docked

    def test_force_mode_requires_residual(self):
        """In force_mode, low distance alone should NOT trigger CONTACT."""
        cfg = ContactObserverConfig(
            F_threshold=5.0, d_proximity=0.020,
            d_contact=0.010, d_reset=0.030, debounce_count=3)
        sm = ContactStateMachine(cfg)

        sm.update(0.0, 0.015, force_mode=True)  # -> PROXIMITY
        assert sm.state == ContactState.PROXIMITY

        # Close but no force: stays PROXIMITY
        sm.update(0.0, 0.008, force_mode=True)
        assert sm.state == ContactState.PROXIMITY

        # Close with force: -> CONTACT
        sm.update(10.0, 0.008, force_mode=True)
        assert sm.state == ContactState.CONTACT

    def test_hysteresis_reset(self):
        """Moving far away resets to NO_CONTACT."""
        cfg = ContactObserverConfig(
            d_proximity=0.020, d_reset=0.030, debounce_count=1)
        sm = ContactStateMachine(cfg)

        sm.update(0.0, 0.015)  # -> PROXIMITY
        assert sm.state == ContactState.PROXIMITY

        # Move away past hysteresis
        sm.update(0.0, 0.035)
        assert sm.state == ContactState.NO_CONTACT


# ═══════════════════════════════════════════════════════════════════════════════
# T4 — Debounce prevents false CONFIRMED
# ═══════════════════════════════════════════════════════════════════════════════

class TestDebounce:
    """Transient contact should not trigger CONFIRMED."""

    def test_transient_does_not_confirm(self):
        cfg = ContactObserverConfig(
            F_threshold=5.0, d_proximity=0.020,
            d_contact=0.010, d_reset=0.030, debounce_count=5)
        sm = ContactStateMachine(cfg)

        # Enter proximity
        sm.update(0.0, 0.015)
        # Momentary contact (only 2 cycles, need 5)
        sm.update(0.0, 0.008, force_mode=False)  # CONTACT
        sm.update(0.0, 0.008, force_mode=False)  # counter=1
        assert sm.state == ContactState.CONTACT

        # Bounce out
        sm.update(0.0, 0.015, force_mode=False)  # back to PROXIMITY
        assert sm.state == ContactState.PROXIMITY
        assert not sm.is_docked

    def test_sustained_contact_confirms(self):
        cfg = ContactObserverConfig(
            d_proximity=0.020, d_contact=0.010,
            d_reset=0.030, debounce_count=5)
        sm = ContactStateMachine(cfg)

        sm.update(0.0, 0.015)  # PROXIMITY
        for i in range(7):
            sm.update(0.0, 0.005, force_mode=False)

        assert sm.state == ContactState.CONFIRMED


# ═══════════════════════════════════════════════════════════════════════════════
# T5 — Reset behavior
# ═══════════════════════════════════════════════════════════════════════════════

class TestReset:
    """After reset, observer returns to initial state."""

    def test_observer_reset(self):
        nv = 6
        cfg = ContactObserverConfig(K_O=80.0, dt=0.01, nv=nv)
        gmo = GeneralizedMomentumObserver(cfg)

        M = np.eye(nv)
        v = np.ones(nv)
        tau_ext = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        gmo.reset(M, v)
        # Run a few steps with external force
        for _ in range(50):
            v = v + cfg.dt * tau_ext
            gmo.update(M, v, np.zeros((nv, nv)), np.zeros(nv))

        assert np.linalg.norm(gmo.residual) > 1.0  # nonzero

        # Reset
        gmo.reset(M, v)
        assert np.linalg.norm(gmo.residual) < 1e-10  # back to zero

    def test_state_machine_reset(self):
        cfg = ContactObserverConfig(d_proximity=0.020, debounce_count=1)
        sm = ContactStateMachine(cfg)

        sm.update(0.0, 0.015)
        sm.update(0.0, 0.005, force_mode=False)
        sm.update(0.0, 0.005, force_mode=False)
        assert sm.state == ContactState.CONFIRMED

        sm.reset()
        assert sm.state == ContactState.NO_CONTACT
        assert not sm.is_docked
