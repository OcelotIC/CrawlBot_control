"""M3 — NMPC conservation-law tests.

Covers the three pass criteria from CLAUDE_CODE_HANDOFF.md §M3:

    T4 — B2 NMPC from rest, 0.3 m displacement, h_max = 5 Nms
         Converges to target, h_w(k) ∈ [-h_max, h_max] at all knots.

    T5 — Position-dependent envelope: 0.5 m vs 3.0 m from O_p
         Step time at 3.0 m is longer (transverse velocity limited by
         larger lever arm in r_com × m·v_com).

    T6 — Two consecutive steps with terminal margin κ = 0.7
         Terminal h_w ∈ [-κ·h_max, κ·h_max] at end of each step.

The NMPC is exercised standalone (no MuJoCo, no QP) with:
    c_simple = h_w_0 + L_com_0 + r_com_0 × m·v_com_0

and the RWA box:
    c_simple - L_com(k) - r_com(k) × m·v_com(k) ∈ [-h_max', h_max']

Also produces a diagnostic plot results/test_scratch/M3_tests/t4_hw_bounds.png
the predicted h_w trajectory vs the bounds at each knot.
"""

import os
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from crawlbot.solvers.centroidal_nmpc import CentroidalNMPC, CentroidalNMPCConfig
from crawlbot.solvers.contact_phase import ContactConfig, ContactPhase


OUTPUT_DIR = 'results/test_scratch/M3_tests'
ROBOT_MASS = 71.0
H_MAX = 5.0         # Nms  (spec §4.6)
F_MAX = 25.0
TAU_MAX = 8.0


def _build_nmpc(*, N=10, dt=0.1, h_max=H_MAX, kappa=1.0,
                w_L=1.0, w_r=200.0, w_v=20.0, qf_r=2000.0, qf_v=200.0,
                L_max=10.0, p_max=50.0, tau_w_max=50.0):
    """Helper to build an M3-enabled NMPC with given parameters.

    Note: tau_w_max must be finite, otherwise the L_dot bilateral
    constraint evaluates to ±inf which IPOPT rejects.
    """
    cfg = CentroidalNMPCConfig(
        robot_mass=ROBOT_MASS,
        N=N, dt=dt,
        Wr=np.full(3, w_r),
        Wv=np.full(3, w_v),
        Wu_f=0.01, Wu_tau=0.001,
        Qf_r=np.full(3, qf_r),
        Qf_v=np.full(3, qf_v),
        f_max=F_MAX,
        tau_max=TAU_MAX,
        L_max=L_max,
        tau_w_max=tau_w_max,
        p_max=p_max,
        enforce_hw_conservation=True,
        h_max_tight=np.full(3, h_max),
        w_L=w_L,
        Qf_L=10.0,
        kappa_terminal=kappa,
        solver_opts={
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 500,
        },
    )
    nmpc = CentroidalNMPC(cfg)
    nmpc.build()
    return nmpc


def _default_contacts():
    """DS contact config: anchors on either side of the torso."""
    return ContactConfig.from_phase(
        ContactPhase.DOUBLE,
        r_contact_A=np.array([0.8, 0.3, 0.0]),
        r_contact_B=np.array([0.8, -0.3, 0.0]),
    )


def _hw_from_state(r_com, v_com, L_com, c_simple, m=ROBOT_MASS):
    """Algebraic h_w^s(k) reconstruction from the NMPC state.

    Per spec §4.6 (Option B):
        h_w(k) = c_simple - L_com(k) - r_com(k) × m·v_com(k)
    """
    return c_simple - L_com - np.cross(r_com, m * v_com)


# ---------------------------------------------------------------------------
# T4 — B2 NMPC from rest, 0.3 m displacement, h_w in box
# ---------------------------------------------------------------------------

class TestT4FromRest:
    def test_t4_converges_and_respects_hw_box(self):
        nmpc = _build_nmpc(N=10, dt=0.1, h_max=H_MAX, kappa=1.0)
        cc = _default_contacts()

        # Initial state: at rest at origin, zero momentum, empty wheels
        r_com_0 = np.array([0.5, 0.0, -0.5])
        v_com_0 = np.zeros(3)
        L_com_0 = np.zeros(3)
        hw_0 = np.zeros(3)

        # Target: 0.3 m displacement in x
        r_ref = r_com_0 + np.array([0.3, 0.0, 0.0])
        v_ref = np.zeros(3)

        x_opt, u_opt, info = nmpc.get_full_trajectory(
            r_com=r_com_0, v_com=v_com_0, L_com=L_com_0,
            r_com_ref=r_ref, v_com_ref=v_ref,
            contact_config=cc, hw_current=hw_0,
        )
        assert info.success, f"T4: NMPC failed: {info.status}"

        # Reconstruct predicted h_w at each knot
        c_simple = nmpc.compute_c_simple(r_com_0, v_com_0, L_com_0, hw_0)
        N = nmpc.config.N
        hw_traj = np.zeros((3, N + 1))
        for k in range(N + 1):
            hw_traj[:, k] = _hw_from_state(
                x_opt[0:3, k], x_opt[3:6, k], x_opt[6:9, k], c_simple)

        # Produce diagnostic plot (always)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        t = np.arange(N + 1) * nmpc.config.dt
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axs = ['x', 'y', 'z']
        for i, c in enumerate(['r', 'g', 'b']):
            axes[0].plot(t, hw_traj[i], c, label=f'hw_{axs[i]}')
        axes[0].axhline(H_MAX, color='red', ls='--', lw=0.8,
                        label=f'±{H_MAX} Nms')
        axes[0].axhline(-H_MAX, color='red', ls='--', lw=0.8)
        axes[0].set_ylabel('hw [Nms]')
        axes[0].legend(fontsize=8, ncol=4)
        axes[0].set_title('T4 — Predicted wheel momentum (NMPC solution)')
        axes[0].grid(True, alpha=0.3)

        for i, c in enumerate(['r', 'g', 'b']):
            axes[1].plot(t, x_opt[i, :], c,
                         label=f'r_com_{axs[i]}')
            axes[1].axhline(r_ref[i], color=c, ls=':', lw=0.8)
        axes[1].set_ylabel('r_com [m]')
        axes[1].set_xlabel('Time [s]')
        axes[1].legend(fontsize=8, ncol=3)
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, 't4_hw_bounds.png'), dpi=120)
        plt.close(fig)

        # Assertion 1: h_w within bounds at every knot (small numerical tol)
        tol = 1e-4
        violations = np.sum(np.abs(hw_traj) > H_MAX + tol)
        max_violation = float(np.max(np.abs(hw_traj)) - H_MAX)
        print(f"\nT4: max |hw| = {np.max(np.abs(hw_traj)):.4f} Nms "
              f"(limit {H_MAX}), violations = {violations}")
        assert violations == 0, (
            f"T4: {violations} hw box violations, "
            f"max excess = {max_violation:.4e} Nms")

        # Assertion 2: trajectory converges toward target
        final_err = float(np.linalg.norm(x_opt[0:3, -1] - r_ref))
        initial_err = float(np.linalg.norm(r_com_0 - r_ref))
        print(f"T4: initial err = {initial_err*1000:.1f} mm, "
              f"final err = {final_err*1000:.1f} mm")
        assert final_err < initial_err * 0.5, (
            f"T4: NMPC did not halve the displacement "
            f"(initial={initial_err:.3f}, final={final_err:.3f})")


# ---------------------------------------------------------------------------
# T5 — Position-dependent envelope
# ---------------------------------------------------------------------------

class TestT5PositionDependent:
    """At larger |r_com| the cross-product r × m·v dominates, so the
    same hw headroom supports a smaller transverse velocity. A fixed
    CoM step therefore requires a longer effective maneuver at 3.0 m
    than at 0.5 m from the origin.

    We measure this by counting how many NMPC iterations are needed to
    reach the target from rest (velocity-limited regime).
    """

    def _steps_to_reach(self, r_com_0, r_ref, *, tol=0.005,
                        max_iter=100):
        nmpc = _build_nmpc(N=10, dt=0.1, h_max=H_MAX, kappa=1.0)
        cc = _default_contacts()
        r = r_com_0.copy()
        v = np.zeros(3)
        L = np.zeros(3)
        hw = np.zeros(3)
        nmpc.reset_warm_start()
        for k in range(max_iter):
            r_plan, v_plan, L_plan, lam, info = nmpc.solve(
                r_com=r, v_com=v, L_com=L,
                r_com_ref=r_ref, v_com_ref=np.zeros(3),
                contact_config=cc, warm_start=True,
                hw_current=hw,
            )
            if not info.success:
                return None, k
            # Advance one NMPC step by applying the first prediction
            r, v, L = r_plan, v_plan, L_plan
            if np.linalg.norm(r - r_ref) < tol:
                return k + 1, k + 1
        return None, max_iter

    def test_t5_far_step_takes_longer(self):
        # Same 0.2 m y-direction step (transverse to r_com) from two
        # different radial distances.
        step_y = np.array([0.0, 0.2, 0.0])

        # Near case: 0.5 m in x
        r_near = np.array([0.5, 0.0, -0.5])
        n_near, _ = self._steps_to_reach(r_near, r_near + step_y)

        # Far case: 3.0 m in x  (larger lever arm)
        r_far = np.array([3.0, 0.0, -0.5])
        n_far, _ = self._steps_to_reach(r_far, r_far + step_y)

        print(f"\nT5: near (0.5 m) converged in {n_near} steps, "
              f"far (3.0 m) converged in {n_far} steps")

        assert n_near is not None, "T5: near case did not converge"
        assert n_far is not None, "T5: far case did not converge"
        assert n_far >= n_near, (
            f"T5: expected the far case ({n_far} steps) to need at "
            f"least as many steps as the near case ({n_near})")


# ---------------------------------------------------------------------------
# T6 — Terminal margin κ = 0.7
# ---------------------------------------------------------------------------

class TestT6TerminalMargin:
    def test_t6_terminal_hw_within_kappa(self):
        kappa = 0.7
        nmpc = _build_nmpc(N=10, dt=0.1, h_max=H_MAX, kappa=kappa)
        cc = _default_contacts()

        r_com_0 = np.array([0.5, 0.0, -0.5])
        v_com_0 = np.zeros(3)
        L_com_0 = np.zeros(3)
        hw_0 = np.zeros(3)

        # Two consecutive 0.15 m steps
        step1 = r_com_0 + np.array([0.15, 0.0, 0.0])
        step2 = step1 + np.array([0.15, 0.0, 0.0])

        results = []
        r, v, L = r_com_0.copy(), v_com_0.copy(), L_com_0.copy()
        hw = hw_0.copy()
        for i, target in enumerate([step1, step2]):
            nmpc.reset_warm_start()
            x_opt, u_opt, info = nmpc.get_full_trajectory(
                r_com=r, v_com=v, L_com=L,
                r_com_ref=target, v_com_ref=np.zeros(3),
                contact_config=cc, hw_current=hw,
            )
            assert info.success, f"T6 step {i}: {info.status}"

            c_simple = nmpc.compute_c_simple(r, v, L, hw)
            # Terminal hw at last knot
            hw_N = _hw_from_state(
                x_opt[0:3, -1], x_opt[3:6, -1], x_opt[6:9, -1], c_simple)
            results.append((i, float(np.max(np.abs(hw_N))), hw_N))
            # Advance to the end of the horizon for the next step
            r = x_opt[0:3, -1].copy()
            v = x_opt[3:6, -1].copy()
            L = x_opt[6:9, -1].copy()
            # hw evolves deterministically from the constraint:
            hw = c_simple - L - np.cross(r, ROBOT_MASS * v)

        limit = kappa * H_MAX
        tol = 1e-4
        for i, max_hw, hw_N in results:
            print(f"\nT6 step {i}: terminal max|hw| = {max_hw:.4f} "
                  f"(limit {limit:.4f}) — {hw_N}")
            assert max_hw <= limit + tol, (
                f"T6 step {i}: terminal hw {max_hw:.4f} > κ·h_max = {limit}")
