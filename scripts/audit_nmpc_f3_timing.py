"""F3 proof: prediction step vs control period, decoupled.

`nmpc_pred_dt` is the NLP's prediction step; `nmpc_period` is how often the controller
re-solves. They are different quantities (standard MPC lets them differ), but
three code paths advanced the plan by ONE prediction knot per CONTROL period,
which is correct only when the two are equal:

  1. sim_loop plan interpolation across the QP sub-loop
  2. CentroidalNMPC.get_shifted_fallback
  3. NMPCSolver.shift_warm_start

Checks:

  A. REDUCTION — the new time-indexed interpolation must equal the old
     `alpha = qs / n_qp_per_nmpc` form EXACTLY when nmpc_pred_dt == nmpc_period, so the
     committed configuration is bit-for-bit unchanged.
  B. EFFECT — when nmpc_pred_dt = nmpc_period/2 the two forms must differ, and the new
     one must track the plan at real speed while the old one lags 2x.
  C. SHIFT — n_shift_per_control_period must be 1 when equal and 2 when the
     prediction step is half the control period, and the shifted trajectory
     must advance by that many knots.

Run:
    MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_nmpc_f3_timing.py
"""
import numpy as np

from crawlbot.solvers.centroidal_nmpc import CentroidalNMPC, CentroidalNMPCConfig

DT_QP = 0.01


def old_interp(plan, qs, n_qp_per_nmpc):
    """The pre-fix form: knots 0->1 across one control period."""
    a = qs / n_qp_per_nmpc
    return (1.0 - a) * plan[:, 0] + a * plan[:, 1]


def new_interp(plan, qs, nmpc_pred_dt):
    """The fixed form: index the plan by elapsed time in INTEGER knot units.

    `qp_per_knot` is an int, so `u` is formed exactly as the old code formed
    its alpha — algebraically equivalent forms like `(qs*dt_qp)/nmpc_pred_dt` differ
    by 1 ULP and would break bit-identity.
    """
    qp_per_knot = int(round(nmpc_pred_dt / DT_QP))
    u = qs / qp_per_knot
    k = min(int(np.floor(u)), plan.shape[1] - 2)
    a = min(max(u - k, 0.0), 1.0)
    return (1.0 - a) * plan[:, k] + a * plan[:, k + 1]


def make_plan(N, nmpc_pred_dt, speed=1.0):
    """A CoM plan moving at constant `speed` m/s along +x, knots at nmpc_pred_dt."""
    return np.stack([np.array([speed * k * nmpc_pred_dt, 0.0, 0.0])
                     for k in range(N + 1)], axis=1)


def main():
    ok = True

    # ---- A. reduction when the periods are equal ----------------------
    print('=' * 68)
    print('F3 TIMING PROOF')
    print('=' * 68)
    nmpc_period, nmpc_pred_dt, N = 0.1, 0.1, 20
    n_qp = int(round(nmpc_period / DT_QP))
    plan = make_plan(N, nmpc_pred_dt)
    worst = 0.0
    for qs in range(n_qp):
        worst = max(worst, float(np.max(np.abs(
            new_interp(plan, qs, nmpc_pred_dt) - old_interp(plan, qs, n_qp)))))
    print(f'\n[A] REDUCTION  nmpc_pred_dt = nmpc_period = {nmpc_pred_dt}')
    print(f'  max |new - old| over the {n_qp} sub-steps = {worst:.3e} m')
    a_ok = worst == 0.0
    print(f'  -> {"PASS" if a_ok else "FAIL"} (must be EXACTLY 0 — bit-identity)')
    ok &= a_ok

    # ---- B. effect when the prediction step is finer -------------------
    nmpc_period, nmpc_pred_dt = 0.1, 0.05
    n_qp = int(round(nmpc_period / DT_QP))
    plan = make_plan(N, nmpc_pred_dt, speed=1.0)     # 1 m/s
    print(f'\n[B] EFFECT  nmpc_pred_dt = {nmpc_pred_dt}, nmpc_period = {nmpc_period} '
          f'(plan moves 1.0 m/s)')
    print(f'  {"qs":>3} {"t [s]":>7} {"truth":>9} {"new":>9} {"old":>9} '
          f'{"old err":>9}')
    max_old_err, max_new_err = 0.0, 0.0
    for qs in range(0, n_qp, 2):
        t = qs * DT_QP
        truth = t                                # x = 1.0 * t
        xn = float(new_interp(plan, qs, nmpc_pred_dt)[0])
        xo = float(old_interp(plan, qs, n_qp)[0])
        max_new_err = max(max_new_err, abs(xn - truth))
        max_old_err = max(max_old_err, abs(xo - truth))
        print(f'  {qs:>3} {t:>7.2f} {truth:>9.4f} {xn:>9.4f} {xo:>9.4f} '
              f'{xo - truth:>+9.4f}')
    print(f'  max error   new = {max_new_err:.3e} m   old = {max_old_err:.3e} m')
    # The old form walks knot0->knot1 over 0.1 s, but that segment is only
    # 0.05 s of plan time: it delivers half the motion. Reference dilated 2x.
    b_ok = max_new_err < 1e-15 and max_old_err >= 0.039
    print(f'  -> {"PASS" if b_ok else "FAIL"} '
          f'(new exact; old lags ~2x as predicted)')
    ok &= b_ok

    # ---- C. the warm-start / fallback shift ---------------------------
    print('\n[C] SHIFT  n_shift_per_control_period')
    for period, step, want in ((0.1, 0.1, 1), (0.1, 0.05, 2), (0.1, 0.025, 4)):
        cfg = CentroidalNMPCConfig(N=8, dt=step, control_period=period)
        got = CentroidalNMPC(cfg).n_shift_per_control_period
        good = got == want
        ok &= good
        print(f'  control_period={period}  dt={step:<6} -> n={got}  '
              f'(expect {want})  {"ok" if good else "FAIL"}')
    # Default (control_period=None) must behave exactly as before.
    d = CentroidalNMPC(CentroidalNMPCConfig(N=8, dt=0.1)).n_shift_per_control_period
    print(f'  control_period=None (legacy default)     -> n={d}  (expect 1)  '
          f'{"ok" if d == 1 else "FAIL"}')
    ok &= (d == 1)

    # And the fallback must actually advance by n knots.
    print('\n[C2] fallback advances by n knots')
    for step, want in ((0.1, 1), (0.05, 2)):
        cfg = CentroidalNMPCConfig(N=8, dt=step, control_period=0.1)
        nm = CentroidalNMPC(cfg)
        x = np.stack([np.arange(9, dtype=float) + 100 * i for i in range(9)])
        nm._last_x_opt = x.copy()
        nm._last_u_opt = np.zeros((12, 8))
        xs, _ = nm.get_shifted_fallback()
        good = np.allclose(xs[:, 0], x[:, want])
        ok &= good
        print(f'  dt={step:<6} n={want}: shifted[:,0] == original[:,{want}]  '
              f'{"ok" if good else "FAIL"}')

    print(f'\nF3 TIMING: {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
