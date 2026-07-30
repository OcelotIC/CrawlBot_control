"""Is the h_w box redundant with the wheel-torque rate cap?

The two NMPC momentum constraints are not independent — they bound the same
physical quantity at different orders:

    rate    |Hdot_s,i| <= tau_w_max        the moment the wheels must absorb NOW
    level   |h_w,i|    <= h_max_tight      the momentum they have accumulated

and h_w is (up to sign and wheel inertia) the integral of that moment. So over a
horizon of length T the level bound is IMPLIED by the rate bound whenever

    T * tau_w_max <= h_max_tight

This measures where the canonical sits against that inequality, per axis, and
how much of the box is actually reachable — expressed in SECONDS OF SATURATED
tau_w, which is the units that matter operationally.

It also quantifies the structure's own angular momentum I_s.omega_s, which the
conservation-law reconstruction does not attribute to the wheels.

Run:
    MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_nmpc_momentum_budget.py
"""
import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT,
                   'results/j2_adjconv/nmpc_sweep/F2off_ctl_N20/fulldiag_fulldiag.csv')
MJCF = os.path.join(ROOT, 'models/VISPA_crawling_rwa3.xml')
AXES = 'xyz'
TAU_W_MAX = 2.5      # config.py tau_w_max
H_MAX = 5.0          # _make_m7_config h_max_tight
N, DT = 20, 0.1      # horizon


def col(rows, n):
    o = []
    for r in rows:
        try:
            o.append(float(r[n]))
        except (TypeError, ValueError, KeyError):
            o.append(np.nan)
    return np.asarray(o)


def main():
    rows = list(csv.DictReader(open(RUN)))
    tw = np.stack([col(rows, f'tauw_{k}_Nm') for k in AXES], axis=1)
    hw = np.stack([col(rows, f'hw_{k}_Nms') for k in AXES], axis=1)
    om = np.stack([col(rows, f'omega_s_{k}_radps') for k in AXES], axis=1)
    ok = np.isfinite(tw).all(1) & np.isfinite(hw).all(1)
    tw, hw = tw[ok], hw[ok]

    T = N * DT
    print('=' * 72)
    print('MOMENTUM BUDGET — is the level box redundant with the rate cap?')
    print('=' * 72)
    print(f'  horizon T = N*dt = {T:.1f} s      tau_w_max = {TAU_W_MAX} Nm'
          f'      h_max_tight = {H_MAX} Nms')
    print(f'  T * tau_w_max = {T * TAU_W_MAX:.1f} Nms  vs  h_max = {H_MAX} Nms'
          f'   -> {"CRITICALLY BALANCED" if abs(T*TAU_W_MAX - H_MAX) < 1e-9 else "not balanced"}')
    print('\n  Reading: from h_w(0)=0 the rate cap alone already implies the box,')
    print('  so the box adds information ONLY through the initial condition.')

    print(f'\n{"axis":>5}{"|tau_w| peak":>14}{"at cap":>10}'
          f'{"|h_w| peak":>12}{"box headroom":>14}{"= sat. sec":>12}')
    out = {}
    for i, k in enumerate(AXES):
        sat = int((np.abs(tw[:, i]) > TAU_W_MAX - 0.025).sum())
        pk = float(np.abs(hw[:, i]).max())
        room = H_MAX - pk
        print(f'{k:>5}{np.abs(tw[:, i]).max():>14.4f}{sat:>7}/{len(tw)}'
              f'{pk:>12.4f}{room:>14.4f}{room / TAU_W_MAX:>12.2f}')
        out[k] = {'tau_w_peak': float(np.abs(tw[:, i]).max()),
                  'ticks_at_cap': sat, 'hw_peak': pk,
                  'headroom_Nms': float(room),
                  'headroom_saturated_seconds': float(room / TAU_W_MAX)}

    print('\n  "sat. sec" = how long tau_w could stay saturated in one direction')
    print('  before the box is reached. Compare against the 2.0 s horizon:')
    for k in AXES:
        s = out[k]['headroom_saturated_seconds']
        verdict = ('REACHABLE within the horizon' if s < T
                   else 'not reachable within the horizon')
        print(f'    {k}: {s:.2f} s  -> {verdict}')

    # ---- the term the reconstruction does not carry --------------------
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_path(MJCF)
        I = np.asarray(m.body_inertia[1], dtype=float)
        omf = om[np.isfinite(om).all(1)]
        Iw = np.abs(omf) * I
        print(f'\n  structure inertia diag = {np.round(I, 1).tolist()} kg m^2')
        print(f'  {"axis":>5}{"|omega_s| peak":>16}{"|I_s omega_s| peak":>21}')
        for i, k in enumerate(AXES):
            print(f'{k:>5}{np.abs(omf[:, i]).max():>16.6f}{Iw[:, i].max():>21.4f}')
            out[k]['I_s_omega_s_peak_Nms'] = float(Iw[:, i].max())
        print(f'\n  peak structure momentum {Iw.max():.3f} Nms vs the tightest box'
              f' headroom {min(out[k]["headroom_Nms"] for k in AXES):.3f} Nms')
        print('  -> the same order. `c_simple` omits I_s.omega_s, so the inferred')
        print('     h_w and the physical wheel momentum can differ by this much.')
    except Exception as exc:                       # pragma: no cover
        print(f'\n  (structure inertia unavailable: {exc})')

    dest = os.path.join(ROOT, 'results/j2_adjconv/nmpc_sweep/momentum_budget.json')
    with open(dest, 'w') as fh:
        json.dump({'T_horizon_s': T, 'tau_w_max': TAU_W_MAX, 'h_max': H_MAX,
                   'T_times_tau_w_max': T * TAU_W_MAX, 'per_axis': out}, fh, indent=2)
    print(f'\nwrote {dest}')


if __name__ == '__main__':
    main()
