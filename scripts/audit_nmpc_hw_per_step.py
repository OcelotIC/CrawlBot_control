"""Would a LONGER PATH saturate the h_w box? Per-step decomposition.

The box is reached either by (a) a bigger excursion inside one step, or (b)
residues that add up across steps. `audit_nmpc_momentum_budget.py` showed the
net drift over the whole run is small, but that number is contaminated: the run
ends with a 20 s DS_terminal settle which unwinds the wheels completely. What
matters for a longer traversal is the carry ACROSS crawl steps, with the
terminal settle excluded.

So this splits the run at the phase level:

    step n = SS (swing) + DS_interstep (double support)      <- the crawl
    step 5   ends in DS_terminal, 20 s                       <- the settle

and measures the per-step carry as h_w at successive SS entries. A systematic
carry grows like N and a 20 m path reaches the box; a zero-mean one grows like
sqrt(N) and it does not.

Run:
    MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_nmpc_hw_per_step.py
"""
import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT,
                   'results/j2_adjconv/nmpc_sweep/F2off_ctl_N20/fulldiag_fulldiag.csv')
AXES = 'xyz'
TAU_W_MAX = 2.5
H_MAX = 5.0
H_HONEST = 3.1          # h_max_tight if the I_s.dw_s allowance were paid


def col(rows, n, cast=float):
    o = []
    for r in rows:
        try:
            o.append(cast(r[n]))
        except (TypeError, ValueError, KeyError):
            o.append(np.nan)
    return np.asarray(o)


def longest_run(mask, dt):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best * dt


def main():
    rows = list(csv.DictReader(open(RUN)))
    t = col(rows, 't_s')
    ph = np.array([r['phase'] for r in rows])
    si = np.array([r['step_index'] for r in rows])
    hw = np.stack([col(rows, f'hw_{k}_Nms') for k in AXES], axis=1)
    tw = np.stack([col(rows, f'tauw_{k}_Nm') for k in AXES], axis=1)
    rc = np.stack([col(rows, f'r_com_{k}_m') for k in AXES], axis=1)
    ok = np.isfinite(hw).all(1) & np.isfinite(tw).all(1) & np.isfinite(t)
    t, ph, si, hw, tw, rc = t[ok], ph[ok], si[ok], hw[ok], tw[ok], rc[ok]
    dt = float(np.median(np.diff(t)))

    steps = sorted({s for s in si if s not in ('', 'None') and float(s) >= 0},
                   key=lambda s: int(float(s)))

    print('=' * 78)
    print('WOULD A LONGER PATH REACH THE BOX?   crawl carry vs terminal settle')
    print('=' * 78)

    # ---- how far does one step actually travel? ------------------------
    fin = np.isfinite(rc).all(1)
    reach = []
    for s in steps:
        m = (si == s) & fin
        if m.sum() > 2:
            reach.append(float(np.linalg.norm(rc[m][-1] - rc[m][0])))
    print(f'  CoM travel per step: {np.round(reach, 3).tolist()} m')
    print(f'  mean {np.mean(reach):.3f} m  ->  a 20 m path is '
          f'~{20.0/np.mean(reach):.0f} steps')

    # ---- per-step markers: h_w at each SS entry ------------------------
    ss_entry_t, ss_entry_h, ss_exit_t, ss_exit_h = [], [], [], []
    for s in steps:
        m = (si == s) & (ph == 'SS')
        if m.sum() < 3:
            continue
        ss_entry_t.append(t[m][0]);  ss_entry_h.append(hw[m][0])
        ss_exit_t.append(t[m][-1]);  ss_exit_h.append(hw[m][-1])
    ss_entry_h = np.asarray(ss_entry_h)
    ss_exit_h = np.asarray(ss_exit_h)

    term = ph == 'DS_terminal'
    print(f'\n  crawl = {len(ss_entry_h)} steps (SS + DS_interstep), '
          f't {ss_entry_t[0]:.2f} .. {t[term][0]:.2f} s')
    print(f'  settle = DS_terminal, {t[term][-1]-t[term][0]:.1f} s '
          f'-- EXCLUDED from the carry')

    out = {}
    for ax, k in enumerate(AXES):
        print(f'\n--- axis {k} ---')
        print(f'  {"step":>5}{"SS entry":>11}{"peak":>10}{"excursion":>12}'
              f'{"carry":>10}{"sat run[s]":>12}{"sat frac":>10}')
        carries, exc, runs = [], [], []
        for i, s in enumerate(steps[:len(ss_entry_h)]):
            m = si == s
            h, w = hw[m, ax], tw[m, ax]
            pk = h[int(np.argmax(np.abs(h)))]
            e = ss_entry_h[i, ax]
            c = (ss_entry_h[i + 1, ax] - e) if i + 1 < len(ss_entry_h) else np.nan
            sat = np.abs(w) > TAU_W_MAX - 0.025
            print(f'  {int(float(s)):>5}{e:>+11.3f}{pk:>+10.3f}{pk-e:>+12.3f}'
                  f'{c:>+10.3f}{longest_run(sat, dt):>12.2f}{sat.mean():>9.1%}')
            exc.append(abs(pk - e)); runs.append(longest_run(sat, dt))
            if np.isfinite(c):
                carries.append(float(c))
        carries = np.asarray(carries)
        ratio = abs(carries.mean()) / (carries.std() + 1e-12)
        verdict = ('SYSTEMATIC — grows ~N' if ratio > 1.0
                   else 'zero-mean — grows ~sqrt(N)')
        print(f'\n  carry across crawl steps: mean {carries.mean():+.4f} '
              f'std {carries.std():.4f}  |mean|/std = {ratio:.2f}  -> {verdict}')
        settle = float(hw[term][-1, ax] - hw[term][0, ax])
        print(f'  the {t[term][-1]-t[term][0]:.0f} s DS_terminal settle then '
              f'moves h_w by {settle:+.3f} Nms (undoes '
              f'{100*abs(settle)/(abs(carries.sum())+1e-9):.0f} % of the crawl carry)')
        print(f'  within-step excursion from entry: mean {np.mean(exc):.3f} '
              f'max {np.max(exc):.3f} Nms   longest saturated run {np.max(runs):.2f} s')
        out[k] = {'carry_per_step': carries.tolist(),
                  'carry_mean': float(carries.mean()),
                  'carry_std': float(carries.std()),
                  'systematic_ratio': float(ratio),
                  'terminal_settle_delta': settle,
                  'excursion_mean': float(np.mean(exc)),
                  'excursion_max': float(np.max(exc)),
                  'sat_run_max_s': float(np.max(runs))}

    # ---- extrapolate ----------------------------------------------------
    print('\n' + '=' * 78)
    print('EXTRAPOLATION — peak(N) ~ |carry|*N + excursion, no terminal settle')
    print('=' * 78)
    print(f'  {"axis":>5}{"carry/step":>12}{"excursion":>11}{"peak now":>10}'
          f'{"N to 3.1":>10}{"N to 5.0":>10}{"= metres":>10}')
    step_m = float(np.mean(reach))
    for k in AXES:
        o = out[k]
        mu, e = abs(o['carry_mean']), o['excursion_max']
        pk_now = max(abs(np.nanmax(hw[:, AXES.index(k)])),
                     abs(np.nanmin(hw[:, AXES.index(k)])))
        n31 = (H_HONEST - e) / mu if mu > 1e-9 else np.inf
        n50 = (H_MAX - e) / mu if mu > 1e-9 else np.inf
        print(f'{k:>5}{o["carry_mean"]:>+12.4f}{e:>11.3f}{pk_now:>10.3f}'
              f'{max(n31,0):>10.0f}{max(n50,0):>10.0f}{max(n50,0)*step_m:>10.1f}')
        o['steps_to_3p1'] = float(max(n31, 0))
        o['steps_to_5p0'] = float(max(n50, 0))
        o['metres_to_5p0'] = float(max(n50, 0) * step_m)

    dest = os.path.join(ROOT, 'results/j2_adjconv/nmpc_sweep/hw_per_step.json')
    with open(dest, 'w') as fh:
        json.dump({'n_crawl_steps': len(ss_entry_h), 'step_travel_m': step_m,
                   'tau_w_max': TAU_W_MAX, 'h_max': H_MAX, 'h_honest': H_HONEST,
                   'per_axis': out}, fh, indent=2)
    print(f'\nwrote {dest}')


if __name__ == '__main__':
    main()
