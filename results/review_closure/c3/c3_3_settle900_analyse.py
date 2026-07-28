#!/usr/bin/env python3
"""C3.3 §3 — close the tau_w vs dh_w/dt anomaly on the instrumented 900 s settle.

C3.3 found mean commanded tau_w exceeding realized dh_w/dt by 45-81x, sign
reversed, on all three axes, using the committed t4b trace. Four candidate
explanations, tested here against the full 10 Hz log:

  H1  ALIASING. t4b_trace_900s.csv is subsampled to 0.2 s from a 10 Hz log. If
      tau_w oscillates near the tick rate, the subsampled mean is not the true
      mean. Test: recompute the mean from every tick and compare.
  H2  WHEEL INERTIA. h_w is logged as rwa_I_w * qvel = 0.01 * qvel, but the
      MJCF wheel joints carry armature="0.01", so the EFFECTIVE joint inertia
      is 0.02. dh_w/dt would then be half the applied torque -- a factor 2,
      not 45x, but it must be accounted for before blaming anything else.
  H3  WHEEL DAMPING. damping="1e-4" N.m.s/rad opposes the motor. At steady
      state tau_motor = b * omega_w, so a sustained torque produces a bounded
      speed, not a ramp. Test: is omega_w near tau/b?
  H4  A GENUINE COMMAND-VS-APPLIED GAP.

H1-H3 are all measurable from this run. H4 is the residual if they fail.
"""
import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
RUN = os.path.join(ROOT, 'results', 'review_closure', 'c3', 'settle900')
OUT = os.path.join(ROOT, 'results', 'review_closure', 'c3')
I_W = 0.01          # config.rwa_I_w  == MJCF wheel spin inertia
ARMATURE = 0.01     # MJCF wheel joint armature
DAMPING = 1e-4      # MJCF wheel joint damping [N.m.s/rad]


def main():
    sl = json.load(open(os.path.join(RUN, 'sim_log.json')))
    t = np.array(sl['t'], float)
    hw = np.array(sl['hw_physical'], float)
    tw = np.array(sl['tau_w'], float)
    ph = np.array(sl['phase'])
    res = {'n_ticks': len(t), 't_end_s': float(t[-1])}

    # settle window: after the last dock, which is the trailing DS
    dev = sl.get('dock_events', [])
    t_last = dev[-1]['t'] if dev else 0.0
    m = t > t_last + 30.0            # skip the first 30 s of transient
    res['settle_window_s'] = [float(t[m].min()), float(t[m].max())]
    res['settle_ticks'] = int(m.sum())
    res['dt_median_s'] = float(np.median(np.diff(t[m])))

    tau_mean = tw[m].mean(0)
    dhw = np.array([np.polyfit(t[m], hw[m, j], 1)[0] for j in range(3)])
    res['H1_full_rate'] = {
        'mean_tau_w_mNm': [round(float(x) * 1000, 5) for x in tau_mean],
        'dhw_dt_mNm': [round(float(x) * 1000, 5) for x in dhw],
        'ratio': [round(float(dhw[j] / tau_mean[j]), 4)
                  if abs(tau_mean[j]) > 1e-12 else None for j in range(3)],
    }
    # the same statistic on a 0.2 s subsample, to reproduce the committed trace
    idx = np.where(m)[0]
    step = max(1, int(round(0.2 / max(res['dt_median_s'], 1e-9))))
    sub = idx[::step]
    res['H1_subsampled'] = {
        'step': int(step),
        'mean_tau_w_mNm': [round(float(x) * 1000, 5) for x in tw[sub].mean(0)],
        'aliasing_shift_mNm': [
            round(float(tw[sub].mean(0)[j] - tau_mean[j]) * 1000, 5)
            for j in range(3)],
    }

    # H2: effective inertia. h_w as logged uses I_W; the joint carries
    # I_W + ARMATURE. Recompute the implied torque both ways.
    res['H2_inertia'] = {
        'I_logged': I_W, 'I_effective': I_W + ARMATURE,
        'dhw_dt_corrected_mNm': [
            round(float(x) * (I_W + ARMATURE) / I_W * 1000, 5) for x in dhw],
        'ratio_after_correction': [
            round(float(dhw[j] * (I_W + ARMATURE) / I_W / tau_mean[j]), 4)
            if abs(tau_mean[j]) > 1e-12 else None for j in range(3)],
    }

    # H3: damping steady state. omega_w = hw / I_W ; predicted tau = b*omega_w
    om_w = hw[m] / I_W
    tau_damp = DAMPING * om_w
    res['H3_damping'] = {
        'mean_omega_w_radps': [round(float(x), 5) for x in om_w.mean(0)],
        'mean_damping_torque_mNm': [round(float(x) * 1000, 5)
                                    for x in tau_damp.mean(0)],
        'damping_explains_frac_of_tau': [
            round(float(tau_damp.mean(0)[j] / tau_mean[j]), 4)
            if abs(tau_mean[j]) > 1e-12 else None for j in range(3)],
        'omega_w_implied_by_steady_state_radps': [
            round(float(tau_mean[j] / DAMPING), 3) for j in range(3)],
    }

    # torque balance: tau_motor - b*omega_w  should equal  I_eff * domega/dt
    lhs = tau_mean - tau_damp.mean(0)
    rhs = dhw * (I_W + ARMATURE) / I_W
    res['balance'] = {
        'tau_minus_damping_mNm': [round(float(x) * 1000, 5) for x in lhs],
        'I_eff_domega_dt_mNm': [round(float(x) * 1000, 5) for x in rhs],
        'residual_mNm': [round(float(lhs[j] - rhs[j]) * 1000, 5)
                         for j in range(3)],
        'residual_frac_of_tau': [
            round(float((lhs[j] - rhs[j]) / tau_mean[j]), 4)
            if abs(tau_mean[j]) > 1e-12 else None for j in range(3)],
    }

    # C2.3 decomposition, if present: which term carries the mean
    if 'aocs_tau_ff' in sl:
        d = {}
        for k in ('tau_ff', 'tau_att_p', 'tau_rate_d', 'tau_accel_d',
                  'tau_antiwindup', 'tau_w_preclip'):
            v = np.array(sl[f'aocs_{k}'], float)[m]
            d[k] = [round(float(x) * 1000, 5) for x in v.mean(0)]
        res['decomposition_mean_mNm'] = d
        res['clip_active'] = bool(
            np.abs(np.array(sl['aocs_tau_w_preclip'], float)[m]).max() > 2.5)

    with open(os.path.join(OUT, 'c3_3_settle900.json'), 'w') as fh:
        json.dump(res, fh, indent=2)

    ax = 'xyz'
    print('=' * 72)
    print(f"C3.3 §3 — settle900, {res['settle_ticks']} ticks over "
          f"{res['settle_window_s'][0]:.1f}..{res['settle_window_s'][1]:.1f} s "
          f"(dt {res['dt_median_s']:.3f} s)")
    print()
    h = res['H1_full_rate']
    print('H1  full-rate vs subsampled mean tau_w')
    for j in range(3):
        print(f"   {ax[j]}: mean tau_w {h['mean_tau_w_mNm'][j]:+9.5f}   "
              f"dhw/dt {h['dhw_dt_mNm'][j]:+9.5f}   ratio {h['ratio'][j]}")
    s = res['H1_subsampled']
    print(f"   subsampled (every {s['step']} ticks ~0.2 s): "
          f"{s['mean_tau_w_mNm']}")
    print(f"   aliasing shift vs full rate: {s['aliasing_shift_mNm']} mNm")
    print()
    h2 = res['H2_inertia']
    print(f"H2  inertia: logged I={h2['I_logged']}, effective "
          f"(with armature) {h2['I_effective']}")
    print(f"   dhw/dt corrected: {h2['dhw_dt_corrected_mNm']} mNm")
    print(f"   ratio after correction: {h2['ratio_after_correction']}")
    print()
    h3 = res['H3_damping']
    print(f"H3  damping b={DAMPING}: mean omega_w {h3['mean_omega_w_radps']} rad/s")
    print(f"   damping torque {h3['mean_damping_torque_mNm']} mNm")
    print(f"   fraction of tau_w explained: {h3['damping_explains_frac_of_tau']}")
    print(f"   omega_w that a sustained tau would need: "
          f"{h3['omega_w_implied_by_steady_state_radps']} rad/s")
    print()
    b = res['balance']
    print('    torque balance  tau_motor - b*omega_w  =?=  I_eff * domega/dt')
    print(f"   LHS {b['tau_minus_damping_mNm']}")
    print(f"   RHS {b['I_eff_domega_dt_mNm']}")
    print(f"   residual {b['residual_mNm']} mNm "
          f"({b['residual_frac_of_tau']} of tau)")
    if 'decomposition_mean_mNm' in res:
        print()
        print('    AOCS decomposition, settle means [mNm]:')
        for k, v in res['decomposition_mean_mNm'].items():
            print(f"      {k:16s} {v}")
    print()
    print(f"wrote {os.path.join(OUT, 'c3_3_settle900.json')}")


if __name__ == '__main__':
    main()
