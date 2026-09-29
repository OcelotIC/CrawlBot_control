#!/usr/bin/env python3
"""M0 — the AOCS wheel command at 100 Hz, and the 900 s settle momentum gap.

PAPER_CORRECTIONS_MEMO item 7 (d37ad02, OPEN): on the instrumented 900 s
settle, the mean logged wheel torque integrates to +2.884 N·m·s while the
wheel momentum changes by only +0.084 N·m·s (z). Hypothesis (Cowork): the 10 Hz
log records the command of QP sub-step qs=9 only, while the qs=0 command carries
the AOCS restart kick K_d·ω_s/dt of opposite sign (attitude.md §3); the applied
average — all ten sub-steps — would then be near Δh_w.

Test, logging only (cfg.log_hifreq_all -> sim.hifreq_trace, never applied):
    --proof   the canonical run (settle 20 s) with the trace ON, every output
              compared bit-for-bit to the host-local reference
              (gate/_run/local_ref/ref) — the trace must be inert
    --settle  the canonical kwargs with settle_seconds=900 (as c3_3 did);
              over the trailing DS hold, per axis:
                ∫τ_w dt at 100 Hz        (what the wheels receive)
                ∫ of the qs=9 subsample  (what the 10 Hz log integrates)
                Δh_w                     (from the trace, and from sim_log)

    MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_m0_aocs_hifreq.py --proof
    MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_m0_aocs_hifreq.py --settle
    PYTHONPATH=. python3 scripts/diag_m0_aocs_hifreq.py --analyse   # from the trace
Writes results/m0_aocs_hifreq/{proof,settle900}/ and m0_summary.json.
"""
import json
import os
import sys

os.environ.setdefault('MUJOCO_GL', 'disabled')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

import crawlbot.solvers.hierarchical_qp as hq
_ow = hq.HierarchicalQP._solve_weighted


def _pw(self, tasks, x0):
    self.regularization = 1e-6          # == gate/replay_canonical.py
    return _ow(self, tasks, x0)


hq.HierarchicalQP._solve_weighted = _pw

import crawlbot.simulation.sim_loop as sl
import scripts.diag_cooperative_arms as dca
from scripts.diag_cooperative_arms import _mutate_mjcf, _mjcf_md5, MJCF

C_KWARGS = dict(                         # verbatim gate/replay_canonical.py
    legacy=False, alpha_torso_lin=0.0, anchor_dx=0.8, mass_ratio=0.01,
    aocs_mode='legacy_pid_numerical', settle_seconds=20.0,
    K_theta=1.0, K_omega=50.0, tau_w_max=2.5,
    n_steps=6, ss_two_task=True, ss_alpha_mom=400.0,
    alpha_torso_pose=2000.0, ss_alpha_ee=1000.0, ss_alpha_posture=2e1,
    ss_alpha_wrench=1.0, ss_kp_torso=3.0, ss_kd_torso=2.5,
    qp_envelope_exact=True,
    interstep_settle_alpha_wrench=3.0, interstep_settle_epsilon_v=5e-3,
)
OUT = 'results/m0_aocs_hifreq'
SIMS = []

_init = sl.SimulationLoop.__init__


def _init_trace(self, *a, **k):
    _init(self, *a, **k)
    self.cfg.log_hifreq_all = True
    SIMS.append(self)


sl.SimulationLoop.__init__ = _init_trace


def run(tag, settle):
    out_rel = f'm0_aocs_hifreq/{tag}'
    log = f'results/{out_rel}/sim_log.json'
    if os.path.exists(log):
        os.remove(log)
    with open(MJCF) as f:
        orig = f.read()
    pre = _mjcf_md5(MJCF)
    try:
        _mutate_mjcf(damping=0.0, armature=0.05, anchor_dx=0.8, mass_ratio=0.01)
        try:
            dca.main(**dict(C_KWARGS, settle_seconds=settle,
                            out_dir_override=out_rel))
        except Exception as e:
            print(f'[m0] main() raised: {type(e).__name__}: {e}', flush=True)
    finally:
        with open(MJCF, 'w') as f:
            f.write(orig)
        assert _mjcf_md5(MJCF) == pre
    trace = SIMS[-1].hifreq_trace
    json.dump(trace, open(f'results/{out_rel}/hifreq_trace.json', 'w'))
    return f'results/{out_rel}', trace


def proof():
    """Trace ON vs the host-local reference (trace OFF): every file identical."""
    d, trace = run('proof', 20.0)
    from gate.local_ref import _walk
    ref = 'gate/_run/local_ref/ref'
    bad = 0
    for fn in sorted(os.listdir(d)):
        if fn == 'hifreq_trace.json' or not fn.endswith('.json'):
            continue
        rp = os.path.join(ref, fn)
        if not os.path.exists(rp):
            continue
        st = {'floats': 0, 'other': 0, 'fields': 0, 'excluded': set()}
        r = _walk(json.load(open(rp)), json.load(open(os.path.join(d, fn))), fn, st)
        print(f'  {"OK  " if r is None else "DIFF"} {fn}'
              + ('' if r is None else f' -> {r}') + f'  ({st["floats"]} floats)')
        bad |= r is not None
    print(f'[m0] trace records: {len(trace)}')
    print('PROOF: trace inert' if not bad else 'PROOF FAILS')
    return not bad


def settle():
    d, trace = run('settle900', 900.0)
    tr = [r for r in trace if r['src'] == 'nmpc' and r['phase'] == 'DS']
    t = np.array([r['t'] for r in tr])
    qs = np.array([r['qs'] for r in tr])
    tw = np.array([r['tau_w'] for r in tr])
    twp = np.array([r['tau_w_preclip'] for r in tr])
    hw = np.array([r['hw'] for r in tr])
    om = np.array([r['omega_s'] for r in tr])
    dt = 0.01
    # trailing hold = the last contiguous block of NMPC 'DS' sub-steps
    brk = np.where(np.diff(t) > 1.5 * dt)[0]
    s0 = brk[-1] + 1 if brk.size else 0
    t, qs, tw, twp, hw, om = t[s0:], qs[s0:], tw[s0:], twp[s0:], hw[s0:], om[s0:]
    # h_w after the first step of the window minus h_w before it: the trace
    # records h_w AFTER each step, so the window's momentum change is
    # hw[-1] - (hw[0] - τ_0·dt) — use the integral identity check both ways.
    dur = t[-1] - t[0] + dt
    int100 = tw.sum(0) * dt                           # all sub-steps
    m9, m0 = qs == 9, qs == 0
    int_q9 = tw[m9].mean(0) * dur                     # what the 10 Hz log sees
    int_q0 = tw[m0].mean(0) * dur
    dhw = hw[-1] - hw[0]
    slope = np.array([np.polyfit(t, hw[:, j], 1)[0] for j in range(3)])
    sl_log = json.load(open(os.path.join(d, 'sim_log.json')))
    ph = np.array(sl_log['phase'])
    tl = np.array(sl_log['t'])
    hwl = np.array(sl_log['hw_physical'], float)
    twl = np.array(sl_log['tau_w'], float)
    mlog = (ph == 'DS') & (tl >= t[0] - 1e-9)
    res = {
        'window_s': [float(t[0]), float(t[-1] + dt)], 'duration_s': float(dur),
        'n_substeps': int(len(t)),
        'integral_tau_w_100Hz_Nms': int100.tolist(),
        'integral_tau_w_qs9_subsample_Nms': int_q9.tolist(),
        'integral_tau_w_qs0_subsample_Nms': int_q0.tolist(),
        'delta_h_w_trace_Nms': dhw.tolist(),
        'dh_w_dt_slope_trace_Nm': slope.tolist(),
        'mean_tau_w_100Hz_Nm': tw.mean(0).tolist(),
        'mean_tau_w_qs9_Nm': tw[m9].mean(0).tolist(),
        'mean_tau_w_qs0_Nm': tw[m0].mean(0).tolist(),
        'max_abs_preclip_Nm': float(np.abs(twp).max()),
        'n_clipped_substeps': int((np.abs(twp) > 2.5 + 1e-12).any(1).sum()),
        'sim_log_10Hz': {
            'n_ticks': int(mlog.sum()),
            'mean_tau_w_logged_Nm': twl[mlog].mean(0).tolist(),
            'integral_logged_Nms': (twl[mlog].mean(0) * dur).tolist(),
            'delta_hw_physical_Nms': (hwl[mlog][-1] - hwl[mlog][0]).tolist(),
        },
        'omega_s_abs_mean_mrad_s': (np.abs(om).mean(0) * 1e3).tolist(),
    }
    json.dump(res, open(f'{OUT}/m0_summary.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))


def analyse():
    """From results/m0_aocs_hifreq/settle900/hifreq_trace.json (no re-run):
    per-step regression Δh_w = k·τ_w·dt over every pair of consecutive plant
    steps, and the decomposition of the memo's gap on the trailing hold."""
    tr = json.load(open(f'{OUT}/settle900/hifreq_trace.json'))
    t = np.array([r['t'] for r in tr])
    tau = np.array([r['tau_w'] for r in tr])
    hw = np.array([r['hw'] for r in tr])
    dt = 0.01
    ok = np.abs(np.diff(t) - dt) < 1e-9
    dh, tdt = np.diff(hw, axis=0)[ok], (tau[1:] * dt)[ok]
    reg = {a: {'k': float(np.polyfit(tdt[:, j], dh[:, j], 1)[0]),
               'corr': float(np.corrcoef(tdt[:, j], dh[:, j])[0, 1])}
           for j, a in enumerate('xyz')}
    s = json.load(open(f'{OUT}/m0_summary.json'))
    z = 2
    dur = s['duration_s']
    gap = {
        'logged_qs9_integral_z': s['integral_tau_w_qs9_subsample_Nms'][z],
        'applied_100Hz_integral_z': s['integral_tau_w_100Hz_Nms'][z],
        'delta_hw_channel_endpoints_z': s['delta_h_w_trace_Nms'][z],
        'delta_hw_channel_slope_times_T_z':
            s['dh_w_dt_slope_trace_Nm'][z] * dur,
        'wheel_momentum_true_z (channel / k)':
            s['delta_h_w_trace_Nms'][z] / reg['z']['k'],
    }
    gap['factor_sampling_qs9_vs_100Hz'] = (gap['logged_qs9_integral_z']
                                           / gap['applied_100Hz_integral_z'])
    gap['factor_inertia_channel_vs_true'] = 1.0 / reg['z']['k']
    gap['factor_slope_vs_endpoint_metric'] = (
        gap['delta_hw_channel_endpoints_z']
        / gap['delta_hw_channel_slope_times_T_z'])
    gap['residual_applied_vs_true'] = (gap['applied_100Hz_integral_z']
                                       / gap['wheel_momentum_true_z (channel / k)'])
    out = {'per_step_regression_dhw_on_tau_dt': reg, 'gap_decomposition_z': gap}
    json.dump(out, open(f'{OUT}/m0_analysis.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    if '--proof' in sys.argv:
        sys.exit(0 if proof() else 1)
    if '--settle' in sys.argv:
        settle()
    if '--analyse' in sys.argv:
        analyse()
