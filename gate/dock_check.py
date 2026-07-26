#!/usr/bin/env python3
"""Report the headline canonical numbers from a replay ``sim_log.json``.

The gate proves *byte-identity* of the exported CSV; this reports the physical
result the paper quotes, so a replay can be read as "still docks 6/6" without
trusting a hash. Compares against the FROZEN 2.5 CANONICAL table in CLAUDE.md.

Rule 10: dock precision is the at-weld ``dock_events`` d_mm ONLY — never
min-over-swing.

    MUJOCO_GL=disabled PYTHONPATH=. python3 gate/dock_check.py [log.json]

Exit 0 = matches frozen, 1 = divergence.
"""
import datetime
import json
import os
import sys

import numpy as np

LOG = sys.argv[1] if len(sys.argv) > 1 else 'results/gate_run_scratch/sim_log.json'
VERDICT = 'gate/last_verdict.json'
VERDICT_TIMED = 'gate/_run/last_verdict_timed.json'

GATE_MM = 5.0                                    # docking-mechanism capture radius
FROZEN_DOCKS = [4.02, 4.89, 4.99, 4.97, 4.95, 4.62]        # mm, at-weld
DOCK_TOL = 5e-3                                  # mm; artifacts are quoted to 0.01

log = json.load(open(LOG))
ok = True

# ── is this log from the run you think it is? ─────────────────────────────
# CLEANUP-33: a crashed replay left the PREVIOUS run's log in place and every
# number below matched frozen — from a 33-minute-old artifact. The row-count
# check at line ~36 catches a TRUNCATED log; nothing caught a stale COMPLETE
# one. Report the age always, and refuse a log that predates the gate verdict
# it is supposed to accompany.
_mtime = os.path.getmtime(LOG)
_age = datetime.datetime.now() - datetime.datetime.fromtimestamp(_mtime)
print(f'log: {LOG}   ticks={len(log.get("t", []))}')
print(f'     written {datetime.datetime.fromtimestamp(_mtime):%H:%M:%S} '
      f'({_age.total_seconds() / 60:.1f} min ago)')
if os.path.exists(VERDICT):
    # The verdict is written at the END of a gate run and the log during it, so
    # the log is ALWAYS a little older — the question is how much. Anything
    # beyond one gate run means the log came from an earlier one. `seconds_total`
    # lives in the gitignored timed copy (the tracked verdict omits timings to
    # stay byte-stable), so fall back to a generous 10 min when it is absent.
    _v = os.path.getmtime(VERDICT)
    try:
        _budget = json.load(open(VERDICT_TIMED))['seconds_total'] + 120
    except Exception:
        _budget = 600
    _gap = _v - _mtime
    if _gap > _budget:
        print(f'\n*** STALE LOG *** — written {_gap / 60:.1f} min before '
              f'{VERDICT}, which is longer than one gate run '
              f'({_budget / 60:.1f} min). This log came from an EARLIER run, so '
              f'the numbers below would not describe the current code. '
              f'Re-run gate/run_gate.py.')
        sys.exit(1)

# ── at-weld docks ─────────────────────────────────────────────────────────
docks = [float(e['d_mm']) for e in log.get('dock_events', [])]
breach = [i for i, d in enumerate(docks) if d >= GATE_MM]
print(f'\nat-weld docks  {len(docks)}/{len(FROZEN_DOCKS)}'
      + ('  ALL UNDER 5 mm' if docks and not breach else '  *** GATE BREACH ***'))
ok &= len(docks) == len(FROZEN_DOCKS) and not breach
for i, d in enumerate(docks):
    ref = FROZEN_DOCKS[i] if i < len(FROZEN_DOCKS) else float('nan')
    ok &= abs(d - ref) < DOCK_TOL
    print(f'   step {i+1}:  {d:6.2f} mm   (frozen {ref:5.2f}, '
          f'delta {d - ref:+.4f})   margin {GATE_MM - d:5.2f} mm')
if docks:
    print(f'   worst margin: {GATE_MM - max(docks):.2f} mm')


# ── headline scalars ──────────────────────────────────────────────────────
def arr(k):
    return np.asarray(log[k], float) if k in log and log[k] else None


hw = arr('hw')                       # (n,3) wheel momentum [Nms]
euler = arr('struct_euler_deg')      # (n,3) structure attitude [deg]
ecom = arr('e_com')

meas = {}
if euler is not None:                # theta_s := total attitude angle
    meas['theta_s peak [deg]'] = (np.max(np.linalg.norm(euler, axis=1)), 0.540, 5e-3)
if hw is not None:
    meas['h_w peak axis [Nms]'] = (np.max(np.abs(hw)), 4.10, 5e-3)
    meas['h_w peak norm [Nms]'] = (np.max(np.linalg.norm(hw, axis=1)), 4.24, 5e-3)
if ecom is not None:
    meas['e_com peak [m]'] = (np.max(np.abs(ecom)), 0.154, 5e-4)

print()
for k, (v, ref, tol) in meas.items():
    good = abs(v - ref) <= tol
    ok &= good
    print(f'{k:24s} {v:8.3f}   frozen {ref:7.3f}   {"OK" if good else "<-- DIFF"}')

qp_ok = arr('qp_ok')
if qp_ok is not None:
    n = int((~qp_ok.astype(bool)).sum())
    ok &= n == 0
    print(f'{"qp_fail":24s} {n:8d}   frozen {0:7d}   {"OK" if n == 0 else "<-- DIFF"}')

print('\nCANONICAL RESULTS: ' + ('MATCH frozen 2.5' if ok else '*** DIVERGENCE ***'))
sys.exit(0 if ok else 1)
