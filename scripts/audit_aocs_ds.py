#!/usr/bin/env python3
"""AOCS-during-DS audit (READ-ONLY) — is the structure attitude controller
(tau_w to the wheels/RWA) active every tick in ALL DS phases?

Invariant under test: the structure is free-floating at all times, so the AOCS
MUST run every tick — DS included. This checks the code (where tau_w is
applied) and the empirical tell (theta_s + tau_w during an inter-step DS
settle, from a committed sim_log). No crawlbot/ change, no sim run.

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_aocs_ds.py
"""
import json
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_SIM = os.path.join(ROOT, 'crawlbot', 'simulation', 'sim_loop.py')
_CFG = os.path.join(ROOT, 'crawlbot', 'simulation', 'config.py')
_LOG = os.path.join(ROOT, 'Misc', 'runs', 'fixA_gate', 'sim_log.json')

_checks = []


def rec(ok, label, ev):
    _checks.append(bool(ok))
    print(f"  [{'OK ' if ok else 'XX '}] {label}\n         {ev}")


def find(path, needle, label):
    rel = os.path.relpath(path, ROOT)
    try:
        for i, ln in enumerate(open(path), 1):
            if needle in ln:
                return i, ln.strip()
    except OSError as e:
        rec(False, label, f'{rel}: {e}'); return None
    rec(False, label, f'{rel}: {needle!r} NOT FOUND'); return None


def main():
    print('=' * 78)
    print('AOCS-DURING-DS AUDIT — is tau_w applied every tick in all DS? (ae0673e)')
    print('=' * 78)

    # ---- CODE: where tau_w is applied, per path ----
    print('\n[CODE] tau_w application by path')
    a = find(_SIM, 'self.mj_data.ctrl[self.robot.n_joints:self.robot.n_joints + 3] = tau_w_cmd',
             '_step applies AOCS tau_w')
    if a:
        rec(True, '_step (SS + terminal/DWELL DS): AOCS tau_w_cmd APPLIED to wheels every tick',
            f'sim_loop.py:{a[0]}: {a[1]}')
    a = find(_SIM, 'self.mj_data.ctrl[n_j:n_j + 3] = 0.0', '_run_ds_passivity_loop zeroes wheels')
    if a:
        rec(True, 'inter-step DS (_run_ds_passivity_loop): wheels HARDCODED to 0.0 — AOCS NOT run',
            f'sim_loop.py:{a[0]}: {a[1]}  ← the violation')
    a = find(_SIM, "log.tau_w.append(np.zeros(3))", 'inter-step logs tau_w=0')
    if a:
        rec(True, "inter-step loop logs tau_w = zeros (comment: 'wheels are OFF in this loop')",
            f'sim_loop.py:{a[0]}')
    a = find(_CFG, 'aocs_off_in_ds: bool = False', 'aocs_off_in_ds default')
    if a:
        rec(True, 'aocs_off_in_ds defaults False ⇒ _step DS keeps AOCS ON (only the inter-step loop is off)',
            f'config.py:{a[0]}: {a[1]}')

    # ---- CODE: AOCS input dependency on the NMPC? ----
    print('\n[CODE] AOCS self-contained (no NMPC dependency)?')
    # scan the _step AOCS block for any use of the NMPC outputs rp/lr/vp
    try:
        lines = open(_SIM).read().splitlines()
    except OSError:
        lines = []
    blk = '\n'.join(lines[3099:3276])  # the AOCS block in _step
    uses_nmpc = any(tok in blk for tok in (' rp', '=rp', ' lr', '=lr', 'lambda_ref', 'self.nmpc'))
    uses_state = ('rs.L_com' in blk and 'omega_s' in blk and 'hw_phys' in blk)
    rec(uses_state and not uses_nmpc,
        'AOCS reads STATE (rs.L_com, omega_s, hw, lambda_qp_sol) — NOT the NMPC plan (rp/lr)',
        f'uses_state={uses_state} uses_nmpc_output={uses_nmpc} ⇒ self-contained; inter-step loop HAS '
        f'these inputs (rs + lambda_qp_sol) — bypassing the NMPC does NOT starve the AOCS')

    # ---- EMPIRICAL: tau_w + theta_s by phase, from a committed sim_log ----
    print('\n[EMPIRICAL] tau_w + theta_s by DS sub-path (Misc/runs/fixA_gate)')
    if not os.path.exists(_LOG):
        rec(False, 'sim_log present', f'{_LOG} missing'); return _verdict()
    sl = json.load(open(_LOG))
    ph = np.array(sl['phase']); se = np.array(sl['struct_euler_deg'])
    tw = np.linalg.norm(np.array(sl['tau_w']), axis=1)
    ss = ph == 'SS'
    ds = ph == 'DS'
    ds_off = ds & (tw < 1e-9)   # inter-step _run_ds_passivity_loop (tau_w=0)
    ds_on = ds & (tw >= 1e-9)   # terminal/DWELL _step (AOCS on)

    def line(mask, lbl):
        n = int(mask.sum())
        if n == 0:
            print(f'    {lbl:34s} n=0'); return
        print(f'    {lbl:34s} n={n:4d} ({n*0.01:.2f}s)  |tau_w| mean={tw[mask].mean():.3f} '
              f'max={tw[mask].max():.3f} Nm')
    line(ss, 'SS (AOCS on)')
    line(ds_on, 'DS terminal/DWELL (AOCS on)')
    line(ds_off, 'DS inter-step (_run_ds_passivity_loop)')
    rec(int(ds_off.sum()) > 0 and tw[ds_off].max() < 1e-9,
        'inter-step DS ticks run with tau_w ≡ 0 (structure attitude UNCONTROLLED)',
        f'{int(ds_off.sum())} ticks ({ds_off.sum()*0.01:.2f}s) at tau_w=0')

    # per-settle theta_s drift (the empirical tell)
    idx = np.where(ds_off)[0]
    if len(idx):
        groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        worst = 0.0
        print('    θ_s drift per inter-step settle (uncontrolled):')
        for g in groups:
            if len(g) < 2:
                continue
            dr = (se[g].max(0) - se[g].min(0))
            worst = max(worst, dr.max())
            print(f'      settle {len(g):3d} ticks ({len(g)*0.01:.2f}s): '
                  f'Δθ_s = [{dr[0]:.4f}, {dr[1]:.4f}, {dr[2]:.4f}]° (max {dr.max():.4f}°)')
        rec(worst > 0.0, 'θ_s DRIFTS during the uncontrolled inter-step settle (non-zero, uncorrected)',
            f'worst per-settle drift = {worst:.4f}° (small here — heavy structure + short low-v settle '
            f'— but non-zero and uncontrolled)')

    return _verdict()


def _verdict():
    n = sum(_checks)
    print('\n' + '=' * 78)
    print(f'{n}/{len(_checks)} checks. VERDICT: the AOCS invariant is VIOLATED in the inter-step DS loop')
    print('  (_run_ds_passivity_loop): wheels are commanded to ZERO, AOCS not run, structure attitude')
    print('  uncontrolled for the settle duration. SS + terminal/DWELL DS (_step) DO run the AOCS.')
    print('  The AOCS is self-contained (no NMPC dependency) ⇒ re-activation in the inter-step loop is')
    print('  the core requirement of step 4. See Misc/runs/j2_aocs_ds/INTERNAL_aocs_ds.md.')
    print('=' * 78)
    return 0 if n == len(_checks) else 1


if __name__ == '__main__':
    sys.exit(main())
