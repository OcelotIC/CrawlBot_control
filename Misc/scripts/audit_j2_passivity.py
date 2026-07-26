#!/usr/bin/env python3
"""J2 passivity audit — CONFIRM/REFUTE reproducer (read-only, ae0673e).

Re-verifies the three passivity claims the reviewing Claude drew from a
pre-Fix-A clone (5ada364), against the real Fix-A canonical (main = ae0673e),
plus the load-bearing PREMISE that Fix A did not touch the passivity/QP code.

READ-ONLY: live-builds the canonical config exactly as the diag does, locates
each source anchor *by text* (line numbers track the live tree), and shells
out to git only to diff 5ada364..ae0673e. No crawlbot/ mutation, no sim run.

Run:  MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/audit_j2_passivity.py

Exit 0 iff every claim's code evidence holds AND the premise (passivity/QP
code byte-identical 5ada364->ae0673e) holds.
"""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_SIM = os.path.join(ROOT, 'crawlbot', 'simulation', 'sim_loop.py')
_QP = os.path.join(ROOT, 'crawlbot', 'solvers', 'wholebody_qp.py')

_checks = []


def record(ok, label, evidence):
    _checks.append((bool(ok), label, evidence))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print(f"         {evidence}")


def find(path, needle, label):
    rel = os.path.relpath(path, ROOT)
    try:
        with open(path) as f:
            for i, line in enumerate(f, 1):
                if needle in line:
                    return i, line.strip()
    except OSError as e:
        record(False, label, f'{rel}: unreadable ({e})')
        return None
    record(False, label, f'{rel}: anchor {needle!r} NOT FOUND')
    return None


def git(*args):
    return subprocess.run(['git', '-C', ROOT, *args],
                          capture_output=True, text=True)


def main():
    print('=' * 78)
    print('J2 PASSIVITY AUDIT — CONFIRM/REFUTE reproducer  (Fix-A canonical ae0673e)')
    print('=' * 78)

    # --- PREMISE: Fix A did not touch the passivity / QP code -----------
    print('\n[PREMISE] passivity/QP code byte-identical 5ada364 -> ae0673e')
    r = git('rev-parse', '--verify', '-q', '5ada364^{commit}')
    have_commits = (r.returncode == 0)
    if not have_commits:
        record(False, 'commit 5ada364 reachable (needed for premise diff)',
               'git rev-parse 5ada364 failed — run on the j2 branch with full history')
    else:
        r = git('diff', '--quiet', '5ada364', 'ae0673e', '--',
                'crawlbot/solvers/wholebody_qp.py')
        record(r.returncode == 0, 'wholebody_qp.py IDENTICAL 5ada364..ae0673e',
               'git diff --quiet -> ' + ('no diff' if r.returncode == 0 else 'DIFFERS'))
        r = git('diff', '5ada364', 'ae0673e', '--', 'crawlbot/simulation/sim_loop.py')
        n_pass_lines = sum('passivity' in ln.lower()
                           for ln in r.stdout.splitlines()
                           if ln[:1] in '+-' and ln[:3] not in ('+++', '---'))
        record(n_pass_lines == 0,
               'sim_loop.py Fix-A diff touches ZERO passivity lines (only the impact block moved)',
               f'passivity +/- lines in 5ada364..ae0673e sim_loop diff = {n_pass_lines}')

    # --- Canonical config (built the way the diag does) -----------------
    print('\n[CONFIG] canonical flags (via _make_m7_config + diag overrides)')
    import scripts.run_m7_single_step as r_single
    cfg = r_single._make_m7_config()
    cfg.ss_two_task_mode = True       # --ss-two-task
    cfg.ss_alpha_lambda_int = 1.0     # diag:312
    cfg.ds_centroidal_mode = True     # diag:317
    record(cfg.use_m2_stack is True, 'cfg.use_m2_stack == True (so base passivity expr can fire in DS)',
           f'use_m2_stack = {cfg.use_m2_stack}  (SimConfig default is False; _make_m7_config sets True)')
    record(cfg.ds_centroidal_mode is True, 'cfg.ds_centroidal_mode == True (centroidal-DS + force passivity ON)',
           f'ds_centroidal_mode = {cfg.ds_centroidal_mode}')
    record(cfg.alpha_passivity > 0, 'cfg.alpha_passivity > 0 (passivity inequality enabled)',
           f'alpha_passivity = {cfg.alpha_passivity}')

    # --- CLAIM 1: passivity + centroidal tracking coexist in terminal/DWELL DS
    print('\n[CLAIM 1] passivity_active expr + override; tracking & passivity in ONE QP')
    a = find(_SIM, 'cfg.use_m2_stack and (phase ==', 'base passivity_active expression')
    if a:
        record(True, "base expr = use_m2_stack and (phase=='DS' or passivity_hold) [matches the 5ada364 read]",
               f'sim_loop.py:{a[0]}: {a[1]}')
    a = find(_SIM, 'if passivity_override is not None:', 'override path present')
    if a:
        record(True, 'override path: passivity_override (when not None) SUPERSEDES the base expr',
               f'sim_loop.py:{a[0]}: {a[1]}')
    # terminal/DWELL DS pass passivity_override=True
    a = find(_SIM, 'passivity_override=True', 'DWELL forces passivity_override=True')
    if a:
        record(True, 'DWELL _step(...) passes passivity_override=True (+ ds_centroidal_active=True)',
               f'sim_loop.py:{a[0]}: {a[1]}')
    a = find(_SIM, '_pass_override = True', 'trailing-DS forces _pass_override=True under ds_centroidal_mode')
    if a:
        record(True, 'trailing-DS: ds_centroidal_mode -> _pass_override=True (forced ON regardless of abort)',
               f'sim_loop.py:{a[0]}: {a[1]}')
    # QP side: both tasks added concurrently
    a = find(_QP, 'if settle_mode and cfg.ds_centroidal_mode and ds_centroidal_active:',
             'centroidal-DS tracking-task gate')
    if a:
        record(True, 'QP adds centroidal-DS tracking (CoM-3D + torso-ori-3D) under settle+ds_centroidal+active',
               f'wholebody_qp.py:{a[0]}: {a[1]}')
    a = find(_QP, 'if passivity_active and cfg.alpha_passivity > 0:', 'passivity-inequality gate')
    if a:
        record(True, 'QP adds passivity inequality concurrently (same solve) when passivity_active',
               f'wholebody_qp.py:{a[0]}: {a[1]}')

    # --- CLAIM 2: passivity = QP inequality; loop is only the driver ----
    print('\n[CLAIM 2] passivity inequality form + _run_ds_passivity_loop is the driver')
    a = find(_QP, 'A_pass[0, idx[\'tau\'][0]: idx[\'tau\'][1]] = dq', "A_pass[tau] = dq")
    if a:
        record(True, "A_pass row on the tau block = dq  (=> dq^T*tau_q term)",
               f'wholebody_qp.py:{a[0]}: {a[1]}')
    a = find(_QP, 'b_pass = np.array([-2.0 * cfg.alpha_passivity * T_kin])', 'b_pass = -2 alpha T_kin')
    if a:
        record(True, 'b_pass = -2*alpha*T_kin  (=> dq^T*tau_q + 2*alpha*T_kin <= 0, T<=T0*exp(-2at))',
               f'wholebody_qp.py:{a[0]}: {a[1]}')
    a = find(_QP, 'H_jj = H_robot[6:, 6:]', 'T_kin uses the joint block of H')
    if a:
        record(True, 'T_kin = 0.5*dq^T*H_jj*dq with H_jj = H_robot[6:,6:] (joint block; base DOFs excluded)',
               f'wholebody_qp.py:{a[0]}: {a[1]}')
    a = find(_QP, 'qp.add_inequality_constraint(A_pass, b_pass)', 'passivity added as an inequality')
    if a:
        record(True, 'passivity enters as a QP *inequality* constraint (not a cost / not a separate law)',
               f'wholebody_qp.py:{a[0]}: {a[1]}')
    a = find(_SIM, 'passivity_active=True)', '_run_ds_passivity_loop calls qp.solve(passivity_active=True)')
    if a:
        record(True, '_run_ds_passivity_loop drives qp_ss.solve(..., settle_mode=True, passivity_active=True)',
               f'sim_loop.py:{a[0]}: {a[1]}')
    a = find(_SIM, "exit_reason = 'target_met'", 'loop exit on T<T_settle')
    if a:
        record(True, 'loop = iteration driver: repeats until T_kin < T_settle (exit target_met)',
               f'sim_loop.py:{a[0]}: {a[1]}')

    # --- CLAIM 3: positive-work conflict ALREADY observed (SS-hold) -----
    print('\n[CLAIM 3] positive-work conflict observed -> passivity disabled in SS hold window')
    a = find(_SIM, 'doing the positive work required to close the last', 'positive-work comment')
    if a:
        record(True, 'comment records: passivity_hold=True BLOCKED the arm doing positive work (last few mm)',
               f'sim_loop.py:{a[0]}: {a[1]}')
    a = find(_SIM, '(passivity_active=False) is the correct regime', 'SS-hold regime comment')
    if a:
        record(True, 'comment: "Normal SS tracking (passivity_active=False) is the correct regime for the hold"',
               f'sim_loop.py:{a[0]}: {a[1]}')
    a = find(_SIM, 'passivity_hold=False)', 'SS-hold loop disables passivity')
    if a:
        record(True, 'SS convergence-hold loop calls _step(..., passivity_hold=False) -> passivity OFF',
               f'sim_loop.py:{a[0]}: {a[1]}')

    # --- Verdict --------------------------------------------------------
    n_pass = sum(ok for ok, _, _ in _checks)
    n_tot = len(_checks)
    print('\n' + '=' * 78)
    print(f'VERDICT: {n_pass}/{n_tot} checks confirmed.')
    fails = [lbl for ok, lbl, _ in _checks if not ok]
    if fails:
        print('FAILED:')
        for lbl in fails:
            print(f'  - {lbl}')
        return 1
    print('=> all three claims CONFIRMED on ae0673e; premise (passivity code unchanged) holds.')
    print('=> see Misc/runs/j2_audit/INTERNAL_j2_passivity.md for CONFIRM/REFUTE/PARTIAL + nuances.')
    print('=' * 78)
    return 0


if __name__ == '__main__':
    sys.exit(main())
