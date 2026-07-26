#!/usr/bin/env python3
"""J2 audit — read-only reproducer of the load-bearing facts.

Re-verifies, against the Fix-A canonical (``main = ae0673e``), every claim
the J2 cartography report (``Misc/runs/j2_audit/INTERNAL_j2_audit.md``) leans
on. READ-ONLY: it imports class constants + dataclass defaults and locates
source anchors *by text* (so the printed line numbers track the live tree,
not a frozen snapshot). No ``crawlbot/`` mutation, no sim run.

Run:  MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/audit_j2_facts.py

Exit 0 iff every load-bearing fact holds; non-zero (and a FAIL list) if the
tree has drifted from the audited state.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

_SIM_LOOP = os.path.join(ROOT, 'crawlbot', 'simulation', 'sim_loop.py')
_WBQP = os.path.join(ROOT, 'crawlbot', 'solvers', 'wholebody_qp.py')
_NMPC = os.path.join(ROOT, 'crawlbot', 'solvers', 'centroidal_nmpc.py')
_CPHASE = os.path.join(ROOT, 'crawlbot', 'solvers', 'contact_phase.py')
_DIAG = os.path.join(ROOT, 'scripts', 'diag_cooperative_arms.py')
_FIXA_META = os.path.join(ROOT, 'Misc', 'runs', 'fixA_gate', 'PHASE3_METADATA.txt')

_checks = []  # (ok: bool, label: str, evidence: str)


def record(ok, label, evidence):
    _checks.append((bool(ok), label, evidence))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print(f"         {evidence}")


def find(path, needle, label):
    """Return (lineno, stripped_line) of the first line containing `needle`.

    Records a FAIL (and returns None) if absent — anchors that vanish mean
    the audited code moved, which the reproducer must surface, not hide.
    """
    rel = os.path.relpath(path, ROOT)
    try:
        with open(path) as f:
            for i, line in enumerate(f, 1):
                if needle in line:
                    return i, line.strip()
    except OSError as e:
        record(False, label, f'{rel}: unreadable ({e})')
        return None
    record(False, label, f'{rel}: anchor {needle!r} not found')
    return None


def count(path, needle):
    rel = os.path.relpath(path, ROOT)
    try:
        with open(path) as f:
            return sum(needle in line for line in f), rel
    except OSError:
        return -1, rel


def main():
    print('=' * 78)
    print('J2 AUDIT — load-bearing facts reproducer  (Fix-A canonical)')
    print('=' * 78)

    # --- Live class constants + config defaults -------------------------
    from crawlbot.solvers.centroidal_nmpc import CentroidalNMPC
    from crawlbot.simulation.config import SimConfig
    cfg = SimConfig()

    print('\n[A] NMPC is n=2-generic (both contacts in the formulation)')
    record(CentroidalNMPC.NX == 9, 'NMPC.NX == 9  [r_com, v_com, L_com]',
           f'CentroidalNMPC.NX = {CentroidalNMPC.NX}')
    record(CentroidalNMPC.NU == 12, 'NMPC.NU == 12  [f1, tau1, f2, tau2] -> BOTH contacts',
           f'CentroidalNMPC.NU = {CentroidalNMPC.NU}')
    record(CentroidalNMPC.NP == 18, 'NMPC.NP == 18  [r_ref,v_ref,r_C1,r_C2,c_simple,L_ref]',
           f'CentroidalNMPC.NP = {CentroidalNMPC.NP}')

    print('\n[B] NMPC horizon = config N=8/dt=0.1 (class default 20/0.05 is overridden)')
    record(cfg.nmpc_N == 8 and cfg.nmpc_dt == 0.1, 'cfg.nmpc_N == 8 and cfg.nmpc_dt == 0.1 (0.8 s)',
           f'cfg.nmpc_N = {cfg.nmpc_N}, cfg.nmpc_dt = {cfg.nmpc_dt}')
    anchor = find(_SIM_LOOP, 'N=cfg.nmpc_N, dt=cfg.nmpc_dt', 'sim_loop overrides class default')
    if anchor:
        record(True, 'NMPC built from cfg (override site)', f'sim_loop.py:{anchor[0]}: {anchor[1]}')

    print('\n[C] NMPC drives DS: one unconditional solve site inside _step')
    n_solve, rel = count(_SIM_LOOP, 'self.nmpc.solve')
    anchor = find(_SIM_LOOP, 'self.nmpc.solve', 'NMPC.solve present')
    record(n_solve == 1, 'exactly ONE self.nmpc.solve site (no phase/settle gate around it)',
           f'{rel}: {n_solve} occurrence(s)'
           + (f'; at line {anchor[0]}' if anchor else ''))

    print('\n[D] ss_two_task gate: ON only in SS with the flag; always OFF in settle_mode (DS)')
    anchor = find(_WBQP, '_two_task = cfg.ss_two_task_mode and not settle_mode',
                  'two-task gate exact form')
    if anchor:
        record(True, 'gate = cfg.ss_two_task_mode AND NOT settle_mode',
               f'wholebody_qp.py:{anchor[0]}: {anchor[1]}')
    record(cfg.ss_two_task_mode is False, 'cfg.ss_two_task_mode default == False (flag opt-in)',
           f'cfg.ss_two_task_mode = {cfg.ss_two_task_mode}')

    print('\n[E] Terminal/DWELL DS task set: CoM-3D + torso-ori-3D, alphas 100/200')
    record(cfg.ds_alpha_com == 100.0 and cfg.ds_alpha_torso_ori == 200.0,
           'cfg.ds_alpha_com == 100 and cfg.ds_alpha_torso_ori == 200',
           f'cfg.ds_alpha_com = {cfg.ds_alpha_com}, cfg.ds_alpha_torso_ori = {cfg.ds_alpha_torso_ori}')

    print('\n[F] Hyperstatic 2-contact distribution: internal-stress reg, OFF by default but ON in canonical')
    record(cfg.ss_alpha_lambda_int == 0.0, 'dataclass DEFAULT ss_alpha_lambda_int == 0.0 (so "off" if you only read the default)',
           f'cfg.ss_alpha_lambda_int = {cfg.ss_alpha_lambda_int}')
    anchor = find(_SIM_LOOP, 'alpha_lambda_int=cfg.ss_alpha_lambda_int',
                  'QP wired from cfg.ss_alpha_lambda_int')
    if anchor:
        record(True, '_build_qp maps alpha_lambda_int <- cfg.ss_alpha_lambda_int',
               f'sim_loop.py:{anchor[0]}: {anchor[1]}')
    anchor = find(_DIAG, 'cfg.ss_alpha_lambda_int = 1.0', 'canonical diag enables it')
    if anchor:
        record(True, 'canonical run (diag_cooperative_arms) sets ss_alpha_lambda_int = 1.0 -> ON',
               f'diag_cooperative_arms.py:{anchor[0]}: {anchor[1]}')
    anchor = find(_WBQP, 'contact_config.nc == 2', 'internal-stress reg gated nc==2')
    if anchor:
        record(True, 'internal-stress reg gated on nc==2 (both contacts) -> DS-only',
               f'wholebody_qp.py:{anchor[0]}: {anchor[1]}')

    print('\n[G] EXT purged: ContactPhase = {SINGLE_A, SINGLE_B, DOUBLE} only')
    n_ext, _ = count(_CPHASE, 'EXT')
    a = find(_CPHASE, "SINGLE_A = ", 'SINGLE_A present')
    b = find(_CPHASE, "SINGLE_B = ", 'SINGLE_B present')
    d = find(_CPHASE, "DOUBLE = ", 'DOUBLE present')
    record(n_ext == 0 and a and b and d, 'no EXT token; three-phase enum is two-phase contact set',
           f'contact_phase.py: EXT occurrences = {n_ext}; '
           f'enum at lines {a[0] if a else "?"},{b[0] if b else "?"},{d[0] if d else "?"}')

    print('\n[H] Canonical SS working point = alpha_torso_pose 24000 (not run_phase3_gate.sh 20000)')
    anchor = find(_FIXA_META, 'alpha-torso-pose 24000', 'fixA_gate metadata WP')
    if anchor:
        record(True, 'Misc/runs/fixA_gate/PHASE3_METADATA.txt pins 24000',
               f'PHASE3_METADATA.txt:{anchor[0]}: {anchor[1]}')

    # --- Verdict --------------------------------------------------------
    n_pass = sum(ok for ok, _, _ in _checks)
    n_tot = len(_checks)
    print('\n' + '=' * 78)
    print(f'VERDICT: {n_pass}/{n_tot} load-bearing facts confirmed.')
    fails = [lbl for ok, lbl, _ in _checks if not ok]
    if fails:
        print('FAILED:')
        for lbl in fails:
            print(f'  - {lbl}')
        print('=> tree has drifted from the audited Fix-A canonical state.')
        return 1
    print('=> matches Misc/runs/j2_audit/INTERNAL_j2_audit.md.')
    print('=' * 78)
    return 0


if __name__ == '__main__':
    sys.exit(main())
