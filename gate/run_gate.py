#!/usr/bin/env python3
"""CLEANUP chantier — reproduction & consistency GATE.

One command, four checks, one verdict. Run from anywhere:

    PYTHONPATH=. python3 gate/run_gate.py

Checks
------
1. Canonical replay — re-run the managed (C) scenario from the frozen config
   into a scratch dir (gate/replay_canonical.py), then export the 66-col
   fulldiag CSV from it (scripts/diag_full_diag_export.py). No committed
   artifact is touched.
2. Artifact identity — field-by-field vs the committed baseline
   results/j2_adjconv/c25_fulldiag.csv (the founding base, commit bfd5509 on main).
   Every column is byte-compared EXCEPT the two wall-clock timing columns
   (qp_time_ms, nmpc_time_ms), which are measured per-tick during the run and
   are non-reproducible by construction — see gate/EXCEPTIONS.md. Byte-identical
   on the reproducible columns = PASS; first mismatching field/row otherwise.
3. Two-model consistency — load the MJCF plant and the URDF controller model;
   diff the BIN A-a2 hand-duplicated quantities (per-link composite mass / COM /
   principal inertia, total mass, joint order + limits, tool/torso frame
   placements) and the BIN B-b1 naming contract. Any disagreement FAILs with
   the quantity named. (PORT_SYNTHESIS roadmap item 1.)
4. Environment pin — compare the live interpreter/library versions against
   gate/environment.lock. Mismatch WARNs today (bit-identity is meaningless on
   an unpinned stack); a later revision may promote this to FAIL. Creates the
   lock from the live env on first run.

Overall verdict = PASS iff checks 1+2+3 PASS. Check 4 is advisory (WARN-only).
Writes gate/last_verdict.json. Exit 0 on PASS, 1 on FAIL.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'disabled')
import sys
import json
import csv
import time
import subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

MJCF = 'models/VISPA_crawling_rwa3.xml'
URDF = 'models/VISPA_crawling_fixed.urdf'
BASELINE_CSV = 'results/j2_adjconv/c25_fulldiag.csv'
REPLAY_DIR = 'results/gate_run_scratch'
REPLAY_PREFIX = 'gate/_run/c25_replay'
REPLAY_CSV = REPLAY_PREFIX + '_fulldiag.csv'
LOCK = 'gate/environment.lock'
VERDICT = 'gate/last_verdict.json'

# Non-reproducible columns excluded from artifact identity: wall-clock per-tick
# timings. Documented in gate/EXCEPTIONS.md. NOT a metric-equivalence exception
# (which needs sign-off) — a definitional exclusion of nondeterministic
# instrumentation. All 64 physics/control columns are byte-compared.
EXCLUDE_COLS = ['qp_time_ms', 'nmpc_time_ms']

os.makedirs('gate/_run', exist_ok=True)


# ── check 4: environment pin ─────────────────────────────────────────
def live_versions():
    import platform
    import numpy
    import mujoco
    import pinocchio
    import casadi
    import scipy
    return {
        'python': platform.python_version(),
        'numpy': numpy.__version__,
        'mujoco': mujoco.__version__,
        'pinocchio': pinocchio.__version__,
        'casadi': casadi.__version__,
        'scipy': scipy.__version__,
        'ipopt_available': bool(casadi.has_nlpsol('ipopt')),
    }


def check_env():
    live = live_versions()
    if not os.path.exists(LOCK):
        json.dump(live, open(LOCK, 'w'), indent=2)
        return {'status': 'WARN', 'note': 'environment.lock created from live env',
                'live': live}
    locked = json.load(open(LOCK))
    mism = {k: {'locked': locked.get(k), 'live': live.get(k)}
            for k in set(locked) | set(live) if locked.get(k) != live.get(k)}
    return {'status': 'PASS' if not mism else 'WARN',
            'mismatches': mism, 'locked': locked, 'live': live}


# ── check 3: two-model consistency (BIN A-a2 + B-b1) ─────────────────
def check_two_model():
    import mujoco
    import pinocchio as pin
    TOL_M, TOL_C, TOL_L, TOL_K = 1e-6, 1e-6, 1e-4, 1e-5
    fails = []
    m = mujoco.MjModel.from_xml_path(MJCF)
    rm = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
    rd = rm.createData()
    d = mujoco.MjData(m)
    d.qpos[:] = m.key_qpos[0]
    mujoco.mj_forward(m, d)
    q0 = pin.neutral(rm)
    pin.forwardKinematics(rm, rd, q0)
    pin.updateFramePlacements(rm, rd)

    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    sid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)

    def rigid_group(root):
        grp, stack = [root], [root]
        while stack:
            p = stack.pop()
            for b in range(m.nbody):
                if m.body_parentid[b] == p and b != p and m.body_jntnum[b] == 0:
                    grp.append(b); stack.append(b)
        return grp

    def mj_inv(bodies):
        M = 0.0; c = np.zeros(3)
        for b in bodies:
            M += m.body_mass[b]; c += m.body_mass[b] * d.xipos[b]
        c /= M
        I = np.zeros((3, 3))
        for b in bodies:
            R = d.ximat[b].reshape(3, 3)
            Ib = R @ np.diag(m.body_inertia[b]) @ R.T
            r = d.xipos[b] - c
            I += Ib + m.body_mass[b] * ((r @ r) * np.eye(3) - np.outer(r, r))
        return M, c, np.sort(np.linalg.eigvalsh(I))

    def pin_inv(j):
        Y = rm.inertias[j]; oMi = rd.oMi[j]
        return float(Y.mass), oMi.act(Y.lever), np.sort(np.linalg.eigvalsh(np.asarray(Y.inertia)))

    # b1 naming contract
    for nm in ('torso', 'structure'):
        if bid(nm) < 0: fails.append({'quantity': f'naming/body:{nm}', 'detail': 'missing in MJCF'})
    for nm in ('gripper_a', 'gripper_b', 'anchor_1a', 'anchor_1b'):
        if sid(nm) < 0: fails.append({'quantity': f'naming/site:{nm}', 'detail': 'missing in MJCF'})
    for nm in ('rw_x', 'Joint_1_a', 'Joint_1_b', 'Joint_swivel_a', 'Joint_swivel_b'):
        if jid(nm) < 0: fails.append({'quantity': f'naming/joint:{nm}', 'detail': 'missing in MJCF'})
    for fn in ('tool_a', 'tool_b', 'Link_0'):
        if not rm.existFrame(fn): fails.append({'quantity': f'naming/frame:{fn}', 'detail': 'missing in URDF'})

    # joint order (+ suffix contract)
    pin_names = [rm.names[i] for i in range(2, rm.njoints)]
    mj_arm = [n for n in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt))
              if n and n.startswith('Joint_') and (n.endswith('_a') or n.endswith('_b'))]
    if mj_arm != pin_names:
        fails.append({'quantity': 'joint_order', 'mjcf': mj_arm, 'urdf': pin_names})

    # per-link composite mass / COM / principal inertia
    links = [('torso(Link_0)', bid('torso'), 1)]
    links += [(jn, int(m.jnt_bodyid[jid(jn)]), rm.getJointId(jn)) for jn in pin_names]
    for label, mjb, pj in links:
        Mm, cm, Im = mj_inv(rigid_group(mjb))
        Mp, cp, Ip = pin_inv(pj)
        dM, dC, dI = abs(Mm - Mp), float(np.max(np.abs(cm - cp))), float(np.max(np.abs(Im - Ip)))
        if not (dM < TOL_M and dC < TOL_C and dI < TOL_M):
            fails.append({'quantity': f'link:{label}',
                          'dmass': dM, 'dcom': dC, 'dinertia': dI,
                          'mjcf_mass': Mm, 'urdf_mass': Mp})

    # total robot mass
    def robot_bodies():
        tid = bid('torso'); out = set()
        for b in range(m.nbody):
            p = b
            while p > 0:
                if p == tid: out.add(b); break
                p = m.body_parentid[p]
        return out
    mjt = float(sum(m.body_mass[b] for b in robot_bodies()))
    pit = float(pin.computeTotalMass(rm))
    if abs(mjt - pit) > TOL_M:
        fails.append({'quantity': 'total_mass', 'mjcf': mjt, 'urdf': pit})

    # joint limits
    for jn in pin_names:
        mr = np.asarray(m.jnt_range[jid(jn)], float)
        qi = rm.idx_qs[rm.getJointId(jn)]
        pl = np.array([rm.lowerPositionLimit[qi], rm.upperPositionLimit[qi]], float)
        if float(np.max(np.abs(mr - pl))) >= TOL_L:
            fails.append({'quantity': f'joint_limit:{jn}', 'mjcf': mr.tolist(), 'urdf': pl.tolist()})

    # tool frame placements (relative to torso, all-joints-zero)
    tb = bid('torso'); Rt = d.xmat[tb].reshape(3, 3)
    oMt = rd.oMf[rm.getFrameId('Link_0')]
    for tool in ('tool_a', 'tool_b'):
        a = Rt.T @ (d.xpos[bid(tool)] - d.xpos[tb])
        oMe = rd.oMf[rm.getFrameId(tool)]
        b = oMt.rotation.T @ (oMe.translation - oMt.translation)
        if float(np.max(np.abs(a - b))) >= TOL_K:
            fails.append({'quantity': f'frame:{tool}', 'mjcf': a.tolist(), 'urdf': b.tolist()})

    return {'status': 'PASS' if not fails else 'FAIL',
            'n_links_checked': len(links), 'n_arm_joints': len(pin_names),
            'total_mass_kg': round(mjt, 6), 'fails': fails}


# ── checks 1+2: canonical replay + export + artifact identity ─────────
def run_subprocess(argv, tag, timeout):
    env = dict(os.environ, PYTHONPATH=ROOT, MUJOCO_GL='disabled')
    t0 = time.time()
    p = subprocess.run([sys.executable] + argv, cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    tail = (p.stdout or '').strip().splitlines()[-6:]
    return {'tag': tag, 'returncode': p.returncode, 'seconds': round(dt, 1),
            'stdout_tail': tail, 'stderr_tail': (p.stderr or '').strip().splitlines()[-6:]}


def check_replay_and_identity():
    rep = run_subprocess(['gate/replay_canonical.py'], 'replay', timeout=3600)
    sim_log = f'{REPLAY_DIR}/sim_log.json'
    if not os.path.exists(sim_log):
        return {'status': 'FAIL', 'reason': 'replay produced no sim_log.json',
                'replay': rep}, None
    exp = run_subprocess(['scripts/diag_full_diag_export.py',
                          '--run-dir', REPLAY_DIR, '--out-prefix', REPLAY_PREFIX],
                         'export', timeout=600)
    if not os.path.exists(REPLAY_CSV):
        return {'status': 'FAIL', 'reason': 'export produced no fulldiag CSV',
                'replay': rep, 'export': exp}, None
    ident = diff_csv(REPLAY_CSV, BASELINE_CSV, EXCLUDE_COLS)
    return {'status': ident['status'], 'replay': rep, 'export': exp, 'identity': ident}, ident


def diff_csv(replay, baseline, exclude):
    with open(replay, newline='') as f:
        R = list(csv.reader(f))
    with open(baseline, newline='') as f:
        B = list(csv.reader(f))
    if not R or not B:
        return {'status': 'FAIL', 'reason': 'empty csv',
                'replay_rows': len(R), 'baseline_rows': len(B)}
    if R[0] != B[0]:
        return {'status': 'FAIL', 'reason': 'header mismatch',
                'replay_header': R[0], 'baseline_header': B[0]}
    header = R[0]
    excl = {header.index(c) for c in exclude if c in header}
    if len(R) != len(B):
        return {'status': 'FAIL', 'reason': 'row count',
                'replay_rows': len(R), 'baseline_rows': len(B)}
    ncmp = 0
    for i in range(1, len(R)):
        if len(R[i]) != len(B[i]):
            return {'status': 'FAIL', 'reason': f'column count differs at row {i}'}
        for j in range(len(R[i])):
            if j in excl:
                continue
            ncmp += 1
            if R[i][j] != B[i][j]:
                return {'status': 'FAIL', 'reason': 'field mismatch',
                        'row': i, 'column': header[j],
                        'replay': R[i][j], 'baseline': B[i][j]}
    return {'status': 'PASS', 'rows': len(R) - 1,
            'columns_total': len(header), 'columns_excluded': exclude,
            'fields_compared': ncmp}


def main():
    t0 = time.time()
    print('=' * 70)
    print('CLEANUP GATE — canonical reproduction + two-model consistency')
    print('=' * 70, flush=True)

    env = check_env()
    print(f"[4] environment pin           : {env['status']}"
          + (f"  ({env.get('note')})" if env.get('note') else '')
          + (f"  mismatches={list(env.get('mismatches', {}))}" if env.get('mismatches') else ''),
          flush=True)

    tm = check_two_model()
    print(f"[3] two-model consistency     : {tm['status']}"
          f"  ({tm['n_links_checked']} links, {tm['n_arm_joints']} joints, "
          f"total {tm['total_mass_kg']} kg)"
          + (f"  FAILS={[x['quantity'] for x in tm['fails']]}" if tm['fails'] else ''),
          flush=True)

    print('[1] canonical replay + export : running (this re-runs the 6-step '
          'managed traversal; minutes) ...', flush=True)
    ri, ident = check_replay_and_identity()
    print(f"[1] canonical replay + export : "
          f"replay rc={ri['replay']['returncode']} ({ri['replay']['seconds']}s), "
          f"export rc={ri.get('export', {}).get('returncode')}", flush=True)
    if ident is None:
        print(f"[2] artifact identity         : FAIL  ({ri.get('reason')})", flush=True)
    else:
        extra = (f"  {ident.get('reason')} @ row {ident.get('row')} col {ident.get('column')}"
                 if ident['status'] == 'FAIL' else
                 f"  ({ident['rows']} rows × {ident['fields_compared']} fields, "
                 f"excl {sorted(EXCLUDE_COLS)})")
        print(f"[2] artifact identity         : {ident['status']}{extra}", flush=True)

    passed = (tm['status'] == 'PASS'
              and ri['status'] == 'PASS'
              and ri.get('identity', {}).get('status') == 'PASS')
    verdict = {
        'verdict': 'PASS' if passed else 'FAIL',
        'baseline': 'bfd5509 (main HEAD at founding)',
        'seconds_total': round(time.time() - t0, 1),
        'checks': {
            'canonical_replay': {'status': ri['status'], 'replay': ri.get('replay'),
                                 'export': ri.get('export')},
            'artifact_identity': ri.get('identity') or {'status': 'FAIL',
                                                         'reason': ri.get('reason')},
            'two_model_consistency': tm,
            'environment_pin': env,
        },
        'notes': [
            'Tier-0 identity is byte-equality on all fulldiag columns EXCEPT '
            'the wall-clock timings qp_time_ms/nmpc_time_ms (nondeterministic '
            'instrumentation; see gate/EXCEPTIONS.md).',
            'environment_pin is advisory (WARN-only) in this revision.',
        ],
    }
    json.dump(verdict, open(VERDICT, 'w'), indent=2)
    print('-' * 70)
    print(f"VERDICT: {verdict['verdict']}   "
          f"(env {env['status']}, {verdict['seconds_total']}s)   -> {VERDICT}")
    print('=' * 70, flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
