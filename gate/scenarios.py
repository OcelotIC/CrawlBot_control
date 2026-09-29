#!/usr/bin/env python3
"""Differential replay of NON-canonical branches: old tree vs new tree.

The canonical replay cannot see DWELL, SKIP (pre-planner infeasible),
dock TIMEOUT or stop_on_failed_step — it never
takes them. Each scenario below forces one, on a shortened traversal, and is run
twice: once from an old source tree (a git worktree at the pre-refactor commit)
and once from the working tree. Outputs are compared bit-for-bit with
gate/local_ref.py's comparator, and stdout is compared with wall-clock numbers
masked.

    git worktree add /tmp/old_tree <pre-change commit>
    python3 gate/scenarios.py run  /tmp/old_tree old   [--cov] [names...]
    python3 gate/scenarios.py run  .             new   [names...]
    python3 gate/scenarios.py diff [--new-side S] [names...]  # exit 0 iff identical
    python3 gate/scenarios.py canonical-cov /tmp/old_tree old  # once, same commit
    python3 gate/scenarios.py coverage [names...]      # paths beyond canonical
    git worktree remove --force /tmp/old_tree

The "old" outputs are a FROZEN baseline: each carries a _manifest.json with the
commit it was produced from, and stays valid for every later commit that is
bit-identical to it — re-run only the "new" side after each commit.

--cov runs the scenario under coverage.py (source=crawlbot) and writes
gate/_run/local_ref/cov_<name>_<side>.json; ``coverage`` then lists, per
scenario, the lines it executes that the canonical replay does not
(gate/_run/cov/cov.json, which must come from the same commit), grouped by
enclosing function — i.e. what the scenario actually covers, measured.

Outputs land in gate/_run/local_ref/scn_<name>_<side>/ (git-ignored). The
scenario is run by THIS file's copy of the harness against either tree, so the
old tree does not need to contain it.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BASE = os.path.join(REPO, 'gate/_run/local_ref')

# name -> (dca.main kwargs overrides, SimConfig overrides[, extras])
# extras: {'attrs': {sim attribute: value}  — set after __init__ (diag hooks),
#          'nmpc_fail': {call indices}, 'qp_fail': {call indices},
#          'qp_fail_track': {indices among settle_mode=False QP calls},
#          'nmpc_unsuccessful': {call index: status}} — the
#          CentroidalNMPC.solve / WholeBodyQP.solve call with that 0-based
#          index raises RuntimeError (forced-failure fallbacks).
_SHORT = {'n_steps': 1, 'settle_seconds': 1.0}
SCENARIOS = {
    'timeout': ({'n_steps': 2, 'settle_seconds': 1.0},
                {'weld_radius': 0.0005, 't_hold_max': 0.5, 't_ss_margin': 0.3,
                 'stop_on_failed_step': False}),
    'stop': ({'n_steps': 2, 'settle_seconds': 1.0},
             {'weld_radius': 0.0005, 't_hold_max': 0.5, 't_ss_margin': 0.3,
              'stop_on_failed_step': True}),
    'skip': ({'n_steps': 2, 'settle_seconds': 1.0},
             {'preplanner_max_iter': 1, 'stop_on_failed_step': False}),
    'skipstop': ({'n_steps': 2, 'settle_seconds': 1.0},
                 {'preplanner_max_iter': 1, 'stop_on_failed_step': True}),
    'dwell': ({'n_steps': 2, 'settle_seconds': 1.0, 'dt_ds': 3.0}, {}),
    # Torso CoM reference time compression (0 < ff < 1) in PlannerReferences.
    'ff_compress': (_SHORT, {'torso_early_finish_fraction': 0.7}),
    # ── L8: runtime diagnostic hooks (DiagHooks via sim._diag_*) ────────
    'diag_pure_pd': (_SHORT, {}, {'attrs': {'_diag_pure_pd': True}}),
    'diag_freeze_ref': (_SHORT, {}, {'attrs': {'_diag_freeze_ref': True}}),
    'diag_disable_aocs': (_SHORT, {}, {'attrs': {'_diag_disable_aocs': True}}),
    'diag_lock_arm_joints': (_SHORT, {},
                             {'attrs': {'_diag_lock_arm_joints': True}}),
    # ── L8: forced solver failures ───────────────────────────────────────
    # NMPC call 0 fails (no previous solve -> reference-level fallback),
    # calls 3-4 fail (receding-horizon shifted fallback).
    'nmpc_fail': (_SHORT, {}, {'nmpc_fail': {0, 3, 4}}),
    # QP call 2 lands in the setup settle (joint-damping fallback).
    'qp_fail': (_SHORT, {}, {'qp_fail': {2}}),
    # Indices counted over tracking solves only (settle_mode=False, i.e. SS
    # sub-steps): the zero-torque QP-FAIL path of WholeBodyController.track.
    'qp_fail_track': (_SHORT, {}, {'qp_fail_track': {30, 31, 32}}),
    # NMPC returns success=False WITHOUT raising (status codes 2 and 1).
    'nmpc_unsuccessful': (_SHORT, {}, {'nmpc_unsuccessful': {
        3: 'Infeasible_Problem_Detected', 5: 'Maximum_Iterations_Exceeded'}}),
    # ── L8: inter-step settle with the AOCS off (wheels commanded to 0) ──
    'aocs_off_interstep': (_SHORT, {'aocs_active_in_interstep': False}),
    # Inter-step settle QP fed the entry-frozen h_w (no per-tick refresh).
    'hw_refresh_off': (_SHORT, {'interstep_hw_refresh': False}),
    # ── R0: the paper's Table 2 configurations (short traversal) ─────────
    # Arms of the three-way envelope ablation (e37b9ca, branch
    # claude/review-closure-bloc-2-uwu1x7) + the published unmanaged run.
    # full = the canonical (covered by the canonical replay itself).
    # rate: NMPC rate cap kept, NMPC storage (h_w) box off.
    'table2_rate': (_SHORT, {'enforce_hw_conservation': False}),
    # none: NMPC rate cap AND storage box off; QP box + AOCS clip held at 2.5
    # (SimConfig.nmpc_tau_w_max, ported from e37b9ca in a44a3b6; the frozen
    # 36c44bf baseline was produced by a harness injection of the same single
    # effect — CentroidalNMPCConfig(tau_w_max=inf) — which the diff equates).
    'table2_none': (_SHORT, {'enforce_hw_conservation': False,
                             'nmpc_tau_w_max': float('inf')}),
    # u25: tau_w_max=1e6 through dca -> NMPC cap + QP box + AOCS clip lifted.
    'table2_u25': (dict(_SHORT, tau_w_max=1e6), {}),
}

C_KWARGS = dict(
    legacy=False, alpha_torso_lin=0.0, anchor_dx=0.8, mass_ratio=0.01,
    aocs_mode='legacy_pid_numerical', settle_seconds=20.0,
    K_theta=1.0, K_omega=50.0, tau_w_max=2.5,
    n_steps=6, ss_two_task=True, ss_alpha_mom=400.0,
    alpha_torso_pose=2000.0, ss_alpha_ee=1000.0, ss_alpha_posture=2e1,
    ss_alpha_wrench=1.0, ss_kp_torso=3.0, ss_kd_torso=2.5,
    qp_envelope_exact=True,
    interstep_settle_alpha_wrench=3.0, interstep_settle_epsilon_v=5e-3,
)


def child(name, out_rel):
    """Runs inside the tree under test (cwd = that tree)."""
    sys.path.insert(0, os.getcwd())
    os.environ.setdefault('MUJOCO_GL', 'disabled')
    import crawlbot.solvers.hierarchical_qp as hq
    _ow = hq.HierarchicalQP._solve_weighted

    def _pw(self, tasks, x0):
        self.regularization = 1e-6
        return _ow(self, tasks, x0)
    hq.HierarchicalQP._solve_weighted = _pw

    import crawlbot.simulation.sim_loop as sl
    spec = SCENARIOS[name]
    kw_over, cfg_over = spec[0], spec[1]
    extras = spec[2] if len(spec) > 2 else {}
    _init = sl.SimulationLoop.__init__

    def _init_over(self, *a, **k):
        _init(self, *a, **k)
        for key, val in cfg_over.items():
            assert hasattr(self.cfg, key), key
            setattr(self.cfg, key, val)
        for key, val in extras.get('attrs', {}).items():
            assert hasattr(self, key), key
            setattr(self, key, val)
        # dca's per-step q-log calls sim._step_q_start.tolist() even when the
        # pre-planner failed on the FIRST step (still None) and crashes —
        # a driver defect. Seed placeholders so SKIP scenarios run through.
        import numpy as _np
        self._step_q_start = _np.zeros(1)
        self._step_q_end = _np.zeros(1)
    sl.SimulationLoop.__init__ = _init_over

    def _failing(cls, meth, fail_at, label, only=None):
        orig = getattr(cls, meth)
        n = [0]

        def wrapped(self, *a, **k):
            if only is not None and not only(k):
                return orig(self, *a, **k)
            i = n[0]
            n[0] += 1
            if i in fail_at:
                print(f'[scenario] forced {label} failure, call {i}', flush=True)
                raise RuntimeError(f'forced {label} failure (scenario)')
            return orig(self, *a, **k)
        setattr(cls, meth, wrapped)

    if extras.get('nmpc_fail'):
        import crawlbot.solvers.centroidal_nmpc as cn
        _failing(cn.CentroidalNMPC, 'solve', extras['nmpc_fail'], 'NMPC')
    if extras.get('qp_fail'):
        import crawlbot.solvers.wholebody_qp as wq
        _failing(wq.WholeBodyQP, 'solve', extras['qp_fail'], 'QP')
    if extras.get('nmpc_unsuccessful'):
        import crawlbot.solvers.centroidal_nmpc as cn
        orig_solve = cn.CentroidalNMPC.solve
        unsucc = extras['nmpc_unsuccessful']
        cnt = [0]

        def solve_unsucc(self, *a, **k):
            out = orig_solve(self, *a, **k)
            i = cnt[0]
            cnt[0] += 1
            if i in unsucc:
                info = out[-1]
                info.success = False
                info.status = unsucc[i]
                print(f'[scenario] forced NMPC status {unsucc[i]}, call {i}',
                      flush=True)
            return out
        cn.CentroidalNMPC.solve = solve_unsucc
    if extras.get('qp_fail_track'):
        import crawlbot.solvers.wholebody_qp as wq
        _failing(wq.WholeBodyQP, 'solve', extras['qp_fail_track'], 'QP-track',
                 only=lambda k: not k.get('settle_mode', False))

    import scripts.diag_cooperative_arms as dca
    from scripts.diag_cooperative_arms import _mutate_mjcf, _mjcf_md5, MJCF
    kw = dict(C_KWARGS, **kw_over, out_dir_override=out_rel)
    with open(MJCF) as f:
        orig = f.read()
    pre = _mjcf_md5(MJCF)
    try:
        _mutate_mjcf(damping=0.0, armature=0.05, anchor_dx=0.8, mass_ratio=0.01)
        try:
            dca.main(**kw)
        except Exception as e:
            print(f'[scenario] main() raised: {type(e).__name__}: {e}', flush=True)
    finally:
        with open(MJCF, 'w') as f:
            f.write(orig)
        assert _mjcf_md5(MJCF) == pre


def run(tree, side, names, cov=False):
    import json
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=tree,
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', 'crawlbot/'],
                           cwd=tree, capture_output=True, text=True).stdout.strip()
    for name in names:
        tag = f'scn_{name}_{side}'
        out_abs = os.path.join(BASE, tag)
        subprocess.run(['rm', '-rf', out_abs])
        os.makedirs(out_abs)
        # dca writes to results/<out_dir_override> inside the tree under test
        out_rel = f'_scn_{name}_{side}'
        env = dict(os.environ, MUJOCO_GL='disabled', PYTHONPATH=tree)
        cov_data = os.path.join(BASE, f'cov_{name}_{side}.data')
        pre = ([sys.executable, '-m', 'coverage', 'run', f'--data-file={cov_data}',
                '--source=crawlbot'] if cov else [sys.executable])
        p = subprocess.run(pre + [os.path.abspath(__file__), '_child',
                                  name, out_rel], cwd=tree, env=env,
                           capture_output=True, text=True, timeout=3600)
        if cov:
            subprocess.run([sys.executable, '-m', 'coverage', 'json',
                            f'--data-file={cov_data}', '-o',
                            os.path.join(BASE, f'cov_{name}_{side}.json')],
                           cwd=tree, capture_output=True)
        src = os.path.join(tree, 'results', out_rel)
        if os.path.isdir(src):
            for fn in os.listdir(src):
                os.replace(os.path.join(src, fn), os.path.join(out_abs, fn))
            os.rmdir(src)
        with open(os.path.join(out_abs, '_stdout.txt'), 'w') as f:
            f.write(p.stdout)
        with open(os.path.join(out_abs, '_stderr.txt'), 'w') as f:
            f.write(p.stderr)
        with open(os.path.join(out_abs, '_manifest.json'), 'w') as f:
            json.dump({'commit': commit, 'crawlbot_dirty': bool(dirty),
                       'tree': tree, 'coverage': cov}, f, indent=1)
        print(f'[{side}] {name}: rc={p.returncode}, '
              f'{len(os.listdir(out_abs))} files', flush=True)


MASK = re.compile(r'  Output: .*|\d+\.\d+ ms|\d+\.\d+s sim|written .*|'
                  r't_max|\s+\d+\.\d\s+\d+\.\d\s+(?=\d+\s+Solve)')


def diff(names, new='new'):
    sys.path.insert(0, REPO)
    os.chdir(REPO)
    import gate.local_ref as lr
    bad = 0
    for name in names:
        a, b = f'scn_{name}_old', f'scn_{name}_{new}'
        print(f'=== {name}')
        # stdout / stderr are compared separately (masked), not as files
        for side in (a, b):
            for fn in ('_stdout.txt', '_stderr.txt'):
                src = os.path.join(BASE, side, fn)
                if os.path.exists(src):
                    os.replace(src, os.path.join(BASE, side + fn))
        rc = lr.diff(a, b)
        so = [MASK.sub('#', open(os.path.join(BASE, t + '_stdout.txt')).read())
              for t in (a, b)]
        same = so[0] == so[1]
        print(f'  stdout (wall-clock masked): {"identical" if same else "DIFFERS"}'
              f' ({len(so[0].splitlines())} lines)')
        if not same:
            la, lb = so[0].splitlines(), so[1].splitlines()
            for i, (x, y) in enumerate(zip(la, lb)):
                if x != y:
                    print(f'    first diff line {i}:\n    old {x!r}\n    new {y!r}')
                    break
            else:
                print(f'    line counts {len(la)} vs {len(lb)}')
        for kw in ('DWELL', 'SKIP', 'TIMEOUT', 'stop_on_failed_step', 'DOCK',
                   'HOLD', 'Traceback', 'raised'):
            n = so[1].count(kw)
            if n:
                print(f'  branch marker {kw!r}: {n}')
        bad |= rc != 0 or not same
    print('ALL IDENTICAL' if not bad else 'DIFFERENCES FOUND')
    return bad


def _functions(path, commit=None):
    """line -> 'Class.method' / 'function' for every line of a module, read
    at ``commit`` (the coverage's own commit) when given."""
    import ast
    if commit:
        src = subprocess.run(['git', 'show', f'{commit}:{path}'], cwd=REPO,
                             capture_output=True, text=True).stdout
    else:
        src = open(os.path.join(REPO, path)).read()
    tree = ast.parse(src)
    owner = {}

    def visit(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.ClassDef)):
                q = f'{prefix}{ch.name}'
                for ln in range(ch.lineno, (ch.end_lineno or ch.lineno) + 1):
                    owner[ln] = q
                visit(ch, q + '.')
    visit(tree, '')
    return owner


def canonical_cov(tree, side):
    """Canonical replay under coverage in ``tree`` -> cov_canonical_<side>.json."""
    data = os.path.join(BASE, f'cov_canonical_{side}.data')
    env = dict(os.environ, MUJOCO_GL='disabled', PYTHONPATH=tree)
    subprocess.run([sys.executable, '-m', 'coverage', 'run', f'--data-file={data}',
                    '--source=crawlbot', 'gate/replay_canonical.py'],
                   cwd=tree, env=env, capture_output=True, timeout=3600)
    subprocess.run([sys.executable, '-m', 'coverage', 'json', f'--data-file={data}',
                    '-o', os.path.join(BASE, f'cov_canonical_{side}.json')],
                   cwd=tree, capture_output=True)
    print(f'[{side}] canonical coverage -> cov_canonical_{side}.json')


def coverage(names, side='old'):
    """Lines each scenario executes that the canonical replay does not."""
    import json
    # The canonical coverage must come from the SAME commit as the scenario
    # coverage, or line numbers do not line up: prefer the one recorded next
    # to the baseline (`canonical-cov`), else the routine's gate/_run/cov.
    cc = os.path.join(BASE, f'cov_canonical_{side}.json')
    if not os.path.exists(cc):
        cc = os.path.join(REPO, 'gate/_run/cov/cov.json')
    canon = json.load(open(cc))['files']
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()
    out = ['| scenario | beyond-canonical lines, by function |', '|---|---|']
    for name in names:
        cp = os.path.join(BASE, f'cov_{name}_{side}.json')
        if not os.path.exists(cp):
            out.append(f'| `{name}` | (no coverage run) |')
            continue
        man = json.load(open(os.path.join(BASE, f'scn_{name}_{side}',
                                          '_manifest.json')))
        same = subprocess.run(['git', 'diff', '--quiet', man['commit'], head,
                               '--', 'crawlbot/'], cwd=REPO).returncode == 0
        if not same and not cc.endswith(f'cov_canonical_{side}.json'):
            print(f'!! {name}: coverage from {man["commit"][:8]}; crawlbot/ '
                  f'differs at HEAD {head[:8]} — line numbers may not match',
                  file=sys.stderr)
        files = json.load(open(cp))['files']
        cells = []
        for f in sorted(files):
            extra = (set(files[f]['executed_lines'])
                     - set(canon.get(f, {}).get('executed_lines', [])))
            if not extra:
                continue
            own = _functions(f, man['commit'])
            by = {}
            for ln in extra:
                by.setdefault(own.get(ln, '<module>'), []).append(ln)
            short = f.replace('crawlbot/', '')
            for fn, lns in sorted(by.items()):
                cells.append(f'{short}:{fn} ({len(lns)}: L{min(lns)}–{max(lns)})')
        out.append(f'| `{name}` | ' + ('<br>'.join(cells) or '— none —') + ' |')
    text = '\n'.join(out)
    open(os.path.join(BASE, 'scenario_coverage.md'), 'w').write(text + '\n')
    print(text)


if __name__ == '__main__':
    new_side = 'new'
    if '--new-side' in sys.argv:
        k = sys.argv.index('--new-side')
        new_side = sys.argv[k + 1]
        del sys.argv[k:k + 2]
    args = [a for a in sys.argv[1:] if a != '--cov']
    if args[0] == '_child':
        child(args[1], args[2])
    elif args[0] == 'run':
        run(os.path.abspath(args[1]), args[2], args[3:] or list(SCENARIOS),
            cov='--cov' in sys.argv)
    elif args[0] == 'diff':
        sys.exit(diff(args[1:] or list(SCENARIOS), new=new_side))
    elif args[0] == 'canonical-cov':
        canonical_cov(os.path.abspath(args[1]), args[2])
    elif args[0] == 'coverage':
        coverage(args[1:] or list(SCENARIOS))
