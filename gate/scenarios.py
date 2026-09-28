#!/usr/bin/env python3
"""Differential replay of NON-canonical branches: old tree vs new tree.

The canonical replay cannot see DWELL, SKIP (pre-planner infeasible),
dock TIMEOUT, stop_on_failed_step, or the diag_*_on_abort overrides — it never
takes them. Each scenario below forces one, on a shortened traversal, and is run
twice: once from an old source tree (a git worktree at the pre-refactor commit)
and once from the working tree. Outputs are compared bit-for-bit with
gate/local_ref.py's comparator, and stdout is compared with wall-clock numbers
masked.

    git worktree add /tmp/old_tree <pre-change commit>
    python3 gate/scenarios.py run  /tmp/old_tree old   [names...]
    python3 gate/scenarios.py run  .             new   [names...]
    python3 gate/scenarios.py diff [names...]          # exit 0 iff identical
    git worktree remove --force /tmp/old_tree

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

# name -> (dca.main kwargs overrides, SimConfig overrides)
SCENARIOS = {
    'timeout': ({'n_steps': 2, 'settle_seconds': 1.0},
                {'weld_radius': 0.0005, 't_hold_max': 0.5, 't_ss_margin': 0.3,
                 'stop_on_failed_step': False}),
    'abortdiag': ({'n_steps': 1, 'settle_seconds': 1.0},
                  {'weld_radius': 0.0005, 't_hold_max': 0.5, 't_ss_margin': 0.3,
                   'stop_on_failed_step': False,
                   'diag_force_single_contact_on_abort': True,
                   'diag_freeze_torso_ref_on_abort': True,
                   'diag_disable_passivity_on_abort': True,
                   'ds_centroidal_mode': False,
                   'ds_torso_ref_from_state': False}),
    'stop': ({'n_steps': 2, 'settle_seconds': 1.0},
             {'weld_radius': 0.0005, 't_hold_max': 0.5, 't_ss_margin': 0.3,
              'stop_on_failed_step': True}),
    'skip': ({'n_steps': 2, 'settle_seconds': 1.0},
             {'preplanner_max_iter': 1, 'stop_on_failed_step': False}),
    'skipstop': ({'n_steps': 2, 'settle_seconds': 1.0},
                 {'preplanner_max_iter': 1, 'stop_on_failed_step': True}),
    'dwell': ({'n_steps': 2, 'settle_seconds': 1.0, 'dt_ds': 3.0}, {}),
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
    kw_over, cfg_over = SCENARIOS[name]
    _init = sl.SimulationLoop.__init__

    def _init_over(self, *a, **k):
        _init(self, *a, **k)
        for key, val in cfg_over.items():
            assert hasattr(self.cfg, key), key
            setattr(self.cfg, key, val)
        # dca's per-step q-log calls sim._step_q_start.tolist() even when the
        # pre-planner failed on the FIRST step (still None) and crashes —
        # a driver defect. Seed placeholders so SKIP scenarios run through.
        import numpy as _np
        self._step_q_start = _np.zeros(1)
        self._step_q_end = _np.zeros(1)
    sl.SimulationLoop.__init__ = _init_over

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


def run(tree, side, names):
    for name in names:
        tag = f'scn_{name}_{side}'
        out_abs = os.path.join(BASE, tag)
        subprocess.run(['rm', '-rf', out_abs])
        os.makedirs(out_abs)
        # dca writes to results/<out_dir_override> inside the tree under test
        out_rel = f'_scn_{name}_{side}'
        env = dict(os.environ, MUJOCO_GL='disabled', PYTHONPATH=tree)
        p = subprocess.run([sys.executable, os.path.abspath(__file__), '_child',
                            name, out_rel], cwd=tree, env=env,
                           capture_output=True, text=True, timeout=3600)
        src = os.path.join(tree, 'results', out_rel)
        if os.path.isdir(src):
            for fn in os.listdir(src):
                os.replace(os.path.join(src, fn), os.path.join(out_abs, fn))
            os.rmdir(src)
        with open(os.path.join(out_abs, '_stdout.txt'), 'w') as f:
            f.write(p.stdout)
        with open(os.path.join(out_abs, '_stderr.txt'), 'w') as f:
            f.write(p.stderr)
        print(f'[{side}] {name}: rc={p.returncode}, '
              f'{len(os.listdir(out_abs))} files', flush=True)


MASK = re.compile(r'  Output: .*|\d+\.\d+ ms|\d+\.\d+s sim|written .*|'
                  r't_max|\s+\d+\.\d\s+\d+\.\d\s+(?=\d+\s+Solve)')


def diff(names):
    sys.path.insert(0, REPO)
    os.chdir(REPO)
    import gate.local_ref as lr
    bad = 0
    for name in names:
        a, b = f'scn_{name}_old', f'scn_{name}_new'
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


if __name__ == '__main__':
    if sys.argv[1] == '_child':
        child(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'run':
        run(os.path.abspath(sys.argv[2]), sys.argv[3],
            sys.argv[4:] or list(SCENARIOS))
    elif sys.argv[1] == 'diff':
        sys.exit(diff(sys.argv[2:] or list(SCENARIOS)))
