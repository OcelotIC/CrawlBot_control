"""NMPC horizon/discretization sweep driver.

Runs the canonical 6-step traversal under several (nmpc_N, nmpc_pred_dt, nmpc_period)
settings, saving each run's artifacts to its own directory so the variants can
be compared and plotted.

Why `nmpc_period` is a swept variable and not a constant
----------------------------------------------------
`sim_loop.py:2342` interpolates the NMPC plan across the QP sub-loop as

    alpha = qs / n_qp_per_nmpc,     n_qp_per_nmpc = nmpc_period / dt_qp
    r_ref = (1-alpha)*x_plan[:,0] + alpha*x_plan[:,1]

i.e. it walks from plan knot 0 to plan knot 1 over one CONTROL period
`nmpc_period`. Those knots are `nmpc_pred_dt` apart in PLAN time. The interpolation is
therefore time-correct only when `nmpc_pred_dt == nmpc_period`; otherwise the reference
the QP tracks is dilated by `nmpc_period / nmpc_pred_dt`. Setting `nmpc_pred_dt = 0.05` while
leaving `nmpc_period = 0.1` gives a 2x-slow CoM reference. Both readings are run
here so the effect is measured rather than assumed.

Each variant edits the SimConfig defaults in place, runs the same
`gate/replay_canonical.py` the gate uses, exports the 66-column fulldiag CSV,
and copies the artifacts out. `config.py` is restored from its original text in
a `finally`, whatever happens.

Usage:
    MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_nmpc_horizon_sweep.py
    MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_nmpc_horizon_sweep.py --only N20_dt05_p05
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, 'crawlbot/simulation/config.py')
SCRATCH = os.path.join(ROOT, 'results/gate_run_scratch')
DEST = os.path.join(ROOT, 'results/j2_adjconv/nmpc_sweep')

# tag -> (nmpc_N, nmpc_pred_dt, nmpc_period, nmpc_per_stage_refs)
HORIZON_VARIANTS = [
    # The literal reading of "N=20, dt=0.05": prediction step only. Leaves the
    # control period at 0.1 s, so the plan-interpolation is dilated 2x.
    ('N20_dt05_p10', 20, 0.05, 0.10, False, False, None),
    # The consistent reading: control period follows the prediction step, so
    # the interpolation invariant holds and the NMPC runs at 20 Hz.
    ('N20_dt05_p05', 20, 0.05, 0.05, False, False, None),
]

# F1 (NMPC_AUDIT): per-knot references. Rule 12 -- one variable per run, so the
# refactor is proven inert BEFORE the flag flips, and the flag flips BEFORE N
# moves. Each row differs from the one above it in exactly one field.
F1_VARIANTS = [
    ('F1off_N15', 15, 0.10, 0.10, False, False, None),   # must reproduce the committed N=15
    ('F1on_N15',  15, 0.10, 0.10, True, False, None),    # F1 effect, isolated
    ('F1on_N20',  20, 0.10, 0.10, True, False, None),    # then N: 15 -> 20 at 10 Hz
]

# F3: the prediction step / control period decoupling.
#   - the first row is the COMMITTED config; its fulldiag must come out
#     byte-identical to nmpc_sweep/F1on_N20 or the fix is not inert
#   - the second differs only in nmpc_pred_dt, which the pre-fix code could not
#     run correctly (it dilated the QP's reference 2x)
F3_VARIANTS = [
    ('F3_N20_dt10', 20, 0.10, 0.10, True, False, None),
    ('F3_N20_dt05', 20, 0.05, 0.10, True, False, None),
]

# F2 (NMPC_AUDIT): reactivate the RWA conservation box, staged.
#   step A — PATH box only, terminal still off
#   step B — terminal set on top, run only if A is sound
F2_VARIANTS_A = [('F2box_N20',     20, 0.10, 0.10, True, True, False)]
F2_VARIANTS_B = [('F2boxterm_N20', 20, 0.10, 0.10, True, True, True)]
# Control: F2 OFF on the CURRENT tree. Needed because the F2-off baseline
# (F3_N20_dt10) was produced several commits ago; without re-running it here,
# any "box vs off" difference is confounded with everything committed since.
F2_VARIANTS_CTL = [('F2off_ctl_N20',  20, 0.10, 0.10, True, False, None)]
# BITE TEST: the same box tightened BELOW the realized h_w peak (3.815 Nms).
# If this does not change the trajectory, the constraint is not wired to
# anything and "reactivating F2" would be a no-op.
F2_VARIANTS_BITE = [('F2bite_h35_N20', 20, 0.10, 0.10, True, True, True, 3.5)]

VARIANT_SETS = {'horizon': HORIZON_VARIANTS, 'f1': F1_VARIANTS,
                'f3': F3_VARIANTS, 'f2a': F2_VARIANTS_A, 'f2b': F2_VARIANTS_B,
                'f2ctl': F2_VARIANTS_CTL,
                'f2bite': F2_VARIANTS_BITE}

FIELDS = {
    'nmpc_N':  (r'^(\s*nmpc_N: int = )(\d+)', '{}'),
    'nmpc_pred_dt': (r'^(\s*nmpc_pred_dt: float = )([0-9.]+)', '{}'),
    'nmpc_period': (r'^(\s*nmpc_period: float = )([0-9.]+)', '{}'),
    'nmpc_per_stage_refs': (r'^(\s*nmpc_per_stage_refs: bool = )(True|False)', '{}'),
    'enforce_hw_conservation': (r'^(\s*enforce_hw_conservation: bool = )(True|False)', '{}'),
    'enforce_hw_terminal': (r'^(\s*enforce_hw_terminal: Optional\[bool\] = )(True|False|None)', '{}'),
    'h_max_tight': (r'^(\s*h_max_tight: np\.ndarray = field\(default_factory=lambda: np\.full\(3, )([0-9.]+)', '{}'),
}


def patch_config(text, n, ndt, pdt, per_stage, hw_box, hw_term, h_max=5.0):
    """Rewrite the SimConfig defaults; assert each substitution landed."""
    vals = {'nmpc_N': n, 'nmpc_pred_dt': ndt, 'nmpc_period': pdt,
            'nmpc_per_stage_refs': per_stage,
            'enforce_hw_conservation': hw_box, 'enforce_hw_terminal': hw_term,
            'h_max_tight': h_max}
    for name, (pat, fmt) in FIELDS.items():
        rx = re.compile(pat, re.M)
        if not rx.search(text):
            raise RuntimeError(f'could not locate {name} in config.py')
        text = rx.sub(lambda m: m.group(1) + fmt.format(vals[name]), text, count=1)
    # Verify by reading back.
    for name, (pat, _) in FIELDS.items():
        m = re.search(pat, text, re.M)
        got = m.group(2)
        want = str(vals[name])
        same = (got == want) if got in ('True', 'False', 'None') else (
            float(got) == float(want))
        if not same:
            raise RuntimeError(f'{name}: wrote {want}, read back {got}')
    return text


def run(argv, tag, timeout):
    env = dict(os.environ, MUJOCO_GL='disabled', PYTHONPATH=ROOT)
    t0 = time.time()
    p = subprocess.run([sys.executable] + argv, cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    print(f'    {tag}: rc={p.returncode} ({dt:.0f}s)', flush=True)
    if p.returncode != 0:
        print('    --- stderr tail ---')
        print('\n'.join(p.stderr.strip().splitlines()[-15:]))
    return p.returncode, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None, help='run a single variant tag')
    ap.add_argument('--set', default='horizon', choices=sorted(VARIANT_SETS),
                    help='which variant list to run')
    args = ap.parse_args()
    variants = VARIANT_SETS[args.set]

    original = open(CONFIG).read()
    os.makedirs(DEST, exist_ok=True)
    summary = []

    try:
        for tag, *rest in variants:
            n, ndt, pdt, per_stage, hw_box, hw_term = rest[:6]
            h_max = rest[6] if len(rest) > 6 else 5.0
            if args.only and tag != args.only:
                continue
            print(f'\n=== {tag}: nmpc_N={n} nmpc_pred_dt={ndt} nmpc_period={pdt} '
                  f'per_stage={per_stage} hw_box={hw_box} hw_term={hw_term} '
                  f'(horizon {n * ndt:.2f}s, {1 / pdt:.0f} Hz) ===', flush=True)
            open(CONFIG, 'w').write(
                patch_config(original, n, ndt, pdt, per_stage, hw_box, hw_term, h_max))

            rc, secs = run(['gate/replay_canonical.py'], 'replay', 3600)
            if rc != 0:
                summary.append({'tag': tag, 'status': 'REPLAY_FAIL'})
                continue
            run(['scripts/diag_full_diag_export.py', '--run-dir', SCRATCH,
                 '--out-prefix', os.path.join(SCRATCH, 'fulldiag')], 'export', 900)

            out = os.path.join(DEST, tag)
            if os.path.exists(out):
                shutil.rmtree(out)
            os.makedirs(out)
            for f in os.listdir(SCRATCH):
                src = os.path.join(SCRATCH, f)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(out, f))
            summary.append({'tag': tag, 'status': 'OK', 'nmpc_N': n,
                            'nmpc_pred_dt': ndt, 'nmpc_period': pdt,
                            'per_stage_refs': per_stage,
                            'hw_box': hw_box, 'hw_terminal': hw_term,
                            'h_max_tight': h_max,
                            'horizon_s': round(n * ndt, 3),
                            'replay_seconds': round(secs, 1)})
            print(f'    saved -> {out}', flush=True)
    finally:
        open(CONFIG, 'w').write(original)
        print(f'\nrestored {CONFIG}', flush=True)

    with open(os.path.join(DEST, f'sweep_summary_{args.set}.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
