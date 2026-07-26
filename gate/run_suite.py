#!/usr/bin/env python3
"""Make the component test suite enforceable — the half of verification
`run_gate.py` structurally cannot do.

`gate/run_gate.py` proves the CANONICAL RUN reproduces byte-identically against
`results/j2_adjconv/c25_fulldiag.csv`. That is a very strong claim about ONE
configuration and it is indifferent to correctness: a wrong controller
reproduces just as faithfully as a right one, and byte-identity then locks the
wrong behaviour in as the new baseline. The gate protects the past.

The suite is the only thing in this repository that can say a CHANGED
configuration is still physically valid — the dynamics residual, the passivity
inequality, the CoM-Jacobian mass form, the J̇ assembly, the quaternion
conventions, the momentum identity. Those hold or fail independently of any
tuning, which is exactly why they survive re-tuning and the byte-identity check
does not.

It went un-run and un-gated long enough to drift from a documented "2 failures"
to a measured 12 without anyone noticing, and 11 of those 12 turned out to be
the cleanup chantier's own breakage (CLEANUP-27 §4). A suite nobody runs is not
a safety net; it is a claim about a safety net. Hence this.

Two modes, same pass criterion:

    --fast     deselects @pytest.mark.slow      — the per-commit gate
    (default)  everything                      — the pre-merge gate

PASS requires zero failures, zero errors, AND zero XPASS. A test marked
`xfail(strict=True)` that starts passing is news — the open question behind it
was resolved by something nobody wrote down — so it breaks the gate instead of
quietly turning green.

Usage
-----
    PYTHONPATH=. python3 gate/run_suite.py --fast          # per-commit
    PYTHONPATH=. python3 gate/run_suite.py                 # pre-merge
    PYTHONPATH=. python3 gate/run_suite.py tests/test_x.py # while iterating

A trailing path narrows the run. CI passes none, so narrowing cannot leak into
the gate itself; the verdict line says what was actually run.
"""
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

JUNIT = 'gate/_run/suite_junit.xml'
VERDICT = 'gate/_run/suite_verdict.json'


def main() -> int:
    args = sys.argv[1:]
    fast = '--fast' in args
    targets = [a for a in args if not a.startswith('-')] or ['tests/']

    env = dict(os.environ)
    env['PYTHONPATH'] = '.'
    # Not a preference — a requirement. Under MUJOCO_GL=osmesa this container
    # aborts pytest COLLECTION (`tests/test_diagnostics.py` ->
    # `PyOpenGL AttributeError: 'NoneType' object has no attribute
    # 'glGetError'`), which reads identically to a broken suite in CI output.
    # `disabled` is what run_gate.py uses.
    env['MUJOCO_GL'] = 'disabled'

    os.makedirs(os.path.dirname(JUNIT), exist_ok=True)
    # --durations keeps the fast/slow split honest: a test that quietly drifts
    # past ~5 s shows up here, in the gate's own output, instead of silently
    # inflating the per-commit run until someone stops using it.
    cmd = [sys.executable, '-m', 'pytest', *targets, '-q',
           '--tb=short', '--durations=25', f'--junitxml={JUNIT}']
    if fast:
        cmd += ['-m', 'not slow']

    mode = 'FAST (per-commit)' if fast else 'FULL (pre-merge)'
    if targets != ['tests/']:
        mode += f' — NARROWED to {" ".join(targets)}, not a gate run'
    print(f'gate/run_suite.py — {mode}')
    print(f'  $ {" ".join(cmd[2:])}\n')

    proc = subprocess.run(cmd, env=env)

    if not os.path.exists(JUNIT):
        print('\nFAIL — pytest produced no junit report; it died before '
              'collection finished. Run the command above directly.')
        return 1

    suite = ET.parse(JUNIT).getroot()
    if suite.tag == 'testsuites':
        suite = suite.find('testsuite')

    n = int(suite.get('tests', 0))
    failures = int(suite.get('failures', 0))
    errors = int(suite.get('errors', 0))
    skipped = int(suite.get('skipped', 0))
    seconds = float(suite.get('time', 0.0))

    # junitxml records an unexpectedly-passing strict xfail as a failure, so it
    # is already inside `failures`. Name them anyway: "the open question closed
    # itself" needs different handling from "the code broke".
    bad, xpass, xfail = [], [], []
    for case in suite.iter('testcase'):
        nid = f"{case.get('classname')}::{case.get('name')}"
        # junit files an xfail under <skipped type="pytest.xfail">. Counting it
        # as a skip would read as "not run"; it is a KNOWN OPEN QUESTION that
        # ran and behaved as documented. Different thing, so report it apart.
        sk = case.find('skipped')
        if sk is not None and (sk.get('type') or '').endswith('xfail'):
            xfail.append(nid)
        for tag in ('failure', 'error'):
            el = case.find(tag)
            if el is not None:
                msg = (el.get('message') or '').strip().replace('\n', ' ')
                if 'XPASS' in msg or 'unexpectedly passing' in msg.lower():
                    xpass.append(nid)
                else:
                    bad.append((tag, nid, msg[:140]))

    passed = n - failures - errors - skipped
    true_skips = skipped - len(xfail)
    ok = failures == 0 and errors == 0

    print(f'\n{"=" * 72}')
    print(f'{mode}: {n} tests, {passed} passed, {failures} failed, '
          f'{errors} errors, {true_skips} skipped, {len(xfail)} xfail  '
          f'({seconds:.0f} s)')
    if xfail:
        print('\nxfail — documented open questions, behaving as documented:')
        for nid in xfail:
            print(f'  {nid}')
    if bad:
        print('\nFAILED / ERRORED:')
        for tag, nid, msg in bad:
            print(f'  [{tag}] {nid}\n         {msg}')
    if xpass:
        print('\nXPASS — a strict xfail started passing. Do NOT just delete '
              'the marker: find out what changed, then write it down.')
        for nid in xpass:
            print(f'  {nid}')
    print(f'\nVERDICT: {"PASS" if ok else "FAIL"}')
    if ok and fast:
        print('  (fast mode — run without --fast before merging)')
    print('=' * 72)

    with open(VERDICT, 'w', encoding='utf-8') as fh:
        json.dump({'mode': 'fast' if fast else 'full', 'tests': n,
                   'passed': passed, 'failed': failures, 'errors': errors,
                   'skipped': true_skips, 'seconds': round(seconds, 1),
                   'xfail': xfail, 'xpass': xpass,
                   'failed_ids': [nid for _, nid, _ in bad],
                   'verdict': 'PASS' if ok else 'FAIL'}, fh, indent=2)
    print(f'wrote {VERDICT}')

    return 0 if ok else (proc.returncode or 1)


if __name__ == '__main__':
    sys.exit(main())
