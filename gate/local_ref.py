#!/usr/bin/env python3
"""Local bit-identity reference for refactors — stricter than run_gate.py.

Why this exists. ``run_gate.py`` compares the fulldiag CSV against the
committed baseline, and that CSV is written at 7 significant digits: it is a
rounding-tolerant comparison, and the committed baseline does not reproduce on
every host (a 3.11.15 -> 3.11.16 interpreter drift flips ``tauw_x_Nm`` at the
1e-15 level). For a no-behaviour-change refactor we want the stronger claim:
the whole replay directory is bit-identical to a reference produced ON THIS
HOST from the pre-refactor code.

What is compared (every file the replay writes, plus the fulldiag export):
  * ``*.json`` — parsed and walked; floats compared by ``float.hex()`` (so
    -0.0 != 0.0 and NaN == NaN bit-for-bit), everything else by ``==``.
  * ``*.csv``  — every field compared as text.
  * anything else — SHA-256 of the bytes.
Excluded: wall-clock instrumentation only — JSON keys / CSV columns whose name
ends in ``_ms`` (qp_time_ms, nmpc_time_ms, solve_ms, time_ms). The excluded
names are printed, so an exclusion can never hide silently.

Usage:
    PYTHONPATH=. python3 gate/local_ref.py run <tag>          # replay -> gate/_run/local_ref/<tag>/
    PYTHONPATH=. python3 gate/local_ref.py diff <tagA> <tagB> # compare two runs
    PYTHONPATH=. python3 gate/local_ref.py freeze <tag>       # <tag> becomes 'ref'
    PYTHONPATH=. python3 gate/local_ref.py check              # run 'cand', diff ref cand
Exit 0 on identity, 1 otherwise.
"""
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
BASE = 'gate/_run/local_ref'                 # git-ignored (gate/_run/)
REPLAY_DIR = 'results/gate_run_scratch'


def _is_timing(name):
    return isinstance(name, str) and name.endswith('_ms')


# Non-physical fields beyond the *_ms rule, each named and printed on use:
#   c25_fulldiag_meta.json run/csv — the export's own output path (differs by
#     construction between two tags);
#   nmpc_per_step.txt t_max/t_mean — wall-clock solve times, a summary of
#     nmpc_step_log.json whose time_ms is already excluded (its iteration
#     counts and statuses ARE compared).
PATH_KEYS = {'c25_fulldiag_meta.json': {'run', 'csv'}}
TABLE_SKIP = {'nmpc_per_step.txt': {'t_max', 't_mean'}}


def _sh(argv, timeout):
    env = dict(os.environ, MUJOCO_GL='disabled', PYTHONPATH='.')
    p = subprocess.run([sys.executable] + argv, cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        sys.stderr.write(p.stdout[-2000:] + p.stderr[-2000:])
        raise SystemExit(f'{argv[0]} exited {p.returncode}')


def run(tag):
    out = f'{BASE}/{tag}'
    if os.path.exists(out):
        shutil.rmtree(out)
    # Empty the scratch dir first: replay_canonical.py deletes only
    # sim_log.json, so any file this replay fails to rewrite would otherwise
    # be copied stale — and compare equal to an equally stale reference.
    if os.path.exists(REPLAY_DIR):
        shutil.rmtree(REPLAY_DIR)
    t0 = time.time()
    _sh(['gate/replay_canonical.py'], 3600)
    stale = [p for p in os.listdir(REPLAY_DIR)
             if os.path.getmtime(f'{REPLAY_DIR}/{p}') < t0]
    if stale:
        raise SystemExit(f'files older than this replay: {stale}')
    log = f'{REPLAY_DIR}/sim_log.json'
    if not os.path.exists(log) or os.path.getmtime(log) < t0:
        raise SystemExit('replay produced no fresh sim_log.json')
    shutil.copytree(REPLAY_DIR, out)
    _sh(['scripts/diag_full_diag_export.py', '--run-dir', out,
         '--out-prefix', f'{out}/c25'], 600)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                            text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', 'crawlbot/'],
                           capture_output=True, text=True).stdout.strip()
    with open(f'{out}/_manifest.json', 'w') as f:
        json.dump({'commit': commit, 'crawlbot_dirty': bool(dirty),
                   'python': sys.version.split()[0],
                   'seconds': round(time.time() - t0, 1)}, f, indent=1)
    print(f'[local_ref] {tag}: {time.time() - t0:.0f}s  commit {commit[:10]}'
          f'{" (crawlbot/ dirty)" if dirty else ""}')


def _walk(a, b, path, st):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return f'{path}: key sets differ {sorted(set(a) ^ set(b))[:5]}'
        for k in sorted(a):
            if _is_timing(k) or (path in PATH_KEYS and k in PATH_KEYS[path]):
                st['excluded'].add(f'{path}:{k}' if path in PATH_KEYS else k)
                continue
            r = _walk(a[k], b[k], f'{path}.{k}', st)
            if r:
                return r
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f'{path}: length {len(a)} vs {len(b)}'
        for i, (x, y) in enumerate(zip(a, b)):
            r = _walk(x, y, f'{path}[{i}]', st)
            if r:
                return r
        return None
    if isinstance(a, float) and isinstance(b, float):
        st['floats'] += 1
        if a.hex() != b.hex() and not (math.isnan(a) and math.isnan(b)):
            return f'{path}: {a!r} vs {b!r}'
        return None
    st['other'] += 1
    if type(a) is not type(b) or a != b:
        return f'{path}: {a!r} vs {b!r}'
    return None


def _cmp_csv(pa, pb, st):
    with open(pa, newline='') as f:
        A = list(csv.reader(f))
    with open(pb, newline='') as f:
        B = list(csv.reader(f))
    if not A or A[0] != B[0] or len(A) != len(B):
        return 'header or row count differs'
    skip = {j for j, h in enumerate(A[0]) if _is_timing(h)}
    st['excluded'].update(A[0][j] for j in skip)
    for i in range(1, len(A)):
        for j, (x, y) in enumerate(zip(A[i], B[i])):
            if j in skip:
                continue
            st['fields'] += 1
            if x != y:
                return f'row {i} col {A[0][j]}: {x} vs {y}'
    return None


def _cmp_table(pa, pb, skip, name, st):
    """Whitespace table with a header row; columns in ``skip`` ignored."""
    A = open(pa).read().splitlines()
    B = open(pb).read().splitlines()
    if len(A) != len(B) or A[0] != B[0]:
        return 'header or row count differs'
    hdr = A[0].split()
    idx = {hdr.index(c) for c in skip}
    st['excluded'].update(f'{name}:{c}' for c in skip)
    for i, (x, y) in enumerate(zip(A, B)):
        fx, fy = x.split(), y.split()
        if len(fx) != len(fy):
            return f'line {i}: {x!r} vs {y!r}'
        for j, (u, w) in enumerate(zip(fx, fy)):
            if j in idx and i > 1:
                continue
            if u != w:
                return f'line {i} field {j}: {u} vs {w}'
    return None


def diff(ta, tb):
    da, db = f'{BASE}/{ta}', f'{BASE}/{tb}'
    fa = sorted(p for p in os.listdir(da) if p != '_manifest.json')
    fb = sorted(p for p in os.listdir(db) if p != '_manifest.json')
    if fa != fb:
        print(f'FAIL file sets differ: {sorted(set(fa) ^ set(fb))}')
        return 1
    st = {'floats': 0, 'other': 0, 'fields': 0, 'excluded': set()}
    bad = 0
    for name in fa:
        pa, pb = f'{da}/{name}', f'{db}/{name}'
        if os.path.isdir(pa):
            print(f'  ?? {name}: subdirectory not compared')
            bad += 1
            continue
        if name.endswith('.json'):
            r = _walk(json.load(open(pa)), json.load(open(pb)), name, st)
        elif name.endswith('.csv'):
            r = _cmp_csv(pa, pb, st)
        elif name in TABLE_SKIP:
            r = _cmp_table(pa, pb, TABLE_SKIP[name], name, st)
        else:
            ha = hashlib.sha256(open(pa, 'rb').read()).hexdigest()
            hb = hashlib.sha256(open(pb, 'rb').read()).hexdigest()
            r = None if ha == hb else f'sha256 {ha[:12]} vs {hb[:12]}'
        print(f'  {"OK  " if r is None else "DIFF"} {name}' + ('' if r is None else f'  -> {r}'))
        bad += r is not None
    print(f'  compared: {st["floats"]} floats (bit-exact), {st["other"]} other '
          f'JSON leaves, {st["fields"]} CSV fields; {len(fa)} files')
    print(f'  excluded (wall-clock): {sorted(st["excluded"])}')
    print(f'{"PASS" if bad == 0 else "FAIL"}: {ta} vs {tb}')
    return 0 if bad == 0 else 1


def freeze(tag):
    if os.path.exists(f'{BASE}/ref'):
        shutil.rmtree(f'{BASE}/ref')
    shutil.copytree(f'{BASE}/{tag}', f'{BASE}/ref')
    print(f'[local_ref] {tag} frozen as ref')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'run':
        run(sys.argv[2])
    elif cmd == 'diff':
        sys.exit(diff(sys.argv[2], sys.argv[3]))
    elif cmd == 'freeze':
        freeze(sys.argv[2])
    elif cmd == 'check':
        run('cand')
        sys.exit(diff('ref', 'cand'))
    else:
        print(__doc__)
        sys.exit(2)
