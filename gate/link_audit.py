#!/usr/bin/env python3
"""Link audit: which repo-relative paths cited in tracked text do not exist?

Run after any restructuring. Classifies each unresolved citation so a
migration's own breakage is separable from pre-existing rot:

  BROKEN BY MOVE  the same basename exists elsewhere -> the citation is stale
  DELETED         existed once, removed in an earlier commit
  DANGLING        no such file anywhere, and none in history

Exits non-zero if anything is BROKEN BY MOVE.

    PYTHONPATH=. python3 gate/link_audit.py
"""
import os
import re
import subprocess
import sys

TOP = ('crawlbot', 'gate', 'scripts', 'tests', 'models', 'docs', 'results',
       'Misc', 'scenarios', 'benchmarks', 'lutze_baseline')

pat = re.compile(r'\b((?:' + '|'.join(TOP) + r')/[A-Za-z0-9_./-]*[A-Za-z0-9_])')

files = [f for f in subprocess.run(['git', 'ls-files'], capture_output=True,
                                   text=True).stdout.split()
         if f.endswith(('.md', '.py'))]

cited = {}
for f in files:
    if not os.path.exists(f):
        continue
    try:
        s = open(f, encoding='utf-8').read()
    except UnicodeDecodeError:
        continue
    for m in set(pat.findall(s)):
        cited.setdefault(m, set()).add(f)

missing = {p: v for p, v in cited.items() if not os.path.exists(p)}

# Two classes are not rot and must not be reported as such:
#  - gitignored paths (scratch dirs created at run time, e.g.
#    results/test_scratch/... and results/gate_run_scratch/...)
#  - prose ellipses ("results/.../foo.md") caught by the tokenizer
if missing:
    ign = subprocess.run(['git', 'check-ignore'] + sorted(missing),
                         capture_output=True, text=True).stdout.split()
    missing = {p: v for p, v in missing.items()
               if p not in set(ign) and '...' not in p}
print(f'{len(cited)} distinct paths cited in {len(files)} tracked files; '
      f'{len(missing)} do not resolve\n')

# index every real basename, to recognise "this moved" vs "this never existed"
index = {}
for root, dirs, fs in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__')]
    for f in fs:
        index.setdefault(f, []).append(os.path.join(root, f)[2:])
    for d in dirs:
        index.setdefault(d, []).append(os.path.join(root, d)[2:])

buckets = {'BROKEN BY MOVE': [], 'DELETED': [], 'DANGLING': []}
for p in sorted(missing):
    base = p.rstrip('/').split('/')[-1]
    elsewhere = [c for c in index.get(base, []) if c != p]
    # "moved" requires an UNAMBIGUOUS relocation: exactly one candidate
    # elsewhere, and the cited parent directory gone too. Without the
    # uniqueness test, every citation of a common basename (sim_log.json
    # lives in dozens of run dirs, and most are gitignored) reads as
    # broken -- 35 false alarms, which makes the check worthless.
    parent_gone = not os.path.isdir(os.path.dirname(p))
    if len(elsewhere) == 1 and parent_gone:
        buckets['BROKEN BY MOVE'].append((p, elsewhere[0]))
        continue
    hist = subprocess.run(['git', 'log', '--all', '--oneline', '-1',
                           '--diff-filter=A', '--', p, f'{p}/*'],
                          capture_output=True, text=True).stdout.strip()
    buckets['DELETED' if hist else 'DANGLING'].append((p, None))

for k in ('BROKEN BY MOVE', 'DELETED', 'DANGLING'):
    v = buckets[k]
    cap = 20 if k == 'BROKEN BY MOVE' else 6
    print(f'{k}: {len(v)}')
    for p, alt in v[:cap]:
        print(f'   {p}' + (f'   -> {alt}' if alt else '')
              + f'   [{len(cited[p])} citing file(s)]')
    if len(v) > cap:
        print(f'   ... and {len(v) - cap} more')
    print()

sys.exit(1 if buckets['BROKEN BY MOVE'] else 0)
