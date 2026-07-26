#!/usr/bin/env python3
"""Static import closure of the canonical run.

Answers the only question that matters for restructuring `scripts/`:
which modules does the gate's canonical replay actually pull in?
Everything else in scripts/ is, by definition, not on the canonical path.

Walks `import`/`from ... import` statements from a set of roots, following
only first-party modules (crawlbot, scripts, lutze_baseline, gate).
"""
import ast
import os
import sys

ROOTS = ['gate/replay_canonical.py', 'gate/run_gate.py',
         'scripts/diag_full_diag_export.py', 'scripts/render_traversal.py']
FIRST_PARTY = ('crawlbot', 'scripts', 'lutze_baseline', 'gate', 'benchmarks')


def mod_to_path(m):
    p = m.replace('.', '/')
    for cand in (p + '.py', p + '/__init__.py'):
        if os.path.exists(cand):
            return cand
    return None


def path_to_mod(p):
    return p[:-3].replace('/', '.').removesuffix('.__init__')


seen, queue = set(), list(ROOTS)
edges = []
while queue:
    path = queue.pop()
    if path in seen or not os.path.exists(path):
        continue
    seen.add(path)
    try:
        tree = ast.parse(open(path).read())
    except SyntaxError:
        continue
    here = path_to_mod(path)
    for n in ast.walk(tree):
        targets = []
        if isinstance(n, ast.Import):
            targets = [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            if n.level:                       # relative import
                base = here.rsplit('.', n.level)[0]
                targets = [f'{base}.{n.module}' if n.module else base]
            elif n.module:
                targets = [n.module]
                # `from pkg import mod` where mod is a submodule
                targets += [f'{n.module}.{a.name}' for a in n.names]
        for t in targets:
            if not t.startswith(FIRST_PARTY):
                continue
            np_ = mod_to_path(t)
            if np_:
                edges.append((path, np_))
                queue.append(np_)

by_pkg = {}
for p in sorted(seen):
    by_pkg.setdefault(p.split('/')[0], []).append(p)

print(f'CANONICAL IMPORT CLOSURE — {len(seen)} first-party modules\n')
for pkg, files in sorted(by_pkg.items()):
    print(f'{pkg}/  ({len(files)})')
    for f in files:
        print(f'    {f}')
    print()

for pkg, glob in [('scripts', 'scripts'), ('crawlbot', 'crawlbot')]:
    allf = set()
    for root, _, fs in os.walk(glob):
        if '__pycache__' in root:
            continue
        allf |= {os.path.join(root, f) for f in fs if f.endswith('.py')}
    on = allf & seen
    print(f'{glob}/: {len(on)} of {len(allf)} .py files on the canonical path '
          f'({100*len(on)//max(len(allf),1)}%)')
