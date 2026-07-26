#!/usr/bin/env python3
"""Which public symbols actually execute on the canonical run?

Uses the coverage of gate/replay_canonical.py measured in CLEANUP-19
(crawlbot/ is unchanged since). Lets the documentation say "this is the
canonical path" and "this is an alternative mode" from evidence rather
than from the name of the function.
"""
import ast
import json
import os
import sys

cov = json.load(open('gate/_run/cov/cov.json'))
PKG = sys.argv[1] if len(sys.argv) > 1 else None

for root, dirs, files in os.walk('crawlbot'):
    dirs[:] = sorted(d for d in dirs if d != '__pycache__')
    if PKG and root.split('/')[-1] != PKG:
        continue
    for f in sorted(files):
        if not f.endswith('.py') or f == '__init__.py':
            continue
        p = os.path.join(root, f)
        entry = cov['files'].get(p)
        if entry is None:
            print(f'\n### {p}   NEVER IMPORTED on the canonical')
            continue
        ex = set(entry['executed_lines'])
        print(f'\n### {p}   {entry["summary"]["percent_covered"]:.0f}% covered')
        t = ast.parse(open(p).read())

        def report(n, prefix=''):
            body = [x for x in range(n.lineno, (n.end_lineno or n.lineno) + 1)]
            hits = len(ex & set(body))
            # a def line is always "executed" (definition); look at the body
            inner = [x for x in body if x > n.lineno]
            ihits = len(ex & set(inner))
            tag = 'LIVE  ' if ihits else 'unused'
            print(f'  {tag} {prefix}{n.name}  ({ihits}/{len(inner)} lines)')

        for n in t.body:
            if isinstance(n, ast.ClassDef):
                for m in n.body:
                    if isinstance(m, ast.FunctionDef) and not m.name.startswith('__'):
                        report(m, f'{n.name}.')
            elif isinstance(n, ast.FunctionDef) and not n.name.startswith('_'):
                report(n)
