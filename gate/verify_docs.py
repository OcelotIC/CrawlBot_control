#!/usr/bin/env python3
"""Verify every factual reference in docs/crawlbot/ actually resolves.

Documentation that cannot be checked is documentation that will rot. This
checks the two kinds of claim the new docs make:

  file:line   -> the file exists and has at least that many lines
  Class.method / bare symbol -> defined somewhere in crawlbot/
"""
import ast
import os
import re
import sys

DOCS = sorted(f'docs/crawlbot/{f}' for f in os.listdir('docs/crawlbot')
              if f.endswith('.md'))

# every symbol defined in crawlbot/
defined, qualified = set(), set()
for root, dirs, files in os.walk('crawlbot'):
    dirs[:] = [d for d in dirs if d != '__pycache__']
    for f in files:
        if not f.endswith('.py'):
            continue
        t = ast.parse(open(os.path.join(root, f)).read())
        for n in ast.walk(t):
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                defined.add(n.name)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                defined.add(n.id)
            elif isinstance(n, ast.arg):
                defined.add(n.arg)
            elif isinstance(n, ast.Attribute):
                defined.add(n.attr)
        for n in ast.walk(t):
            if isinstance(n, ast.ClassDef):
                for m in n.body:
                    if isinstance(m, ast.FunctionDef):
                        qualified.add(f'{n.name}.{m.name}')

LINEREF = re.compile(r'`?([A-Za-z0-9_./]+\.(?:py|md|json|csv|xml|urdf|sh))'
                     r':(\d+)(?:-(\d+))?`?')
# `Class.method` — but not `FILE.md` / `FILE.py`, which match the same shape
SYMREF = re.compile(r'`([A-Z][A-Za-z0-9_]*\.[a-z_][A-Za-z0-9_]*)`(?<!\.md`)(?<!\.py`)'
                    r'(?<!\.json`)(?<!\.csv`)(?<!\.sh`)(?<!\.xml`)')

bad = []
for d in DOCS:
    s = open(d, encoding='utf-8').read()
    for m in LINEREF.finditer(s):
        path, lo, hi = m.group(1), int(m.group(2)), m.group(3)
        # allow bare basenames only if they resolve from repo root
        if not os.path.exists(path):
            cands = [os.path.join(r, os.path.basename(path))
                     for r, _, fs in os.walk('.') if os.path.basename(path) in fs]
            if not cands:
                bad.append((d, m.group(0), 'file not found'))
                continue
            path = cands[0][2:]
        n = len(open(path, encoding='utf-8', errors='replace').read().split('\n'))
        top = int(hi) if hi else lo
        if top > n:
            bad.append((d, m.group(0), f'{path} has only {n} lines'))
    for m in SYMREF.finditer(s):
        q = m.group(1)
        cls, meth = q.split('.', 1)
        if q in qualified or meth in defined or cls in defined:
            continue
        bad.append((d, q, 'symbol not defined in crawlbot/'))

print(f'checked {len(DOCS)} documents')
if bad:
    print(f'\n*** {len(bad)} unresolved reference(s) ***')
    for d, ref, why in bad:
        print(f'  {d}: {ref}   -- {why}')
    sys.exit(1)
print('every file:line and symbol reference resolves — OK')
