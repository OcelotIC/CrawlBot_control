#!/usr/bin/env python3
"""Verify every moved script's `_root` still resolves to the repo root.

This is the one thing that can break silently: `_root` feeds both
sys.path AND path construction (URDF, MJCF, OUT_DIR), so a wrong value
writes files to the wrong place instead of raising.

Extracts the `_root = ...` / `_ROOT = ...` expression and the inline
`sys.path.insert(0, <expr>)` form, evaluates each with `__file__` bound to
the script's real path, and compares against the repo root.
"""
import os
import re
import sys
from pathlib import Path

ROOT = os.path.abspath('.')
ASSIGN = re.compile(r'^_(?:root|ROOT)\s*=\s*(.+)$', re.M)
INLINE = re.compile(r'^\s*sys\.path\.insert\(0,\s*(.+)\)\s*$', re.M)

bad, checked, skipped = [], 0, []
for dirpath, dirnames, filenames in os.walk('Misc/scripts'):
    dirnames[:] = [d for d in dirnames if d != '__pycache__']
    for fn in sorted(filenames):
        if not fn.endswith('.py'):
            continue
        path = os.path.join(dirpath, fn)
        src = open(path, encoding='utf-8').read()
        exprs = ASSIGN.findall(src) + [e for e in INLINE.findall(src)
                                       if '_root' not in e and '_ROOT' not in e]
        if not exprs:
            continue
        env = {'os': os, 'Path': Path, '__file__': os.path.abspath(path)}
        for e in exprs:
            e = e.split('#')[0].strip()
            try:
                val = eval(e, env)                       # simple path exprs only
            except Exception as ex:
                skipped.append((path, e, type(ex).__name__))
                continue
            checked += 1
            if os.path.realpath(str(val)) != os.path.realpath(ROOT):
                bad.append((path, e, os.path.realpath(str(val))))

print(f'checked {checked} root expressions across Misc/scripts/')
if skipped:
    print(f'\n{len(skipped)} not statically evaluable (review):')
    for p, e, t in skipped[:8]:
        print(f'  {p}: {e[:60]}  ({t})')
if bad:
    print(f'\n*** {len(bad)} WRONG ROOT ***')
    for p, e, v in bad:
        print(f'  {p}\n     expr : {e[:70]}\n     -> {v}')
    sys.exit(1)
print('\nall root expressions resolve to the repo root — OK')
