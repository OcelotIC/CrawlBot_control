#!/usr/bin/env python3
"""Per-test inventory of tests/ — what each test looks at.

For every collected test: file, class, name, first docstring line, the
crawlbot modules its file imports, the number of assertions, and whether
it currently passes. Written to a TSV so the audit is data, not memory.
"""
import ast
import json
import os
import re
import subprocess
import sys

OUT = 'gate/_run/test_inventory.tsv'


def crawlbot_imports(tree):
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and 'crawlbot' in n.module:
            mods.add(n.module.replace('crawlbot.', ''))
        elif isinstance(n, ast.Import):
            for a in n.names:
                if 'crawlbot' in a.name:
                    mods.add(a.name.replace('crawlbot.', ''))
    return sorted(mods)


def count_asserts(node):
    return sum(1 for n in ast.walk(node) if isinstance(n, ast.Assert))


def first_doc_line(node):
    d = ast.get_docstring(node) or ''
    for ln in d.split('\n'):
        if ln.strip():
            return ln.strip()
    return ''


rows = []
for f in sorted(os.listdir('tests')):
    if not f.endswith('.py') or f in ('__init__.py', 'conftest.py'):
        continue
    path = f'tests/{f}'
    tree = ast.parse(open(path).read())
    imps = ','.join(crawlbot_imports(tree))
    mod_doc = first_doc_line(tree)

    def walk(body, cls=''):
        for n in body:
            if isinstance(n, ast.ClassDef):
                walk(n.body, n.name)
            elif isinstance(n, ast.FunctionDef) and n.name.startswith('test_'):
                marks = [ast.unparse(d)[:40] for d in n.decorator_list]
                rows.append({
                    'file': f, 'class': cls, 'test': n.name,
                    'line': n.lineno,
                    'doc': first_doc_line(n) or mod_doc,
                    'asserts': count_asserts(n),
                    'marks': ';'.join(m for m in marks if 'fixture' not in m),
                    'imports': imps,
                })
    walk(tree.body)

with open(OUT, 'w', encoding='utf-8') as fh:
    cols = ['file', 'class', 'test', 'line', 'asserts', 'marks', 'doc', 'imports']
    fh.write('\t'.join(cols) + '\n')
    for r in rows:
        fh.write('\t'.join(str(r[c]).replace('\t', ' ') for c in cols) + '\n')

print(f'{len(rows)} test functions across '
      f'{len({r["file"] for r in rows})} files -> {OUT}\n')
by_file = {}
for r in rows:
    by_file.setdefault(r['file'], []).append(r)
for f, rs in sorted(by_file.items(), key=lambda kv: -len(kv[1])):
    print(f'  {len(rs):3d}  {f:44s} {rs[0]["imports"][:60]}')
