#!/usr/bin/env python3
"""Post-migration link audit: which cited paths no longer resolve?

Scans tracked .md/.py for `results/X` and `Misc/runs/X` tokens and checks
whether each exists. Distinguishes:
  - BROKEN BY MIGRATION : existed before, cited path now wrong  (must fix)
  - PRE-EXISTING DANGLE : never existed at HEAD or in history   (report)
"""
import os
import re
import subprocess

files = [f for f in subprocess.run(['git', 'ls-files'], capture_output=True,
                                   text=True).stdout.split()
         if f.endswith(('.md', '.py'))]

pat = re.compile(r'\b((?:results|Misc/runs)/[A-Za-z0-9_]+)')
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
print(f'{len(cited)} distinct paths cited; {len(missing)} do not resolve\n')

for p in sorted(missing):
    name = p.split('/')[-1]
    alt = [c for c in (f'results/{name}', f'Misc/runs/{name}') if os.path.exists(c)]
    if alt:
        tag = f'BROKEN BY MIGRATION -> should be {alt[0]}'
    else:
        r = subprocess.run(['git', 'log', '--all', '--oneline',
                            '--diff-filter=A', '--', f'{p}/*', p],
                           capture_output=True, text=True).stdout.strip()
        tag = 'pre-existing dangle (never in history)' if not r \
            else 'deleted earlier in history'
    print(f'  {p:52s} {tag}')
    print(f'      cited by {len(cited[p])} file(s), e.g. {sorted(cited[p])[0]}')
