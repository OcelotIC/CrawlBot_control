#!/usr/bin/env python3
"""Verify CLAUDE.md's "Key Parameters" table against the code.

CLAUDE.md is the single source of truth for canonical parameter values, and
every row cites a `file.py:LINE`. Nothing checked those citations, so when the
chantier shrank `config.py` (610 -> 507) and `wholebody_qp.py` (1385 -> 950),
**8 of 14 references silently drifted by 20-50 lines** while the values stayed
correct. This closes that gap.

For each row carrying a `file.py:LINE` reference it checks:

  1. the file exists and is long enough;
  2. the cited line declares the parameter the row names;
  3. the value in the table matches the value in the code, compared
     NUMERICALLY so `2e3` in the source satisfies `2000` in the table.

    PYTHONPATH=. python3 gate/verify_params.py

Exits non-zero on any mismatch. Run it whenever a canonical value or a line
number could have moved -- i.e. as part of the Rule 15 routine.
"""
import glob
import re
import sys

CLAUDE = 'CLAUDE.md'
ROW = re.compile(r'^\|(?P<name>[^|]+)\|(?P<value>[^|]+)\|(?P<unit>[^|]*)\|(?P<ref>[^|]+)\|\s*$',
                 re.M)
REF = re.compile(r'`([A-Za-z0-9_/]+\.py):(\d+)(?:-(\d+))?`')
NUM = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')

# table label -> the identifier actually declared in the code
IDENT = {
    'tau_w_max': 'tau_w_max', 'tau_max': 'tau_max', 'hw_max': 'hw_max',
    'dt_nmpc': 'dt_nmpc', 'dt_qp': 'dt_qp', 'weight_ratio': 'weight_ratio',
    'α torso-pose': 'alpha_torso_pose', 'α swing-EE': 'ss_alpha_ee',
    'α momentum (T-MOM)': 'ss_alpha_mom', 'w hw-slack': 'w_hw_slack',
    'α posture': 'ss_alpha_posture', 'α wrench-track': 'ss_alpha_wrench',
    'ε (Tikhonov)': 'regularization',
}


def clean(s):
    return re.sub(r'[*`]', '', s).strip()


def nums(s):
    return [float(x) for x in NUM.findall(s.replace(',', ''))]


rows = ROW.findall(open(CLAUDE, encoding='utf-8').read())
checked, bad = 0, []

for name, value, _unit, ref in rows:
    name, value = clean(name), clean(value)
    m = REF.search(ref)
    if not m:
        continue
    f, lo = m.group(1), int(m.group(2))
    hi = int(m.group(3)) if m.group(3) else lo

    cands = ([f] if '/' in f else
             glob.glob(f'crawlbot/**/{f}', recursive=True)
             + glob.glob(f'gate/{f}') + glob.glob(f'scripts/{f}'))
    if not cands:
        bad.append((name, f'{f}: file not found'))
        continue
    path = cands[0]
    src = open(path, encoding='utf-8').read().split('\n')
    if hi > len(src):
        bad.append((name, f'{path}:{hi} beyond EOF ({len(src)} lines)'))
        continue

    seg = '\n'.join(src[lo - 1:hi])
    checked += 1

    ident = IDENT.get(name)
    if ident and not re.search(rf'\b{re.escape(ident)}\b', seg):
        where = [i + 1 for i, l in enumerate(src)
                 if re.match(rf'\s*{re.escape(ident)}\s*[:=]', l)]
        bad.append((name, f'{path}:{lo}-{hi} does not declare `{ident}`'
                          + (f' — it is at :{where[0]}' if where else '')))
        continue

    want, got = nums(value), nums(seg)
    if want and not any(abs(want[0] - g) <= 1e-12 * max(1.0, abs(want[0]))
                        for g in got):
        bad.append((name, f'value {want[0]:g} not found at {path}:{lo}-{hi} '
                          f'(found {got[:4]})'))

print(f'checked {checked} CLAUDE.md parameter row(s) with a file:line reference')
if bad:
    print(f'\n*** {len(bad)} mismatch(es) ***')
    for n, w in bad:
        print(f'  {n:22s} {w}')
    sys.exit(1)
print('every cited line declares its parameter, and every value matches — OK')
