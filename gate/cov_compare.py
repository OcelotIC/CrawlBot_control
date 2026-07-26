#!/usr/bin/env python3
"""What does the test suite cover that the canonical replay does not?

The gate proves the CANONICAL RUN reproduces byte-identically. That is a
very strong regression check on one config, and says nothing about lines
that config never executes. This quantifies the difference, per module:

  both       covered by the canonical AND by the suite
  gate-only  covered only by the canonical replay
  TESTS-ONLY covered only by pytest  <- the suite's unique contribution
  neither    covered by nothing      <- genuinely unverified
"""
import json
import os
import sys

A = 'gate/_run/cov/cov.json'          # canonical replay
B = 'gate/_run/cov_tests/cov.json'    # pytest suite
for p in (A, B):
    if not os.path.exists(p):
        sys.exit(f'missing {p}')

ca, cb = json.load(open(A)), json.load(open(B))
mods = sorted(set(ca['files']) | set(cb['files']))

tot = {'both': 0, 'gate': 0, 'tests': 0, 'neither': 0}
rows = []
for m in mods:
    ea, eb = ca['files'].get(m), cb['files'].get(m)
    stmts = set()
    for e in (ea, eb):
        if e:
            stmts |= set(e['executed_lines']) | set(e['missing_lines'])
    xa = set(ea['executed_lines']) if ea else set()
    xb = set(eb['executed_lines']) if eb else set()
    both, only_a, only_b = xa & xb, xa - xb, xb - xa
    none = stmts - xa - xb
    rows.append((m, len(stmts), len(both), len(only_a), len(only_b), len(none)))
    tot['both'] += len(both)
    tot['gate'] += len(only_a)
    tot['tests'] += len(only_b)
    tot['neither'] += len(none)

w = max(len(r[0]) for r in rows)
print(f'{"module":{w}}  stmts   both  gate-only  TESTS-ONLY  neither')
for m, s, b, a, t, n in sorted(rows, key=lambda r: -r[4]):
    print(f'{m:{w}}  {s:5d}  {b:5d}  {a:9d}  {t:10d}  {n:7d}')

T = sum(tot.values())
print(f'\n{"TOTAL":{w}}  {T:5d}  {tot["both"]:5d}  {tot["gate"]:9d}  '
      f'{tot["tests"]:10d}  {tot["neither"]:7d}')
print(f'\nlines the SUITE covers that the canonical never executes: '
      f'{tot["tests"]}  ({100 * tot["tests"] // max(T, 1)} % of all statements)')
print(f'lines covered by NEITHER: {tot["neither"]} '
      f'({100 * tot["neither"] // max(T, 1)} %)')
