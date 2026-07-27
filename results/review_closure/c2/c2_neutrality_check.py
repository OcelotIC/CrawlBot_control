#!/usr/bin/env python3
"""C2 — control-neutrality check (the Gate-0 pattern, extended for appended columns).

`gate/run_gate.py`'s `diff_csv` requires EXACT header equality (`R[0] != B[0]`
-> FAIL "header mismatch"), so it cannot express the criterion this phase is
held to: *the 66 original columns byte-identical, new columns appended only*.
This checker implements exactly that criterion and nothing else. It does not
replace the gate — the gate is still run, unmodified, on the flag-off export.

Verdict is PASS iff, for the 66 baseline columns:
  * every baseline column name is present in the candidate, in the SAME
    POSITION (a moved column is a schema change, not an append);
  * the candidate adds columns only AFTER the 66th;
  * row counts match;
  * every field is byte-identical, EXCEPT the two wall-clock columns the gate
    itself excludes as nondeterministic instrumentation (gate/EXCEPTIONS.md).

Any field difference outside those two columns is a control-path leak and is
reported with its row, column and both values.

Usage:
  python3 results/review_closure/c2/c2_neutrality_check.py CANDIDATE.csv \\
      [--baseline results/j2_adjconv/c25_fulldiag.csv]
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
BASELINE = os.path.join(ROOT, 'results', 'j2_adjconv', 'c25_fulldiag.csv')
EXCLUDE = ['qp_time_ms', 'nmpc_time_ms']       # == gate/run_gate.py EXCLUDE_COLS


def check(candidate, baseline=BASELINE, exclude=EXCLUDE):
    with open(candidate, newline='') as fh:
        C = list(csv.reader(fh))
    with open(baseline, newline='') as fh:
        B = list(csv.reader(fh))
    if not C or not B:
        return {'verdict': 'FAIL', 'reason': 'empty csv'}

    chdr, bhdr = C[0], B[0]
    nb = len(bhdr)

    if len(chdr) < nb:
        return {'verdict': 'FAIL', 'reason': 'candidate has FEWER columns',
                'candidate_cols': len(chdr), 'baseline_cols': nb}
    if chdr[:nb] != bhdr:
        moved = [(i, bhdr[i], chdr[i])
                 for i in range(nb) if chdr[i] != bhdr[i]]
        return {'verdict': 'FAIL',
                'reason': 'a baseline column moved or was renamed',
                'first_mismatches': moved[:10]}
    if len(C) != len(B):
        return {'verdict': 'FAIL', 'reason': 'row count',
                'candidate_rows': len(C) - 1, 'baseline_rows': len(B) - 1}

    excl = {bhdr.index(c) for c in exclude if c in bhdr}
    ncmp = 0
    diffs = []
    for i in range(1, len(B)):
        if len(C[i]) < nb:
            return {'verdict': 'FAIL',
                    'reason': f'short row {i}: {len(C[i])} < {nb}'}
        for j in range(nb):
            if j in excl:
                continue
            ncmp += 1
            if C[i][j] != B[i][j]:
                diffs.append({'row': i, 'column': bhdr[j],
                              'candidate': C[i][j], 'baseline': B[i][j]})
                if len(diffs) >= 10:
                    break
        if diffs:
            break

    return {
        'verdict': 'FAIL' if diffs else 'PASS',
        'candidate': os.path.relpath(candidate, ROOT),
        'baseline': os.path.relpath(baseline, ROOT),
        'rows': len(B) - 1,
        'baseline_columns': nb,
        'candidate_columns': len(chdr),
        'appended_columns': chdr[nb:],
        'columns_excluded': sorted(excl and [bhdr[j] for j in excl] or []),
        'fields_compared': ncmp,
        'diffs': diffs,
    }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not args:
        print(__doc__)
        return 2
    baseline = BASELINE
    if '--baseline' in sys.argv:
        baseline = sys.argv[sys.argv.index('--baseline') + 1]
    res = check(args[0], baseline)
    print('=' * 74)
    print('C2 CONTROL-NEUTRALITY CHECK  (66 baseline columns, byte-identical)')
    print('=' * 74)
    print(f"  candidate : {res.get('candidate', args[0])}")
    print(f"  baseline  : {res.get('baseline', baseline)}")
    if res['verdict'] == 'PASS':
        print(f"  rows      : {res['rows']}")
        print(f"  columns   : {res['baseline_columns']} baseline "
              f"+ {len(res['appended_columns'])} appended "
              f"= {res['candidate_columns']}")
        print(f"  appended  : {res['appended_columns']}")
        print(f"  excluded  : {res['columns_excluded']} "
              f"(wall-clock, per gate/EXCEPTIONS.md)")
        print(f"  compared  : {res['fields_compared']} fields")
    else:
        print(f"  reason    : {res.get('reason', 'field mismatch')}")
        for d in res.get('diffs', [])[:10]:
            print(f"    row {d['row']:5d}  {d['column']:28s} "
                  f"candidate={d['candidate']}  baseline={d['baseline']}")
        for m in res.get('first_mismatches', [])[:10]:
            print(f"    position {m[0]}: baseline '{m[1]}' vs candidate '{m[2]}'")
    print()
    print(f"  VERDICT: {res['verdict']}")
    out = os.path.join(ROOT, 'results', 'review_closure', 'c2',
                       'c2_neutrality_verdict.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as fh:
        json.dump(res, fh, indent=2)
    print(f"  wrote {os.path.relpath(out, ROOT)}")
    return 0 if res['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
