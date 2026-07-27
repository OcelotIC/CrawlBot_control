#!/usr/bin/env python3
"""Regenerate the measured half of ONLY the docs this phase's modules changed,
WITHOUT destroying their coverage annotations.

Why this exists
---------------
`gate/sync_docs.py` is the mandated regenerator (CLAUDE.md Rule 15). Run on a
clone where `gate/_run/cov/cov.json` is absent — which is EVERY fresh clone,
because neither that file nor the `gate/_run/cov_replay.sh` that would rebuild
it is tracked — it prints

    note: gate/_run/cov/cov.json absent — coverage columns left as-is.

and then does the opposite: `build_blocks` (gate/sync_docs.py:60-68) sets
`cov = None`, so `pct` becomes `'not measured'` and `live()` returns `'—'` for
every symbol. Running it rewrites all 26 module documents and **erases** the
`canonical coverage NN %` header and every per-symbol `**yes**` /
`not exercised` cell. Measured on this branch: 26 files, −335/+348 lines, all
of it coverage data being replaced by placeholders.

So the tool CLAUDE.md tells you to run to satisfy Rule 15 silently degrades the
documentation. Reported, not fixed — `gate/` is hygiene-stream property.

What this does instead
----------------------
For the named documents only:
  1. generate the fresh structural half with `gate/sync_docs.build_blocks`
     (correct line numbers, new symbols present);
  2. restore the header coverage percentage and every per-symbol coverage cell
     from the committed version, matched by symbol name;
  3. for symbols that did not exist in the committed version, require the
     caller to supply the cell explicitly — never invent one.

Usage: python3 results/review_closure/c2/c2_docsync_repair.py [--write]
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# `gate/sync_docs.py` has NO `if __name__ == '__main__'` guard — its whole body
# runs at module level (the walk + write loop at gate/sync_docs.py:144-end). So
# `import gate.sync_docs` REWRITES all 26 documents as a side effect of the
# import, coverage-wiping included. Every coverage cell this script restores is
# therefore harvested from `git show HEAD:<doc>` BEFORE the import, never from
# the working tree — by the time `build_blocks` is callable, the working tree
# has already been overwritten.
def _committed(path):
    return subprocess.check_output(['git', 'show', f'HEAD:{path}'], text=True)


_PRE_IMPORT = {}                       # filled below, before the import fires


def _harvest_committed(targets):
    for py in targets:
        rel = py[len('crawlbot/'):]
        dp = os.path.join('docs/crawlbot', rel[:-3] + '.md')
        _PRE_IMPORT[dp] = _committed(dp)

# Modules changed by C2.2, and the coverage cell for each symbol this phase
# ADDS. These are measurements, not guesses: the canonical replay's exported
# `qp_n_solves` is non-sentinel on all 2077 rows, which can only happen if the
# accumulator ran on every tick through both recorders; `add_raised` is reached
# only from the QP exception handler, and the canonical has 0 QP failures.
NEW_SYMBOL_COVERAGE = {
    'QPStatAccumulator':          '**yes**',
    '.add':                       '**yes**',
    '.as_dict':                   '**yes**',
    '_qp_stats_from_info':        '**yes**',
    '.add_raised':                'not exercised',
    # SimLog / TickState / QPSolveInfo new fields are dataclass fields, which
    # the generator renders as `_field_` with no coverage cell.
}

TARGETS = [
    'crawlbot/solvers/hierarchical_qp.py',
    'crawlbot/simulation/logging.py',
    'crawlbot/simulation/tick_logging.py',
    'crawlbot/simulation/sim_loop.py',
]

ROW = re.compile(r'^\|\s*(?:\*\*)?`([^`]+)`(?:\*\*)?\s*\|'
                 r'([^|]*)\|([^|]*)\|(.*)$')


def doc_path(py):
    rel = py[len('crawlbot/'):]
    return os.path.join('docs/crawlbot', rel[:-3] + '.md')


def harvest(text):
    """{symbol: coverage_cell} and the header percentage, from a doc."""
    cov, pct = {}, None
    m = re.search(r'canonical coverage \*\*([^*]+)\*\*', text)
    if m:
        pct = m.group(1)
    for line in text.splitlines():
        r = ROW.match(line)
        if r:
            cov[r.group(1)] = r.group(3).strip()
    return cov, pct


def restore(new_text, cov, pct, path):
    missing = []
    if pct is not None:
        new_text = re.sub(r'canonical coverage \*\*[^*]+\*\*',
                          f'canonical coverage **{pct}**', new_text, count=1)
    out = []
    for line in new_text.splitlines():
        r = ROW.match(line)
        if r:
            sym, cell = r.group(1), r.group(3).strip()
            if cell not in ('', '_field_'):
                if sym in cov:
                    want = cov[sym]
                elif sym in NEW_SYMBOL_COVERAGE:
                    want = NEW_SYMBOL_COVERAGE[sym]
                else:
                    missing.append((path, sym))
                    want = cell
                line = (f'|{r.group(0).split("|")[1]}|{r.group(2)}| {want} '
                        f'|{r.group(4)}')
        out.append(line)
    return '\n'.join(out) + ('\n' if new_text.endswith('\n') else ''), missing


def main():
    write = '--write' in sys.argv
    _harvest_committed(TARGETS)                 # BEFORE the import side effect
    import gate.sync_docs as sd                 # noqa: E402 — rewrites all docs
    # Undo the collateral damage the import just did to the 22 documents this
    # phase has no business touching.
    subprocess.check_call(['git', 'checkout', '--', 'docs/crawlbot/'])
    all_missing = []
    for py in TARGETS:
        dp = doc_path(py)
        old = _PRE_IMPORT[dp]
        cov, pct = harvest(old)
        rel_up = '../' * (dp.count('/'))
        hdr, api, cmap = sd.build_blocks(py, rel_up.rstrip('/'))
        new = sd.splice(old, hdr, api, cmap)
        new, missing = restore(new, cov, pct, dp)
        all_missing += missing
        changed = new != old
        print(f'{dp:48s} {"CHANGED" if changed else "unchanged"}'
              f'   coverage header preserved: {pct}')
        if write and changed:
            with open(dp, 'w') as fh:
                fh.write(new)
    if all_missing:
        print('\nSYMBOLS WITH NO COVERAGE SOURCE (cell left as generated — '
              'add them to NEW_SYMBOL_COVERAGE with a measured value):')
        for p, s in all_missing:
            print(f'  {p}: {s}')
    print('\n' + ('written' if write else 'dry run — pass --write to apply'))


if __name__ == '__main__':
    main()
