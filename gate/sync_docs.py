#!/usr/bin/env python3
"""Regenerate the measured blocks of docs/crawlbot/ from the code.

Run this after ANY change to crawlbot/. It rewrites, in place:

  - the header line (line count, canonical coverage)
  - the "## Public API" table, with a line-anchored link per symbol
  - the "## Code map" table, linking each class/function to its span

and leaves every hand-written prose section untouched. That is the whole
point: the parts that can be derived from the code are derived, so they
cannot drift, and the parts that require judgement are preserved.

    PYTHONPATH=. python3 gate/sync_docs.py            # rewrite
    PYTHONPATH=. python3 gate/sync_docs.py --check    # fail if out of date

`--check` is what makes updating the docs a routine rather than a wish:
it exits non-zero when a symbol was added, removed or moved without the
document following.
"""
import ast
import json
import os
import sys

DOCS = 'docs/crawlbot'
COV = 'gate/_run/cov/cov.json'
CHECK = '--check' in sys.argv


def load_cov():
    if not os.path.exists(COV):
        print(f'note: {COV} absent — coverage columns left as-is.\n'
              '      regenerate with gate/_run/cov_replay.sh', file=sys.stderr)
        return None
    return json.load(open(COV))


def sig(n):
    a = n.args
    parts, defaults = [], [None] * (len(a.args) - len(a.defaults)) + list(a.defaults)
    for arg, d in zip(a.args, defaults):
        if arg.arg == 'self':
            continue
        parts.append(arg.arg + (f'={ast.unparse(d)}' if d is not None else ''))
    if a.vararg:
        parts.append('*' + a.vararg.arg)
    parts += [x.arg for x in a.kwonlyargs]
    if a.kwarg:
        parts.append('**' + a.kwarg.arg)
    s = ', '.join(parts)
    return s if len(s) <= 58 else s[:55] + '...'


def build_blocks(py_path, rel_up):
    """Return (header_line, api_table, code_map) for one module."""
    src = open(py_path).read()
    tree = ast.parse(src)
    nlines = len(src.split('\n'))
    cov = load_cov.cache if hasattr(load_cov, 'cache') else None
    entry = cov['files'].get(py_path) if cov else None
    pct = f"{entry['summary']['percent_covered']:.0f} %" if entry else 'not measured'
    ex = set(entry['executed_lines']) if entry else set()
    link = f'{rel_up}/{py_path}'

    def live(n):
        if not ex:
            return '—'
        inner = range(n.lineno + 1, (n.end_lineno or n.lineno) + 1)
        return '**yes**' if (ex & set(inner)) else 'not exercised'

    api, cmap = [], []
    for n in tree.body:
        if isinstance(n, ast.ClassDef):
            deco = ' *(dataclass)*' if any('dataclass' in ast.unparse(d)
                                           for d in n.decorator_list) else ''
            api.append(f'| **`{n.name}`**{deco} |  |  | [L{n.lineno}]({link}#L{n.lineno}) |')
            cmap.append(f'| `class {n.name}` | [L{n.lineno}-{n.end_lineno}]'
                        f'({link}#L{n.lineno}-L{n.end_lineno}) |')
            for m in n.body:
                if isinstance(m, ast.FunctionDef) and not m.name.startswith('__'):
                    api.append(f'| `.{m.name}` | `({sig(m)})` | {live(m)} | '
                               f'[L{m.lineno}]({link}#L{m.lineno}) |')
                    cmap.append(f'| `{n.name}.{m.name}` | [L{m.lineno}-{m.end_lineno}]'
                                f'({link}#L{m.lineno}-L{m.end_lineno}) |')
                elif isinstance(m, ast.AnnAssign) and isinstance(m.target, ast.Name):
                    v = ast.unparse(m.value) if m.value else ''
                    pre = 'field(default_factory=lambda: '
                    if v.startswith(pre):
                        v = v[len(pre):-1] if v.endswith(')') else v[len(pre):]
                    api.append(f'|   `{m.target.id}` | `{v[:46]}` | _field_ | '
                               f'[L{m.lineno}]({link}#L{m.lineno}) |')
        elif isinstance(n, ast.FunctionDef) and not n.name.startswith('_'):
            api.append(f'| `{n.name}` | `({sig(n)})` | {live(n)} | '
                       f'[L{n.lineno}]({link}#L{n.lineno}) |')
            cmap.append(f'| `{n.name}()` | [L{n.lineno}-{n.end_lineno}]'
                        f'({link}#L{n.lineno}-L{n.end_lineno}) |')

    header = (f'**File**: [`{py_path}`]({link}) — **{nlines} lines** — '
              f'canonical coverage **{pct}**')
    api_tbl = ('| symbol | signature | canonical? | code |\n|---|---|---|---|\n'
               + '\n'.join(api) + '\n')
    map_tbl = ('| unit | source |\n|---|---|\n' + '\n'.join(cmap) + '\n')
    return header, api_tbl, map_tbl


def splice(md, header, api_tbl, map_tbl):
    lines = md.split('\n')
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith('**File**:'):
            out.append(header)
            i += 1
            continue
        if ln.strip() == '## Public API':
            out.append(ln)
            out.append('')
            out.append(api_tbl.rstrip())
            i += 1
            while i < len(lines) and (lines[i].startswith('|') or not lines[i].strip()):
                i += 1
            out.append('')
            continue
        if ln.strip() == '## Code map':
            out.append(ln)
            out.append('')
            out.append(map_tbl.rstrip())
            i += 1
            while i < len(lines) and (lines[i].startswith('|') or not lines[i].strip()):
                i += 1
            out.append('')
            continue
        out.append(ln)
        i += 1
    s = '\n'.join(out)
    # insert a Code map section before "## See also" if absent
    if '## Code map' not in s:
        s = s.replace('## See also',
                      f'## Code map\n\n{map_tbl.rstrip()}\n\n---\n\n## See also')
    return s


load_cov.cache = load_cov()

stale, written = [], 0
for root, dirs, files in os.walk('crawlbot'):
    dirs[:] = [d for d in dirs if d != '__pycache__']
    pkg = root.split('/')[-1]
    if pkg == 'crawlbot':
        continue
    for f in sorted(files):
        if not f.endswith('.py') or f == '__init__.py':
            continue
        py = os.path.join(root, f)
        md = f'{DOCS}/{pkg}/{f[:-3]}.md'
        if not os.path.exists(md):
            stale.append((md, 'MISSING — module has no document'))
            continue
        cur = open(md, encoding='utf-8').read()
        new = splice(cur, *build_blocks(py, '../../..'))
        if new != cur:
            if CHECK:
                stale.append((md, 'out of date with the code'))
            else:
                open(md, 'w', encoding='utf-8').write(new)
                written += 1

if CHECK:
    if stale:
        print(f'*** {len(stale)} document(s) out of date ***')
        for m, why in stale:
            print(f'  {m}: {why}')
        print('\nrun: PYTHONPATH=. python3 gate/sync_docs.py')
        sys.exit(1)
    print('docs/crawlbot/ is in sync with the code — OK')
else:
    for m, why in stale:
        print(f'  {m}: {why}')
    print(f'updated {written} document(s)')
