#!/usr/bin/env python3
"""Generate the MEASURED skeleton of each per-module document.

Emits, per module: line count, canonical coverage, module docstring, and
the public-API table annotated LIVE / non exerce from the canonical
replay's line coverage. Prose is written by hand on top.

Only writes files that do not exist, so re-running never clobbers prose.
"""
import ast
import json
import os

cov = json.load(open('gate/_run/cov/cov.json'))
OUT = 'docs/crawlbot'


def sig(n):
    a = n.args
    parts = []
    defaults = [None] * (len(a.args) - len(a.defaults)) + list(a.defaults)
    for arg, d in zip(a.args, defaults):
        if arg.arg == 'self':
            continue
        s = arg.arg
        if d is not None:
            s += f'={ast.unparse(d)}'
        parts.append(s)
    if a.vararg:
        parts.append('*' + a.vararg.arg)
    for arg in a.kwonlyargs:
        parts.append(arg.arg)
    if a.kwarg:
        parts.append('**' + a.kwarg.arg)
    s = ', '.join(parts)
    return s if len(s) <= 58 else s[:55] + '...'


made = []
for root, dirs, files in os.walk('crawlbot'):
    dirs[:] = [d for d in dirs if d != '__pycache__']
    pkg = root.split('/')[-1]
    if pkg == 'crawlbot':
        continue
    for f in sorted(files):
        if not f.endswith('.py') or f == '__init__.py':
            continue
        mod = f[:-3]
        dst = f'{OUT}/{pkg}/{mod}.md'
        if os.path.exists(dst):
            continue
        p = os.path.join(root, f)
        src = open(p).read()
        t = ast.parse(src)
        nlines = len(src.split('\n'))
        entry = cov['files'].get(p)
        pct = f"{entry['summary']['percent_covered']:.0f} %" if entry else 'jamais importe'
        ex = set(entry['executed_lines']) if entry else set()
        doc = (ast.get_docstring(t) or '').strip()
        head = doc.split('\n')[0] if doc else ''

        def live(n):
            inner = [x for x in range(n.lineno + 1, (n.end_lineno or n.lineno) + 1)]
            return '**oui**' if (ex & set(inner)) else 'non exerce'

        rows, consts = [], []
        for n in t.body:
            if isinstance(n, ast.ClassDef):
                deco = ' *(dataclass)*' if any(
                    'dataclass' in ast.unparse(d) for d in n.decorator_list) else ''
                rows.append((f'**`{n.name}`**{deco}', '', '', n.lineno))
                for m in n.body:
                    if isinstance(m, ast.FunctionDef) and not m.name.startswith('__'):
                        rows.append((f'`.{m.name}`', f'`({sig(m)})`',
                                     live(m), m.lineno))
                    elif isinstance(m, ast.AnnAssign) and isinstance(m.target, ast.Name):
                        v = ast.unparse(m.value) if m.value else ''
                        pre = 'field(default_factory=lambda: '
                        if v.startswith(pre):          # strip wrapper AND its paren
                            v = v[len(pre):-1] if v.endswith(')') else v[len(pre):]
                        rows.append((f'  `{m.target.id}`',
                                     f'`{v[:46]}`' if v else '', '_champ_',
                                     m.lineno))
            elif isinstance(n, ast.FunctionDef) and not n.name.startswith('_'):
                rows.append((f'`{n.name}`', f'`({sig(n)})`', live(n), n.lineno))
            elif isinstance(n, ast.Assign):
                for tg in n.targets:
                    if isinstance(tg, ast.Name) and tg.id.isupper():
                        consts.append((tg.id, ast.unparse(n.value)[:44]))

        L = [f'# `crawlbot.{pkg}.{mod}`', '',
             '<!-- PROSE: resume en une phrase -->', '',
             f'**Fichier** : `{p}` — **{nlines} lignes** — '
             f'couverture canonique **{pct}**', '']
        if head:
            L += [f'> Docstring du module : *« {head} »*', '']
        L += ['---', '', '## API publique', '',
              '| symbole | signature | canonique ? |', '|---|---|---|']
        for name, s, lv, _ in rows:
            L.append(f'| {name} | {s} | {lv} |')
        if consts:
            L += ['', '### Constantes de module', '',
                  '| nom | valeur |', '|---|---|']
            L += [f'| `{k}` | `{v}` |' for k, v in consts]
        L += ['', '---', '', '<!-- PROSE: fonctionnement, pieges, valeurs canoniques -->',
              '', '## Voir aussi', '',
              f'- vue d\'ensemble du paquet : [`{pkg}.md`]({pkg}.md)', '']
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, 'w', encoding='utf-8').write('\n'.join(L))
        made.append(dst)

print(f'generated {len(made)} module skeletons')
for m in made:
    print(f'  {m}')
