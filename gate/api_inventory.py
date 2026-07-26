#!/usr/bin/env python3
"""Accurate public-API inventory of crawlbot/, straight from the AST.

Documentation written from memory is how docs/api/ rotted. Every symbol
and signature in the new docs comes from this.
"""
import ast
import os
import sys

PKG = sys.argv[1] if len(sys.argv) > 1 else None


def sig(n):
    a = n.args
    parts = []
    defaults = [None] * (len(a.args) - len(a.defaults)) + list(a.defaults)
    for arg, d in zip(a.args, defaults):
        s = arg.arg
        if arg.annotation:
            s += f': {ast.unparse(arg.annotation)}'
        if d is not None:
            s += f' = {ast.unparse(d)}'
        parts.append(s)
    if a.vararg:
        parts.append('*' + a.vararg.arg)
    kwd = [None] * (len(a.kwonlyargs) - len(a.kw_defaults)) + list(a.kw_defaults)
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        s = arg.arg + (f': {ast.unparse(arg.annotation)}' if arg.annotation else '')
        if d is not None:
            s += f' = {ast.unparse(d)}'
        parts.append(s)
    if a.kwarg:
        parts.append('**' + a.kwarg.arg)
    r = f' -> {ast.unparse(n.returns)}' if n.returns else ''
    return f'{n.name}({", ".join(parts)}){r}'


for root, dirs, files in os.walk('crawlbot'):
    dirs[:] = sorted(d for d in dirs if d != '__pycache__')
    sub = root.split('/')[-1]
    if PKG and sub != PKG:
        continue
    for f in sorted(files):
        if not f.endswith('.py') or f == '__init__.py':
            continue
        p = os.path.join(root, f)
        t = ast.parse(open(p).read())
        n_lines = len(open(p).read().split('\n'))
        d = (ast.get_docstring(t) or '').strip().split('\n')[0]
        print(f'\n### {p}  ({n_lines} lines)')
        print(f'    """{d}"""')
        for n in t.body:
            if isinstance(n, ast.ClassDef):
                deco = ','.join(ast.unparse(x) for x in n.decorator_list)
                print(f'  class {n.name}'
                      + (f'  [@{deco}]' if deco else '')
                      + f'   L{n.lineno}-{n.end_lineno}')
                for m in n.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if m.name.startswith('_') and m.name != '__init__':
                            continue
                        print(f'      .{sig(m)}   L{m.lineno}')
                    elif isinstance(m, ast.AnnAssign) and isinstance(m.target, ast.Name):
                        v = f' = {ast.unparse(m.value)}' if m.value else ''
                        print(f'      {m.target.id}: {ast.unparse(m.annotation)}{v}')
            elif isinstance(n, ast.FunctionDef) and not n.name.startswith('_'):
                print(f'  def {sig(n)}   L{n.lineno}')
            elif isinstance(n, ast.Assign):
                for tg in n.targets:
                    if isinstance(tg, ast.Name) and tg.id.isupper():
                        print(f'  {tg.id} = {ast.unparse(n.value)[:70]}')
