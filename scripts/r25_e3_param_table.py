#!/usr/bin/env python3
"""R25 export E3 — parameter table: canonical value vs dataclass default.

Reads the LIVE config objects captured during the instrumented canonical replay
(`e3_live_config.json`) and diffs every field against the pristine dataclass
default, resolving each field's declaration to file:line by AST.

The diff is the point. A dataclass default that differs from what the canonical
run actually used is exactly how a wrong number reaches a paper — three such
divergences were already found this way (torso gains, qp_envelope_exact, wheel
inertia vs the MJCF armature).

Output: results/review_closure/r25/e3_param_table.json + .md
"""
import ast
import dataclasses
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
os.environ.setdefault('MUJOCO_GL', 'disabled')

OUT = 'results/review_closure/r25'
LIVE = json.load(open(f'{OUT}/e3_live_config.json'))

from crawlbot.simulation.config import SimConfig                    # noqa: E402
from crawlbot.solvers.wholebody_qp import WholeBodyQPConfig          # noqa: E402
from crawlbot.solvers.centroidal_nmpc import CentroidalNMPCConfig    # noqa: E402
from crawlbot.planning.coarse_preplanner import CoarsePrePlannerConfig  # noqa: E402

CLASSES = {
    'SimConfig_live': (SimConfig, 'crawlbot/simulation/config.py'),
    'WholeBodyQPConfig_live[qp_ss]': (WholeBodyQPConfig,
                                      'crawlbot/solvers/wholebody_qp.py'),
    'CentroidalNMPCConfig_live': (CentroidalNMPCConfig,
                                  'crawlbot/solvers/centroidal_nmpc.py'),
    'CoarsePrePlannerConfig_live': (CoarsePrePlannerConfig,
                                    'crawlbot/planning/coarse_preplanner.py'),
}


def field_lines(path, cls_name):
    """field -> line number of its declaration in the dataclass body."""
    tree = ast.parse(open(path).read())
    cls = next((n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == cls_name), None)
    if cls is None:
        return {}
    out = {}
    for st in cls.body:
        if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
            out[st.target.id] = st.lineno
    return out


def norm(v):
    if isinstance(v, list):
        return np.asarray(v, float)
    return v


def same(a, b):
    try:
        a, b = norm(a), norm(b)
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            a, b = np.atleast_1d(a).astype(float), np.atleast_1d(b).astype(float)
            return a.shape == b.shape and np.allclose(a, b, equal_nan=True)
        if isinstance(a, float) or isinstance(b, float):
            return float(a) == float(b) or (a != a and b != b)
        return a == b
    except Exception:
        return repr(a) == repr(b)


report = {}
divergences = []
for key, (cls, path) in CLASSES.items():
    live = LIVE.get(key)
    if live is None:
        report[key] = {'available': False}
        continue
    default_obj = cls()
    lines = field_lines(path, cls.__name__)
    fields = {}
    for f in dataclasses.fields(cls):
        dv = getattr(default_obj, f.name)
        if isinstance(dv, np.ndarray):
            dv = dv.tolist()
        lv = live.get(f.name)
        differs = not same(dv, lv)
        rec = {'canonical': lv, 'default': dv, 'differs': differs,
               'source': f'{path}:{lines.get(f.name, "?")}'}
        fields[f.name] = rec
        if differs:
            divergences.append({'config': cls.__name__, 'field': f.name, **rec})
    report[key] = {'available': True, 'class': cls.__name__,
                   'file': path, 'n_fields': len(fields), 'fields': fields}

report['_divergences_default_vs_canonical'] = divergences
json.dump(report, open(f'{OUT}/e3_param_table.json', 'w'), indent=2)

lines_md = ['# E3 — dataclass default vs canonical value', '',
            f'{len(divergences)} field(s) differ. Every one is a place where '
            f'reading the dataclass would put a wrong number in the paper.', '',
            '| config | field | canonical | default | source |',
            '|---|---|---|---|---|']
for d in divergences:
    c, df = d['canonical'], d['default']
    fmt = lambda v: (f'{v:.6g}' if isinstance(v, float)
                     else (str(v)[:52] + '…' if len(str(v)) > 52 else str(v)))
    lines_md.append(f'| `{d["config"]}` | `{d["field"]}` | {fmt(c)} | '
                    f'{fmt(df)} | `{d["source"]}` |')
open(f'{OUT}/e3_param_table.md', 'w').write('\n'.join(lines_md) + '\n')

print(f'{len(divergences)} divergences default vs canonical\n')
for d in divergences:
    print(f'  {d["config"]}.{d["field"]:28s} canonical={str(d["canonical"])[:34]:36s}'
          f' default={str(d["default"])[:26]:28s} {d["source"]}')
print(f'\nwrote {OUT}/e3_param_table.json and .md')
