"""Verify that the three weld_sweep variants actually loaded distinct
mj_model.eq_solref values. No simulation — three MJCF mutations,
three model loads, read eq_solref per equality.

Outputs Misc/runs/M7_settle_diag/weld_sweep/solref_verification.md.
MJCF is restored on script exit and verified.
"""
from __future__ import annotations

import os
import re
import sys

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import numpy as np
import mujoco


MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')
OUT = os.path.join(_root, 'Misc', 'runs', 'M7_settle_diag', 'weld_sweep',
                   'solref_verification.md')

VARIANTS = [
    ('solref_0p003_1',      '0.003 1'),
    ('solref_0p001_1',      '0.001 1'),
    ('solref_direct_stiff', '-1e6 -1e3'),
]

ROBOT_JOINT_RE = re.compile(
    r'(<default class="robot_joint">\s*\n\s*<joint damping=")[^"]+'
    r'(" armature=")[^"]+(")')
WELD_SOLREF_RE = re.compile(
    r'(<weld name="[^"]+" site1="[^"]+" site2="[^"]+" solref=")[^"]+(")')


def _mutate_mjcf(damping: float, armature: float, solref: str):
    with open(MJCF, 'r') as f:
        text = f.read()
    text, n1 = ROBOT_JOINT_RE.subn(
        rf'\g<1>{damping}\g<2>{armature}\g<3>', text, count=1)
    if n1 != 1:
        raise RuntimeError('robot_joint default not found')
    text, n2 = WELD_SOLREF_RE.subn(rf'\g<1>{solref}\g<2>', text)
    if n2 == 0:
        raise RuntimeError('no <weld> solref replacement')
    with open(MJCF, 'w') as f:
        f.write(text)
    return n1, n2


def _read_eq_solref():
    """Fresh model load; return (names[], solref_matrix (neq, 2))."""
    model = mujoco.MjModel.from_xml_path(MJCF)
    names = []
    solref_rows = []
    for eid in range(model.neq):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, eid)
        names.append(name)
        # eq_solref is (neq, mjNREF = 2)
        solref_rows.append(model.eq_solref[eid].copy())
    return names, np.array(solref_rows)


def _arm_joint_params():
    """Read damping/armature for the first arm joint (Joint_1_a) to
    confirm the default-class modification propagated too."""
    model = mujoco.MjModel.from_xml_path(MJCF)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'Joint_1_a')
    if jid < 0:
        return None
    # dof_damping is per dof; jnt_dofadr gives the dof index for a joint.
    dof_adr = model.jnt_dofadr[jid]
    return {
        'joint': 'Joint_1_a',
        'dof_damping':  float(model.dof_damping[dof_adr]),
        'dof_armature': float(model.dof_armature[dof_adr]),
    }


def _verify_restored_exact(original: str):
    """Re-read MJCF, byte-compare to the original text captured at entry."""
    with open(MJCF, 'r') as f:
        current = f.read()
    return current == original


def main():
    with open(MJCF, 'r') as f:
        original_mjcf = f.read()

    per_variant = []
    try:
        for tag, solref_str in VARIANTS:
            _mutate_mjcf(damping=0.0, armature=0.0, solref=solref_str)
            names, solref_mat = _read_eq_solref()
            arm_params = _arm_joint_params()
            per_variant.append({
                'tag': tag,
                'xml_solref': solref_str,
                'names': names,
                'eq_solref': solref_mat,
                'arm_joint': arm_params,
            })
    finally:
        with open(MJCF, 'w') as f:
            f.write(original_mjcf)

    byte_ok = _verify_restored_exact(original_mjcf)

    # One more post-restoration read so the report can quote the
    # HEAD-state eq_solref for comparison.
    names_head, solref_head = _read_eq_solref()
    arm_head = _arm_joint_params()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write('# M7 — weld_sweep `mj_model.eq_solref` verification\n\n')
        f.write('Quick check only, no simulation. For each of the three '
                'weld_sweep variants, the MJCF was mutated in place and a '
                'fresh `mujoco.MjModel.from_xml_path(...)` was invoked; '
                '`model.eq_solref[eid]` was read for every equality `eid` '
                'in `range(model.neq)`.\n\n')

        f.write('## 1. Script loading order\n\n')
        f.write('`scripts/_weld_solref_sweep.py:_run_variant` calls '
                '`_mutate_mjcf(...)` which writes the file to disk, then '
                'instantiates `SimulationLoop(mjcf_path=MJCF, ...)` and '
                'calls `sim.setup(...)`. `SimulationLoop.__init__` '
                '(`sim_loop.py:66-140`) only stores the path; the actual '
                '`mujoco.MjModel.from_xml_path(self.mjcf_path)` call '
                'happens inside `setup()` at `sim_loop.py:149`. No '
                'Python-level caching between calls; each `setup()` '
                'invocation re-reads the file.\n\n')
        f.write('**Order: mutate → load. Correct.**\n\n')

        f.write('## 2. `mj_model.eq_solref` per variant\n\n')
        names0 = per_variant[0]['names']
        f.write('MuJoCo equality constraints, in `eid` order, as '
                'identified by `mj_id2name(..., mjOBJ_EQUALITY, eid)`:\n\n')
        f.write('| eid | name |\n|---|---|\n')
        for i, n in enumerate(names0):
            f.write(f'| {i} | `{n}` |\n')
        f.write('\n')

        for v in per_variant:
            f.write(f'### Variant `{v["tag"]}` — XML `solref="{v["xml_solref"]}"`\n\n')
            f.write(f'`arm-joint check (Joint_1_a)`: '
                    f'`dof_damping = {v["arm_joint"]["dof_damping"]:.6g}`, '
                    f'`dof_armature = {v["arm_joint"]["dof_armature"]:.6g}`\n\n')
            f.write('| eid | name | eq_solref[0] | eq_solref[1] |\n')
            f.write('|---|---|---|---|\n')
            for i, (n, row) in enumerate(zip(v['names'], v['eq_solref'])):
                f.write(f'| {i} | `{n}` | `{row[0]:+.6e}` | `{row[1]:+.6e}` |\n')
            f.write('\n')
            uniq = {tuple(row) for row in v['eq_solref']}
            f.write(f'Unique `eq_solref` rows across all {v["eq_solref"].shape[0]} '
                    f'equalities: `{sorted(uniq)}`.\n\n')

        f.write('## 3. Cross-variant comparison\n\n')
        f.write('One-row summary: the unique `eq_solref` pairs observed '
                'across the twelve welds in each variant.\n\n')
        f.write('| variant | XML solref | unique `eq_solref` rows |\n')
        f.write('|---|---|---|\n')
        for v in per_variant:
            uniq = {tuple(float(x) for x in row)
                    for row in v['eq_solref']}
            f.write(f'| `{v["tag"]}` | `"{v["xml_solref"]}"` | '
                    f'`{sorted(uniq)}` |\n')
        f.write('\n')

        variants_equal = all(
            np.array_equal(v['eq_solref'], per_variant[0]['eq_solref'])
            for v in per_variant[1:])
        f.write(f'**All three variants show identical `eq_solref`?** '
                f'`{variants_equal}`\n\n')

        f.write('## 4. MJCF restoration\n\n')
        f.write('Script-exit byte-exact comparison against the entry '
                f'text: **`{byte_ok}`**.\n\n')
        f.write('Post-restoration read of `eq_solref` (HEAD-state):\n\n')
        uniq_head = {tuple(float(x) for x in row)
                     for row in solref_head}
        f.write(f'- Unique `eq_solref` rows: `{sorted(uniq_head)}`\n')
        f.write(f'- Joint_1_a dof_damping = `{arm_head["dof_damping"]:.6g}`\n')
        f.write(f'- Joint_1_a dof_armature = `{arm_head["dof_armature"]:.6g}`\n')

    # Console summary.
    print()
    print('=' * 78)
    print('weld_sweep eq_solref verification')
    print('=' * 78)
    for v in per_variant:
        uniq = {tuple(float(x) for x in row)
                for row in v['eq_solref']}
        arm = v['arm_joint']
        print(f'  {v["tag"]:22s}  xml="{v["xml_solref"]:>10s}"  '
              f'uniq eq_solref={sorted(uniq)}  '
              f'Joint_1_a damping={arm["dof_damping"]:.4g}  '
              f'armature={arm["dof_armature"]:.4g}')
    print()
    all_equal = all(np.array_equal(v['eq_solref'],
                                   per_variant[0]['eq_solref'])
                    for v in per_variant[1:])
    print(f'  All variants identical? {all_equal}')
    print(f'  Byte-exact restoration: {byte_ok}')
    uniq_head = {tuple(float(x) for x in row) for row in solref_head}
    print(f'  Post-restoration eq_solref unique: {sorted(uniq_head)}')
    print()
    print(f'Report: {OUT}')


if __name__ == '__main__':
    main()
