# M7 — weld_sweep `mj_model.eq_solref` verification

Quick check only, no simulation. For each of the three weld_sweep variants, the MJCF was mutated in place and a fresh `mujoco.MjModel.from_xml_path(...)` was invoked; `model.eq_solref[eid]` was read for every equality `eid` in `range(model.neq)`.

## 1. Script loading order

`scripts/_weld_solref_sweep.py:_run_variant` calls `_mutate_mjcf(...)` which writes the file to disk, then instantiates `SimulationLoop(mjcf_path=MJCF, ...)` and calls `sim.setup(...)`. `SimulationLoop.__init__` (`sim_loop.py:66-140`) only stores the path; the actual `mujoco.MjModel.from_xml_path(self.mjcf_path)` call happens inside `setup()` at `sim_loop.py:149`. No Python-level caching between calls; each `setup()` invocation re-reads the file.

**Order: mutate → load. Correct.**

## 2. `mj_model.eq_solref` per variant

MuJoCo equality constraints, in `eid` order, as identified by `mj_id2name(..., mjOBJ_EQUALITY, eid)`:

| eid | name |
|---|---|
| 0 | `grip_a_to_1a` |
| 1 | `grip_a_to_2a` |
| 2 | `grip_a_to_3a` |
| 3 | `grip_a_to_4a` |
| 4 | `grip_a_to_5a` |
| 5 | `grip_a_to_6a` |
| 6 | `grip_b_to_1b` |
| 7 | `grip_b_to_2b` |
| 8 | `grip_b_to_3b` |
| 9 | `grip_b_to_4b` |
| 10 | `grip_b_to_5b` |
| 11 | `grip_b_to_6b` |

### Variant `solref_0p003_1` — XML `solref="0.003 1"`

`arm-joint check (Joint_1_a)`: `dof_damping = 0`, `dof_armature = 0`

| eid | name | eq_solref[0] | eq_solref[1] |
|---|---|---|---|
| 0 | `grip_a_to_1a` | `+3.000000e-03` | `+1.000000e+00` |
| 1 | `grip_a_to_2a` | `+3.000000e-03` | `+1.000000e+00` |
| 2 | `grip_a_to_3a` | `+3.000000e-03` | `+1.000000e+00` |
| 3 | `grip_a_to_4a` | `+3.000000e-03` | `+1.000000e+00` |
| 4 | `grip_a_to_5a` | `+3.000000e-03` | `+1.000000e+00` |
| 5 | `grip_a_to_6a` | `+3.000000e-03` | `+1.000000e+00` |
| 6 | `grip_b_to_1b` | `+3.000000e-03` | `+1.000000e+00` |
| 7 | `grip_b_to_2b` | `+3.000000e-03` | `+1.000000e+00` |
| 8 | `grip_b_to_3b` | `+3.000000e-03` | `+1.000000e+00` |
| 9 | `grip_b_to_4b` | `+3.000000e-03` | `+1.000000e+00` |
| 10 | `grip_b_to_5b` | `+3.000000e-03` | `+1.000000e+00` |
| 11 | `grip_b_to_6b` | `+3.000000e-03` | `+1.000000e+00` |

Unique `eq_solref` rows across all 12 equalities: `[(0.003, 1.0)]`.

### Variant `solref_0p001_1` — XML `solref="0.001 1"`

`arm-joint check (Joint_1_a)`: `dof_damping = 0`, `dof_armature = 0`

| eid | name | eq_solref[0] | eq_solref[1] |
|---|---|---|---|
| 0 | `grip_a_to_1a` | `+1.000000e-03` | `+1.000000e+00` |
| 1 | `grip_a_to_2a` | `+1.000000e-03` | `+1.000000e+00` |
| 2 | `grip_a_to_3a` | `+1.000000e-03` | `+1.000000e+00` |
| 3 | `grip_a_to_4a` | `+1.000000e-03` | `+1.000000e+00` |
| 4 | `grip_a_to_5a` | `+1.000000e-03` | `+1.000000e+00` |
| 5 | `grip_a_to_6a` | `+1.000000e-03` | `+1.000000e+00` |
| 6 | `grip_b_to_1b` | `+1.000000e-03` | `+1.000000e+00` |
| 7 | `grip_b_to_2b` | `+1.000000e-03` | `+1.000000e+00` |
| 8 | `grip_b_to_3b` | `+1.000000e-03` | `+1.000000e+00` |
| 9 | `grip_b_to_4b` | `+1.000000e-03` | `+1.000000e+00` |
| 10 | `grip_b_to_5b` | `+1.000000e-03` | `+1.000000e+00` |
| 11 | `grip_b_to_6b` | `+1.000000e-03` | `+1.000000e+00` |

Unique `eq_solref` rows across all 12 equalities: `[(0.001, 1.0)]`.

### Variant `solref_direct_stiff` — XML `solref="-1e6 -1e3"`

`arm-joint check (Joint_1_a)`: `dof_damping = 0`, `dof_armature = 0`

| eid | name | eq_solref[0] | eq_solref[1] |
|---|---|---|---|
| 0 | `grip_a_to_1a` | `-1.000000e+06` | `-1.000000e+03` |
| 1 | `grip_a_to_2a` | `-1.000000e+06` | `-1.000000e+03` |
| 2 | `grip_a_to_3a` | `-1.000000e+06` | `-1.000000e+03` |
| 3 | `grip_a_to_4a` | `-1.000000e+06` | `-1.000000e+03` |
| 4 | `grip_a_to_5a` | `-1.000000e+06` | `-1.000000e+03` |
| 5 | `grip_a_to_6a` | `-1.000000e+06` | `-1.000000e+03` |
| 6 | `grip_b_to_1b` | `-1.000000e+06` | `-1.000000e+03` |
| 7 | `grip_b_to_2b` | `-1.000000e+06` | `-1.000000e+03` |
| 8 | `grip_b_to_3b` | `-1.000000e+06` | `-1.000000e+03` |
| 9 | `grip_b_to_4b` | `-1.000000e+06` | `-1.000000e+03` |
| 10 | `grip_b_to_5b` | `-1.000000e+06` | `-1.000000e+03` |
| 11 | `grip_b_to_6b` | `-1.000000e+06` | `-1.000000e+03` |

Unique `eq_solref` rows across all 12 equalities: `[(-1000000.0, -1000.0)]`.

## 3. Cross-variant comparison

One-row summary: the unique `eq_solref` pairs observed across the twelve welds in each variant.

| variant | XML solref | unique `eq_solref` rows |
|---|---|---|
| `solref_0p003_1` | `"0.003 1"` | `[(0.003, 1.0)]` |
| `solref_0p001_1` | `"0.001 1"` | `[(0.001, 1.0)]` |
| `solref_direct_stiff` | `"-1e6 -1e3"` | `[(-1000000.0, -1000.0)]` |

**All three variants show identical `eq_solref`?** `False`

## 4. MJCF restoration

Script-exit byte-exact comparison against the entry text: **`True`**.

Post-restoration read of `eq_solref` (HEAD-state):

- Unique `eq_solref` rows: `[(0.003, 1.0)]`
- Joint_1_a dof_damping = `0.05`
- Joint_1_a dof_armature = `0.05`
