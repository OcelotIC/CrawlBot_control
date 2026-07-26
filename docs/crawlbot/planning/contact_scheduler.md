# `crawlbot.planning.contact_scheduler`

**File**: `crawlbot/planning/contact_scheduler.py` — **350 lines** — canonical coverage **87 %**

> Module docstring: *"ContactScheduler — Gait timing and contact management for VISPA crawling."*

The gait plan: the DS/SS/DS sequence, which anchor each gripper holds, and the
timeline — which is rebuilt in cascade once a step duration becomes known.

---

## Public API

| symbol | signature | canonical? |
|---|---|---|
| `make_anchor_grid` | `(n=DEFAULT_N_ANCHORS, dx=DEFAULT_DX, dy=DEFAULT_DY)` | not exercised |
| `read_anchors_from_mujoco` | `(mj_model, mj_data)` | **yes** |
| **`GaitPhase`** *(dataclass)* |  |  |
|   `phase` |  | _field_ |
|   `duration` |  | _field_ |
|   `anchor_a_idx` |  | _field_ |
|   `anchor_b_idx` |  | _field_ |
|   `swing_arm` | `''` | _field_ |
|   `swing_from_idx` | `-1` | _field_ |
|   `swing_to_idx` | `-1` | _field_ |
| **`GaitPlan`** *(dataclass)* |  |  |
|   `phases` |  | _field_ |
|   `t_start` |  | _field_ |
|   `t_end` |  | _field_ |
|   `total_duration` |  | _field_ |
| `.phase_at` | `(t)` | **yes** |
| `.set_step_duration` | `(idx, T_step)` | **yes** |
| **`ContactScheduler`** |  |  |
| `.plan_traversal` | `(start_a=0, start_b=0, n_steps=4)` | **yes** |
| `.plan` | `()` | **yes** |
| `.contact_config_at` | `(t)` | **yes** |
| `.contact_sequence_over_horizon` | `(t, dt, N)` | not exercised |
| `.anchor_se3` | `(arm, idx)` | **yes** |

### Module constants

| name | value |
|---|---|
| `DEFAULT_DX` | `0.8` |
| `DEFAULT_DY` | `0.3` |
| `DEFAULT_N_ANCHORS` | `6` |

---

---

## 1. Building the plan

`plan_traversal(start_a, start_b, n_steps)` produces a `GaitPlan`: a list of
`GaitPhase` (DOUBLE / SINGLE_A / SINGLE_B), each carrying anchor indices and
which arm swings. Arms alternate, one anchor at a time:

```
DS  ->  SS_A (B swings to b+1)  ->  DS  ->  SS_B (A swings to a+1)  ->  DS ...
```

## 2. ⚠ SS phases are born with `duration = 0.0`

The real duration comes from the pre-planner. `GaitPlan.set_step_duration(idx,
T_step)` installs it and **rebuilds the whole timeline in cascade**
(`sim_loop.py:1495`), preserving

```
t_end[k]   = t_start[k] + duration[k]
t_start[k+1] = t_end[k]
```

Rebuilding from scratch rather than patching offsets is what keeps the invariant
exact after several steps of differing duration — the alternative accumulates
floating-point drift across a traversal.

`dt_ds = 0.5 s` is only a skeleton value so queries before the energy-settle
runs have a valid timeline; the real DS duration is governed by the settle in
`sim_loop`.

## 3. Anchors

Canonical: `read_anchors_from_mujoco` — read straight from the MuJoCo model, so
the plan and the simulated scene cannot disagree. It scans site names
`anchor_1a..anchor_Na` / `anchor_1b..Nb` and stops at the first gap.

`make_anchor_grid` (analytic grid, `dx = 0.8`, `dy = 0.3`, 6 anchors) is the
fallback when no anchors are supplied, and is still used by a diagnostic script.
Unexercised on the canonical.

## 4. Essentially clean

16 uncovered lines out of 120 — **15 are guards and fallbacks**:

- `set_step_duration`: `IndexError` (index out of range), `ValueError`
  (`T_step <= 0`)
- `plan` property: `RuntimeError("Call plan_traversal() first.")`
- `plan_traversal`: the two `break`s when the anchor grid runs out
- `read_anchors_from_mujoco`: `ImportError`, `except: break`, `RuntimeError`
- `__init__`: the `anchors_a is None` fallback
- `phase_at`: the exactly-at-end edge case

All dead **because the system is healthy**. Keep.

**One genuinely dead method**: `contact_sequence_over_horizon` (19 lines, zero
callers) — plumbing intended for the NMPC horizon that the NMPC never took up.

## See also

- package overview: [`planning.md`](planning.md)
