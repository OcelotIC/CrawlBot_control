# `crawlbot.planning.contact_scheduler`

**File**: [`crawlbot/planning/contact_scheduler.py`](../../../crawlbot/planning/contact_scheduler.py) — **350 lines** — canonical coverage **87 %**

> Module docstring: *"ContactScheduler — Gait timing and contact management for VISPA crawling."*

The gait plan: the DS/SS/DS sequence, which anchor each gripper holds, and the
timeline — which is rebuilt in cascade once a step duration becomes known.

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| `make_anchor_grid` | `(n=DEFAULT_N_ANCHORS, dx=DEFAULT_DX, dy=DEFAULT_DY)` | not exercised | [L42](../../../crawlbot/planning/contact_scheduler.py#L42) |
| `read_anchors_from_mujoco` | `(mj_model, mj_data)` | **yes** | [L61](../../../crawlbot/planning/contact_scheduler.py#L61) |
| **`GaitPhase`** *(dataclass)* |  |  | [L103](../../../crawlbot/planning/contact_scheduler.py#L103) |
|   `phase` | `` | _field_ | [L105](../../../crawlbot/planning/contact_scheduler.py#L105) |
|   `duration` | `` | _field_ | [L106](../../../crawlbot/planning/contact_scheduler.py#L106) |
|   `anchor_a_idx` | `` | _field_ | [L107](../../../crawlbot/planning/contact_scheduler.py#L107) |
|   `anchor_b_idx` | `` | _field_ | [L108](../../../crawlbot/planning/contact_scheduler.py#L108) |
|   `swing_arm` | `''` | _field_ | [L110](../../../crawlbot/planning/contact_scheduler.py#L110) |
|   `swing_from_idx` | `-1` | _field_ | [L111](../../../crawlbot/planning/contact_scheduler.py#L111) |
|   `swing_to_idx` | `-1` | _field_ | [L112](../../../crawlbot/planning/contact_scheduler.py#L112) |
| **`GaitPlan`** *(dataclass)* |  |  | [L116](../../../crawlbot/planning/contact_scheduler.py#L116) |
|   `phases` | `` | _field_ | [L121](../../../crawlbot/planning/contact_scheduler.py#L121) |
|   `t_start` | `` | _field_ | [L122](../../../crawlbot/planning/contact_scheduler.py#L122) |
|   `t_end` | `` | _field_ | [L123](../../../crawlbot/planning/contact_scheduler.py#L123) |
|   `total_duration` | `` | _field_ | [L124](../../../crawlbot/planning/contact_scheduler.py#L124) |
| `.phase_at` | `(t)` | **yes** | [L126](../../../crawlbot/planning/contact_scheduler.py#L126) |
| `.set_step_duration` | `(idx, T_step)` | **yes** | [L135](../../../crawlbot/planning/contact_scheduler.py#L135) |
| **`ContactScheduler`** |  |  | [L162](../../../crawlbot/planning/contact_scheduler.py#L162) |
| `.plan_traversal` | `(start_a=0, start_b=0, n_steps=4)` | **yes** | [L203](../../../crawlbot/planning/contact_scheduler.py#L203) |
| `.plan` | `()` | **yes** | [L291](../../../crawlbot/planning/contact_scheduler.py#L291) |
| `.contact_config_at` | `(t)` | **yes** | [L296](../../../crawlbot/planning/contact_scheduler.py#L296) |
| `.contact_sequence_over_horizon` | `(t, dt, N)` | not exercised | [L313](../../../crawlbot/planning/contact_scheduler.py#L313) |
| `.anchor_se3` | `(arm, idx)` | **yes** | [L333](../../../crawlbot/planning/contact_scheduler.py#L333) |

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

## Code map

| unit | source |
|---|---|
| `make_anchor_grid()` | [L42-58](../../../crawlbot/planning/contact_scheduler.py#L42-L58) |
| `read_anchors_from_mujoco()` | [L61-99](../../../crawlbot/planning/contact_scheduler.py#L61-L99) |
| `class GaitPhase` | [L103-112](../../../crawlbot/planning/contact_scheduler.py#L103-L112) |
| `class GaitPlan` | [L116-159](../../../crawlbot/planning/contact_scheduler.py#L116-L159) |
| `GaitPlan.phase_at` | [L126-133](../../../crawlbot/planning/contact_scheduler.py#L126-L133) |
| `GaitPlan.set_step_duration` | [L135-159](../../../crawlbot/planning/contact_scheduler.py#L135-L159) |
| `class ContactScheduler` | [L162-349](../../../crawlbot/planning/contact_scheduler.py#L162-L349) |
| `ContactScheduler.plan_traversal` | [L203-288](../../../crawlbot/planning/contact_scheduler.py#L203-L288) |
| `ContactScheduler.plan` | [L291-294](../../../crawlbot/planning/contact_scheduler.py#L291-L294) |
| `ContactScheduler.contact_config_at` | [L296-311](../../../crawlbot/planning/contact_scheduler.py#L296-L311) |
| `ContactScheduler.contact_sequence_over_horizon` | [L313-331](../../../crawlbot/planning/contact_scheduler.py#L313-L331) |
| `ContactScheduler.anchor_se3` | [L333-349](../../../crawlbot/planning/contact_scheduler.py#L333-L349) |

---

## See also

- package overview: [`planning.md`](planning.md)
