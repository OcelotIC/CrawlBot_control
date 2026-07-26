# `crawlbot.core.ik`

**File**: [`crawlbot/core/ik.py`](../../../crawlbot/core/ik.py) — **786 lines** — canonical coverage **69 %**

> Module docstring: *"Inverse kinematics for VISPA docking configurations."*

Inverse kinematics for docking: place both grippers on their anchors, keep the
torso at a commanded orientation and standoff, and pick — among the infinitely
many solutions a free-flying dual-arm robot admits — one that is far from
singularity.

The largest file in the package, and the one with the largest share **off** the
canonical path.

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| `solve_ik` | `(model, q0, targets, max_iter=500, tol=1e-08, base_gain=...)` | **yes** | [L116](../../../crawlbot/core/ik.py#L116) |
| `dock_configuration` | `(model, anchor_a, anchor_b, torso_pos=None, q_init=None,...)` | not exercised | [L320](../../../crawlbot/core/ik.py#L320) |
| `dock_configuration_fixed_rotation` | `(model, anchor_a, anchor_b, R_torso_fixed, torso_pos=Non...)` | **yes** | [L368](../../../crawlbot/core/ik.py#L368) |
| `manipulability_config` | `(model, anchor_a, anchor_b, level_axis, q_nominal, w_pos...)` | **yes** | [L534](../../../crawlbot/core/ik.py#L534) |
| `solve_ik_waypoints` | `(model, q_start, stance_frame, stance_target, swing_fram...)` | not exercised | [L668](../../../crawlbot/core/ik.py#L668) |

---

---

## 1. `solve_ik` — damped least squares on the whole system

Iterative, on the **full** Jacobian (floating base + arm joints). For each
target frame the 6-D pose error is taken in the Lie algebra:

```
e_i = log6( X_i^-1 * X_i_target )        in R^6
```

and the velocity step solves a damped least-squares problem

```
dq = argmin  sum_i || J_i dq - e_i ||^2 + lambda^2 ||dq||^2
```

Damping keeps the step bounded near singularities, where `J` loses rank and an
undamped pseudo-inverse would explode.

⚠ With several targets the contributions are **summed, not prioritised**. Both
grippers are equal citizens. That is correct here — a docking pose in which one
hand reaches and the other does not is useless — but it means `solve_ik` is not
a hierarchical solver and should not be used as one.

Convergence is tested on `sum_i ||log6(e_i)||` against `tol = 1e-8`.

### The posture term

With `q_nominal` and `w_posture > 0`, the gradient `q_nominal - q_arms` is
projected into the **whole-system task null space** (both tool Jacobians
stacked) and added to each step. The effect is worth stating precisely: the
**base repositions** so the arms can relax toward `q_nominal` *without* moving
either gripper off its target. It is the mechanism that de-contorts the arms
using the freedom a free-flying base provides.

Default off, so behaviour is bit-identical to the legacy solver when unused.

## 2. `dock_configuration_fixed_rotation` — the canonical entry point

Same solver, with the torso **orientation imposed** and, through `com_z_target`,
the **canonical standoff of -0.35 m**: the crawl height at which the CoM is
held.

Fixing the orientation removes 3 DOF of nuisance freedom and makes the docking
pose repeatable step to step, which matters because the docking gate is a
5 mm / 5 deg window.

## 3. `manipulability_config` — choosing among the solutions

The redundancy left after both targets are met is resolved by maximising the
**product of the Yoshikawa manipulability indices** of the two arms,

```
w_j = sqrt( det( J_j J_j^T ) )        objective:  max  w_a * w_b
```

An outer Nelder-Mead optimises **torso xyz only**, re-running the inner
`solve_ik` at each probe; the sigma_min objective is unchanged from the legacy
version.

The product, rather than the sum, is what forbids the degenerate trade: an
arrangement where one arm is comfortable and the other near-singular scores
zero. A sum would accept it.

## 4. What is *not* on the canonical path

Two entries, down from six. CLEANUP-30 retired the other four.

`dock_configuration` — a convenience wrapper (both tools at anchor poses) that
`manipulability_config` and the retired variants used as a seed. Live: called
from `sim_loop.py:325` on the init path and by two test modules.

`solve_ik_waypoints` — a chain of IK solutions along a swing arc. **Zero
callers anywhere**, measured: nothing in `crawlbot/`, nothing in `scripts/`,
nothing in `tests/`. It survived CLEANUP-30 only because the retirement scope
was the four Option-B functions; it is the same class and the obvious next
candidate. 118 lines, and the largest single block of the 173 statements in this
module that the canonical replay never reaches.

### The Option-B path, and why it went

Retired in CLEANUP-30 — 695 lines, 47 % of this module:
`manipulability_config_trajectory`, `manipulability_config_mid_waypoint`,
`check_path_feasibility`, `precompute_torso_map`, plus `_interpolate_q_quintic`,
`_trajectory_worst_w`, `_sigma_min_pair` and `_ik_three_tasks`, which stranded
with them (computed from the call graph, not chosen by eye).

They were IK 3 in `IK_FORMULATION.md` — the trajectory-aware escalation built for
the T15 step-2 path singularity, where the endpoint-only IK could return a
configuration whose *interior* passed near a singularity. The evidence for
retiring: zero callers in `crawlbot/`, 0 lines executed by the canonical replay,
and 87 % of the test suite's runtime spent on their regression tests. Removing
them left the canonical run byte-identical over 2077 rows × 132 928 fields, with
all six docks at delta +0.0000.

The reasoning is preserved rather than deleted: `IK_FORMULATION.md` §7–§9 still
derives the formulation and carries a banner marking it retired, and the tests
live in `Misc/tests/` with their fixture. Revival starts at
`git show d61e1a0:crawlbot/core/ik.py`.

⚠ **One consequence to hold onto:** `check_path_feasibility` was the only
interior-feasibility guard, and it was *already* disconnected — nothing called
it. Retiring it removed no protection that was actually running, but it does mean
the architecture has no path-feasibility check at all. The canonical scenario
does not need one; a new anchor geometry might.

## 5. A design trap this module cannot protect you from

Project rule:

> *Do not generate trajectory acceleration profiles without checking actuator
> feasibility — a quintic on 591 mm of torso displacement saturates 20 Nm
> joints.*

IK returns a configuration that is **reachable**, which says nothing about
whether the trajectory *to* it is realisable within torque limits. Feasibility in
torque is a separate check, and in this architecture it is the pre-planner's job
(`planning/coarse_preplanner.md`), not the IK's.

## Code map

| unit | source |
|---|---|
| `solve_ik()` | [L116-317](../../../crawlbot/core/ik.py#L116-L317) |
| `dock_configuration()` | [L320-365](../../../crawlbot/core/ik.py#L320-L365) |
| `dock_configuration_fixed_rotation()` | [L368-531](../../../crawlbot/core/ik.py#L368-L531) |
| `manipulability_config()` | [L534-665](../../../crawlbot/core/ik.py#L534-L665) |
| `solve_ik_waypoints()` | [L668-785](../../../crawlbot/core/ik.py#L668-L785) |

---

## See also

- package overview: [`core.md`](core.md)
