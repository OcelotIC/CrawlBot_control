# `crawlbot.core.ik`

**File**: `crawlbot/core/ik.py` — **1468 lines** — canonical coverage **40 %**

> Module docstring: *"Inverse kinematics for VISPA docking configurations."*

Inverse kinematics for docking: place both grippers on their anchors, keep the
torso at a commanded orientation and standoff, and pick — among the infinitely
many solutions a free-flying dual-arm robot admits — one that is far from
singularity.

The largest file in the package, and the one with the largest share **off** the
canonical path.

---

## Public API

| symbol | signature | canonical? |
|---|---|---|
| `solve_ik` | `(model, q0, targets, max_iter=500, tol=1e-08, base_gain=...)` | **yes** |
| `dock_configuration` | `(model, anchor_a, anchor_b, torso_pos=None, q_init=None,...)` | not exercised |
| `dock_configuration_fixed_rotation` | `(model, anchor_a, anchor_b, R_torso_fixed, torso_pos=Non...)` | **yes** |
| `manipulability_config` | `(model, anchor_a, anchor_b, level_axis, q_nominal, w_pos...)` | **yes** |
| `precompute_torso_map` | `(model, anchors_a, anchors_b, anchor_pair_sequence, q_in...)` | not exercised |
| `manipulability_config_trajectory` | `(model, anchor_a, anchor_b, q_start, n_samples=5, q_gues...)` | not exercised |
| `manipulability_config_mid_waypoint` | `(model, anchor_a_pose, anchor_b_pose, q_start, q_end, sw...)` | not exercised |
| `check_path_feasibility` | `(model, q_start, q_end, anchor_a_pose, anchor_b_pose, sw...)` | not exercised |
| `solve_ik_waypoints` | `(model, q_start, stance_frame, stance_target, swing_fram...)` | not exercised |

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

`dock_configuration`, `precompute_torso_map`, `manipulability_config_trajectory`,
`manipulability_config_mid_waypoint`, `check_path_feasibility`,
`solve_ik_waypoints`.

These served variants whose wiring was removed from `sim_loop` by CLEANUP-15 —
the FK reference path, trajectory-aware IK, mid-waypoint reshaping, the path
feasibility probe. They are not dead in the strict sense, but they carry **no
gate coverage at all**: changing them will trip nothing.

That is the main verification blind spot inside `crawlbot/`.

## 5. A design trap this module cannot protect you from

Project rule:

> *Do not generate trajectory acceleration profiles without checking actuator
> feasibility — a quintic on 591 mm of torso displacement saturates 20 Nm
> joints.*

IK returns a configuration that is **reachable**, which says nothing about
whether the trajectory *to* it is realisable within torque limits. Feasibility in
torque is a separate check, and in this architecture it is the pre-planner's job
(`planning/coarse_preplanner.md`), not the IK's.

## See also

- package overview: [`core.md`](core.md)
