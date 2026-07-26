# `crawlbot.solvers.wholebody_qp`

**File**: `crawlbot/solvers/wholebody_qp.py` — **950 lines** — canonical coverage **97 %**

> Module docstring: *"WholeBodyQP - Whole-body Quadratic Program for high-rate tracking."*

**Stage 2 of the controller — the canonical one.** An instantaneous QP solved
at 100 Hz that turns the references (torso pose, swing end-effector, momentum,
posture) into joint accelerations, contact wrenches and joint torques, subject
to the full multibody dynamics.

Stage 1 decides *what is feasible* against the wheel envelope; this decides
*how to realise it* with the arms.

---

## Public API

| symbol | signature | canonical? |
|---|---|---|
| **`WholeBodyQPConfig`** *(dataclass)* |  |  |
|   `nq` | `14` | _field_ |
|   `nc_max` | `2` | _field_ |
|   `method` | `'weighted'` | _field_ |
|   `solver` | `'qpoases'` | _field_ |
|   `weight_ratio` | `1.0` | _field_ |
|   `alpha_ee` | `500.0` | _field_ |
|   `alpha_posture` | `100.0` | _field_ |
|   `alpha_wrench` | `10.0` | _field_ |
|   `alpha_torque` | `1.0` | _field_ |
|   `alpha_reg` | `0.01` | _field_ |
|   `alpha_lambda_int` | `0.0` | _field_ |
|   `ds_centroidal_mode` | `False` | _field_ |
|   `ds_alpha_com` | `100.0` | _field_ |
|   `ds_alpha_torso_ori` | `200.0` | _field_ |
|   `ds_alpha_posture` | `50.0` | _field_ |
|   `ss_two_task_mode` | `False` | _field_ |
|   `ss_alpha_mom` | `500.0` | _field_ |
|   `alpha_torso_pose` | `1000.0` | _field_ |
|   `alpha_passivity` | `1.0` | _field_ |
|   `passivity_W_budget` | `0.0` | _field_ |
|   `qp_envelope_exact` | `False` | _field_ |
|   `w_hw_slack` | `800.0` | _field_ |
|   `Kp_com` | `100.0 * np.ones(3)` | _field_ |
|   `Kd_com` | `20.0 * np.ones(3)` | _field_ |
|   `Kp_torso` | `np.array([8.0, 8.0, 8.0, 5.0, 5.0, 5.0])` | _field_ |
|   `Kd_torso` | `np.array([6.0, 6.0, 6.0, 4.0, 4.0, 4.0])` | _field_ |
|   `Kp_ee` | `80.0 * np.ones(3)` | _field_ |
|   `Kd_ee` | `15.0 * np.ones(3)` | _field_ |
|   `Kp_ee_ang` | `5.0 * np.ones(3)` | _field_ |
|   `Kd_ee_ang` | `3.0 * np.ones(3)` | _field_ |
|   `Kp_posture` | `25.0` | _field_ |
|   `Kd_posture` | `10.0` | _field_ |
|   `Kd_settle` | `10.0` | _field_ |
|   `alpha_settle` | `1000.0` | _field_ |
|   `tau_max` | `50.0 * np.ones(14)` | _field_ |
|   `qdd_max` | `50.0` | _field_ |
|   `dt_qp` | `0.008` | _field_ |
|   `f_max` | `3000.0` | _field_ |
|   `tau_contact_max` | `300.0` | _field_ |
|   `L_max` | `np.inf` | _field_ |
|   `tau_w_max` | `np.inf` | _field_ |
| **`WholeBodyQP`** |  |  |
| `.set_nominal_posture` | `(q_nom)` | **yes** |
| `.solve` | `(dq_t, q, dq, r_com_ref, v_com_ref, lambda_ref, a_com_ff...)` | **yes** |
| `._add_equality_constraints` | `(qp, H_robot, C_robot, J_contacts, Jdot_dq_contacts, con...)` | **yes** |
| `._add_inequality_constraints` | `(qp, H_robot, dq, r_com, hw_current, hw_min, hw_max, L_c...)` | **yes** |
| `._set_variable_bounds` | `(qp, contact_config, hw_constraint_active)` | **yes** |
| `._com_task_rows` | `(J_com, Jdot_dq_com, dq_robot, r_com, r_com_ref, v_com_r...)` | **yes** |
| `._compute_indices` | `()` | **yes** |
| `.n_vars` | `()` | not exercised |
| `.variable_indices` | `()` | not exercised |

---

---

## 1. Mathematical formulation

### Decision vector

```
z = [ qdd_t (6) ; qdd (nq) ; lambda (6*nc_max) ; tau_q (nq) ]
```

- `qdd_t` — floating-base (torso) acceleration
- `qdd`   — joint accelerations
- `lambda`— contact wrenches `[f1, tau1, f2, tau2]`
- `tau_q` — joint torques, sent straight to the actuators

Solving for torques **inside** the QP rather than inverting the dynamics
afterwards is what lets actuator limits be hard constraints instead of a
post-hoc clip.

### 1.1 Equality constraints

**Full robot dynamics** (paper Eq. VI-F.7):

```
H_robot * qdd_robot + C_robot = B_u * tau_q + J_robot^T * lambda
```

`H` is the mass matrix, `C` the Coriolis/centrifugal vector, `B_u` the actuation
selection matrix (zero rows on the 6 floating-base DOFs — the base is
unactuated, which is the whole difficulty of free-floating control).

**Contact acceleration**, bilateral (welded grippers, so the contact point must
not accelerate):

```
J_contact * qdd_robot = -Jdot_contact * qd_robot
```

Implemented in `_add_equality_constraints`.

### 1.2 The momentum box

The one inequality that ties stage 2 back to the wheel envelope:

```
h_min <= h_w - dt * M_lambda * lambda <= h_max
```

`M_lambda = compute_momentum_map(r_com, contact_config)` maps the contact
wrench to the rate of change of momentum the wheels will see. So the QP is
forbidden from *commanding a wrench* that would push the wheels out of their box
within one tick.

Slack variables with weight `w_hw_slack = 800` keep the QP feasible if the box
is already violated on entry — they are inactive otherwise.

Also enforced: joint torque limits, and joint acceleration limits derived from
barrier functions on the joint position bounds.

Implemented in `_add_inequality_constraints`, which returns `hw_constraint_active`
so the caller can log when the box actually bound.

### 1.3 The task stack — and why there is no null-space projection

Every task is a least-squares row block `A z = b` with weight `alpha`, and the
QP minimises the **weighted sum**. There is no projection, no cascade, no
lexicographic solve.

Single support — the two-task stack (Phase-2.1), the canonical controller:

| task | alpha | what it does |
|---|---:|---|
| torso pose, 6-D | **2000** | the fine lever on docking precision |
| swing end-effector, 6-D | **1000** | the gross-reach lever; needs >= ~1000 |
| T-MOM linear | **400** | near-inert on H_s_dot — the NMPC owns the envelope |
| posture | **20** | joint-space regularisation |

All phases add: contact-wrench tracking (`alpha_wrench = 1`), joint-torque
minimisation (`alpha_torque = 5`), acceleration regularisation
(`alpha_reg = 1`, the cost floor), and the `h_w` slack penalty.

Double support instead runs a joint-space settle, or — under
`ds_centroidal_mode` — CoM 3-D + torso-angular 3-D + posture, with energy
dissipation handled by the **passivity inequality** rather than by a cost.

### 1.4 Consequences of weight_ratio = 1

Because `weight_ratio = 1.0`, **the alpha magnitudes _are_ the hierarchy** and
the `priority=` integers on each `Task` are inert labels. Three project rules
follow directly:

- no `weight_ratio > 1`;
- no `alpha_wrench > 1` — at 100 the wrench regulariser consumed 20 % of the QP
  budget and starved the torso and EE tasks;
- **rule 14**: `alpha_torque >= ~5 * alpha_reg`. At a 1:1 ratio the single-support
  redundancy resolution degrades into a docking timeout.

Canonical ordering: `torso 2000 > EE 1000 > T-MOM 400 > posture 20 > torque 5 >
accel-reg 1 ~ wrench 1`.

This design is deliberate. A strict hierarchy would need a null-space projection
per level, which costs a pseudo-inverse per level per tick at 100 Hz, and makes
the conditioning depend on rank decisions. The weighted stack keeps one QP with
`kappa_SS(H) ~ 7.5e3`.

---

## 2. Structure of `solve()`

CLEANUP-11 extracted four helpers, all on the canonical path, taking the body
from 543 to about 346 lines:

| helper | role |
|---|---|
| `_add_equality_constraints` | dynamics + contact acceleration (1.1) |
| `_add_inequality_constraints` | boxes; returns `hw_constraint_active` (1.2) |
| `_set_variable_bounds` | bounds on the decision variables |
| `_com_task_rows` | CoM task rows; returns `(A_com, b_com)` |

⚠ Explicitly **not** done: merging or reordering the task blocks. The order
encodes the cost-assembly sequence, so changing it would be a behavioural change
dressed up as a refactor.

---

## 3. Silent canonical values (rule 5)

Eight fields are **never** overridden by `sim_loop`, so their defaults *are* the
canonical values: `method`, `solver`, `weight_ratio`, `w_hw_slack`,
`alpha_settle`, `Kd_settle`, `qdd_max`, `tau_contact_max`. Only `w_hw_slack`
appears in CLAUDE.md (`CLEANUP_CARRYOVER` C4).

Everything else in the table above is overridden per run — read the canonical
values from CLAUDE.md, not from this file.

---

## 4. Known debt

`solve()` takes **40 parameters**, of which 30 are read in exactly one block:
17 by the assembly helpers, 11 by the two-task stack, 2 by wrench tracking. Only
10 span multiple blocks. Restructuring the signature touches both call sites in
`sim_loop`, and was deliberately deferred so it would not blur the diff proving
the helper extraction was inert (`CLEANUP_CARRYOVER` A1).

The file went from 1385 to 950 lines during the chantier and is the
best-covered in the repository at **97 %**.

## See also

- package overview: [`solvers.md`](solvers.md)
