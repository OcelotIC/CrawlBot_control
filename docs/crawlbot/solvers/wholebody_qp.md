# `crawlbot.solvers.wholebody_qp`

**File**: [`crawlbot/solvers/wholebody_qp.py`](../../../crawlbot/solvers/wholebody_qp.py) — **1019 lines** — canonical coverage **96 %**

> Module docstring: *"WholeBodyQP - Whole-body Quadratic Program for high-rate tracking."*

**Stage 2 of the controller — the canonical one.** An instantaneous QP solved
at 100 Hz that turns the references (torso pose, swing end-effector, momentum,
posture) into joint accelerations, contact wrenches and joint torques, subject
to the full multibody dynamics.

Stage 1 decides *what is feasible* against the wheel envelope; this decides
*how to realise it* with the arms.

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| `as_gain_matrix` | `(g, n, name='gain')` | **yes** | [L70](../../../crawlbot/solvers/wholebody_qp.py#L70) |
| **`WholeBodyQPConfig`** *(dataclass)* |  |  | [L130](../../../crawlbot/solvers/wholebody_qp.py#L130) |
|   `nq` | `14` | _field_ | [L140](../../../crawlbot/solvers/wholebody_qp.py#L140) |
|   `nc_max` | `2` | _field_ | [L141](../../../crawlbot/solvers/wholebody_qp.py#L141) |
|   `method` | `'weighted'` | _field_ | [L144](../../../crawlbot/solvers/wholebody_qp.py#L144) |
|   `solver` | `'qpoases'` | _field_ | [L145](../../../crawlbot/solvers/wholebody_qp.py#L145) |
|   `weight_ratio` | `1.0` | _field_ | [L153](../../../crawlbot/solvers/wholebody_qp.py#L153) |
|   `alpha_ee` | `500.0` | _field_ | [L156](../../../crawlbot/solvers/wholebody_qp.py#L156) |
|   `alpha_posture` | `100.0` | _field_ | [L157](../../../crawlbot/solvers/wholebody_qp.py#L157) |
|   `alpha_wrench` | `10.0` | _field_ | [L158](../../../crawlbot/solvers/wholebody_qp.py#L158) |
|   `alpha_torque` | `1.0` | _field_ | [L159](../../../crawlbot/solvers/wholebody_qp.py#L159) |
|   `alpha_reg` | `0.01` | _field_ | [L160](../../../crawlbot/solvers/wholebody_qp.py#L160) |
|   `alpha_lambda_int` | `0.0` | _field_ | [L161](../../../crawlbot/solvers/wholebody_qp.py#L161) |
|   `ds_centroidal_mode` | `False` | _field_ | [L174](../../../crawlbot/solvers/wholebody_qp.py#L174) |
|   `ds_alpha_com` | `100.0` | _field_ | [L175](../../../crawlbot/solvers/wholebody_qp.py#L175) |
|   `ds_alpha_torso_ori` | `200.0` | _field_ | [L176](../../../crawlbot/solvers/wholebody_qp.py#L176) |
|   `ds_alpha_posture` | `50.0` | _field_ | [L177](../../../crawlbot/solvers/wholebody_qp.py#L177) |
|   `ss_two_task_mode` | `False` | _field_ | [L191](../../../crawlbot/solvers/wholebody_qp.py#L191) |
|   `ss_alpha_mom` | `500.0` | _field_ | [L192](../../../crawlbot/solvers/wholebody_qp.py#L192) |
|   `alpha_torso_pose` | `1000.0` | _field_ | [L193](../../../crawlbot/solvers/wholebody_qp.py#L193) |
|   `alpha_passivity` | `1.0` | _field_ | [L196](../../../crawlbot/solvers/wholebody_qp.py#L196) |
|   `passivity_W_budget` | `0.0` | _field_ | [L202](../../../crawlbot/solvers/wholebody_qp.py#L202) |
|   `qp_envelope_exact` | `False` | _field_ | [L206](../../../crawlbot/solvers/wholebody_qp.py#L206) |
|   `w_hw_slack` | `800.0` | _field_ | [L218](../../../crawlbot/solvers/wholebody_qp.py#L218) |
|   `Kp_com` | `100.0 * np.ones(3)` | _field_ | [L226](../../../crawlbot/solvers/wholebody_qp.py#L226) |
|   `Kd_com` | `20.0 * np.ones(3)` | _field_ | [L227](../../../crawlbot/solvers/wholebody_qp.py#L227) |
|   `Kp_torso` | `np.array([8.0, 8.0, 8.0, 5.0, 5.0, 5.0])` | _field_ | [L230](../../../crawlbot/solvers/wholebody_qp.py#L230) |
|   `Kd_torso` | `np.array([6.0, 6.0, 6.0, 4.0, 4.0, 4.0])` | _field_ | [L231](../../../crawlbot/solvers/wholebody_qp.py#L231) |
|   `Kp_ee` | `80.0 * np.ones(3)` | _field_ | [L234](../../../crawlbot/solvers/wholebody_qp.py#L234) |
|   `Kd_ee` | `15.0 * np.ones(3)` | _field_ | [L235](../../../crawlbot/solvers/wholebody_qp.py#L235) |
|   `Kp_ee_ang` | `5.0 * np.ones(3)` | _field_ | [L236](../../../crawlbot/solvers/wholebody_qp.py#L236) |
|   `Kd_ee_ang` | `3.0 * np.ones(3)` | _field_ | [L237](../../../crawlbot/solvers/wholebody_qp.py#L237) |
|   `Kp_posture` | `25.0` | _field_ | [L240](../../../crawlbot/solvers/wholebody_qp.py#L240) |
|   `Kd_posture` | `10.0` | _field_ | [L241](../../../crawlbot/solvers/wholebody_qp.py#L241) |
|   `Kd_settle` | `10.0` | _field_ | [L244](../../../crawlbot/solvers/wholebody_qp.py#L244) |
|   `alpha_settle` | `1000.0` | _field_ | [L245](../../../crawlbot/solvers/wholebody_qp.py#L245) |
|   `tau_max` | `50.0 * np.ones(14)` | _field_ | [L248](../../../crawlbot/solvers/wholebody_qp.py#L248) |
|   `qdd_max` | `50.0` | _field_ | [L251](../../../crawlbot/solvers/wholebody_qp.py#L251) |
|   `dt_qp` | `0.008` | _field_ | [L254](../../../crawlbot/solvers/wholebody_qp.py#L254) |
|   `f_max` | `3000.0` | _field_ | [L257](../../../crawlbot/solvers/wholebody_qp.py#L257) |
|   `tau_contact_max` | `300.0` | _field_ | [L258](../../../crawlbot/solvers/wholebody_qp.py#L258) |
|   `L_max` | `np.inf` | _field_ | [L261](../../../crawlbot/solvers/wholebody_qp.py#L261) |
|   `tau_w_max` | `np.inf` | _field_ | [L262](../../../crawlbot/solvers/wholebody_qp.py#L262) |
| **`WholeBodyQP`** |  |  | [L265](../../../crawlbot/solvers/wholebody_qp.py#L265) |
| `.set_nominal_posture` | `(q_nom)` | **yes** | [L313](../../../crawlbot/solvers/wholebody_qp.py#L313) |
| `.solve` | `(dq_t, q, dq, r_com_ref, v_com_ref, lambda_ref, a_com_ff...)` | **yes** | [L323](../../../crawlbot/solvers/wholebody_qp.py#L323) |
| `._add_equality_constraints` | `(qp, H_robot, C_robot, J_contacts, Jdot_dq_contacts, con...)` | **yes** | [L737](../../../crawlbot/solvers/wholebody_qp.py#L737) |
| `._add_inequality_constraints` | `(qp, H_robot, dq, r_com, hw_current, hw_min, hw_max, L_c...)` | **yes** | [L796](../../../crawlbot/solvers/wholebody_qp.py#L796) |
| `._set_variable_bounds` | `(qp, contact_config, hw_constraint_active)` | **yes** | [L907](../../../crawlbot/solvers/wholebody_qp.py#L907) |
| `._com_task_rows` | `(J_com, Jdot_dq_com, dq_robot, r_com, r_com_ref, v_com_r...)` | **yes** | [L953](../../../crawlbot/solvers/wholebody_qp.py#L953) |
| `._compute_indices` | `()` | **yes** | [L983](../../../crawlbot/solvers/wholebody_qp.py#L983) |
| `.n_vars` | `()` | not exercised | [L1006](../../../crawlbot/solvers/wholebody_qp.py#L1006) |
| `.variable_indices` | `()` | not exercised | [L1010](../../../crawlbot/solvers/wholebody_qp.py#L1010) |

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

### 1.5 The PD gain handover contract (`as_gain_matrix`)

Every task PD gain reaches this module through `as_gain_matrix(g, n, name)`,
which accepts a **scalar**, an **`(n,)` vector**, or a **diagonal `(n, n)`
matrix** and returns the `(n, n)` matrix. A full (non-diagonal) matrix is
**rejected**, not truncated.

This helper exists because of a defect the whole project history carried.
`Kp_com` is *declared* `(3,)` but **every** caller passes a matrix —
`sim_loop._build_qp` sends `np.diag([kpc]*3)`, and the `Misc/` baselines send
`np.diag([3., 3., 5.])` and `np.diag([50., 50., 100.])`. The consumer used to be
a bare `np.diag(cfg.Kp_com)`, which is shape-polymorphic in the worst possible
way:

| input | `np.diag` does | result |
|---|---|---|
| `(3,)` vector | **builds** the matrix | `diag(k)` — correct |
| `(3,3)` matrix | **extracts** the diagonal | `k` as a vector — wrong |

On the extracted vector, `k @ e` is a **scalar**, which NumPy then broadcast
back over all three axes. The applied law was therefore

$$a_\text{fb} = \Big(\textstyle\sum_i k_i e_i\Big)\,[1,1,1]^\top \quad\text{(rank 1)}
\qquad\text{instead of}\qquad a_\text{fb} = \mathrm{diag}(k)\,e \quad\text{(rank 3)}$$

Both are `(3,)` and finite, so nothing raised and no assertion fired. Three
consequences, all measured on the frozen canonical:

- **axes are coupled** — an x-only error commands equal y and z acceleration
  (`e=[1,0,0]` mm: intended `[3,0,0]`, applied `[3,3,3]` mm/s²);
- **anisotropy is destroyed** — `diag([3,3,5])` and `diag([5,3,3])` become the
  same operator, which silently voids the point of the `Misc/` gain choices;
- **any error orthogonal to `[1,1,1]` is invisible** — it sums to zero, so the
  feedback is *exactly* zero. **81 % of the canonical single-support CoM error
  lies in that subspace** (median; max 99.99 %).

The defect was **live on 8458/8458 canonical solves**, and the applied term
differed from the intended one by **134 % of the intended term's own magnitude**
in SS (104 % in DS) — it was a different vector, not a perturbed one.

⚠ It nonetheless moved the docks by **≤ 0.02 mm** (all six still under the 5 mm
capture radius; the tightest went 4.99 → 4.98 mm) and left `e_com` peak
**unchanged at 0.154 m**. That is not because the bug was benign — it is because
the CoM task's authority is bounded by something else entirely, which is the
subject of §4. Fixing the gain does **not** improve CoM tracking.

The same normalizer is applied to `Kp_torso`/`Kd_torso` (which arrive as `(6,)`
and were always correct) purely so that the contract is uniform and this class
of bug is unrepresentable. Regression cover:
`tests/test_reworked_qp.py::TestGainSemantics` — six tests, all six proven to
fail against the pre-fix expression by `scripts/audit_com_gain_bite_check.py`.

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
best-covered in the repository (the header figure above is regenerated; the
`as_gain_matrix` rejection branches are deliberately not on the canonical path,
which is why it is no longer 97 %).

**The CoM task differences the NMPC plan, not the tracking error.** `_com_task_rows`
receives `r_com_ref` from `sim_loop.py:2549`, which passes `rp_interp` — the
NMPC's *own* planned CoM trajectory, interpolated across the QP sub-steps. The
planner reference `cref_r` goes to the **NMPC** (`sim_loop.py:2243`) and is what
gets logged and exported (`sim_loop.py:2969`). The two differ by two orders of
magnitude:

| quantity | source | SS median | SS max |
|---|---|---|---|
| `\|e_r\|` the QP task sees | `rp_interp` (NMPC plan) | 0.165 mm | **1.147 mm** |
| `\|e_com\|` closed loop | `cref_r` (planner) | 61.5 mm | **153.7 mm** |

Because the NMPC re-anchors its plan to the *measured* CoM every 100 ms, `e_r` is
identically zero on 508/5080 SS ticks — exactly one per NMPC cycle — and never
exceeds ~1 mm. The CoM task is therefore a **plan-follower with ~1 mm of
authority**; the 61–154 mm planner-tracking error is owned by the NMPC, and no
gain change in this module can address it. The PD term is 9.8 % of `|a_ff|` in SS
(4.46 vs 54.2 mm/s²), and the stack delivers a median **41 %** of the CoM
acceleration it commands (0.09 % in DS, where `ds_alpha_com=100` leaves the task
nearly inert).

This is a **structural** observation, not a proposed change: rerouting the task
to the planner reference would be a control-architecture decision, and the DS
weighting question is a separate one that rule 12 forbids moving in the same
experiment.

## Code map

| unit | source |
|---|---|
| `as_gain_matrix()` | [L70-126](../../../crawlbot/solvers/wholebody_qp.py#L70-L126) |
| `class WholeBodyQPConfig` | [L130-262](../../../crawlbot/solvers/wholebody_qp.py#L130-L262) |
| `class WholeBodyQP` | [L265-1018](../../../crawlbot/solvers/wholebody_qp.py#L265-L1018) |
| `WholeBodyQP.set_nominal_posture` | [L313-321](../../../crawlbot/solvers/wholebody_qp.py#L313-L321) |
| `WholeBodyQP.solve` | [L323-723](../../../crawlbot/solvers/wholebody_qp.py#L323-L723) |
| `WholeBodyQP._add_equality_constraints` | [L737-794](../../../crawlbot/solvers/wholebody_qp.py#L737-L794) |
| `WholeBodyQP._add_inequality_constraints` | [L796-905](../../../crawlbot/solvers/wholebody_qp.py#L796-L905) |
| `WholeBodyQP._set_variable_bounds` | [L907-951](../../../crawlbot/solvers/wholebody_qp.py#L907-L951) |
| `WholeBodyQP._com_task_rows` | [L953-981](../../../crawlbot/solvers/wholebody_qp.py#L953-L981) |
| `WholeBodyQP._compute_indices` | [L983-1003](../../../crawlbot/solvers/wholebody_qp.py#L983-L1003) |
| `WholeBodyQP.n_vars` | [L1006-1007](../../../crawlbot/solvers/wholebody_qp.py#L1006-L1007) |
| `WholeBodyQP.variable_indices` | [L1010-1012](../../../crawlbot/solvers/wholebody_qp.py#L1010-L1012) |

---

## See also

- package overview: [`solvers.md`](solvers.md)
