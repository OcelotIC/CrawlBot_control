# `crawlbot.solvers.hierarchical_qp`

**File**: [`crawlbot/solvers/hierarchical_qp.py`](../../../crawlbot/solvers/hierarchical_qp.py) — **529 lines** — canonical coverage **70 %**

> Module docstring: *"HierarchicalQP - Generic hierarchical quadratic program solver."*

Generic QP backend. Accumulates weighted least-squares tasks, adds equality
and inequality constraints and variable bounds, then hands a single dense QP to
a CasADi conic solver.

Two formulations are implemented — `strict` (cascade with null-space
projection, exact priorities) and `weighted` (one QP, approximate priorities).
**The canonical controller uses `weighted`.**

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`Task`** *(dataclass)* |  |  | [L46](../../../crawlbot/solvers/hierarchical_qp.py#L46) |
|   `A` | `` | _field_ | [L48](../../../crawlbot/solvers/hierarchical_qp.py#L48) |
|   `b` | `` | _field_ | [L49](../../../crawlbot/solvers/hierarchical_qp.py#L49) |
|   `W` | `` | _field_ | [L50](../../../crawlbot/solvers/hierarchical_qp.py#L50) |
|   `priority` | `` | _field_ | [L51](../../../crawlbot/solvers/hierarchical_qp.py#L51) |
| **`QPSolveInfo`** *(dataclass)* |  |  | [L55](../../../crawlbot/solvers/hierarchical_qp.py#L55) |
|   `method` | `''` | _field_ | [L57](../../../crawlbot/solvers/hierarchical_qp.py#L57) |
|   `success` | `False` | _field_ | [L58](../../../crawlbot/solvers/hierarchical_qp.py#L58) |
|   `exitflag` | `-1` | _field_ | [L59](../../../crawlbot/solvers/hierarchical_qp.py#L59) |
|   `cost` | `np.inf` | _field_ | [L60](../../../crawlbot/solvers/hierarchical_qp.py#L60) |
|   `lambda_eq` | `None` | _field_ | [L61](../../../crawlbot/solvers/hierarchical_qp.py#L61) |
|   `lambda_ineq` | `None` | _field_ | [L62](../../../crawlbot/solvers/hierarchical_qp.py#L62) |
|   `lambda_lb` | `None` | _field_ | [L63](../../../crawlbot/solvers/hierarchical_qp.py#L63) |
|   `lambda_ub` | `None` | _field_ | [L64](../../../crawlbot/solvers/hierarchical_qp.py#L64) |
|   `failed_priority` | `None` | _field_ | [L65](../../../crawlbot/solvers/hierarchical_qp.py#L65) |
|   `solve_time_ms` | `0.0` | _field_ | [L66](../../../crawlbot/solvers/hierarchical_qp.py#L66) |
|   `n_iter` | `0` | _field_ | [L67](../../../crawlbot/solvers/hierarchical_qp.py#L67) |
| **`HierarchicalQP`** |  |  | [L70](../../../crawlbot/solvers/hierarchical_qp.py#L70) |
| `.add_task` | `(A, b, W, priority)` | **yes** | [L129](../../../crawlbot/solvers/hierarchical_qp.py#L129) |
| `.add_equality_constraint` | `(C, d)` | **yes** | [L171](../../../crawlbot/solvers/hierarchical_qp.py#L171) |
| `.add_inequality_constraint` | `(C, d)` | **yes** | [L182](../../../crawlbot/solvers/hierarchical_qp.py#L182) |
| `.set_bounds` | `(lb, ub)` | **yes** | [L193](../../../crawlbot/solvers/hierarchical_qp.py#L193) |
| `.clear_tasks` | `()` | not exercised | [L198](../../../crawlbot/solvers/hierarchical_qp.py#L198) |
| `.clear_constraints` | `()` | not exercised | [L202](../../../crawlbot/solvers/hierarchical_qp.py#L202) |
| `.solve` | `(x0=None)` | **yes** | [L214](../../../crawlbot/solvers/hierarchical_qp.py#L214) |
| `._solve_weighted` | `(sorted_tasks, x0)` | **yes** | [L252](../../../crawlbot/solvers/hierarchical_qp.py#L252) |
| `._solve_strict` | `(sorted_tasks, x0)` | not exercised | [L297](../../../crawlbot/solvers/hierarchical_qp.py#L297) |
| `._solve_qp_raw` | `(H, g, C_eq, d_eq, C_ineq, d_ineq, lb, ub, x0=None)` | **yes** | [L379](../../../crawlbot/solvers/hierarchical_qp.py#L379) |
| `._get_solver_options` | `()` | **yes** | [L493](../../../crawlbot/solvers/hierarchical_qp.py#L493) |
| `.n_tasks` | `()` | not exercised | [L521](../../../crawlbot/solvers/hierarchical_qp.py#L521) |

---

---

## 1. The problem

Per task:

```
minimize    (1/2) || A_i x - b_i ||^2_{W_i}
subject to  C_eq   x  = d_eq
            C_ineq x <= d_ineq
            lb <= x <= ub
```

## 2. Weighted mode — how the stack is actually assembled

`_solve_weighted` builds one dense Hessian by accumulation:

```python
for task in sorted_tasks:
    w_p      = weight_ratio ** (task.priority - 1)
    W_scaled = task.W / w_p
    AtW      = task.A.T @ W_scaled
    H       += AtW @ task.A          # (1/2) x^T (A^T W A) x
    g       -= AtW @ task.b          # - (A^T W b)^T x
H += regularization * I
H  = 0.5 * (H + H^T)                 # re-symmetrise
```

Expanding `sum_i || A_i x - b_i ||^2_{W_i}` gives exactly `x^T H x + 2 g^T x`
plus a constant, so the accumulation *is* the sum of the task costs.

### The consequence that matters

`w_p = weight_ratio^(priority-1)`, and the canonical `weight_ratio = 1.0`.
Therefore `w_p = 1` for **every** task, whatever its priority:

> **At `weight_ratio = 1` the `priority` field is inert. The magnitudes of `W`
> are the entire hierarchy.**

This is why `wholebody_qp` tunes alphas spanning 2000 down to 1 rather than
assigning priority levels, and why raising `weight_ratio` above 1 is forbidden:
it would reintroduce a second, hidden hierarchy on top of the alphas.

## 3. Conditioning

The Tikhonov term `regularization * I` keeps `H` positive definite. Canonically
`eps = 1e-6`, and it is **inert in practice**: the task-only Hessian already has
`lambda_min(H_LS) = 1`, six orders of magnitude above eps.

Measured canonical conditioning: **`kappa_SS(H) ~ 7.5e3`** — about 530x better
than before the 2.5 freeze.

⚠ `gate/replay_canonical.py` sets `regularization = 1e-6` explicitly rather than
relying on the default, because byte-identical reproduction of the frozen
artifacts depends on it.

The `0.5 * (H + H^T)` line is not cosmetic: accumulated floating-point error
makes `H` very slightly asymmetric, and qpOASES assumes symmetry.

## 4. Backends

`qpoases` (active-set, warm-startable — the canonical choice, good for the
small dense QPs here) or `osqp` (ADMM, better for large sparse problems).
Selected by `WholeBodyQPConfig.solver`, which is never overridden, so `qpoases`
is the canonical value by default.

## 5. `_solve_strict` — unexercised, and not obviously removable

76 lines, the largest unexercised block left in `crawlbot/`. It implements the
Escande-style cascade: solve priority 1, project the residual freedom into its
null space, solve priority 2 there, and so on.

Dead because `method='weighted'` canonically — but **2 tests and 6 scripts use
it**. Removing it is a decision about whether the strict-hierarchy path stays
reproducible, not an obvious cleanup (`CLEANUP_CARRYOVER` B2).

## 6. References

- Escande et al., *Hierarchical quadratic programming: fast online humanoid-robot
  motion generation*, IJRR 2014.
- Wensing et al., *Optimization-based control for dynamic legged robots*,
  IEEE T-RO 2024.

## Code map

| unit | source |
|---|---|
| `class Task` | [L46-51](../../../crawlbot/solvers/hierarchical_qp.py#L46-L51) |
| `class QPSolveInfo` | [L55-67](../../../crawlbot/solvers/hierarchical_qp.py#L55-L67) |
| `class HierarchicalQP` | [L70-528](../../../crawlbot/solvers/hierarchical_qp.py#L70-L528) |
| `HierarchicalQP.add_task` | [L129-169](../../../crawlbot/solvers/hierarchical_qp.py#L129-L169) |
| `HierarchicalQP.add_equality_constraint` | [L171-180](../../../crawlbot/solvers/hierarchical_qp.py#L171-L180) |
| `HierarchicalQP.add_inequality_constraint` | [L182-191](../../../crawlbot/solvers/hierarchical_qp.py#L182-L191) |
| `HierarchicalQP.set_bounds` | [L193-196](../../../crawlbot/solvers/hierarchical_qp.py#L193-L196) |
| `HierarchicalQP.clear_tasks` | [L198-200](../../../crawlbot/solvers/hierarchical_qp.py#L198-L200) |
| `HierarchicalQP.clear_constraints` | [L202-207](../../../crawlbot/solvers/hierarchical_qp.py#L202-L207) |
| `HierarchicalQP.solve` | [L214-246](../../../crawlbot/solvers/hierarchical_qp.py#L214-L246) |
| `HierarchicalQP._solve_weighted` | [L252-291](../../../crawlbot/solvers/hierarchical_qp.py#L252-L291) |
| `HierarchicalQP._solve_strict` | [L297-373](../../../crawlbot/solvers/hierarchical_qp.py#L297-L373) |
| `HierarchicalQP._solve_qp_raw` | [L379-491](../../../crawlbot/solvers/hierarchical_qp.py#L379-L491) |
| `HierarchicalQP._get_solver_options` | [L493-514](../../../crawlbot/solvers/hierarchical_qp.py#L493-L514) |
| `HierarchicalQP.n_tasks` | [L521-522](../../../crawlbot/solvers/hierarchical_qp.py#L521-L522) |

---

## See also

- package overview: [`solvers.md`](solvers.md)
