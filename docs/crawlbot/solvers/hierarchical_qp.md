# `crawlbot.solvers.hierarchical_qp`

**File**: `crawlbot/solvers/hierarchical_qp.py` — **529 lines** — canonical coverage **70 %**

> Module docstring: *"HierarchicalQP - Generic hierarchical quadratic program solver."*

Generic QP backend. Accumulates weighted least-squares tasks, adds equality
and inequality constraints and variable bounds, then hands a single dense QP to
a CasADi conic solver.

Two formulations are implemented — `strict` (cascade with null-space
projection, exact priorities) and `weighted` (one QP, approximate priorities).
**The canonical controller uses `weighted`.**

---

## Public API

| symbol | signature | canonical? |
|---|---|---|
| **`Task`** *(dataclass)* |  |  |
|   `A` |  | _field_ |
|   `b` |  | _field_ |
|   `W` |  | _field_ |
|   `priority` |  | _field_ |
| **`QPSolveInfo`** *(dataclass)* |  |  |
|   `method` | `''` | _field_ |
|   `success` | `False` | _field_ |
|   `exitflag` | `-1` | _field_ |
|   `cost` | `np.inf` | _field_ |
|   `lambda_eq` | `None` | _field_ |
|   `lambda_ineq` | `None` | _field_ |
|   `lambda_lb` | `None` | _field_ |
|   `lambda_ub` | `None` | _field_ |
|   `failed_priority` | `None` | _field_ |
|   `solve_time_ms` | `0.0` | _field_ |
|   `n_iter` | `0` | _field_ |
| **`HierarchicalQP`** |  |  |
| `.add_task` | `(A, b, W, priority)` | **yes** |
| `.add_equality_constraint` | `(C, d)` | **yes** |
| `.add_inequality_constraint` | `(C, d)` | **yes** |
| `.set_bounds` | `(lb, ub)` | **yes** |
| `.clear_tasks` | `()` | not exercised |
| `.clear_constraints` | `()` | not exercised |
| `.solve` | `(x0=None)` | **yes** |
| `._solve_weighted` | `(sorted_tasks, x0)` | **yes** |
| `._solve_strict` | `(sorted_tasks, x0)` | not exercised |
| `._solve_qp_raw` | `(H, g, C_eq, d_eq, C_ineq, d_ineq, lb, ub, x0=None)` | **yes** |
| `._get_solver_options` | `()` | **yes** |
| `.n_tasks` | `()` | not exercised |

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

## See also

- package overview: [`solvers.md`](solvers.md)
