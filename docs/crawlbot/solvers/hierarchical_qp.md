# `crawlbot.solvers.hierarchical_qp`

**File**: [`crawlbot/solvers/hierarchical_qp.py`](../../../crawlbot/solvers/hierarchical_qp.py) — **568 lines** — canonical coverage **70 %**

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
|   `solver_success` | `None` | _field_ | [L80](../../../crawlbot/solvers/hierarchical_qp.py#L80) |
|   `return_status` | `''` | _field_ | [L81](../../../crawlbot/solvers/hierarchical_qp.py#L81) |
| **`HierarchicalQP`** |  |  | [L84](../../../crawlbot/solvers/hierarchical_qp.py#L84) |
| `.add_task` | `(A, b, W, priority)` | **yes** | [L143](../../../crawlbot/solvers/hierarchical_qp.py#L143) |
| `.add_equality_constraint` | `(C, d)` | **yes** | [L185](../../../crawlbot/solvers/hierarchical_qp.py#L185) |
| `.add_inequality_constraint` | `(C, d)` | **yes** | [L196](../../../crawlbot/solvers/hierarchical_qp.py#L196) |
| `.set_bounds` | `(lb, ub)` | **yes** | [L207](../../../crawlbot/solvers/hierarchical_qp.py#L207) |
| `.clear_tasks` | `()` | not exercised | [L212](../../../crawlbot/solvers/hierarchical_qp.py#L212) |
| `.clear_constraints` | `()` | not exercised | [L216](../../../crawlbot/solvers/hierarchical_qp.py#L216) |
| `.solve` | `(x0=None)` | **yes** | [L228](../../../crawlbot/solvers/hierarchical_qp.py#L228) |
| `._solve_weighted` | `(sorted_tasks, x0)` | **yes** | [L266](../../../crawlbot/solvers/hierarchical_qp.py#L266) |
| `._solve_strict` | `(sorted_tasks, x0)` | not exercised | [L315](../../../crawlbot/solvers/hierarchical_qp.py#L315) |
| `._solve_qp_raw` | `(H, g, C_eq, d_eq, C_ineq, d_ineq, lb, ub, x0=None)` | **yes** | [L397](../../../crawlbot/solvers/hierarchical_qp.py#L397) |
| `._get_solver_options` | `()` | **yes** | [L532](../../../crawlbot/solvers/hierarchical_qp.py#L532) |
| `.n_tasks` | `()` | not exercised | [L560](../../../crawlbot/solvers/hierarchical_qp.py#L560) |

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

### ⚠ `info.success` cannot report a solver failure

`_get_solver_options` sets `error_on_fail: False`. That is what the option
says on the tin: CasADi **returns normally** instead of raising when the
backend fails. But `_solve_qp_raw` derives its outcome from whether the call
raised —

```python
try:    sol = solver(**solve_args); success = True;  exitflag = 0
except RuntimeError: ...            success = False; exitflag = -1
```

— so on the weighted path `success` is **True for an infeasible QP as well as a
feasible one**, and `exitflag` is 0 either way. Verified directly: a QP with
`lba = uba = 5` against `lbx = ubx = ±1` makes qpOASES print *"Premature
homotopy termination because QP is infeasible"*, and the Python call returns
without raising.

Everything downstream inherits the blindness. `WholeBodyQP` passes `info`
through, `sim_loop` sets `qp_ok` from the same try/except, and the fulldiag
`qp_ok` column is therefore **structurally incapable of ever being 0** on the
canonical path. "0 QP failures over the run" is not a measurement; it is the
only value that column can take.

**What to use instead.** `casadi.Function.stats()` carries the backend's own
verdict, and `_solve_qp_raw` now reads it into three additive `QPSolveInfo`
fields:

| field | source | meaning |
|---|---|---|
| `solver_success` | `stats()['success']` | the backend's verdict; `None` if stats are unavailable |
| `return_status` | `stats()['return_status']` | its own string, e.g. *"Initial QP could not be solved due to infeasibility!"* |
| `n_iter` | `stats()['iter_count']` | qpOASES working-set recalculations. Previously declared on `QPSolveInfo` and **never assigned** — it read 0 for every solve |

These are **additive**: `success`, `exitflag` and `cost` keep their old values
bit-for-bit, so `qp_ok` stays byte-stable and the frozen baseline is untouched.
The honest fix — deriving `success` from `stats()` — changes an existing
fulldiag column and needs the baseline regenerated first.

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
| `class QPSolveInfo` | [L55-81](../../../crawlbot/solvers/hierarchical_qp.py#L55-L81) |
| `class HierarchicalQP` | [L84-567](../../../crawlbot/solvers/hierarchical_qp.py#L84-L567) |
| `HierarchicalQP.add_task` | [L143-183](../../../crawlbot/solvers/hierarchical_qp.py#L143-L183) |
| `HierarchicalQP.add_equality_constraint` | [L185-194](../../../crawlbot/solvers/hierarchical_qp.py#L185-L194) |
| `HierarchicalQP.add_inequality_constraint` | [L196-205](../../../crawlbot/solvers/hierarchical_qp.py#L196-L205) |
| `HierarchicalQP.set_bounds` | [L207-210](../../../crawlbot/solvers/hierarchical_qp.py#L207-L210) |
| `HierarchicalQP.clear_tasks` | [L212-214](../../../crawlbot/solvers/hierarchical_qp.py#L212-L214) |
| `HierarchicalQP.clear_constraints` | [L216-221](../../../crawlbot/solvers/hierarchical_qp.py#L216-L221) |
| `HierarchicalQP.solve` | [L228-260](../../../crawlbot/solvers/hierarchical_qp.py#L228-L260) |
| `HierarchicalQP._solve_weighted` | [L266-309](../../../crawlbot/solvers/hierarchical_qp.py#L266-L309) |
| `HierarchicalQP._solve_strict` | [L315-391](../../../crawlbot/solvers/hierarchical_qp.py#L315-L391) |
| `HierarchicalQP._solve_qp_raw` | [L397-530](../../../crawlbot/solvers/hierarchical_qp.py#L397-L530) |
| `HierarchicalQP._get_solver_options` | [L532-553](../../../crawlbot/solvers/hierarchical_qp.py#L532-L553) |
| `HierarchicalQP.n_tasks` | [L560-561](../../../crawlbot/solvers/hierarchical_qp.py#L560-L561) |

---

## See also

- package overview: [`solvers.md`](solvers.md)
