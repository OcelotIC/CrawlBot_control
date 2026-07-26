# `crawlbot.solvers.nmpc_solver`

**File**: `crawlbot/solvers/nmpc_solver.py` — **650 lines** — canonical coverage **95 %**

> Module docstring: *"NMPCSolver - Generic Nonlinear Model Predictive Control solver with CasADi."*

Problem-agnostic multiple-shooting NLP builder. You supply continuous
dynamics, stage and terminal costs, path and terminal constraints and bounds; it
transcribes to a CasADi NLP and solves it with IPOPT.

`centroidal_nmpc` is its only production instance. Keeping the two separate is
what lets the centroidal model be swapped or tested without touching the
transcription.

---

## Public API

| symbol | signature | canonical? |
|---|---|---|
| **`NMPCSolveInfo`** *(dataclass)* |  |  |
|   `cost` | `np.inf` | _field_ |
|   `success` | `False` | _field_ |
|   `status` | `''` | _field_ |
|   `iterations` | `0` | _field_ |
|   `solve_time_ms` | `0.0` | _field_ |
|   `solver_stats` | `None` | _field_ |
| **`NMPCSolver`** |  |  |
| `.set_continuous_dynamics` | `(ode_func)` | **yes** |
| `.set_stage_cost` | `(cost_func)` | **yes** |
| `.set_terminal_cost` | `(cost_func)` | **yes** |
| `.set_path_constraints` | `(constraint_func, ng)` | **yes** |
| `.set_terminal_constraints` | `(constraint_func, ng)` | **yes** |
| `.set_state_bounds` | `(x_min, x_max)` | **yes** |
| `.set_control_bounds` | `(u_min, u_max)` | **yes** |
| `.apply_control_bounds_all_stages` | `(u_min, u_max)` | **yes** |
| `.set_parameters` | `(np_)` | **yes** |
| `.build` | `(solver_opts=None)` | **yes** |
| `.solve` | `(x0, params=None, warm_start=True)` | **yes** |
| `.shift_warm_start` | `()` | **yes** |
| `.reset_warm_start` | `()` | **yes** |
| `._build_initial_guess` | `(x0, warm_start)` | **yes** |
| `._build_w0_from_trajectories` | `(x_traj, u_traj)` | **yes** |
| `._parse_solution` | `(w)` | **yes** |
| `._get_default_solver_options` | `()` | **yes** |

---

---

## 1. Multiple shooting

Decision variables are the states at every node **and** the controls on every
interval:

```
w = [ x_0, u_0, x_1, u_1, ..., u_{N-1}, x_N ]
```

with the dynamics imposed as equality constraints between consecutive nodes:

```
x_{k+1} - Phi(x_k, u_k, p) = 0        k = 0 .. N-1
```

`Phi` is one RK4 step of the user-supplied continuous ODE over `dt`.

Compared with single shooting this gives a larger but far better-conditioned
problem: the nonlinearity is spread across nodes instead of compounding through
one long integration, and every node can be warm-started independently.

## 2. Build once, solve many

`build()` performs the CasADi transcription and code generation, which is
expensive. It is called **once**, at simulation `setup()`. Subsequent `solve()`
calls only update the parameter vector `p` and the bounds, then re-run IPOPT.

This is why the problem dimensions are fixed at build time and inactive contacts
are zeroed *by bounds* rather than removed (see `centroidal_nmpc.md`).

## 3. Two contributions from the chantier

**`apply_control_bounds_all_stages`** (CLEANUP-1/3) applies control bounds
across every stage of the horizon in one call, replacing a scattered loop.

**Fix F2** (CLEANUP-3) — the important one. The warm start is now reused only
when `info.success` is true:

> Previously a failed solve could seed the next one. On a horizon that is
> re-solved at 10 Hz, that propagates a diverging iterate forward instead of
> restarting clean.

## 4. Coverage

**95 %** — every public method is on the canonical path. The unexercised
remainder is failure handling, dead because no solve fails.

## See also

- package overview: [`solvers.md`](solvers.md)
