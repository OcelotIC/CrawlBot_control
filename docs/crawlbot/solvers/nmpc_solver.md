# `crawlbot.solvers.nmpc_solver`

**File**: [`crawlbot/solvers/nmpc_solver.py`](../../../crawlbot/solvers/nmpc_solver.py) — **650 lines** — canonical coverage **95 %**

> Module docstring: *"NMPCSolver - Generic Nonlinear Model Predictive Control solver with CasADi."*

Problem-agnostic multiple-shooting NLP builder. You supply continuous
dynamics, stage and terminal costs, path and terminal constraints and bounds; it
transcribes to a CasADi NLP and solves it with IPOPT.

`centroidal_nmpc` is its only production instance. Keeping the two separate is
what lets the centroidal model be swapped or tested without touching the
transcription.

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`NMPCSolveInfo`** *(dataclass)* |  |  | [L52](../../../crawlbot/solvers/nmpc_solver.py#L52) |
|   `cost` | `np.inf` | _field_ | [L54](../../../crawlbot/solvers/nmpc_solver.py#L54) |
|   `success` | `False` | _field_ | [L55](../../../crawlbot/solvers/nmpc_solver.py#L55) |
|   `status` | `''` | _field_ | [L56](../../../crawlbot/solvers/nmpc_solver.py#L56) |
|   `iterations` | `0` | _field_ | [L57](../../../crawlbot/solvers/nmpc_solver.py#L57) |
|   `solve_time_ms` | `0.0` | _field_ | [L58](../../../crawlbot/solvers/nmpc_solver.py#L58) |
|   `solver_stats` | `None` | _field_ | [L59](../../../crawlbot/solvers/nmpc_solver.py#L59) |
| **`NMPCSolver`** |  |  | [L62](../../../crawlbot/solvers/nmpc_solver.py#L62) |
| `.set_continuous_dynamics` | `(ode_func)` | **yes** | [L138](../../../crawlbot/solvers/nmpc_solver.py#L138) |
| `.set_stage_cost` | `(cost_func)` | **yes** | [L158](../../../crawlbot/solvers/nmpc_solver.py#L158) |
| `.set_terminal_cost` | `(cost_func)` | **yes** | [L168](../../../crawlbot/solvers/nmpc_solver.py#L168) |
| `.set_path_constraints` | `(constraint_func, ng)` | **yes** | [L178](../../../crawlbot/solvers/nmpc_solver.py#L178) |
| `.set_terminal_constraints` | `(constraint_func, ng)` | **yes** | [L197](../../../crawlbot/solvers/nmpc_solver.py#L197) |
| `.set_state_bounds` | `(x_min, x_max)` | **yes** | [L210](../../../crawlbot/solvers/nmpc_solver.py#L210) |
| `.set_control_bounds` | `(u_min, u_max)` | **yes** | [L215](../../../crawlbot/solvers/nmpc_solver.py#L215) |
| `.apply_control_bounds_all_stages` | `(u_min, u_max)` | **yes** | [L220](../../../crawlbot/solvers/nmpc_solver.py#L220) |
| `.set_parameters` | `(np_)` | **yes** | [L246](../../../crawlbot/solvers/nmpc_solver.py#L246) |
| `.build` | `(solver_opts=None)` | **yes** | [L263](../../../crawlbot/solvers/nmpc_solver.py#L263) |
| `.solve` | `(x0, params=None, warm_start=True)` | **yes** | [L411](../../../crawlbot/solvers/nmpc_solver.py#L411) |
| `.shift_warm_start` | `()` | **yes** | [L528](../../../crawlbot/solvers/nmpc_solver.py#L528) |
| `.reset_warm_start` | `()` | **yes** | [L547](../../../crawlbot/solvers/nmpc_solver.py#L547) |
| `._build_initial_guess` | `(x0, warm_start)` | **yes** | [L563](../../../crawlbot/solvers/nmpc_solver.py#L563) |
| `._build_w0_from_trajectories` | `(x_traj, u_traj)` | **yes** | [L579](../../../crawlbot/solvers/nmpc_solver.py#L579) |
| `._parse_solution` | `(w)` | **yes** | [L593](../../../crawlbot/solvers/nmpc_solver.py#L593) |
| `._get_default_solver_options` | `()` | **yes** | [L608](../../../crawlbot/solvers/nmpc_solver.py#L608) |

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

## Code map

| unit | source |
|---|---|
| `class NMPCSolveInfo` | [L52-59](../../../crawlbot/solvers/nmpc_solver.py#L52-L59) |
| `class NMPCSolver` | [L62-634](../../../crawlbot/solvers/nmpc_solver.py#L62-L634) |
| `NMPCSolver.set_continuous_dynamics` | [L138-156](../../../crawlbot/solvers/nmpc_solver.py#L138-L156) |
| `NMPCSolver.set_stage_cost` | [L158-166](../../../crawlbot/solvers/nmpc_solver.py#L158-L166) |
| `NMPCSolver.set_terminal_cost` | [L168-176](../../../crawlbot/solvers/nmpc_solver.py#L168-L176) |
| `NMPCSolver.set_path_constraints` | [L178-195](../../../crawlbot/solvers/nmpc_solver.py#L178-L195) |
| `NMPCSolver.set_terminal_constraints` | [L197-208](../../../crawlbot/solvers/nmpc_solver.py#L197-L208) |
| `NMPCSolver.set_state_bounds` | [L210-213](../../../crawlbot/solvers/nmpc_solver.py#L210-L213) |
| `NMPCSolver.set_control_bounds` | [L215-218](../../../crawlbot/solvers/nmpc_solver.py#L215-L218) |
| `NMPCSolver.apply_control_bounds_all_stages` | [L220-244](../../../crawlbot/solvers/nmpc_solver.py#L220-L244) |
| `NMPCSolver.set_parameters` | [L246-257](../../../crawlbot/solvers/nmpc_solver.py#L246-L257) |
| `NMPCSolver.build` | [L263-405](../../../crawlbot/solvers/nmpc_solver.py#L263-L405) |
| `NMPCSolver.solve` | [L411-526](../../../crawlbot/solvers/nmpc_solver.py#L411-L526) |
| `NMPCSolver.shift_warm_start` | [L528-545](../../../crawlbot/solvers/nmpc_solver.py#L528-L545) |
| `NMPCSolver.reset_warm_start` | [L547-557](../../../crawlbot/solvers/nmpc_solver.py#L547-L557) |
| `NMPCSolver._build_initial_guess` | [L563-577](../../../crawlbot/solvers/nmpc_solver.py#L563-L577) |
| `NMPCSolver._build_w0_from_trajectories` | [L579-591](../../../crawlbot/solvers/nmpc_solver.py#L579-L591) |
| `NMPCSolver._parse_solution` | [L593-606](../../../crawlbot/solvers/nmpc_solver.py#L593-L606) |
| `NMPCSolver._get_default_solver_options` | [L608-623](../../../crawlbot/solvers/nmpc_solver.py#L608-L623) |

---

## See also

- package overview: [`solvers.md`](solvers.md)
