# `crawlbot.solvers.nmpc_solver`

**File**: [`crawlbot/solvers/nmpc_solver.py`](../../../crawlbot/solvers/nmpc_solver.py) — **708 lines** — canonical coverage **94 %**

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
| `.set_continuous_dynamics` | `(ode_func)` | **yes** | [L140](../../../crawlbot/solvers/nmpc_solver.py#L140) |
| `.set_stage_cost` | `(cost_func)` | **yes** | [L160](../../../crawlbot/solvers/nmpc_solver.py#L160) |
| `.set_terminal_cost` | `(cost_func)` | **yes** | [L170](../../../crawlbot/solvers/nmpc_solver.py#L170) |
| `.set_path_constraints` | `(constraint_func, ng)` | **yes** | [L180](../../../crawlbot/solvers/nmpc_solver.py#L180) |
| `.set_terminal_constraints` | `(constraint_func, ng)` | **yes** | [L199](../../../crawlbot/solvers/nmpc_solver.py#L199) |
| `.set_state_bounds` | `(x_min, x_max)` | **yes** | [L212](../../../crawlbot/solvers/nmpc_solver.py#L212) |
| `.set_control_bounds` | `(u_min, u_max)` | **yes** | [L217](../../../crawlbot/solvers/nmpc_solver.py#L217) |
| `.apply_control_bounds_all_stages` | `(u_min, u_max)` | **yes** | [L222](../../../crawlbot/solvers/nmpc_solver.py#L222) |
| `.set_parameters` | `(np_, per_stage=False)` | **yes** | [L248](../../../crawlbot/solvers/nmpc_solver.py#L248) |
| `.n_param_blocks` | `()` | **yes** | [L281](../../../crawlbot/solvers/nmpc_solver.py#L281) |
| `.build` | `(solver_opts=None)` | **yes** | [L289](../../../crawlbot/solvers/nmpc_solver.py#L289) |
| `.solve` | `(x0, params=None, warm_start=True)` | **yes** | [L451](../../../crawlbot/solvers/nmpc_solver.py#L451) |
| `.shift_warm_start` | `()` | **yes** | [L586](../../../crawlbot/solvers/nmpc_solver.py#L586) |
| `.reset_warm_start` | `()` | **yes** | [L605](../../../crawlbot/solvers/nmpc_solver.py#L605) |
| `._build_initial_guess` | `(x0, warm_start)` | **yes** | [L621](../../../crawlbot/solvers/nmpc_solver.py#L621) |
| `._build_w0_from_trajectories` | `(x_traj, u_traj)` | **yes** | [L637](../../../crawlbot/solvers/nmpc_solver.py#L637) |
| `._parse_solution` | `(w)` | **yes** | [L651](../../../crawlbot/solvers/nmpc_solver.py#L651) |
| `._get_default_solver_options` | `()` | **yes** | [L666](../../../crawlbot/solvers/nmpc_solver.py#L666) |

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

### 1.1 Parameter blocks: setpoint vs trajectory (F1)

`set_parameters(np_, per_stage=...)` chooses how the parameter vector is laid
out, and that choice decides whether the NLP can track a **trajectory** at all.

| mode | vector | stage `k` reads | reference can vary along the horizon? |
|---|---|---|---|
| `per_stage=False` (legacy) | `[x0; p]` | the *same* `p` | **no** — it is a setpoint |
| `per_stage=True` | `[x0; p_0; …; p_N]` | `p_k` (terminal reads `p_N`) | **yes** |

The symbol passed to `set_stage_cost` / `set_path_constraints` /
`set_continuous_dynamics` has the same shape either way, so the user callbacks
are written identically and are unaware of the choice. Only `build()` slices
differently, via a local `p_at(k)`.

**Why it matters.** Under the legacy layout a reference carried in `p` is
constant over the whole horizon, so a caller who wants tracking has to sample
its reference in the *future* to avoid systematic lag — which silently couples
the horizon length to the target. That is exactly what happened here: `nmpc_N`
was two knobs (`NMPC_AUDIT` F1, `NMPC_HORIZON_N15` §1).

**Equivalence.** Feeding a `per_stage=True` problem a single `(np_,)` vector
broadcasts it to every block and reproduces the legacy NLP **exactly** — proven
to Δcost = Δsolution = 0.000e+00 by `scripts/audit_nmpc_f1_equivalence.py`.
That is deliberate: it makes the switch auditable, because any later
behavioural difference is attributable to the reference varying rather than to
the refactor. `solve()` rejects a parameter vector whose length matches neither
one block nor `n_param_blocks`, so a wrong knot count raises instead of being
broadcast or truncated.

The problem *size* is unchanged — same decision variables, same constraint rows.
Only the parameterization grows, from `nx + np_` to `nx + (N+1)·np_`.

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
| `class NMPCSolver` | [L62-692](../../../crawlbot/solvers/nmpc_solver.py#L62-L692) |
| `NMPCSolver.set_continuous_dynamics` | [L140-158](../../../crawlbot/solvers/nmpc_solver.py#L140-L158) |
| `NMPCSolver.set_stage_cost` | [L160-168](../../../crawlbot/solvers/nmpc_solver.py#L160-L168) |
| `NMPCSolver.set_terminal_cost` | [L170-178](../../../crawlbot/solvers/nmpc_solver.py#L170-L178) |
| `NMPCSolver.set_path_constraints` | [L180-197](../../../crawlbot/solvers/nmpc_solver.py#L180-L197) |
| `NMPCSolver.set_terminal_constraints` | [L199-210](../../../crawlbot/solvers/nmpc_solver.py#L199-L210) |
| `NMPCSolver.set_state_bounds` | [L212-215](../../../crawlbot/solvers/nmpc_solver.py#L212-L215) |
| `NMPCSolver.set_control_bounds` | [L217-220](../../../crawlbot/solvers/nmpc_solver.py#L217-L220) |
| `NMPCSolver.apply_control_bounds_all_stages` | [L222-246](../../../crawlbot/solvers/nmpc_solver.py#L222-L246) |
| `NMPCSolver.set_parameters` | [L248-278](../../../crawlbot/solvers/nmpc_solver.py#L248-L278) |
| `NMPCSolver.n_param_blocks` | [L281-283](../../../crawlbot/solvers/nmpc_solver.py#L281-L283) |
| `NMPCSolver.build` | [L289-445](../../../crawlbot/solvers/nmpc_solver.py#L289-L445) |
| `NMPCSolver.solve` | [L451-584](../../../crawlbot/solvers/nmpc_solver.py#L451-L584) |
| `NMPCSolver.shift_warm_start` | [L586-603](../../../crawlbot/solvers/nmpc_solver.py#L586-L603) |
| `NMPCSolver.reset_warm_start` | [L605-615](../../../crawlbot/solvers/nmpc_solver.py#L605-L615) |
| `NMPCSolver._build_initial_guess` | [L621-635](../../../crawlbot/solvers/nmpc_solver.py#L621-L635) |
| `NMPCSolver._build_w0_from_trajectories` | [L637-649](../../../crawlbot/solvers/nmpc_solver.py#L637-L649) |
| `NMPCSolver._parse_solution` | [L651-664](../../../crawlbot/solvers/nmpc_solver.py#L651-L664) |
| `NMPCSolver._get_default_solver_options` | [L666-681](../../../crawlbot/solvers/nmpc_solver.py#L666-L681) |

---

## See also

- package overview: [`solvers.md`](solvers.md)
