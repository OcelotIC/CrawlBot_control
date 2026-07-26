# `crawlbot.planning.coarse_preplanner`

**File**: `crawlbot/planning/coarse_preplanner.py` — **540 lines** — canonical coverage **81 %**

> Module docstring: *"CoarsePrePlanner — momentum-feasible CoM trajectory optimization (M6)."*

**Decides how long a step lasts.** A centroidal NLP solved once per step,
producing `T_step` and a momentum-feasible CoM trajectory over that horizon.

The central idea of the architecture is here: step duration is not a tuning
parameter, it is an *output* of the momentum envelope.

---

## Public API

| symbol | signature | canonical? |
|---|---|---|
| **`CoarsePrePlannerConfig`** *(dataclass)* |  |  |
|   `M` | `15` | _field_ |
|   `robot_mass` | `71.0` | _field_ |
|   `h_max` | `np.full(3, 5.0)` | _field_ |
|   `kappa_terminal` | `0.7` | _field_ |
|   `f_max` | `25.0` | _field_ |
|   `tau_max` | `8.0` | _field_ |
|   `tau_w_max` | `2.5` | _field_ |
|   `w_L` | `1.0` | _field_ |
|   `w_u` | `0.01` | _field_ |
|   `w_v_terminal` | `100.0` | _field_ |
|   `w_L_terminal` | `100.0` | _field_ |
|   `eps_v_terminal` | `0.005` | _field_ |
|   `eps_L_terminal` | `0.05` | _field_ |
|   `T_step_default` | `6.0` | _field_ |
|   `a_cruise_max` | `0.0` | _field_ |
|   `cruise_ramp_frac` | `0.2` | _field_ |
|   `ipopt_print_level` | `0` | _field_ |
|   `ipopt_max_iter` | `300` | _field_ |
|   `ipopt_tol` | `1e-06` | _field_ |
| **`CoarsePlanResult`** *(dataclass)* |  |  |
|   `T_step` |  | _field_ |
|   `t_grid` |  | _field_ |
|   `r_com` |  | _field_ |
|   `v_com` |  | _field_ |
|   `L_com` |  | _field_ |
|   `f_stance` |  | _field_ |
|   `tau_stance` |  | _field_ |
|   `success` |  | _field_ |
|   `solve_time_ms` |  | _field_ |
|   `cost` |  | _field_ |
|   `status` |  | _field_ |
|   `iter_count` | `0` | _field_ |
| `.r_com_at` | `(t)` | **yes** |
| `.v_com_at` | `(t)` | **yes** |
| `.L_com_at` | `(t)` | not exercised |
| `._interp` | `(t, traj)` | **yes** |
| `.hw_at_knots` | `(c_const)` | not exercised |
| `.from_heuristic` | `(cls, r_com_0, r_com_goal, h_max, robot_mass, M=15, leve...)` | not exercised |
| **`CoarsePrePlanner`** |  |  |
| `.build` | `()` | **yes** |
| `.solve` | `(r_com_0, v_com_0, L_com_0, r_com_goal, r_C_stance, c_co...)` | **yes** |

---

---

## 1. Why the step duration cannot be chosen

Move the CoM by `d` in time `T` and you generate transverse momentum of order
`m * d / T`, which shows up in the wheels as `r_com x m*v_com`. Halve `T` and you
double it. So for a given reach and a given `h_max`, there is a **minimum
feasible step duration** — and it depends on the geometry of that particular
step, not on a global setting.

That is what this module computes. `ContactScheduler` creates SS phases with
`duration = 0.0`; the real value only exists after this solve, and is installed
by `GaitPlan.set_step_duration()`.

## 2. The optimisation problem

M = 15 collocation intervals. Decision variables: `r_com, v_com, L_com` at the
M+1 nodes, `f_stance, tau_stance` on the M intervals (a single active contact in
single support, so 3+3 per interval).

### Dynamics

RK4 on the one-contact centroidal ODE:

```
r_com_dot = v_com
v_com_dot = f / m
L_com_dot = (r_C - r_com) x f + tau
```

### Constraints

**1. Momentum box at every node** — the same conservation reconstruction as
stage 1:

```
c - L_com(k) - r_com(k) x m*v_com(k)  in  [-h_max', +h_max']
```

**2. Rate bound**, evaluated at the start of each interval:

```
| H_s_dot |_inf <= tau_w_max = 2.5 Nm      with  H_s_dot = r_C x f + tau
```

⚠ Note this uses the arm **from the structure origin**, matching the NMPC's cap
and *not* the centroidal `L_com_dot`. The comment at `:339-344` is explicit: the
centroidal quantity under-counts wheel torque at non-zero standoff. The `L` state
in the ODE stays centroidal; only the constraint uses the origin arm.

**3. Wrench box**: `|f|_inf <= 25 N`, `|tau|_inf <= 8 Nm`.

**4. Boundary conditions**: `x(0) = x0`, `r_com(M) = r_goal`, plus soft boxes on
terminal velocity and momentum.

**5. Terminal margin**: at `k = M` the momentum box is tightened by
`kappa = 0.7`. The step must not merely end inside the envelope but with 30 %
headroom — otherwise the *next* step starts with no room to manoeuvre.

### Cost

```
J = sum_k [ w_L ||L_com(k)||^2 + w_u ( ||f(k)||^2 + ||tau(k)||^2 ) ]
    + w_v_terminal ||v_com(M)||^2 + w_L_terminal ||L_com(M)||^2
```

Penalising `||L_com||` throughout, not just at the end, keeps the whole
trajectory momentum-quiet rather than allowing a large excursion that is
cancelled late.

## 3. Build once, re-solve per step

The NLP is built once via CasADi `Opti` with the initial state, goal, contact
point, conservation constant and timestep bound as **parameters**. Each `solve()`
only sets values and re-runs IPOPT. Initial guess: linear interpolation on
`r_com`, zeros elsewhere.

## 4. On failure the step is skipped — deliberately

No silent heuristic fallback. `sim_loop` logs the failure, holds position and
skips the step. A pre-planner that quietly degrades to a guess would defeat the
entire feasibility argument.

## 5. ⚠ `from_heuristic`: 83 lines documented as a test fixture, with no test

Four comments (`sim_loop.py:1369`, `:1599`, `:1633`, `config.py:207`) state that
unit tests use it to avoid the IPOPT dependency. **No test calls it.** Searched
across *every reachable commit*, the only `.from_heuristic(` in the repository is
that comment.

Worse, the comments are load-bearing: `sim_loop.py:1632-1640` **re-implements its
envelope formula inline** and cites the fixture to justify the duplication.
Removal is pending (CLEANUP-19 audit).

## 6. Silent canonical values (rule 5)

Five fields are never overridden by `sim_loop`, so their defaults *are* the
canonical values — and **none is in CLAUDE.md**:

| field | value | what it decides |
|---|---|---|
| `eps_v_terminal` | 5e-3 m/s | **hard box** on terminal CoM velocity |
| `eps_L_terminal` | 5e-2 Nms | **hard box** on terminal momentum |
| `w_v_terminal`, `w_L_terminal` | 1e2 | soft penalties on the same residual |
| `ipopt_tol` | 1e-6 | convergence tolerance |

The first two decide **where a step is allowed to end**. That is physics living
in a dataclass default.

`T_step_default` (6.0 s) is never used at all: `sim_loop` always passes `T_step`.

## 7. The unexercised remainder is fallback, not sediment

The 12 dead lines in `solve()` are **entirely** the error ladder — IPOPT
`RuntimeError`, the value-extraction fallback, the stats fallback — plus two API
defaults. Signature of a healthy system. Keep.

The cruise-acceleration block (`a_cruise_max`, M7 v21) is disabled (`0.0`) and
reachable only by hand-editing `SimConfig`. Its removal awaits a ruling: it is a
*documented* CLAUDE.md parameter.

## See also

- package overview: [`planning.md`](planning.md)
