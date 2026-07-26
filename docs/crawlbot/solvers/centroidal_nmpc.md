# `crawlbot.solvers.centroidal_nmpc`

**File**: `crawlbot/solvers/centroidal_nmpc.py` — **702 lines** — canonical coverage **88 %**

> Module docstring: *"CentroidalNMPC - Centroidal NMPC for momentum-feasible trajectory generation."*

**Stage 1 of the two-stage controller.** Generates, over a receding horizon, a
CoM and angular-momentum trajectory that is *feasible against the reaction-wheel
envelope* of the host structure.

The problem it solves is specific to orbital crawling: a robot moving along a
satellite transfers angular momentum to it. The AOCS wheels must absorb that
momentum, and their capacity is finite. A geometrically sensible CoM trajectory
can therefore be **physically inadmissible**. This module rejects it at
optimisation time instead of discovering it mid-swing.

---

## Public API

| symbol | signature | canonical? |
|---|---|---|
| **`CentroidalNMPCConfig`** *(dataclass)* |  |  |
|   `robot_mass` | `90.0` | _field_ |
|   `N` | `20` | _field_ |
|   `dt` | `0.05` | _field_ |
|   `Wr` | `100.0 * np.ones(3)` | _field_ |
|   `Wv` | `10.0 * np.ones(3)` | _field_ |
|   `Wu_f` | `0.01` | _field_ |
|   `Wu_tau` | `0.001` | _field_ |
|   `Qf_r` | `1000.0 * np.ones(3)` | _field_ |
|   `Qf_v` | `100.0 * np.ones(3)` | _field_ |
|   `f_max` | `3000.0` | _field_ |
|   `tau_max` | `300.0` | _field_ |
|   `L_max` | `np.inf` | _field_ |
|   `tau_w_max` | `np.inf` | _field_ |
|   `p_max` | `np.inf` | _field_ |
|   `enforce_hw_conservation` | `False` | _field_ |
|   `h_max_tight` | `np.full(3, 5.0)` | _field_ |
|   `w_L` | `1.0` | _field_ |
|   `Qf_L` | `10.0` | _field_ |
|   `kappa_terminal` | `1.0` | _field_ |
|   `solver_name` | `'ipopt'` | _field_ |
|   `solver_opts` | `field(default_factory=dict)` | _field_ |
| **`CentroidalNMPC`** |  |  |
| `.build` | `(solver_opts=None)` | **yes** |
| `.solve` | `(r_com, v_com, L_com, r_com_ref, v_com_ref, contact_conf...)` | **yes** |
| `.get_last_trajectory` | `()` | **yes** |
| `.get_shifted_fallback` | `()` | not exercised |
| `.compute_c_simple` | `(r_com, v_com, L_com, hw_current=None)` | **yes** |
| `.reset_warm_start` | `()` | **yes** |
| `.get_full_trajectory` | `(r_com, v_com, L_com, r_com_ref, v_com_ref, contact_conf...)` | not exercised |
| `.compute_feedforward_acceleration` | `(lambda_ref)` | **yes** |
| `._assemble_params` | `(r_com, v_com, L_com, r_com_ref, v_com_ref, contact_conf...)` | **yes** |
| `._apply_contact_bounds` | `(contact_config)` | **yes** |

---

---

## 1. Mathematical formulation

### State, control, parameters

```
x = [ r_com ; v_com ; L_com ]                      nx = 9
u = [ f1 ; tau1 ; f2 ; tau2 ]                      nu = 12
p = [ r_ref ; v_ref ; r_C1 ; r_C2 ; c ; L_ref ]    np = 18
```

Everything is in the structure frame **R_s**. `L_com` is the robot's centroidal
angular momentum *about its own CoM*.

Note `h_w` (wheel momentum) is **not a state**: the AOCS manages the wheels
independently. The coupling happens through a conservation law (1.3), not a
shared variable. That is what keeps the two subsystems decentralised.

### 1.1 Centroidal dynamics

In orbit, no gravity, up to two active contacts:

```
r_com_dot = v_com
v_com_dot = ( f1 + f2 ) / m
L_com_dot = (r_C1 - r_com) x f1 + tau1 + (r_C2 - r_com) x f2 + tau2
```

The moment arm is taken **from the robot CoM** — that is the definition of
centroidal momentum. Implemented as `centroidal_ode` (`:173-190`), integrated
with RK4 inside `NMPCSolver`.

### 1.2 Cost

Stage cost, summed over `k = 0..N-1`:

```
l(x,u) = ||r_com - r_ref||^2_Wr + ||v_com - v_ref||^2_Wv
       + w_L ||L_com - L_ref||^2 + ||u||^2_Wu
```

Terminal cost has the same structure with `Qf_r`, `Qf_v`, `Qf_L`.

`L_ref` is **not a stub**: it is supplied on every solve from
`TorsoPlanner.l_com_reference_at(t_mid)` and enters both costs. This is what
lets the NMPC track a planned *momentum* profile, not just a position.

### 1.3 The constraint that gives the module its purpose

Total angular momentum (robot + wheels) about the origin of R_s is conserved.
Measuring it once at the start of a step:

```
c = h_w_0 + L_com_0 + r_com_0 x m*v_com_0
```

lets you **reconstruct what the wheels will be carrying** at any node k of the
plan:

```
h_w(k) = c - L_com(k) - r_com(k) x m*v_com(k)
```

and bound it:

```
h_w(k) in [ -h_max' , +h_max' ]           (enforce_hw_conservation)
```

This is the "M3 Option B" formulation. It does not simulate the wheels: it
*deduces* their state from a conservation law, which makes it exact and cheap —
no extra states, no extra dynamics.

`c` is computed once per step by `compute_c_simple()` (`:514`).

### 1.4 The torque cap, and a moment-arm correction worth knowing

```
| H_s_dot,i | <= tau_w_max     where   H_s_dot = sum_j ( r_Cj x f_j + tau_j )
```

`H_s_dot` is the **exact** moment the wheels must counter to hold the structure
still (Newton's third law about the structure CoM = origin of R_s).

Note the moment arm: **`r_Cj`, from the origin** — not `r_Cj - r_com`. An earlier
version used `L_com_dot` as a proxy, i.e. the arm from the *robot* CoM, which
**under-counts the wheel torque whenever the standoff is non-zero**. With the
canonical standoff at -0.35 m the error was not small
(`Misc/reports/architecture/CAMPAIGN_5STEP_TRAVERSAL_2026-05.md` section 9).

The constraint is **linear in the controls**, which is what makes it a clean
decentralised contract between the robot and the AOCS.

### 1.5 Conic and box constraints

```
||f_j||^2 <= f_max^2        ||tau_j||^2 <= tau_max^2
||m*v_com||^2 <= p_max^2    | L_com,i | <= L_max
```

The linear-momentum bound limits the **orbital** disturbance transferred to the
structure — the other half of `H_{r/O} = L_com + r_com x m*v_com`.

Implementation detail (fix **F7**, CLEANUP-2): a constraint row is emitted **only
when its bound is finite**. Previously an infinite bound produced a constant
`-inf` row handed to IPOPT. Canonically `tau_w_max` and `p_max` are both finite,
so the emitted NLP is unchanged — but the code no longer depends on that.

---

## 2. What the code does, in order

| step | where | note |
|---|---|---|
| build the NLP | `build()` `:149` | **once**, at simulation `setup()` |
| assemble parameters | `_assemble_params()` | 18 values per solve |
| bounds for the phase | `_apply_contact_bounds()` | inactive contacts are **zeroed by bounds**, not removed |
| solve | `solve()` `:379` | warm-started from the previous solution |
| feedforward accel | `compute_feedforward_acceleration()` `:601` | consumed by stage 2 |

Keeping the problem size constant (12 controls even in single support) avoids
rebuilding the NLP at every phase change — only the bounds move. Rebuilding
would cost a full CasADi code-generation pass every step.

---

## 3. The `CentroidalNMPCConfig` defaults are not the canonical values

The table above shows `robot_mass=90.0`, `N=20`, `dt=0.05`. **None is used**:
`sim_loop.py:383-398` overrides every field from `SimConfig` (about 71 kg,
**N = 8**, **dt = 0.1 s** — a 0.8 s horizon at 10 Hz).

The costliest case is `enforce_hw_conservation`, whose default is `False` while
**the canonical run sets it `True`**. This is the error the chantier retracted
(F1): the entire `h_w` path — the constraint of 1.3, the reason this module
exists — had been declared dead from that default. Measured on the real run:
`enforce_hw=True`, `ng_path=17`, `ng_term=6`.

> A dataclass default is not the canonical value.

---

## 4. Two unexercised methods, opposite verdicts

| method | why unexercised | verdict |
|---|---|---|
| `get_shifted_fallback` | **no solve ever fails** on the canonical | **keep** — it is the fallback |
| `get_full_trajectory` | zero callers in `crawlbot/` | 5 tests + 1 script use it — pending |

`get_shifted_fallback` shifts the previous trajectory by one step to supply an
admissible command when IPOPT fails. It is dead *because the system is healthy* —
the most dangerous class of dead code to remove.

**Fix F2** (CLEANUP-3): the warm start is reused only when `info.success` is
true. Previously a failed solve could seed the next one and propagate its
divergence.

## See also

- package overview: [`solvers.md`](solvers.md)
