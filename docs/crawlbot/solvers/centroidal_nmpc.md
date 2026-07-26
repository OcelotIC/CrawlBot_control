# `crawlbot.solvers.centroidal_nmpc`

**File**: [`crawlbot/solvers/centroidal_nmpc.py`](../../../crawlbot/solvers/centroidal_nmpc.py) — **702 lines** — canonical coverage **88 %**

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

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`CentroidalNMPCConfig`** *(dataclass)* |  |  | [L73](../../../crawlbot/solvers/centroidal_nmpc.py#L73) |
|   `robot_mass` | `90.0` | _field_ | [L81](../../../crawlbot/solvers/centroidal_nmpc.py#L81) |
|   `N` | `20` | _field_ | [L84](../../../crawlbot/solvers/centroidal_nmpc.py#L84) |
|   `dt` | `0.05` | _field_ | [L85](../../../crawlbot/solvers/centroidal_nmpc.py#L85) |
|   `Wr` | `100.0 * np.ones(3)` | _field_ | [L88](../../../crawlbot/solvers/centroidal_nmpc.py#L88) |
|   `Wv` | `10.0 * np.ones(3)` | _field_ | [L89](../../../crawlbot/solvers/centroidal_nmpc.py#L89) |
|   `Wu_f` | `0.01` | _field_ | [L90](../../../crawlbot/solvers/centroidal_nmpc.py#L90) |
|   `Wu_tau` | `0.001` | _field_ | [L91](../../../crawlbot/solvers/centroidal_nmpc.py#L91) |
|   `Qf_r` | `1000.0 * np.ones(3)` | _field_ | [L92](../../../crawlbot/solvers/centroidal_nmpc.py#L92) |
|   `Qf_v` | `100.0 * np.ones(3)` | _field_ | [L93](../../../crawlbot/solvers/centroidal_nmpc.py#L93) |
|   `f_max` | `3000.0` | _field_ | [L96](../../../crawlbot/solvers/centroidal_nmpc.py#L96) |
|   `tau_max` | `300.0` | _field_ | [L97](../../../crawlbot/solvers/centroidal_nmpc.py#L97) |
|   `L_max` | `np.inf` | _field_ | [L100](../../../crawlbot/solvers/centroidal_nmpc.py#L100) |
|   `tau_w_max` | `np.inf` | _field_ | [L101](../../../crawlbot/solvers/centroidal_nmpc.py#L101) |
|   `p_max` | `np.inf` | _field_ | [L102](../../../crawlbot/solvers/centroidal_nmpc.py#L102) |
|   `enforce_hw_conservation` | `False` | _field_ | [L110](../../../crawlbot/solvers/centroidal_nmpc.py#L110) |
|   `h_max_tight` | `np.full(3, 5.0)` | _field_ | [L111](../../../crawlbot/solvers/centroidal_nmpc.py#L111) |
|   `w_L` | `1.0` | _field_ | [L113](../../../crawlbot/solvers/centroidal_nmpc.py#L113) |
|   `Qf_L` | `10.0` | _field_ | [L114](../../../crawlbot/solvers/centroidal_nmpc.py#L114) |
|   `kappa_terminal` | `1.0` | _field_ | [L115](../../../crawlbot/solvers/centroidal_nmpc.py#L115) |
|   `solver_name` | `'ipopt'` | _field_ | [L118](../../../crawlbot/solvers/centroidal_nmpc.py#L118) |
|   `solver_opts` | `field(default_factory=dict)` | _field_ | [L119](../../../crawlbot/solvers/centroidal_nmpc.py#L119) |
| **`CentroidalNMPC`** |  |  | [L122](../../../crawlbot/solvers/centroidal_nmpc.py#L122) |
| `.build` | `(solver_opts=None)` | **yes** | [L149](../../../crawlbot/solvers/centroidal_nmpc.py#L149) |
| `.solve` | `(r_com, v_com, L_com, r_com_ref, v_com_ref, contact_conf...)` | **yes** | [L379](../../../crawlbot/solvers/centroidal_nmpc.py#L379) |
| `.get_last_trajectory` | `()` | **yes** | [L469](../../../crawlbot/solvers/centroidal_nmpc.py#L469) |
| `.get_shifted_fallback` | `()` | not exercised | [L481](../../../crawlbot/solvers/centroidal_nmpc.py#L481) |
| `.compute_c_simple` | `(r_com, v_com, L_com, hw_current=None)` | **yes** | [L514](../../../crawlbot/solvers/centroidal_nmpc.py#L514) |
| `.reset_warm_start` | `()` | **yes** | [L549](../../../crawlbot/solvers/centroidal_nmpc.py#L549) |
| `.get_full_trajectory` | `(r_com, v_com, L_com, r_com_ref, v_com_ref, contact_conf...)` | not exercised | [L565](../../../crawlbot/solvers/centroidal_nmpc.py#L565) |
| `.compute_feedforward_acceleration` | `(lambda_ref)` | **yes** | [L601](../../../crawlbot/solvers/centroidal_nmpc.py#L601) |
| `._assemble_params` | `(r_com, v_com, L_com, r_com_ref, v_com_ref, contact_conf...)` | **yes** | [L628](../../../crawlbot/solvers/centroidal_nmpc.py#L628) |
| `._apply_contact_bounds` | `(contact_config)` | **yes** | [L663](../../../crawlbot/solvers/centroidal_nmpc.py#L663) |

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

## Code map

| unit | source |
|---|---|
| `class CentroidalNMPCConfig` | [L73-119](../../../crawlbot/solvers/centroidal_nmpc.py#L73-L119) |
| `class CentroidalNMPC` | [L122-701](../../../crawlbot/solvers/centroidal_nmpc.py#L122-L701) |
| `CentroidalNMPC.build` | [L149-377](../../../crawlbot/solvers/centroidal_nmpc.py#L149-L377) |
| `CentroidalNMPC.solve` | [L379-467](../../../crawlbot/solvers/centroidal_nmpc.py#L379-L467) |
| `CentroidalNMPC.get_last_trajectory` | [L469-479](../../../crawlbot/solvers/centroidal_nmpc.py#L469-L479) |
| `CentroidalNMPC.get_shifted_fallback` | [L481-512](../../../crawlbot/solvers/centroidal_nmpc.py#L481-L512) |
| `CentroidalNMPC.compute_c_simple` | [L514-547](../../../crawlbot/solvers/centroidal_nmpc.py#L514-L547) |
| `CentroidalNMPC.reset_warm_start` | [L549-563](../../../crawlbot/solvers/centroidal_nmpc.py#L549-L563) |
| `CentroidalNMPC.get_full_trajectory` | [L565-599](../../../crawlbot/solvers/centroidal_nmpc.py#L565-L599) |
| `CentroidalNMPC.compute_feedforward_acceleration` | [L601-622](../../../crawlbot/solvers/centroidal_nmpc.py#L601-L622) |
| `CentroidalNMPC._assemble_params` | [L628-661](../../../crawlbot/solvers/centroidal_nmpc.py#L628-L661) |
| `CentroidalNMPC._apply_contact_bounds` | [L663-689](../../../crawlbot/solvers/centroidal_nmpc.py#L663-L689) |

---

## See also

- package overview: [`solvers.md`](solvers.md)
