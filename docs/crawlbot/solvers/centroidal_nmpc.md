# `crawlbot.solvers.centroidal_nmpc`

**File**: [`crawlbot/solvers/centroidal_nmpc.py`](../../../crawlbot/solvers/centroidal_nmpc.py) — **756 lines** — canonical coverage **86 %**

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
|   `per_stage_refs` | `False` | _field_ | [L126](../../../crawlbot/solvers/centroidal_nmpc.py#L126) |
|   `solver_name` | `'ipopt'` | _field_ | [L129](../../../crawlbot/solvers/centroidal_nmpc.py#L129) |
|   `solver_opts` | `field(default_factory=dict)` | _field_ | [L130](../../../crawlbot/solvers/centroidal_nmpc.py#L130) |
| **`CentroidalNMPC`** |  |  | [L133](../../../crawlbot/solvers/centroidal_nmpc.py#L133) |
| `.build` | `(solver_opts=None)` | **yes** | [L160](../../../crawlbot/solvers/centroidal_nmpc.py#L160) |
| `.solve` | `(r_com, v_com, L_com, r_com_ref, v_com_ref, contact_conf...)` | **yes** | [L394](../../../crawlbot/solvers/centroidal_nmpc.py#L394) |
| `.get_last_trajectory` | `()` | **yes** | [L484](../../../crawlbot/solvers/centroidal_nmpc.py#L484) |
| `.get_shifted_fallback` | `()` | not exercised | [L496](../../../crawlbot/solvers/centroidal_nmpc.py#L496) |
| `.compute_c_simple` | `(r_com, v_com, L_com, hw_current=None)` | **yes** | [L529](../../../crawlbot/solvers/centroidal_nmpc.py#L529) |
| `.reset_warm_start` | `()` | **yes** | [L564](../../../crawlbot/solvers/centroidal_nmpc.py#L564) |
| `.get_full_trajectory` | `(r_com, v_com, L_com, r_com_ref, v_com_ref, contact_conf...)` | not exercised | [L580](../../../crawlbot/solvers/centroidal_nmpc.py#L580) |
| `.compute_feedforward_acceleration` | `(lambda_ref)` | **yes** | [L616](../../../crawlbot/solvers/centroidal_nmpc.py#L616) |
| `._assemble_params` | `(r_com, v_com, L_com, r_com_ref, v_com_ref, contact_conf...)` | **yes** | [L643](../../../crawlbot/solvers/centroidal_nmpc.py#L643) |
| `._apply_contact_bounds` | `(contact_config)` | **yes** | [L717](../../../crawlbot/solvers/centroidal_nmpc.py#L717) |

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

**`p` is per-knot** when `per_stage_refs=True` (the canonical setting): the NLP
carries N+1 parameter blocks and stage `k` reads `p_k`, so `r_ref`, `v_ref` and
`L_ref` vary along the horizon and the problem is a **trajectory tracker**.

With `per_stage_refs=False` there is one shared block, the reference is
necessarily a constant **setpoint**, and `sim_loop` has to sample it in the
future (horizon end for CoM, midpoint for `L_com`) to avoid systematic lag —
which is what used to tie `nmpc_N` to the reference lead. See
`nmpc_solver.md` §1.1 for the mechanism and the equivalence proof, and
`results/j2_adjconv/NMPC_F1_PER_STAGE_REFS.md` for the measured effect
(θ_s 0.554 → 0.455°, all six docks held).

`_assemble_params` accepts a `(3,)` setpoint or a `(K, 3)` per-knot reference
and raises when `K` matches neither 1 nor N+1 — silently dropping knots would be
the dangerous failure, so it is made impossible rather than documented.

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
`sim_loop.py:416-433` overrides every field from `SimConfig` (about 71 kg,
**N = 20**, **dt = 0.1 s** — a 2.0 s horizon at 10 Hz).

> A dataclass default is not the canonical value.

⚠ **Correction (NMPC_AUDIT, 2026-07-30).** This section previously stated that
"the canonical run sets `enforce_hw_conservation` `True`" with `ng_path=17`,
`ng_term=6`. **That is false for the current tree, and it was asserted as a
measurement.** Built from the canonical `SimConfig` and read back by
`scripts/audit_nmpc_structure.py`:

```
[OFF] RWA conservation box h_w(k)          0 rows
[OFF] terminal |h_w(N)| <= kappa*h_max     0 rows
ng_path = 11   (4 SOC + 6 wheel-torque + 1 linear momentum)
ng_term = 0
```

`SimConfig.enforce_hw_conservation` is `False` (`config.py:193`) and **nothing
overrides it** — not `sim_loop`, not `dca.main`, not the gate's `C_KWARGS`. The
only `True` sites in the tree are `scripts/run_m7_single_step.py:44` and
`tests/test_nmpc_conservation.py:64`, neither of which is the canonical run.

So the `h_w` box and its terminal constraint are **not part of the canonical
controller**, and `c_simple` — computed by `compute_c_simple()` on every solve —
is read by nothing. Whether to enable it is an open decision (`NMPC_AUDIT` F2);
it is recorded here so the document stops contradicting the code.

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
| `class CentroidalNMPCConfig` | [L73-130](../../../crawlbot/solvers/centroidal_nmpc.py#L73-L130) |
| `class CentroidalNMPC` | [L133-755](../../../crawlbot/solvers/centroidal_nmpc.py#L133-L755) |
| `CentroidalNMPC.build` | [L160-392](../../../crawlbot/solvers/centroidal_nmpc.py#L160-L392) |
| `CentroidalNMPC.solve` | [L394-482](../../../crawlbot/solvers/centroidal_nmpc.py#L394-L482) |
| `CentroidalNMPC.get_last_trajectory` | [L484-494](../../../crawlbot/solvers/centroidal_nmpc.py#L484-L494) |
| `CentroidalNMPC.get_shifted_fallback` | [L496-527](../../../crawlbot/solvers/centroidal_nmpc.py#L496-L527) |
| `CentroidalNMPC.compute_c_simple` | [L529-562](../../../crawlbot/solvers/centroidal_nmpc.py#L529-L562) |
| `CentroidalNMPC.reset_warm_start` | [L564-578](../../../crawlbot/solvers/centroidal_nmpc.py#L564-L578) |
| `CentroidalNMPC.get_full_trajectory` | [L580-614](../../../crawlbot/solvers/centroidal_nmpc.py#L580-L614) |
| `CentroidalNMPC.compute_feedforward_acceleration` | [L616-637](../../../crawlbot/solvers/centroidal_nmpc.py#L616-L637) |
| `CentroidalNMPC._assemble_params` | [L643-715](../../../crawlbot/solvers/centroidal_nmpc.py#L643-L715) |
| `CentroidalNMPC._apply_contact_bounds` | [L717-743](../../../crawlbot/solvers/centroidal_nmpc.py#L717-L743) |

---

## See also

- package overview: [`solvers.md`](solvers.md)
