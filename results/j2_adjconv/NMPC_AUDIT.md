# NMPC audit — what is actually implemented

**Branch** `claude/com-gain-semantics-audit-j0u6yr`
**Method** every structural claim below is produced by
`scripts/audit_nmpc_structure.py`, which *builds* the NLP from the canonical
`SimConfig` and reads the result back — not by reading source. Activity claims
come from the coverage of the canonical replay (`gate/_run/cov/cov.json`) and
from the exported fulldiag CSV. Machine output: `results/j2_adjconv/nmpc_structure.json`.

---

## 0. Summary of findings, ranked

| # | finding | severity |
|---|---|---|
| **F1** | **The reference is a constant setpoint over the whole horizon, not a trajectory.** One parameter vector serves every stage; the pre-planner's momentum-feasible *trajectory* is sampled at a single instant per solve and held. | **structural** |
| ~~**F2**~~ | **RETRACTED — the box and its terminal set are ON in the canonical** (`ng_path=17`, `ng_term=6`). The finding was built from a bare `SimConfig()`; `dca` uses `_make_m7_config()`, which passes `enforce_hw_conservation=True`, `h_max_tight=5.0`, `kappa_terminal=1.0` as explicit kwargs. See `NMPC_F2_RWA_BOX.md` §0. | **withdrawn** |
| **F3** | **Three code paths assume `nmpc_dt == dt_nmpc`** (plan interpolation, shifted fallback, warm-start shift). Nothing enforces it. Setting them unequal silently dilates the reference the QP tracks. | **latent bug** |
| **F4** | The wheel-torque cap `\|Ḣ_s\|_∞ ≤ τ_w,max` is the **only constraint that binds** — 58.1 % of SS ticks at the cap. The SOC force/torque bounds and the linear-momentum bound are non-binding throughout. (All NMPC constraints are HARD — `lbg=-inf, ubg=0`; only the QP's momentum box is soft, via `w_hw_slack` slack variables.) | informational |
| **F5** | The infeasibility fallback (`get_shifted_fallback`) is **never executed** by the canonical run — 11/11 body statements uncovered. It is the designed recovery path and it is unverified by both gates. | coverage gap |
| **F6** | `Solved_To_Acceptable_Level` counts as success. At `acceptable_tol=1e-4` vs `tol=1e-6` a solve can be accepted at 100× looser tolerance after 5 iterations. Frequency is configuration-dependent (2/634 at N=15, **27/711** at N=20/dt=0.05). | informational |
| **F7** | `L_com` is bounded by a **state box** `\|L_i\| ≤ 10` Nms, not by the wheel envelope. ⚠ Partly retracted with F2: the envelope DOES bound accumulated `h_w` inside the horizon, via the live box. | informational |

---

## 1. Role and placement

Stage 1 of the two-stage controller. It plans **robot centroidal motion only**;
the AOCS manages the wheels independently. It is invoked **once per control
period** from `sim_loop._step`, and only in **SS** and the **terminal DS
settle** — never in `DS_interstep`. On the frozen canonical that is 508 SS +
201 DS_terminal = **709 solves**, which is exactly the length of
`nmpc_step_log.json`.

⚠ The exported `nmpc_ok` column conflates "not called" with "failed": the 1368
`DS_interstep` ticks export `nmpc_ok=0`. Reading the column unfiltered gives a
false 34.1 % success rate; the true rate is 709/709. Always filter by `phase`.

---

## 2. Measured NLP structure

Built from the canonical `SimConfig` (values below are for `N=20, dt=0.05`; the
committed config at time of writing is `N=15, dt=0.1`):

```
state / control    nx = 9, nu = 12, np = 18
transcription      multiple shooting
integrator         RK4, zero-order hold on u
decision variables n_w = (N+1)·nx + N·nu
constraint rows    n_g = nx + N·nx + N·11 + 0
```

- **State** `x = [r_com(3), v_com(3), L_com(3)]`. Wheel momentum `h_w` is *not*
  a state — it was removed when the AOCS became independent.
- **Control** `u = [f₁(3), τ₁(3), f₂(3), τ₂(3)]` — two contact wrenches.
  Inactive contacts are pinned to zero by bounds, per contact phase.
- **Parameters** `p = [r_ref(3), v_ref(3), r_C1(3), r_C2(3), c_simple(3), L_ref(3)]`.

### Dynamics (`centroidal_ode`, `centroidal_nmpc.py:170-189`)

$$\dot r = v,\qquad \dot v = \frac{f_1+f_2}{m},\qquad
\dot L = \sum_j \big[(r_{Cj}-r_{com})\times f_j + \tau_j\big]$$

No gravity (orbital). Integrated RK4 at step `dt`. **What is not modelled:**
joint-level dynamics, the arm kinematics, the structure's attitude, and the
wheels. The NMPC sees a free-floating point mass with an angular-momentum state
and two contact wrenches — everything else is the QP's problem.

---

## 3. F1 — the reference is a setpoint, not a trajectory

`nmpc_solver.build()` constructs **one** parameter symbol `p_param = P[nx:]`
and passes **the same one** to every stage:

```python
for k in range(self.N):
    J += L_eval(x=Xk, u=Uk, p=p_param)['cost']     # same p_param, all k
    g_val = g_eval(x=Xk, u=Uk, p=p_param)['g']     # same p_param, all k
```

There are no per-stage parameters. Consequently `r_ref`, `v_ref`, `L_ref` and
both contact positions are **constant across the horizon**. The NMPC is a
*regulator to a fixed point*, not a trajectory tracker.

`sim_loop` compensates by sampling the reference **at the horizon end** rather
than at the current time (`sim_loop.py:2128-2132`, comment: *"passing the
current-time reference causes systematic lag"*). That is a sound workaround for
a setpoint regulator, and it is the mechanism behind the `nmpc_N` coupling
documented in `NMPC_HORIZON_N15.md` §1.

**Consequence.** The coarse pre-planner produces a momentum-feasible CoM
*trajectory* over the full `T_step`; the NMPC consumes exactly **one point of
it per solve**. The intermediate shape of that trajectory never reaches the
NMPC. Adding per-stage parameters would let the horizon track the pre-planner's
actual curve and would decouple `nmpc_N` from the reference lead — the two
issues have the same root.

---

## 4. F2 — declared constraints that are not emitted
> ⚠ **§4 below is superseded for the F2 rows.** It was measured from a bare
> `SimConfig()`, which is not what the canonical runs. The box and terminal
> set are ON (`ng_path=17`, `ng_term=6`) and `c_simple`, `h_max_tight`,
> `kappa_terminal` are all live. Full retraction:
> `results/j2_adjconv/NMPC_F2_RWA_BOX.md` §0.


| block | rows | canonical | governing value |
|---|---|---|---|
| SOC `‖f_j‖² ≤ f_max²`, `‖τ_j‖² ≤ τ_max²` | 4 | **ON** | `f_max=300` N, `τ_max=8` N·m |
| wheel-torque cap `\|Ḣ_s,i\| ≤ τ_w,max` | 6 | **ON** | `τ_w,max = 2.5` N·m |
| linear momentum `‖m·v‖² ≤ p_max²` | 1 | **ON** | `p_max = 50` kg·m/s |
| **RWA conservation box** `h_w(k) ∈ [−h', h']` | 6 | **ON** | `h_max_tight = 5.0` (per axis) |
| **terminal** `\|h_w(N)\| ≤ κ·h'` | 6 | **ON** | `κ = 1.0` |

`enforce_hw_conservation` defaults to `False` at `config.py:193` and **nothing
overrides it** — not `sim_loop`, not `dca.main`, not the gate's `C_KWARGS`. So
the M3/Option-B conservation law described at length in the module docstring
(lines 56-57), in `CentroidalNMPCConfig` (lines 105-115) and in the spec is
**not part of the canonical controller**.

Inert as a direct consequence: `c_simple` (parameter slots `p[12:15]`, computed
by `compute_c_simple()` on every solve), `h_max_tight`, `kappa_terminal`. The
code's own docstring concedes this — *"It is only read by the NLP when
enforce_hw_conservation is True"* (`centroidal_nmpc.py:649-650`).

This is not necessarily wrong: the wheel envelope is still protected by the
rate cap (F4) and by the QP's own momentum box. But the repository documents a
constraint that does not run, and the paper's §V momentum narrative should be
checked against that.

---

## 5. F3 — the `nmpc_dt == dt_nmpc` invariant, assumed in three places

`nmpc_dt` (prediction step) and `dt_nmpc` (control period) are **independent
fields**. Three consumers assume they are equal:

| site | what it does | breaks how |
|---|---|---|
| `sim_loop.py:2342` | `alpha = qs / n_qp_per_nmpc`, walks plan knot 0 → knot 1 across one **control period** | knots are `nmpc_dt` apart ⇒ reference dilated by `dt_nmpc / nmpc_dt` |
| `centroidal_nmpc.get_shifted_fallback` | shifts the stored plan by **one knot**, used as the fallback for one control period | fallback advances the wrong amount of time |
| `nmpc_solver.shift_warm_start` | shifts the warm start by **one knot** between solves separated by one control period | warm start misaligned |

The `sim_loop.py:2338` comment explicitly claims the interpolation *"matches
the time parameterisation `tq = t + qs·dt_qp`"* — true only under the
invariant. Nothing asserts it. The structural probe now reports it:

```
control period     dt_nmpc = 0.1 s  (10 Hz)
  knot spacing == control period? *** NO ***
```

**Recommended fix** (not applied): either assert equality at construction, or
make the interpolation time-based — compute `alpha` from
`(qs·dt_qp) / nmpc_dt` and index the correct knot pair — which would make
`nmpc_dt` a genuinely free parameter.

---

## 6. F4 — which constraint actually binds

Measured on the frozen canonical fulldiag, SS ticks only:

| quantity | value |
|---|---|
| planned `‖Ḣ_s‖_∞` max | **2.5000** N·m (exactly the cap) |
| SS ticks within 1 % of the cap | **295/508 = 58.1 %** |
| realized `‖Ḣ_s‖_∞` max | 2.5000 N·m; at the cap on 27/508 |

The wheel-torque cap is the binding constraint and effectively defines the
plan. The SOC bounds (300 N, 8 N·m) and the linear-momentum bound (50 kg·m/s)
are far from active — they are guards, not shaping constraints.

---

## 7. Cost

$$\ell = \|r-r^*\|^2_{W_r} + \|v-v^*\|^2_{W_v} + w_L\|L-L^*\|^2 + \|u\|^2_{W_u}$$
$$\ell_f = \|r-r^*\|^2_{Q_{f,r}} + \|v-v^*\|^2_{Q_{f,v}} + Q_{f,L}\|L-L^*\|^2$$

| weight | value | note |
|---|---|---|
| `Wr` | 100 | CoM position |
| `Wv` | 10 | CoM velocity |
| `w_L` | 1.0 | `L_com` tracking — **live** reference from `TorsoPlanner.l_com_reference_at(t_mid)`, not a stub |
| `Wu_f` / `Wu_tau` | 0.01 / 0.001 | wrench regularization |
| `Qf_r` / `Qf_v` / `Qf_L` | 1000 / 100 / 10 | terminal |

Terminal position weight is 10× the stage weight, which — combined with F1's
constant setpoint — makes the horizon end the dominant target.

---

## 8. Solver, warm start, fallback

IPOPT via CasADi, MUMPS linear solver. `CentroidalNMPCConfig.solver_opts` is
`{}` and is never overridden, so `_get_default_solver_options()` **is** the
canonical setting:

```
ipopt.max_iter = 200      ipopt.tol = 1e-06
ipopt.acceptable_tol = 1e-04    ipopt.acceptable_iter = 5
ipopt.warm_start_init_point = yes    ipopt.linear_solver = mumps
ipopt.print_level = 0     print_time = 0
```

- **Warm start** — primal *and dual* (`lam_g0`, `lam_x0`), stored **only on
  success** so a failed iterate cannot entrench itself. Shifted by one knot
  after each successful solve. Reset at phase transitions (three call sites).
- **Fallback on infeasibility** — `get_shifted_fallback()` advances the last
  successful plan by one knot rather than jumping to the geometric reference.
  **F5: its body never executes in the canonical run** (coverage: only the
  `def` line). `sim_loop.py:2186` states this is by design — *"the canonical
  run never has nmpc_ok False"*. The designed recovery path is therefore
  verified by neither gate.
- **F6: success semantics.** `info.success` comes from CasADi's
  `stats['success']`, which is `True` for both `Solve_Succeeded` and
  `Solved_To_Acceptable_Level`. The latter accepts a 100× looser tolerance.
  Counts are configuration-dependent — see §9.

---

## 9. Results: N=20, dt=0.05

Two readings of "dt = 0.05" were run, because F3 makes them different
experiments. Both use `N=20` (lookahead 1.0 s). Driver:
`scripts/audit_nmpc_horizon_sweep.py`; artifacts under
`results/j2_adjconv/nmpc_sweep/<tag>/`; figure `nmpc_variants.png`.

### 9.1 The two readings

| tag | `nmpc_N` | `nmpc_dt` | `dt_nmpc` | meaning |
|---|---|---|---|---|
| `N20_dt05_p10` | 20 | 0.05 | **0.10** | literal: prediction step only. Violates F3 ⇒ the QP's CoM reference is **dilated 2×** |
| `N20_dt05_p05` | 20 | 0.05 | **0.05** | consistent: control period follows ⇒ F3 satisfied, NMPC at **20 Hz** |

### 9.2 Outcome

Baseline column is the frozen canonical (N=8, dt=0.1). N=15 is the current
committed config, both with the Phase-3 gain fix.

| metric | frozen N=8 | N=15 dt=0.1 | **N=20 dt=0.05 @10 Hz** | **N=20 dt=0.05 @20 Hz** |
|---|---|---|---|---|
| dock 1..6 [mm] | 4.02 / 4.89 / 4.99 / 4.97 / 4.95 / 4.62 | 4.40 / 4.53 / 4.93 / 4.48 / 2.12 / 4.40 | 4.89 / 4.64 / 4.85 / 4.65 / 4.95 / 4.47 | 4.92 / 4.66 / **5.00** / 4.80 / 4.29 / 4.60 |
| under 5 mm | 6/6 | 6/6 | **6/6** | **5/6 — GATE BREACH** |
| worst margin [mm] | 0.01 | 0.07 | 0.05 | **0.00** |
| θ_s peak [deg] | 0.540 | 0.554 | **0.494 (best)** | **0.621 (worst)** |
| h_w peak norm [Nms] | 4.240 | 4.172 | 4.233 | **4.068 (best)** |
| e_com peak [m] | 0.154 | 0.190 | 0.167 | 0.170 |
| qp_fail | 0 | 0 | 0 | 0 |
| ticks | 2077 | 1981 | 2053 | 2642 |
| NMPC solves | 709 | 634 | 711 | **1274** |
| solve median / max [ms] | 22.0 / 61.9 | 34.3 / 117.9 | 34.4 / **53.9** | 35.2 / 58.1 |
| period budget [ms] | 100 | 100 | 100 | **50** |
| **solves over budget** | 0/709 | 1/634 | **0/711** | **8/1274** |
| `Solved_To_Acceptable_Level` | — | 2 | 27 | 55 |

Figure: `results/j2_adjconv/nmpc_sweep/nmpc_variants.png` — CoM error, θ_s,
‖h_w‖ vs the 5 N·m·s envelope, realized ‖Ḣ_s‖_∞ vs the 2.5 N·m cap, solve time
vs each variant's own period, and the solver summary.

### 9.3 Reading the result

**The consistent configuration is the one that fails.** Running the NMPC at its
own declared design point (N=20, dt=0.05) *with the timing invariant satisfied*
docks step 3 at exactly 5.00 mm — zero margin, which `dock_check` scores as a
gate breach — and produces the worst attitude excursion of any configuration
measured (θ_s 0.621°, vs 0.540 frozen). It also puts 8 solves over the halved
50 ms budget.

**The dilated configuration is the one that looks good.** N=20/dt=0.05 at 10 Hz
gives the *best* θ_s of all four (0.494°), keeps 6/6 docks, and is the only
variant with real-time headroom (max 53.9 ms against 100 ms). But its CoM
reference advances at half rate — it is good *because of* an F3 bug, not in
spite of it. It must not be adopted as a configuration.

Two further observations:

- **Smaller `dt` buys real-time headroom, longer `N` costs it.** N=20/dt=0.05
  (429 decision variables) solves faster in the worst case than N=15/dt=0.1
  (~330 variables): max 53.9 vs 117.9 ms. The finer step makes each RK4 defect
  easier to satisfy, so IPOPT terminates sooner — the `Solved_To_Acceptable_Level`
  count rising 2 → 27 → 55 is the same effect. **Solve cost is not monotone in
  problem size here**, which is worth knowing before any further tuning.
- **θ_s and the docks disagree across the sweep.** N=15 has the best dock margin
  and a worse θ_s; N=20@10 Hz has the best θ_s and a mid dock margin; N=20@20 Hz
  is worst on both dock margin and θ_s while being best on h_w. There is no
  single dominant setting among those measured.

### 9.4 What is left in the tree

`config.py` is restored to the committed **N=15, dt=0.1** — the sweep driver
restores it in a `finally`, and this was verified after the run. Neither N=20
variant is adopted, because one breaches the dock gate and the other is only
viable through the F3 defect. Adopting N=20/dt=0.05 properly requires fixing F3
first, then re-measuring.

---

## 10. Recommendations

1. **Assert the F3 invariant** or make the interpolation time-based. One line
   either way; without it `nmpc_dt` is a trap.
2. **Decide F2**: either enable `enforce_hw_conservation` and re-measure, or
   mark the box and `c_simple` as retired in the docstring and the spec so the
   code stops advertising a constraint it does not apply.
3. **F1 is the substantive one.** Per-stage parameters would let the NMPC track
   the pre-planner's actual trajectory and would decouple the horizon from the
   reference lead. It is the single change with the most headroom.
4. **Cover F5.** A test that forces an infeasible solve and asserts the shifted
   fallback is used would close the largest verification gap in this module.
