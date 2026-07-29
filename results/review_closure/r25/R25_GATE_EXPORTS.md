# R25_GATE_EXPORTS — final submission-gate exports (r25 → r26)

**Commit worked on:** `d15d2ba` (branch `review/r25-gate-exports`, cut from `main`).
`main` at the time of writing is `d15d2ba`; the two open-branch passes
(`cleanup-simloop`, PR #32) are **not** included, so every `file:line` below
resolves in the public `main`.

**Artifact base:** the four committed CSVs were verified byte-identical to
commit `bfd5509` before use (`md5sum` against `git show bfd5509:<path>`):
`results/j2_adjconv/{c25,u25}_fulldiag.csv`, `t4b_trace_900s.csv`,
`t4b_ltot_900s.csv`.

**Environment** (`gate/environment.lock`; gate check 4 = PASS, so the live
interpreter matches the lock exactly): python **3.11.15**, numpy **2.3.5**,
mujoco **3.10.0**, pinocchio **3.9.0**, casadi **3.7.2**, scipy **1.17.1**,
IPOPT available.

**Per-tick state:** `results/gate_run_scratch/sim_log.json`, regenerated on this
commit by `gate/replay_canonical.py` immediately before the exports. That run
passed the gate: artifact identity **PASS**, 2077 rows × 132 928 fields against
the committed baseline, log written 13:21:38 with the verdict at 13:21:39. The
sim_log is required because the 66-column CSV does **not** carry `L_com`.

**Scripts:** `scripts/r25_e1_e2_exports.py`,
`scripts/r25_e3_instrumented_replay.py`, `scripts/r25_e3_param_table.py`.
**Machine-readable outputs:** `results/review_closure/r25/*.{json,csv,md}`.

> ⚠ `results/review_closure/c2/c25_c2_fulldiag.csv` — named in the brief as the
> C2 instrumentation artifact — **does not exist in this repository**, and no C2
> branch is present on the remote (`origin/{main,cleanup,cleanup-nmpc,cleanup-simloop,claude/lucid-gates-rsigzt}`).
> Nothing below depends on it: the AOCS decomposition and solver channels it was
> to supply were obtained instead from the canonical sim_log and from a
> read-only instrumented replay (§E3).

---

## E1 — Neglected-term residuals of the Stage-1 model (gate M1)

```
rho_rot(t)    = ||omega_s x L_robot||  / ||sum_j (r_Cj x f_j + tau_j)||
rho_origin(t) = ||v_s      x m v_com|| / ||sum_j (r_Cj x f_j + tau_j)||
L_robot       = L_com + r_com x m v_com                (structure frame)
m             = 71.0560 kg
```

Denominator: `Hdot_s_realized_cont_{x,y,z}_Nm` from `c25_fulldiag.csv`, built by
`scripts/diag_full_diag_export.py::hdot()` as
`cross(r_a,f_a) + tau_a + cross(r_b,f_b) + tau_b` with levers in the structure
frame — the transmitted moment as specified.

### Results

| window | ratio | median | p95 | max | max at |
|---|---|---|---|---|---|
| traversal (to last SS, ticks 0–1875) | `rho_rot` | 1.459e-04 | 1.126e-03 | **8.446e-03** | t = 39.66 s, SS, step 3 |
| | `rho_origin` | 3.519e-06 | 4.450e-04 | **5.535e-03** | t = 9.82 s, SS, step 1 |
| full logged run (2077 ticks) | `rho_rot` | 1.463e-04 | 1.005e-03 | **8.446e-03** | t = 39.66 s, SS, step 3 |
| | `rho_origin` | 3.657e-06 | 3.934e-04 | **5.535e-03** | t = 9.82 s, SS, step 1 |

Valid ticks: 1866/1876 (traversal), 2067/2077 (full). The 10 excluded ticks have
a denominator of exactly zero — the exporter sets the transmitted moment to 0
outside steps 0–5, where the ratio is undefined rather than large. They are
excluded and counted, not clipped.

Per-tick values with their numerators and denominators:
`e1_ratios.csv`. Summary: `e1_summary.json`.

### `v_s` — not logged, derived, and the derivation validated

**`omega_s` is logged** (per tick, both CSV and sim_log; they agree to 4.97e-10).
**`v_s` is not logged anywhere** — neither the 66-column CSV nor the sim_log
carries the structure's linear velocity (`mj_data.qvel[0:3]`).

It was derived by central finite difference of `struct_pos`
(= `mj_data.qpos[0:3]`, logged per tick, dt = 0.1 s). Two checks on that
derivation, because `rho_origin` rests entirely on it:

1. **The operator is validated in-band.** The same central difference applied to
   `struct_quat`, converted to a body rate, reproduces the independently logged
   `omega_s`:

   | | median | p95 | max |
   |---|---|---|---|
   | absolute error | 3.34e-08 | 6.71e-06 | 2.82e-03 rad/s |
   | relative, where \|omega_s\| > 1e-3 rad/s (n = 266) | 2.87e-03 | 7.08e-03 | 8.37e-01 |

   The unconditioned relative error reaches O(1) only on ticks where
   `|omega_s| < 1e-6` rad/s — division by a vanishing denominator, not a failure
   of the operator. That is why the absolute figure is the primary one.

2. **Truncation bound on `v_s` itself**: central (2nd order) against a 5-point
   (4th order) stencil differ by at most **1.19e-02 m/s**, median relative to
   `|v_s|` 1.36e-04.

Cross-checks, CSV against sim_log (max abs diff): `r_com` 4.99e-07,
`v_com` 4.96e-08, `omega_s` 4.97e-10, `hw` vs `hw_physical` 4.99e-07.

Interpretation is not offered; the numbers are above.

---

## E2 — Reduced storage-prediction error (gate P1)

The constraint's storage model, verbatim from `centroidal_nmpc.py:37-38, 56,
107-108`:

```
c_simple      = h_w_0 + L_com_0 + r_com_0 x m v_com_0
hhat_w(i|k)   = c_simple - L_com,i - r_com,i x m v_com,i
```

which matches the brief's definition exactly.

### `e_h^(N)` is not computable from committed data — and no export was added

The NMPC's predicted trajectory is **not persisted**. The sim_log carries no
`x_plan` or predicted-state channel (the only plan-like channels are
`preplanner_T_steps` and `gait_plan`), so `hhat_w(k+N|k)` cannot be evaluated
along the solved trajectory. Per the brief, the one-step error is reported and
**no horizon export was added**.

### One-step error, 703 NMPC-active consecutive pairs

`e_h^(1)(k) = h_w(k+1) - [ c(k) - L_com(k+1) - r_com(k+1) x m v_com(k+1) ]`

| axis | median | p95 | max | max as % of h_max |
|---|---|---|---|---|
| x | 9.288e-03 | 3.132e-02 | 6.448e-02 | 1.290 % |
| y | 1.842e-02 | 7.938e-02 | 2.863e-01 | 5.727 % |
| z | 3.802e-02 | 1.204e-01 | 1.670e-01 | 3.341 % |
| **inf-norm** | 4.924e-02 | 1.204e-01 | **2.863e-01** | **5.727 %** |

Units N·m·s; `h_max = 5.0` N·m·s. Per-pair values: `e2_storage_error.csv`.
Summary: `e2_summary.json`.

Two conventions the reader needs:

- **Which `h_w`.** The CSV `hw_*_Nms` = sim_log `hw_physical` =
  `rwa_I_w * qvel[6:9]`, which is the quantity `sim_loop` feeds the NMPC as
  `c_simple`'s `h_w_0` (`hw_for_nmpc`). The other sim_log channel, `hw`, is a
  controller-internal value and was not used.
- **Which instant.** The recorder writes end-of-tick values (post QP sub-loop,
  post `mj_forward`), so all quantities here share one instant. This is
  therefore a consistent-instant conservation residual, not a replay of the
  solver's internal `c_simple`, which is read at tick *start*. The half-tick
  offset between the two conventions is not corrected for and is the dominant
  known bias in the table above.
- **Why 703 pairs.** `nmpc_ok` is set on 508 SS ticks and 201 DS_terminal ticks;
  it is 0 on all 1368 DS_interstep ticks, where the NMPC does not run.

---

## E3 — Table 3 reproducibility dump (gate P2)

Every value below is the **live** value captured from the running canonical
configuration (`e3_live_config.json`), not read off a dataclass. It was obtained
by attaching two read-only observers to a canonical replay
(`scripts/r25_e3_instrumented_replay.py`); nothing in `crawlbot/` was modified.
That replay reproduced the canonical exactly — **docks 6/6, every step delta
+0.0000, MATCH frozen 2.5** — which is the proof the observation was neutral.

### E3.1 NMPC (`crawlbot/solvers/centroidal_nmpc.py`)

The paper's `W_r`, `W_v`, `W_u` are named `Wr`, `Wv` and a **split** pair
`Wu_f` / `Wu_tau` in code — the control weight is not a single scalar. All
values below are the live canonical ones.

| quantity | symbol in code | canonical | default | source |
|---|---|---|---|---|
| stage weight, position | `Wr` | [100, 100, 100] | same | `:88` |
| stage weight, velocity | `Wv` | [10, 10, 10] | same | `:89` |
| control weight, force | `Wu_f` | 0.01 | same | `:90` |
| control weight, moment | `Wu_tau` | 0.001 | same | `:91` |
| terminal weight, position | `Qf_r` | [1000, 1000, 1000] | same | `:92` |
| terminal weight, velocity | `Qf_v` | [100, 100, 100] | same | `:93` |
| stage weight, `L_com` tracking | `w_L` | 1.0 | same | `:113` |
| terminal weight, `L_com` | `Qf_L` | 10.0 | same | `:114` |
| contact-force bound | `f_max` | **300.0 N** | 3000.0 | `:96` |
| contact-moment bound | `tau_max` | **8.0 N·m** | 300.0 | `:97` |
| angular-momentum bound | `L_max` | **10.0 N·m·s** | inf | `:100` |
| momentum-rate bound | `tau_w_max` | **2.5 N·m** | inf | `:101` |
| linear-momentum bound | `p_max` | **50.0** | inf | `:102` |
| horizon | `N` | **8** | 20 | `:84` |
| step | `dt` | **0.1 s** | 0.05 | `:85` |
| robot mass | `robot_mass` | **71.056 kg** | 90.0 | `:81` |
| storage box active | `enforce_hw_conservation` | **True** | False | `:110` |
| terminal margin | `kappa_terminal` | 1.0 | same | `:115` |
| IPOPT feasibility / optimality tolerance, iteration cap | `solver_opts` | **`{}` — empty.** No tolerance and no iteration cap is set by this project; CasADi/IPOPT defaults are in force | same | `:119` |

### E3.2 QP (`crawlbot/solvers/wholebody_qp.py`, `hierarchical_qp.py`)

| quantity | symbol | canonical | source |
|---|---|---|---|
| contact-force bound | `f_max` | 3000.0 N | `wholebody_qp.py:193` |
| contact-moment bound | `tau_contact_max` | 300.0 N·m | `:194` |
| joint-acceleration bound | `qdd_max` | 50.0 rad/s² | `:187` |
| joint-torque bound | `tau_max` | 20.0 N·m (14 joints) | `:184` |
| storage-slack penalty | `w_hw_slack` | 800.0 | `:159` |
| storage slack dimension | `_dim_slack_hw` | 6 (3 upper + 3 lower) | `:227` |
| QP box on storage | `hw_qp_tight` | **±3.0 N·m·s** | `config.py:77` |
| angular-momentum bound | `L_max` | 10.0 N·m·s | `:197` |
| momentum-rate bound | `tau_w_max` | 2.5 N·m | `:198` |
| QP step | `dt_qp` | 0.01 s | `:190` |
| solver | `solver` | `qpoases` (CasADi conic) | `:86`, `hierarchical_qp.py:99` |
| solver version | — | CasADi 3.7.2 (qpOASES bundled) | `gate/environment.lock` |
| print level | `printLevel` | `none` | `hierarchical_qp.py:498` |
| user override hook | `solver_opts` | `{}` (empty — nothing overridden) | `hierarchical_qp.py:513` |
| working-set cap | `nWSR` | **500** | `hierarchical_qp.py:500` |
| per-solve CPU budget | `CPUtime` | **0.005 s (5 ms)** | `hierarchical_qp.py:501` |
| fail behaviour | `error_on_fail` | `False` | `hierarchical_qp.py:495` |
| feasibility / optimality tolerance | — | **not set** — no `terminationTolerance` is passed; the qpOASES default is in force | `hierarchical_qp.py:493-503` |
| Tikhonov regularization | `regularization` | 1e-6 | `hierarchical_qp.py:98` |
| weight ratio | `weight_ratio` | 1.0 | `wholebody_qp.py:94` |

**From the canonical run** (8458 QP solves, `e3_qp_diagnostics.json`):

| quantity | median | p95 | max |
|---|---|---|---|
| equality residual `\|C_eq z − d_eq\|_inf` | 2.93e-16 | 1.22e-15 | **1.45e-11** |
| inequality violation `max(C_ineq z − d_ineq, 0)` | 0.0 | 2.03e-20 | **1.33e-15** |
| variable-bound violation | 0.0 | 0.0 | **0.0** |

| storage (h_w) slacks | value |
|---|---|
| solves with a non-zero slack | **389 of 8458** (4.60 %) |
| max upper slack `s_up` | 0.0 |
| max lower slack `s_lo` | **1.1118 N·m·s** |
| max slack norm | 1.1118 N·m·s |

Source: `WholeBodyQP.hw_slack_log`, populated on every solve
(`wholebody_qp.py:636-650`). The asymmetry is as recorded: the upper slack never
activates in this run; only the lower bound is relaxed, and at most by 1.11 N·m·s
(22 % of the 5 N·m·s box).

### E3.3 AOCS and plant

| quantity | canonical | source |
|---|---|---|
| number of wheels | 3 | `models/VISPA_crawling_rwa3.xml:117,125,133` |
| wheel axes | `rw_x` (1,0,0), `rw_y` (0,1,0), `rw_z` (0,0,1) | same |
| `B_w` (wheel → body torque map) | **identity, 3×3** — an orthogonal triad aligned with the body axes; no distribution matrix is formed in code | implied by the axes above |
| wheel inertia / momentum-per-speed | `rwa_I_w` = 0.01 kg·m²; `h_w = rwa_I_w · qvel[6:9]` | `config.py:84`; MJCF `armature="0.01"` |
| **wheel inertia vs MJCF armature** | **0.01 vs 0.01 — they agree.** (This is the divergence flagged in the brief; it is closed in the frozen configuration.) | `:117,125,133` |
| storage box, physical | `hw_min` / `hw_max` = ∓5.0 / ±5.0 N·m·s | `config.py:70-71` |
| storage box, as the NMPC sees it | `h_max_tight = ±5.0` N·m·s — **equal to the physical box**, so "tightened" is a no-op in this configuration | `config.py:194` |
| storage box, as the QP sees it | `hw_qp_tight = ±3.0` N·m·s — **the QP's box is tighter than physical, at 60 %** | `config.py` (live) |
| box implementation | NMPC: hard constraint on `c_simple − L_com(k) − r_com(k)×m v_com(k)` at every knot (`centroidal_nmpc.py:56, 105-110`). QP: soft, via 6 slack variables with penalty `w_hw_slack = 800` (`wholebody_qp.py:149-159`) | |
| torque saturation | clipped, three times: AOCS command `np.clip(±aocs_tau_w_max = 2.5)` (`sim_loop.py:3085`, `aocs/force_estimator.py:287,376`), and the plant motor `ctrlrange="-2.5 2.5"` (`VISPA_crawling_rwa3.xml:323-325`) | |

### E3.4 Divergences — dataclass default ≠ canonical value

**38 fields differ.** Full machine-readable list: `e3_param_table.json`,
rendered in `e3_param_table.md`. Reading any of these off the dataclass would put
a wrong number in the paper. The ones bearing on Table 3:

| config | field | canonical | default | source |
|---|---|---|---|---|
| `CentroidalNMPCConfig` | `robot_mass` | **71.056** | 90.0 | `centroidal_nmpc.py:81` |
| | `N` | 8 | 20 | `:84` |
| | `dt` | 0.1 | 0.05 | `:85` |
| | `f_max` | **300.0** | 3000.0 | `:96` |
| | `tau_max` | **8.0** | 300.0 | `:97` |
| | `L_max` | 10.0 | inf | `:100` |
| | `tau_w_max` | 2.5 | inf | `:101` |
| | `enforce_hw_conservation` | True | False | `:110` |
| `WholeBodyQPConfig` | `alpha_ee` | 1000.0 | 500.0 | `wholebody_qp.py:97` |
| | `alpha_posture` | 20.0 | 100.0 | `:98` |
| | `alpha_wrench` | 1.0 | 10.0 | `:99` |
| | `alpha_torque` | 5.0 | 1.0 | `:100` |
| | `alpha_reg` | 1.0 | 0.01 | `:101` |
| | `alpha_lambda_int` | **1.0** | 0.0 | `:102` |
| | `ss_alpha_mom` | 400.0 | 500.0 | `:133` |
| | `alpha_torso_pose` | 2000.0 | 1000.0 | `:134` |
| | `qp_envelope_exact` | **True** | False | `:147` |
| | `Kp_torso` / `Kd_torso` | 3.0 / 2.5 (all 6) | [8,8,8,5,5,5] / [6,6,6,4,4,4] | `:166-167` |
| | `Kp_ee` / `Kd_ee` | 10.0 / 12.0 | 80.0 / 15.0 | `:170-171` |
| | `Kp_posture` / `Kd_posture` | 1.0 / 1.5 | 25.0 / 10.0 | `:176-177` |
| | `tau_max` | 20.0 | 50.0 | `:184` |
| | `dt_qp` | 0.01 | 0.008 | `:190` |
| `CoarsePrePlannerConfig` | `robot_mass` | 71.056 | 71.0 | `coarse_preplanner.py:68` |

### ⚠ E3.5 One divergence is not a value but a *shape*, and it changes the gain

`sim_loop.py:1132` constructs the QP with

```python
Kp_com=np.diag([kpc]*3), Kd_com=np.diag([kdc]*3)      # 3x3 MATRICES
```

whereas the dataclass declares `Kp_com`/`Kd_com` as 3-**vectors**
(`wholebody_qp.py:162-163`, default `[100,100,100]` / `[20,20,20]`). The task row
then does (`wholebody_qp.py:902-906`):

```python
Kp_com_mat = np.diag(cfg.Kp_com)                       # np.diag of a 2-D array
a_com_des  = a_com_ff + Kp_com_mat @ (r_com_ref - r_com) + Kd_com_mat @ (v_com_ref - v_com)
```

`np.diag` applied to a **matrix extracts its diagonal**, so `Kp_com_mat` is the
1-D vector `[3,3,3]` and the product is a **dot product — a scalar**, broadcast
to all three axes. Evaluated on a sample error `e = [0.01, −0.02, 0.005]`:

```
as coded    np.diag(Kp) @ e  = -0.015          (one scalar, applied to x, y and z)
as nominal  diag(3) @ e      = [0.03, -0.06, 0.015]
```

So the canonical CoM PD term is `kp · Σ_i e_i` rather than the per-axis
`kp · e_i` that a reader of "Kp_com = 3" would assume. This affects the SS
two-task momentum row and the DS centroidal task, both of which are built by
`_com_task_rows`.

Stated as a fact about the frozen configuration, not a proposed change: the run
is the run, and the published numbers come from it. Flagged because Table 3 would
otherwise report a per-axis gain that the implementation does not apply.

---

## E4 — Two definitional questions

### E4.1 The internal-stress projector `P_int = I − G⁺G`

**It is assembled, and it is active in the canonical run.**

- **Where:** `crawlbot/solvers/wholebody_qp.py:574-594`.
- **`G` is 6×12**, mapping the stacked contact wrench
  `λ = (f_A, τ_A, f_B, τ_B)` to the net wrench on the structure body about the
  origin. Stacking as coded (`:578-586`):

  ```
  G[0:3, 0:3] = I        f_A -> net force
  G[0:3, 6:9] = I        f_B -> net force
  G[3:6, 0:3] = skew(r_CA)   r_CA x f_A -> torque about origin
  G[3:6, 3:6] = I            tau_A      -> torque
  G[3:6, 6:9] = skew(r_CB)   r_CB x f_B -> torque about origin
  G[3:6, 9:12] = I           tau_B      -> torque
  ```

  The comment records the approximation: torque is taken about the origin
  because the structure CoM ≈ world origin at small structure rotation.

- **The term as coded** (`:587-594`):

  ```python
  G_pinv = np.linalg.pinv(G, rcond=1e-8)
  P_int  = np.eye(12) - G_pinv @ G            # projector onto the 6-D internal-stress null space
  A_lint[:, lambda_block] = P_int
  qp.add_task(A_lint, b_lint=0, cfg.alpha_lambda_int, priority=4)
  ```

  i.e. a least-squares task minimising `‖P_int λ‖²` at weight `alpha_lambda_int`.

- **Weight:** canonical **`alpha_lambda_int = 1.0`**, set at
  `scripts/diag_cooperative_arms.py:342`. The dataclass default is **0.0**
  (`wholebody_qp.py:102`) — so reading the default would wrongly conclude the
  term is inert. It is not.

- **It is DS-only.** The block is gated on `alpha_lambda_int > 0` **and**
  `contact_config.nc == 2` with both contacts active (`:574-577`). In single
  support `rank(G) = 6 = dim(λ)`, so there is no internal-stress subspace to
  project onto. The `--baseline_ds_rework` flag sets the weight back to 0.0
  (`diag_cooperative_arms.py:356`), but that is not the canonical run.

The discussion should therefore be kept, with `G` defined as above and the
DS-only scope stated.

### E4.2 The coarse pre-planner

- **Purpose** (`crawlbot/planning/coarse_preplanner.py:1-8`): a momentum-feasible
  CoM trajectory. It solves a centroidal NLP over the whole step horizon so that
  the *position-dependent* momentum box holds at every collocation point — the
  geometric TorsoPlanner path carries no such guarantee.
- **Formulation** (`:9-33`): decision variables `r_com, v_com, L_com` at M+1
  collocation points plus `f_stance, tau_stance` at M intervals (one active
  contact in SS, so 3-vectors); centroidal ODE dynamics; constraints = momentum
  box at every knot, `|L̇_com|_inf ≤ tau_w_max`, force/moment boxes, boundary
  conditions `state(0) = x0`, `r_com(M) = r_goal`, `v_com(M) ≈ 0`, `L_com(M) ≈ 0`,
  and a terminal margin tightened by `kappa < 1`.
- **Inputs:** initial state, goal CoM, contact point, the conservation constant
  `c`, and the step duration — all bound as CasADi parameters.
- **Outputs:** `CoarsePlanResult` — a dense interpolant for `r_com(t)`, `v_com(t)`
  plus the raw collocation trajectory. Consumed by `sim_loop` **as the NMPC
  reference in place of the TorsoPlanner geometric path** (`:38-41`).
- **When it runs:** **once per step, before the step starts** (`:3`). The call
  order in `run()` is `_setup_torso_for_step()` → `_run_preplanner()`
  (`sim_loop.py:1470`) → then `_step()` (`sim_loop.py:1865`), which is the online
  10 Hz NMPC / 100 Hz QP loop. The NLP is **built once** via CasADi Opti;
  subsequent steps only rebind parameters and re-run IPOPT.
- **Does its cost belong in an online-performance statement?** **No.** It is
  per-step setup executed before the online loop begins, not per-tick work
  inside it. Six solves occur across the whole 6-step traversal, producing the
  six step durations `preplanner_T_steps = [2.775, 7.652, 3.452, 7.924, 3.433,
  7.772] s`. Quoting its solve time alongside the 10 Hz NMPC and 100 Hz QP
  timings would misattribute an offline, once-per-step cost to the online loop.

---

## Outputs

| file | contents |
|---|---|
| `e1_ratios.csv` | per-tick `rho_rot`, `rho_origin`, numerators, denominator, valid flag |
| `e1_summary.json` | E1 statistics, both windows, plus the `v_s` provenance and validation |
| `e2_storage_error.csv` | per-pair one-step storage error, per axis and inf-norm |
| `e2_summary.json` | E2 statistics vs `h_max`, and why `e_h^(N)` is absent |
| `e3_live_config.json` | every field of the four live config objects |
| `e3_param_table.json` / `.md` | default vs canonical, per field, with `file:line` |
| `e3_qp_diagnostics.json` | residuals, violations and storage slacks over 8458 solves |

Scripts: `scripts/r25_e1_e2_exports.py`,
`scripts/r25_e3_instrumented_replay.py`, `scripts/r25_e3_param_table.py`.
