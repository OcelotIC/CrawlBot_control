# C1 — Paper ↔ code exactness

**Review-Closure Bloc 2, Phase C1.** Read-only. Nothing was fixed, retuned or
re-run; the two probe scripts below only read committed artifacts and the
controller model.

---

## Header (mandatory)

| item | value |
|---|---|
| commit worked on | **`54a0a4a`** (C0 report + nav-doc refresh), branch `claude/review-closure-bloc-2-uwu1x7`; controller code identical to `eecbf94` |
| date | 2026-07-26 |
| python / mujoco / pinocchio | 3.11.15 / 3.10.0 / 3.9.0 |
| casadi + ipopt / numpy / scipy | 3.7.2 + IPOPT / 2.3.5 / 1.17.1 |
| env vs `gate/environment.lock` | exact match (C0) |

**Artifacts cited**

- `results/j2_adjconv/c25_fulldiag.csv`, `u25_fulldiag.csv` (+ `_meta.json`)
- `results/gate_run_scratch/sim_log.json` — the C0 replay, byte-identical to the
  committed canonical; used for `dock_gate_trace`, `dock_events`,
  `preplanner_T_steps`, `q_torso`, `q_torso_ref`, `q_ee` (channels the CSV does
  not carry)
- `results/review_closure/c1/c1_log_analysis.py` → `c1_log_analysis.json`
- `results/review_closure/c1/c1_jacobian_probe.py` → `c1_jacobian_probe.json`

**Anchor protocol.** Every `file:line` below was **re-located against the current
package layout at `eecbf94`** and read before being quoted. No anchor is
inherited from `PORT_AUDIT.md` / `PORT_SYNTHESIS.md` or from the earlier audits.

**Summary of verdicts**

| item | verdict |
|---|---|
| C1.1 QP stack + momentum task rows | **CONFIRMED** (2 omissions in the paper's weight list) |
| C1.2 desired-acceleration composition | **CONFIRMED** on FF, gains, error map · **DEFECT** on the orientation-error *frame* |
| C1.3 task Jacobians | **CONFIRMED** (1 notational caveat) |
| C1.4 AOCS law | **CONFIRMED** on structure, no-integral, FF, anti-windup ≡ 0 · **DEFECT** on the **signs** of the three feedback terms |
| C1.5 docking gate | **CONFIRMED** — thresholds located; *not* where the brief expected |
| C1.6 traversal-time comparison | **DEFECT — the paper's attribution is wrong.** Durations confirmed; the mechanism is not the constrained plan |
| C1.7 structure inertia | **REPORTED** — value confirmed; a latent axis-order defect found in the fallback path |

---

## C1.1 — QP stack and the momentum task — **CONFIRMED**

### Weighted, not strictly prioritised

`WholeBodyQP.solve` builds `HierarchicalQP(n_vars, method=cfg.method,
solver=cfg.solver, weight_ratio=cfg.weight_ratio)` at
`crawlbot/solvers/wholebody_qp.py:389-392`, with
`method = 'weighted'` (`:85`) and `weight_ratio = 1.0` (`:94`). Nothing in
`sim_loop.py`, `diag_cooperative_arms.py` or `run_m7_single_step.py` overrides
either (grep for `method=` / `weight_ratio` over all three returns only an
unrelated print). `HierarchicalQP`'s own defaults are `strict` / 1000
(`hierarchical_qp.py:95-96`) and are **not** what the canonical uses.
**There is no null-space projection anywhere in `wholebody_qp.py`.** At
`weight_ratio = 1` each task enters at its face-value α and the `priority=`
arguments are inert labels. ✔

### The SS cost, as actually assembled

Gate `_two_task = cfg.ss_two_task_mode and not settle_mode`
(`wholebody_qp.py:422`); on the canonical `ss_two_task_mode=True`
(`gate/replay_canonical.py:41`).

| # | task | α | source of α | `add_task` |
|---|---|---:|---|---|
| 1 | T-MOM linear (CoM-Jacobian rows) | **400** | `config.py:290` | `:425` |
| 2 | torso-pose 6-D on `J_torso` | **2000** | `config.py:303` | `:442-443` |
| 3 | swing-EE 6-D on `J_ee` | **1000** | `config.py:282` | `:460` |
| 4 | posture | **20** | `config.py:283` | `:464-466` |
| 5 | contact-wrench tracking | **1.0** | `config.py:284` | `:553` |
| 6 | joint-torque minimisation | **5** | `sim_loop.py:951` literal | `:601` |
| 7 | acceleration regularisation | **1** | `sim_loop.py:951` literal | `:609` |
| 8 | h_w slack penalty | **800** | `wholebody_qp.py:159` | `:622` |

The four weights the paper states (torso 2000, EE 1000, momentum 400,
torque-min 5) are **all confirmed**, and `alpha_torso_pose = 2e3` carries the
annotation `CANONICAL-2p5 / Add-5 freeze (was 5e3)` exactly as the brief says.

**Two entries the paper's list omits, flagged rather than treated as defects:**

- **`w_hw_slack = 800`** is a cost term and is *numerically larger than the
  momentum task's 400*. A weight table that stops at "posture + wrench
  regularisation" understates the stack. It is **inert on this run** — the
  slacks are pinned to zero at the optimum whenever the momentum box is
  satisfiable, and C1.4 shows the box is never violated (`|h_w|_∞ ≤ 4.1019 < 5`
  on all 2077 ticks) — but it belongs in the table with that caveat.
- **`alpha_reg = 1`** (the cost floor) is what `alpha_torque = 5` must stay
  ≳5× above (Rule 14). Quoting torque-min = 5 without the floor it is measured
  against loses the constraint's meaning.

One wording point: task 5 is a **tracking** task, `b = lambda_ref` from the
NMPC (`wholebody_qp.py:545`), not a regulariser toward zero. "Wrench
regularisation" is loose; "wrench tracking at regulariser-tier weight" is what
the code does. `alpha_lambda_int = 1.0` (set at `diag_cooperative_arms.py:342`)
is a genuine regulariser but is gated to `contact_config.nc == 2`
(`wholebody_qp.py:574-577`) and is therefore **inert in SS**.

### The momentum task uses the linear CMM rows A_{G,v} — **CONFIRMED numerically**

`_com_task_rows` (`wholebody_qp.py:888-912`) assembles `A_com` from `J_com`
(rows `J_com[:, :6]` → `q̈_t`, `J_com[:, 6:]` → `q̈`), with
`b_com = a_com_ff + K_p(r_ref − r) + K_d(v_ref − v) − J̇_com q̇` and
`Kp_com = Kd_com = 3·I₃` (`sim_loop.py:957`, from `ss_Kp_com`/`ss_Kd_com = 3.0`).
`J_com` is Pinocchio's `data.Jcom` (`robot_interface.py:301`).

Measured on the canonical controller model (`c1_jacobian_probe.py`):

```
max | J_com  −  A_G[0:3,:] / m |  =  2.220e-16      (scale max|J_com| = 0.997)
```

i.e. `A_G[0:3,:] = m · J_com` to machine precision. **The task rows are the
linear centroidal-momentum matrix rows A_{G,v}, scaled by 1/m.** ✔
The angular rows `A_G[3:6,:]` are assembled nowhere in the QP.

### `L_com` is not tracked by any Stage-2 task — **CONFIRMED**

Every occurrence of `L_com` in `wholebody_qp.py` is at lines 286, 399, 733,
735, 778-786 — the parameter, its pass-through, and the **inequality** box
`|L_com + dt·M_λ·λ| ≤ L_max` (`L_max = 10 Nms`, piped at `sim_loop.py:969`).
It never reaches `add_task`. So `L_com` is **constrained, never tracked**, and
the QP has no angular-momentum cost. ✔ (Robot angular momentum is tracked only
upstream, by the NMPC's `w_L_nmpc = 1.0` term against `L_com_ref`.)

---

## C1.2 — Desired-acceleration composition — **CONFIRMED**, with a **frame DEFECT**

### What the code assembles

Torso (`wholebody_qp.py:427-443`) and swing-EE (`:445-460`), verbatim:

```python
e6      = concat([p_ref − p,  pin.log3(R.T @ R_ref)])
a_des   = a_ff + Kp @ e6 + Kd @ (v_ref − v_act)
add_task(A, a_des − Jdot_dq, alpha, priority=2)
```

| paper claim | verdict | evidence |
|---|---|---|
| tasks are at **acceleration** level | CONFIRMED | rows are `J q̈ = a_des − J̇q̇`, `:439-443`, `:457-460` |
| **feedforward `ẍ_ref` present** for torso | CONFIRMED | `a_ft = a_torso_ff` , `:435`, entering `a_t_des` at `:436` |
| **feedforward present** for swing-EE | CONFIRMED | `a_fe = a_ee_ff`, `:455`, entering `a_e_des` at `:456` |
| pose-error **PD** | CONFIRMED | `Kp @ e6 + Kd @ (v_ref − v_act)`, `:436` / `:456` |
| orientation error via the **SO(3) logarithm** | CONFIRMED | `pin.log3(...)`, `:433` / `:451` |
| **torso gains `K_p = 3`, `K_d = 2.5`** | **CONFIRMED** | `sim_loop.py:964-965` `Kp_torso = [kpt]*6`, `Kd_torso = [kdt]*6`, fed from `cfg.ss_Kp_torso/ss_Kd_torso` (`sim_loop.py:473`), set to **3.0 / 2.5** by the canonical kwargs (`gate/replay_canonical.py:43`) |

The gains are **uniform across all six DOF** — no separate angular scaling
(`sim_loop.py:958-963` documents the removal of the legacy 0.6× angular
factor). The paper's single (K_p, K_d) pair is therefore exactly right.

**Trap, recorded because it will bite the next reader:** `config.py:351-352`
declares `ss_Kp_torso = 6.0`, `ss_Kd_torso = 5.0`. Those are the *dataclass
defaults* and are 2× the canonical. Anyone verifying the paper against
`config.py` alone will conclude the paper is wrong by a factor of two. It is
not — the kwarg chain wins.

For completeness, the other effective gains: EE `Kp_ee = 10`, `Kd_ee = 12`,
`Kp_ee_ang = 6`, `Kd_ee_ang = 4.5` (`sim_loop.py:966-967`); posture
`Kp_posture = 1.0`, `Kd_posture = 1.5` (`sim_loop.py:968` — again *not* the
dataclass 25/10); T-MOM `Kp_com = Kd_com = 3`.

### DEFECT — the orientation error is in the **body** frame, the Jacobian is **world-aligned**

`J_torso` and `J_tool_{a,b}` are obtained with `pin.LOCAL_WORLD_ALIGNED`
(`robot_interface.py:317-357`), so the angular rows of `J q̈` are an angular
acceleration expressed in **world-aligned axes**. The driving term
`pin.log3(R.T @ R_ref)` is the rotation vector expressed in the **body**
frame. The frame-consistent form for a world-aligned Jacobian is
`log3(R_ref @ R.T) = R · log3(R.T @ R_ref)`.

The two have identical magnitude and differ in **direction** by the rotation
`R` itself. Measured (`c1_jacobian_probe.py`, `C1_2_frame`): at an 8° frame
attitude the relative direction gap is a constant **9.87 %** of the error
magnitude, independent of the tracking error — it scales with **how far the
frame's attitude is from identity**, not with how badly it is tracking.

**How much does this matter on the canonical run?** Measured from the replay
log, not assumed:

| quantity | value |
|---|---|
| torso attitude `‖log3(R_torso)‖` over 2077 ticks | **3.943° – 6.129°**, mean 5.305° |
| ⇒ relative direction gap `2·sin(θ_R/2)` | **6.88 % – 10.69 %** |
| logged `e_torso_ori` | mean 0.664°, p95 1.652°, max 1.683° |
| ⇒ absolute angular-error **direction** discrepancy | mean 0.060°, **max 0.130°** |
| swing-EE attitude `‖log3(R_ee)‖` | 0.000° – **1.293°** ⇒ gap ≤ 2.3 %, negligible |

So the torso task's angular PD term is driven by a vector whose direction is
wrong by ~7–11 %, bounded at 0.13° of commanded error. **This is a real
formal inconsistency, not a rounding artifact** — the torso is *not* near
identity (it sits at a fixed 5.157° yaw, see below) — but it is small in
absolute terms and the run docks 6/6 regardless.

**Paper action:** the composition as written must state its frame. If §VI-D.5/6
writes the world-frame form `log(R_ref R^T)`, it is describing something the
code does not do. The one-line fix the brief anticipates is to write
`e_R = log(R^T R_ref)` **and** say the task is expressed in the body frame —
or to state the discrepancy is second-order and bounded at 0.13°.

### Incidental, and material for the paper's torso narrative

`q_torso_ref` in the replay log is **constant to 6.9e-18 across all 2077
ticks** — a single value, `(w,x,y,z) = (0.998988, 0, 0, −0.044988)`, i.e. a
pure 5.157° yaw. The torso **orientation** reference is a constant regulation
target for the entire six-step traversal, so `ω_ref = α_ref = 0` and the
angular half of the feedforward is identically zero. This is the intended
consequence of `ik_fixed_rotation = True` (`config.py:376`, "the robot crawls
forward, it doesn't pirouette"). The measured torso attitude does move
(2068 distinct values, max deviation 0.0117 in quaternion components).
A paper sentence describing the torso reference as a "quintic + SLERP
trajectory" is formally true of the planner but, on the canonical, the SLERP is
between identical endpoints.

---

## C1.3 — Task Jacobians — **CONFIRMED**

Measured on the controller model at a non-trivial configuration (random arm
joints, 8° base attitude, non-zero base translation) —
`c1_jacobian_probe.py`, block `C1_3`. Column order is
`[q̇_t(6) | arm A(7) | arm B(7)]` (`robot_interface._detect_arm_slices`,
`:29-67`; `n_joints = 14`, `nv = 20`).

| Jacobian | base block | arm-A block | arm-B block | exactly zero? |
|---|---:|---:|---:|---|
| `J_torso` | 0.9967 | **0.0000** | **0.0000** | **yes** — `all(J_torso[:, 6:] == 0.0)` |
| `J_tool_a` | 1.4669 | 1.1155 | **0.0000** | **yes** — `all(J_tool_a[:, 13:] == 0.0)` |
| `J_tool_b` | 1.2179 | **0.0000** | 1.0783 | **yes** — `all(J_tool_b[:, 6:13] == 0.0)` |

- **`J_torso = [J̃_torso  0_{6×7}  0_{6×7}]`** — confirmed exactly, both arm
  blocks identically zero. The QP splits it back into the two decision blocks
  at `wholebody_qp.py:440-441`, so the torso task drives only `q̈_t`. ✔
- **`J_ee = [J̃^j_t  0 (stance)  J̃^j_q (swing)]`** — confirmed: each tool
  Jacobian is identically zero on the *other* arm's columns
  (`wholebody_qp.py:458-459`). ✔

**Caveat, notational not numerical.** The zero blocks arise **structurally from
the kinematic tree** (a tool frame simply does not depend on the other arm's
joints), not from an explicit masking step in the controller. And the literal
matrix `[J̃^j_t  0_{6×7}  J̃^j_q]` as the paper prints it is correct only when
the **swing arm is B**; when the swing arm is A the layout is
`[J̃_t  J̃_q  0_{6×7}]`. The brief's own phrasing — "the zero block on the
stance chain's columns" — is the accurate statement and is what the paper
should print, since the canonical alternates swing arms every step
(b, a, b, a, b, a per `c25_fulldiag_meta.json`).

---

## C1.4 — AOCS law — **CONFIRMED** in structure, **DEFECT** in the signs

Active mode is `legacy_pid_numerical` (`gate/replay_canonical.py:39`), which
dispatches to `compute_aocs_command_legacy_pid_numerical`
(`crawlbot/aocs/force_estimator.py:514-595`), called at `sim_loop.py:2854-2866`
for SS and in-step DS, and at `sim_loop.py:920-931`
(`_interstep_aocs_command`) for the inter-step settle.

**The law as coded** (`force_estimator.py:575-595`):

```python
hw_error       = np.clip(hw_current, hw_min, hw_max) - hw_current     # :575
omega_dot_est  = (omega_s - omega_s_prev) / dt                        # :576
pid_term       = K_theta*theta_s + K_omega*omega_s + K_d*omega_dot_est  # :577
ff_term        = tau_struct_ff  if given  else  -L_dot_est - orbital  # :579-592
tau_w          = ff_term + K_hw*hw_error + pid_term                   # :594
return np.clip(tau_w, -tau_w_max, tau_w_max)                          # :595
```

| paper claim | verdict | evidence |
|---|---|---|
| anti-windup term `K_hw(sat_{±5}(h_w) − h_w)` | **CONFIRMED** verbatim | `:575` + `:594`; `hw_min/hw_max = ∓5` (`config.py:70-71`) |
| **no integral term** | **CONFIRMED** | no integrator state anywhere in the function or the estimator it belongs to. Despite the name, `legacy_pid_*` is **P on attitude + D on rate + D on acceleration** — the "I" is a misnomer for the added attitude-P term (`config.py:90-97`) |
| FF = full measured momentum rate about O | **CONFIRMED** | see below |
| **SS**: `−L̇_com − r_com × m·a_com` | **CONFIRMED** | `:585-588`, taken when `tau_struct_ff is None`; `a_com` is the one-step FD `(v_com − v_com_prev)/dt` |
| **DS**: contact-wrench couple | **CONFIRMED** | `tau_struct_ff = −Σ_i (r_Ci × f_i + τ_i)` assembled from `λ_qp` at `sim_loop.py:2747-2757` (in-step DS) and `:908-916` (inter-step). Gated on `aocs_use_wrench_ff_in_ds` **and** `phase == 'DS'` **and** mode ∈ {`legacy_pid_numerical`, `legacy_pid_model`} (`:2743-2746`) — all three hold on the canonical (`diag_cooperative_arms.py:318`) |
| gains | CONFIRMED | `K_θ = 1.0`, `K_ω = 50.0`, `K_d = 25.0`, `K_hw = 2.0` (`config.py:102-104, 82`); output clipped to `aocs_tau_w_max = 2.5` (`config.py:83`) |

### DEFECT — the three feedback terms carry **`+`**, the paper writes **`−`**

The brief states the paper's law as

> τ_w = τ_ff **−** K_θ e_θ **−** K_ω ω_s **−** K_d ω̇_s + K_hw(sat(h_w) − h_w)

The code is `pid_term = **+** K_θ·θ_s **+** K_ω·ω_s **+** K_d·ω̇_s`
(`force_estimator.py:577`), added with `+` at `:594`. The positive sign is
**deliberate and reasoned in the source**, `force_estimator.py:545-548`:

> "Sign on K_θ is positive (same derivation as K_ω, K_d): Newton-Euler about
> structure CoM with τ_w on wheels giving −τ_w reaction on the structure. For
> θ_s > 0 to decrease, need negative angular acceleration ⇒ τ_w > −Ḣ_s ⇒ K_θ
> contribution adds positive."

With the code's own definitions — `θ_s` the structure attitude *error* from its
initial reference and `ω_s` the *measured* structure rate — all three signs are
`+`. The paper's minus signs are consistent only under an opposite sign
convention for `e_θ` and `ω_s`, which the paper would then have to state.
**As printed, the paper's equation does not describe the implemented law.**
The fix is one of: flip the three signs, or define `e_θ ≜ −θ_s` and `ω ≜ −ω_s`
explicitly. This is a sign convention, not a stability error — the run is
stable and θ_s peaks at 0.540°.

### Second divergence — `θ_s` is the Lee–McClamroch error, **not** `log3`

`force_estimator.py:552-553` documents `theta_s` as "Computed in sim_loop as
`log3(R_init.T @ R_now)`". It is not. Both call sites compute

```python
R_err   = R_init.T @ R_now
theta_s = 0.5 * vee(R_err − R_errᵀ)          # sim_loop.py:2846-2850, :899-903
```

which is `sin(θ)·axis`, not `θ·axis`. The **call-site** comment
(`sim_loop.py:2826-2837`) is explicit and correct about this and gives the
three reasons (frame consistency with ω_s, boundedness `|e_R| ≤ 1`, no
singularity at π); it is the *function docstring* that is stale. At the
canonical's ≤ 0.54° drift the two agree to better than 1e-5 relative, so this
is a documentation defect with no numerical consequence — but the paper must
print `½(R_ref^T R − R^T R_ref)^∨`, not `log`, if it prints the error map.

### Anti-windup term is identically zero over the whole run — **CONFIRMED**

Asserted from the committed CSVs, all three axes, every tick
(`c1_log_analysis.py`):

| run | `|h_w|_∞` peak | ticks with `|h_w| > 5 Nms` |
|---|---:|---:|
| **C (managed)** | **4.1019 Nms** | **0 / 2077** |
| U (unmanaged) | 4.5502 Nms | 0 / 1905 |

`np.clip(hw, −5, +5) − hw ≡ 0` whenever `|h_w| ≤ 5`, which holds on every tick
of both runs. **The `K_hw` branch contributes exactly nothing to the reported
results.** The brief's stated peak, 4.102, is reproduced to four decimals.
Note this also makes the h_w slack task of C1.1 inert, by the same argument.

---

## C1.5 — Docking gate — **CONFIRMED** (and not where the brief expected)

**The thresholds are `SimConfig` fields, not in `sequence_loader.py` or
`contact_estimator.py`.** Both files were read: `sequence_loader.py` parses the
`.seq` gait description and has no capture threshold; `contact_estimator.py`
carries the **GMO contact-state** thresholds (`F_threshold = 5.0 N`,
`d_proximity = 0.020 m`, `d_contact = 0.005 m`, `d_reset = 0.030 m`,
`debounce = 3`, `config.py:60-66`) which drive the estimator's state machine and
**do not gate the weld**.

The weld gate is `SimulationLoop._dock_gate`,
**`crawlbot/simulation/sim_loop.py:1103-1135`**:

```python
pos_ok  = d          < cfg.weld_radius                       # :1121
ori_ok  = ori_err_deg < cfg.dock_ori_threshold_deg           # :1122
vel_ok  = twist_norm  < cfg.dock_twist_max                   # :1124
docked  = pos_ok and ori_ok and vel_ok                       # :1127
```

**Table III values — exact, with locations:**

| symbol | quantity | value | unit | `file:line` |
|---|---|---|---|---|
| ε_pos | `weld_radius` — gripper↔anchor site distance | **0.005** | m | `crawlbot/simulation/config.py:35` |
| ε_ori | `dock_ori_threshold_deg` — angle between gripper R and the anchor frame (Identity in the structure frame) | **5.0** | deg | `config.py:42` |
| ε_twist | `dock_twist_max` — **‖J_c·v⁻‖**, the 6-D weld-relative twist | **0.05** | mixed `√(‖v‖²[m/s]² + ‖ω‖²[rad/s]²)` | `config.py:58` |
| — | `dock_use_6d_twist` — selects the 6-D twist gate over the legacy linear-speed gate | **True** | — | `config.py:57` |
| — | `dock_check_delay` — gate not evaluated for the first N s of SS | **0.5** | s | `config.py:28` |
| (unused) | `dock_vel_max` — legacy linear `_gripper_speed` gate | 0.01 | m/s | `config.py:36`; live only when `dock_use_6d_twist=False` |

`J_c = [j_pos(gripper) − j_pos(anchor) ; j_rot(gripper) − j_rot(anchor)]` from
`mj_jacSite`, contracted with the **full** `mj_data.qvel` (structure freejoint
and wheels included, so common-mode structure motion cancels between the two
sites) — `_weld_relative_twist`, `sim_loop.py:1075-1101`.

Two prerequisites gate the gate itself (`sim_loop.py:1783-1786`):
`(t − t_ss_start) > dock_check_delay` **and**
`(t − t_ss_start) ≥ swing_early_finish_fraction · T_step` with
`swing_early_finish_fraction = 1.0` (`config.py:442`).
**Consequence, load-bearing for C1.6: the dock can never fire before the
planned swing has completed.**

**⚠ The ε_twist value the paper is about to print is documented in-source as
untuned.** `config.py:54-56`, verbatim:

> "dock_twist_max units are a mixed lin+ang twist norm
> (sqrt(‖v_lin‖²[m/s] + ‖ω‖²[rad/s])); default is a starting point for the J2
> characterization sweep, **NOT a tuned value**."

C1.6 shows this untuned 0.05 is the single largest term in the paper's
cost-of-constraint number, and that it sets the worst-case reported dock.

---

## C1.6 — Traversal-time comparison — **DEFECT: the paper's sentence is wrong**

### Both durations — CONFIRMED

| run | first tick | last tick | span | last dock (`dock_events`) |
|---|---:|---:|---:|---:|
| **C managed** | 0.01 s | 84.54 s | **84.53 s** | 64.54 s |
| **U unmanaged** | 0.01 s | 74.99 s | **74.98 s** | 54.99 s |
| **Δ** | | | **+9.55 s** | **+9.55 s** |

The paper's "84.5 s against 75.0 s" is reproduced from the committed artifacts.
✔ (`c25_fulldiag.csv`, `u25_fulldiag.csv`; the 20.0 s trailing settle,
`settle_seconds = 20.0`, is common to both and cancels in the difference.)

### Where the 9.55 s actually goes

**Method.** From `sim_loop.py:1783-1786` the gate opens exactly at
`t_ss_start + T_step`, so the first tick with `gate_eval == 1` marks the end
of the planned swing:

```
T_step        =  t(first gate evaluation)  −  t(SS start)
post-swing hold =  t(dock)                 −  t(first gate evaluation)
```

**Inference validated** against the C run's logged plan — `preplanner_T_steps`
= `[2.775, 7.652, 3.452, 7.924, 3.433, 7.772]` vs inferred
`[2.80, 7.70, 3.50, 8.00, 3.50, 7.80]`: agreement to the 0.1 s export cadence
on all six steps.

| step | T_step C | T_step U | ΔT | hold C | hold U | Δhold | SS C | SS U | ΔSS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.775 | *n/m* | — | 0.125 | ~0.1 | — | 2.90 | 2.80 | +0.10 |
| 1 | 7.652 | 7.90 | −0.25 | 1.348 | 1.30 | +0.05 | 9.00 | 9.20 | −0.20 |
| 2 | 3.452 | *n/m* | — | **6.848** | **~0.1** | **+6.7** | 10.30 | 3.40 | **+6.90** |
| 3 | 7.924 | 7.20 | +0.72 | 1.276 | 1.40 | −0.12 | 9.20 | 8.60 | +0.60 |
| 4 | 3.433 | 4.20 | −0.77 | **6.867** | **5.30** | +1.57 | 10.30 | 9.50 | +0.80 |
| 5 | 7.772 | 7.20 | +0.57 | 1.328 | 1.40 | −0.07 | 9.10 | 8.60 | +0.50 |
| **Σ** | **33.01** | — | **+0.28** ⁽¹⁾ | **17.79** | — | **+1.42** ⁽¹⁾ | **50.80** | **42.10** | **+8.70** |

⁽¹⁾ summed over the four steps where U's `T_step` is measurable. *n/m* = not
measurable: the U run's `sim_log.json` is **not committed** (only its CSV and
meta), and the exporter emitted no `gate_eval` row for U steps 0 and 2, so
those two `T_step` values are bounded rather than measured — bounded tightly,
because the gate cannot fire before `T_step` and both steps docked essentially
at the swing's end (`T_step_U(0) ≤ 2.80`, `T_step_U(2) ≤ 3.40`, against
C's 2.775 and 3.452).

**Decomposition of the +9.55 s:**

| component | Δ | share |
|---|---:|---:|
| longer **planned swings** (`T_step`, i.e. the constrained plan) | **≈ +0.3 s** | **3 %** |
| longer **post-swing convergence hold** (the dock gate not yet satisfiable) | **≈ +8.4 s** | **88 %** |
| longer **DS / settle** | +0.85 s | 9 % |

**The constrained plan is not the cause.** The momentum-constrained
pre-planner produces swings within ~0.3 s of the unconstrained ones across the
whole traversal. On the four directly comparable steps the managed plan is a
net **0.28 s** longer — 3 % of the reported cost.

### The actual mechanism, from the source-of-truth `dock_gate_trace`

**~70 % of the entire 9.55 s is one event: the twist gate rejecting a
pose-valid capture on step 2.**

```
C step 2 — dock_gate_trace (results/gate_run_scratch/sim_log.json)
   t = 21.90   d = 4.334 mm   ori = 0.105°   twist = 0.060522   pos✓ ori✓ twist✗   NOT FIRED
   t = 22.00   d = 4.941 mm   ori = 0.234°   twist = 0.057386   pos✓ ori✓ twist✗   NOT FIRED
   t = 28.50   d = 4.990 mm   ori = 0.019°   twist = 0.005781   pos✓ ori✓ twist✓   ***DOCK***
```

Both pose criteria were satisfied at **t = 21.90 s**, at **4.334 mm**. The
capture was refused because the weld-relative twist was `0.0605` against the
`0.05` threshold — a **21 % overshoot** of an explicitly untuned bound. The arm
then recoiled and took **6.6 s** to return, docking at a *worse* distance,
4.990 mm.

The same pattern repeats on step 4:

```
C step 4:  t = 46.58  d = 3.409 mm  ori = 0.157°  twist = 0.057616  → rejected
           t = 46.68  d = 3.326 mm  ori = 0.230°  twist = 0.050607  → rejected (by 1.2 %)
           t = 53.38  d = 4.954 mm  ori = 0.012°  twist = 0.004990  → ***DOCK***
```

Across all six steps there are exactly **4 pose-valid, twist-rejected
evaluations — 2 on step 2, 2 on step 4** — and those are precisely the two
steps carrying a ~6.85 s hold. Every other step holds ~1.3 s.

The gripper-distance trace makes the physical picture concrete
(`d_grip_swing_mm`, C step 2, times relative to SS start):

```
C:  0.00 → 799.8 mm    2.80 →  55.9    3.50 →   7.7  (swing ends, T_step=3.45)
    4.20 →  21.3       4.90 →  24.5    ← recoil, ~17 mm away from the anchor
    6.30 →  18.7       7.70 →  14.6    9.10 →   8.5   9.80 → 5.9   → dock 10.30
U:  0.00 → 799.7 mm    2.60 →  61.9    3.00 →  11.2   3.20 →   2.9  → dock 3.40
```

The managed run overshoots at the end of the swing and re-converges over ~6 s;
the unmanaged run closes monotonically and docks immediately.

### What the paper must say instead

The envelope constraint **is** the upstream cause — the momentum-limited wrench
is what leaves the swing unable to arrest cleanly, producing the recoil and the
excess weld-relative twist. But the sentence as shipped in r17 attributes the
9.5 s to the constraint in the sense of *a longer constrained plan*, and that
is measurably false (3 % of the effect). The honest mechanism is two-stage:

> the envelope limits the wrench available to terminate the swing → the
> managed run arrives with ~10× more weld-relative twist → the 6-D capture gate
> (ε_twist = 0.05, an untuned bound) refuses the capture → ~6.8 s of
> re-convergence on two of six steps.

**Two consequences the paper should not ship without:**

1. **The reported worst-case dock is set by the gate, not by tracking.** Step 2
   reached **4.334 mm** at t = 21.90 s and is reported at **4.990 mm** — the
   4.99 mm figure, and its celebrated **0.01 mm margin**, is the *second*
   approach after a twist rejection. Step 4 likewise reached 3.326 mm and is
   reported at 4.954 mm. The "worst margin 0.01 mm" is not a tracking limit.
2. **The cost-of-constraint number is threshold-sensitive.** Step 4's second
   rejection missed by 1.2 % (0.050607 vs 0.050). A sentence quantifying the
   cost of the envelope in seconds is, at this operating point, largely
   quantifying the cost of one untuned gate constant. C4 should report ε_twist
   alongside the timing so the two are not conflated.

---

## C1.7 — Structure inertia — **REPORTED**, with a latent defect in the fallback path

### As declared in the plant model the canonical actually loads

`models/VISPA_crawling_rwa3.xml:80-83` (the file named at `gate/run_gate.py:45`
and `scripts/diag_cooperative_arms.py:51`):

```xml
<body name="structure" pos="0 0 -1.8">
  <freejoint name="structure_free"/>
  <inertial pos="0 0 0" mass="7110"
            fullinertia="597 1493 1777 0 0 0"/>
```

**For Table II:**

| quantity | value | unit |
|---|---|---|
| structure mass | **7110** | kg |
| structure inertia, body frame, about its CoM (which is at the body origin, `pos="0 0 0"`) | **I_xx = 597, I_yy = 1493, I_zz = 1777**, products of inertia all zero | kg·m² |
| implied radius of gyration `√(I/M)` | **x 0.290, y 0.458, z 0.500** | m |
| structure + 3 wheels (subtree) | 7111.5 | kg |
| robot | 71.056 | kg (⇒ mass ratio 1.00 %) |
| whole model | 7182.556 | kg |

`_mutate_mjcf` scales structure mass **and** inertia by `0.01/mass_ratio`, but
is a **no-op at the canonical `mass_ratio = 0.01`** (guard
`abs(mass_ratio − 0.01) > 1e-9`, `scripts/diag_cooperative_arms.py:88`), so the
committed values above are what ran, unscaled, and the file is restored under an
md5 assert.

**The ~0.5 m radius of gyration is confirmed and worth the flag the brief
anticipates** — for a stronger reason than the 2.2 m traverse. The *same file*
draws the structure as a beam of half-extents `2.4 × 0.4 × 0.025`
(`:85`), i.e. **4.8 m × 0.8 m × 0.05 m**. A uniform box of 7110 kg with those
dimensions would have `I ≈ [381, 13653, 14032]` kg·m² — radius of gyration
1.39 m about y and z, **~9× the declared value**. The explicit `<inertial>`
overrides any geom-derived inertia, so the *dynamics* use 597/1493/1777 and the
geometry is visual only. If the paper prints the inertia next to a figure
showing the beam, it should say the inertia is a declared parameter rather than
implied by the drawn geometry.

### The fallback literal still exists — and disagrees with the live value

`crawlbot/simulation/sim_loop.py:254-256`:

```python
sid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, 'structure')
self._struct_I = self.mj_model.body_inertia[sid].copy() \
    if sid >= 0 else np.array([597.0, 1493.0, 1777.0])
```

Measured on the compiled model:

```
m.body_inertia[structure] = [1777., 1493.,  597.]
m.body_iquat [structure]  = [0., 0.70710678, 0., 0.70710678]
fallback literal          = [ 597., 1493., 1777.]
np.allclose(live, fallback) -> False
```

**There is no physical disagreement, but there is a real axis-order bug.**
MuJoCo diagonalises the declared inertia and stores the principal moments in a
rotated principal frame; `body_iquat` (a 180° rotation about `(1,0,1)/√2`,
mapping x↔z) is what reconciles `[1777, 1493, 597]` with the XML's
`I_xx=597, I_yy=1493, I_zz=1777`. The two arrays describe the same tensor.

But `_struct_I` is consumed as a plain per-axis `[I_x, I_y, I_z]` diagonal —
`body_iquat` is never read. So:

- the **live** path treats the structure as `I_x = 1777, I_z = 597`;
- the **fallback** path treats it as `I_x = 597, I_z = 1777`;
- **the x and z entries are transposed between the two**, a 3× error on both
  axes, and neither path is frame-aware.

**Severity: latent, and inert on the canonical.** `_struct_I` is read at
exactly two sites — `sim_loop.py:2814` (`legacy_pd_model`) and `:2876`
(`legacy_pid_model`) — and the canonical mode is **`legacy_pid_numerical`**,
which derives `ω̇_s` by finite difference and never touches `_struct_I`. So no
committed result is affected. It is exactly the class of silent
model/constant disagreement the brief asked C1.7 to look for, and it should
become a hygiene-stream ticket: read `body_iquat`, or drop the fallback.

---

## Cross-cutting: one CLAUDE.md "Known Issue" is stale

CLAUDE.md states, under Known Issues:

> "**Fig-3 conservation quantity ‖L_total‖ is NOT in the fulldiag export**
> (verified: no `Ltot` column in `c25_fulldiag.csv`)"

**This is no longer true at `eecbf94`.** `c25_fulldiag.csv` columns **63–66**
are `Ltot_x_Nms`, `Ltot_y_Nms`, `Ltot_z_Nms`, `Ltot_norm_Nms`, and columns
60–62 are `omega_s_{x,y,z}_radps` — added by the T2 drift-closure work
(`REPO_STATE.md` §2.2). Both fulldiag CSVs carry them, and
`c25_fulldiag_meta.json` additionally carries 44 `ltot_snapshots`.
**C3.2 (a)–(f) is therefore not blocked on a new export**, and the paper's
Fig-3 pending item may already be satisfiable from committed data. Flagged
here rather than edited, since CLAUDE.md is shared with the hygiene stream.

---

## STOP

C1 is complete: 7 items, 4 CONFIRMED, 2 with DEFECTs to carry to the paper
(C1.2 frame, C1.4 signs), 1 with a wrong attribution that a shipped r17
sentence depends on (C1.6), 1 reported with a latent code defect (C1.7).

Nothing was fixed. Awaiting explicit GO before **C3.4** (which the brief
schedules ahead of C2, and which is unblocked — the timing and iteration
channels are columns 54–56 of the committed canonical CSV).
