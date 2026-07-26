# T15-post-5 — Torso reference generation pipeline audit

Read-only architectural analysis of how the 6D torso reference is
produced and consumed across the planning / solver stack that
drove the T15 bug1fix_vel run. No edits, no simulation.

Paths referenced are relative to `/home/user/CrawlBot_control/`.

---

## §1 Pipeline map (Q1)

### 1.1 Modules and entry points

The full stack that runs between scheduler output and joint-torque
command, in the order they execute per SS step:

| # | Stage | Module | Class / function | Entry | Cadence |
|---:|---|---|---|---|---|
| 0 | Static setup | `crawlbot/simulation/sim_loop.py` | `SimulationLoop.setup` | `sim_loop.py:151` | once at run init |
| 1 | Contact scheduler | `crawlbot/planning/contact_scheduler.py` | `ContactScheduler.plan_traversal` | `contact_scheduler.py:203` | once at `setup()`; `set_step_duration` called once per step |
| 2 | Coarse pre-planner (IPOPT NLP) | `crawlbot/planning/coarse_preplanner.py` | `CoarsePrePlanner.solve` | `coarse_preplanner.py:416`; invoked from `sim_loop._run_preplanner` (`sim_loop.py:962`) and dispatched inside `_setup_torso_for_step` (`sim_loop.py:896`) | **once per step** at SS setup |
| 3 | Torso planner | `crawlbot/planning/torso_planner.py` | `TorsoPlanner.add_phase`, `.set_hold` | `torso_planner.py:155`, `:85`; programmed inside `_setup_torso_for_step` at `sim_loop.py:930–937` | programmed once per step; queried per NMPC tick and per QP sub-step |
| 4 | Swing planner | `crawlbot/planning/swing_planner.py` | `SwingPlanner.reference_at` | `swing_planner.py:221`; set up in `sim_loop.py:1214` (`set_swing_orientation`) | queried per NMPC tick (QP loop at `sim_loop.py:1815`) |
| 5 | Centroidal NMPC (IPOPT NLP) | `crawlbot/solvers/centroidal_nmpc.py` + `crawlbot/solvers/nmpc_solver.py` | `CentroidalNMPC` (problem build), `NMPCSolver.solve` | `nmpc_solver.py:388`; invoked per-tick at `sim_loop.py:1615` | **once per NMPC tick** (`dt_nmpc = 0.1 s` = 10 Hz, `config.py:23`) |
| 6 | CoM→torso mapping | `crawlbot/core/com_to_torso_mapping.py` | `CoMToTorsoMapping.compute` | `com_to_torso_mapping.py:133`; invoked per QP sub-step at `sim_loop.py:1749` | **per QP sub-step** (N sub-steps per NMPC tick) |
| 7 | Whole-body QP | `crawlbot/solvers/wholebody_qp.py` + `crawlbot/solvers/hierarchical_qp.py` | `WholeBodyQP.solve` (atop `HierarchicalQP`) | `wholebody_qp.py:195`; invoked at `sim_loop.py:~1690` within the inner QP loop `for qs in range(self.n_qp_per_nmpc)` (`sim_loop.py:1691`) | **per QP sub-step** (`dt_qp = 0.01 s` = 100 Hz, `config.py:24`) |

`n_qp_per_nmpc` is defined as `int(round(dt_nmpc / dt_qp))` = 10
sub-steps per NMPC tick.

### 1.2 Data flow (text-level graph)

```
              ┌─────────────────────────────────────────────────────┐
              │ setup() sim_loop.py:151                             │
              │  anchors (structure-local) from MJCF sites          │
              └───────────────────┬─────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ [1] ContactScheduler.plan_traversal(start_a,start_b,n_steps)        │
│     Produces: GaitPlan = list[GaitPhase] with                       │
│       anchor_a_idx, anchor_b_idx, swing_arm, swing_from_idx,        │
│       swing_to_idx per phase, plus t_start/t_end.                   │
│     contact_scheduler.py:203                                        │
└───────────────────┬─────────────────────────────────────────────────┘
                    │  ss_phase_idx, stance_a, stance_b, swing_arm,
                    │  target_idx              (sim_loop.py:1101–1103)
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ [2] CoarsePrePlanner.solve(r_com_0, v_com_0, L_com_0, r_com_goal,   │
│          r_C_stance, c_const, T_step=None)                          │
│     IPOPT NLP over knots k=0..M with x_k=(r_com,v_com,L_com) and    │
│     u_k=(f1,tau1,f2,tau2). Returns CoarsePlanResult with            │
│       T_step, r_com[], v_com[], L_com[], interpolants r/v/L_com_at. │
│     coarse_preplanner.py:416, :129–150                              │
└───────────────────┬─────────────────────────────────────────────────┘
                    │  T_step, r_com[], v_com[], L_com[]
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ [2b] plan.set_step_duration(ss_phase_idx, T_step)                   │
│      Updates GaitPhase.duration and cascades t_start/t_end.         │
│      contact_scheduler.py:135–160 via sim_loop.py:911               │
│      Plus Option Z reset:                                           │
│        self._t_plan_offset = t_ss_start - plan.t_start[ss_phase_idx]│
│        sim_loop.py:919                                              │
└───────────────────┬─────────────────────────────────────────────────┘
                    │  T_step, p_t0/R_t0 (IK start), p_t1/R_t1 (IK end),
                    │  delta_com_start/_end, cfg.torso_early_finish_fraction
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ [3] TorsoPlanner.add_phase(t_ss_start, t_ss_start+T_step,           │
│          p_t0, R_t0, p_t1, R_t1,                                    │
│          delta_com_start, delta_com_end,                            │
│          early_finish_fraction=cfg.torso_early_finish_fraction)     │
│     torso_planner.py:155 via sim_loop.py:932                        │
│     Produces: quintic-profile 6D torso + per-phase ComReference     │
│       (trapezoidal-CoM) accessible via reference_at / com_ref_at.   │
│     NO optimization — pure quintic/linear interpolation.            │
└───────────────────┬─────────────────────────────────────────────────┘
                    │  [queried per tick]
                    ▼
         ─── PER NMPC TICK (10 Hz) loop starts here ──────────────────
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ [3q] TorsoPlanner.reference_at(t) → TorsoReference(p,R,v,a,         │
│                                                   r_com_ref,...)   │
│      sim_loop.py:1516                                               │
│ [3q] TorsoPlanner.com_reference_at(t + N·dt_nmpc) → ComReference    │
│      sim_loop.py:1521                                               │
│ [3q] TorsoPlanner.l_com_reference_at(t_mid) → (3,) angular mom.     │
│      sim_loop.py:1593                                               │
└───────────────────┬─────────────────────────────────────────────────┘
                    │  r_com_ref, v_com_ref, L_com_ref_nmpc
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ [5] NMPCSolver.solve(r_com, v_com, L_com,                           │
│         r_com_ref, v_com_ref, contact_config,                       │
│         warm_start=True, hw_current, L_com_ref)                     │
│     wraps CentroidalNMPC (x = (r_com,v_com,L_com) ∈ R^9,            │
│     u = (f1,tau1,f2,tau2) ∈ R^12, N=20 knots, dt=0.05)             │
│     centroidal_nmpc.py:58–105 config, :120– build                   │
│     nmpc_solver.py:388 solve                                        │
│     Produces: (rp, vp, _, lr, info)                                 │
│       rp=x_opt[0:3,1] (r_com one knot ahead), vp similar,           │
│       lr = contact-wrench reference (12,),                          │
│       af = feedforward CoM acceleration from lr (sim_loop.py:1621). │
│     Invoked per NMPC tick at sim_loop.py:1615                       │
└───────────────────┬─────────────────────────────────────────────────┘
                    │  rp, vp, af (r/v/a_com_ref for the upcoming tick)
                    ▼
         ─── PER QP SUB-STEP (100 Hz) inner loop starts here ─────────
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ [3q'] TorsoPlanner.reference_at(tq_planner) for the *angular* ref   │
│       sim_loop.py:1734                                              │
│ [6]  CoMToTorsoMapping.compute(r_com_ref=rp_interp,                 │
│           v_com_ref=vp_interp, a_com_ff=af,                         │
│           q_current=q_map, dq_current=dq_map)                       │
│       com_to_torso_mapping.py:133 via sim_loop.py:1749              │
│       Produces: r_b_ref, v_b_ref, a_b_ff — torso LINEAR pos/vel/acc │
│       refs from CoM refs via the mass-weighted delta(q) correction. │
│                                                                     │
│  Result fed to QP as (p_torso_ref_used, v_torso_ref_used,           │
│                       a_torso_ff_used). Angular part still from     │
│  torso_planner.reference_at. See sim_loop.py:1740 'mapping_bypass_  │
│  in_ss' branch: when set, the linear torso reference is FROZEN at   │
│  its SS-entry value (`self._ss_entry_p_torso`) throughout SS —      │
│  mapping bypassed for linear (angular still from TorsoPlanner).     │
└───────────────────┬─────────────────────────────────────────────────┘
                    │  p_torso_ref, v_torso_ref, a_torso_ff (used by QP)
                    │  swing-arm EE ref from [4] — same cadence
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ [4q] SwingPlanner.reference_at(tq_plan = tq − t_plan_offset)        │
│      swing_planner.py:221 via sim_loop.py:1815                      │
│      Returns: p_ee, R_ee, v_ee, omega_ee, a_ee, alpha_ee            │
│      in the structure frame (quintic + bump).                       │
└───────────────────┬─────────────────────────────────────────────────┘
                    │  p_ee_ref, R_ee_ref, v_ee_ref, a_ee_ff
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ [7] WholeBodyQP.solve(...)  ← torso + EE + CoM + posture tasks      │
│     wholebody_qp.py:195 via sim_loop.py:~1690                       │
│     Uses HierarchicalQP (hierarchical_qp.py:70, method='weighted')  │
│     Produces: qdd, tau (joint torques), lambda_qp (wrenches)        │
│     Inputs: rs (Pinocchio RobotState), J_torso/J_ee Jacobians,      │
│       p/v/a torso refs, p/R/v/a EE refs, CoM ref (rp/vp/af), hw,    │
│       contact Jacobians Jc/Jdc, posture q_nom.                      │
└─────────────────────────────────────────────────────────────────────┘
                    │  tau → MuJoCo step (physics)
                    ▼
```

### 1.3 Notes on cadence

- `dt_nmpc = 0.1 s` (10 Hz, `config.py:23`); `dt_qp = 0.01 s` (100 Hz, `config.py:24`); **10 QP sub-steps per NMPC solve.**
- CoarsePrePlanner runs **once per step** (at SS setup inside `_setup_torso_for_step`, `sim_loop.py:896`).
- TorsoPlanner is **programmed once per step** (`sim_loop.py:930–937`) and **queried at every NMPC tick and every QP sub-step**.
- Centroidal NMPC runs **once per NMPC tick** (`sim_loop.py:1615`); produces a one-step-ahead plan that is consumed by the next 10 QP sub-steps via interpolation (`sim_loop.py:1680–1720` linear interp between knot 0 and knot 1).
- CoM→torso mapping runs **once per QP sub-step** (`sim_loop.py:1749`).
- Whole-body QP solves **once per QP sub-step**.

### 1.4 Unused / legacy modules in the current production path

- `crawlbot/planning/locomotion_planner.py` (`LocomotionPlanner`) is not imported by `sim_loop.py` at HEAD of `claude/t15-bug1-fix`. `sim_loop.py:46` comment reads `# LocomotionPlanner removed — CoM reference comes from TorsoPlanner`. Grep result: referenced only from `Misc/scripts/test_integration.py` and `crawlbot/planning/__init__.py` (re-export). Flagging it as **not in scope** for the live reference pipeline.

---

*(§2 Torso reference identification follows — one section at a
time per the prompt's instruction.)*

---

## §2 Torso reference identification (Q2)

### 2.1 What "the torso reference used by the QP" actually is

The `WholeBodyQP.solve` call consumes **four** torso-reference
quantities (`wholebody_qp.py:234–241`):

- `p_torso_ref` ∈ ℝ³ (linear position reference, structure frame)
- `R_torso_ref` ∈ SO(3) (angular reference)
- `v_torso_ref` ∈ ℝ⁶ (twist: linear + angular velocity)
- `a_torso_ff` ∈ ℝ⁶ (feedforward acceleration)

These are assembled per-QP-sub-step at `sim_loop.py:1730–1796`
into a dict `tkw` (`sim_loop.py:1797–1802`) that is handed to
`qp.solve(...)`. The **composite source** for each field depends
on a branch structure controlled by `cfg.mapping_bypass_in_ss`
and by the phase (SS / DS).

### 2.2 Source branches (conditional data flow)

```
                 ┌─ TorsoPlanner.reference_at(tq_planner)        ◄─ always
                 │    TorsoReference(p, R, v, a) where
                 │      p, R        = quintic IK-endpoint interp
                 │      v = [v_lin(3), omega(3)]
                 │      a = [a_lin(3), alpha(3)]
      tr ←─────┤
                 │
  Branch A  ── SS and cfg.mapping_bypass_in_ss and self._ss_entry_p_torso
              (sim_loop.py:1734–1742):
                 p_torso_ref_used  = self._ss_entry_p_torso.copy()
                 v_torso_ref_used  = [0, 0, 0,  tr.v[3:6]]
                 a_torso_ff_used   = [0, 0, 0,  tr.a[3:6]]
                 # i.e. linear pos FROZEN at SS-entry torso position;
                 # linear velocity/accel zeroed; angular ref, omega, alpha
                 # from TorsoPlanner quintic.

  Branch B  ── (SS or DS) and self.mapping is not None and cfg.use_m2_stack
              (sim_loop.py:1743–1773):
                 q_map, dq_map = self._planned_arm_config(tq, rs) (SS)
                                 or rs.q, rs.v                     (DS)
                 (r_b_ref_m, v_b_ref_m, a_b_ff_m, _) =
                     self.mapping.compute(
                         r_com_ref = rp_interp,
                         v_com_ref = vp_interp,
                         a_com_ff  = af_for_mapping,
                         q_current = q_map,
                         dq_current = dq_map)
                 # DS: optional quintic blend from SS-exit pose to live
                 # mapping output over cfg.ds_ramp_duration_s
                 # (sim_loop.py:1755–1770).
                 p_torso_ref_used  = r_b_ref_m
                 v_torso_ref_used  = [v_b_ref_m,  tr.v[3:6]]
                 a_torso_ff_used   = [a_b_ff_m,   tr.a[3:6]]

  Branch C  ── else (neither of the above)
              (sim_loop.py:1774–1777):
                 p_torso_ref_used  = tr.p
                 v_torso_ref_used  = tr.v
                 a_torso_ff_used   = tr.a

  Finally (sim_loop.py:1795–1796):
                 R_torso_ref_used  = (self._diag_frozen_R_b_ref
                                      if self._diag_freeze_ref else tr.R)
                 # By default: tr.R from TorsoPlanner.
```

### 2.3 Which branch T15 bug1fix_vel takes

From the T15 bug1fix_vel run config (T15_report.md §1.2):
`mapping_bypass_in_ss = True`, `aocs_off_in_ds = True`,
`swing_early_finish_fraction = 0.80`, and `use_m2_stack` is the
default True. So:

- **During SS:** Branch **A** — linear torso reference frozen at
  `self._ss_entry_p_torso`; angular reference quintic-interpolated
  by TorsoPlanner. The CoMToTorsoMapping is not invoked for
  linear torso reference in SS.
- **During DS:** Branch **B** — linear torso reference via
  `CoMToTorsoMapping.compute(rp_interp, vp_interp, af,
  q_current=rs.q, dq_current=rs.v)`, with the post-dock
  quintic blend from the SS-exit pose for `cfg.ds_ramp_duration_s`
  seconds (`sim_loop.py:1756–1770`). Angular reference still from
  TorsoPlanner.

### 2.4 TorsoPlanner representation (what tr.p / tr.R / tr.v / tr.a *are*)

The TorsoPlanner is **not** an optimization — it is a piecewise
quintic interpolator (`torso_planner.py:362–395`) between
IK-derived endpoints (`p_t0, R_t0`) → (`p_t1, R_t1`) over
`[t_ss_start, t_ss_start + effective_duration]`, where
`effective_duration = T_step · early_finish_fraction`.

- Linear: `p(t) = p_t0 + s(τ) · (p_t1 − p_t0)`, with quintic
  `s(τ) = 10τ³ − 15τ⁴ + 6τ⁵` and τ = clip((t − t_start)/effective_duration, 0, 1).
- Angular: `R(t) = R_t0 · exp( s(τ) · log(R_t0ᵀ R_t1) )` — also
  time-parameterised by the same `s(τ)`.
- Derivatives: `v_lin = ṡ(τ) · Δp`, `ω = R · (ṡ · ω_total)`,
  `a_lin = s̈(τ) · Δp`, `α = R · (s̈ · ω_total)`.

After the quintic reaches τ = 1 (at `t_ss_start +
effective_duration`), all four quantities are clipped — `p, R`
at the terminal pose, `v = 0`, `a = 0` — and the phase "holds"
for the remainder of `[t_start, t_end]`. `SimConfig.torso_early_finish_fraction`
default is `1.0` (`config.py:178`).

### 2.5 IK endpoints — where (p_t0, R_t0) and (p_t1, R_t1) come from

`_setup_torso_for_step` (`sim_loop.py:794–951`):

- **Start pose `(p_t0, R_t0)`** = live torso pose read from the
  current Pinocchio state at SS-entry:
  ```
  pq_live, pv_live = mujoco_to_pinocchio(qpos, qvel)    (sim_loop.py:820)
  rs_s = self.robot.update(pq_live, pv_live)
  p_t0 = rs_s.oMf_torso.translation.copy()               (sim_loop.py:824)
  R_t0 = rs_s.oMf_torso.rotation.copy()                  (sim_loop.py:825)
  ```
- **End pose `(p_t1, R_t1)`** = result of dock-configuration IK
  against the post-swing anchor pair `(end_a, end_b)`
  (`sim_loop.py:828–879`):
  ```
  if target_arm == 'b':
      end_a, end_b = stance_a, target_idx
  else:
      end_a, end_b = target_idx, stance_b
  se3_a = self.sched.anchor_se3('a', end_a)             (sim_loop.py:837)
  se3_b = self.sched.anchor_se3('b', end_b)             (sim_loop.py:838)
  # Prefer fixed-R_torso IK, fall back to manipulability IK
  q_end = dock_configuration_fixed_rotation(...)  or
          dock_configuration(...)                        (sim_loop.py:847–874)
  rs_e = self.robot.update(q_end, zeros)
  p_t1 = rs_e.oMf_torso.translation.copy()               (sim_loop.py:877)
  R_t1 = rs_e.oMf_torso.rotation.copy()                  (sim_loop.py:878)
  ```

The IK selects end-configuration anchor indices from **scheduler
state** (`stance_a`, `stance_b`, `target_idx`), not from live
physical weld state (see T15-post-2 Q2.4 for the step-2 divergence
between scheduler `stance_a` and physical weld).

### 2.6 Horizon and recompute cadence

| Field | Horizon / validity | Recomputed |
|---|---|---|
| `(p_t0, R_t0, p_t1, R_t1)` | Entire SS window `[t_ss_start, t_ss_start + T_step]` (with early-finish quintic completing at `t_ss_start + ef·T_step`) | Once per step at `_setup_torso_for_step` (`sim_loop.py:896`) |
| `tr = torso_planner.reference_at(t)` | Single-tick evaluation of the quintic at `t` | Once per QP sub-step (100 Hz) |
| `p_torso_ref_used` (Branch A, SS) | Entire SS window (frozen scalar) | Once per SS entry (`self._ss_entry_p_torso` set at `sim_loop.py:946`) |
| `p_torso_ref_used` (Branch B, DS) | Single-tick evaluation | Per QP sub-step via `mapping.compute` |
| `R_torso_ref_used` | Single-tick quintic evaluation | Per QP sub-step |

- CoarsePrePlanner produces an r_com trajectory over `[0, T_step]`
  once per step but **does not enter the torso reference
  directly**; it feeds the NMPC's r_com reference (`sim_loop.py:1538–1551` —
  not shown, but the interpolants `T_step`-bounded) and the IK
  ingredients for `p_t1` / `R_t1`.
- Centroidal NMPC has its own **1-s prediction horizon** (`N = 20`,
  `dt = 0.05 s`, `centroidal_nmpc.py:67–68`), receives
  `r_com_ref, v_com_ref` from `torso_planner.com_reference_at`
  (`sim_loop.py:1521`) and `L_com_ref_nmpc` from
  `torso_planner.l_com_reference_at` (`sim_loop.py:1593`). The
  NMPC's output `(rp, vp)` feeds Branch B of §2.2 via `rp_interp`,
  `vp_interp` — i.e. it influences the *linear* torso reference
  only when mapping is NOT bypassed (i.e. DS, or when
  `mapping_bypass_in_ss=False`).

### 2.7 Representation summary (T15 bug1fix_vel, SS phase)

During SS with `mapping_bypass_in_ss=True` (the T15 bug1fix_vel
configuration), the effective torso reference seen by the QP over
the whole SS window is:

- `p_torso_ref(t) = self._ss_entry_p_torso` (constant, 3-D)
- `v_torso_ref(t) = [0, 0, 0, ω_ref(t)]` where `ω_ref` is the quintic
  angular reference from TorsoPlanner
- `a_torso_ff(t)   = [0, 0, 0, α_ref(t)]`
- `R_torso_ref(t)  = R_t0 · exp(s(τ) · log(R_t0ᵀ R_t1))` (quintic SLERP)

with the quintic τ running on `[0, T_step · early_finish_fraction]`
and the reference holding at the terminal pose `R_t1` for the
remainder of the SS. **The linear torso position reference is
therefore a constant** (SS-entry value) throughout each SS phase
in the current run, and only the **angular** reference is a
piecewise quintic.

### 2.8 Summary (facts only)

- The 6D torso reference fed to the QP is assembled at **100 Hz**
  from two sources: `TorsoPlanner` (always, for angular + for
  linear when Branches A and B are inactive) and
  `CoMToTorsoMapping` (for linear during Branch B).
- `TorsoPlanner` is a **piecewise quintic** interpolator between
  **IK-derived torso endpoints** `(p_t0, R_t0, p_t1, R_t1)`. It
  does no optimization; it stores two waypoints per SS step and
  interpolates them.
- In SS with `mapping_bypass_in_ss = True` (T15 configuration),
  the **linear** torso reference is a **constant** = SS-entry
  torso position; only the **angular** reference follows the
  quintic.
- In DS, the linear torso reference comes from
  `CoMToTorsoMapping.compute(r_com_ref = NMPC's rp_interp, …)`,
  with a quintic blend from SS-exit for the first
  `ds_ramp_duration_s` seconds.
- `(p_t1, R_t1)` (the IK-derived target torso pose) is computed
  once per step at SS-setup and carries all of the step's
  pre-determined torso "plan" into the QP. The torso planner
  does not revisit this target during SS.

*(§3 Cost functions follows.)*

---

## §3 Cost and constraint structure (Q3)

### 3.1 CoarsePrePlanner (IPOPT NLP)

`crawlbot/planning/coarse_preplanner.py`. Similar stage/terminal
cost structure to the NMPC, solving over `M` knots across
`[0, T_step]` with the same centroidal dynamics. Outputs
`(T_step, r_com[], v_com[], L_com[])` used downstream.

Per `coarse_preplanner.py:58–103` (config) and
`:129–150` (result interface):

- **Decision variables**: centroidal state `(r_com, v_com, L_com)`
  at each knot, contact wrenches `(f1, τ1, f2, τ2)` per stage,
  `T_step` (the step duration itself).
- **Cost**: tracking terms against `r_com_goal` + regularization
  of wrenches (same structure as the NMPC, see §3.2).
- **Constraints**: wrench feasibility (SOC), wheel-momentum box
  (same h_w formula as NMPC), state bounds.

The pre-planner's *output* that actually enters the torso
reference pipeline is just `T_step` and the `r_com/v_com/L_com`
interpolants — the CoM path and the step duration. The torso-pose
endpoints `(p_t0, R_t0, p_t1, R_t1)` for the TorsoPlanner are
**not** chosen by the pre-planner; they are computed by IK
(`dock_configuration_fixed_rotation` / `dock_configuration`) at
`sim_loop.py:847–874`.

### 3.2 Centroidal NMPC (IPOPT NLP)

`crawlbot/solvers/centroidal_nmpc.py`, horizon `N = 20`,
`dt = 0.05 s` (1 s lookahead; `centroidal_nmpc.py:67–68`).

#### 3.2.1 Cost terms

**Stage cost** (`centroidal_nmpc.py:188–205`):

```
L(x, u) = (r_com - r_ref)ᵀ Wr (r_com - r_ref)
        + (v_com - v_ref)ᵀ Wv (v_com - v_ref)
        + w_L · ||L_com - L_ref||²
        + uᵀ Wu u
```

with defaults (`centroidal_nmpc.py:69–76`):
- `Wr = diag(100, 100, 100)` — CoM position tracking
- `Wv = diag(10, 10, 10)` — CoM velocity tracking
- `w_L = 1.0` — angular momentum tracking
- `Wu` built from `Wu_f = 0.01` (force) and `Wu_tau = 0.001` (torque)

**Terminal cost** (`centroidal_nmpc.py:213–226`):

```
Lf(x) = (r_com - r_ref)ᵀ Qf_r (r_com - r_ref)
      + (v_com - v_ref)ᵀ Qf_v (v_com - v_ref)
      + Qf_L · ||L_com - L_ref||²
```

with `Qf_r = diag(1000, 1000, 1000)`, `Qf_v = diag(100, 100, 100)`,
`Qf_L = 10.0`.

**No term is a function of arm joint angles or arm Jacobians.**
The cost is expressed entirely in the centroidal state
(`r_com, v_com, L_com`) and contact wrenches.

#### 3.2.2 Path and terminal constraints

Per `centroidal_nmpc.py:234–330`:

| Constraint | Form | Default status |
|---|---|---|
| Contact wrench SOC | `||fⱼ||² ≤ f_max²`, `||τⱼ||² ≤ τ_max²` | active; `f_max=3000 N`, `τ_max=300 Nm` |
| L̇_com rate bound | `|L̇_com,i| ≤ τ_w_max` per component | `τ_w_max = ∞` (default, disabled) |
| Structure disturbance | `|Ḣ_s,i| ≤ τ_struct_max` | `τ_struct_max = ∞` (default, disabled) |
| Linear momentum | `||m·v_com||² ≤ p_max²` | `p_max = ∞` (default, disabled) |
| Wheel-momentum conservation box | `h_w^s(k) = c_simple − L_com(k) − r_com(k) × m·v_com(k) ∈ [-h_max_tight, +h_max_tight]` (path) and `|h_w(N)| ≤ κ·h_max_tight` (terminal) | gated by `enforce_hw_conservation` (default **False**) |
| State bounds | `L_com ∈ [-L_max, +L_max]`, `r_com, v_com` unbounded | `L_max = ∞` (default) |
| Control bounds | Component-wise `|f| ≤ f_max`, `|τ| ≤ τ_max` (then SOC tightens) | always active |

**No manipulability / whole-body-kinematics constraint is present
in the NMPC.** The tightest kinematic information the NMPC sees
is `r_C1, r_C2` (contact point positions from the scheduler) and
the robot mass `m` — no joint angles, no Jacobians.

### 3.3 TorsoPlanner (`torso_planner.py`)

No optimization, no cost. Pure piecewise quintic interpolator
between IK-derived endpoints. `grep -n "cost\|optim\|minimize" torso_planner.py` matches only the header comment.

### 3.4 SwingPlanner (`swing_planner.py`)

Same — no optimization, no cost. Quintic (position) + bump
(clearance) + delayed-cosine SLERP (orientation), evaluated
per-tick at `reference_at(t)`.

### 3.5 CoMToTorsoMapping (`com_to_torso_mapping.py`)

No optimization. Closed-form algebraic map:

```
r_b_ref = (m_total / m_b) · r_com_ref − δ(q_current) / m_b
v_b_ref = (m_total / m_b) · v_com_ref − δ̇(q, q̇) / m_b
a_b_ff  = (m_total / m_b) · a_com_ff
```

with `δ(q) = Σ_{i ≠ torso} m_i · r_i(q)` the mass-weighted sum of
non-torso body CoM positions (`com_to_torso_mapping.py:133–170`).
No cost, no optimization.

### 3.6 WholeBodyQP — task stack

`crawlbot/solvers/wholebody_qp.py`. Runs on `HierarchicalQP` with
`method='weighted'`, `weight_ratio=1.0` (M2 configuration —
`wholebody_qp.py:69–75`; task isolation comes from null-space
projection, not weight scaling).

#### 3.6.1 Task list (priorities assigned under M2 stack)

| Priority | Task | Form | Default α (SS) | Geometric projection | File:line |
|---:|---|---|---:|---|---|
| 1 | Torso 6D tracking | `J_torso · q̈ + J̇_torso · q̇ = a_torso_des(p_ref, R_ref, v_ref, a_ff, …)` | α_torso = **5e2** (ss_alpha_torso, `config.py:135`) | none — primary | `wholebody_qp.py:566` |
| 1 | hw slack penalty | Quadratic penalty on slack variables for `h_w(k+1) ∈ box` | w_hw_slack = 1e4 | none | `wholebody_qp.py:784` |
| 2 | EE 6D tracking | `J_ee · q̈ + J̇_ee · q̇ = a_ee_des(p_ee_ref, R_ee_ref, v_ee_ref, a_ee_ff)` | α_ee = **3e3** (ss_alpha_ee, `config.py:136`) | optionally projected into `null(A_torso)` via `cfg.ee_null_space` | `wholebody_qp.py:641, :645` |
| 3 | Posture regulation | `q̈ = Kp(q_nom − q) − Kd·q̇`, α_posture per joint, skipped in DS settle mode | α_posture = **2e1** (ss_alpha_posture, `config.py:137`) | projected into `null(A_torso) ∩ null(A_ee)` | `wholebody_qp.py:688–720` |
| 4 | Soft CoM residual | CoM tracking, projected into null-space of P1/P2 | α_com_soft = 5.0 | projected | `wholebody_qp.py:674` |
| 4 | Wrench tracking | `λ_contact = lr` (from NMPC) | α_wrench = 1e-2 | none | `wholebody_qp.py:739` |
| 4 | Reaction null-space | `H_base ← swing_arm · q̈_sw = 0` — minimizes base disturbance from swing arm | α_reaction = **0.0** (ss_alpha_reaction, `config.py:139` — **disabled**) | none | `wholebody_qp.py:744–757` |
| 5 | Joint torque minimization | `τ = 0` | α_torque = 1e0 | — | `wholebody_qp.py:764` |
| 6 | Acceleration regularization | `q̈ = 0` | α_reg = 1e-2 | — | `wholebody_qp.py:772` |

The `ee_null_space` projection is **not** active by default in
the T15 configuration. The P1/P2 cascade uses priority weighting
only; task isolation relies on the P3 posture task and below
being projected into `null(A_torso) ∩ null(A_ee)`.

#### 3.6.2 QP structural constraints

Not all QP content is a cost — hard constraints include:

- **Contact kinematic constraints**: `J_c · q̈ + J̇_c · q̇ = 0` for
  each active contact (consumed via the contact Jacobians
  `Jc, Jdc` from `rs.get_contact_jacobians`).
- **Momentum-safety slack box**: `h_w(k+1) ≤ h_max + s_upper`,
  `h_w(k+1) ≥ h_min − s_lower`, with `s_{upper,lower} ≥ 0` and a
  heavy penalty `w_hw_slack · (||s_upper||² + ||s_lower||²)` on
  the slacks in the cost (`wholebody_qp.py:91–100, :784`).
- **Actuator bounds**: joint torque box `τ ∈ [−τ_max, τ_max]`
  (config: `tau_max = 50·ones(14)` Nm by default,
  `wholebody_qp.py:131`). Joint acceleration bounds `|q̈| ≤ qdd_max`
  (default 50 rad/s², `wholebody_qp.py:134`).
- **Passivity inequality** (DS only, optional): energy-decay
  bound `dqᵀτ_q + 2α·T_kin ≤ 0` with `α = alpha_passivity = 1.0`
  (`wholebody_qp.py:93`).

**No manipulability / singularity-avoidance task or constraint
is present** in the QP task list or constraint set. The only
Jacobian-rank-related operation is the damped pseudo-inverse
`np.linalg.pinv(J_torso, rcond=1e-8)` used when forming the
null-space projector `N_torso` for lower-priority tasks
(`wholebody_qp.py:582, :620`). That damping protects the
projector itself from numerical degradation but does not propagate
any cost/constraint that would dissuade the planners upstream
from *commanding* configurations where `J_ee` is near-singular.

### 3.7 Summary — kinematic awareness per stage

| Stage | Optimization? | Cost/constraint involves … | Reads arm Jacobian? | Reads arm joint angles? |
|---|---|---|---|---|
| CoarsePrePlanner | Yes (NLP) | `(r_com, v_com, L_com)`, wrenches, wheel momentum | no | no (uses `r_C_stance` point only) |
| ContactScheduler | No | — | no | no |
| TorsoPlanner | No | — | no (but `δ_com` inputs come from IK) | via the IK that produced `p_t0/p_t1` |
| SwingPlanner | No | — | no | no |
| CoMToTorsoMapping | No | — | no (uses `δ(q)` — FK mass-weighted) | **yes** (needs current `q` for `δ(q)`) |
| Centroidal NMPC | Yes (NLP) | `(r_com, v_com, L_com)`, wrenches, hw box | no | no |
| WholeBodyQP | Yes (weighted QP) | `q̈, τ, λ`; tasks use `J_torso, J_ee, J_contact`; posture PD on `q` | **yes** (via `J_ee`) | **yes** (via posture task `q_nom − q`) |

The only two places where arm-kinematic information enters at all
are:

1. **CoMToTorsoMapping** (`com_to_torso_mapping.py:133`): uses
   `q_current` to compute the `δ(q)` correction on the linear
   torso reference. Not active in SS under
   `mapping_bypass_in_ss = True` (T15 configuration).

2. **WholeBodyQP** (`wholebody_qp.py`): consumes `J_ee, J_torso,
   J_contact` as task matrices and uses `q, dq` for the posture
   task. Task priorities are static; the QP never adapts them to
   the current conditioning of `J_ee` or `J_arm`. The
   `np.linalg.pinv(J_torso, rcond=1e-8)` calls use a fixed
   damping parameter independent of `σ_min(J_ee)`.

The IK step inside `_setup_torso_for_step` (`sim_loop.py:847–874`)
does consult a `w_fixed` weight (product of arm manipulability
measures) to choose between fixed-rotation and manipulability-
optimized IK modes (`sim_loop.py:851–862`), but this choice only
affects the `(p_t1, R_t1)` endpoints — it does not produce a
per-tick manipulability-aware trajectory, nor does it flag
trajectories whose *interior* (between `p_t0` and `p_t1`) passes
through near-singular configurations.

*(§4 Torso DOF flexibility follows.)*

---

## §4 Torso reference DOF flexibility (Q4)

### 4.1 What "free variables" the torso reference has during SS

The 6D torso reference at any instant during SS is parameterised
by **two static waypoints** and a **quintic time profile**:

- `p_t0` (3) — torso position at SS entry (live; read from
  Pinocchio state, `sim_loop.py:824`).
- `R_t0` (3 rotational DOFs via `log(R_t0ᵀ R_t1)`) — torso
  rotation at SS entry (live; read from Pinocchio state,
  `sim_loop.py:825`).
- `p_t1` (3) — torso position at SS exit (IK output; `sim_loop.py:877`).
- `R_t1` (3 rotational DOFs) — torso rotation at SS exit (IK
  output or `R_t0` in fixed-rotation mode; `sim_loop.py:878`).
- `T_step` (1) — quintic duration (from pre-planner;
  `sim_loop.py:896`).
- `early_finish_fraction` (1, config; `config.py:178`) — fraction
  of `T_step` over which the quintic completes before holding at
  the terminal value.

Given these inputs, the instantaneous torso reference is
**completely determined** by the TorsoPlanner quintic
(`torso_planner.py:362–395`). The quintic profile is fixed
(`s(τ) = 10τ³ − 15τ⁴ + 6τ⁵`); the time scaling `τ` is fixed;
the linear and angular interpolations between endpoints are
fixed (straight line in position, SLERP in rotation). No run-time
shape parameter remains free.

### 4.2 Per-endpoint flexibility

The endpoints themselves have some latitude:

| Endpoint | Derivation | DOFs available to the planner |
|---|---|---|
| `p_t0, R_t0` | Live Pinocchio state at SS entry | **None**: fixed by the physics at `_setup_torso_for_step` time |
| `T_step` | CoarsePrePlanner NLP (`coarse_preplanner.py:416`) | Chosen by the NLP subject to momentum-feasibility. Per `coarse_preplanner.py:108–115` it is part of the NLP decision vector. |
| `p_t1, R_t1` | IK at `_setup_torso_for_step` (`sim_loop.py:847–874`) | See 4.3. |
| `early_finish_fraction` | Config scalar (`torso_early_finish_fraction`, default 1.0) | Tunable offline, not per-step |

### 4.3 IK-level flexibility at `(p_t1, R_t1)`

Two IK branches are available (`sim_loop.py:845–862`), selected
automatically per-step:

#### 4.3.1 Fixed-rotation IK (`dock_configuration_fixed_rotation`, `ik.py:168–270`)

- **Constraint**: `R_torso = R_t0` (torso orientation HELD at the
  SS-entry rotation).
- **Free DOFs**: 3 torso-position DOFs + 14 arm-joint DOFs = 17.
- **Constraints imposed**: both tool frames at their respective
  anchor poses (2 × SE3 = 12).
- **Redundancy**: 17 − 12 = **5-DOF null space** (`ik.py:187–188`),
  iteratively projected by the Gauss-Newton loop.
- **Selection criterion**: used when the converged IK residual
  `err < 1e−4` *and* the Yoshikawa manipulability product
  `w_a · w_b ≥ ik_fixed_rotation_w_min` (default 1e−4,
  `config.py:170`). Otherwise falls back to 4.3.2.

#### 4.3.2 Manipulability-optimized IK (`manipulability_config`, `ik.py:274–378`)

- **Constraint**: both tools at anchor poses (12 SE3 constraints).
- **Optimization variable**: torso xyz position (3 DOFs), via
  `scipy.optimize.minimize(method='Nelder-Mead')` from 3 seed
  positions (`ik.py:346–360`).
- **Cost**: `−σ_min(J_a) · σ_min(J_b)` (negated product of minimum
  singular values of the arm Jacobians in `LOCAL` frame, each
  Jacobian restricted to arm-joint columns; `ik.py:334–337`).
- **Free DOFs after optimization**: torso rotation is fully
  determined by the IK (consistent with whatever IK branch the
  neutral-seed Gauss-Newton converges to); arm joints are
  whatever `solve_ik` returns at the optimal torso position.
- **Selection criterion**: used whenever fixed-rotation mode is
  disabled, fails, or has `w_product` below threshold.
- **Cache**: a pre-computed `torso_map[(a_idx, b_idx)]` table
  (`precompute_torso_map`, `ik.py:381`) is checked before solving;
  if the `(end_a, end_b)` pair is in the table, its pre-computed
  manipulability-optimized config is used directly
  (`sim_loop.py:858`).

### 4.4 Does the NMPC "fully determine" the torso reference?

**No.** Per §2 branch analysis:

- In **SS** with `mapping_bypass_in_ss = True` (T15 config): the
  linear torso reference during the SS is a constant = SS-entry
  torso position; no NMPC output enters it. The angular reference
  is a TorsoPlanner quintic between the IK-derived endpoints.
  **NMPC output does not enter the torso reference at all during
  SS in this config.**
- In **SS** with `mapping_bypass_in_ss = False`: linear ref comes
  from `mapping.compute(r_com_ref = rp_interp, v_com_ref = vp_interp, …)`,
  where `rp_interp / vp_interp` are interpolated between two NMPC
  `r_com` knots (`sim_loop.py:1680–1720`). NMPC output drives
  linear; angular still from TorsoPlanner.
- In **DS** (`mapping_bypass_in_ss` does not apply): same Branch
  B as above — NMPC drives linear (with a quintic blend from
  SS-exit pose for the first `ds_ramp_duration_s`).

### 4.5 Where torso-reference DOF flexibility actually lives

Summarising, the planner layers hold three tiers of torso-reference freedom:

| Tier | Stage | Decision | Flexibility |
|---|---|---|---|
| 1. Step-level | Pre-planner (NLP) | `T_step`, `r_com[]` trajectory | Chosen subject to wrench SOC, wheel-momentum box, CoM terminal match. **No arm-kinematic awareness.** |
| 1. Step-level | IK at SS setup (`dock_configuration_fixed_rotation` / `manipulability_config`) | `p_t1, R_t1` (torso end pose) | Fixed-rotation picks `R_t1 = R_t0` unless the 5-DOF null-space collapses (`w_product < 1e−4`). Manipulability mode chooses torso xyz to maximize `σ_min(J_a)·σ_min(J_b)` at the end pose. **Step-local single-instant optimization; does not consider trajectory interior.** |
| 2. Sample-level | TorsoPlanner quintic | Fixed-shape interpolation between (`p_t0, R_t0`) and (`p_t1, R_t1`) | **No flexibility** — no cost, no adjustable shape parameter. |
| 3. Sub-tick-level | CoMToTorsoMapping / QP | QP resolves `q̈, τ, λ` subject to task weights and contact constraints | Weights and null-space projections are static per phase; no manipulability-based adaptation. |

### 4.6 Where a manipulability-aware extension could enter (facts only — no design)

The analysis above identifies the architectural surfaces where
manipulability-awareness *could* be added without rearchitecting
the stack, each with the data already available at that point:

1. **IK end-pose (`_setup_torso_for_step`).** `manipulability_config`
   already optimizes `σ_min(J_a) · σ_min(J_b)` at the single
   endpoint (`ik.py:334–337`). A version that considered the
   trajectory's *minimum* `σ_min` across the straight-line
   interpolation from `(p_t0, R_t0)` to candidate `(p_t1, R_t1)`
   could be slotted in here. Data required: `q(τ)` along the
   trajectory, which requires running the IK at intermediate
   poses — possible (costly) within the same routine. **Runs
   once per step, not per tick.**

2. **TorsoPlanner shape function.** Currently a pure quintic with
   no parameters. Replacing `add_phase` (`torso_planner.py:155`)
   with a parametrised family (e.g. via control points or spline
   knots) and a cost that penalises trajectories passing through
   low-`σ_min(J_ee)` configurations would require adding an
   optimization loop (currently absent). **Runs once per step at
   setup time.**

3. **Centroidal NMPC cost.** Currently centroidal-state-only
   (§3.2.1). Adding an arm-kinematic proxy term would require
   feeding arm joint angles into the NMPC parameter vector (9-D
   state, 12-D control, plus `p = [r_ref, v_ref, r_C1, r_C2,
   c_simple, L_ref]` currently, `centroidal_nmpc.py:192–196`) and
   evaluating a Jacobian condition metric in CasADi. This is
   invasive — the NMPC is not currently aware of the arm
   configuration. **Runs per NMPC tick.**

4. **CoMToTorsoMapping.** Could adjust the linear torso reference
   to move the torso toward a better-conditioned configuration
   subject to the CoM constraint. Already reads `q_current`
   (`com_to_torso_mapping.py:133`). Not active in SS under T15
   config (`mapping_bypass_in_ss = True`). Would require flipping
   that flag or adjusting the bypass logic. **Runs per QP
   sub-step.**

5. **WholeBodyQP tasks.** Could make `α_ee` adaptive to
   `σ_min(J_ee)` (reduce EE-tracking weight when near-singular),
   or add an explicit manipulability-gradient task. Already
   reads `J_ee` per sub-step. **Runs per QP sub-step.** This
   would affect reactive behavior only, not the commanded
   trajectory.

### 4.7 Summary

- Once `_setup_torso_for_step` has run, the torso reference is
  fully prescribed — no per-tick freedom.
- The only step-level freedom is `(p_t1, R_t1)` (via IK mode
  selection) and `T_step` (via pre-planner).
- The IK modes already evaluate manipulability, but only at the
  single endpoint `(p_t1, R_t1)` — the step-2 singularity
  observed in T15-post-4 is in the *trajectory interior*, not at
  the endpoint.
- The NMPC's output feeds the torso reference only when
  `mapping_bypass_in_ss = False` (i.e. not the current T15 SS
  configuration).

*(§5 Arm-kinematics coupling points follows.)*

---

## §5 Arm-kinematics coupling points (Q5)

Every location where the running controller reads arm-joint
angles (`q_joints`), arm-joint velocities (`dq_joints`), or arm
Jacobians (`J_tool_a`, `J_tool_b` restricted to arm-joint columns,
`J_contacts`, or intermediate Jacobians). Ordered by stage.

### 5.1 Setup / one-time

| # | File:line | Quantities read | Downstream effect |
|---:|---|---|---|
| 5.1a | `sim_loop.py:215–222` (calling `crawlbot/core/ik.py:274`) | `J_a, J_b` (arm Jacobians) → `σ_min(J_a)·σ_min(J_b)` inside `manipulability_config` | Builds `torso_map[(ai, bi)]` — cached manipulability-optimized IK configurations for every anchor pair appearing in the plan. Used as initial qpos (`sim_loop.py:234`) and as IK fallback (`sim_loop.py:858`) and as `q_nominal` posture reference for the QP (`sim_loop.py:1192`). Runs **once per `setup()` call** at sim start. |

### 5.2 Per step (SS-setup)

| # | File:line | Quantities read | Downstream effect |
|---:|---|---|---|
| 5.2a | `sim_loop.py:820` → `robot.update(pq_live, pv_live)` | full `q, v` including arm slices | Produces `rs_s` with `p_t0, R_t0, r_com0, delta0`; `p_t0/R_t0` are the TorsoPlanner SS-entry endpoints (`sim_loop.py:824–825`, §2.5). |
| 5.2b | `sim_loop.py:847–856` → `dock_configuration_fixed_rotation` (`ik.py:168`) | `J_a, J_b` at converged q (to compute `w_product = w_a · w_b`) | Branch selector: if `err < 1e−4` AND `w_product ≥ 1e−4` (`config.py:170`), use fixed-rotation q_end; otherwise fall back. Feeds `p_t1, R_t1` to TorsoPlanner add_phase. |
| 5.2c | `sim_loop.py:858–862` → `torso_map.get(...)` or `manipulability_config` (`ik.py:274`) | Pre-cached or newly-solved manipulability-optimized q | Same — determines `p_t1, R_t1` of the TorsoPlanner. |
| 5.2d | `sim_loop.py:876–879` → `robot.update(q_end, 0)` | — (forward kinematics only; q passed in, not read live) | Computes `p_t1, R_t1, r_com1, delta1` at the IK end pose. |
| 5.2e | `sim_loop.py:960` | `self._step_q_start = pq_live.copy()`, `self._step_q_end = q_end.copy()` (full 21-dim q vectors) | Saved for per-tick `_planned_arm_config` interpolation during SS. |

### 5.3 Per NMPC tick (10 Hz, inside `_step`)

| # | File:line | Quantities read | Downstream effect |
|---:|---|---|---|
| 5.3a | `sim_loop.py:~1515` → `robot.update(pq, pv)` | full `q, v` | Produces `rs` with `J_torso, J_tool_a, J_tool_b, r_com, v_com, L_com, H, C, …`. Called each NMPC-tick entry. |
| 5.3b | `sim_loop.py:1616` → `nmpc.solve(r_com, v_com, L_com, …)` | **only** `rs.r_com, rs.v_com, rs.L_com` | NMPC does NOT see joint angles or arm Jacobians. (Confirmed by reading `centroidal_nmpc.py:249–270` — `x = [r_com, v_com, L_com]` and `p = [r_ref, v_ref, r_C1, r_C2, c_simple, L_ref]` are the only inputs.) |

### 5.4 Per QP sub-step (100 Hz, inner QP loop)

| # | File:line | Quantities read | Downstream effect |
|---:|---|---|---|
| 5.4a | `sim_loop.py:1693` → `robot.update(pq, pv, omega_struct)` | full `q, v` | Per-sub-step Pinocchio refresh (structure pose drifts within the 10 sub-steps of one NMPC tick). |
| 5.4b | `sim_loop.py:1696` → `robot.get_contact_jacobians(…)` (`robot_interface.py:433`) | `rs.J_tool_a, Jdot_dq_tool_a, rs.J_tool_b, Jdot_dq_tool_b` (stacked for active contacts) | Provides `J_contacts, Jdot_dq_contacts` to the WholeBodyQP (hard constraint `J_c q̈ + J̇_c q̇ = 0`). |
| 5.4c | `sim_loop.py:1746` → `_planned_arm_config(tq, rs)` (`sim_loop.py:759`) | `rs.q, rs.v` (floating-base part), `_step_q_start/_end` (arm slices) | Produces `q_map, dq_map` = live floating base + quintic-interpolated arm plan; fed to `mapping.compute(q_current=q_map, dq_current=dq_map)` (`sim_loop.py:1749`). Not active in SS when `mapping_bypass_in_ss=True`. |
| 5.4d | `sim_loop.py:1749` → `mapping.compute(…, q_current, dq_current)` (`com_to_torso_mapping.py:133`) | Full `q, dq` — uses `δ(q) = Σ m_i r_i(q)` (FK-derived, mass-weighted position sum; `com_to_torso_mapping.py:97–122`) | Produces `r_b_ref, v_b_ref, a_b_ff` — the linear torso reference for the QP (Branch B). |
| 5.4e | `sim_loop.py:1748` (DS branch) | `rs.q, rs.v` directly (no quintic interp; live state) | Same downstream — fed to `mapping.compute` during DS. |
| 5.4f | `sim_loop.py:1809` → `_get_ee_data(rs, swing_arm)` (`sim_loop.py:2219`) | `rs.J_tool_a` or `rs.J_tool_b` (the swinging arm's full-Jacobian, plus `Jdot_dq_tool_*` and `oMf_tool_*`) | Produces `J_ee, Jdq_ee, oMf_ee` — fed to the QP `ek` kwargs (`sim_loop.py:1811–1814`) as the EE-task Jacobian. |
| 5.4g | `sim_loop.py:1824–1828` → swing-arm v-slice lookup (`rs.arm_b_v_slice` / `rs.arm_a_v_slice`) | Arm-B or arm-A v-slice of `rs.H` (mass matrix) as `H_bs = rs.H[:6, sw_slice]` | Reaction-null-space task input (`qp.solve(..., H_base_swing=H_bs, swing_v_slice=sw_slice)`). **Gated by `alpha_reaction = 0.0` in SS config** — so this input currently does not affect the QP cost. |
| 5.4h | `sim_loop.py:1845` → `qp.solve(q_t=rs.q_torso, dq_t=rs.dq_torso, q=rs.q_joints, dq=rs.dq_joints, …)` | `rs.q_joints, rs.dq_joints` (14-dim arm-joint slices) | Fed to the WholeBodyQP for: (a) posture task `q̈ = Kp(q_nom − q) − Kd·dq` (`wholebody_qp.py:685–691`); (b) task-space PD on torso/EE uses `rs.q_torso, rs.dq_torso` directly. |
| 5.4i | `sim_loop.py:1850` → `qp.solve(..., J_com=rs.J_com, Jdot_dq_com=rs.Jdot_dq_com, …)` | `rs.J_com` (CoM Jacobian, which depends on arm joints by construction) | CoM-tracking task (priorities 1 if `use_m2_stack=False`, else 4 soft). |

### 5.5 Post-dock (one-shot per step boundary)

| # | File:line | Quantities read | Downstream effect |
|---:|---|---|---|
| 5.5a | `sim_loop.py:1366` → `robot.get_contact_jacobians(True, True)` | Both `J_tool_a, J_tool_b` | Used to compute the inelastic-impact velocity projection (`Jc_both`, `Lambda_inv`, impulse, `pv_post`). Modifies `mj_data.qvel` to zero the velocity projected onto the new constraint manifold. Runs once on successful dock. |

### 5.6 Summary — arm-kinematic coupling map

Grouped by which stage *depends on* arm kinematics, either
directly (Jacobian / joint angles consumed as input) or
indirectly (quantity derived from FK):

| Stage | Reads arm joint angles? | Reads arm Jacobians? | Frequency | Notes |
|---|---|---|---|---|
| CoarsePrePlanner NLP | **no** | **no** | per step | Centroidal-only (§3.2.1) |
| ContactScheduler | no | no | once | Anchors only |
| TorsoPlanner `add_phase` / `reference_at` | indirect: `p_t0, R_t0, p_t1, R_t1` come from FK on live/IK q | no | per step (program) / per sub-step (query) | Endpoints bake in one snapshot of arm-dependent FK |
| SwingPlanner | no | no | per sub-step | Anchors only |
| CoMToTorsoMapping | **yes** (`δ(q)`, `δ̇(q, q̇)`) | no | per sub-step when active | Bypassed in SS under T15 config |
| Centroidal NMPC solve | **no** | **no** | per NMPC tick | Only receives `r_com, v_com, L_com` and scheduler-derived contact positions |
| Reaction null-space task (QP) | via arm mass matrix slice (`rs.H[:6, sw_slice]`) | no (uses H slice, not J directly) | per sub-step | Gated by `alpha_reaction = 0.0` — **inactive in T15** |
| Posture task (QP) | **yes** (`q_nominal − q`) | no | per sub-step | `q_nominal` = `torso_map` output = manipulability-optimized IK from setup |
| EE task (QP) | via `J_ee` Jacobian | **yes** (swing-arm `J_tool_{a,b}`) | per sub-step | Fixed weight `α_ee = 3e3` |
| Contact task (QP) | via stance-arm `J_contacts` | **yes** (stance-arm `J_tool_*`) | per sub-step | Hard constraint |
| Impact projection (dock) | via `Jc_both` | **yes** | once per dock | `sim_loop.py:1366` |
| IK at SS setup | yes (q_end contains arms) | **yes** (for `w_product` check) | per step | Drives `p_t1, R_t1` |
| `manipulability_config` cache (setup) | yes | **yes** (`σ_min(J_a)·σ_min(J_b)`) | once at `setup()` | Caches per-anchor-pair manipulability-optimal q |

### 5.7 Where arm-kinematic information enters the *torso reference*

Tracing the torso-reference production path against §5.1–5.5:

- **IK at SS setup (5.2b, 5.2c):** `J_a, J_b` drive `(p_t1, R_t1)` endpoint choice. This is the **only place** in the live pipeline where any arm-Jacobian condition number influences the commanded torso trajectory. Per `sim_loop.py:851–862`, the influence is **binary**: if `w_product ≥ 1e−4` the fixed-rotation solution is accepted; otherwise the manipulability-optimized `torso_map` entry is used. Neither branch evaluates Jacobian conditioning along the trajectory interior.
- **CoMToTorsoMapping (5.4d):** `q_current` drives the `δ(q)` correction on linear torso reference. In SS under T15 config this is bypassed (`mapping_bypass_in_ss = True`). In DS it is active.
- **No other site feeds arm-kinematic information into the torso reference.** All other arm-kinematic reads (5.4f, 5.4g, 5.4h, 5.4i) feed the QP directly — affecting how the QP *resolves* the task-space command into joint torques, not what the torso reference is.

### 5.8 New data channels required to add interior manipulability-awareness

The TorsoPlanner (`torso_planner.py`) is currently told only
about the two endpoints and the time window. To make it interior-
aware, it would need:

- A Pinocchio model handle (to evaluate Jacobians at interior
  `q`) — **not currently present** in `TorsoPlanner.__init__`
  (`torso_planner.py:61`).
- A reference arm configuration at each interior time (since the
  quintic only defines torso pose, not arm joints). The
  `_planned_arm_config` quintic (`sim_loop.py:759`) *does*
  already provide this for the mapping — an analogous quintic
  could be used to evaluate `q(τ)` inside the TorsoPlanner,
  provided the Pinocchio handle.

Alternatively, the interior check could run inside
`_setup_torso_for_step` (which already has Pinocchio access via
`self.robot`) by sampling the quintic path offline at step setup
time — no new data channel needed, only additional computation
once per step.

*(§6 Swing/torso coupling follows.)*

---

## §6 Swing/torso coupling in the QP (Q6)

### 6.1 The QP tracks both as simultaneous, prioritized tasks

Per §3.6.1 (task list), the WholeBodyQP adds:

- **Priority 1:** Torso 6D tracking, weight `α_torso = 5e2` (SS)
  (`wholebody_qp.py:540–567`).
- **Priority 2:** EE 6D tracking (swing-arm), weight `α_ee = 3e3`
  (SS) (`wholebody_qp.py:594–647`).

Both are added to the same `HierarchicalQP` instance. The solver
is configured with `method='weighted'`, `weight_ratio=1.0`
(M2 stack, §3.6). In `weighted` mode, all tasks become rows of
one big least-squares QP; each task's weight matrix enters its
block of the cost. Task isolation is done **geometrically** via
null-space projection, not via weight-ratio priority.

### 6.2 Geometric coupling via null-space projection (M2 stack)

`wholebody_qp.py:579–583, :633–644`:

```
N_torso = I − A_torsoᵀ (A_torso A_torsoᵀ)⁻¹ A_torso   # ≈ I − A_torso⁺ A_torso
        = I − pinv(A_torso, rcond=1e-8) @ A_torso

# EE task projected into null(A_torso):
if (cfg.use_m2_stack or cfg.ee_null_space) and N_torso is not None:
    A_ee_proj   = A_ee @ N_torso
    b_ee_res    = b_ee − A_ee @ A_torso_pinv @ b_torso
    qp.add_task(A_ee_proj, b_ee_res, cfg.alpha_ee, priority=2)
else:
    qp.add_task(A_ee,      b_ee,     cfg.alpha_ee, priority=2)
```

T15 scripts set `use_m2_stack=True` (`scripts/run_m7_single_step.py:40`),
and `sim_loop.py:689` passes `ee_null_space=cfg.use_m2_stack`.
So in T15 the **EE task is projected into the null space of the
torso task**. The EE residual `b_ee_res` subtracts the particular
solution contribution `A_ee · A_torso_pinv · b_torso`.

Effect: the EE task can only use degrees of freedom that do not
move the torso. The solver therefore resolves a "best EE fit" in
`null(A_torso)` rather than trading EE tracking against torso
tracking — so long as `null(A_torso)` is rich enough.

Posture (`A_posture_proj = A_posture @ N_combo` with `N_combo =
I − A_combo⁺ A_combo` and `A_combo = [A_torso; A_ee]`,
`wholebody_qp.py:706–717`) and soft CoM are further projected
into `null([A_torso; A_ee])`.

### 6.3 Task-consistent feedforward (M7 v17)

Beyond the null-space projection, `wholebody_qp.py:617–621`
adds a feedforward coupling term to the EE task:

```
if torso_task_active and J_torso is not None:
    J_torso_pinv = np.linalg.pinv(J_torso, rcond=1e-8)
    a_ff_ee = a_ff_ee + J_ee @ J_torso_pinv @ a_torso_des
```

Rationale recorded in the code comment (`wholebody_qp.py:613–617`):
"When the torso accelerates at `a_torso_des`, the EE sees
`J_ee · J_torso⁺ · a_torso_des` of induced motion through the
shared floating base. Pre-add this to the EE feedforward so the
PD doesn't have to chase the coupling reactively."

So the EE task's desired acceleration is:

```
a_ee_des = a_ee_ff                                         # planner FF
         + J_ee · J_torso_pinv · a_torso_des               # torso-coupling FF
         + Kp_ee · (p_ee_ref − p_ee_actual)                # position PD
         + Kd_ee · (v_ee_ref − v_ee_actual)                # velocity PD
         + orientation PD terms                            # (Kp_ee_ang, Kd_ee_ang)
```

The shared floating-base coupling between torso and EE is
explicitly pre-accounted for.

### 6.4 What happens when the torso ref and EE ref are kinematically incompatible

The QP is a **soft-constrained weighted least-squares** — there
is no feasibility check, no branch for "incompatible references",
no abort. Whatever torso / EE / posture residuals remain after the
geometric and weighted blend are the ones the solver carries as
cost, and the QP returns a best-effort `q̈, τ, λ`.

Practical consequences:

- **In `null(A_torso)`**: the projected EE task `A_ee_proj` has
  `rank = rank(A_ee) − dim(image(A_torso) ∩ image(A_ee))`. If
  `A_ee` happens to be fully contained in `image(A_torso)`
  (e.g. at a singularity where the arm's reachable velocity
  space collapses onto the torso's), `A_ee_proj` becomes
  rank-deficient. The QP solution then produces zero EE
  correction in the direction `ŝ_min(J_ee) ∈ image(A_torso)`.

- **Weighted fallback**: if `null_space` projection is disabled
  (fallback branch at `wholebody_qp.py:644–645`), the torso and
  EE tasks trade off via their `α` weights. `α_torso = 5e2`
  vs `α_ee = 3e3` would favour EE 6× in the cost.

- **Posture task**: projected into `null([A_torso; A_ee])`. If
  the combined rank of `[A_torso; A_ee]` fills the configuration
  space, the posture task becomes inert.

There is no warning mechanism or metric logged that reports how
much of the EE residual is being absorbed geometrically vs cost-
tradeoff vs lost to rank-deficiency.

### 6.5 Swing ref vs torso ref during T15 SS

Swing reference per §2 (T15 bug1fix_vel):

- `p_ee_ref(t)` — SwingPlanner quintic from `anchors_b[k] →
  anchors_b[k+1]` (or `anchors_a[k] → anchors_a[k+1]`) over
  `[0, T_step · early_finish_fraction]`.
- `R_ee_ref(t)` — SLERP with delayed-cosine timing; starts at
  `R_release` (set via `set_swing_orientation` at
  `sim_loop.py:1214`) and rotates into anchor-frame identity.

Torso reference per §2 (T15 bug1fix_vel, SS, `mapping_bypass_in_ss=True`):

- `p_torso_ref(t)` — **constant** = `self._ss_entry_p_torso`.
- `R_torso_ref(t)` — TorsoPlanner quintic SLERP from `R_t0` to
  `R_t1`.

The QP is asked to simultaneously produce:

- `J_torso · q̈ ≈ a_torso_des` (linear → 0, angular → quintic
  interpolation)
- `J_ee · q̈ ≈ a_ee_des` (linear → large x-sweep per §2; angular
  → SLERP)

These references are **not coordinated by any planner** — the
SwingPlanner and TorsoPlanner share only `T_step` and
`early_finish_fraction` (plus the IK-derived endpoints that made
both achievable at their terminal time). The QP enforces the
coupling geometrically via null-space projection per §6.2.

### 6.6 Compatibility guarantee (absence thereof)

There is no stage in the pipeline that checks whether the
SwingPlanner's commanded EE trajectory is kinematically
compatible with the TorsoPlanner's commanded torso trajectory at
every interior instant — only at the endpoints (via the single
IK call at `_setup_torso_for_step`, §2.5). The guarantees are:

- **At SS entry (`τ=0`)**: both references start at the live
  state (TorsoPlanner.p_t0 = live torso; SwingPlanner p_start =
  anchors_b[swing_from_idx] which is where the swing arm is
  physically docked). Compatible by construction.
- **At SS end (`τ=1`)**: both terminate at an IK-consistent pose
  — the IK at `_setup_torso_for_step` solves for a `q_end` that
  simultaneously satisfies `oMf_tool_a = se3_a`, `oMf_tool_b =
  se3_b`, and a torso pose chosen by the IK mode. By construction
  the TorsoPlanner endpoint and the SwingPlanner endpoint live on
  the same `q_end` manifold.
- **At `τ ∈ (0, 1)`**: nothing enforces compatibility. The swing
  reference is a geometric template (quintic + bump + SLERP); the
  torso reference is a separate quintic. Whether `q̈` in the
  tangent space admits both depends on the arm Jacobian's
  conditioning along the path — which per T15-post-4 is where
  step 2 fails.

### 6.7 Summary

| Question | Answer (pointers) |
|---|---|
| Does the QP track both simultaneously? | **Yes** — both are tasks in the same `HierarchicalQP` (`wholebody_qp.py:566, :641`). |
| Priority / weighting scheme? | **Geometric priority via null-space projection** under M2 (active in T15). Torso: P1 unprojected. EE: P2 projected into `null(A_torso)`. Weight ratio is 1 under M2; actual isolation comes from the projector `N_torso`. |
| Feedforward coupling? | **Yes**: `a_ff_ee += J_ee · J_torso⁺ · a_torso_des` (`wholebody_qp.py:617–621`) pre-accounts for the shared floating-base-induced EE motion. |
| What if the references are incompatible? | **Best-effort blend**. No feasibility check, no abort, no warning. EE residual absorbed against `null(A_torso)` rank; what remains goes into the weighted cost. |
| Compatibility guarantee? | **Only at the endpoints** (τ=0 and τ=1, via the SS-setup IK). No guarantee for interior τ. |

*(§7 Multi-step horizon considerations follows.)*

---

## §7 Multi-step horizon considerations (Q7)

### 7.1 Per-stage horizon audit

| Stage | Horizon visible to the stage | Inputs specific to the *next* step? |
|---|---|---|
| ContactScheduler | **Entire traversal** — all phases pre-built at `plan_traversal(start_a, start_b, n_steps)` (`contact_scheduler.py:203–284`); subsequent phases' `anchor_a_idx, anchor_b_idx, swing_arm, swing_from_idx, swing_to_idx` are fixed at setup. | n/a (this *is* the full-horizon plan) |
| `torso_map` cache | All `(anchor_a_idx, anchor_b_idx)` pairs that appear in the plan (`sim_loop.py:210–222`) | No — pairs are pre-computed once, used by lookup per step |
| CoarsePrePlanner NLP | **Single-step window `[0, T_step]`** (`coarse_preplanner.py:416` signature takes `r_com_0, r_com_goal, r_C_stance, c_const, T_step`; all scalars/vectors refer to the current step only) | **No** — `r_com_goal` is the *current* step's terminal CoM (from IK at the current step's end pose). The stance contact point `r_C_stance` is the *current* stance. The conservation constant `c_const` is computed from the current live state at SS-setup. |
| IK at `_setup_torso_for_step` | Single step: picks `(p_t1, R_t1)` for the current step's end pose only. Inputs `stance_a, stance_b, target_idx` come from `phases[i+1]` (current upcoming SS), `sim_loop.py:1101–1110`. | **No** — the IK goal is derived from current `(end_a, end_b)` only. It does not look at step K+1's anchor pair. |
| TorsoPlanner | Single step: one phase added per SS setup at `_setup_torso_for_step` (`sim_loop.py:932–936`); `clear_phases()` is called before each `add_phase` to reset (`sim_loop.py:930`). | **No** — only current step's endpoints. |
| SwingPlanner | Nominally the whole plan (has a reference to `self.plan`), but evaluates only the phase returned by `plan.phase_at(t)` at the query time. Walks back to prior swing at `_last_swing_position` (`swing_planner.py:397–410`) for DOUBLE-phase fallback only. | **No** — it doesn't read step K+1. Forward lookup at t returns the current phase. |
| Centroidal NMPC | **1 s prediction horizon** (N=20 × dt=0.05s; `centroidal_nmpc.py:67–68`), but entirely within the current SS window. Parameters `r_ref, v_ref, r_C1, r_C2, c_simple, L_ref` (`centroidal_nmpc.py:192–196`) are drawn from the current step's TorsoPlanner queries and the current SS's stance. | **No** |
| CoMToTorsoMapping | Single-tick evaluation (`com_to_torso_mapping.py:133`). | No state carried beyond the current tick. |
| WholeBodyQP | Single-tick QP — no time horizon. | — |

### 7.2 Main loop structure

`sim_loop.py:1090–1400` is a per-phase `while i < len(phases):`
loop (`sim_loop.py:1090`). At each iteration it **only processes
one DS + one SS pair**:

```python
while i < len(phases):
    gp = phases[i]
    if gp.phase.value == 'double':
        if i + 1 < len(phases) and phases[i+1].phase.value != 'double':
            ss_gp = phases[i+1]         # ← only looks one phase ahead
            ss_phase_idx = i + 1
            swing_arm = ss_gp.swing_arm
            stance_a = ss_gp.anchor_a_idx
            stance_b = ss_gp.anchor_b_idx
            target_idx = ss_gp.swing_to_idx
            # ... _setup_torso_for_step, SS loop, optional dock, etc.
            step_idx += 1
            i += 2   # skip SS phase (already processed)
```

The loop **does not peek at `phases[i+2]` or beyond**. Each call
to `_setup_torso_for_step` and `_run_preplanner` is independent.
No warm-start, trajectory initial guess, or terminal cost is
carried across steps.

### 7.3 State carried across steps

Fields that persist between steps (not explicitly reset on
step boundary):

| Field | Persistence | Effect on next step |
|---|---|---|
| MuJoCo physical state (`mj_data.qpos, qvel`) | Persistent | Next step's IK seed via `q_init=pq_live` (`sim_loop.py:864`); next step's live `(v_com_0, L_com_0)` for the pre-planner (`sim_loop.py:995–996`) |
| `self.plan` (scheduler plan with installed durations) | Persistent | `plan.t_start/t_end` for all phases; current-step's `T_step` is installed before its own SS runs |
| `self.torso_map` | Persistent | Cache of manipulability-optimized IK configs |
| `self.qp_ss` nominal posture | Updated per-step (`sim_loop.py:1192`: `qp_ss.set_nominal_posture(q_dock[joints_q_slice])`) | Posture task uses the *current step's* `q_dock` as nominal — not step K+1's |
| `hw_current` (wheel momentum) | Persistent physical | Feeds `c_const` of the next pre-planner call and `hw_current` of the next NMPC solve |
| `self._last_x_opt, _w0_prev` (NMPC warm start) | Persistent within a step; reset at step boundary via `nmpc.reset_warm_start()` (`sim_loop.py:1206`) | No cross-step warm start for the NMPC |

### 7.4 Is the pre-planner for step K aware of step K+1?

**No.** The CoarsePrePlanner's `solve(r_com_0, v_com_0, L_com_0,
r_com_goal, r_C_stance, c_const, T_step)` signature
(`coarse_preplanner.py:416`) takes only current-step quantities:

- `r_com_goal` = current step's end-CoM (from current step's IK).
- `r_C_stance` = current step's stance contact (single point).
- `c_const` = current step's conservation constant from live state.
- `T_step` = current step's duration guess, optimized by the NLP
  within the single-step problem.

There is no parameter for "the next step will require CoM at
`r_com_goal_K+1` with stance `r_C_K+1`", no terminal cost that
penalises configurations unfavourable for the next step, no
multi-step rollout, and no receding-horizon logic beyond the
single step.

### 7.5 Is any stage multi-step lookahead-aware?

- **No stage runs a multi-step optimization.** The whole planning
  pipeline is single-step-local.
- **The scheduler is full-horizon** — it has the entire gait plan
  from setup — but it does not solve anything; it just stores the
  anchor sequence.
- **The `torso_map` cache** is pre-computed for every anchor pair
  in the plan. It carries the outcome of per-pair
  `manipulability_config` optimizations (each itself single-pair,
  no cross-pair coupling). Used as a lookup when step K happens
  to land at a cached pair.
- **Centroidal NMPC** has a 1-second internal prediction horizon
  (N=20 × dt=0.05 s), *but* `T_step` in the T15 run is 7.3 – 9.4 s
  (§2 of T15_report.md). The NMPC horizon therefore spans a small
  fraction of the SS window and never reaches the next SS. Its
  reference `r_com_ref` is evaluated at `t + N·dt_nmpc`
  (`sim_loop.py:1521`, `t + 8·0.1 = t + 0.8 s`) — still within
  the current SS.

### 7.6 Consequence for manipulability-aware extensions

Facts the prompt §7 asked about:

- **Per-step-local is admissible.** Since the pipeline is already
  single-step-local at every stage, a manipulability-aware
  extension can be scoped to a single step without introducing
  multi-step coherence requirements.
- **No multi-step lookahead to keep consistent with.** There is no
  existing receding-horizon structure that a new term would have
  to fit into.
- **The `torso_map` cache is the only full-plan structure.** It
  is pre-computed once per `setup()` and caches per-pair manipulability-
  optimized configs. Any step-interior manipulability work that
  wanted to piggyback on it would need to extend it from
  "single-pose cache" to something richer (e.g. a per-pair
  trajectory cache).

*(§8 Summary follows.)*

---

## §8 Factual summary

### 8.1 Where the torso reference is decided

The torso reference handed to the whole-body QP is produced by
**three stages operating at three different cadences**, each
contributing to a subset of the four QP-input fields
(`p_torso_ref, R_torso_ref, v_torso_ref, a_torso_ff`):

| Stage | Runs at | Produces | Which QP fields |
|---|---|---|---|
| IK inside `_setup_torso_for_step` (`sim_loop.py:794–951`) | **Once per step**, at SS entry | `(p_t0, R_t0)` = live torso; `(p_t1, R_t1)` = IK-derived end pose; `T_step` = pre-planner output | Endpoints of the TorsoPlanner quintic → *all four fields over the SS window* |
| TorsoPlanner (`torso_planner.py:155–210`) | Programmed per step, **queried per QP sub-step (100 Hz)** | `tr = TorsoReference(p, R, v, a)` via piecewise quintic between endpoints | `R_torso_ref` always; `v_torso_ref, a_torso_ff` angular always; `p_torso_ref, v_torso_ref, a_torso_ff` linear *in Branch C only* |
| CoMToTorsoMapping (`com_to_torso_mapping.py:133–170`) | Per QP sub-step (100 Hz) **when active** (DS, or SS with `mapping_bypass_in_ss=False`) | `(r_b_ref, v_b_ref, a_b_ff)` = mass-weighted map from NMPC CoM output + live `δ(q)` | `p_torso_ref, v_torso_ref, a_torso_ff` linear (Branch B) |

T15 bug1fix_vel runs Branch A during SS (`mapping_bypass_in_ss =
True`): the linear torso reference is frozen at
`self._ss_entry_p_torso` for the whole SS; only the angular
reference follows the TorsoPlanner quintic. No NMPC output
reaches the torso reference in SS under this configuration.

### 8.2 The decision point

**All torso-reference trajectory content is baked at
`_setup_torso_for_step` time.** After the call returns, the
reference is fully prescribed for the entire SS window:

- Linear position: constant (Branch A) or mapping-driven (Branch
  B), with both endpoints (start = live torso, end = IK output)
  fixed at setup.
- Angular: piecewise quintic between `R_t0` and `R_t1`, both
  fixed at setup.
- `T_step`: fixed at setup by the CoarsePrePlanner.

Per-tick queries just evaluate these pre-decided curves at the
current time. There is no planner loop that adjusts the torso
reference during the SS.

### 8.3 Where manipulability of the arm Jacobian currently enters

Two sites in the running pipeline, both binary or endpoint-only:

1. **`dock_configuration_fixed_rotation`** (`ik.py:168–270`,
   called from `sim_loop.py:847–856`). After convergence it
   returns `w_product = w_a · w_b` (Yoshikawa manipulability of
   the two arms at `q_end`). `sim_loop.py:854` uses it as a
   **binary threshold** (`w_product ≥ 1e−4`, `config.py:170`) to
   choose between fixed-rotation IK and manipulability-optimized
   IK.

2. **`manipulability_config`** (`ik.py:274–378`, called from
   `sim_loop.py:220` at setup and as fallback from
   `sim_loop.py:860`). Optimizes torso xyz to maximize
   `σ_min(J_a) · σ_min(J_b)` at a single `q_end` pose.

Both are **endpoint-only** (evaluate at `q_end`, not along the
trajectory interior). Both feed into `(p_t1, R_t1)` — the
TorsoPlanner's end waypoint.

No stage evaluates arm-Jacobian conditioning at interior times
`τ ∈ (0, 1)`.

### 8.4 Stages that would be touched to add interior
manipulability-awareness

For each of the candidate insertion surfaces identified in §4.6
and §5.8, the stage, cadence, existing inputs, and what would
need to be added to the stage's input channels. No fix design;
this is an enumeration of architectural touch-points.

#### Candidate 1 — Endpoint IK already evaluates manipulability; extend to trajectory

- **Stage**: `_setup_torso_for_step` (`sim_loop.py:794–951`).
  Already holds `self.robot` (Pinocchio model), `q_end`,
  `pq_live`, and the anchor SE3s.
- **Existing inputs**: current `(stance_a, stance_b, target_idx)`,
  live q, cached `torso_map`.
- **New data required**: none beyond what is already in scope.
  The straight-line torso path and the arm quintic path are
  implicit in the endpoints and `T_step`.
- **Cadence**: once per step. No per-tick cost change.

#### Candidate 2 — TorsoPlanner shape optimization

- **Stage**: `TorsoPlanner.add_phase` (`torso_planner.py:155`).
  Currently a dataclass append; no optimization.
- **Existing inputs**: endpoints and time window.
- **New data required**: Pinocchio model handle (not currently
  passed to TorsoPlanner; would need to extend `__init__`), and
  a per-interior-time arm configuration reference (currently
  available as `_step_q_start, _step_q_end` via
  `_planned_arm_config`, but not provided to the planner).
- **Cadence**: once per step at programming time.

#### Candidate 3 — CoarsePrePlanner cost augmentation

- **Stage**: `CoarsePrePlanner.solve` (`coarse_preplanner.py:416`),
  NLP via CasADi + IPOPT.
- **Existing inputs**: centroidal state, stance contact, `c_const`,
  `T_step`, `h_max`.
- **New data required**: arm-joint angles over the step (the NLP
  currently does not carry `q` in its state). Adding a manipulability
  proxy would require either (a) feeding arm-q as a CasADi
  parameter evaluated at fixed interior times, or (b) expanding
  the NLP state to include arm kinematics — the latter is
  structurally invasive.
- **Cadence**: once per step.

#### Candidate 4 — Centroidal NMPC cost augmentation

- **Stage**: `CentroidalNMPC` (`centroidal_nmpc.py:106–420`), NLP
  via CasADi.
- **Existing inputs**: `x = [r_com, v_com, L_com] ∈ ℝ⁹`,
  `u = [f1, τ1, f2, τ2] ∈ ℝ¹²`, parameters `p = [r_ref, v_ref,
  r_C1, r_C2, c_simple, L_ref] ∈ ℝ¹⁸`.
- **New data required**: arm-joint angles in the parameter vector
  (expand `p` to include a `q_arm` block at each knot; requires
  rebuilding the NLP with a larger `np_total`). The NMPC's
  dynamics model would also need to know how arm motion affects
  `L_com` — not currently modelled (arm joints are hidden inside
  the `c_simple` correction).
- **Cadence**: per NMPC tick (10 Hz).

#### Candidate 5 — CoMToTorsoMapping retargeting

- **Stage**: `CoMToTorsoMapping.compute` (`com_to_torso_mapping.py:133`).
  Already reads `q_current`.
- **Existing inputs**: `r_com_ref, v_com_ref, a_com_ff, q_current,
  dq_current`.
- **New data required**: none — all arm state already available.
  Extension would mean modifying the closed-form map to include a
  manipulability-gradient correction. Note: currently bypassed in
  SS under T15 config.
- **Cadence**: per QP sub-step (100 Hz) when active.

#### Candidate 6 — WholeBodyQP adaptive weighting or explicit task

- **Stage**: `WholeBodyQP.solve` (`wholebody_qp.py:195`). Already
  reads `J_ee`, `q`, `dq`.
- **Existing inputs**: all Jacobians, q, dq, references, task
  weights from config.
- **New data required**: none for making `α_ee` a runtime
  function of `σ_min(J_ee)`. Adding an explicit manipulability-
  gradient task would require exposing the gradient of
  `σ_min(J_ee)` w.r.t. q (available via Pinocchio's frame-Jacobian
  derivative API but not currently computed).
- **Cadence**: per QP sub-step (100 Hz). Reactive, not predictive.

### 8.5 Map of single-step-locality (relevant to where a fix plugs in)

Per §7, every planner stage is single-step-local:

- Scheduler has the full plan but doesn't optimize it.
- Torso_map cache is pre-populated once per setup but holds only
  single-pose configs.
- Every other planner stage resets per step (`TorsoPlanner.clear_phases`,
  pre-planner `solve`, NMPC `reset_warm_start`).

Any manipulability-aware extension at any of the 6 candidate
stages can therefore be scoped per-step without changing
multi-step-horizon semantics (because the semantics is already
"no multi-step horizon").

### 8.6 Pointers index

| Item | File:line |
|---|---|
| Main SS setup | `crawlbot/simulation/sim_loop.py:794` |
| IK branch selector (`w_product` binary) | `sim_loop.py:845–862`, threshold at `config.py:170` |
| TorsoPlanner programming | `sim_loop.py:930–937` → `torso_planner.py:155` |
| Torso reference per-tick assembly (three branches) | `sim_loop.py:1734–1777` |
| Per-SS-entry Option Z offset reset | `sim_loop.py:919` |
| CoMToTorsoMapping | `crawlbot/core/com_to_torso_mapping.py:133–170` |
| CoarsePrePlanner NLP | `crawlbot/planning/coarse_preplanner.py:416` |
| Centroidal NMPC NLP build | `crawlbot/solvers/centroidal_nmpc.py:133–370` |
| Centroidal NMPC solve call site | `sim_loop.py:1615` |
| WholeBodyQP solve | `crawlbot/solvers/wholebody_qp.py:195` |
| Torso task (P1) | `wholebody_qp.py:540–567` |
| EE task (P2) with null-space projection | `wholebody_qp.py:594–644` |
| Task-consistent feedforward | `wholebody_qp.py:617–621` |
| `manipulability_config` | `crawlbot/core/ik.py:274–378` |
| `dock_configuration_fixed_rotation` | `crawlbot/core/ik.py:168–270` |
| `torso_map` precomputation | `sim_loop.py:210–222` |
| `_planned_arm_config` quintic | `sim_loop.py:759–792` |

---

*End of T15-post-5 pipeline audit.*
