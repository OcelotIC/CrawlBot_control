> **⚠ SUPERSEDED (2026-05-27).** This file is stale — it predates or does not
> track the reworked controller. For what the code actually does,
> `docs/architecture/STACK_OVERVIEW.md` is the code-ground-truth reference and
> supersedes any current-state claim here (e.g. the NMPC is 9-state
> `[r_com,v_com,L_com]`, not 12; module APIs and parameters have changed).

# STATUS — Session Handoff

**Last updated:** 2026-04-15
**Branch:** `claude/m7-core-task-replace-3Hmf4`
**Head commit:** `90d7b13` (QP tracking test: realistic swing geometry 800 mm / 45° / 7.3 s)
**Active milestone:** M7 — two-phase state machine wired; dock still failing; QP tracking defect isolated (α_wrench fix) but closed-loop regression not yet explained

This document is a handoff for the next Claude Code session. It is not a
status report for Idriss. Read the two authoritative documents first:

1. `docs/architecture/brainstorming_reworked_architecture.md` — spec
2. `docs/architecture/CLAUDE_CODE_HANDOFF.md` — implementation plan

Then read this file.

---

## 1. Milestones completed

| M    | Description                                                                     | Commit(s)                          |
|------|---------------------------------------------------------------------------------|------------------------------------|
| M-1  | Codebase cleanup (shim removal, canonical imports)                              | `1f76c11`                          |
| M-1.5| Remove dead methods                                                             | `4cdd2cb`                          |
| M0   | Diagnostic suite (metrics, plots, invariants)                                   | `1d35b8c`, `6b37a58`, `1d917ef`    |
| M1   | CoM-to-torso mapping layer: `r_b_ref = (m_total/m_b)·r_com_ref − δ(q)/m_b`      | `103bcc6`                          |
| M2   | Reworked whole-body QP task stack (P1 torso/EE → P3 posture → P4 soft-CoM)      | `a339c19`                          |
| M3   | Centroidal NMPC with B2 Option B conservation-law box                           | `ce380b0`                          |
| M4   | AOCS feedforward + wiring (`τ_w = −L̇_est − orbital + K_hw·hw_err`)              | `c126405`, `8680d47`               |
| M4+  | **AOCS desat sign fix** (was destabilizing RWAs)                                | `80028ac`, `0049503` (diag)        |
| M5   | TorsoPlanner `L_com_ref` + SwingPlanner + full wiring                           | `4f8a7ff` → `9bd003c`              |
| M5+  | **weight_ratio=1 + null-space projection hierarchy**                            | `dd892bc`, `34893b0`/`4d52fd6`     |
| M6   | Coarse pre-planner (CasADi/IPOPT centroidal NLP, M=15 knots, momentum box)      | `078fff5`                          |
| M6+  | Gain bumps → first successful dock at 1% mass ratio                             | `506674f`, `0971a6e` (diag)        |
| M6+  | **Inter-step DS settle: energy-based passivity** (replaces timer)               | `208a155`                          |
| M6+  | **Dock gate: d<5mm AND ori<5°** (both gates, not position-only)                 | `7797211`                          |
| M6+  | **7-DOF arms: SEW elbow-swivel joint per spec §4.9.6**                          | `a9a36f5`                          |

### What the 7-DOF upgrade changed

- **Per-arm joint order** (Pinocchio chain): `J1, J2, SW, J3, J4, J5, J6`
  - `SW` = elbow-swivel, axis along shoulder→elbow direction, between
    `Link_2` (upper arm) and `Link_3` (elbow), MIRROR-style (Mishra et al., 2022).
- **Pinocchio model:** `nq=21, nv=20, n_joints=14` (was 19/18/12)
- **MuJoCo model:** `nq=31, nv=29, nu=17` (was 29/27/15)
- **Model files touched:** URDF + all four MJCF variants (1%, 8%, hw50, hw100)
- **Code files touched:** `robot_interface.py`, `state_conversions.py`,
  `config.py` (SimConfig n_joints), and `tests/test_invariants.py`
  (replaced hardcoded `np.zeros(18)` with `robot_interface.model.nv`).
- **Actuator order in MJCF** was reordered to match Pinocchio joint order
  exactly — this is important for torque mapping. Do not assume the arm
  block is contiguous in MuJoCo's default actuator layout.

### What the dock-gate fix proved

Spec §2 predicts that 6 DOF is insufficient for orientation-controlled
docking. After enabling the `d<5mm AND ori<5°` gate:

- **6-DOF + gate ON:** 0 / 162 samples satisfied both thresholds.
  Best ori in close-approach band (d<20mm, 9 samples): **14.44° – 56.80°**.
- **7-DOF + gate ON:** 20 / 162 samples with ori<5°.
  But min distance at those samples: **41.60 mm @ ori=4.22°, t=8.40s**.
  The 7-DOF arm *can* orient, but the torso quintic drags the base
  during approach and the swing EE never closes to <5 mm.

**Conclusion:** 6-DOF was the kinematic bottleneck for orientation.
7-DOF is kinematically sufficient. What remains is a **trajectory
coupling** problem between torso and swing, which motivates the
state-machine rework below.

---

## 2. Test suite

**Total: 191 tests passing** across 15 files (run with
`PYTHONPATH=. MUJOCO_GL=osmesa python3 -m pytest tests/ -v`).

| File                          | Tests |
|-------------------------------|-------|
| test_mapping_layer.py         | 30    |
| test_liabilities.py           | 20    |
| test_coarse_preplanner.py     | 19    |
| test_planners_6d.py           | 16    |
| test_invariants.py            | 16    |
| test_momentum.py              | 14    |
| test_frame_conversions.py     | 11    |
| test_contact_estimator.py     | 11    |
| test_contact_dynamics.py      | 11    |
| test_aocs_physics.py          | 11    |
| test_nmpc_qp_consistency.py   | 10    |
| test_diagnostics.py           | 10    |
| test_aocs_orbital.py          | 5     |
| test_reworked_qp.py           | 4     |
| test_nmpc_conservation.py     | 3     |

All 191 pass on `a9a36f5`. Run the suite at session start per `CLAUDE.md`.

---

## 3. Critical decision — the M7 rework

The SS → EXT → DS three-phase state machine is **being eliminated**.
The spec (§6) defines a two-phase machine — only `SS` (single support,
one arm swinging) and `DS` (double support, both welded). The EXT phase
is a legacy from the previous architecture and is not present in the
spec.

### Why eliminate EXT

1. **Not in the spec.** Spec §6 is explicit: two phases, DS ↔ SS.
2. **EXT is the reason step 0 can't close.** The 7-DOF single-step run
   showed the torso quintic starting at t=1.7s and running to t=16.5s
   (SS_duration baked in), while the swing arm reaches minimum distance
   at t=8.4s when torso is still mid-quintic and pulling the base away
   from the target. There is no mechanism for the two trajectories to
   agree on "finish together."
3. **Gain scheduling is a workaround for phase boundaries** — with no
   phase boundary, no gain scheduling is needed. Spec §6 already wants
   distance-dependent EE gains removed; the user's direction is further:
   no schedule at all.

### What the two-phase machine looks like

```
DS (settle)  ────┐
                 │  coarse pre-planner fires once per step
                 │  → outputs T_step, torso spline, swing spline,
                 │     momentum/hw plan, contact plan
                 ▼
SS (single support)
  t ∈ [0, T_step]
  torso_ref(t)  = quintic with zero-velocity endpoints
  swing_ref(t)  = quintic with zero-velocity endpoints
  BOTH terminate at t = T_step simultaneously
  dock check at every QP step: d<5mm AND ori<5° → weld → DS
  timeout: t > T_step + margin → abort step
                 │
                 ▼
DS (settle)      energy-based passivity exit (already implemented,
                 see _run_ds_passivity_loop in sim_loop.py)
  exit when T_kin < T_settle = 0.5·ε_v²·λ_min(H)
                 │
                 ▼
next step ───────┘
```

### Synchronized trajectories — key constraint

The coarse pre-planner outputs a single `T_step` per step. Both the
torso and swing trajectories are planned over `[0, T_step]`:

- `p_b_ref(0) = p_b_start, p_b_ref(T_step) = p_b_end, v_b_ref(0)=v_b_ref(T_step)=0`
- `p_ee_ref(0) = p_ee_start, p_ee_ref(T_step) = p_anchor, v_ee_ref(0)=v_ee_ref(T_step)=0`
- Quintic (or higher-order) in both, with zero-velocity boundary
  conditions on **both endpoints** of **both** trajectories.

This means at `t = T_step`, if the controller has tracked well, the
EE has reached the anchor with zero velocity, while the torso has
reached its planned end pose also with zero velocity. Dock is then
a natural terminal condition, not a race.

### No gain scheduling, no EXT phase

- One gain set for SS. One gain set for DS. No blending, no ramps,
  no distance-dependent modulation.
- The only mode switching is the SS → DS transition on dock
  (or on timeout abort).

---

## 4. Remaining M7 work

Four concrete tasks, in order:

### Task 1 — Rewire `sim_loop.py` to two-phase state machine

File: `crawlbot/simulation/sim_loop.py`

Current structure (approximate):
```
per-step loop:
  phase = SS
  while not (docked or timeout):
    if dist < ext_thresh: phase = EXT
    run QP
  if phase in {SS, EXT}: dock
  inter-step energy settle (phase = DS)
```

Target structure:
```
per-step loop:
  # one coarse pre-plan call → T_step, torso_spline, swing_spline
  phase = SS
  t_local = 0
  while t_local < T_step + t_timeout_margin:
    torso_ref = torso_spline(t_local)
    swing_ref = swing_spline(t_local)
    run QP
    if docked: break
    t_local += dt_qp
  if not docked: abort / log and stop traversal
  phase = DS
  _run_ds_passivity_loop(post_dock_contact_config, ...)
```

- Delete all `EXT` branches, `ext_thresh` config, distance-dependent
  gain ramps.
- Delete `ss_duration`-like config; `T_step` comes from the coarse
  pre-planner.
- Keep `_t_plan_offset` bookkeeping for multi-step traversals (the
  swing_planner offset fix from commit `208a155` — do not regress it).
- Keep `_run_ds_passivity_loop` as is — it already takes a
  `contact_config` parameter and works for both setup-stage and
  inter-step settling.

### Task 2 — Synchronized trajectory planning

Files:
- `crawlbot/planners/torso_planner.py`
- `crawlbot/planners/swing_planner.py`

Both planners must be rewritten (or extended) to accept `T_step` and
produce a quintic with zero-velocity boundary conditions at both
endpoints. The torso planner already uses a quintic spline; the key
change is:

- Both planners use the **same** `T_step` (comes from coarse
  pre-planner output).
- Both planners parameterize over `t ∈ [0, T_step]` (not absolute
  sim time).
- Both planners return `v_ref(T_step) = 0` *by construction*.

The coarse pre-planner already estimates `T_step` — wire it through.
Check `crawlbot/planners/coarse_preplanner.py` for the output field
name (may be `T_step`, `step_duration`, or similar — verify, don't
guess).

### Task 3 — Inter-step DS settling with passivity constraint

Already implemented via `_run_ds_passivity_loop`. Verify it still
works after the state-machine rewire:

- DS entry: `passivity_active=True`, reset NMPC warm start.
- Energy monitor: `T_kin < T_settle = 0.5·ε_v²·λ_min(H)` → exit.
- Max step budget and plateau-detection safeguards remain.
- Logged in `SimLog.inter_step_settles` per-transition telemetry.

### Task 4 — Validation

Four runs, in order. Each must produce diagnostics (rule 3).

1. **1% single-step** — first target is a clean dock within `T_step`
   (no timeout). Metric: `ee_pos_err_at_dock < 5 mm`,
   `ee_ori_err_at_dock < 5°`, `|Δr_com| < 2 cm`.
2. **14% single-step** — scales up the mass ratio. Same metrics.
3. **1% 3-step traversal** — three successive steps, no abort, energy
   settle between each. Metric: each step meets 1%-single-step
   criteria.
4. **14% 3-step traversal** — final validation for M7 completion.

All four runs must dump logs to `results/` and run `run_diagnostics()`.

---

## 5. Files to know

- `crawlbot/simulation/sim_loop.py` — main loop, to be rewired
- `crawlbot/simulation/config.py` — SimConfig, remove EXT-related fields
- `crawlbot/planners/torso_planner.py` — quintic torso trajectory
- `crawlbot/planners/swing_planner.py` — quintic swing trajectory
- `crawlbot/planners/coarse_preplanner.py` — centroidal NLP, outputs T_step
- `crawlbot/solvers/wholebody_qp.py` — P1/P3/P4 task stack, null-space projection
- `crawlbot/solvers/nmpc.py` — centroidal NMPC with conservation-law box
- `crawlbot/diagnostics/plots.py` — diagnostic plot generation
- `crawlbot/diagnostics/metrics.py` — metric extraction
- `tests/test_invariants.py` — physics invariants, nv-generic after 7-DOF
- `models/VISPA_crawling_fixed.urdf` — Pinocchio URDF with 7-DOF arms
- `models/VISPA_crawling_rwa3*.xml` — MuJoCo MJCFs with 7-DOF arms
- `docs/architecture/brainstorming_reworked_architecture.md` — spec
- `docs/architecture/CLAUDE_CODE_HANDOFF.md` — impl plan

---

## 6. Anti-patterns to avoid (from CLAUDE.md + HANDOFF §0)

1. Do not reintroduce EXT. If you find yourself adding an `ext_thresh`,
   stop and re-read §3 of this file.
2. Do not gain-schedule by distance. One gain set per phase.
3. Do not silently change SimConfig parameters. All changes need
   justification with units.
4. Do not skip diagnostics. Every sim ends with `run_diagnostics()`.
5. Do not proceed past a failing metric by rationalizing it. Fix it
   or document that it's known and why.
6. Do not edit model files by copy-paste. One URDF, one MJCF per mass
   ratio. Parametric variations via Python, not new files.

---

## 7. 2026-04-15 session — QP tracking isolation (in progress)

Goal this session: find why 7-DOF + two-phase + dock-gate still times out at
close approach (~20–38 mm from the 5 mm weld gate).

### Cascade of closed-loop runs

| Tag | Key change from prior | Result (best `d_swing`, notes) |
|-----|------------------------|--------------------------------|
| v7  | Baseline two-phase, stage-1 damping on, bump_peak=0.25, torso stagger=0.7 | dock fails; κ(N_t·J_ee)=1e5 elbow-unfold singularity |
| v8  | Manipulability-optimized qpos init (torso_map entry instead of `dock_configuration`) | reach ≥ 650 mm (fixed fold); κ(N_t·J_ee) < 7 in SS |
| v9  | Reverted trajectory shaping: `swing_bump_peak_tau=0.5`, `torso_early_finish_fraction=1.0` | torso ori error 32°, EE peak far from anchor; QP delivering 14.5°/s² when PD wants 102°/s² (7× attenuation) |
| v10 | `ss_alpha_wrench=0.01`, `alpha_com_soft=0.0` in `SimConfig` | SS tracking great; **closed-loop breaks** — NMPC infeasibility 11%, hw saturates, post-SS DS explodes |
| v11 | Added `hw_qp_tight = ±3 Nms` (tight QP momentum bound; NMPC still sees ±5) | bound never activates (\|hw\| < 3); same failure mode as v10 |

All logs + physics traces in `results/M7_1pct_1step_v{4..11}/`.

### Standalone QP-only diagnosis (no NMPC, no AOCS, no planners)

Script: `scripts/test_qp_tracking.py`. Setup: manipulability-optimized DS
init, release arm `b`, drive EE with septic 6D reference (C³-continuous,
zero vel/acc/jerk at both endpoints). Override QP weights after `setup()`:
`sim.qp_ss.config.alpha_wrench = 0.01`, `sim.qp_ss.config.alpha_com_soft = 0`.

- **Small swing (200 mm / 15° / 8 s)** with default `α_wrench=100`:
  all 4 metrics fail; pre-vs-post torso-accel ratio ≈ 0.14 (7× attenuation).
- **Same swing with `α_wrench=0.01`**:
  all 4 metrics pass; ratio → ~1.0; torque commands 5–7 Nm (vs 0.22 Nm before).
- **Realistic swing (800 mm / 45° / 7.3 s, matching closed-loop step geometry)**
  with `α_wrench=0.01, α_com_soft=0`:
  **EE position peak 15.6 mm**, torso/ori/ee-ori metrics clean. Standalone QP
  can track the real step geometry with ~16 mm peak EE error.

### The unexplained delta

- Standalone QP with full-geometry septic trajectory: **EE peak ≈ 16 mm.**
- Closed-loop v10/v11 with the same α values and similar-shape reference:
  **EE peak ≈ 193 mm** (12× worse), NMPC infeasible, hw saturated, dock fails.

The QP is not intrinsically unable to track this trajectory. Something in
the cascade (coarse pre-planner / NMPC / CoM→torso mapping / AOCS) is
injecting a disturbance that the QP then fights.

### Next diagnostic (planned, not yet run)

Run the full cascade **minus** the coarse pre-planner: feed `T_step=7.3 s`
directly, keep the same septic swing reference, but enable NMPC + CoM→torso
mapping + AOCS. Expected outcomes:

- If EE peak stays ≈ 16 mm → the coarse pre-planner's `T_step` or
  reference shape is the disturbance source. Narrow to pre-planner.
- If EE peak jumps to ≈ 190 mm → one of {NMPC, mapping, AOCS} is the
  source. Next cut: bisect NMPC vs mapping vs AOCS individually.

Script stub to be created: `scripts/test_cascade_no_preplanner.py`. Use
`SimulationLoop.setup()` for initialization, then a manual 100 Hz loop
that calls `sim.nmpc.solve(...)` every 10 ticks (constant `r_com_ref`,
`L_com_ref=0`), `sim.mapping.compute(...)` for torso position reference
(orientation held at `R_torso_0`), septic 800/45/7.3 for EE reference,
QP solve with overrides, `compute_aocs_command_legacy_corrected(...)`
for `τ_w`. Release weld `('b', 2)` before entering the loop. Compare
EE peak error against standalone (15.6 mm) and closed-loop v11 (193 mm).

APIs confirmed this session:
- `NMPC.solve(r_com, v_com, L_com, r_com_ref, v_com_ref, contact_config,
  warm_start, hw_current, L_com_ref) -> (rp, vp, _, lr, info)` then
  `nmpc.compute_feedforward_acceleration(lr)` (see `sim_loop.py:1462`).
- `CoMToTorsoMapping.compute(r_com_ref, v_com_ref, a_com_ff, q_current,
  dq_current) -> (r_b_ref, v_b_ref, a_b_ff, delta)` (see
  `crawlbot/core/com_to_torso_mapping.py:109`).
- `compute_aocs_command_legacy_corrected(L_com, L_com_prev, r_com, v_com,
  v_com_prev, hw_current, dt, robot_mass, K_hw, hw_min, hw_max, tau_w_max)`
  (see `crawlbot/aocs/force_estimator.py:286`).

### SimConfig state at end of session

Current values (committed, in effect for any re-run):

- `n_settle_damping_steps = 0` — stage-1 damping skipped (manipulability
  init places arms near weld equilibrium, no impulse to absorb).
- `swing_bump_peak_tau = 0.5` — symmetric sin²(πτ) bump (v9 revert).
- `torso_early_finish_fraction = 1.0` — torso runs full [0, T_step] (v9 revert).
- `alpha_com_soft = 0.0` — soft CoM residual disabled (v10).
- `ss_alpha_wrench = 0.01` — pure regularization (v10 fix, justified in config.py).
- `hw_qp_tight = np.full(3, 3.0)` — tight QP momentum bound (v11, inactive in practice).
- `ss_Kp_torso = 6.0, ss_Kd_torso = 5.0, ss_Kp_ee = 10.0, ss_Kp_ee_ang = 6.0`
  (v8 post-init values).

### Rule reminder

Do not rationalize the 193 mm EE peak by blaming trajectory speed before
the cascade bisection runs. The QP can track this geometry with 16 mm peak
error *in isolation* — the disturbance source is inside the
NMPC/mapping/AOCS/pre-planner chain and will be isolated by the next test.
