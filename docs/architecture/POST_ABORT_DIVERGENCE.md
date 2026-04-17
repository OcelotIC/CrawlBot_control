# POST_ABORT_DIVERGENCE — three hypotheses for the trailing-DS torso drift

**Scope.** Step 3 of the M7 per-phase metrics audit. This document is
**research only**: no source files outside `crawlbot/diagnostics/` were
modified, no simulations were re-run, no fixes are proposed. For each
hypothesis the output is: (1) file + line numbers consulted, (2)
conclusion against the v21 observation, (3) the minimal experimental test
that would settle it.

## Observation under scrutiny

Archive re-diagnostic (`results/archive_rediagnostic.md`) on M7 v21:

| field | value |
|---|---|
| SS window | `[0.11, 11.31] s` (113 samples) |
| `torso_ori_peak_deg_SS` | **0.53°** |
| `ss_end_torso_ori_deg` | 0.20° |
| `q_torso_ref_ss_to_ds_jump_deg` | **3.42°** |
| `ds_entry_torso_ori_deg` | 3.33° |
| DS window | `[11.41, 31.31] s` (200 samples) |
| `torso_ori_peak_deg_DS` | **45.47°** (drifts monotonically over 20 s) |
| `tau_peak_Nm_DS` | **20.00 Nm** (saturated; 24 % of DS ticks reported externally) |
| `tau_w_peak_ratio_DS` | **1.00** (wheel torque saturated; 35 % of ticks externally) |
| `aborted_steps[0]` | `reason='dock_timeout', d_mm=40.84, ori_deg=6.97` |

The reference `q_torso_ref` is held strictly constant for t ∈ [11.41, 31.31]
(quaternion path length 0.000, per the external analysis) yet the actual
torso tumbles ~38° over 20 s while both torque channels saturate.
Across the four archived versions the SS peak stays in
`[0.53°, 0.54°]` and the DS peak ranges `45° → 180°` — the "29–45°"
headline was a DS-window artefact on every version.

---

## H_DS1 — contact-config mismatch (QP believes DOUBLE, MuJoCo has one weld)

**Code consulted.**

- `crawlbot/simulation/sim_loop.py:1282-1298` — the `if not docked:` branch
  on dock timeout only logs `aborted_steps` and prints; it does **not**
  change `docked` to something that downstream code reads, and it does
  not deactivate any weld-related state.
- `crawlbot/simulation/sim_loop.py:1300-1331` — the entire
  `if docked:` block (weld activation at 1302, `mj_forward` at 1303,
  NMPC warm-start reset at 1304, inelastic-impact projection at
  1306-1331) is **skipped** on timeout. MuJoCo therefore retains
  exactly one active weld: `arm 'a'` to `stance_a` (activated during
  setup at `sim_loop.py:241-242` and never deactivated because swing
  arm is `'b'`, released at `sim_loop.py:1168-1169`).
- `crawlbot/simulation/sim_loop.py:1337-1338` — `step_idx += 1; i += 2`
  advances unconditionally, so the run falls into the trailing-DS
  branch.
- `crawlbot/simulation/sim_loop.py:1339-1390` — trailing DS branch.
  Line 1343 computes
  `cc_ds = self.sched.contact_config_at(plan.t_start[i] + 0.1)`.
- `crawlbot/planning/contact_scheduler.py:296-311` —
  `contact_config_at(t)` returns
  `ContactConfig.from_phase(gp.phase, r_a, r_b)` where `gp.phase` is
  the gait-plan phase enum at `t`. The plan for `n_steps=1` is
  `[DS, SS, DS]`, so the trailing `i` hits a DS phase.
- `crawlbot/solvers/contact_phase.py:50-65` —
  `ContactConfig.from_phase(ContactPhase.DOUBLE, …)` returns
  `active_contacts=(True, True), nc=2` with `r_contact_A, r_contact_B`
  equal to the scheduled anchors (A=stance, B=**target** — the anchor
  the swing arm failed to reach).
- `crawlbot/simulation/sim_loop.py:1597-1598` — inside `_step()` the
  contact Jacobians fed to the QP are built from
  `cc_ss.active_contacts[0], cc_ss.active_contacts[1]`
  (variable-named `cc_ss` but populated from `cc_ds` in this branch).
- `crawlbot/core/robot_interface.py:421-443` —
  `get_contact_jacobians(True, True)` stacks both tool Jacobians
  unconditionally from Pinocchio kinematics; it does **not** consult
  MuJoCo's `eq_active` state.
- `crawlbot/simulation/sim_loop.py:1382-1388` — the trailing-DS loop
  calls `self._step(..., cc_ds, ..., settle_mode=True)` for the full
  20 s hold window with this mismatched `cc_ds`.

**Conclusion: consistent.** The QP plans a task-consistent double-contact
solution: 12 wrench DOFs available, Jacobian stacked for both tools, a
second contact constraint `J_tool_B · v = 0` enforced in the QP even
though the physical weld does not exist. Any wrench the QP allocates to
the (imaginary) B contact is not felt by the torso in MuJoCo, so the
torque balance the controller expects is wrong. This biases the control
authority on the torso's rotational DOFs — consistent with a monotone
drift and with both `τ_q` and `τ_w` saturating as the QP tries to hold a
reference that its internal model believes is achievable with fictitious
reaction at B.

**Minimal experimental test (no fix proposed — this is just the
experiment that would settle the hypothesis).** Branch the trailing-DS
branch on `log.aborted_steps[-1]['reason'] == 'dock_timeout'` and
substitute
`cc_ds := ContactConfig.from_phase(ContactPhase.SINGLE_A, anchor_a,
anchor_b_current_actual)` (or `SINGLE_B`, whichever corresponds to the
surviving weld), then re-simulate the single step. If
`torso_ori_peak_deg_DS` drops materially (e.g. below 5°) and
`tau_peak_Nm_DS` drops out of saturation, H_DS1 is confirmed. If the DS
drift survives, the mismatch is not the dominant term.

---

## H_DS2 — reference discontinuity at the SS→DS boundary

**Code consulted.**

- `crawlbot/simulation/sim_loop.py:911-917` — at SS setup the torso
  planner is cleared and a single phase is added over
  `[t_ss_start, t_ss_start + T_step]`. With
  `cfg.torso_early_finish_fraction` (default 0.7) the trajectory
  completes at `t_ss_start + 0.7·T_step` and the phase holds at
  `(p_t1, R_t1)` for the remainder of the window.
- `crawlbot/planning/torso_planner.py:206-213` — `reference_at(t)`
  interpolates any phase covering `t`; outside all phases it returns
  `_hold_reference()`.
- `crawlbot/planning/torso_planner.py:277-290` — `_hold_reference()`
  returns `TorsoReference(p=_hold_p, R=_hold_R, …)`.
- v21 log: SS phase was added for `[0.11, 0.11 + 7.28] = [0.11, 7.39] s`
  (from `preplanner_T_steps[0] = 7.284 s`), so `reference_at(t)` during
  `t ∈ [7.39, 11.31]` falls through to `_hold_reference()` — held at
  `(p_t0, R_t0)` per `sim_loop.py:912`. The log shows SS torso_ori
  still peaks at only 0.53° in this window, so the in-SS fallback is
  not itself the issue.
- `crawlbot/simulation/sim_loop.py:1365-1375` — **entry to trailing
  DS** overwrites `_hold_p, _hold_R`:

      anchor_a_se3 = self.sched.anchor_se3('a', last_sa)
      anchor_b_se3 = self.sched.anchor_se3('b', last_sb)
      q_eq = dock_configuration(
          self.robot.model, anchor_a_se3, anchor_b_se3,
          q_init=pq)
      rs_eq = self.robot.update(q_eq, np.zeros(...))
      self.torso_planner.set_hold(
          rs_eq.oMf_torso.translation.copy(),
          rs_eq.oMf_torso.rotation.copy(),
          r_com=rs_eq.r_com.copy())

- `crawlbot/core/ik.py:128-167` — `dock_configuration(anchor_a,
  anchor_b, q_init=pq)` solves for a configuration with **both** tools
  at their anchors. On timeout, tool B is ~40 mm and ~7° away from
  its anchor; the IK therefore returns a `q_eq` — and hence
  `rs_eq.oMf_torso.rotation` — that differs from the actual torso
  rotation by a small angle (a rigid-body correction of the torso to
  compensate for the fictitious relocation of tool B onto its
  anchor).
- `crawlbot/planning/torso_planner.py:85-98` — `set_hold` simply
  stores `_hold_p, _hold_R`; nothing reconciles them with the
  current torso attitude.
- `crawlbot/planning/torso_planner.py:206-213` again — at the first
  DS tick `t = 11.41 s` the phase list still contains the old SS
  phase (nothing clears it), but `11.41 > 7.39`, so `reference_at`
  falls through to the new `_hold_reference()` with the IK-derived
  `(p, R)`.

**Diagnostic field (Step 1 refactor).**
`q_torso_ref_ss_to_ds_jump_deg = 3.42°` on v21 — the geodesic angle
between `q_torso_ref[i_ss_last]` and `q_torso_ref[i_ds_first]`. This is
**purely a reference discontinuity**, independent of the state
estimator, and its magnitude matches `ds_entry_torso_ori_deg = 3.33°`
(error at the first DS sample, before any controller reaction).

**Conclusion: consistent.** Across v17/v19/v20/v21 the jump is
2.94°/3.32°/3.32°/3.42° — all the same order as the final timeout
orientation error (`abort ori_deg ≈ 6.3–7.0°`), and roughly the
rigid-body rotation needed to bring tool B onto its anchor. The
reference injects an instantaneous ~3° step; in a system with
no passivity constraint on torso rotation (see H_DS3) and with the
wrong internal contact model (see H_DS1), even a ~3° step is a
sufficient excitation to see the controller chase it with saturating
torque.

**Minimal experimental test.** In the trailing-DS branch of
`sim_loop.py`, replace the `dock_configuration` + `set_hold(rs_eq, …)`
call with `set_hold(rs_hold.oMf_torso.translation,
rs_hold.oMf_torso.rotation, r_com=rs_hold.r_com)` — i.e. freeze the
reference at the **actual** torso pose at the SS→DS boundary. Re-
simulate v21. If `q_torso_ref_ss_to_ds_jump_deg` drops to <0.01° and
`torso_ori_peak_deg_DS` drops below 5°, H_DS2 is confirmed.

---

## H_DS3 — passivity formulation in single support

**Code consulted.**

- `crawlbot/simulation/sim_loop.py:1382-1388` — trailing-DS calls
  `_step(..., phase='DS', ..., settle_mode=True)` for the full 20 s
  window.
- `crawlbot/simulation/sim_loop.py:1712-1713` —

      passivity_active = bool(
          cfg.use_m2_stack and (phase == 'DS' or passivity_hold))

  so `passivity_active = True` for every QP call in the trailing-DS
  window.
- `crawlbot/simulation/sim_loop.py:1715-1730` — the call to
  `qp.solve(...)` passes `passivity_active=True` and
  `contact_config=cc_ss` (populated from `cc_ds` = DOUBLE in this
  branch).
- `crawlbot/solvers/wholebody_qp.py:444-459` — the passivity inequality
  is:

      # 4. Passivity constraint (M2, DS only): dq_j^T * τ_q + 2α·T ≤ 0
      #
      # T = 0.5 * dq_j^T * H_jj * dq_j uses the joint block of the
      # mass matrix (the full v=[dq_t; dq] is constrained by welds at
      # both EEs so only the joint kinetic energy matters here).
      if passivity_active and cfg.alpha_passivity > 0:
          H_jj = H_robot[6:, 6:]
          T_kin = 0.5 * float(dq @ H_jj @ dq)
          A_pass = np.zeros((1, n))
          A_pass[0, idx['tau'][0]: idx['tau'][1]] = dq
          b_pass = np.array([-2.0 * cfg.alpha_passivity * T_kin])
          qp.add_inequality_constraint(A_pass, b_pass)

  The comment at lines 444-450 is explicit: `T` uses only the joint
  mass block `H_jj`, which is valid **under the assumption that the
  free-floating base velocity `dq_t` is fully constrained by welds at
  both EEs** (nc=2 kinematic constraints on the 6-DOF base ⇒ 0 free
  base modes for a non-singular contact set).
- v21 log: `phase[i_ds_first:] == 'DS'` throughout, so the inequality
  is imposed on every one of the 200 DS ticks. `T_kinetic` is logged
  as a scalar and remains small; `dq_t` (the torso velocity) is not
  in the expression.
- Spec § 7.2.5 (brainstorming doc) — the passivity derivation is
  presented for rigid double-support; the single-support version is
  not derived, consistent with the comment in `wholebody_qp.py:449`.

**Conclusion: consistent.** On a timeout the physical system has one
weld, not two: the base's 6 DOFs minus the 6 kinematic constraints of
tool A give 0 constrained modes of the base only if the base and tool
A are rigidly coupled *and* the stance weld fully constrains all 6
rotational/translational axes at that contact; in any case a
single-contact weld at a point cannot constrain three base rotational
modes about axes through that point (the weld is a kinematic-coupling
constraint, not a 6-DOF ball-and-socket-plus-torque-clamp). Those
rotational modes have nonzero kinetic energy which is **not** counted
in `T = 0.5 · dq_j^T · H_jj · dq_j`. The constraint therefore bounds
only joint kinetic energy while the unregulated base rotation can
accumulate — consistent with the monotone 38°/20 s drift observed
with saturating torques.

This is a structural issue with the formulation as it is written, not
a parameter choice; it shows up only when the sim finds itself in a
de-facto single-support state while still flagged `passivity_active =
True`. That combination occurs exactly when trailing-DS is entered
after a dock_timeout — the same condition under which H_DS1 and H_DS2
activate.

**Minimal experimental test.** Two independent variants, to separate
engagement from formulation:

1. *Engagement test.* Force `passivity_active = False` for the
   trailing-DS ticks when `aborted_steps[-1]['reason'] ==
   'dock_timeout'`. Re-simulate v21. If the DS drift disappears (or
   changes sign / goes oscillatory rather than monotone), passivity is
   dominating the drift via the fake-constraint pathway. If it stays
   the same, passivity is not the primary cause.

2. *Formulation test (only if test 1 says passivity matters).*
   Replace `T_kin = 0.5 · dq_j · H_jj · dq_j` with the full
   free-floating kinetic energy projected onto the null space of the
   **actually active** contact Jacobian:
   `T_kin = 0.5 · v^T · (P_N^T · H · P_N) · v` with
   `P_N = I − Jc_active^+ · Jc_active`. Re-simulate v21 with both
   H_DS1 and H_DS2 still unfixed. If passivity then dissipates the
   torso rotation and the DS peak drops, the formulation is the
   fault, not the engagement.

---

## Cross-hypothesis observations (no interpretation — fact pattern only)

1. All three hypotheses require, as their triggering condition,
   exactly what happens on a dock_timeout: trailing-DS entry while
   MuJoCo still holds a single weld. They can therefore all be active
   on every archived v17–v21 run.
2. The SS-side value of each hypothesis' signature is ≈ zero:
   (H_DS1) SS uses a correctly-scheduled SINGLE contact config;
   (H_DS2) the intra-SS `q_torso_ref` is continuous and
   `torso_ori_peak_deg_SS ≈ 0.53°`;
   (H_DS3) `passivity_active = False` during SS (line 1712, `phase ==
   'DS' or passivity_hold`; `passivity_hold` defaults to `False`).
   All three are therefore silent in SS and loud in the post-abort
   DS — matching the per-phase metrics.
3. The three tests are mostly orthogonal: H_DS1 touches `cc_ds`,
   H_DS2 touches `set_hold` target, H_DS3 toggles
   `passivity_active`. They can be run independently to apportion the
   drift.

---

## Referenced files (single list for ease of audit)

- `crawlbot/simulation/sim_loop.py` — lines 241-242, 911-917,
  1168-1169, 1282-1298, 1300-1331, 1337-1338, 1339-1390, 1365-1375,
  1382-1388, 1597-1598, 1712-1713, 1715-1730.
- `crawlbot/planning/torso_planner.py` — lines 85-98, 206-213,
  277-290.
- `crawlbot/planning/contact_scheduler.py` — lines 296-311.
- `crawlbot/solvers/contact_phase.py` — lines 50-65.
- `crawlbot/solvers/wholebody_qp.py` — lines 444-459.
- `crawlbot/core/robot_interface.py` — lines 421-443.
- `crawlbot/core/ik.py` — lines 128-167.
- Spec — `docs/architecture/brainstorming_reworked_architecture.md`
  § 7.2.5.
- Log — `results/M7_1pct_1step_v21/sim_log.json` via
  `crawlbot.diagnostics.metrics.compute_metrics` (Step 1 refactor).
- Archive table — `results/archive_rediagnostic.md`.
