# INTERNAL — J1 Part B: terminal centroidal-DS characterization (Fix-A canonical)

**Goal:** characterize the **terminal centroidal-DS precedent** (`ds_centroidal_active=True`, AOCS on) —
the working active-DS controller **J2 generalizes** — and **re-confirm it on the Fix-A canonical**
(`main = ae0673e`), since Fix A changed the dock impact map and the terminal DS sits immediately after
the last dock.

**Mode:** READ-ONLY — code-path archaeology + characterization on the committed gate logs
(`results/fixA_gate` = Fix-A canonical; `results/p3b_gate_w24000_kp3` = pre-Fix-A). **No re-run, no
mutation, no `main` write, no PR.** The terminal-DS per-tick detail is in the committed runs. Tooling:
`scripts/audit_lemma2_partB.py`.

---

## ✅ RE-CONFIRMED — terminal-DS metrics equal/slightly-better under Fix-A; residual momentum 113× less

…with **one correction to the brief's premise** (B.1): the terminal DS does **not** run the SS two-task
working point (`ss_alpha_mom=5000` / `alpha_torso_pose=24000` / `Kp3`). It runs a **distinct
centroidal-DS task set** (CoM-3D + torso-**orientation**-3D) with its **own** weights
`ds_alpha_com=100` / `ds_alpha_torso_ori=200`. The 5k/24k/Kp3 working point is the **SS-swing**
controller. J2 must generalize the *DS-centroidal* task set, not the SS one.

## B.1 — Code path active in the terminal DS (confirmed on `ae0673e`)

The trailing/terminal DS is `sim_loop._step(phase='DS', settle_mode=True,
ds_centroidal_active=cfg.ds_centroidal_mode)` (sim_loop.py:2300–2307), with `cfg.ds_centroidal_mode=True`
(set by diag). It activates the Stage-2 centroidal-DS branch (`wholebody_qp.py:1065`,
`settle_mode and ds_centroidal_mode and ds_centroidal_active`), which **replaces** the joint-velocity-
damping cost (1053). The QP tasks:

| task | priority | reference (source) | weight | gains |
|---|---|---|---|---|
| **CoM-3D** (`J_com`·q̈ form ≡ `[A_G]_lin` up to mass scalar) | P1 | `r_com_ref = rp_interp` — interpolated **NMPC/preplanner CoM** ref, held at the docked target | **`ds_alpha_com=100`** | `Kp_com`/`Kd_com` |
| **torso-angular-3D** (`J_torso[3:]`) | P1 | `R_torso_ref` — **TorsoPlanner** orientation, held (`set_hold`) at the docked pose | **`ds_alpha_torso_ori=200`** | `Kp_torso[3:]=5`, `Kd[3:]=4` |
| passivity inequality (dq·τ + 2α·T ≤ 0) | — | energy dissipation (replaces joint-damping) | — | — |

- **F-SAT genuinely OFF — 0 calls:** the F-SAT rate-limiter (sim_loop.py:2662) is gated `phase == 'SS'`,
  so `_sat_total_calls` never increments in the terminal DS. The centroidal-DS references `r_com_ref`
  (NMPC) and `R_torso_ref` (TorsoPlanner) **directly** — no δ(q)-mapping rate-clamp band-aid, no
  algebraic loop. ✔ confirmed.
- **AOCS:** `legacy_pid_numerical` (PD + feedforward + desaturation), envelope ‖Ḣ_s‖∞ ≤ 5, tight wheel
  bounds `±hw_qp_tight` in DS.
- **⚠ Weight correction (flag for J2):** terminal-DS weights are `ds_alpha_com=100`,
  `ds_alpha_torso_ori=200` (config defaults; **not** overridden by the gate flags); torso-ori gains are
  `cfg.Kp_torso=[8,8,8,5,5,5]`/`Kd=[6,6,6,4,4,4]` (angular Kp5/Kd4). The brief's `5k:24k, Kp3` is the
  **SS** two-task working point and is **not** active here. Also the DS task is **CoM-position +
  torso-orientation**, not a full 6-D torso pose — torso *position* is not directly tasked (see B.3).

## B.2 — Terminal-DS metrics, Fix-A vs pre-Fix-A (the re-confirmation)

Terminal-DS window = trailing settle after the last swing (`phase=='DS'`, t > last-SS-end), ≈ t 30.3–50.3 s
(200 ticks @10 Hz) in both runs:

| metric | Fix-A | pre-Fix-A | note |
|---|---|---|---|
| torso pos error — **peak** [mm] | 152.5 | 152.5 | reference transient at DS entry (see below) |
| torso pos error — **mean** [mm] | 7.4 | 7.4 | small |
| torso **actual** displacement [mm] | 3.6 | 3.5 | torso barely moves (weld-anchored) |
| torso **ori** RMS [deg] | **0.135** | 0.136 | ≤0.68 ✔ |
| θ_s peak [deg] | **0.555** | 0.558 | ≤1.9 ✔ |
| θ_s final [deg] | **0.103** | 0.151 | ≤1.65 ✔ (matches full-run C4) |
| h_w peak ∞-norm | **1.348** | 1.429 | ≤4.5 ✔ |
| **residual ‖ΔH_sys‖ over terminal DS** | **0.0006** | 0.0724 | **113× less** |
| ‖H_sys‖ final (full run) | **0.0040** | 0.2030 | 51× (the Fix-A headline) |

- **Every metric is equal or slightly better under Fix-A** — nothing regressed. The biggest shift is the
  **residual system angular momentum across the terminal DS: 0.0724 → 0.0006 (113×)** — the terminal DS
  follows the last dock, so it directly inherits Fix A's corrected impact (the step-4 dock impact, the
  bulk of that 0.0724 pre-Fix-A, is now ~0.0004). C4-final and C5 also improve, matching the full-run
  gate (C4 final 0.15→0.10, C5 4.41→4.37).
- **The 152 mm pos-error PEAK is a reference transient, not a control failure:** the torso *actually*
  moves only 3.6 mm and holds orientation to 0.13°; the mean error is 7.4 mm. The DS torso-position
  *reference* (mapping/ramp output) jumps at DS entry while the weld-anchored torso stays put, spiking
  the error transiently. Identical across Fix-A. (SS `e_torso_pos` peak = 16.0 mm reproduces the gate
  C2 exactly — the metric is correct; the DS value is genuinely a reference transient.)

## B.3 — References & tracking (the J2 precedent)

- **What it tracks:** CoM-3D follows the **held NMPC/preplanner CoM** reference (`rp_interp`, the
  TorsoPlanner `com_reference_at` → NMPC, pinned at the docked target via `set_hold`); torso-angular-3D
  follows the **held TorsoPlanner orientation** `R_torso_ref`. Both held (not moving) in the terminal
  settle. NOT the structure-relative δ-mapping, NOT `lambda_qp`.
- **Tracking quality:** torso orientation RMS **0.13°**, θ_s ≤ 0.56°, torso actual motion 3.6 mm — the
  docked torso pose is well-regulated. CoM tracked at P1; AOCS holds attitude and desaturates the wheels
  (h_w peak 1.35 ≪ 4.5) within the ‖Ḣ_s‖ ≤ 5 envelope.
- **The docked-torso physical point (carry into J2):** in the free-floating **docked** configuration the
  stance arm welds the gripper to the anchor, kinematically anchoring the torso (manipulator-like chain).
  The terminal-DS controller embodies the consequence: it regulates **CoM position + torso orientation**
  at P1 (the terrestrial pelvis-free paradigm does **not** transfer — you cannot leave the base free).
  Torso *position* is not a separate task — it **emerges** from the CoM task + the welded-arm kinematics
  (which is why the position-vs-planner-ref error is reference-dependent, while the torso itself is
  pinned and stable). J2's active-DS-during-locomotion generalizes exactly this: CoM + torso-orientation
  regulation under the (moving) contact set, with the dock rework folding **Fix C** (zero approach
  velocity).

## B.4 — What this gives J2 (with flags)

A re-confirmed, Fix-A-canonical characterization of the working active-DS controller: task set
(**CoM-3D + torso-orientation-3D**, P1, + passivity), references (held NMPC CoM + TorsoPlanner
orientation), F-SAT off, AOCS `legacy_pid_numerical`. **Flags for J2:**
1. **The DS working point is `ds_alpha_com=100` / `ds_alpha_torso_ori=200` (Kp5/Kd4)** — distinct from the
   SS two-task `5k:24k/Kp3`. Generalize the DS task set, not the SS one.
2. **The DS task is CoM + torso-orientation, not a 6-D torso pose** — torso position emerges from CoM +
   the welded chain.
3. Under Fix-A the terminal DS is cleaner (momentum residual 113× lower); J2's active DS inherits the
   conserving plant.

**Verdict:** terminal centroidal-DS precedent re-confirmed on the Fix-A canonical — metrics equal or
slightly better, residual momentum 113× less, qualitatively unchanged. With the Lemma-2 SS validation,
this **closes J1**. (No commit unless Idriss directs.)
