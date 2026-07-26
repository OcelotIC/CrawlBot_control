# INTERNAL — J2 audit: architecture cartography (Fix-A canonical `ae0673e`)

**Goal:** establish factually, on the Fix-A canonical (`main = ae0673e`), what the code actually is, so
the J2 DS-rework spec is built on verified ground. **READ-ONLY** archaeology (no `crawlbot/` change, no
`main` write, no PR, no implementation). Branch `j2/ds-active-rework` off `ae0673e`. Method: 4 Explore
sub-agents (one per block) + direct verification of every load-bearing claim. Reproducer:
`Misc/scripts/audit_j2_facts.py`.

**TL;DR — the J2 starting position is much stronger than the framing assumed.** The NMPC is **already
n=2-formulated** and **already drives DS** (terminal + dwell) with **live** solves; the hyperstatic
2-contact distribution is **already resolved** (internal-stress regularization ON in canonical); EXT is
purged; the SS-flag removal is clean. J2 is mostly **wiring an existing capability into the inter-step DS
with moving references**, plus Fix C as an NMPC terminal condition — not building a DS controller from
scratch. Details + every code-vs-framing divergence in **MISMATCHES** at the end.

---

## Bloc 0 — `ss_two_task` flag + cost to make two-task the default

**Gate** (`wholebody_qp.py:630`): `_two_task = cfg.ss_two_task_mode and not settle_mode` — ON only in SS
with the flag set; **always OFF in `settle_mode` (DS)**.

**The two stacks, side by side** (all `qp.add_task(A, b, weight, priority)`):

| | flag ON (`_two_task`, `wholebody_qp.py:631–674`) | flag OFF (legacy, the 5 `not _two_task` blocks) |
|---|---|---|
| momentum/CoM | **T-MOM linear** `A_com`, `ss_alpha_mom`, **P2** (L633) | CoM-3D `A_com`, `alpha_com`, P1 (L677-679) |
| torso | **6-D torso-pose** `J_torso`, `alpha_torso_pose`, **P2** (L650-651) | torso-6D (+ coop angular/linear split, Option-D tube), `alpha_torso`, P1 (L685-816, gate L690) |
| swing EE | EE-6D `J_ee`, `alpha_ee`, P2, **unprojected** (L668) | EE-6D, `alpha_ee`, P2, **null-space projected** (L847-903) |
| posture | posture, `alpha_posture`, P3, unprojected (L672-674) | posture, `alpha_posture`, P3, projected (L989-1040) |
| soft-CoM | — | soft-CoM residual, `alpha_com_soft`, P4 (L962-978) |
| structure | strictly-**weighted** co-equal P2 trio (no projection) | null-space-**projected** hierarchy (M2 stack) |

So ON = the **fully-weighted two-task stack** (T-MOM ∥ torso-pose co-equal at P2 + EE + posture); OFF =
the **null-space-projected M2 hierarchy**. The wrench task (P4) and all dynamics/contact/momentum
**constraints** (`wholebody_qp.py:430–555`, `1093`) are **shared** (run regardless of the flag).

**Where OFF is used:** `--ss-two-task` (ON) is passed by `run_phase3_gate.sh`, `run_iterB_gate.sh`,
`run_fixA_gate.sh`, `run_phase2_2_sweep.sh`. The **only** consumer of the OFF path is the **C6(a)
byte-identical determinism check** (`run_fixA_rebaseline.sh` generates `phase3_off` /
`ssmom_phase1_baseline_fixA` with the flag OFF; `gate_phase3.py` compares them). **No test asserts the
legacy single-task behaviour** — `tests/test_reworked_qp.py` runs the default (OFF) M2/T-MOM configs but
nothing *relies on* the legacy SS task semantics. So OFF is a determinism-reference only, not a
functional dependency.

**Cost to make two-task default = CLEAN removal.** The legacy intermediates (`A_torso`, `A_ee`,
`_coop_A_lin`, `N_torso`, `A_torso_pinv`) are initialised to `None` and only populated *inside* the
`not _two_task` blocks; every downstream use is `None`-guarded (e.g. L831-833, L916-917, L1022). So
deleting the `if _two_task:` block + the five `and not _two_task` guards leaves the legacy blocks as dead
code that drops cleanly; nothing shared reads a variable defined only in an OFF block. (Recommend doing
it as the final cleanup pass per the framing, after the DS design lands.)

**⚠ Canonical SS working point — pinned (FLAG):** the real canonical is **`alpha_torso_pose=24000`,
`ss_alpha_mom=5000`, `ss_Kp_torso=3`/`Kd=2.5`** (`Misc/runs/p3b_gate_w24000_kp3/PHASE3_METADATA.txt` and
`Misc/runs/fixA_gate/PHASE3_METADATA.txt`; runner `run_iterB_gate.sh 24000 3 2.5`). **`run_phase3_gate.sh:15`
passes `--alpha-torso-pose 20000`** — this is a **stale exploratory runner**: its `phase3_wp` (20k) is
*not* the canonical and is unused by the gate verdict; only its `phase3_off` (flag-OFF) feeds C6(a). Two
gate-runner lineages coexist (`run_phase3_gate.sh` 20k-exploratory vs `run_iterB_gate.sh`/`run_fixA_gate.sh`
24k-canonical). Use **24k**; treat `run_phase3_gate.sh`'s 20k as legacy.

## Bloc 1 — can the NMPC drive DS? (YES — and it's n=2)

**NMPC:** `crawlbot/solvers/centroidal_nmpc.py`, class `CentroidalNMPC` (CasADi+IPOPT via `NMPCSolver`).
`NX=9` `[r_com, v_com, L_com]` (no h_w — AOCS independent), **`NU=12` `[f1,τ1,f2,τ2]`**, `NP=18`
`[r_ref, v_ref, r_C1, r_C2, c_simple, L_ref]`. **Real horizon `N=8`, `dt=0.1` (0.8 s)** — set in
`sim_loop.py:381-383` from `cfg.nmpc_N=8`/`cfg.nmpc_dt=0.1` (the class defaults `N=20`/`dt=0.05` are
**overridden**). Re-solved at 10 Hz (`dt_nmpc`).

**The NMPC IS called in DS.** `self.nmpc.solve(...)` appears at exactly **one** site, `sim_loop.py:2471`,
**inside `_step`, unconditional** (no phase/`settle_mode` gate). Since the **terminal DS** and the
**DWELL** (long inter-step DS, gated `cfg.ds_centroidal_mode and _dwell_target>1.0`, L1867) both run via
`_step(phase='DS', settle_mode=True, ds_centroidal_active=True)`, the **NMPC solves live there**. The
**only** DS regime where the NMPC is bypassed is the short **inter-step energy-dissipation settle**
(`_run_ds_passivity_loop`, L601-793) — it calls the QP directly (passivity-constrained) and never enters
`_step`. (A stale comment near L1794 says "NMPC is bypassed during DS" — true only for that passivity
loop, misleading in general.)

**n=2 — NO SS assumption baked in (verified).** The centroidal ODE uses **both** contacts
(`v̇=(f1+f2)/m`, `L̇=Σ(r_Cj−r_com)×f_j+τ_j`, `centroidal_nmpc.py:161-180`); SOC limits and the wheel-torque
(envelope) cap sum over both contacts (L251-268); active/inactive contacts are gated by **zeroing the
inactive contact's bounds** (`_apply_contact_bounds`, L631-656), *not* by reformulating. The
contact→centroidal map `compute_momentum_map` (`contact_phase.py:101-137`) is a fixed **3×12** for all
phases. `ContactConfig.from_phase` sets `nc=1, active=(T,F)/(F,T)` for SS and `nc=2, active=(T,T)` for DS.
So the NMPC, the envelope, and the wrench map are **contact-count-generic**; SS is the special case.

**Why DS-NMPC is "terminal/hold" today, not locomotion:** in terminal/DWELL DS the **TorsoPlanner is held
(`set_hold`)**, so the NMPC tracks a *static* docked setpoint (live solve, frozen target). The short
inter-step DS deliberately uses the passivity loop (energy dissipation, spec §7.1.1) — a reference-tracking
NMPC there would fight the dissipation. **No code gate forbids NMPC-driven DS during locomotion** — it's
just never given a *moving* DS reference, and the inter-step DS is routed to the passivity path.

## Bloc 2 — gait cycle structure

**Two-phase** (`ContactScheduler.plan_traversal`, `contact_scheduler.py:203-289`): initial DS → [SS_A | SS_B]
→ DS → … Durations: **DS `dt_ds=0.5 s` nominal but energy-based exit** (actual exit `T_kin<T_settle`,
`sim_loop.py:719-727,692`); **SS `dt_ss=0` placeholder**, real `T_step` installed per-step by the coarse
pre-planner (`sim_loop.py:1466`). Roles: SS = swing + dock; DS = settle/transition.

**Transitions & contact set n=1→n=2→n=1:**
- **SS→DS (dock, REACTIVE):** gate `sim_loop.py:2010-2031` fires when swing trajectory done **and**
  `d<5mm` **and** `ori<5°` **and** `v_ee<0.01 m/s` (`weld_radius`, `dock_ori_threshold_deg`,
  `dock_vel_max`; GMO variant uses contact-confirm). **Weld engages immediately** (`_activate_weld`,
  L2121 → `eq_active=1`). n: 1→2.
- **DS→SS (undock, PROACTIVE):** the planner schedule advances to the next SS phase; the old stance weld
  is released at SS entry (`_deactivate_weld`, L1949 → `eq_active=0`). n: 2→1.
- So **dock timing is sensor/proximity-driven; undock timing is schedule-driven.**

**EXT purged — confirmed, no residual.** `ContactPhase` enum = `{SINGLE_A, SINGLE_B, DOUBLE}`
(`contact_phase.py:19-23`); no `EXT`/`ext_phase`/three-phase branch anywhere in the runtime. Matches the
two-phase framing. ✔

## Bloc 3 — NMPC→QP in DS: hyperstatic distribution, ε_wrench, Fix-C surface

**12-D wrench distribution over 2 welds — IS resolved in canonical (verified, corrects a sub-agent
default-vs-actual error):**
- Primary: a **wrench-tracking task** (`A_wrench=I`, `b_wrench=lambda_ref`, weight `alpha_wrench`, P4,
  `wholebody_qp.py:1093-1135`) tracks the **NMPC's 12-D `lambda_ref`** — the NMPC resolves the net 6-D
  wrench.
- Redundancy: an **internal-stress regularization** (`wholebody_qp.py:1156-1176`) builds
  `P_int = I − G⁺G` (projector onto the **6-D internal-stress null space** — the wrench component that
  does *not* affect centroidal dynamics) and minimises it at P4, gated `alpha_lambda_int>0 and nc==2 and
  both active`. **It is ON in the canonical run:** `_build_qp` sets `alpha_lambda_int=cfg.ss_alpha_lambda_int`
  (`sim_loop.py:1013`) and **diag sets `ss_alpha_lambda_int=1.0`** (`diag_cooperative_arms.py:312`). (The
  QP dataclass *default* is 0.0 — hence "disabled" if you only read `wholebody_qp.py:86`; but the canonical
  config enables it.) So the hyperstatic case is well-posed: **NMPC `lambda_ref` for the net wrench +
  internal-stress minimisation for the redundant 6-D.** It does **not** rely on a quasi-passive second
  contact. (J2 should still verify the distribution quality in *active* DS, but the mechanism exists.)
- `compute_momentum_map` confirmed 3×12 for 2 contacts (`contact_phase.py:101-137`).

**ε_wrench (bridge to J3) — observable, partially logged.** `lambda_qp` (QP command) is logged at 100 Hz
and 10 Hz (`sim_loop.py:982-983, 3346-3347`). The **plant** side logs only the **torque** component of the
weld reaction (`qfrc_constraint[3:6]`, `sim_loop.py:3263-3264`) — the **full 6-D plant weld wrench is NOT
in the committed log**. (The J1 Lemma-2 work reconstructed the full wrench from `qfrc_constraint` via the
relative-site weld Jacobian — `scripts/audit_lemma2.py` — and found SS |f|≈3.4 N / ‖λ_qp‖≈5.7.)
Characterising ε_wrench *in DS* therefore needs that full reconstruction; the hook exists, the number is
not yet computed (left to J3).

**Fix-C surface — the lock is NOT fully "dry", but the existing guards are EE-velocity heuristics, not the
constraint twist:**
- The swing planner already drives `v_ee=0` (and accel 0) at terminal τ=1 (`swing_planner.py:532-540`),
  and the dock gate requires `v_ee<0.01 m/s` (`sim_loop.py:2026-2027`). **But both gate the gripper's
  *linear EE speed*, not the 6-D weld-relative constraint twist** `Jc·v⁻` that Fix A showed is what the
  impact map mis-handles. So the lock is a **heuristic-soft, not constraint-velocity-zero**.
- **Fix C insertion points** (zero 6-D weld-relative velocity at lock):
  1. **NMPC terminal constraint** (`centroidal_nmpc.py` terminal block ~L309-326, the `c_simple`/terminal
     tightening) — **the framing's preferred home; cleanest** (drive the predicted weld-relative twist to
     0 at the horizon end → the dock arrives with `Jc·v⁻≈0`).
  2. swing-planner terminal velocity expressed as the **weld-relative** twist (not just EE linear v).
  3. the pre-weld dock gate (`sim_loop.py:2026`) — replace `v_ee<0.01` with `‖Jc·v⁻‖<ε`.
  These are the **planning-side analogue** of the physics-side Fix A: Fix A made the *impact map*
  momentum-consistent; Fix C makes the *approach* arrive with no constraint velocity so there is no impact
  to absorb.

---

## MISMATCHES vs the J2 framing (every code-vs-framing divergence)

1. **"Live NMPC instead of frozen settle_mode refs" — the NMPC is ALREADY live in DS.** Terminal/DWELL DS
   call `self.nmpc.solve` every `_step` (unconditional, `sim_loop.py:2471`). What is "frozen" is the
   **TorsoPlanner target** (`set_hold`), not the NMPC. ⇒ **J2's real gap is a *moving* DS reference**
   (locomotion vs hold) **and routing the inter-step DS through the NMPC path** — not adding NMPC to DS.

2. **Two DS regimes exist, not one.** Inter-step DS = **passivity loop, NMPC bypassed** (energy
   dissipation, `_run_ds_passivity_loop`); terminal/DWELL DS = **NMPC-driven hold** (`_step` settle).
   ⇒ The "generic DS controller" must **unify** these — generalise the terminal/DWELL NMPC-DS to the
   inter-step window with moving refs (and decide what happens to the energy-dissipation requirement).

3. **No n=1 (SS) assumption is baked into the NMPC — POSITIVE, de-risks J2.** `NU=12`, both-contact
   dynamics/SOC/envelope, 3×12 momentum map; SS is `nc=1` via zeroed bounds. The framing's worry ("does
   the code hard-code a single contact?") is **answered: no**.

4. **The hyperstatic 2-contact distribution is ALREADY resolved in canonical — POSITIVE.** Internal-stress
   regularization is **ON** (`alpha_lambda_int=1.0` via `ss_alpha_lambda_int`), not the dataclass-default
   0.0. ⇒ J2 inherits a working redundancy resolver (NMPC `lambda_ref` + internal-stress min); verify, do
   not rebuild.

5. **Fix C is a *strengthening*, not a from-scratch add.** The lock already has EE-speed guards
   (`v_ee<0.01`, planner `v_ee=0` at τ=1) — but they gate **EE linear speed**, not the **6-D weld-relative
   twist**. ⇒ Fix C = replace the heuristic with the constraint-velocity condition, ideally as an NMPC
   terminal constraint (matches the framing).

6. **ε_wrench in DS is not directly logged** — only `qfrc_constraint[3:6]` (torque) is. Full plant weld
   wrench needs the Lemma-2-style reconstruction. ⇒ J3 surface; quantify later, not a J2 blocker.

7. **NMPC horizon is N=8 / dt=0.1 (0.8 s)** in config (not the class-default 20/0.05). Matches CLAUDE.md;
   noted so the spec uses the right horizon.

8. **Working-point discrepancy (FLAG):** `run_phase3_gate.sh:15` = `alpha_torso_pose 20000` is **stale
   exploratory**; the canonical SS working point is **24000** (run_iterB/fixA + frozen `p3b_gate_w24000_kp3`).

9. **Terminal-DS task set = CoM-3D + torso-orientation-3D, `ds_alpha_com=100`/`ds_alpha_torso_ori=200`,
   `Kp_torso[3:]=5`/`Kd=4` — matches the framing** (Part-B correction already folded in). ✔ Confirmed, no
   divergence: it is **not** a 6-D torso pose and **not** the SS `5k:24k/Kp3`.

10. **EXT fully purged — matches framing.** ✔

---

## What this means for the J2 spec (scoping, not design)

- The **DS controller largely exists**: n=2 NMPC + centroidal-DS QP tasks (CoM-3D + torso-ori-3D) +
  internal-stress-resolved hyperstatic distribution, already running live in terminal/DWELL DS. J2 is
  **(a)** give it **moving** locomotion references across the switch, **(b)** route the **inter-step DS**
  through it (reconciling energy dissipation), **(c)** add **Fix C** as an NMPC terminal constraint, and
  **(d)** the final SS-flag-removal cleanup (clean, deferred).
- The audit found **no baked-in SS assumption** and **no unresolved hyperstatic distribution** — the two
  risks the framing flagged are absent. The open design questions are the **inter-step DS unification** and
  the **moving-reference generation across n=1↔n=2**.

**STOP — doc-first.** Awaiting the digest before any design or implementation. No `crawlbot/` change, no
`main` write, no PR.
