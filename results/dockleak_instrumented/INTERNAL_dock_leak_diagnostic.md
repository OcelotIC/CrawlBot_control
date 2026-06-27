# INTERNAL — Dock-Leak Diagnostic, Part 1 (confirm the mechanism)

**Study:** test whether the ~0.203 N·m·s DS angular-momentum leak (J1 A′.1) is a
soft-weld / finite-dock-gap constraint-stabilization artefact, and if so, suppress it.
**Branch:** `diag/dock-leak` (off `21cec74`; push, never merge). **Mode:** RE-RUN.
**Frozen baseline (untouched):** `results/p3b_gate_w24000_kp3/` + controller weights.
**Instrumented re-run:** `results/dockleak_instrumented/` (this dir). **Metric:**
`subtree_angmom[0]` — the validated J1 A′.1 ground truth.
**Tooling:** instrument in `crawlbot/{simulation/logging.py, simulation/sim_loop.py}`
(per-tick `H_sys`/`n_weld` + per-dock `dock_probe`); `scripts/run_dockleak_instrumented.sh`,
`audit_dock_leak.py`, `plot_dock_leak.py`. Figure: `dockleak_plots/dock_leak_attribution.png`.

---

## ⛔ HEADLINE — the leak is REAL, but the hypothesised mechanism is WRONG

The leak reproduces **bit-identically** (final ‖H_sys‖ = **0.2030 N·m·s**, matching A′.1)
and localises to a **single logged tick at each dock**. But the per-dock sub-event probe
shows it is **not** soft-weld / finite-gap stabilisation. It is the **inelastic impact
velocity projection** (`sim_loop.py` ~L2164–2187) — a discrete, controller-side reset of
the torso+arm velocities applied at every dock that does not conserve total angular
momentum. Attribution over the 5 engagements (Σ of per-dock ‖ΔH‖):

| sub-event | code | Σ‖ΔH‖ [N·m·s] | share |
|---|---|---|---|
| weld activation (`eq_active`+`mj_forward`) | L2166–2167 | **0.0000** | 0 % |
| **inelastic impact velocity projection** | **L2182–2187** | **0.3565** | **≈100 %** |
| gap-closing weld stabilisation (1st DS step) | L766/3085 | **0.0022** | 0.6 % |

The brief's falsifiable predictions for the soft-weld mechanism are **not** met:
prediction (1) localises ✅ — but to the *impact projection*, not the weld; (2) "scales
with gap" ✗ (corr = **−0.40**, weakly *negative*); (5) "ΔH ∥ g×F" ✗ (no consistent
direction). The soft-weld / finite-gap term exists but is **0.6 %** of the leak.
Per the brief — *"if gap/stiffness scaling is absent, the mechanism is wrong — report it"* —
**we stop and report.** Part 2 (solref/solimp + gap sweep) as written would tune a 0.6 %
effect; the real lever is the impact map. Recommendation + corrected test below.

---

## 0. Non-perturbation proof (the instrument is read-only)

Instrumented run vs frozen canonical, all shared log fields: **69 fields compared,
worst |Δ| = 0.000e+00 → BIT-IDENTICAL.** The instrument (`mj_kinematics/comPos/comVel/
subtreeVel` read-out + appends) never touches `qpos/qvel/ctrl`, so the physical
trajectory is unchanged. Anything the instrument reports is a property of the canonical
plant, not of the instrumentation. (This also re-confirms `21cec74` reproduces the
canonical run exactly.)

**Regression check:** `pytest tests/` → **220 passed, 1 failed**. The single failure
(`test_E7_t15_step2_dock_under_fk_mode`) is **pre-existing on clean `21cec74`** (verified
by stashing the instrument and re-running: identical abort signature) — it validates a
committed FK-reference-mode artefact (`results/M7_1pct_3step_v22_t15_fk/sim_log.json`)
that already carries aborted steps. This is the loop-free / FK-mapping track listed under
CLAUDE.md "Open"; it is unrelated to the (read-only) dock-leak instrument and out of scope
here.

## a. Localization — ΔH is a single-tick step at each dock

**(a.1) Coarse (per `n_weld` segment, `n_weld`=1 SS / =2 DS):** SS swings conserve H_sys
**exactly** (max ‖ΔH_vec‖ = **0.0000** over all 5 swings); the *interior* of every DS
window is also flat (max ‖ΔH_vec‖ = **0.0016**, mean 0.0005). So the injection is neither
in SS nor in the DS dwell — it is at the **SS→DS transition**.

**(a.2) The step is one logged tick wide** at each `n_weld` 1→2 (dock) transition:

| dock | t [s] | ‖H‖ before | ‖H‖ after | ‖ΔH‖ (1 tick) | ‖ΔH‖ (rest of DS) |
|---|---|---|---|---|---|
| 0 | 3.32 | 0.0000 | 0.0410 | **0.0410** | 0.0016 |
| 1 | 13.44 | 0.0419 | 0.0225 | **0.0480** | 0.0001 |
| 2 | 17.25 | 0.0226 | 0.0618 | **0.0491** | 0.0014 |
| 3 | 26.37 | 0.0617 | 0.1350 | **0.1457** | 0.0001 |
| 4 | 30.37 | 0.1349 | 0.2030 | **0.0724** | 0.0000 |

97–100 % of each engagement's injection is in the dock tick; the DS dwell adds nothing.

**(a.3) Sub-event attribution (the decisive test).** The `dock_probe` reads H_sys at
H0 (pre-activation), H1 (weld active, before any Δqvel), H2 (after the impact projection),
and H3 = first DS tick. Probe placement verified: H0 == last-SS tick exactly, H2 ≈ H3 to
3e-4. Per dock:

| step | gap [mm] | v_app [m/s] | ‖H0→H1‖ | ‖H1→H2‖ | ‖H2→H3‖ | dominant |
|---|---|---|---|---|---|---|
| 0 | 4.940 | 0.0060 | 0.0000 | 0.0409 | 0.0002 | impact-proj |
| 1 | 4.507 | 0.0039 | 0.0000 | 0.0480 | 0.0004 | impact-proj |
| 2 | 4.934 | 0.0061 | 0.0000 | 0.0491 | 0.0003 | impact-proj |
| 3 | 4.618 | 0.0041 | 0.0000 | 0.1456 | 0.0004 | impact-proj |
| 4 | 4.891 | 0.0059 | 0.0000 | 0.0729 | 0.0009 | impact-proj |

- **Weld activation alone is H-neutral (0.0000).** Setting `eq_active` + `mj_forward`
  changes no velocity, so H_sys is unchanged — exactly as it must be.
- **The impact velocity projection is the injector.** Σ = 0.3565; and the *vector* sum of
  the per-dock (H2−H1) is [−0.016, −0.009, −0.205], ‖·‖ = **0.2057 ≈ final H_sys 0.203**.
  The summed norms (0.357) exceed the final norm because successive injections partially
  cancel (the z-axis kicks alternate sign) — but the projection accounts for essentially
  **all** of the residual leak.
- **The soft-weld gap-closing step contributes 0.0022** (≈0.6 %).

## b. Gap regression — does NOT scale with the dock gap

| step | gap [mm] | v_app [m/s] | ‖ΔH‖ | ‖ΔH_impact‖ | ‖ΔH_gapstab‖ |
|---|---|---|---|---|---|
| 0 | 4.940 | 0.0060 | 0.0410 | 0.0409 | 0.0002 |
| 1 | 4.507 | 0.0039 | 0.0480 | 0.0480 | 0.0004 |
| 2 | 4.934 | 0.0061 | 0.0491 | 0.0491 | 0.0003 |
| 3 | 4.618 | 0.0041 | **0.1457** | 0.1456 | 0.0004 |
| 4 | 4.891 | 0.0059 | 0.0724 | 0.0729 | 0.0009 |

`corr(‖ΔH‖, gap) = −0.40`, `corr(‖ΔH‖, v_approach) = −0.50` (both weakly **negative**),
`corr(‖ΔH_gapstab‖, gap) = +0.09` (the genuine soft-weld term is tiny and ~uncorrelated).
The largest injection (dock 3, 0.146) has neither the largest gap nor the largest approach
speed — its size is set by the **whole-body velocity field** the projection annihilates at
that configuration, not by the gap. *(5 engagements — descriptive; a deliberate sweep is
deferred. But the sign is unambiguous: there is no positive gap-scaling.)*

## c. Direction — no consistent ΔH ∥ g×F

| step | ΔH direction (unit) | cos(ΔH, gap) | cos(ΔH, lever×gap) | cos(ΔH, lever) |
|---|---|---|---|---|
| 0 | [−0.27, −0.77, +0.58] | −0.87 | +0.33 | +0.27 |
| 1 | [+0.18, +0.39, −0.90] | +0.36 | −0.93 | +0.34 |
| 2 | [−0.20, −0.96, +0.19] | −0.98 | +0.01 | +0.05 |
| 3 | [+0.06, +0.49, −0.87] | +0.45 | −0.89 | −0.17 |
| 4 | [−0.21, −0.29, −0.93] | −0.22 | −0.96 | −0.17 |

The alignment with both the gap vector and the geometric weld-couple `lever×gap` **flips
sign** engagement-to-engagement — inconsistent with a single gap-induced weld couple. The
direction is what an inertia-weighted velocity projection produces (set by the arm velocity
field at impact), not a `g×F` constraint couple. (Prediction 5 not supported.)

---

## Root cause (high confidence; one confirmatory check pending)

The impact map (`sim_loop.py` L2164–2187) is the standard inertia-weighted projection
`v⁺ = v − M⁻¹Jcᵀ(Jc M⁻¹ Jcᵀ)⁻¹ Jc v`, which conserves momentum **only when applied to the
full free system and written back in full**. As implemented it does **neither**:

1. It is computed in the **structure-fixed-base** Pinocchio model (`robot.update`, root =
   structure) — so the projection treats the structure as ground and cannot represent the
   structure recoil that conservation requires.
2. It is **written back only to `qvel[6+off:]`** (torso base + arm joints; L2186), leaving
   the structure base `qvel[0:6]` and wheels `qvel[6:9]` at their pre-impact values.

The result is a velocity discontinuity that is **not** a momentum-conserving impact, so
each dock injects Δ(subtree_angmom). The codebase already half-flags this: the
`_gripper_speed` docstring (L1157–1159) notes "the weld's inelastic impact projection
injects a momentum impulse (recoil / force spike)." Gravity is `0 0 0` (microgravity, RK4),
so the free system *must* conserve H about its CoM — the 0.203 N·m·s is purely this artefact.

---

## Verdict & recommendation (STOP — reviewer's call before Part 2/3)

1. **Leak confirmed and characterised:** 0.203 N·m·s (≈4 % of the ±5 N·m·s wheel budget,
   z-dominated), bit-identically reproduced, single-tick at each dock, vector-summing to
   the final value.
2. **Hypothesised mechanism refuted:** soft-weld / finite-gap stabilisation is **0.6 %**;
   gap/speed scaling is absent (corr −0.40/−0.50); no `g×F` direction. **The mechanism is
   the inelastic impact velocity projection (≈100 %).**
3. **Part 2 as written should be redirected.** Sweeping weld `solref/solimp` and the dock
   gap tunes the 0.6 % term; it will *not* move the leak. The lever is the impact map.
4. **Corrected falsifiable test for the next phase** (cheap, static-eval first):
   - Recompute the impact projection in the **full MuJoCo DOF space** (project + write back
     all DOFs incl. structure base & wheels) at each dock snapshot and check Δ(subtree_angmom)
     → should be ≈0 if the partial/fixed-base write-back is the cause.
   - **Zero-gap probe note:** snapping the gripper to exact alignment before `eq_active=1`
     (the brief's Part-2 probe) will *not* null this leak — the projection acts on velocity,
     not position. The momentum-consistent test is zero pre-impact constraint **velocity**
     (`Jc·v⁻ = 0`) or a full-DOF conserving impact map.
5. **Part 3 gating:** the brief gates the fix on "mechanism confirmed." The confirmed
   mechanism differs from the hypothesised one, which changes the fix surface (impact map,
   not weld parameters). **Deferring the fix to the reviewer's go**, with the
   full-DOF/momentum-consistent impact map as the recommended candidate, re-gated 6/6.

---

# PART 2 (REDIRECTED) — full-DOF confirmatory test

**Mode:** READ-ONLY, static evaluation on the 5 logged pre-activation dock snapshots — the
impact projection is recomputed two ways and compared. No re-run, no `sim_loop.py` change.
**Tooling:** `scripts/audit_dock_leak_part2.py`, `plot_dock_leak_part2.py`. Figure:
`dockleak_plots/dock_leak_part2.png`. **Snapshot fidelity:** `subtree_angmom[0]` of each
`dock_stepN` snapshot equals the Part-1 probe H0 to <1e-6 (the snapshots are the exact
pre-impact states the in-plant map acts on).

## ✅ Test A — DISCRIMINATOR CONFIRMED: full-DOF conserves, current partial leaks

Both maps recomputed from identical pre-impact `qpos/qvel`; `Δ = ‖subtree_angmom[0] after −
before‖`:

| dock | gap [mm] | **A.2 current partial** | Part-1 leak | **A.1 full-DOF** | reduction |
|---|---|---|---|---|---|
| 0 | 0.005 | 0.0409 | 0.0409 | 0.0004 | 111× |
| 1 | 0.003 | 0.0480 | 0.0480 | 0.0001 | 956× |
| 2 | 0.005 | 0.0491 | 0.0491 | 0.0003 | 146× |
| 3 | 0.002 | 0.1456 | 0.1456 | 0.0001 | 2276× |
| 4 | 0.006 | 0.0729 | 0.0729 | 0.0003 | 252× |
| **Σ** | | **0.3565** | 0.3565 | **0.0011** | |

- **A.2 reproduces the Part-1 leak bit-for-bit** (Σ = 0.3565) — the offline re-projection is
  an exact replica of the in-plant map (robot-only Pinocchio `H`/`Jc`, write back only
  `qvel[6+off:]`).
- **A.1 (full MuJoCo `mj_fullM` + relative-site weld Jacobian over all DOFs, write back all
  DOFs) collapses the leak 110×–2276×** to Σ = 0.0011 — matching the Part-1 gap-stabilisation
  residual (0.0022). This residual is the expected **O(gap×f) couple** (a full action-reaction
  impulse pair across a 2–6 mm gap is not *exactly* torque-free), not the one-sided defect.
- **Discriminator met:** A.1 ≪ A.2 **and** A.2 reproduces the observed leak ⇒ the root cause is
  the partial/structure-fixed-base impact map; the full-DOF momentum-consistent map (**Fix A**)
  removes it down to the gap couple.

## Test B — the lever is VELOCITY, not gap (and the partial map has two velocity defects)

**B.1 — decompose the partial map** (`Δ` vs the pre-impact H0; `ΔH_impulse = ‖after − no-impulse‖`):

| dock | ΔH_full | ΔH_no-impulse (conversion only) | ΔH_impulse |
|---|---|---|---|
| 0 | 0.0409 | 0.0140 | 0.0443 |
| 1 | 0.0480 | 0.0635 | 0.0202 |
| 2 | 0.0491 | 0.0736 | 0.1071 |
| 3 | 0.1456 | 0.1951 | 0.0518 |
| 4 | 0.0729 | 0.2060 | 0.1618 |

The partial map injects **even with the impulse removed** (Σ conversion-only = 0.55): the
`pinocchio_to_mujoco` write-back is documented as *"assumes v_struct ≈ 0"*, but at a dock
`v_struct ≠ 0`, so it drops the structure-coupling terms (`v_s + ω_s×Δp`) when mapping the
torso velocity back to world. So the net leak is the **partial cancellation of two
velocity-driven defects** — the one-sided impulse and the lossy structure-relative conversion
(they are ~antiparallel, `cos ≈ −0.95` at dock 4). **Both are eliminated by the full-DOF,
MuJoCo-native map** (A.1) — which is why a half-fix that only widens the write-back but keeps
the Pinocchio conversion would be insufficient; Fix A sidesteps the conversion entirely.

**B.2 — velocity scaling (gap held fixed):** scale the pre-impact velocity by α and recompute
the partial-map injection:

| dock | gap [mm] | α=1.0 | α=0.5 | α=0.25 | α=0.0 |
|---|---|---|---|---|---|
| 0 | 0.005 | 0.0409 | 0.0205 | 0.0102 | 0.0000 |
| 3 | 0.002 | 0.1456 | 0.0728 | 0.0364 | 0.0000 |
| 4 | 0.006 | 0.0729 | 0.0364 | 0.0182 | 0.0000 |

ΔH is **exactly linear in approach velocity** and **→ 0 at zero approach velocity**, while the
(fixed, nonzero) gap is unchanged. This is the corrected null test (Part 1 already refuted the
zero-*gap* null: `corr(‖ΔH‖, gap) = −0.40`). **The lever is the pre-impact velocity.** This
validates **Fix C** (null the approach velocity before `eq_active = 1` — a soft-dock) as a
momentum-consistent target, alongside Fix A.

## Part 2 verdict (STOP — Part 3 gated on paper-timing decision)

1. **Root cause confirmed at high confidence (Test A):** the leak is the structure-fixed-base /
   partial-write-back impact map; the full-DOF momentum-consistent map conserves
   `subtree_angmom[0]` to the O(gap×f) residual (0.0011, ~0.3 % of the leak).
2. **Refinement (Test B.1):** the partial map carries *two* velocity-driven defects — the
   one-sided impulse **and** the setup-only `pinocchio_to_mujoco` conversion used at nonzero
   `v_struct`. Both vanish under the full-DOF map.
3. **Lever is velocity, not gap (Test B.2):** ΔH ∝ approach velocity, zero at zero velocity.
   Two fixes are validated: **Fix A** (full-DOF conservative impact map) and **Fix C** (soft-dock
   = null approach velocity before welding).
4. **Did NOT proceed to Part 3.** No `sim_loop.py` impact-map edit, no re-gate. The Fix-A-now
   vs fold-into-J2-dock-rework choice depends on the paper-timing decision — **awaiting explicit
   go.**
