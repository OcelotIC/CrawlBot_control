# Campaign — 5-Step Cooperative-Arms Traversal: results & dock validation

**Date:** 2026-05-27
**Branch:** `claude/rework-controller-tasks-0MAgl`
**Scenario:** `scripts/diag_cooperative_arms.py` (cooperative-arms mode, 1% mass-ratio canonical, anchor grid dx=0.8m, 6 anchors/arm).
**Companion:** `STACK_OVERVIEW.md` is the code-ground-truth reference for what each layer actually does. This file is the *results* trace.

---

## 1. Headline

The cooperative-arms traversal now **docks all 5 steps (0–4) end-to-end**, and the docking accuracy is **within the real HOTDOCK capture envelope** with margin. This is the first end-to-end traversal on this branch.

| | before campaign | after |
|---|---|---|
| steps docked | died at step 2 | **0,1,2,3,4** |
| regression | — | **212 passed, 1 deselected** (pre-existing FK) |
| crashes / QP-FAIL | — | 0 / 0 |

---

## 2. What was fixed to get here (commit trail)

| # | fix | commit |
|---|---|---|
| 1 | **F-SAT rate-cap rescale** — the torso-ref rate limiter was capped at the body's 2-tick startup distance (~0.125mm/tick), throttling sustained motion so the reference never reached the dock. Re-scaled to planned-reference velocity + jitter slack. Steps 0–3 then dock (was die@2). | `87a2661` |
| 2 | **Constant CoM-z standoff (−0.35m)** — the missing crawl-height spec; dock-IK + initial config pinned so the body holds a uniform standoff. Torso clearance −24mm→+259mm; CoM-z range 408mm→60mm. | `61c3d5a` |
| 3 | **Step-4 = 3 bugs, not control** (the EE reached the anchor; gate/dynamics bugs blocked it): (a) `_cache_site_ids` cached only anchors 1–5 → `_gripper_distance` returned inf for anchor 6 → dock gate never fired; (b) stale `tau=zeros(12)` (6-DOF) crashed terminal DS settle on `ctrl[:14]`; (c) `_coop_A_lin` UnboundLocalError in `settle_mode`. | `7ac9124` |
| 4 | **Dock-gate velocity guard** — gate was purely positional; added `|v_ee| < dock_vel_max` (0.01 m/s) clean-dock criterion. Verified no-op at current speeds. | `dfa541b` |

Earlier supporting: swing-reference logging clamp (`1d11c9d`, killed a phantom 820mm `e_ee_pos`).

---

## 3. Dock-quality results (verified fresh run)

Values at the dock instant, per step:

| step | swing | d_grip [mm] | ori [deg] | \|v_ee\| [mm/s] | \|a_ee\| [mm/s²] | torso travel [mm] |
|---|---|---|---|---|---|---|
| 0 | b | 1.49 | 0.15 | 2.6 | 94.9 | 22 |
| 1 | a | 4.72 | 0.04 | 3.8 | 3.8 | 553 |
| 2 | b | 4.94 | 0.06 | 3.2 | 2.5 | 522 |
| 3 | a | 4.82 | 0.03 | 4.2 | 3.3 | 690 |
| 4 | b | 2.13 | 0.15 | 2.3 | 117 | 42 |

**Interpretation:**
- **Velocity at dock is low** (2–4 mm/s) and **orientation tight** (≤0.15°) → gentle, well-aligned docks, not impacts.
- **Acceleration splits two regimes:** long strides (1–3) are *quiescent* at dock (a≈3 mm/s²) — a **settled steady-state ~4.8mm offset**; short steps (0,4) reach ~1.5–2mm but **mid-transient** (a≈100–600 mm/s²).
- Recoil (torso travel, 22–690mm) is the **intended locomotion**, not a dock impact — it does not correlate with dock acceleration.
- **Tightening to 1mm/1° is infeasible** with current control: the EE only transiently grazes its closest approach (1.15mm best) then drifts; a 1mm gate misses it (step 0 → TIMEOUT 26.7mm). Reverted to 5mm/5°.

Diagnostics: `scripts/diag_dock_quality.py`, fig `results/diag_cooperative_arms/dock_quality.png`.

---

## 4. Realism benchmark vs HOTDOCK (the interface this system uses)

HOTDOCK (Space Applications Services; MOSAR/PERIOD) is validated to **mate from 23.5 mm lateral and 24° tilt misalignment**; its form-fit contour + chamfers mechanically guide/latch from there. Sister interface SIROM uses ~6–10 mm/s nominal capture approach velocity.

| quantity | our docks | HOTDOCK/SIROM | margin |
|---|---|---|---|
| lateral | ≤4.8 mm | **23.5 mm** capture | ~5× inside |
| angular | 0.15° | **24°** | ~160× inside |
| approach vel | 2–4 mm/s | ~6–10 mm/s | within/below |

**Conclusion: the docking is realistic and comfortably in-spec.** The current `weld_radius=5mm` gate is ~5× *stricter* than HOTDOCK — a conservative proxy. The ~4.8mm steady-state EE offset is **not a dockability problem** (it's ~20% of the real capture envelope). Chasing 1mm via control is the wrong goal — the real hardware captures these docks trivially.

Sources: Deremetz et al., "HOTDOCK: Design and Validation…" (ResearchGate 344871962); Space Applications Services HOTDOCK product page; MOSAR D2.7 HOTDOCK User Manual; PERIOD (OG12, DFKI); SIROM (SENER).

---

## 5. Group B — momentum / AOCS health (`scripts/diag_momentum_aocs.py`)

Checked against spec thresholds on the verified 5-dock run:

| metric | value | thresh | pass |
|---|---|---|---|
| hw saturation peak | 0.652 (3.26/5 Nms) | <1.0 | ✓ |
| hw saturation rms | 0.376 | <0.7 | ✓ |
| platform rotation total | 3.80° | <5° | ✓ (76% of budget) |
| platform ω peak | 0.41°/s | <2°/s | ✓ |
| **τ_w peak ratio** | **1.732 (8.66/5 Nm)** | <1.0 | **✗** |

**Verdict: momentum *state* healthy, margin eroding.**
- Wheels stay in box (65% peak), ω_s gentle, L_com low (mean 0.25 Nms) — NMPC momentum constraint + AOCS keep the state feasible.
- **τ_w over-commands transiently** (3% of ticks): bursts on step-2 SS (peak 8.66 Nm, ~1s) and step-4 end; steps 0,1,3 ≤3 Nm. Wheel `ctrlrange=±5` clamps it → wheels saturate, structure briefly under-actuated for attitude. **The step-2 burst is the same disturbance event as that step's 62N contact-force spike** (longest stride, 522mm torso travel).
- **Platform attitude accumulates** monotonically to 3.8° over 5 steps (~0.76°/step → would breach 5° ≈ step 7, extrapolated).
- Both concerns expected to **worsen at 14% mass ratio** (untested).

Fig: `results/diag_cooperative_arms/momentum_aocs.png`.

## 6. Verdict & open items

**Verdict:** end-to-end 5-step traversal **works**; dock criterion **validated in-spec** (A); momentum **state** healthy but with **eroding margin** (B: transient AOCS torque saturation + ~0.76°/step attitude accumulation).

**Still pending:**
- **C cascade band-aid** (CoM-z hold all steps, F-SAT clip rate, torso tracking) and **D actuator/solver** (the step-2 62N spike root, per-joint τ, NMPC solve rate). Note B already linked the step-2 τ_w burst ↔ the 62N spike — likely one event for D.
- Only the **1% mass-ratio canonical scenario**; 14% (spec T12) unchecked — and B suggests it's where margins bite.
- **FK-mode test** still red (deselected, not fixed).

**Open control items (not blocking the traversal):**
- Settled **~4.8mm steady-state EE offset** on long strides (cooperative torso-linear vs EE tension).
- **Loop-free mapping angular drift** (committed OFF) — spec §3 constrained-dynamic-singularity; §6 mitigation never implemented.
- **F-SAT / δ(q_current) debt** — live cascade is the world-frame δ + F-SAT band-aid.
- **AOCS torque margin** — transient saturation on the hardest stride; attitude accumulation per step.
