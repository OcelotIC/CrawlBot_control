# INTERNAL — DS-Active J1 Part A / A′ (conservation validation)

**Target:** commit `21cec74` (canonical), run `results/p3b_gate_w24000_kp3/`. **Mode:** READ-ONLY,
static evaluation on logged snapshots (no stepping, no re-run, no source edits). Scripts:
`scripts/audit_J1_partA.py`, `audit_J1_partAprime.py`. Theory record:
`ENVELOPE_NCONTACT_6D_DERIVATION_2026-06.md` (§B inertial).

**A.0 ✓** `21cec74:crawlbot == HEAD == fca58ab`; `crawlbot/` clean; `sim_log.json` present.

---

## ⛔ HEADLINE FLAG — total angular momentum is NOT conserved in the canonical plant

The MuJoCo-direct ground truth (A′.1) overturns the working premise that the free system conserves
H. **Total system angular momentum `subtree_angmom[0]` grows 0 → 0.203 N·m·s over the traversal,
injected during DS phases and exactly conserved during SS swings.** The earlier A.1 non-closure was
therefore **correctly detecting a real leak**, NOT the frame-mixing artefact hypothesised in the A′
brief (A′.2 refutes the co-rotation explanation). Conservation — the J0 envelope backbone — does
**not** hold exactly in this plant; it leaks at the DS welds. First-order flag for J2.

---

## A′.1 — MuJoCo-direct total angular momentum (decisive)

Static eval per snapshot: set `qpos/qvel` → `mj_kinematics, mj_comPos, mj_comVel, mj_subtreeVel` →
`subtree_angmom[0]` (whole system about system CoM, world frame). Free system from rest ⇒ must be ≈0.

| snapshot | t [s] | ‖H_sys‖ | H_sys [x,y,z] | ‖v_comSys‖ |
|---|---|---|---|---|
| initial / release_step0 | 0.00 / 0.11 | **0.0000** | [0,0,0] | 0 |
| dock_step0 | 3.31 | **0.0000** | [0,0,0] | 0 |
| release_step1 → dock_step1 | 5.03 → 13.43 | 0.0419 → 0.0419 | [−.012,−.031,+.025] | ~0 |
| release_step2 → dock_step2 | 13.94 → 17.24 | 0.0226 → 0.0226 | | ~0 |
| release_step3 → dock_step3 | 18.26 → 26.36 | 0.0617 → 0.0617 | [−.014,−.060,−.008] | ~0 |
| release_step4 → dock_step4 | 26.87 → 30.37 | 0.1349 → 0.1349 | [−.005,+.012,−.134] | ~0 |
| **final** | 50.37 | **0.2030** | [−.020,−.009,−.202] | ~0 |

**The pattern is unambiguous:**
- **SS swings conserve H_sys EXACTLY** — every `release_stepN == dock_stepN` to 4 decimals (e.g.
  5.03→13.43: 0.0419→0.0419). This validates the computation (it conserves perfectly when it
  should) and rules out integration noise.
- **Every DS phase injects momentum** — H_sys changes only across DS windows: +0.042 (DS0), −0.019
  (DS1), +0.039 (DS2), +0.073 (DS3), +0.068 (terminal), accumulating to **0.203 N·m·s** ≈ 4 % of
  the ±5 N·m·s wheel budget. Linear momentum stays conserved (`v_comSys ≈ 0`).

**⇒ FLAG (model/contact leak).** Per the brief this halts the "conservation holds" conclusion. The
leak is structured (DS-only), so it is a real plant effect, not numerics. **Most likely mechanism
(reported, not asserted):** the DS phase is the only regime with the *second* weld active (both
grippers welded → closed loop); the just-docked weld closes a non-zero dock gap (~4.5–5.0 mm, the
gate tolerance) under MuJoCo's soft-constraint/Baumgarte stabilisation, which is not momentum-
conserving. SS (single steady weld) conserves; DS (second weld engaging + hyperstatic closed-loop
constraint solve) leaks. Exact cause to be confirmed before J2 — not guessed here.

## A′.2 — Inertial decomposition: co-rotation hypothesis REFUTED, I_s=1777 fine

`H_sys = H_robot/Os + I_s·ω_s + h_w` at settled ticks (world frame; `I_s·ω_s`, `h_w` rotated R_s→world):

| tick | ‖ω_s‖ [mrad/s] | I_s·ω_s | h_w | H_robot/Os (inertial) | H_robot (struct-rel, log) | I_robot back-out |
|---|---|---|---|---|---|---|
| release_step1 | 0.365 | 0.467 | 0.438 | **0.024** | 0.046 | 66 |
| release_step3 | 0.818 | 1.339 | 1.404 | **0.134** | 0.071 | 163 |
| final | 0.150 | 0.257 | 0.479 | **0.035** | 0.002 | 234 |

- The A′ brief predicted a missing robot co-rotation `H_robot/Os ≈ 0.244` (back-out `I_robot ≈ 1294`)
  that would close the budget to 0. **The data refutes this:** `H_robot/Os` (inertial) is small
  (0.02–0.13), ≈ the structure-relative logged value (both ~0), and `I_robot` back-out is 66–234
  (not ~1294) and not constant. There is **no large co-rotation term**.
- Consequently the inertial budget does **not** close to 0 — it closes to `H_sys` (the leak):
  `H_robot/Os + I_s·ω_s + h_w = H_sys ≠ 0`. The original A.1 residual (≈ `I_s·ω_s + h_w` since the
  robot terms are small either frame) was numerically ≈ `H_sys` — i.e. it was **measuring the real
  non-conservation**, not a frame mix.
- The frame diagnosis was *partially* right (the logged robot momentum IS structure-relative and ≈0
  at settle, confirmed), but fixing the frame does **not** close conservation. `I_s = 1777` needs no
  inflation — it was never the cause.

## A′.3 — Controller structure-relative invariant `c` does NOT close

`c(t) = h_w + L_com + r_com×m·v_com` (all structure-relative, per-tick logged), drift per DS window:

| window | n | ‖c_ref‖ | max‖Δc‖ | mean‖Δc‖ | max‖Δc‖/‖c‖ |
|---|---|---|---|---|---|
| post-step0 | 171 | 0.527 | 0.854 | 0.311 | **1.62** |
| post-step1 | 50 | 1.473 | 0.403 | 0.245 | 0.27 |
| post-step2 | 101 | 1.787 | 1.134 | 0.584 | 0.64 |
| post-step3 | 50 | 2.416 | 1.142 | 0.536 | 0.47 |
| terminal | 200 | 2.685 | 2.214 | 1.748 | 0.82 |

The controller's invariant drifts **27–162 %** of `‖c_ref‖` within DS windows — it does **not** close.
This is consistent with the A′.1 leak (the structure-relative bookkeeping inherits the same
non-conservation). So the controller's `c_simple` is not an exactly-conserved quantity in DS on this
run.

## Secondary — structure mass/inertia (model hygiene)

`body_mass = 7110 kg`, principal inertia `[1777, 1493, 597] kg·m²` (iquat `(0,.707,0,.707)`), radius
of gyration ≈ 0.43 m. The 7110-vs-"500 kg" MJCF comment is a **stale comment** (the mass is really
7110). This is **not** the conservation cause (the leak is ~0.2 N·m·s / DS-structured, not an
inertia scale error; `I_s=1777` is consistent — A′.2).

## Verdict & flags

1. **Conservation does NOT hold in the canonical plant (A′.1):** ~0.203 N·m·s leak, injected during
   DS, exactly conserved during SS. The J0 envelope-conservation backbone is leaky at the DS welds
   in this MuJoCo model. **First-order flag — resolve before J2.**
2. **The A′ frame-resolution hypothesis is refuted (A′.2):** no large robot co-rotation (≈0.03, not
   0.244); the original A.1 residual was the real leak, not a frame mix; `I_s=1777` is fine.
3. **Controller invariant `c` drifts 27–162 % in DS (A′.3)** — not exactly conserved on this run.
4. **Likely mechanism (to confirm, not asserted):** soft-weld / non-zero dock-gap (~5 mm) constraint
   stabilisation when the second weld engages in DS; SS (single steady weld) is leak-free.
5. **Lemma 2 (A.4)** remains untested cleanly (DS rates are low-SNR; deferred to SS regime per the
   prior note). Given (1), it should be tested plant-side, not controller-side (the QP enforces
   `L̇_com=M_λ·λ` by construction).
