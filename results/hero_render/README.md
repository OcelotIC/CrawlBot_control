# HERO-RENDER (v3) — figure set + provenance

**Rendering only, dynamics-neutral.** No controller/config/model *dynamics* change. Poses are a **replay of
stored state** from the frozen canonical run `results/figC25_addfive/sim_log.json` (τ_w,max = 2.5 canonical,
freeze `32aefaf`). The canonical MJCF **on disk is byte-identical** (CLAUDE.md rule 4 + freeze); v3 visual
scene elements are injected into an **in-memory** XML copy (all-primitive model → `from_xml_string`).

Scripts: `scripts/hero_render_v3.py` (injection + per-camera styling), `scripts/hero_render_v3_run.py`
(driver), `scripts/diag_v3_belt.py` (neutrality belt). Backend: MuJoCo 3.10 osmesa offscreen.

## Arm colour mapping (Table-III deliverable)

**Arm A = BLUE, Arm B = RED** — MJCF ground truth: `default class="arm_a"` rgba `0.2 0.4 0.8` (blue),
`class="arm_b"` rgba `0.8 0.3 0.2` (red); source comment line 150 reads *"ARM A (blue)"*.
*(This corrects the v2 report, which had the mapping inverted.)*

## v3 visual scene additions (injected, dynamics-neutral)

- `v3_rail_a`, `v3_rail_b` — slender rails under each anchor row (anchors sit on them).
- `v3_rwa_mount` — pedestal under the reaction-wheel cluster (no longer floating).
- All three: `contype=0`, `conaffinity=0`, `group=2` (visual); the structure body's explicit `<inertial>`
  (mass 7110) is untouched ⇒ **zero mass/inertia change by construction**.
- Dark gradient **skybox** asset (rgb1 `0.035 0.045 0.065` → rgb2 `0.005 0.006 0.010`; no stars/Earth) for cam B.

## Neutrality belt — `scripts/diag_v3_belt.py` → `results/j2_adjconv/v3_belt_result.json`

| check | result |
|---|---|
| dims nq/nv/nu, nbody, njnt | identical; ngeom +3 (the new geoms) |
| body mass/inertia/ipos/iquat, dof damping/armature, jnt range/axis | **identical** |
| existing-geom pos/size/type/contype/conaffinity | **identical** (new geoms excluded); new geoms `contype=conaffinity=0` |
| structure mass / inertia | 7110 / [1777,1493,597] — unchanged |
| **forward dynamics (qacc, qfrc_bias) at all 5 stored canonical states (t≤1.51s)** | **max diff = 0.00e+00** |
| **closed-loop 2 s replay, M0 (canonical) vs M1 (injected), 200 ticks** | **max\|Δqpos\| = 0, max\|Δqvel\| = 0 — BIT-IDENTICAL** |

**Verdict: dynamics-NEUTRAL.** The injection changes the plant nowhere: identical accelerations at every
logged canonical state, and a byte-identical 2 s closed-loop trajectory. (The visual model diverges from the
stored log *identically* to the canonical model under this lightweight belt harness — a runner-fidelity gap,
not an injection effect; since M1 ≡dynamics M0 and M0 ≡ stored [T4 Gate 0], M1 ≡ stored by transitivity.)

## Frames & ticks (Camera A — unchanged from v2)

F0 `initial`; F1–F6 = `frame_step{0-5}_2` (mid-swing max-clearance apex, ~33–36 mm) at
**t = 1.51 / 9.62 / 20.00 / 34.86 / 44.88 / 59.34 s**; F7 `final`.
Camera B: **B1 = F0**; **B2 = step 4** (`frame_step4_2`) — swing = **Arm B (red)** reaching, stance = Arm A (blue).

## Camera parameters (exact; MuJoCo free camera, z-up, anchor plane z=−1.775)

| camera | lookat [m] | distance | azimuth | elevation | fovy | render |
|---|---|---|---|---|---|---|
| **A** (wide stroboscopic, WHITE) | (1.10, −0.35, −2.05) | 6.5 | 35° | 18° | 27° | 3840×2400 |
| **B1** (docked, DARK) | (0.0, −0.30, −2.02) | 3.5 | 35° | 18° | 27° | 2880×2400 |
| **B2** (mid-swing, DARK) | (1.0, −0.30, −2.02) | 3.8 | 35° | 18° | 27° | 2880×2400 |

Camera A fixed across F0…F7 (progression axis = lower-left→upper-right diagonal). Cam B re-framed to the
per-pose robot bbox (~70 % fill) with the structure now visible.

## Backgrounds, lighting, rgba (reported)

- **Camera A — pure white** via segmentation matte (robot pixels over white); structure hidden (alpha 0);
  shadows/skybox/reflection/haze off; arms/anchors = v2 colours (arm_a blue, arm_b red, anchors blue). Unchanged.
- **Camera B — dark space**: dark gradient skybox (above); **low ambient (0.24) + overhead key light (0.85) +
  warm rim (0.75)**, headlight diffuse 0.40; shadows/reflection/haze off. Material re-balance for dark bg:
  - structure **light grey** rgba `(0.80, 0.82, 0.85)` (lighter than torso 0.6),
  - **Arm A (blue)** `(0.15, 0.35, 0.92)`, **Arm B (red)** `(0.90, 0.22, 0.15)` (saturated),
  - **anchors gold** `(1.0, 0.72, 0.12)` (distinct accent, not red/blue), upright posts r=0.030 h=0.050.

## Deliverables (same paths)

`frame_0.png … frame_7.png` · `composite_v1.png` · `contact_sheet.png` (all Camera A, white) ·
`sysview_docked.png` · `sysview_midswing.png` (Camera B, dark) · `render_meta.json` · this README.

*Optional Camera-A replay mp4: not produced (needs a per-tick qpos re-run); available on request.*
