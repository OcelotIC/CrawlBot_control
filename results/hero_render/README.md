# HERO-RENDER (v4) — figure set + provenance

**Rendering only, dynamics-neutral.** No controller/config/model *dynamics* change. Poses are a **replay of
stored state** from the frozen canonical run `results/figC25_addfive/sim_log.json` (τ_w,max = 2.5 canonical,
freeze `32aefaf`). The canonical MJCF **on disk is byte-identical** (CLAUDE.md rule 4 + freeze); the v3 visual
scene elements are injected into an **in-memory** XML copy (all-primitive model → `from_xml_string`).

Scripts: `Misc/scripts/hero_render_v4.py` (injection re-exported from v3 + v4 render-time styling: RWA hidden,
structured white camera A, full-system dark camera B), `Misc/scripts/hero_render_v4_run.py` (driver),
`Misc/scripts/diag_v3_belt.py` (neutrality belt — still valid, see below). Backend: MuJoCo 3.10 osmesa offscreen.

## What changed v3 → v4 (render-time only)

1. **Full-system dark view** (`sysview_full`): camera B pulled back to frame the **entire platform + the whole
   robot in one shot** — the robot is deliberately *small* (scale ratio). Beam runs the frame diagonal; robot
   sits on a rule-of-thirds line (lower-right), not centred.
2. **No actuator geometry**: the RGB reaction-wheel disc stack (`rwa_x/y/z`) **and** its mount (`v3_rwa_mount`)
   are hidden (alpha 0). AOCS actuation is a generic RWA/CMG class — no specific wheel hardware is depicted.
3. **Anchor rail legible at scale**: gold anchor posts enlarged for the wide shot (sizes per-camera, below).
4. **Gripper inset** (`sysview_gripper_inset`): one tight close-up of a docked gripper on its interface.
5. **Camera A now renders ON the structured scene** (white bg): structure is a visible opaque base and the
   robot poses are ghosted over it, occlusion-correct (see compositing below). v3's camera A was structure-hidden.

**Neutrality is unaffected:** v4 changes only render-time rgba / camera / geom-group visibility. The injected
XML (`inject_visual()`) is **md5-identical to v3** (`97cd24cc7eb68e639bb302dc2724bdf9`) — the v3 belt below
applies verbatim.

## Arm colour mapping (Table-III deliverable)

**Arm A = BLUE, Arm B = RED** — MJCF ground truth: `default class="arm_a"` rgba `0.2 0.4 0.8` (blue),
`class="arm_b"` rgba `0.8 0.3 0.2` (red); source comment line 150 reads *"ARM A (blue)"*.
*(This corrects the v2 report, which had the mapping inverted.)* In the mid-swing frames the reaching (swing)
arm alternates by step: step 3 → **Arm A (blue)**, step 4 → **Arm B (red)**.

## v3 visual scene additions (injected, dynamics-neutral) — carried into v4 unchanged

- `v3_rail_a`, `v3_rail_b` — slender rails under each anchor row (anchors sit on them).
- `v3_rwa_mount` — pedestal under the reaction-wheel cluster (**hidden in v4** along with the wheels).
- All three: `contype=0`, `conaffinity=0`, `group=2` (visual); the structure body's explicit `<inertial>`
  (mass 7110) is untouched ⇒ **zero mass/inertia change by construction**.
- Dark gradient **skybox** asset (rgb1 `0.035 0.045 0.065` → rgb2 `0.005 0.006 0.010`; no stars/Earth) for cam B.

## Neutrality belt — `Misc/scripts/diag_v3_belt.py` → `results/j2_adjconv/v3_belt_result.json`

Belt runs on `inject_visual()`, which v4 re-exports **byte-for-byte** from v3 (md5
`97cd24cc7eb68e639bb302dc2724bdf9`). The v4 hero edits (rgba, camera, `geom_group` toggling, alpha-0 hiding)
are applied to the compiled `MjModel` **after** injection and touch no dynamics field, so the belt below covers
v4 unchanged.

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

## Frames & ticks

- **Camera A** (structured white, stroboscopic): F0 `initial`; F1–F6 = `frame_step{0-5}_2` (mid-swing
  max-clearance apex, ~33–36 mm) at **t = 1.51 / 9.62 / 20.00 / 34.86 / 44.88 / 59.34 s**; F7 `final`.
  Camera fixed across F0…F7 (progression axis = lower-left→upper-right diagonal).
- **Camera B wide** (`sysview_full`): pose = **step 4** (`frame_step4_2`), swing = **Arm B (red)**.
- **Gripper inset** (`sysview_gripper_inset`): pose = **step 4**, lookat = docked `gripper_a` site.

## Camera parameters (exact; MuJoCo free camera, z-up, anchor plane z=−1.775)

| camera | lookat [m] | distance | azimuth | elevation | fovy | render |
|---|---|---|---|---|---|---|
| **A** (structured stroboscopic, WHITE) | (1.10, −0.35, −2.05) | 6.5 | 35° | 18° | 27° | 3840×2400 |
| **B wide** (full system, DARK) | (0.15, 0.10, −1.98) | 10.0 | 36° | 20° | 27° | 3600×2250 |
| **gripper inset** (DARK) | (1.185, 0.300, −1.827) `=gripper_a@step4` | 0.90 | 42° | 24° | 27° | 2200×2200 |

## Compositing (Camera A, structured)

The composite must not punch robot-shaped holes in the structure. Procedure:
1. Render a **structure-only base** (robot geoms toggled off via `geom_group[GROBOT]=0`) → matte over white.
2. For each pose, render the full scene + a **robot-only segmentation mask** (`np.isin(seg, robot_geom_ids)`).
3. Alpha-blend each pose's robot pixels onto the base (**F0 α=0.5, F1–F6 α=0.30, F7 α=1.0**), so the docked
   end pose is opaque and the swing history ghosts. Occlusion is correct because only robot pixels blend and
   the structure base is already opaque behind them.

## Backgrounds, lighting, rgba (reported)

- **Camera A — pure white.** Structure now **visible** (cool grey `(0.62, 0.65, 0.70)`, distinct from torso 0.6);
  arms = model colours (arm_a blue `0.2 0.4 0.8`, arm_b red `0.8 0.3 0.2`); anchors **gold** `(1.0, 0.74, 0.16)`
  posts r=0.028 h=0.05; RWA + mount hidden. Shadows/skybox/reflection/haze off; white via segmentation matte.
- **Camera B (wide + inset) — dark space**: dark gradient skybox (above); low ambient (0.24) + overhead key
  (0.85) + warm rim (0.75), headlight diffuse 0.40; shadows/reflection/haze off. Material re-balance for dark bg:
  - structure **light grey** `(0.80, 0.82, 0.85)`,
  - **Arm A (blue)** `(0.15, 0.35, 0.92)`, **Arm B (red)** `(0.90, 0.22, 0.15)` (saturated),
  - **anchors gold** `(1.0, 0.74, 0.16)` posts — **wide**: r=0.050 h=0.085 (legible at full-platform scale);
    **inset**: r=0.016 h=0.075. RWA disc stack + mount hidden (no actuator geometry).

## Deliverables (paths)

`frame_0.png … frame_7.png` · `composite_v1.png` · `contact_sheet.png` (all Camera A, structured white) ·
`sysview_full.png` (Camera B, full-system dark) · `sysview_gripper_inset.png` (docked-gripper close-up, dark) ·
`render_meta.json` · this README.

*Superseded v3 tight system views `sysview_docked.png` / `sysview_midswing.png` removed — they showed the
now-hidden RWA disc stack. Optional Camera-A replay mp4: not produced (needs a per-tick qpos re-run); on request.*
