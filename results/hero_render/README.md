# HERO-RENDER v2 — figure set + provenance

**Rendering only.** No controller/config/model dynamics change. Poses are a **replay of stored state**
from the frozen canonical run `results/figC25_addfive/sim_log.json` (its snapshots) — the exact qpos
recorded during the τ_w,max=2.5 canonical traversal (freeze `32aefaf`; this run's traversal is bit-identical
to a re-run, proven in T4/T4b Gate 0). No re-run was needed.

Scripts: `scripts/hero_render.py` (render helpers), `scripts/hero_render_run.py` (driver). Backend: MuJoCo
3.10 osmesa offscreen.

## Frames & ticks (Camera A)

| frame | snapshot | tick | note |
|---|---|---|---|
| F0 | `initial` | t=0.00 s | both docked (start) |
| F1 | `frame_step0_2` | **t=1.51 s** | step 0 mid-swing apex |
| F2 | `frame_step1_2` | **t=9.62 s** | step 1 mid-swing apex |
| F3 | `frame_step2_2` | **t=20.00 s** | step 2 mid-swing apex |
| F4 | `frame_step3_2` | **t=34.86 s** | step 3 mid-swing apex |
| F5 | `frame_step4_2` | **t=44.88 s** | step 4 mid-swing apex |
| F6 | `frame_step5_2` | **t=59.34 s** | step 5 mid-swing apex |
| F7 | `final` | t=84.64 s | both docked (end) |

The mid-swing frame is the **tick of max swing-gripper clearance**: `frame_stepN_2` is the apex of the 5
stored swing frames per step (perpendicular gripper offset from the lift-off→target chord ≈ **33–36 mm**,
vs ≈16–25 mm at the flanking frames). Camera B: **B1 = F0 pose**, **B2 = F5 (step-4 mid-swing)**.

## Camera parameters (exact, MuJoCo free camera; z-up, anchor plane z=−1.775)

| camera | lookat [m] | distance [m] | azimuth [°] | elevation [°] | fovy [°] |
|---|---|---|---|---|---|
| **A (wide, fixed)** | (1.10, −0.35, −2.05) | 6.5 | 35 | 18 | 27 |
| **B1 (docked)** | (−0.048, −0.385, −2.103) | 3.6 | 35 | 18 | 27 |
| **B2 (mid-swing)** | (0.969, −0.377, −2.109) | 4.3 | 35 | 18 | 27 |

Camera A is **fixed across F0…F7** (the robot progresses +X through the fixed frame; progression axis is the
lower-left→upper-right diagonal). B1/B2 lookat = per-pose robot-geom bbox centre; distance set for ~70 % fill.

## Render settings

- Offscreen PNG; **Camera A 3840×2400**, **Camera B 2880×2400** (both ≥2400 px).
- **Pure white background** via a **segmentation matte**: render RGB + a segmentation pass; robot/anchor
  pixels (`seg geom-id ≥ 0`) are composited onto a white canvas, background pixels set to 255,255,255.
- **Shadows / skybox / reflection / haze OFF** (scene render flags). **No ground plane** — none exists in the
  model (0 plane geoms).

## Composite matte method (`composite_v1.png`)

Per-frame segmentation masks drive alpha compositing over a white canvas, painted back-to-front:
`canvas[mask] = α·rgb[mask] + (1−α)·canvas[mask]`, with **F0 α=0.50 (light)**, **F1…F6 α=0.30 (ghosted)**,
**F7 α=1.00 (opaque, on top)**. The masks give clean per-pixel robot/background separation (no depth matte
needed at this resolution).

## rgba / visibility changes (rendering-only, reported)

- **structure geoms → alpha 0** (hidden) so the hero reads as robot-on-white (the structure is the 4.8 m
  platform; "no ground plane"/"white background" intent). Dynamics untouched — this is a render-time rgba edit.
- **anchor_* sites → rgba (0.15, 0.45, 0.9, 1) + size ≥ 0.03** so the docking anchors read as small blue path
  markers (needed for Camera-B "stance + target anchor visible"). No robot material rgba was changed — arm-a
  (red) / arm-b (blue) / torso (grey) are already distinguishable in the source model.

## Deliverables

`frame_0.png … frame_7.png` · `composite_v1.png` · `contact_sheet.png` · `sysview_docked.png` ·
`sysview_midswing.png` · `render_meta.json` (machine-readable params).

*Optional fixed-camera replay mp4 (Camera A): not produced — it needs a per-tick qpos re-run + a few-hundred
frame render; available on request.*
