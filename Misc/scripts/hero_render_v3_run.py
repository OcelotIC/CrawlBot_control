#!/usr/bin/env python3
"""HERO-RENDER v3 — produce all deliverables to results/hero_render/ (same paths).

Camera A (wide stroboscopic, WHITE bg, UNCHANGED from v2): F0..F7 + composite +
contact sheet — structure hidden, seg-matte over white, v2 arm/anchor colours.
Camera B (tight system view, DARK space bg): sysview_docked (F0) + sysview_midswing
(step 4, Arm B reaching) — light-grey structure on rails, gold anchor posts,
saturated arms, dark gradient skybox + directional key light.

Poses = replay of stored state (figC25_addfive snapshots). The canonical MJCF on
disk is untouched; visual scene elements are injected in-memory (dynamics-neutral,
proven by Misc/scripts/diag_v3_belt.py).

Run: MUJOCO_GL=osmesa PYTHONPATH=. python3 Misc/scripts/hero_render_v3_run.py
"""
import os
os.environ.setdefault('MUJOCO_GL', 'osmesa')
import json
import numpy as np
import mujoco
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Misc.scripts.hero_render import make_scene_option, free_cam, build_model, render_pose
from Misc.scripts.hero_render_v3 import build_v3_model, render_dark

OUT = 'results/hero_render'
os.makedirs(OUT, exist_ok=True)
SL = 'results/figC25_addfive/sim_log.json'
FOVY = 27.0
CAM_A = dict(lookat=[1.10, -0.35, -2.05], distance=6.5, azimuth=35.0, elevation=18.0)
CAM_B1 = dict(lookat=[0.0, -0.30, -2.02], distance=3.5, azimuth=35.0, elevation=18.0)
CAM_B2 = dict(lookat=[1.0, -0.30, -2.02], distance=3.8, azimuth=35.0, elevation=18.0)
A_W, A_H = 3840, 2400
B_W, B_H = 2880, 2400
LABELS = ['F0 initial', 'F1 step0 t=1.51s', 'F2 step1 t=9.62s', 'F3 step2 t=20.00s',
          'F4 step3 t=34.86s', 'F5 step4 t=44.88s', 'F6 step5 t=59.34s', 'F7 final']

sl = json.load(open(SL))
bylabel = {s[3]: np.asarray(s[1], float) for s in sl['snapshots']}
pose_labels = ['initial'] + [f'frame_step{k}_2' for k in range(6)] + ['final']
poses = [bylabel[l] for l in pose_labels]

# ── Camera A (white) — rendered from the UNMODIFIED v2 model so it is
#    byte-identical to v2 (the injected geoms would otherwise enter the
#    segmentation matte even while hidden). Camera A is unchanged. ──
mA, chA = build_model(offwidth=A_W, offheight=A_H, fovy=FOVY)
chA = ['camera A: v2 model (no v3 injection), white seg-matte — UNCHANGED from v2'] + chA
dA = mujoco.MjData(mA); opt = make_scene_option()
rA = mujoco.Renderer(mA, height=A_H, width=A_W)
cam = free_cam(**CAM_A)
framesA = []
for i, q in enumerate(poses):
    rgb, mask = render_pose(rA, mA, dA, q, cam, opt)
    framesA.append((rgb, mask))
    imageio.imwrite(f'{OUT}/frame_{i}.png', rgb)
canvas = np.full((A_H, A_W, 3), 255.0)
for i, a in [(0, 0.5)] + [(k, 0.30) for k in range(1, 7)] + [(7, 1.0)]:
    rgb, mask = framesA[i]
    canvas[mask] = a * rgb[mask].astype(float) + (1 - a) * canvas[mask]
imageio.imwrite(f'{OUT}/composite_v1.png', canvas.astype(np.uint8))
fig, axes = plt.subplots(2, 4, figsize=(20, 7.5))
for ax, i in zip(axes.ravel(), range(8)):
    ax.imshow(framesA[i][0]); ax.set_title(LABELS[i], fontsize=12); ax.axis('off')
plt.tight_layout(); plt.savefig(f'{OUT}/contact_sheet.png', dpi=130, facecolor='white'); plt.close()
print('camera A (white): frame_0..7 + composite_v1 + contact_sheet')

# ── Camera B (dark system view) ───────────────────────────────────────
mB, chB = build_v3_model(dark_bg=True, offwidth=B_W, offheight=B_H, fovy=FOVY)
dB = mujoco.MjData(mB)
rB = mujoco.Renderer(mB, height=B_H, width=B_W)
for cam_p, pose_i, name in [(CAM_B1, 0, 'sysview_docked'), (CAM_B2, 5, 'sysview_midswing')]:
    img = render_dark(rB, mB, dB, poses[pose_i], free_cam(**cam_p), opt)
    imageio.imwrite(f'{OUT}/{name}.png', img)
print('camera B (dark): sysview_docked + sysview_midswing')

# ── provenance / meta ─────────────────────────────────────────────────
json.dump(dict(
    source=SL, poses=pose_labels, apex_ticks=[1.51, 9.62, 20.00, 34.86, 44.88, 59.34],
    fovy=FOVY, cam_A=CAM_A, cam_B1=CAM_B1, cam_B2=CAM_B2,
    render_A=[A_W, A_H], render_B=[B_W, B_H],
    arm_colour_mapping={'Arm A': 'blue', 'Arm B': 'red',
                        'note': 'MJCF ground truth: class arm_a rgba 0.2 0.4 0.8 (blue), '
                                'class arm_b rgba 0.8 0.3 0.2 (red); comment line 150 "ARM A (blue)"'},
    B2_midswing='step 4 (frame_step4_2), swing = Arm B (red) reaching; stance = Arm A (blue)',
    camera_A_changes=chA, camera_B_changes=chB,
    background_A='pure white via segmentation matte', background_B='dark gradient skybox',
    visual_geoms_injected=['v3_rail_a', 'v3_rail_b', 'v3_rwa_mount'],
    dynamics_neutral='proven by Misc/scripts/diag_v3_belt.py (see v3_belt_result.json)'),
    open(f'{OUT}/render_meta.json', 'w'), indent=2)
print('wrote render_meta.json')
