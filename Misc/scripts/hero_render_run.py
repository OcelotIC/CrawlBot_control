#!/usr/bin/env python3
"""HERO-RENDER v2 — produce all committed deliverables to results/hero_render/.

Rendering only (no dynamics/config/model change). Poses = REPLAY of stored
state from figC25_addfive/sim_log.json snapshots.

Camera A (wide stroboscopic, FIXED): F0 initial, F1..F6 mid-swing apex
(frame_stepN_2), F7 final -> frame_0..7.png, composite_v1.png, contact_sheet.png.
Camera B (tight system view): B1 = F0 pose, B2 = step-4 mid-swing ->
sysview_docked.png, sysview_midswing.png.

Run: MUJOCO_GL=osmesa PYTHONPATH=. python3 Misc/scripts/hero_render_run.py
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
from Misc.scripts.hero_render import build_model, make_scene_option, render_pose, free_cam

OUT = 'results/hero_render'
os.makedirs(OUT, exist_ok=True)
SL = 'results/figC25_addfive/sim_log.json'

# ── FIXED camera parameters (report these) ────────────────────────────
FOVY = 27.0
CAM_A = dict(lookat=[1.10, -0.35, -2.05], distance=6.5, azimuth=35.0, elevation=18.0)
CAM_B1 = dict(lookat=[-0.048, -0.385, -2.103], distance=3.6, azimuth=35.0, elevation=18.0)
CAM_B2 = dict(lookat=[0.969, -0.377, -2.109], distance=4.3, azimuth=35.0, elevation=18.0)
A_W, A_H = 3840, 2400          # camera A render size (>=2400 px)
B_W, B_H = 2880, 2400          # camera B render size
LABELS = ['F0 initial', 'F1 step0 t=1.51s', 'F2 step1 t=9.62s', 'F3 step2 t=20.00s',
          'F4 step3 t=34.86s', 'F5 step4 t=44.88s', 'F6 step5 t=59.34s', 'F7 final']

# ── poses: replay of stored state ─────────────────────────────────────
sl = json.load(open(SL))
bylabel = {s[3]: np.asarray(s[1], float) for s in sl['snapshots']}
pose_labels = ['initial'] + [f'frame_step{k}_2' for k in range(6)] + ['final']
poses = [bylabel[l] for l in pose_labels]

m, changes = build_model(offwidth=A_W, offheight=A_H, fovy=FOVY)
d = mujoco.MjData(m)
opt = make_scene_option()
print('rgba/vis changes:', changes)


def render_A(renderer):
    cam = free_cam(**CAM_A)
    frames = []
    for i, q in enumerate(poses):
        rgb, mask = render_pose(renderer, m, d, q, cam, opt)
        frames.append((rgb, mask))
        imageio.imwrite(f'{OUT}/frame_{i}.png', rgb)      # each pose opaque on white
    print(f'wrote frame_0..7.png ({A_W}x{A_H})')
    # composite: F0 light(0.5), F1..F6 ghost(0.30), F7 opaque(1.0)
    canvas = np.full((A_H, A_W, 3), 255.0)
    for i, a in [(0, 0.5)] + [(k, 0.30) for k in range(1, 7)] + [(7, 1.0)]:
        rgb, mask = frames[i]
        canvas[mask] = a * rgb[mask].astype(float) + (1 - a) * canvas[mask]
    imageio.imwrite(f'{OUT}/composite_v1.png', canvas.astype(np.uint8))
    print('wrote composite_v1.png')
    # contact sheet
    fig, axes = plt.subplots(2, 4, figsize=(20, 7.5))
    for ax, i in zip(axes.ravel(), range(8)):
        ax.imshow(frames[i][0]); ax.set_title(LABELS[i], fontsize=12); ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{OUT}/contact_sheet.png', dpi=130, facecolor='white')
    plt.close()
    print('wrote contact_sheet.png')


def render_B(renderer):
    for cam_p, pose_i, name in [(CAM_B1, 0, 'sysview_docked'),
                                (CAM_B2, 5, 'sysview_midswing')]:
        cam = free_cam(**cam_p)
        rgb, _ = render_pose(renderer, m, d, poses[pose_i], cam, opt)
        imageio.imwrite(f'{OUT}/{name}.png', rgb)
        print(f'wrote {name}.png ({B_W}x{B_H})')


rA = mujoco.Renderer(m, height=A_H, width=A_W)
render_A(rA)
rB = mujoco.Renderer(m, height=B_H, width=B_W)
render_B(rB)

# provenance
json.dump(dict(source=SL, poses=pose_labels, apex_ticks=[1.51, 9.62, 20.00, 34.86, 44.88, 59.34],
               fovy=FOVY, cam_A=CAM_A, cam_B1=CAM_B1, cam_B2=CAM_B2,
               render_A=[A_W, A_H], render_B=[B_W, B_H], rgba_vis_changes=changes,
               background='pure white via segmentation matte (robot pixels over white)',
               flags_off=['shadow', 'skybox', 'reflection', 'haze'], ground_plane='none in model'),
          open(f'{OUT}/render_meta.json', 'w'), indent=2)
print('wrote render_meta.json')
