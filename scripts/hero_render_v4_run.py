#!/usr/bin/env python3
"""HERO-RENDER v4 — produce deliverables to results/hero_render/.

Camera A (WHITE, structured): frame_0..7 (pose on the visible structure) +
composite_v1 (structure static base + robot poses ghosted, occlusion-correct via
robot-only segmentation masks + a robot-free structure base) + contact sheet.
Camera B (DARK): sysview_full (full-platform wide, robot small = scale ratio) +
sysview_gripper_inset (tight docked-gripper close-up). RWA discs+mount hidden.

Run: MUJOCO_GL=osmesa PYTHONPATH=. python3 scripts/hero_render_v4_run.py
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
from scripts.hero_render_v4 import (build_v4_model, free_cam, render_dark,
                                    render_rgb_seg, matte_white, robot_geom_mask_ids)

OUT = 'results/hero_render'
os.makedirs(OUT, exist_ok=True)
SL = 'results/figC25_addfive/sim_log.json'
FOVY = 27.0
A_W, A_H = 3840, 2400
BW_W, BW_H = 3600, 2250            # camera B wide
BI_W, BI_H = 2200, 2200           # gripper inset
CAM_A = dict(lookat=[1.10, -0.35, -2.05], distance=6.5, azimuth=35.0, elevation=18.0)
CAM_WIDE = dict(lookat=[0.15, 0.10, -1.98], distance=10.0, azimuth=36.0, elevation=20.0)
INSET = dict(distance=0.90, azimuth=42.0, elevation=24.0)   # lookat = docked gripper_a @step4
LABELS = ['F0 initial', 'F1 step0 t=1.51s', 'F2 step1 t=9.62s', 'F3 step2 t=20.00s',
          'F4 step3 t=34.86s', 'F5 step4 t=44.88s', 'F6 step5 t=59.34s', 'F7 final']

sl = json.load(open(SL))
bylabel = {s[3]: np.asarray(s[1], float) for s in sl['snapshots']}
pose_labels = ['initial'] + [f'frame_step{k}_2' for k in range(6)] + ['final']
poses = [bylabel[l] for l in pose_labels]

GROBOT = 4                          # geomgroup used to toggle the robot on/off


def scene_opt(robot_on):
    o = mujoco.MjvOption()
    o.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1
    o.geomgroup[GROBOT] = 1 if robot_on else 0
    return o


# ── Camera A (white, structured) ──────────────────────────────────────
mA, chA = build_v4_model('white', offwidth=A_W, offheight=A_H, fovy=FOVY,
                         anchor_size=(0.028, 0.05))
robot_ids = robot_geom_mask_ids(mA)
for g in robot_ids:
    mA.geom_group[g] = GROBOT       # put robot in a togglable group
dA = mujoco.MjData(mA)
rA = mujoco.Renderer(mA, height=A_H, width=A_W)
cam = free_cam(**CAM_A)
opt_on, opt_off = scene_opt(True), scene_opt(False)

# structure-only base (robot hidden -> no holes behind ghosts)
brgb, bseg = render_rgb_seg(rA, mA, dA, poses[0], cam, opt_off)
base, _ = matte_white(brgb, bseg)   # structure (+ anchors) on white, robot absent

# individual frames: full pose on the structure (white)
framesA = []
for i, q in enumerate(poses):
    rgb, seg = render_rgb_seg(rA, mA, dA, q, cam, opt_on)
    out, _ = matte_white(rgb, seg)
    imageio.imwrite(f'{OUT}/frame_{i}.png', out)
    rmask = np.isin(seg, list(robot_ids))
    framesA.append((rgb, rmask))
# composite: structure base + ghosted robots (F0 light, F1-6 0.30, F7 opaque)
canvas = base.astype(float)
for i, a in [(0, 0.5)] + [(k, 0.30) for k in range(1, 7)] + [(7, 1.0)]:
    rgb, rmask = framesA[i]
    canvas[rmask] = a * rgb[rmask].astype(float) + (1 - a) * canvas[rmask]
imageio.imwrite(f'{OUT}/composite_v1.png', canvas.astype(np.uint8))
# contact sheet from the saved frame files (avoid re-render)
fig, axes = plt.subplots(2, 4, figsize=(20, 7.5))
for ax, i in zip(axes.ravel(), range(8)):
    ax.imshow(imageio.imread(f'{OUT}/frame_{i}.png')); ax.set_title(LABELS[i], fontsize=12); ax.axis('off')
plt.tight_layout(); plt.savefig(f'{OUT}/contact_sheet.png', dpi=130, facecolor='white'); plt.close()
print('camera A (white, structured): frame_0..7 + composite_v1 + contact_sheet')

# ── Camera B wide (dark, full system) ─────────────────────────────────
mW, chW = build_v4_model('dark', offwidth=BW_W, offheight=BW_H, fovy=FOVY,
                         anchor_size=(0.050, 0.085))
dW = mujoco.MjData(mW); optW = mujoco.MjvOption(); optW.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1
rW = mujoco.Renderer(mW, height=BW_H, width=BW_W)
imageio.imwrite(f'{OUT}/sysview_full.png',
                render_dark(rW, mW, dW, poses[5], free_cam(**CAM_WIDE), optW))
print('camera B wide: sysview_full')

# ── Gripper inset (dark, tight) ───────────────────────────────────────
mI, chI = build_v4_model('dark', offwidth=BI_W, offheight=BI_H, fovy=FOVY,
                         anchor_size=(0.016, 0.075))
dI = mujoco.MjData(mI); optI = mujoco.MjvOption(); optI.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1
dI.qpos[:] = poses[5]; mujoco.mj_forward(mI, dI)   # step-4: stance gripper_a docked
gp = dI.site_xpos[mujoco.mj_name2id(mI, mujoco.mjtObj.mjOBJ_SITE, 'gripper_a')].copy()
la_inset = [gp[0], gp[1], gp[2] - 0.05]
imageio.imwrite(f'{OUT}/sysview_gripper_inset.png',
                render_dark(rI := mujoco.Renderer(mI, height=BI_H, width=BI_W),
                            mI, dI, poses[5], free_cam(la_inset, **INSET), optI))
print('camera B inset: sysview_gripper_inset')

# ── meta ──────────────────────────────────────────────────────────────
json.dump(dict(
    source=SL, poses=pose_labels, fovy=FOVY,
    cam_A=CAM_A, cam_B_wide=CAM_WIDE,
    inset=dict(lookat=[round(float(x), 4) for x in la_inset], pose='step4 docked gripper_a', **INSET),
    render_A=[A_W, A_H], render_B_wide=[BW_W, BW_H], render_inset=[BI_W, BI_H],
    arm_colour_mapping={'Arm A': 'blue (0.2 0.4 0.8)', 'Arm B': 'red (0.8 0.3 0.2)'},
    B_wide_pose='step 4 mid-swing (frame_step4_2); swing=Arm B (red)',
    changes_A=chA, changes_B_wide=chW, changes_inset=chI,
    hidden='rwa_x/y/z disc geoms + v3_rwa_mount (no actuator geometry)',
    dynamics_neutral='injected model identical to v3; belt v3_belt_result.json'),
    open(f'{OUT}/render_meta.json', 'w'), indent=2)
print('wrote render_meta.json')
