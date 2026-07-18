#!/usr/bin/env python3
"""HERO-RENDER v4 — full-system dark framing + structured white camera A.

Same in-memory injected model as v3 (rails + RWA mount + dark skybox; canonical
MJCF on disk byte-identical; dynamics-neutral, proven by scripts/diag_v3_belt.py).
v4 render-time changes only:
  * HIDE the RGB reaction-wheel disc stack + its mount (no actuator geometry shown
    — AOCS actuation is generic RWA/CMG).
  * anchors enlarged/brightened to stay legible in the wide full-platform shot.
  * camera A now renders WITH the structure visible (white bg): structure static
    opaque base + robot poses ghosted (occlusion-correct via robot-only seg mask).

Arm colour mapping (MJCF ground truth): Arm A = BLUE (arm_a 0.2 0.4 0.8),
Arm B = RED (arm_b 0.8 0.3 0.2).
"""
import os
os.environ.setdefault('MUJOCO_GL', 'osmesa')
import numpy as np
import mujoco
from scripts.hero_render_v3 import inject_visual

WHITE = np.array([255, 255, 255], np.uint8)

ARM_A = (0.15, 0.35, 0.92, 1.0)          # Arm A blue (saturated)
ARM_B = (0.90, 0.22, 0.15, 1.0)          # Arm B red  (saturated)
STRUCT_DARK = (0.80, 0.82, 0.85, 1.0)    # light grey on dark bg
STRUCT_WHITE = (0.62, 0.65, 0.70, 1.0)   # cool grey on white bg (distinct from torso 0.6)
ANCHOR_GOLD = (1.0, 0.74, 0.16, 1.0)
RWA_NAMES = ('rwa_x', 'rwa_y', 'rwa_z')
MOUNT_NAME = 'v3_rwa_mount'


def _robot_body_ids(m):
    """Body ids of the robot (torso + all descendants)."""
    tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'torso')
    out = set()
    for b in range(m.nbody):
        p = b
        while p > 0:
            if p == tid:
                out.add(b); break
            p = m.body_parentid[p]
    return out


def robot_geom_mask_ids(m):
    """Set of geom ids that belong to the robot (for seg-based compositing)."""
    rb = _robot_body_ids(m)
    return set(g for g in range(m.ngeom) if m.geom_bodyid[g] in rb)


def build_v4_model(mode, offwidth=3840, offheight=2400, fovy=27.0, anchor_size=(0.030, 0.050)):
    """mode: 'dark' (system view) or 'white' (structured stroboscopic).
    Returns (model, changes)."""
    m = mujoco.MjModel.from_xml_string(inject_visual())
    m.vis.global_.offwidth = offwidth
    m.vis.global_.offheight = offheight
    m.vis.global_.fovy = fovy
    ch = ['injected model identical to v3 (rails + mount + skybox); dynamics-neutral (v3 belt)']

    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'structure')
    rb = _robot_body_ids(m)

    # hide RWA disc stack + mount (no actuator geometry)
    hide_bodies = set(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm) for nm in RWA_NAMES)
    for g in range(m.ngeom):
        gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        if m.geom_bodyid[g] in hide_bodies or gn == MOUNT_NAME:
            m.geom_rgba[g, 3] = 0.0
    ch.append('HIDDEN: rwa_x/y/z disc geoms + v3_rwa_mount (alpha 0) — no actuator geometry')

    struct_rgba = STRUCT_DARK if mode == 'dark' else STRUCT_WHITE
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == sid and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) != MOUNT_NAME:
            if m.geom_rgba[g, 3] > 0:            # don't un-hide the mount
                m.geom_rgba[g] = struct_rgba
    if mode == 'dark':
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in rb:
                nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
        # arm colours
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ''
            if nm.endswith('_a') or nm == 'tool_a':
                m.geom_rgba[g] = ARM_A
            elif nm.endswith('_b') or nm == 'tool_b':
                m.geom_rgba[g] = ARM_B
        m.vis.headlight.ambient[:] = [0.24, 0.24, 0.26]
        m.vis.headlight.diffuse[:] = [0.40, 0.40, 0.40]
        if m.nlight >= 2:
            m.light_diffuse[0] = [0.85, 0.85, 0.90]
            m.light_diffuse[1] = [0.75, 0.72, 0.66]
        ch.append(f'dark: structure {STRUCT_DARK[:3]}, ArmA{ARM_A[:3]}, ArmB{ARM_B[:3]}')
    else:
        ch.append(f'white: structure {STRUCT_WHITE[:3]} (cool grey), arms = model colours '
                  f'(arm_a blue, arm_b red)')

    # anchors: bright gold posts, legible in the wide shot
    for i in range(m.nsite):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i)
        if nm and nm.startswith('anchor_'):
            m.site_rgba[i] = ANCHOR_GOLD
            m.site_size[i] = [anchor_size[0], anchor_size[1], 0.0]
    ch.append(f'anchors gold {ANCHOR_GOLD[:3]} posts r={anchor_size[0]} h={anchor_size[1]}')
    return m, ch


def free_cam(lookat, distance, azimuth, elevation):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = lookat; c.distance = distance
    c.azimuth = azimuth; c.elevation = elevation
    return c


def _kill_flags(sc, skybox):
    sc.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    sc.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    sc.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 0
    sc.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1 if skybox else 0


def render_dark(renderer, m, d, qpos, cam, opt):
    d.qpos[:] = qpos; mujoco.mj_forward(m, d)
    renderer.update_scene(d, camera=cam, scene_option=opt)
    _kill_flags(renderer.scene, skybox=True)
    return renderer.render().copy()


def render_rgb_seg(renderer, m, d, qpos, cam, opt):
    """Return (rgb, seg_geomid) for the full scene, skybox off (white matte use)."""
    d.qpos[:] = qpos; mujoco.mj_forward(m, d)
    renderer.update_scene(d, camera=cam, scene_option=opt)
    _kill_flags(renderer.scene, skybox=False)
    rgb = renderer.render().copy()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(d, camera=cam, scene_option=opt)
    seg = renderer.render()[:, :, 0].copy()
    renderer.disable_segmentation_rendering()
    return rgb, seg


def matte_white(rgb, seg, geom_ids=None):
    """Matte over white. If geom_ids given, only those geoms; else any geom (>=0)."""
    if geom_ids is None:
        mask = seg >= 0
    else:
        mask = np.isin(seg, list(geom_ids))
    out = np.empty_like(rgb); out[:] = WHITE; out[mask] = rgb[mask]
    return out, mask
