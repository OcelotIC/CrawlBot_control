#!/usr/bin/env python3
"""HERO-RENDER v2 — offscreen MuJoCo rendering of the frozen canonical poses.

Rendering only: no controller/config/model dynamics change. Poses are REPLAY of
stored state (figC25_addfive/sim_log.json snapshots): F0 initial, F1..F6 the
mid-swing max-clearance frame (frame_stepN_2) of each step, F7 final.

Pure white background via a segmentation matte (robot pixels composited over
white). Structure geoms hidden (alpha->0) so the hero shows the robot; anchor
sites shown as markers for path/system context. Shadows/skybox/reflection off.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'osmesa')
import numpy as np
import mujoco

MJCF = 'models/VISPA_crawling_rwa3.xml'
WHITE = np.array([255, 255, 255], np.uint8)


def build_model(show_anchors=True, hide_structure=True, anchor_rgba=(0.15, 0.45, 0.9, 1.0),
                offwidth=3600, offheight=2400, fovy=27.0):
    """Load model; hide structure; style anchor sites as visible markers.
    Returns (model, list-of-rgba-changes)."""
    m = mujoco.MjModel.from_xml_path(MJCF)
    m.vis.global_.offwidth = offwidth      # offscreen framebuffer >= render size
    m.vis.global_.offheight = offheight
    m.vis.global_.fovy = fovy              # free-camera vertical FOV (long focal)
    changes = []
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'structure')
    if hide_structure:
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == sid:
                m.geom_rgba[g, 3] = 0.0
        changes.append('structure geoms alpha->0 (hidden for white-bg hero)')
    # anchor sites: enlarge + color so they read as small path markers
    if show_anchors:
        for i in range(m.nsite):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i)
            if nm and nm.startswith('anchor_'):
                m.site_rgba[i] = anchor_rgba
                m.site_size[i] = np.maximum(m.site_size[i], 0.03)
        changes.append(f'anchor_* sites rgba={anchor_rgba}, size>=0.03 (path markers)')
    return m, changes


def make_scene_option(show_anchors=True):
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1
    # sites on only for anchors (their group); keep default groups
    return opt


def render_pose(renderer, model, data, qpos, cam, opt, seg=True):
    """Render qpos with free camera `cam`; return RGB composited over white."""
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam, scene_option=opt)
    # kill shadows / skybox / reflection / haze on the built scene
    sc = renderer.scene
    sc.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    sc.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
    sc.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    sc.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 0
    rgb = renderer.render().copy()
    if not seg:
        return rgb, None
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=cam, scene_option=opt)
    segimg = renderer.render()
    renderer.disable_segmentation_rendering()
    mask = segimg[:, :, 0] >= 0            # geom id >=0 => robot/anchor pixel
    out = np.empty_like(rgb)
    out[:] = WHITE
    out[mask] = rgb[mask]
    return out, mask


def free_cam(lookat, distance, azimuth, elevation):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = lookat
    c.distance = distance
    c.azimuth = azimuth
    c.elevation = elevation
    return c
