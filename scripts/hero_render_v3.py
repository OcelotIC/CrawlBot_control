#!/usr/bin/env python3
"""HERO-RENDER v3 — visual scene upgrade (dynamics-neutral) + dark-bg system view.

The canonical MJCF on disk is NEVER modified (freeze + CLAUDE.md rule 4: one
canonical MJCF, variations applied programmatically). Visual scene elements are
injected into an in-memory XML copy (ElementTree) and compiled via
from_xml_string (the model is all-primitive, no external assets):

  * anchor RAILS (two slender boxes under each anchor row; anchors sit on them)
  * RWA MOUNT (pedestal under the reaction-wheel cluster)
    -- all with contype=0, conaffinity=0, group=2 (visual); the structure body's
       explicit <inertial> (mass 7110) is untouched -> zero mass/inertia change.
  * a dark gradient SKYBOX asset (near-black blue-grey; no stars/Earth) for cam B.

Colours/lights are set at RENDER time (runtime model fields), per camera.
Arm colour mapping (MJCF ground truth): Arm A = BLUE (class arm_a rgba 0.2 0.4 0.8),
Arm B = RED (class arm_b rgba 0.8 0.3 0.2).
"""
import os
os.environ.setdefault('MUJOCO_GL', 'osmesa')
import numpy as np
import mujoco

MJCF = 'models/VISPA_crawling_rwa3.xml'
WHITE = np.array([255, 255, 255], np.uint8)

# String injection (the MJCF header comment contains "+--", which strict XML
# parsers reject but MuJoCo accepts — so we anchor on unique literal markers).
_RAILS = '''
      <!-- v3 visual-only additions: anchor rails + RWA mount.
           contype=0/conaffinity=0 => excluded from collision; the structure
           body's explicit <inertial> is untouched => zero mass/inertia change. -->
      <geom class="structure" type="box" size="2.15 0.045 0.018" pos="0 0.3 0.007" contype="0" conaffinity="0" group="2" name="v3_rail_a"/>
      <geom class="structure" type="box" size="2.15 0.045 0.018" pos="0 -0.3 0.007" contype="0" conaffinity="0" group="2" name="v3_rail_b"/>
      <geom class="structure" type="box" size="0.10 0.10 0.014" pos="0 0 0.039" contype="0" conaffinity="0" group="2" name="v3_rwa_mount"/>'''
_GEOM_MARKER = '<geom class="structure" type="box" size="0.02 0.4 0.04" pos="1.6 0 0"/>'
_SKYBOX = '''  <asset>
    <texture name="v3_skybox" type="skybox" builtin="gradient" rgb1="0.035 0.045 0.065" rgb2="0.005 0.006 0.010" width="256" height="256"/>
  </asset>

'''
_WB_MARKER = '<worldbody>'


def inject_visual(mjcf=MJCF):
    """Return a modified XML string (canonical file on disk untouched)."""
    with open(mjcf) as f:
        xml = f.read()
    assert xml.count(_GEOM_MARKER) == 1 and xml.count(_WB_MARKER) == 1, 'marker not unique'
    xml = xml.replace(_GEOM_MARKER, _GEOM_MARKER + _RAILS, 1)
    xml = xml.replace('  ' + _WB_MARKER, _SKYBOX + '  ' + _WB_MARKER, 1)
    return xml


# arm/structure/anchor palettes
ARM_A_SAT = (0.15, 0.35, 0.92, 1.0)     # Arm A (blue), saturated for dark bg
ARM_B_SAT = (0.90, 0.22, 0.15, 1.0)     # Arm B (red), saturated for dark bg
STRUCT_LIGHT = (0.80, 0.82, 0.85, 1.0)  # light grey, lighter than torso (0.6)
ANCHOR_GOLD = (1.0, 0.72, 0.12, 1.0)    # distinct accent (not red/blue)
ANCHOR_BLUE = (0.15, 0.45, 0.90, 1.0)   # v2 camera-A anchor colour (unchanged)


def build_v3_model(dark_bg, offwidth=3840, offheight=2400, fovy=27.0):
    """Compile the injected model and apply per-camera colours/lights.
    Returns (model, changes)."""
    m = mujoco.MjModel.from_xml_string(inject_visual())
    m.vis.global_.offwidth = offwidth
    m.vis.global_.offheight = offheight
    m.vis.global_.fovy = fovy
    changes = ['injected: v3_rail_a/b (anchor rails) + v3_rwa_mount, '
               'contype=0/conaffinity=0/group=2, structure <inertial> untouched',
               'injected: dark gradient skybox rgb1=0.035 0.045 0.065 / rgb2=0.005 0.006 0.010']

    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'structure')

    def geoms_of(body_name):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]

    def arm_geoms(suffix):
        out = []
        for b in range(m.nbody):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
            if nm and (nm.endswith(f'_{suffix}') or nm == f'tool_{suffix}'):
                out += [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
        return out

    if dark_bg:
        # structure light grey (beam + cross-beams + rails + mount)
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == sid:
                m.geom_rgba[g] = STRUCT_LIGHT
        for g in arm_geoms('a'):
            m.geom_rgba[g] = ARM_A_SAT
        for g in arm_geoms('b'):
            m.geom_rgba[g] = ARM_B_SAT
        for i in range(m.nsite):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i)
            if nm and nm.startswith('anchor_'):
                m.site_rgba[i] = ANCHOR_GOLD
                m.site_size[i] = [0.030, 0.050, 0.0]   # upright gold post on the rail (r, half-h)
        # low-ish ambient + directional key (overhead lights the slab top face)
        m.vis.headlight.ambient[:] = [0.24, 0.24, 0.26]
        m.vis.headlight.diffuse[:] = [0.40, 0.40, 0.40]
        if m.nlight >= 2:
            m.light_diffuse[0] = [0.85, 0.85, 0.90]   # overhead key (lights slab top)
            m.light_diffuse[1] = [0.75, 0.72, 0.66]   # angled warm rim
        changes += ['structure rgba->(0.80,0.82,0.85); ArmA(blue)->(0.15,0.35,0.92); '
                    'ArmB(red)->(0.90,0.22,0.15); anchors->gold(1.0,0.72,0.12) size 0.045',
                    'headlight ambient 0.24 / diffuse 0.40; light0 key 0.85, light1 rim 0.75']
    else:
        # camera A: v2 look -> structure hidden, v2 arm/anchor colours
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == sid:
                m.geom_rgba[g, 3] = 0.0
        for i in range(m.nsite):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i)
            if nm and nm.startswith('anchor_'):
                m.site_rgba[i] = ANCHOR_BLUE
                m.site_size[i] = np.maximum(m.site_size[i], 0.03)
        changes += ['camera A: structure geoms alpha->0 (hidden, unchanged v2 white hero); '
                    'anchors blue (v2)']
    return m, changes


def render_white(renderer, m, d, qpos, cam, opt):
    """Camera A: seg-matte over pure white (skybox off)."""
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    renderer.update_scene(d, camera=cam, scene_option=opt)
    sc = renderer.scene
    for f in ('mjRND_SHADOW', 'mjRND_SKYBOX', 'mjRND_REFLECTION', 'mjRND_HAZE'):
        sc.flags[getattr(mujoco.mjtRndFlag, f)] = 0
    rgb = renderer.render().copy()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(d, camera=cam, scene_option=opt)
    seg = renderer.render()
    renderer.disable_segmentation_rendering()
    mask = seg[:, :, 0] >= 0
    out = np.empty_like(rgb); out[:] = WHITE; out[mask] = rgb[mask]
    return out, mask


def render_dark(renderer, m, d, qpos, cam, opt):
    """Camera B: dark gradient skybox as background (no matte)."""
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    renderer.update_scene(d, camera=cam, scene_option=opt)
    sc = renderer.scene
    sc.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
    sc.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    sc.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    sc.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 0
    return renderer.render().copy()
