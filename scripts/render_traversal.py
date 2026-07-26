"""Render PNG frames of the cooperative-arms 5-step traversal.

Reads snapshots from ``Misc/runs/diag_cooperative_arms/sim_log.json``
(captured with ``cfg.frames_per_step > 0``) and renders an isometric
view of each snapshot using MuJoCo's offscreen Renderer.

Overlays drawn into the MjvScene:
  - Anchor spheres (stance=green, dock_target=red, future=grey)
  - Torso frame axes (RGB = XYZ, 100 mm)
  - EE position spheres (ee_A=blue, ee_B=orange)

Frame label ("Step X — t=Xs — d=Xmm") drawn with PIL after render.

Output:
  Misc/runs/frames/step{S}_frame{K}.png
  Misc/runs/t15fk_traversal_overview.png   (3×5 grid of all frames)

Usage:
  MUJOCO_GL=osmesa PYTHONPATH=. python3 scripts/render_traversal.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import mujoco

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from crawlbot.planning.contact_scheduler import ContactScheduler  # noqa: E402

MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')
_default_simlog = os.path.join(_root, 'Misc', 'runs', 'diag_cooperative_arms', 'sim_log.json')
SIMLOG = sys.argv[1] if len(sys.argv) > 1 else _default_simlog
FRAMES_DIR = os.path.join(_root, 'Misc', 'runs', 'frames')
OVERVIEW_PNG = os.path.join(_root, 'Misc', 'runs',
                            't15fk_traversal_overview.png')
WIDTH, HEIGHT = 1280, 720

LABEL_RE = re.compile(r'frame_step(\d+)_(\d+)')


# ─── Helpers ──────────────────────────────────────────────────────


def add_sphere(scene, pos, rgba, r=0.06):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([r, 0, 0], dtype=np.float64),
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def add_arrow(scene, frm, to, rgba, width=0.012):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.zeros(3, dtype=np.float64),
        pos=np.zeros(3, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        g,
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        width=float(width),
        from_=np.asarray(frm, dtype=np.float64),
        to=np.asarray(to, dtype=np.float64),
    )
    scene.ngeom += 1


def torso_pose_from_data(data, model):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
    pos = data.xpos[bid].copy()
    rot = data.xmat[bid].reshape(3, 3).copy()
    return pos, rot


def ee_pos_from_data(data, model, arm: str):
    # End-effector body name convention: gripper_a / gripper_b
    name = f'tool_{arm}'
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        return None
    return data.xpos[bid].copy()


# ─── Step / phase context per snapshot ────────────────────────────


def build_step_context(log) -> Dict[int, Dict]:
    """For each step 0..N-1, derive (stance arm, dock anchor idx,
    arm-of-dock, all anchor ids past + future)."""
    sidx_log = np.array(log['step_idx'])
    swing_log = np.array(log['swing_arm'], dtype=object)
    t_log = np.array(log['t'])

    ctx: Dict[int, Dict] = {}
    for s in sorted(set(int(x) for x in sidx_log if x >= 0)):
        m = sidx_log == s
        sa = swing_log[m]
        swing_arm = str(sa[len(sa) // 2])
        ctx[s] = {'swing_arm': swing_arm}
    return ctx


def per_frame_dock_distance(log, t_query: float, step_idx: int) -> float:
    """Closest sample distance d_grip_swing at this time/step (mm)."""
    sidx_log = np.array(log['step_idx'])
    t_log = np.array(log['t'])
    d_swing = np.array(log['d_grip_swing'])
    m = sidx_log == step_idx
    if not m.any():
        return float('nan')
    t_sub = t_log[m]
    d_sub = d_swing[m]
    k = int(np.argmin(np.abs(t_sub - t_query)))
    return float(d_sub[k] * 1000.0)


# ─── Main ────────────────────────────────────────────────────────


def main():
    if not os.path.exists(SIMLOG):
        print(f'Missing log: {SIMLOG}', file=sys.stderr)
        sys.exit(1)
    with open(SIMLOG) as f:
        log = json.load(f)

    snaps = log.get('snapshots', [])
    if not snaps:
        print('No snapshots in log — re-run sim with cfg.frames_per_step > 0',
              file=sys.stderr)
        sys.exit(2)

    # Filter to frame_step{S}_{K} snapshots only
    frame_snaps: List[Tuple[int, int, float, list, list]] = []
    for snap in snaps:
        t, qpos, qvel, label = snap
        m = LABEL_RE.match(str(label) or '')
        if m:
            s_idx = int(m.group(1))
            k_idx = int(m.group(2))
            frame_snaps.append((s_idx, k_idx, float(t), qpos, qvel))
    if not frame_snaps:
        print('No frame_step*_* snapshots found.', file=sys.stderr)
        sys.exit(3)
    frame_snaps.sort(key=lambda x: (x[0], x[1]))
    steps_present = sorted({s for s, _, _, _, _ in frame_snaps})
    frames_per = max(k for _, k, _, _, _ in frame_snaps) + 1
    print(f'Found {len(frame_snaps)} frame snapshots across '
          f'{len(steps_present)} steps (k_max+1 = {frames_per})')

    # Load model + schedule
    model = mujoco.MjModel.from_xml_path(MJCF)
    # Bump offscreen framebuffer to the requested width/height (model
    # default is 640×480 which is too small for 1280×720 rendering).
    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = HEIGHT
    data = mujoco.MjData(model)
    sched = ContactScheduler()
    # Match diag run defaults
    sched.plan_traversal(start_a=2, start_b=2, n_steps=5)

    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = 45.0
    cam.elevation = -35.0

    os.makedirs(FRAMES_DIR, exist_ok=True)
    rendered: Dict[Tuple[int, int], Image.Image] = {}

    # Camera framing: per-frame lookat centered on the torso so the
    # robot stays in frame as it crawls. Distance chosen to comfortably
    # show the robot body plus the next 2 anchor pairs (≈ 2.5 m).
    anchors_all = np.array(
        [np.asarray(p, dtype=float)
         for arm_list in (sched.anchors_a, sched.anchors_b) for p in arm_list])

    ctx = build_step_context(log)

    # Resolve docked/future anchors per step. The scheduling assumes both
    # arms start at start_a=start_b=2 and each step advances the swing
    # arm to start_idx + step_idx_for_that_arm + 1 (b-first convention).
    # We'll re-derive each step's swing-target anchor by looking at the
    # log's d_grip_swing minimum and matching anchors.
    # Simpler approach: from contact_scheduler, target indices per step
    # follow the alternation b, a, b, a, b starting from (2, 2):
    #   step 0: swing=b, target_b=3 -> stance pair (a=2, b=3) afterward
    #   step 1: swing=a, target_a=3 -> stance pair (a=3, b=3) afterward
    #   step 2: swing=b, target_b=4 -> stance pair (a=3, b=4) afterward
    # Derive per-step swing target + pre-swing stance pair from the
    # logged GaitPlan when available; fall back to the 5-step canonical
    # gait for older logs. Each entry: step_idx → (swing_arm, target_idx).
    gait_plan = log.get('gait_plan', None)
    plan_targets = {}
    stance_pair_before = {}
    if gait_plan and 'phases' in gait_plan:
        ia0, ib0 = gait_plan.get('start_a', 2), gait_plan.get('start_b', 2)
        ia, ib = ia0, ib0
        step_i = 0
        for ph in gait_plan['phases']:
            if ph.get('phase') == 'SINGLE_A':
                plan_targets[step_i] = ('b', ph['swing_to_idx'])
                stance_pair_before[step_i] = (ia, ib)
                ib = ph['swing_to_idx']
                step_i += 1
            elif ph.get('phase') == 'SINGLE_B':
                plan_targets[step_i] = ('a', ph['swing_to_idx'])
                stance_pair_before[step_i] = (ia, ib)
                ia = ph['swing_to_idx']
                step_i += 1
    if not plan_targets:
        # Fallback to canonical 5-step
        plan_targets = {0: ('b', 3), 1: ('a', 3), 2: ('b', 4),
                        3: ('a', 4), 4: ('b', 5)}
        stance_pair_before = {0: (2, 2), 1: (2, 3), 2: (3, 3),
                              3: (3, 4), 4: (4, 4)}

    def anchor_pos(arm: str, idx: int) -> np.ndarray:
        lst = sched.anchors_a if arm == 'a' else sched.anchors_b
        if 0 <= idx < len(lst):
            return np.asarray(lst[idx], dtype=float)
        return None

    for (s_idx, k_idx, t_snap, qpos, qvel) in frame_snaps:
        data.qpos[:] = np.asarray(qpos)
        data.qvel[:] = np.asarray(qvel)
        mujoco.mj_forward(model, data)

        # ── Per-frame camera: track the torso ───────────────────
        bid_torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                       'torso')
        cam.lookat = data.xpos[bid_torso].copy()
        cam.distance = 3.5  # m — robot body fits comfortably

        # ── Update scene then add overlays ───
        renderer.update_scene(data, camera=cam)
        scene = renderer.scene

        # Anchor spheres
        swing_arm, target_idx = plan_targets.get(s_idx, ('b', 0))
        stance_a, stance_b = stance_pair_before.get(s_idx, (0, 0))
        # Stance anchors (green)
        for arm, idx in (('a', stance_a), ('b', stance_b)):
            p = anchor_pos(arm, idx)
            if p is not None:
                add_sphere(scene, p, rgba=(0.1, 0.85, 0.15, 0.9), r=0.06)
        # Dock target (red)
        p_dock = anchor_pos(swing_arm, target_idx)
        if p_dock is not None:
            add_sphere(scene, p_dock, rgba=(0.95, 0.15, 0.15, 0.95), r=0.07)
        # Future anchors (grey)
        for arm, n_anchors in (('a', len(sched.anchors_a)),
                               ('b', len(sched.anchors_b))):
            for i in range(n_anchors):
                if (arm, i) in {('a', stance_a), ('b', stance_b),
                                (swing_arm, target_idx)}:
                    continue
                p = anchor_pos(arm, i)
                if p is not None:
                    add_sphere(scene, p, rgba=(0.55, 0.55, 0.55, 0.5),
                               r=0.045)

        # Torso frame axes (XYZ → RGB)
        t_pos, t_rot = torso_pose_from_data(data, model)
        L_axis = 0.10  # 100 mm
        for axis_i, color in enumerate(((1, 0, 0, 1),
                                        (0, 1, 0, 1),
                                        (0, 0.4, 1, 1))):
            tip = t_pos + t_rot[:, axis_i] * L_axis
            add_arrow(scene, t_pos, tip, rgba=color, width=0.01)

        # EE spheres
        p_ee_a = ee_pos_from_data(data, model, 'a')
        p_ee_b = ee_pos_from_data(data, model, 'b')
        if p_ee_a is not None:
            add_sphere(scene, p_ee_a, rgba=(0.15, 0.4, 0.95, 0.95), r=0.04)
        if p_ee_b is not None:
            add_sphere(scene, p_ee_b, rgba=(1.0, 0.55, 0.0, 0.95), r=0.04)

        # Render base image
        pixels = renderer.render()
        img = Image.fromarray(pixels)

        # Text label
        d_mm = per_frame_dock_distance(log, t_snap, s_idx)
        label_txt = (
            f'Step {s_idx} — t={t_snap:.2f}s — '
            f'd={d_mm:.1f}mm — swing={ctx.get(s_idx, {}).get("swing_arm", "?")}')
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 24)
        except OSError:
            font = ImageFont.load_default()
        pad = 12
        bbox = draw.textbbox((pad, pad), label_txt, font=font)
        draw.rectangle([bbox[0] - 6, bbox[1] - 4, bbox[2] + 6, bbox[3] + 4],
                       fill=(0, 0, 0, 180))
        draw.text((pad, pad), label_txt, fill=(255, 255, 255), font=font)

        out_path = os.path.join(FRAMES_DIR, f'step{s_idx}_frame{k_idx}.png')
        img.save(out_path)
        rendered[(s_idx, k_idx)] = img
        print(f'  wrote {out_path}  ({label_txt})')

    # ── Per-step strips (6 views/step) + full overview grid ──────────
    rows = sorted(set(s for s, _ in rendered.keys()))      # ALL steps
    cols = sorted(set(k for _, k in rendered.keys()))      # ALL frames
    if rows and cols:
        thumb_w = WIDTH // 2
        thumb_h = HEIGHT // 2

        # One strip image per step (its frames left→right in time).
        strips_dir = os.path.join(FRAMES_DIR, 'strips')
        os.makedirs(strips_dir, exist_ok=True)
        for s in rows:
            ks = sorted(k for (ss, k) in rendered.keys() if ss == s)
            strip = Image.new('RGB', (thumb_w * len(ks), thumb_h),
                              (20, 20, 20))
            for ci, k in enumerate(ks):
                strip.paste(rendered[(s, k)].resize(
                    (thumb_w, thumb_h), Image.LANCZOS), (ci * thumb_w, 0))
            sp = os.path.join(strips_dir, f'step{s}_strip.png')
            strip.save(sp)
            print(f'  wrote {sp}')

        # Full overview: all steps (rows) × all frames (cols).
        grid = Image.new('RGB', (thumb_w * len(cols), thumb_h * len(rows)),
                         (20, 20, 20))
        for ri, s in enumerate(rows):
            for ci, k in enumerate(cols):
                img = rendered.get((s, k))
                if img is None:
                    continue
                grid.paste(img.resize((thumb_w, thumb_h), Image.LANCZOS),
                           (ci * thumb_w, ri * thumb_h))
        grid.save(OVERVIEW_PNG)
        print(f'  wrote {OVERVIEW_PNG}')


if __name__ == '__main__':
    main()
