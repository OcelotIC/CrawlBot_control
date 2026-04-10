"""Render MuJoCo frames at key simulation instants for visual diagnostics.

Requires MUJOCO_GL=osmesa (headless) or egl. If rendering is unavailable,
functions degrade gracefully and print a warning.
"""

import os
import numpy as np


def capture_snapshots(model, data, sim_log, output_dir,
                      width=1280, height=720, camera=None):
    """Render MuJoCo frames from stored snapshot states.

    Args:
        model: mujoco.MjModel
        data: mujoco.MjData (will be modified to restore each snapshot state)
        sim_log: SimLog with .snapshots list of (t, qpos, qvel, label)
        output_dir: directory for PNG files
        width, height: render resolution
        camera: camera name (str) or None for default
    """
    if not hasattr(sim_log, 'snapshots') or len(sim_log.snapshots) == 0:
        print("[snapshots] No snapshots to render.")
        return []

    os.makedirs(output_dir, exist_ok=True)

    try:
        import mujoco
        renderer = mujoco.Renderer(model, height=height, width=width)
    except Exception as e:
        print(f"[snapshots] Cannot create renderer: {e}")
        print("[snapshots] Skipping snapshot rendering (set MUJOCO_GL=osmesa)")
        return []

    paths = []
    try:
        from PIL import Image
    except ImportError:
        print("[snapshots] Pillow not installed, skipping snapshot rendering")
        renderer.close()
        return []

    for i, snap in enumerate(sim_log.snapshots):
        t_snap, qpos, qvel, label = snap
        qpos = np.asarray(qpos)
        qvel = np.asarray(qvel)

        data.qpos[:len(qpos)] = qpos
        data.qvel[:len(qvel)] = qvel
        import mujoco as mj
        mj.mj_forward(model, data)

        try:
            if camera:
                renderer.update_scene(data, camera=camera)
            else:
                renderer.update_scene(data)
            frame = renderer.render()
            img = Image.fromarray(frame)
            fname = f"snap_{i:02d}_t{t_snap:.2f}_{label}.png"
            path = os.path.join(output_dir, fname)
            img.save(path)
            paths.append(path)
        except Exception as e:
            print(f"[snapshots] Frame {i} ({label}) failed: {e}")

    renderer.close()
    return paths
