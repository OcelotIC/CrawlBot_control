#!/usr/bin/env python3
"""Test structure-frame refactoring at 0.1% mass ratio (71000 kg structure).

This is the easiest test case: the massive structure barely moves,
so structure frame ≈ world frame. Expected: 3/3 docks, <1° rotation,
hw peak <1 Nms.

Usage:
    PYTHONPATH=. MUJOCO_GL=disabled python3 scripts/test_struct_frame_0p1.py
"""
import os, sys, time, re, tempfile
import numpy as np

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

from simulation_loop import SimulationLoop, SimConfig, SimLog

URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')
MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')
OUTDIR_LOGS = os.path.join(_root, 'results', 'logs')
OUTDIR_FIGS = os.path.join(_root, 'results', 'figures')
os.makedirs(OUTDIR_LOGS, exist_ok=True)
os.makedirs(OUTDIR_FIGS, exist_ok=True)

TARGET_STRUCT_MASS = 71000.0  # kg  (0.1% ratio with ~71 kg robot)

print('=' * 70)
print('  Structure-frame test — 0.1% mass ratio (71000 kg structure)')
print('=' * 70)

# --- Create patched MJCF with correct mass and inertia ---
with open(MJCF, 'r') as f:
    mjcf_src = f.read()

pattern = r'(<body\s+name="structure"[^>]*>.*?<inertial[^>]*mass=")(\d+)(".*?fullinertia=")([\d\s]+)(")'
m = re.search(pattern, mjcf_src, re.DOTALL)
assert m, "Could not find structure inertial tag in MJCF"

old_mass = float(m.group(2))
ratio = TARGET_STRUCT_MASS / old_mass
old_inertia = [float(x) for x in m.group(4).split()]
new_inertia = [v * ratio for v in old_inertia]
new_inertia_str = ' '.join(f'{v:.0f}' for v in new_inertia)

mjcf_patched = (mjcf_src[:m.start(2)] +
                f'{TARGET_STRUCT_MASS:.0f}' +
                mjcf_src[m.end(2):m.start(4)] +
                new_inertia_str +
                mjcf_src[m.end(4):])

tmp = tempfile.NamedTemporaryFile(suffix='.xml', delete=False, mode='w')
tmp.write(mjcf_patched)
tmp.close()
print(f'  Structure mass: {old_mass:.0f} -> {TARGET_STRUCT_MASS:.0f} kg (x{ratio:.1f})')
print(f'  Inertia scaled: {old_inertia[:3]} -> {new_inertia[:3]}')

# --- Setup and run ---
cfg = SimConfig(
    tau_max=20.0,
    hw_min=np.full(3, -5.0),
    hw_max=np.full(3, 5.0),
    L_max=10.0,
    tau_w_max=5.0,
    nmpc_Wv=10.0,
    nmpc_W_hw=0.0,
    t_settle_final=20.0,
)

sim = SimulationLoop(mjcf_path=tmp.name, urdf_path=URDF, config=cfg)
sim.setup(n_steps=3, start_a=2, start_b=2)
os.unlink(tmp.name)

t0 = time.time()
log = sim.run(verbose=True)
wall = time.time() - t0
print(f'\n  Wall time: {wall:.1f}s')

# --- Save results ---
log_path = os.path.join(OUTDIR_LOGS, 'structframe_0p1pct.json')
log.save(log_path)
print(f'  Saved: {log_path}')

fig = SimulationLoop.plot(log, cfg=cfg,
                          save_path=os.path.join(OUTDIR_FIGS, 'structframe_0p1pct.png'))
print(f'  Saved: results/figures/structframe_0p1pct.png')

# --- Metrics ---
t = np.array(log.t)
Ln = np.array(log.L_com_norm)
euler = np.array(log.struct_euler_deg)
sp = np.array(log.struct_pos)
hw_phys = np.array(log.hw_physical)

print(f'\n{"="*70}')
print(f'  METRICS — 0.1% mass ratio')
print(f'{"="*70}')
print(f'  Docks:           {len(log.dock_events)}/3')
print(f'  Max |angle|:     {np.max(np.abs(euler)):.2f}°')
print(f'  Struct drift:    {np.linalg.norm(sp[-1]-sp[0])*1000:.1f} mm')
print(f'  Max ||L_com||:   {Ln.max():.3f} Nms')
print(f'  Max ||hw_phys||: {np.max(np.linalg.norm(hw_phys, axis=1)):.3f} Nms')
print(f'  NMPC fails:      {sum(1 for x in log.nmpc_ok if not x)}/{len(log.nmpc_ok)}')
print(f'  QP fails:        {sum(1 for x in log.qp_ok if not x)}/{len(log.qp_ok)}')
print(f'{"="*70}')
