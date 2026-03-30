#!/usr/bin/env python3
"""AOCS tuning at 8% mass ratio — NMPC passivity vs torque budget.

Usage:
    PYTHONPATH=. MUJOCO_GL=disabled python3 scripts/compare_aocs.py
"""
import os, sys, time
import numpy as np

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

from simulation_loop import SimulationLoop, SimConfig, SimLog

URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')
MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3_8pct.xml')
OUTDIR_LOGS = os.path.join(_root, 'results', 'logs')
OUTDIR_FIGS = os.path.join(_root, 'results', 'figures')
os.makedirs(OUTDIR_LOGS, exist_ok=True)
os.makedirs(OUTDIR_FIGS, exist_ok=True)

N_STEPS = 3


def run_sim(mjcf, cfg, label):
    print(f'\n{"="*70}')
    print(f'  {label}')
    print(f'{"="*70}')
    sim = SimulationLoop(mjcf_path=mjcf, urdf_path=URDF, config=cfg)
    sim.setup(n_steps=N_STEPS, start_a=2, start_b=2)
    t0 = time.time()
    log = sim.run(verbose=True)
    print(f'  Wall time: {time.time()-t0:.1f}s')
    return log


def metrics(log, label):
    euler = np.array(log.struct_euler_deg)
    sp = np.array(log.struct_pos)
    hw = np.array(log.hw_physical)
    hwn = np.linalg.norm(hw, axis=1)
    nf = sum(1 for x in log.nmpc_ok if not x)
    nd = len(log.dock_events)
    rot = np.max(np.abs(euler))
    print(f'  {"─"*55}')
    print(f'  {label}')
    print(f'  {"─"*55}')
    print(f'  Docks:         {nd}/{N_STEPS}')
    print(f'  Max |angle|:   {rot:.2f}°')
    print(f'  Final euler:   r={euler[-1,0]:.1f}° p={euler[-1,1]:.1f}° y={euler[-1,2]:.1f}°')
    print(f'  Max ||L_com||: {np.array(log.L_com_norm).max():.2f} Nms')
    print(f'  Max ||hw||:    {hwn.max():.2f} Nms')
    print(f'  NMPC fails:    {nf}/{len(log.nmpc_ok)}')
    return dict(docks=nd, rot=rot, hw_max=hwn.max(), nmpc_fails=nf)


base = dict(
    tau_max=20.0,
    hw_min=np.full(3, -5.0),
    hw_max=np.full(3, 5.0),
    L_max=10.0,
    tau_w_max=5.0,
    nmpc_Wv=10.0,
    t_settle_final=20.0,
    aocs_mode='legacy',
    aocs_use_H_estimator=False,
    aocs_K_hw=2.0,
)

variants = [
    ('B', 'legacy ±0.5, W_hw=0',   dict(aocs_tau_w_max=0.5, nmpc_W_hw=0.0)),
    ('G', 'legacy ±0.5, W_hw=1',   dict(aocs_tau_w_max=0.5, nmpc_W_hw=1.0)),
    ('H', 'legacy ±0.5, W_hw=10',  dict(aocs_tau_w_max=0.5, nmpc_W_hw=10.0)),
    ('I', 'legacy ±2.0, W_hw=0',   dict(aocs_tau_w_max=2.0, nmpc_W_hw=0.0)),
    ('J', 'legacy ±2.0, W_hw=10',  dict(aocs_tau_w_max=2.0, nmpc_W_hw=10.0)),
]

results = {}
logs = {}

for tag, label, overrides in variants:
    cfg = SimConfig(**{**base, **overrides})
    full_label = f'{tag}: {label}'
    log = run_sim(MJCF, cfg, full_label)
    log.save(os.path.join(OUTDIR_LOGS, f'{tag}_8pct.json'))
    logs[tag] = log

print(f'\n\n{"="*70}')
print(f'  RESULTS — 8% mass ratio (888 kg)')
print(f'{"="*70}\n')

for tag, label, _ in variants:
    results[tag] = metrics(logs[tag], f'{tag}: {label}')

print(f'\n  {"="*55}')
for tag, label, _ in variants:
    m = results[tag]
    print(f'  {tag}: {m["docks"]}/3 docks | {m["rot"]:5.1f}° | '
          f'hw={m["hw_max"]:.2f} | fails={m["nmpc_fails"]}')
print(f'  {"="*55}')


# ── Plot ─────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
fig.suptitle('AOCS Tuning — 888 kg (8%)', fontsize=14, fontweight='bold')

styles = {'B': ('--', 'C0'), 'G': ('-', 'C1'), 'H': ('-.', 'C2'),
          'I': (':', 'C3'), 'J': ('-', 'C4')}

ax = axes[0]
for tag, label, _ in variants:
    t = np.array(logs[tag].t)
    euler = np.array(logs[tag].struct_euler_deg)
    ls, c = styles[tag]
    ax.plot(t, np.max(np.abs(euler), axis=1), ls, color=c,
            label=f'{tag}: {label}', alpha=0.8, linewidth=1.5)
ax.set_ylabel('Max |angle| [°]')
ax.legend(fontsize=8, loc='upper left'); ax.grid(True, alpha=0.3)
ax.set_title('Structure rotation')

ax = axes[1]
for tag, label, _ in variants:
    t = np.array(logs[tag].t)
    hw = np.array(logs[tag].hw_physical)
    ls, c = styles[tag]
    ax.plot(t, np.linalg.norm(hw, axis=1), ls, color=c,
            label=f'{tag}', alpha=0.8, linewidth=1.5)
ax.axhline(5.0, color='r', ls=':', alpha=0.5, label='h_max')
ax.set_ylabel('||hw|| [Nms]'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_title('Wheel momentum')

ax = axes[2]
for tag, label, _ in variants:
    t = np.array(logs[tag].t)
    ls, c = styles[tag]
    ax.plot(t, np.array(logs[tag].L_com_norm), ls, color=c,
            label=f'{tag}', alpha=0.8, linewidth=1.5)
ax.axhline(10.0, color='r', ls=':', alpha=0.5, label='L_max')
ax.set_ylabel('||L_com|| [Nms]'); ax.set_xlabel('Time [s]')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_title('Centroidal angular momentum')

plt.tight_layout()
save_path = os.path.join(OUTDIR_FIGS, 'aocs_comparison_8pct.png')
plt.savefig(save_path, dpi=150)
print(f'\n  Saved: {save_path}')
