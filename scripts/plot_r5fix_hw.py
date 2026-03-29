#!/usr/bin/env python3
"""Generate hw_physical(t) plots for R5-fix validation.

Runs the RWA-3 simulation with n_steps=3 and saves:
  - results/figures/r5fix_hw_physical.png
  - results/logs/r5fix_rwa3_log.json
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('MUJOCO_GL', 'disabled')

import numpy as np

_root = os.path.join(os.path.dirname(__file__), '..')

from simulation_loop import SimulationLoop, SimConfig

print("Running R5-fix simulation (RWA-3, n_steps=3) ...")
cfg = SimConfig()
sim = SimulationLoop(
    mjcf_path=os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml'),
    urdf_path=os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf'),
    config=cfg)
sim.setup(n_steps=3, start_a=2, start_b=2)
log = sim.run(verbose=True)

# Save log
log_path = os.path.join(_root, 'results', 'logs', 'r5fix_rwa3_log.json')
log.save(log_path)
print(f"Log saved to {log_path}")

# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

t = np.array(log.t)
hw_phys = np.array(log.hw_physical)
hw_norms = np.linalg.norm(hw_phys, axis=1)
tau_w = np.array(log.tau_w)
rw_speed = np.array(log.rw_speed)
phases = log.phase

h_max = cfg.hw_max[0]

# Phase shading helper
def shade(ax):
    for i in range(len(t)):
        if phases[i] == 'DS':
            ax.axvspan(t[i] - 0.04, t[i] + 0.04, alpha=0.08, color='blue')
        elif phases[i] == 'EXT':
            ax.axvspan(t[i] - 0.04, t[i] + 0.04, alpha=0.08, color='red')
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            ax.axvline(t[i], color='gray', ls=':', alpha=0.5)

fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
fig.suptitle(
    f'R5-fix Validation — Physical RWA Model (3 wheels, $h_{{max}}$={h_max} Nm$\\cdot$s)',
    fontsize=14, fontweight='bold')

# Panel 1: hw_physical components
ax = axes[0]; shade(ax)
ax.plot(t, hw_phys[:, 0], 'r-', lw=1.5, label='$h_{w,x}$')
ax.plot(t, hw_phys[:, 1], 'g-', lw=1.5, label='$h_{w,y}$')
ax.plot(t, hw_phys[:, 2], 'b-', lw=1.5, label='$h_{w,z}$')
ax.axhline(h_max, color='k', ls='--', lw=1, alpha=0.5)
ax.axhline(-h_max, color='k', ls='--', lw=1, alpha=0.5)
ax.set_ylabel('$h_w$ [Nm$\\cdot$s]')
ax.set_title('(a) Wheel momentum components $h_w(t)$')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Panel 2: ||hw_physical||(t)
ax = axes[1]; shade(ax)
ax.plot(t, hw_norms, 'k-', lw=2, label='$\\|h_w\\|$')
ax.axhline(h_max, color='r', ls='--', lw=2, label=f'$h_{{max}}$={h_max}')
ax.axhline(np.sqrt(3) * h_max, color='r', ls=':', lw=1,
           label=f'$\\sqrt{{3}} \\cdot h_{{max}}$={np.sqrt(3)*h_max:.1f}')
for ev in log.dock_events:
    ax.axvline(ev['t'], color='green', ls='-', lw=2, alpha=0.4)
ax.set_ylabel('$\\|h_w\\|$ [Nm$\\cdot$s]')
ax.set_title('(b) Wheel momentum norm $\\|h_w\\|(t)$')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Panel 3: tau_w (AOCS torques)
ax = axes[2]; shade(ax)
ax.plot(t, tau_w[:, 0], 'r-', lw=1.5, label='$\\tau_{w,x}$')
ax.plot(t, tau_w[:, 1], 'g-', lw=1.5, label='$\\tau_{w,y}$')
ax.plot(t, tau_w[:, 2], 'b-', lw=1.5, label='$\\tau_{w,z}$')
ax.axhline(cfg.aocs_tau_w_max, color='k', ls='--', lw=1, alpha=0.5)
ax.axhline(-cfg.aocs_tau_w_max, color='k', ls='--', lw=1, alpha=0.5)
ax.set_ylabel('$\\tau_w$ [Nm]')
ax.set_title('(c) AOCS wheel torques $\\tau_w(t)$')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Panel 4: Wheel speeds
ax = axes[3]; shade(ax)
ax.plot(t, rw_speed[:, 0], 'r-', lw=1.5, label='$\\omega_{x}$')
ax.plot(t, rw_speed[:, 1], 'g-', lw=1.5, label='$\\omega_{y}$')
ax.plot(t, rw_speed[:, 2], 'b-', lw=1.5, label='$\\omega_{z}$')
ax.set_xlabel('Time [s]')
ax.set_ylabel('$\\omega$ [rad/s]')
ax.set_title('(d) Reaction wheel speeds $\\omega(t)$')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(_root, 'results', 'figures', 'r5fix_hw_physical.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved to {fig_path}")

# Also save PDF
fig_path_pdf = fig_path.replace('.png', '.pdf')
plt.savefig(fig_path_pdf, bbox_inches='tight')
print(f"Figure saved to {fig_path_pdf}")

# Print summary metrics
print(f"\n{'='*60}")
print("R5-fix Metrics:")
print(f"  Docks:           {len(log.dock_events)}/3")
print(f"  Peak ||hw||:     {hw_norms.max():.3f} Nms (limit {h_max})")
print(f"  Mean ||hw||:     {hw_norms.mean():.3f} Nms")
print(f"  hw viol. rate:   {100*np.sum(hw_norms > h_max)/len(hw_norms):.1f}%")
print(f"  Peak |tau_w|:    {np.max(np.abs(tau_w)):.3f} Nm")
print(f"  Peak |omega_w|:  {np.max(np.abs(rw_speed)):.1f} rad/s")
print(f"{'='*60}")
