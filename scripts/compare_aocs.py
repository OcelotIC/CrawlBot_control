#!/usr/bin/env python3
"""Comparative AOCS test at 8% mass ratio.

Three variants:
  A. Legacy AOCS (L_dot FF, ±0.5 Nm) — baseline matching status_report
  B. NMPC orbital fix only (L_dot AOCS, ±0.5 Nm) — isolates NMPC correction
  C. NMPC fix + H_{r/O} AOCS (FF + feedback, ±0.5 Nm) — full correction

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
    print(f'\n  {"─"*55}')
    print(f'  {label}')
    print(f'  {"─"*55}')
    print(f'  Docks:         {nd}/{N_STEPS}')
    print(f'  Max |angle|:   {rot:.2f}°')
    print(f'  Final euler:   r={euler[-1,0]:.1f}° p={euler[-1,1]:.1f}° y={euler[-1,2]:.1f}°')
    print(f'  Struct drift:  {np.linalg.norm(sp[-1]-sp[0])*100:.1f} cm')
    print(f'  Max ||L_com||: {np.array(log.L_com_norm).max():.2f} Nms')
    print(f'  Max ||hw||:    {hwn.max():.2f} Nms')
    print(f'  hw viol (>5):  {np.sum(hwn>5)}/{len(hwn)}')
    print(f'  NMPC fails:    {nf}/{len(log.nmpc_ok)}')
    return dict(docks=nd, rot=rot, hw_max=hwn.max(), nmpc_fails=nf)


# ── Common parameters ─────────────────────────────────────────────────────────
base = dict(
    tau_max=20.0,
    hw_min=np.full(3, -5.0),
    hw_max=np.full(3, 5.0),
    L_max=10.0,
    tau_w_max=5.0,
    nmpc_Wv=10.0,
    nmpc_W_hw=0.0,
    t_settle_final=20.0,
)

# ── A. Legacy (old NMPC + old AOCS) ──────────────────────────────────────────
# To reproduce the old NMPC behavior (r_mid proxy), we'd need to revert
# centroidal_nmpc.py. Instead we just run with the FIXED NMPC and note that
# variant B isolates its effect. We skip the true legacy for now; the
# status_report gives the baseline at 14% (24°, 3/3).

# ── B. NMPC fix + legacy AOCS (L_dot FF, ±0.5 Nm) ───────────────────────────
cfg_B = SimConfig(
    **base,
    aocs_use_H_estimator=False,
    aocs_K_hw=2.0,
    aocs_tau_w_max=0.5,
)

log_B = run_sim(MJCF, cfg_B, 'B: NMPC fix + legacy AOCS (L_dot, ±0.5 Nm)')
log_B.save(os.path.join(OUTDIR_LOGS, 'nmpc_fix_legacy_aocs_8pct.json'))

# ── C. NMPC fix + H_{r/O} AOCS (light FB, ±0.5 Nm) ─────────────────────────
cfg_C = SimConfig(
    **base,
    aocs_use_H_estimator=True,
    aocs_tau_w_max=0.5,
    aocs_filter_tau=0.032,
    aocs_K_omega=2.0,
    aocs_K_h=0.1,
)

log_C = run_sim(MJCF, cfg_C, 'C: NMPC fix + H_{r/O} AOCS (FF+FB, ±0.5 Nm)')
log_C.save(os.path.join(OUTDIR_LOGS, 'nmpc_fix_H_aocs_8pct.json'))

# ── D. NMPC fix + H_{r/O} AOCS (full torque, ±5 Nm) ────────────────────────
cfg_D = SimConfig(
    **base,
    aocs_use_H_estimator=True,
    aocs_tau_w_max=5.0,
    aocs_filter_tau=0.032,
    aocs_K_omega=2.0,
    aocs_K_h=0.1,
)

log_D = run_sim(MJCF, cfg_D, 'D: NMPC fix + H_{r/O} AOCS (FF+FB, ±5 Nm)')
log_D.save(os.path.join(OUTDIR_LOGS, 'nmpc_fix_H_aocs_full_8pct.json'))


# ── Metrics ──────────────────────────────────────────────────────────────────
print(f'\n\n{"="*70}')
print(f'  COMPARISON — 8% mass ratio (888 kg structure)')
print(f'{"="*70}')

mB = metrics(log_B, 'B: NMPC fix + legacy AOCS (±0.5 Nm)')
mC = metrics(log_C, 'C: NMPC fix + H AOCS (±0.5 Nm)')
mD = metrics(log_D, 'D: NMPC fix + H AOCS (±5.0 Nm)')

print(f'\n  {"="*55}')
print(f'  Summary:')
print(f'    B  →  {mB["docks"]}/3 docks, {mB["rot"]:.1f}° rotation, '
      f'{mB["nmpc_fails"]} NMPC fails')
print(f'    C  →  {mC["docks"]}/3 docks, {mC["rot"]:.1f}° rotation, '
      f'{mC["nmpc_fails"]} NMPC fails')
print(f'    D  →  {mD["docks"]}/3 docks, {mD["rot"]:.1f}° rotation, '
      f'{mD["nmpc_fails"]} NMPC fails')
print(f'  {"="*55}')


# ── Comparison plot ──────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)
fig.suptitle('AOCS Comparison — 888 kg structure (8% ratio)',
             fontsize=14, fontweight='bold')

configs = [
    (log_B, 'B: NMPC fix + L_dot AOCS', '--'),
    (log_C, 'C: NMPC fix + H AOCS ±0.5', '-'),
    (log_D, 'D: NMPC fix + H AOCS ±5.0', '-.'),
]
colors = ['C0', 'C1', 'C2']

# 1. Structure rotation (max absolute angle)
ax = axes[0]
for (log, lbl, ls), c in zip(configs, colors):
    t = np.array(log.t)
    euler = np.array(log.struct_euler_deg)
    max_abs = np.max(np.abs(euler), axis=1)
    ax.plot(t, max_abs, ls, color=c, label=lbl, alpha=0.8)
ax.set_ylabel('Max |angle| [°]')
ax.legend(fontsize=8)
ax.set_title('Structure rotation (max absolute Euler angle)')
ax.grid(True, alpha=0.3)

# 2. Wheel momentum
ax = axes[1]
for (log, lbl, ls), c in zip(configs, colors):
    t = np.array(log.t)
    hw = np.array(log.hw_physical)
    ax.plot(t, np.linalg.norm(hw, axis=1), ls, color=c, label=lbl, alpha=0.8)
ax.axhline(5.0, color='r', linestyle=':', alpha=0.5, label='h_max')
ax.set_ylabel('||hw|| [Nms]')
ax.legend(fontsize=8)
ax.set_title('Wheel angular momentum')
ax.grid(True, alpha=0.3)

# 3. ||L_com|| (angular momentum budget)
ax = axes[2]
for (log, lbl, ls), c in zip(configs, colors):
    t = np.array(log.t)
    ax.plot(t, np.array(log.L_com_norm), ls, color=c, label=lbl, alpha=0.8)
ax.axhline(10.0, color='r', linestyle=':', alpha=0.5, label='L_max')
ax.set_ylabel('||L_com|| [Nms]')
ax.legend(fontsize=8)
ax.set_title('Centroidal angular momentum')
ax.grid(True, alpha=0.3)

# 4. H_{r/O} estimator diagnostics (variant C/D only)
ax = axes[3]
for (log, lbl, ls), c in zip(configs[1:], colors[1:]):
    t = np.array(log.t)
    if log.H_rO and len(log.H_rO) == len(t):
        H = np.array(log.H_rO)
        ax.plot(t, np.linalg.norm(H, axis=1), ls, color=c,
                label=f'||H_{{r/O}}|| ({lbl})', alpha=0.8)
# Also show constraint torque ground truth for D
if log_D.qfrc_constraint_torque:
    t = np.array(log_D.t)
    qfrc = np.array(log_D.qfrc_constraint_torque)
    ax.plot(t, np.linalg.norm(qfrc, axis=1), ':', color='gray',
            label='||qfrc_constraint|| (ground truth)', alpha=0.5)
ax.set_ylabel('[Nms]')
ax.set_xlabel('Time [s]')
ax.legend(fontsize=8)
ax.set_title('H_{r/O} and constraint torque')
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(OUTDIR_FIGS, 'aocs_comparison_8pct.png')
plt.savefig(save_path, dpi=150)
print(f'\n  Saved: {save_path}')

# Standard 9-panel for each
for log, name in [(log_B, 'nmpc_fix_legacy_8pct'),
                  (log_C, 'nmpc_fix_H_aocs_8pct'),
                  (log_D, 'nmpc_fix_H_aocs_full_8pct')]:
    p = os.path.join(OUTDIR_FIGS, f'{name}.png')
    SimulationLoop.plot(log, cfg=cfg_B, save_path=p)
    print(f'  Saved: {p}')
