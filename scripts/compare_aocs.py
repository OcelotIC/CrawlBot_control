#!/usr/bin/env python3
"""Wheel sizing with decoupled NMPC constraints.

Key idea: use large physical wheels (hw_phys_max=50 Nms) but keep tight
NMPC planning constraints (hw_plan_max=5-10 Nms) to force conservative
trajectory planning. The physical headroom absorbs execution errors.

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
MJCF_5  = os.path.join(_root, 'models', 'VISPA_crawling_rwa3_8pct.xml')
MJCF_50 = os.path.join(_root, 'models', 'VISPA_crawling_rwa3_8pct_hw50.xml')
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


def metrics(log, label, hw_lim):
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
    print(f'  Max ||hw||:    {hwn.max():.2f} Nms (plan lim {hw_lim})')
    print(f'  NMPC fails:    {nf}/{len(log.nmpc_ok)}')
    return dict(docks=nd, rot=rot, hw_max=hwn.max(), nmpc_fails=nf)


# ── B: Baseline (small wheels, hw=5) ─────────────────────────────────────────
cfg_B = SimConfig(
    tau_max=20.0, hw_min=np.full(3, -5.0), hw_max=np.full(3, 5.0),
    L_max=10.0, tau_w_max=5.0, rwa_I_w=0.01,
    nmpc_Wv=10.0, nmpc_W_hw=0.0, t_settle_final=20.0,
    aocs_mode='legacy', aocs_use_H_estimator=False,
    aocs_K_hw=2.0, aocs_tau_w_max=0.5,
)
log_B = run_sim(MJCF_5, cfg_B, 'B: small wheels, hw_plan=5, ±0.5 Nm')
log_B.save(os.path.join(OUTDIR_LOGS, 'B_baseline.json'))

# ── N: Big wheels, tight NMPC (hw_plan=5, ±0.5) ─────────────────────────────
cfg_N = SimConfig(
    tau_max=20.0, hw_min=np.full(3, -5.0), hw_max=np.full(3, 5.0),
    L_max=10.0, tau_w_max=5.0, rwa_I_w=0.1,
    nmpc_Wv=10.0, nmpc_W_hw=0.0, t_settle_final=20.0,
    aocs_mode='legacy', aocs_use_H_estimator=False,
    aocs_K_hw=2.0, aocs_tau_w_max=0.5,
)
log_N = run_sim(MJCF_50, cfg_N, 'N: big wheels, hw_plan=5, ±0.5 Nm')
log_N.save(os.path.join(OUTDIR_LOGS, 'N_big_tight5.json'))

# ── O: Big wheels, moderate NMPC (hw_plan=10, ±0.5) ─────────────────────────
cfg_O = SimConfig(
    tau_max=20.0, hw_min=np.full(3, -10.0), hw_max=np.full(3, 10.0),
    L_max=10.0, tau_w_max=5.0, rwa_I_w=0.1,
    nmpc_Wv=10.0, nmpc_W_hw=0.0, t_settle_final=20.0,
    aocs_mode='legacy', aocs_use_H_estimator=False,
    aocs_K_hw=2.0, aocs_tau_w_max=0.5,
)
log_O = run_sim(MJCF_50, cfg_O, 'O: big wheels, hw_plan=10, ±0.5 Nm')
log_O.save(os.path.join(OUTDIR_LOGS, 'O_big_tight10.json'))

# ── P: Big wheels, tight NMPC (hw_plan=5, ±2.0) ─────────────────────────────
cfg_P = SimConfig(
    tau_max=20.0, hw_min=np.full(3, -5.0), hw_max=np.full(3, 5.0),
    L_max=10.0, tau_w_max=5.0, rwa_I_w=0.1,
    nmpc_Wv=10.0, nmpc_W_hw=0.0, t_settle_final=20.0,
    aocs_mode='legacy', aocs_use_H_estimator=False,
    aocs_K_hw=2.0, aocs_tau_w_max=2.0,
)
log_P = run_sim(MJCF_50, cfg_P, 'P: big wheels, hw_plan=5, ±2.0 Nm')
log_P.save(os.path.join(OUTDIR_LOGS, 'P_big_tight5_tw2.json'))


# ── Metrics ──────────────────────────────────────────────────────────────────
print(f'\n\n{"="*70}')
print(f'  RESULTS — Decoupled wheel sizing')
print(f'{"="*70}\n')

mB = metrics(log_B, 'B: small, plan=5, ±0.5', 5)
mN = metrics(log_N, 'N: big, plan=5, ±0.5', 5)
mO = metrics(log_O, 'O: big, plan=10, ±0.5', 10)
mP = metrics(log_P, 'P: big, plan=5, ±2.0', 5)

print(f'\n  {"="*65}')
for tag, m in [('B small plan=5 ±0.5', mB), ('N big plan=5 ±0.5', mN),
               ('O big plan=10 ±0.5', mO), ('P big plan=5 ±2.0', mP)]:
    print(f'  {tag:22s} | {m["docks"]}/3 | {m["rot"]:6.2f}° | '
          f'hw={m["hw_max"]:5.2f} | fails={m["nmpc_fails"]}')
print(f'  {"="*65}')


# ── Plot ─────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

configs = [
    (log_B, 'B: small, ±0.5', '--', 'C0'),
    (log_N, 'N: big, plan=5, ±0.5', '-', 'C1'),
    (log_O, 'O: big, plan=10, ±0.5', '-', 'C2'),
    (log_P, 'P: big, plan=5, ±2.0', '-.', 'C3'),
]

fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
fig.suptitle('Decoupled Wheel Sizing — 888 kg (8%)', fontsize=14, fontweight='bold')

ax = axes[0]
for log, lbl, ls, c in configs:
    t = np.array(log.t)
    euler = np.array(log.struct_euler_deg)
    ax.plot(t, np.max(np.abs(euler), axis=1), ls, color=c, label=lbl, alpha=0.8, lw=1.5)
ax.set_ylabel('Max |angle| [°]'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_title('Structure rotation')

ax = axes[1]
for log, lbl, ls, c in configs:
    t = np.array(log.t)
    hw = np.array(log.hw_physical)
    ax.plot(t, np.linalg.norm(hw, axis=1), ls, color=c, label=lbl, alpha=0.8, lw=1.5)
ax.axhline(5, color='gray', ls=':', alpha=0.5, label='plan limit=5')
ax.set_ylabel('||hw|| [Nms]'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_title('Wheel momentum')

ax = axes[2]
for log, lbl, ls, c in configs:
    t = np.array(log.t)
    tw = np.array(log.tau_w)
    ax.plot(t, np.linalg.norm(tw, axis=1), ls, color=c, label=lbl, alpha=0.8, lw=1.5)
ax.set_ylabel('||τ_w|| [Nm]'); ax.set_xlabel('Time [s]')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_title('Wheel torque')

plt.tight_layout()
save_path = os.path.join(OUTDIR_FIGS, 'wheel_sizing_8pct.png')
plt.savefig(save_path, dpi=150)
print(f'\n  Saved: {save_path}')
