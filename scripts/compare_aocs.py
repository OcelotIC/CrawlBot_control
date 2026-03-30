#!/usr/bin/env python3
"""Comparative AOCS test: legacy (L_dot only) vs H_{r/O} estimator.

Runs 3-step simulation at 14% mass ratio (500 kg structure) with both AOCS
variants, then plots comparison figures.

Usage:
    PYTHONPATH=. MUJOCO_GL=disabled python3 scripts/compare_aocs.py
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

TARGET_STRUCT_MASS = 500.0  # kg  → 14% mass ratio with ~71 kg robot
N_STEPS = 3

# ── Patch MJCF to set structure mass ───────────────────────────────────────────

def create_patched_mjcf(target_mass):
    with open(MJCF, 'r') as f:
        mjcf_src = f.read()

    pattern = r'(<body\s+name="structure"[^>]*>.*?<inertial[^>]*mass=")(\d+)(".*?fullinertia=")([\d\s]+)(")'
    m = re.search(pattern, mjcf_src, re.DOTALL)
    assert m, "Could not find structure inertial tag in MJCF"

    old_mass = float(m.group(2))
    ratio = target_mass / old_mass
    old_inertia = [float(x) for x in m.group(4).split()]
    new_inertia = [v * ratio for v in old_inertia]
    new_inertia_str = ' '.join(f'{v:.1f}' for v in new_inertia)

    mjcf_patched = (mjcf_src[:m.start(2)] +
                    f'{target_mass:.0f}' +
                    mjcf_src[m.end(2):m.start(4)] +
                    new_inertia_str +
                    mjcf_src[m.end(4):])

    tmp = tempfile.NamedTemporaryFile(suffix='.xml', delete=False, mode='w')
    tmp.write(mjcf_patched)
    tmp.close()
    print(f'  Structure mass: {old_mass:.0f} -> {target_mass:.0f} kg '
          f'(ratio {71.0/target_mass*100:.0f}%)')
    return tmp.name


def run_simulation(mjcf_path, cfg, label):
    print(f'\n{"="*70}')
    print(f'  Running: {label}')
    print(f'{"="*70}')

    sim = SimulationLoop(mjcf_path=mjcf_path, urdf_path=URDF, config=cfg)
    sim.setup(n_steps=N_STEPS, start_a=2, start_b=2)

    t0 = time.time()
    log = sim.run(verbose=True)
    wall = time.time() - t0
    print(f'  Wall time: {wall:.1f}s')
    return log


def print_metrics(log, label):
    t = np.array(log.t)
    Ln = np.array(log.L_com_norm)
    euler = np.array(log.struct_euler_deg)
    sp = np.array(log.struct_pos)
    hw_phys = np.array(log.hw_physical)
    hw_norms = np.linalg.norm(hw_phys, axis=1)
    nf_nmpc = sum(1 for x in log.nmpc_ok if not x)
    nf_qp = sum(1 for x in log.qp_ok if not x)

    print(f'\n  {"─"*50}')
    print(f'  {label}')
    print(f'  {"─"*50}')
    print(f'  Docks:           {len(log.dock_events)}/{N_STEPS}')
    print(f'  Max |angle|:     {np.max(np.abs(euler)):.2f}°')
    print(f'  Final rotation:  roll={euler[-1,0]:.2f}° '
          f'pitch={euler[-1,1]:.2f}° yaw={euler[-1,2]:.2f}°')
    print(f'  Struct drift:    {np.linalg.norm(sp[-1]-sp[0])*100:.1f} cm')
    print(f'  Max ||L_com||:   {Ln.max():.2f} Nms')
    print(f'  Max ||hw_phys||: {hw_norms.max():.2f} Nms')
    print(f'  hw violations:   {np.sum(hw_norms > 5.0)}/{len(hw_norms)}')
    print(f'  NMPC fails:      {nf_nmpc}/{len(log.nmpc_ok)}')
    print(f'  QP fails:        {nf_qp}/{len(log.qp_ok)}')
    return {
        'docks': len(log.dock_events),
        'max_angle': np.max(np.abs(euler)),
        'max_hw': hw_norms.max(),
        'nmpc_fails': nf_nmpc,
    }


# ── Run both variants ─────────────────────────────────────────────────────────

mjcf_tmp = create_patched_mjcf(TARGET_STRUCT_MASS)

# Common config
base_cfg = dict(
    tau_max=20.0,
    hw_min=np.full(3, -5.0),
    hw_max=np.full(3, 5.0),
    L_max=10.0,
    tau_w_max=5.0,
    nmpc_Wv=10.0,
    nmpc_W_hw=0.0,
    t_settle_final=20.0,
)

# Variant 1: Legacy AOCS (L_dot only), with original ctrlrange behavior
# Clip to 0.5 Nm to reproduce the status_report baseline
cfg_legacy = SimConfig(
    **base_cfg,
    aocs_use_H_estimator=False,
    aocs_K_hw=2.0,
    aocs_tau_w_max=0.5,     # reproduce original ctrlrange clip (MuJoCo had ±0.5)
)

log_legacy = run_simulation(mjcf_tmp, cfg_legacy, 'Legacy AOCS (-L_dot, tau_w≤0.5Nm)')
log_legacy.save(os.path.join(OUTDIR_LOGS, 'aocs_legacy_14pct.json'))

# Variant 2: H_{r/O} estimator AOCS
# Same torque budget as legacy (0.5 Nm) to isolate the effect of the
# correct feedforward DIRECTION (H_dot vs L_dot only).
cfg_H = SimConfig(
    **base_cfg,
    aocs_use_H_estimator=True,
    aocs_tau_w_max=0.5,         # same budget as legacy
    aocs_filter_tau=0.032,      # 32 ms → ~5 Hz cutoff
    aocs_K_omega=2.0,           # modest attitude damping
    aocs_K_h=0.1,               # gentle desaturation
)

log_H = run_simulation(mjcf_tmp, cfg_H, 'H_{r/O} estimator AOCS')
log_H.save(os.path.join(OUTDIR_LOGS, 'aocs_H_estimator_14pct.json'))

os.unlink(mjcf_tmp)

# ── Print comparison ──────────────────────────────────────────────────────────

print(f'\n\n{"="*70}')
print(f'  COMPARISON — 14% mass ratio ({TARGET_STRUCT_MASS:.0f} kg structure)')
print(f'{"="*70}')

m1 = print_metrics(log_legacy, 'Legacy AOCS (-L_dot)')
m2 = print_metrics(log_H, 'H_{r/O} estimator AOCS')

print(f'\n  {"="*50}')
print(f'  Improvement:')
print(f'    Rotation: {m1["max_angle"]:.1f}° → {m2["max_angle"]:.1f}° '
      f'({(1 - m2["max_angle"]/max(m1["max_angle"],0.01))*100:.0f}% reduction)')
print(f'    Docks:    {m1["docks"]}/{N_STEPS} → {m2["docks"]}/{N_STEPS}')
print(f'  {"="*50}')


# ── Plot comparison figures ───────────────────────────────────────────────────

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(5, 1, figsize=(14, 22), sharex=True)
fig.suptitle(f'AOCS Comparison — {TARGET_STRUCT_MASS:.0f} kg structure (14% ratio)',
             fontsize=14, fontweight='bold')

t1 = np.array(log_legacy.t)
t2 = np.array(log_H.t)

# 1. Structure rotation
ax = axes[0]
euler1 = np.array(log_legacy.struct_euler_deg)
euler2 = np.array(log_H.struct_euler_deg)
for i, lbl in enumerate(['roll', 'pitch', 'yaw']):
    ax.plot(t1, euler1[:, i], '--', alpha=0.7, label=f'{lbl} (legacy)')
    ax.plot(t2, euler2[:, i], '-', alpha=0.9, label=f'{lbl} (H estimator)')
ax.set_ylabel('Structure rotation [°]')
ax.legend(ncol=3, fontsize=8)
ax.set_title('Structure attitude')
ax.grid(True, alpha=0.3)

# 2. hw physical (wheel momentum)
ax = axes[1]
hw1 = np.array(log_legacy.hw_physical)
hw2 = np.array(log_H.hw_physical)
hw1_norm = np.linalg.norm(hw1, axis=1)
hw2_norm = np.linalg.norm(hw2, axis=1)
ax.plot(t1, hw1_norm, '--', label='||hw|| legacy', alpha=0.7)
ax.plot(t2, hw2_norm, '-', label='||hw|| H estimator', alpha=0.9)
ax.axhline(5.0, color='r', linestyle=':', alpha=0.5, label='h_max')
ax.set_ylabel('||hw|| [Nms]')
ax.legend(fontsize=8)
ax.set_title('Wheel angular momentum')
ax.grid(True, alpha=0.3)

# 3. tau_w (wheel torque command)
ax = axes[2]
tw1 = np.array(log_legacy.tau_w)
tw2 = np.array(log_H.tau_w)
tw1_norm = np.linalg.norm(tw1, axis=1)
tw2_norm = np.linalg.norm(tw2, axis=1)
ax.plot(t1, tw1_norm, '--', label='||τ_w|| legacy', alpha=0.7)
ax.plot(t2, tw2_norm, '-', label='||τ_w|| H estimator', alpha=0.9)
ax.set_ylabel('||τ_w|| [Nm]')
ax.legend(fontsize=8)
ax.set_title('Wheel torque command')
ax.grid(True, alpha=0.3)

# 4. H_{r/O} and H_dot (estimator diagnostics, H variant only)
ax = axes[3]
if log_H.H_rO:
    H_rO = np.array(log_H.H_rO)
    H_dot = np.array(log_H.H_dot_est)
    H_rO_norm = np.linalg.norm(H_rO, axis=1)
    H_dot_norm = np.linalg.norm(H_dot, axis=1)
    ax.plot(t2, H_rO_norm, '-', label='||H_{r/O}||', color='C0')
    ax.plot(t2, H_dot_norm, '-', label='||Ḣ_{r/O}||', color='C1', alpha=0.7)
    # Also plot L_com norm for comparison
    L1_norm = np.array(log_legacy.L_com_norm)
    L2_norm = np.array(log_H.L_com_norm)
    ax.plot(t1, L1_norm, '--', label='||L_com|| legacy', color='C2', alpha=0.5)
    ax.plot(t2, L2_norm, ':', label='||L_com|| H est.', color='C3', alpha=0.5)
ax.set_ylabel('[Nms] / [Nm]')
ax.legend(fontsize=8)
ax.set_title('Angular momentum: H_{r/O} vs L_com')
ax.grid(True, alpha=0.3)

# 5. Validation: H_dot vs qfrc_constraint
ax = axes[4]
if log_H.qfrc_constraint_torque and log_H.H_dot_est:
    qfrc = np.array(log_H.qfrc_constraint_torque)
    H_dot = np.array(log_H.H_dot_est)
    qfrc_norm = np.linalg.norm(qfrc, axis=1)
    H_dot_norm = np.linalg.norm(H_dot, axis=1)
    ax.plot(t2, qfrc_norm, '-', label='||qfrc_constraint[3:6]|| (ground truth)', alpha=0.7)
    ax.plot(t2, H_dot_norm, '--', label='||Ḣ_est|| (estimator)', alpha=0.7)
    # Residual
    resid = np.linalg.norm(qfrc + H_dot, axis=1)
    ax.plot(t2, resid, ':', label='||qfrc + Ḣ_est|| (residual)', color='red', alpha=0.5)
ax.set_ylabel('[Nm]')
ax.set_xlabel('Time [s]')
ax.legend(fontsize=8)
ax.set_title('Estimator validation: Ḣ_est vs MuJoCo ground truth')
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(OUTDIR_FIGS, 'aocs_comparison_14pct.png')
plt.savefig(save_path, dpi=150)
print(f'\n  Saved comparison figure: {save_path}')

# Also generate the standard 9-panel plot for each variant
fig1 = SimulationLoop.plot(log_legacy, cfg=cfg_legacy,
    save_path=os.path.join(OUTDIR_FIGS, 'aocs_legacy_14pct.png'))
print(f'  Saved: results/figures/aocs_legacy_14pct.png')

fig2 = SimulationLoop.plot(log_H, cfg=cfg_H,
    save_path=os.path.join(OUTDIR_FIGS, 'aocs_H_estimator_14pct.png'))
print(f'  Saved: results/figures/aocs_H_estimator_14pct.png')
