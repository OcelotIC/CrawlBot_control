#!/usr/bin/env python3
"""AOCS validation — Compare legacy vs H_est modes with orbital fix.

Runs 3-step crawling at 8% mass ratio (888 kg structure) with both AOCS modes
and generates diagnostic plots showing the impact of the orbital correction.

Also renders MuJoCo frames at key moments (dock events, phase transitions).

Usage:
    PYTHONPATH=. MUJOCO_GL=osmesa python3 scripts/validate_aocs_fix.py
    # or with disabled rendering (plots only, no robot images):
    PYTHONPATH=. MUJOCO_GL=disabled python3 scripts/validate_aocs_fix.py
"""
import os, sys, time
import numpy as np

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)

from crawlbot.simulation.sim_loop import SimulationLoop
from crawlbot.simulation.config import SimConfig
from crawlbot.simulation.plotting import plot_simulation

URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')
MJCF_SMALL = os.path.join(_root, 'models', 'VISPA_crawling_rwa3_8pct.xml')
MJCF_BIG = os.path.join(_root, 'models', 'VISPA_crawling_rwa3_8pct_hw50.xml')
OUTDIR = os.path.join(_root, 'results', 'figures')
LOGDIR = os.path.join(_root, 'results', 'logs')
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(LOGDIR, exist_ok=True)

N_STEPS = 3


def run_sim(cfg, label, mjcf=None):
    if mjcf is None:
        mjcf = MJCF_SMALL
    print(f'\n{"="*70}')
    print(f'  {label}')
    print(f'{"="*70}')
    sim = SimulationLoop(mjcf_path=mjcf, urdf_path=URDF, config=cfg)
    sim.setup(n_steps=N_STEPS, start_a=2, start_b=2)
    t0 = time.time()
    log = sim.run(verbose=True)
    wall = time.time() - t0
    print(f'  Wall time: {wall:.1f}s, {len(log.t)} NMPC steps')
    return log, sim


def extract_metrics(log, label):
    euler = np.array(log.struct_euler_deg)
    Lnorm = np.array(log.L_com_norm)
    e_torso = np.array(log.e_torso_pos) * 100  # cm
    nd = len(log.dock_events)
    nf_nmpc = sum(1 for x in log.nmpc_ok if not x)
    rot = np.max(np.abs(euler))

    hw_phys = np.array(log.hw_physical) if log.hw_physical else np.zeros((len(log.t), 3))
    hwn = np.linalg.norm(hw_phys, axis=1) if hw_phys.size > 3 else np.zeros(len(log.t))

    print(f'\n  ── {label} ──')
    print(f'  Docks:           {nd}/{N_STEPS}')
    print(f'  Max |angle|:     {rot:.2f}°')
    print(f'  Final euler:     r={euler[-1,0]:.1f}° p={euler[-1,1]:.1f}° y={euler[-1,2]:.1f}°')
    print(f'  Max ||L_com||:   {Lnorm.max():.2f} Nm·s')
    print(f'  Max ||hw||:      {hwn.max():.2f} Nm·s')
    print(f'  Torso error:     max={e_torso.max():.1f} cm, mean={e_torso.mean():.1f} cm')
    print(f'  NMPC fails:      {nf_nmpc}/{len(log.nmpc_ok)}')
    return dict(docks=nd, rot=rot, L_peak=Lnorm.max(), hw_peak=hwn.max(),
                torso_err_max=e_torso.max(), torso_err_mean=e_torso.mean(),
                nmpc_fails=nf_nmpc)


# ── Config A: Legacy AOCS (spin-only feedforward) ──────────────────────────
cfg_legacy = SimConfig(
    tau_max=20.0,
    hw_min=np.full(3, -5.0), hw_max=np.full(3, 5.0),
    L_max=10.0, tau_w_max=5.0, rwa_I_w=0.01,
    nmpc_Wv=10.0, nmpc_W_hw=0.0, t_settle_final=20.0,
    aocs_mode='legacy', aocs_use_H_estimator=False,
    aocs_K_hw=2.0, aocs_tau_w_max=0.5,
)

# ── Config B: H_est AOCS (full H_{r/O} + attitude damping) ─────────────────
# Gains tuned for the small wheel model (tau_w_max=0.5 Nm):
#   K_omega = 0.5 (gentle damping — wheels only have 0.5 Nm authority)
#   K_h = 0.1 (slow desaturation — don't compete with feedforward)
#   filter_tau = 0.05 (heavier filtering to reduce noise-induced torque)
cfg_H_est = SimConfig(
    tau_max=20.0,
    hw_min=np.full(3, -5.0), hw_max=np.full(3, 5.0),
    L_max=10.0, tau_w_max=5.0, rwa_I_w=0.01,
    nmpc_Wv=10.0, nmpc_W_hw=0.0, t_settle_final=20.0,
    aocs_mode='H_est', aocs_use_H_estimator=True,
    aocs_K_hw=2.0, aocs_tau_w_max=0.5,
    aocs_filter_tau=0.05,
    aocs_K_omega=0.5,
    aocs_K_h=0.1,
)

# ── Config C: H_est AOCS with big wheels (50 Nms capacity, 2 Nm torque) ────
cfg_H_est_big = SimConfig(
    tau_max=20.0,
    hw_min=np.full(3, -5.0), hw_max=np.full(3, 5.0),
    L_max=10.0, tau_w_max=5.0, rwa_I_w=0.1,      # bigger wheels
    nmpc_Wv=10.0, nmpc_W_hw=0.0, t_settle_final=20.0,
    aocs_mode='H_est', aocs_use_H_estimator=True,
    aocs_K_hw=2.0, aocs_tau_w_max=2.0,             # 2 Nm authority
    aocs_filter_tau=0.05,
    aocs_K_omega=5.0,                               # moderate damping
    aocs_K_h=0.2,
)

# ── Run simulations ────────────────────────────────────────────────────────
log_legacy, sim_legacy = run_sim(cfg_legacy, 'A: Legacy AOCS (small wheels, spin-only)')
log_H_est, sim_H_est = run_sim(cfg_H_est, 'B: H_est AOCS (small wheels)', mjcf=MJCF_SMALL)
log_H_big, sim_H_big = run_sim(cfg_H_est_big, 'C: H_est AOCS (big wheels, 2 Nm)', mjcf=MJCF_BIG)

# Save logs
log_legacy.save(os.path.join(LOGDIR, 'aocs_legacy_8pct.json'))
log_H_est.save(os.path.join(LOGDIR, 'aocs_H_est_small_8pct.json'))
log_H_big.save(os.path.join(LOGDIR, 'aocs_H_est_big_8pct.json'))

# ── Metrics comparison ─────────────────────────────────────────────────────
print(f'\n\n{"="*70}')
print(f'  AOCS COMPARISON — 8% mass ratio (888 kg structure)')
print(f'{"="*70}')
m_legacy = extract_metrics(log_legacy, 'A: Legacy (spin-only, small wheels)')
m_H_est = extract_metrics(log_H_est, 'B: H_est (small wheels)')
m_H_big = extract_metrics(log_H_big, 'C: H_est (big wheels, 2 Nm)')

print(f'\n  {"="*75}')
print(f'  {"Mode":<35} | {"Docks":>5} | {"Rotation":>8} | {"L_peak":>7} | {"Torso err":>9} | {"NMPC":>4}')
print(f'  {"─"*75}')
for tag, m in [('A: Legacy (small, spin-only)', m_legacy),
               ('B: H_est (small, 0.5 Nm)', m_H_est),
               ('C: H_est (big, 2.0 Nm)', m_H_big)]:
    print(f'  {tag:<35} | {m["docks"]}/3   | {m["rot"]:6.2f}°  | {m["L_peak"]:5.2f}  | {m["torso_err_mean"]:5.1f} cm   | {m["nmpc_fails"]:>4}')
print(f'  {"="*75}')


# ── Generate 9-panel diagnostic plots ──────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print('\n  Generating diagnostic plots...')
plot_simulation(log_legacy, save_path=os.path.join(OUTDIR, 'diag_legacy_8pct.png'), cfg=cfg_legacy)
plot_simulation(log_H_est, save_path=os.path.join(OUTDIR, 'diag_H_est_small_8pct.png'), cfg=cfg_H_est)
plot_simulation(log_H_big, save_path=os.path.join(OUTDIR, 'diag_H_est_big_8pct.png'), cfg=cfg_H_est_big)
print(f'  Saved: diag_legacy_8pct.png, diag_H_est_small_8pct.png, diag_H_est_big_8pct.png')


# ── Side-by-side comparison plot ───────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)
fig.suptitle('AOCS Comparison — 8% mass ratio, 3 steps',
             fontsize=14, fontweight='bold')

configs = [
    (log_legacy, 'A: Legacy (spin-only, 0.5 Nm)', '--', 'C3'),
    (log_H_est, 'B: H_est (small wheels, 0.5 Nm)', ':', 'C1'),
    (log_H_big, 'C: H_est (big wheels, 2.0 Nm)', '-', 'C0'),
]

# Panel 1: Structure rotation
ax = axes[0]
for log, lbl, ls, c in configs:
    t = np.array(log.t)
    euler = np.array(log.struct_euler_deg)
    ax.plot(t, np.max(np.abs(euler), axis=1), ls, color=c, label=lbl, lw=2)
ax.set_ylabel('Max |angle| [°]')
ax.set_title('Structure rotation')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

# Panel 2: Angular momentum
ax = axes[1]
for log, lbl, ls, c in configs:
    t = np.array(log.t)
    Lnorm = np.array(log.L_com_norm)
    ax.plot(t, Lnorm, ls, color=c, label=lbl, lw=2)
ax.axhline(10.0, color='r', ls=':', alpha=0.5, label='$L_{max}$')
ax.set_ylabel('$||L_{com}||$ [Nm·s]')
ax.set_title('Robot angular momentum')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

# Panel 3: Torso tracking error
ax = axes[2]
for log, lbl, ls, c in configs:
    t = np.array(log.t)
    e_t = np.array(log.e_torso_pos) * 100
    ax.plot(t, e_t, ls, color=c, label=lbl, lw=2)
ax.set_ylabel('$||e_{torso}||$ [cm]')
ax.set_title('Torso tracking error')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

# Panel 4: Wheel momentum
ax = axes[3]
for log, lbl, ls, c in configs:
    t = np.array(log.t)
    hw = np.array(log.hw_physical) if log.hw_physical else np.zeros((len(t), 3))
    if hw.size > 3:
        ax.plot(t, np.linalg.norm(hw, axis=1), ls, color=c, label=lbl, lw=2)
ax.axhline(5.0, color='r', ls=':', alpha=0.5, label='$h_{w,max}$')
ax.set_ylabel('$||h_w||$ [Nm·s]')
ax.set_xlabel('Time [s]')
ax.set_title('Wheel angular momentum')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(OUTDIR, 'aocs_comparison_8pct.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'  Saved: {save_path}')


# ── MuJoCo frame rendering (if renderer available) ────────────────────────
try:
    import mujoco
    # Use the best-performing sim for rendering
    best_sim = sim_H_big if m_H_big['docks'] >= m_legacy['docks'] else sim_legacy
    best_cfg = cfg_H_est_big if m_H_big['docks'] >= m_legacy['docks'] else cfg_legacy
    best_mjcf = MJCF_BIG if m_H_big['docks'] >= m_legacy['docks'] else MJCF_SMALL
    renderer = mujoco.Renderer(best_sim.mj_model, height=480, width=640)

    print('\n  Rendering robot frames...')
    sim_render = SimulationLoop(mjcf_path=best_mjcf, urdf_path=URDF, config=best_cfg)
    sim_render.setup(n_steps=N_STEPS, start_a=2, start_b=2)

    # Capture frames at specific times
    frame_times = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0]
    frames = {}

    # Run through simulation capturing frames
    mujoco.mj_forward(sim_render.mj_model, sim_render.mj_data)
    for ft in frame_times:
        # Step simulation to target time
        while sim_render.mj_data.time < ft:
            mujoco.mj_step(sim_render.mj_model, sim_render.mj_data)

        renderer.update_scene(sim_render.mj_data,
                              camera=mujoco.MjvCamera())
        img = renderer.render()
        frames[ft] = img.copy()

    # Save individual frames
    from PIL import Image
    for ft, img in frames.items():
        img_path = os.path.join(OUTDIR, f'robot_t{ft:.0f}s.png')
        Image.fromarray(img).save(img_path)
        print(f'    t={ft:.0f}s → {img_path}')

    # Create montage
    fig_montage, axes_m = plt.subplots(1, len(frames), figsize=(4*len(frames), 4))
    for i, (ft, img) in enumerate(sorted(frames.items())):
        axes_m[i].imshow(img)
        axes_m[i].set_title(f't = {ft:.0f}s', fontsize=10)
        axes_m[i].axis('off')
    fig_montage.suptitle('VISPA Crawling — 3-step locomotion (H_est AOCS)',
                         fontsize=13, fontweight='bold')
    plt.tight_layout()
    montage_path = os.path.join(OUTDIR, 'robot_montage.png')
    plt.savefig(montage_path, dpi=150, bbox_inches='tight')
    print(f'  Saved montage: {montage_path}')
    renderer.close()

except Exception as e:
    print(f'\n  Rendering skipped: {e}')
    print('  (Run with MUJOCO_GL=osmesa for frame rendering)')


print(f'\n{"="*70}')
print(f'  All outputs in: {OUTDIR}')
print(f'{"="*70}')
