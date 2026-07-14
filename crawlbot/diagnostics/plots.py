"""Generate the fixed set of 8 diagnostic figures from SimLog.

Every simulation produces the same plots. Missing data shows "N/A".
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Phase colors (M7: two-phase state machine — DS and SS only)
_PHASE_COLORS = {'DS': 'blue', 'SS': 'orange'}


def _to_np(lst):
    if len(lst) == 0:
        return np.array([])
    return np.array(lst, dtype=float)


def _to_np2d(lst, cols=3):
    if len(lst) == 0:
        return np.empty((0, cols))
    arr = np.array(lst, dtype=float)
    if arr.ndim == 2:
        # List of per-step vectors — numpy stacked them directly.
        # Trust the inner dimension regardless of the `cols` hint so
        # DOF-generic callers (e.g. 6- vs 7-DOF arms) don't break.
        return arr
    # Flat/1D: best-effort infer cols from total size
    if arr.size > 0 and arr.size % max(len(lst), 1) == 0:
        cols = arr.size // len(lst)
    return arr.reshape(-1, cols)


def _phase_shading(ax, t, phases):
    """Add phase-colored vertical spans."""
    if len(t) < 2 or len(phases) < 2:
        return
    prev = phases[0]
    start = t[0]
    for i in range(1, len(phases)):
        if phases[i] != prev or i == len(phases) - 1:
            color = _PHASE_COLORS.get(prev, 'gray')
            ax.axvspan(start, t[i], alpha=0.06, color=color, lw=0)
            prev = phases[i]
            start = t[i]


def _dock_lines(ax, dock_events):
    for ev in dock_events:
        ax.axvline(ev['t'], color='green', ls='--', lw=0.8, alpha=0.6)


def _has(log, attr):
    return hasattr(log, attr) and len(getattr(log, attr)) > 0


def _na_text(ax, msg="N/A — not logged"):
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha='center', va='center',
            fontsize=12, color='gray')
    ax.set_xticks([])
    ax.set_yticks([])


def generate_plots(log, output_dir, cfg=None, dpi=150):
    """Generate all 8 diagnostic figures and save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    t = _to_np(log.t) if _has(log, 't') else np.array([])
    phases = log.phase if _has(log, 'phase') else []
    docks = log.dock_events if _has(log, 'dock_events') else []

    hw_max = 5.0
    tau_w_max = 2.5
    tau_max = 20.0
    if cfg is not None:
        hw_max = float(np.max(np.abs(cfg.hw_max)))
        tau_w_max = float(cfg.tau_w_max)
        tau_max = float(cfg.tau_max)

    _fig1_tracking(t, phases, docks, log, output_dir, dpi)
    _fig2_momentum(t, phases, docks, log, hw_max, tau_w_max, output_dir, dpi)
    _fig3_com(t, phases, docks, log, output_dir, dpi)
    _fig4_energy(t, phases, log, output_dir, dpi)
    _fig5_nmpc(log, output_dir, dpi)
    _fig6_wrenches(t, phases, log, output_dir, dpi)
    _fig7_joints(t, phases, log, tau_max, output_dir, dpi)
    _fig8_snapshots(log, output_dir, dpi)
    # M7 per-axis 6D tracking — separate EE and torso panels so temporal
    # dynamics are visible (the aggregate metrics hide transients).
    _fig9_ee_6d_tracking(t, phases, docks, log, output_dir, dpi, cfg=cfg)
    _fig10_torso_6d_tracking(t, phases, docks, log, output_dir, dpi, cfg=cfg)


# ── Figure 1: Tracking overview ─────────────────────────────────────────────

def _fig1_tracking(t, phases, docks, log, out, dpi):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # (a) Torso position error
    ax = axes[0]
    if _has(log, 'e_torso_pos'):
        e = _to_np(log.e_torso_pos) * 1000
        ax.plot(t, e, 'k', lw=0.8)
        _phase_shading(ax, t, phases)
        _dock_lines(ax, docks)
    else:
        _na_text(ax)
    ax.set_ylabel('Torso pos err [mm]')
    ax.set_title('Fig 1: Tracking Overview')

    # (b) Torso orientation error
    ax = axes[1]
    if _has(log, 'e_torso_ori'):
        ax.plot(t, _to_np(log.e_torso_ori), 'k', lw=0.8)
        _phase_shading(ax, t, phases)
        _dock_lines(ax, docks)
    else:
        _na_text(ax)
    ax.set_ylabel('Torso ori err [deg]')

    # (c) EE position error (SS only)
    ax = axes[2]
    if _has(log, 'e_ee_pos'):
        e = _to_np(log.e_ee_pos) * 1000
        ax.plot(t, e, 'k', lw=0.8)
        _phase_shading(ax, t, phases)
        _dock_lines(ax, docks)
        for ev in docks:
            idx = int(np.argmin(np.abs(t - ev['t'])))
            ax.plot(t[idx], e[idx], 'ro', ms=5)
    else:
        _na_text(ax)
    ax.set_ylabel('EE pos err [mm]')

    # (d) EE orientation error
    ax = axes[3]
    if _has(log, 'e_ee_ori'):
        ax.plot(t, _to_np(log.e_ee_ori), 'k', lw=0.8)
        _phase_shading(ax, t, phases)
        _dock_lines(ax, docks)
        for ev in docks:
            idx = int(np.argmin(np.abs(t - ev['t'])))
            ax.plot(t[idx], _to_np(log.e_ee_ori)[idx], 'ro', ms=5)
    else:
        _na_text(ax)
    ax.set_ylabel('EE ori err [deg]')
    ax.set_xlabel('Time [s]')

    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig1_tracking.png'), dpi=dpi)
    plt.close(fig)


# ── Figure 2: Momentum & AOCS ───────────────────────────────────────────────

def _fig2_momentum(t, phases, docks, log, hw_max, tw_max, out, dpi):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # (a) hw 3 axes
    ax = axes[0]
    if _has(log, 'hw_physical'):
        hw = _to_np2d(log.hw_physical)
        for i, c in enumerate(['r', 'g', 'b']):
            ax.plot(t, hw[:, i], c, lw=0.8, label=f'hw_{["x","y","z"][i]}')
        ax.axhline(hw_max, color='red', ls='--', lw=0.8)
        ax.axhline(-hw_max, color='red', ls='--', lw=0.8)
        peak = float(np.max(np.linalg.norm(hw, axis=1)))
        ax.set_title(f'Fig 2: Momentum & AOCS  (peak hw={peak:.2f} Nms, '
                     f'ratio={peak/hw_max:.2f})')
        ax.legend(fontsize=8, ncol=3, loc='upper right')
        _phase_shading(ax, t, phases)
    else:
        _na_text(ax)
    ax.set_ylabel('hw [Nms]')

    # (b) tau_w
    ax = axes[1]
    if _has(log, 'tau_w'):
        tw = _to_np2d(log.tau_w)
        for i, c in enumerate(['r', 'g', 'b']):
            ax.plot(t, tw[:, i], c, lw=0.8)
        ax.axhline(tw_max, color='red', ls='--', lw=0.8)
        ax.axhline(-tw_max, color='red', ls='--', lw=0.8)
        _phase_shading(ax, t, phases)
    else:
        _na_text(ax)
    ax.set_ylabel('tau_w [Nm]')

    # (c) Platform angular velocity
    ax = axes[2]
    if _has(log, 'omega_s'):
        om = np.degrees(_to_np2d(log.omega_s))
        for i, c in enumerate(['r', 'g', 'b']):
            ax.plot(t, om[:, i], c, lw=0.8)
        _phase_shading(ax, t, phases)
    else:
        _na_text(ax)
    ax.set_ylabel('omega_s [deg/s]')

    # (d) Platform attitude
    ax = axes[3]
    if _has(log, 'struct_euler_deg'):
        euler = _to_np2d(log.struct_euler_deg)
        for i, (c, lbl) in enumerate(zip(['r', 'g', 'b'], ['roll', 'pitch', 'yaw'])):
            ax.plot(t, euler[:, i], c, lw=0.8, label=lbl)
        ax.legend(fontsize=8, ncol=3)
        _phase_shading(ax, t, phases)
        _dock_lines(ax, docks)
    else:
        _na_text(ax)
    ax.set_ylabel('Euler [deg]')
    ax.set_xlabel('Time [s]')

    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig2_momentum_aocs.png'), dpi=dpi)
    plt.close(fig)


# ── Figure 3: CoM tracking ──────────────────────────────────────────────────

def _fig3_com(t, phases, docks, log, out, dpi):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # (a) CoM position
    ax = axes[0]
    if _has(log, 'r_com') and _has(log, 'r_com_ref'):
        rc = _to_np2d(log.r_com)
        rr = _to_np2d(log.r_com_ref)
        for i, lbl in enumerate(['x', 'y', 'z']):
            ax.plot(t, rc[:, i] * 1000, lw=0.8, label=f'{lbl} actual')
            ax.plot(t, rr[:, i] * 1000, '--', lw=0.8, label=f'{lbl} ref')
        ax.legend(fontsize=7, ncol=3)
        _phase_shading(ax, t, phases)
    else:
        _na_text(ax)
    ax.set_ylabel('CoM pos [mm]')
    ax.set_title('Fig 3: CoM Tracking & Centroidal Momentum')

    # (b) CoM velocity
    ax = axes[1]
    if _has(log, 'v_com') and _has(log, 'v_com_ref'):
        vc = _to_np2d(log.v_com) * 1000
        vr = _to_np2d(log.v_com_ref) * 1000
        for i, lbl in enumerate(['x', 'y', 'z']):
            ax.plot(t, vc[:, i], lw=0.8, label=f'{lbl} actual')
            ax.plot(t, vr[:, i], '--', lw=0.8, label=f'{lbl} ref')
        ax.legend(fontsize=7, ncol=3)
        _phase_shading(ax, t, phases)
    else:
        _na_text(ax)
    ax.set_ylabel('CoM vel [mm/s]')

    # (c) L_com
    ax = axes[2]
    if _has(log, 'L_com'):
        Lc = _to_np2d(log.L_com)
        for i, c in enumerate(['r', 'g', 'b']):
            ax.plot(t, Lc[:, i], c, lw=0.8, label=['Lx', 'Ly', 'Lz'][i])
        if _has(log, 'L_com_ref'):
            Lr = _to_np2d(log.L_com_ref)
            for i, c in enumerate(['r', 'g', 'b']):
                ax.plot(t, Lr[:, i], c + '--', lw=0.8)
        ax.legend(fontsize=8, ncol=3)
        _phase_shading(ax, t, phases)
    else:
        _na_text(ax)
    ax.set_ylabel('L_com [Nms]')
    ax.set_xlabel('Time [s]')

    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig3_com_momentum.png'), dpi=dpi)
    plt.close(fig)


# ── Figure 4: Energy & passivity ────────────────────────────────────────────

def _fig4_energy(t, phases, log, out, dpi):
    has_settling = (hasattr(log, 'settling_T') and len(log.settling_T) > 0)
    n_rows = 3 if has_settling else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 7 if has_settling else 5))

    # (a0) Setup-phase settling energy decay (optional, M4+)
    if has_settling:
        ax = axes[0]
        ts = _to_np(log.settling_t)
        Ts = _to_np(log.settling_T)
        Ts_plot = np.maximum(Ts, 1e-14)
        ax.semilogy(ts, Ts_plot, 'k', lw=1.0, label='T_kin')
        T_target = float(getattr(log, 'settling_T_target', 0.0))
        if T_target > 0:
            ax.axhline(T_target, color='red', ls='--', lw=0.8,
                       label=f'T_settle={T_target:.2e} J')
        n1 = int(getattr(log, 'settling_stage1_steps', 0))
        n2 = int(getattr(log, 'settling_stage2_steps', 0))
        dt_qp = ts[1] - ts[0] if len(ts) >= 2 else 0.01
        t_stage1_end = n1 * dt_qp
        if n1 > 0:
            ax.axvline(t_stage1_end, color='blue', ls=':', lw=0.8,
                       label=f'stage1 end ({n1} steps)')
        exit_reason = getattr(log, 'settling_exit_reason', '')
        ax.set_title(f'Fig 4: Energy & Passivity — setup-phase settling '
                     f'(stage1={n1}, stage2={n2}, exit={exit_reason})')
        ax.set_ylabel('Setup T_kin [J]')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        start_idx = 1  # axes[0] used for settling
    else:
        start_idx = 0

    # (a) Kinetic energy during the main sim
    ax = axes[start_idx]
    if _has(log, 'T_kinetic'):
        Tk = _to_np(log.T_kinetic)
        Tk_plot = np.maximum(Tk, 1e-12)
        ax.semilogy(t, Tk_plot, 'k', lw=0.8)
        _phase_shading(ax, t, phases)
    else:
        _na_text(ax)
    ax.set_ylabel('Run T_kin [J]')
    if not has_settling:
        ax.set_title('Fig 4: Energy & Passivity')

    # (b) Passivity LHS
    ax = axes[start_idx + 1]
    _na_text(ax, "N/A — passivity_lhs not yet logged")
    ax.set_ylabel('Passivity LHS')
    ax.set_xlabel('Time [s]')

    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig4_energy_passivity.png'), dpi=dpi)
    plt.close(fig)


# ── Figure 5: NMPC solver health ────────────────────────────────────────────

def _fig5_nmpc(log, out, dpi):
    fig, axes = plt.subplots(3, 1, figsize=(12, 7))

    n_nmpc = len(log.nmpc_time_ms) if _has(log, 'nmpc_time_ms') else 0
    idx = np.arange(n_nmpc)

    # (a) Solve time
    ax = axes[0]
    if n_nmpc > 0:
        times = _to_np(log.nmpc_time_ms)
        ax.bar(idx, times, width=1.0, color='steelblue', edgecolor='none')
        ax.axhline(50, color='red', ls='--', lw=1, label='50 ms')
        ax.legend(fontsize=8)
    else:
        _na_text(ax)
    ax.set_ylabel('Solve time [ms]')
    ax.set_title('Fig 5: NMPC Solver Health')

    # (b) Status
    ax = axes[1]
    if _has(log, 'nmpc_status'):
        status = _to_np(log.nmpc_status)
        colors = ['green' if s == 0 else 'orange' if s == 1 else 'red'
                  for s in status]
        ax.scatter(idx, status, c=colors, s=8)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['OK', 'max_iter', 'infeasible'])
    elif n_nmpc > 0:
        ok = np.array(log.nmpc_ok, dtype=float)
        colors = ['green' if o else 'red' for o in log.nmpc_ok]
        ax.scatter(idx, ok, c=colors, s=8)
    else:
        _na_text(ax)
    ax.set_ylabel('Status')

    # (c) Cost
    ax = axes[2]
    if _has(log, 'nmpc_cost'):
        cost = _to_np(log.nmpc_cost)
        finite = np.isfinite(cost)
        if np.any(finite):
            ax.semilogy(idx[finite], cost[finite], 'k.', ms=2)
    else:
        _na_text(ax)
    ax.set_ylabel('NMPC cost')
    ax.set_xlabel('NMPC call index')

    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig5_nmpc_health.png'), dpi=dpi)
    plt.close(fig)


# ── Figure 6: Contact wrenches ──────────────────────────────────────────────

def _fig6_wrenches(t, phases, log, out, dpi):
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    ax = axes[0]
    if _has(log, 'lambda_ref') and _has(log, 'lambda_qp'):
        lr = _to_np2d(log.lambda_ref, 12)
        lq = _to_np2d(log.lambda_qp, 12)
        ax.plot(t, np.linalg.norm(lr, axis=1), 'b', lw=0.8, label='NMPC ref')
        ax.plot(t, np.linalg.norm(lq, axis=1), 'r', lw=0.8, alpha=0.7, label='QP actual')
        ax.legend(fontsize=8)
        _phase_shading(ax, t, phases)
    else:
        _na_text(ax)
    ax.set_ylabel('||lambda|| [N]')
    ax.set_title('Fig 6: Contact Wrenches')

    ax = axes[1]
    if _has(log, 'lambda_ref') and _has(log, 'lambda_qp'):
        residual = np.linalg.norm(lq - lr, axis=1)
        ax.plot(t, residual, 'k', lw=0.8)
        _phase_shading(ax, t, phases)
    else:
        _na_text(ax)
    ax.set_ylabel('||lambda_qp - lambda_ref||')
    ax.set_xlabel('Time [s]')

    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig6_contact_wrenches.png'), dpi=dpi)
    plt.close(fig)


# ── Figure 7: Joint-level ───────────────────────────────────────────────────

def _fig7_joints(t, phases, log, tau_max, out, dpi):
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    ax = axes[0]
    if _has(log, 'tau'):
        tau_arr = _to_np2d(log.tau, 12)
        for j in range(tau_arr.shape[1]):
            ax.plot(t, tau_arr[:, j], lw=0.5, alpha=0.6)
        ax.axhline(tau_max, color='red', ls='--', lw=0.8)
        ax.axhline(-tau_max, color='red', ls='--', lw=0.8)
        _phase_shading(ax, t, phases)
    else:
        _na_text(ax)
    ax.set_ylabel('Joint torques [Nm]')
    ax.set_title('Fig 7: Joint-Level Diagnostics')

    ax = axes[1]
    _na_text(ax, "N/A — joint velocities not logged")
    ax.set_ylabel('Joint velocities [rad/s]')
    ax.set_xlabel('Time [s]')

    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig7_joints.png'), dpi=dpi)
    plt.close(fig)


# ── Figure 8: MuJoCo snapshots ──────────────────────────────────────────────

def _fig8_snapshots(log, out, dpi):
    """Create a contact sheet grid from pre-rendered snapshot images."""
    snap_dir = out
    from glob import glob
    pngs = sorted(glob(os.path.join(snap_dir, 'snap_*.png')))
    if not pngs:
        # Create a placeholder figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        _na_text(ax, "N/A — no snapshots rendered\n"
                     "(requires MUJOCO_GL=osmesa)")
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'fig8_snapshots_grid.png'), dpi=dpi)
        plt.close(fig)
        return

    from PIL import Image
    images = [Image.open(p) for p in pngs[:12]]
    n = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_2d(axes)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        axes[r, c].imshow(np.array(img))
        label = os.path.splitext(os.path.basename(pngs[i]))[0]
        axes[r, c].set_title(label.replace('snap_', ''), fontsize=8)
        axes[r, c].axis('off')
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis('off')
    fig.suptitle('Fig 8: MuJoCo Snapshots', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig8_snapshots_grid.png'), dpi=dpi)
    plt.close(fig)


# ── M7 Figure 9: EE 6D tracking ─────────────────────────────────────────────
# Position error per axis [mm] + orientation error [deg] vs time, with
# SS trajectory markers at τ=0.5 and τ=1.0 per step. Aggregate metrics
# hide these temporal dynamics; this plot exposes them.

def _quat_diff_deg(q_act, q_ref):
    """Angle in degrees between two wxyz quaternions."""
    qa = np.asarray(q_act, dtype=float)
    qr = np.asarray(q_ref, dtype=float)
    na = np.linalg.norm(qa); nr = np.linalg.norm(qr)
    if na < 1e-12 or nr < 1e-12:
        return 0.0
    qa = qa / na; qr = qr / nr
    dot = abs(float(qa @ qr))
    dot = min(1.0, max(-1.0, dot))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _ss_tau_markers(ax, t, phases, log):
    """Overlay vertical lines at τ=0.5 and τ=1.0 for each SS window."""
    if len(t) == 0 or len(phases) == 0:
        return
    # Walk SS runs; each run's first sample is τ≈0, last is the
    # trajectory deadline. Use inter_step_settles and aborted_steps to
    # locate SS starts reliably. Fallback: detect DS→SS edges.
    ss_runs = []
    prev = None
    start_i = None
    for i, p in enumerate(phases):
        if p == 'SS' and prev != 'SS':
            start_i = i
        if p != 'SS' and prev == 'SS' and start_i is not None:
            ss_runs.append((start_i, i - 1))
            start_i = None
        prev = p
    if prev == 'SS' and start_i is not None:
        ss_runs.append((start_i, len(phases) - 1))
    # For each run, draw τ=0.5 and τ=1.0 markers. Use the planner
    # T_step captured in _preplanner_stats if available; else fall back
    # to (end_t - start_t) of the SS run.
    T_steps = []
    stats = getattr(log, '_preplanner_stats', None)
    if stats is None and hasattr(log, 'inter_step_settles'):
        stats = None  # not stored on log by default; best-effort below
    for k, (s, e) in enumerate(ss_runs):
        t_start = t[s]
        # Prefer the real T_step from preplanner_stats (attribute may
        # not exist on SimLog; only available when caller attaches it).
        T_step = None
        if hasattr(log, 'preplanner_T_steps') and k < len(log.preplanner_T_steps):
            T_step = log.preplanner_T_steps[k]
        if T_step is None:
            T_step = float(t[e] - t_start)
        ax.axvline(t_start + 0.5 * T_step, color='gray', ls=':',  lw=1.0, alpha=0.6)
        ax.axvline(t_start + 1.0 * T_step, color='gray', ls='--', lw=1.0, alpha=0.7)


def _fig9_ee_6d_tracking(t, phases, docks, log, out, dpi, cfg=None):
    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
    fig.suptitle(
        'Fig 9 — EE 6D tracking (dashed=τ=1 trajectory end, dotted=τ=0.5)',
        fontsize=12)

    have_ee = _has(log, 'p_ee') and _has(log, 'p_ee_ref')
    if not have_ee or len(t) == 0:
        for ax in axes.ravel():
            _na_text(ax, "EE pose not logged")
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'fig9_ee_6d_tracking.png'), dpi=dpi)
        plt.close(fig)
        return

    p_ee = _to_np2d(log.p_ee, cols=3)
    p_ee_ref = _to_np2d(log.p_ee_ref, cols=3)
    n = min(len(t), len(p_ee), len(p_ee_ref))
    t_ = t[:n]
    err = (p_ee[:n] - p_ee_ref[:n]) * 1000.0   # mm
    err_norm = np.linalg.norm(err, axis=1)

    labels = ['x', 'y', 'z']
    for ax, k, lab in zip(axes[:3], range(3), labels):
        _phase_shading(ax, t_, phases[:n])
        ax.axhline(0.0, color='black', lw=0.5, alpha=0.4)
        ax.plot(t_, err[:, k], lw=1.3, color=f'C{k}')
        ax.set_ylabel(f'e_p_ee[{lab}] [mm]')
        ax.grid(True, alpha=0.3)
        _ss_tau_markers(ax, t_, phases[:n], log)
        _dock_lines(ax, docks)

    ax = axes[3]
    _phase_shading(ax, t_, phases[:n])
    ax.plot(t_, err_norm, lw=1.4, color='k', label='||e_p_ee||')
    if cfg is not None:
        ax.axhline(cfg.weld_radius * 1000.0, color='red', ls='--',
                   lw=0.8, alpha=0.5, label=f'{cfg.weld_radius*1000:.1f} mm gate')
    ax.set_ylabel('||e_p_ee|| [mm]')
    ax.grid(True, alpha=0.3)
    _ss_tau_markers(ax, t_, phases[:n], log)
    _dock_lines(ax, docks)
    ax.legend(loc='upper right', fontsize=9)

    # Orientation error (deg)
    ax = axes[4]
    _phase_shading(ax, t_, phases[:n])
    if _has(log, 'q_ee') and _has(log, 'q_ee_ref'):
        q_ee = list(log.q_ee)
        q_ref = list(log.q_ee_ref)
        m = min(n, len(q_ee), len(q_ref))
        ori = np.array([_quat_diff_deg(q_ee[i], q_ref[i]) for i in range(m)])
        ax.plot(t_[:m], ori, lw=1.4, color='C3')
    elif _has(log, 'e_ee_ori'):
        eo = _to_np(log.e_ee_ori)[:n]
        ax.plot(t_[:len(eo)], eo, lw=1.4, color='C3')
    else:
        _na_text(ax, "no EE orientation data")
    if cfg is not None:
        ax.axhline(cfg.dock_ori_threshold_deg, color='red', ls='--',
                   lw=0.8, alpha=0.5,
                   label=f'{cfg.dock_ori_threshold_deg:.1f}° gate')
        ax.legend(loc='upper right', fontsize=9)
    ax.set_ylabel('e_ori_ee [deg]')
    ax.set_xlabel('t [s]')
    ax.grid(True, alpha=0.3)
    _ss_tau_markers(ax, t_, phases[:n], log)
    _dock_lines(ax, docks)

    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig9_ee_6d_tracking.png'), dpi=dpi)
    plt.close(fig)


# ── M7 Figure 10: Torso 6D tracking ─────────────────────────────────────────

def _fig10_torso_6d_tracking(t, phases, docks, log, out, dpi, cfg=None):
    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
    fig.suptitle(
        'Fig 10 — Torso 6D tracking (dashed=τ=1 trajectory end, dotted=τ=0.5)',
        fontsize=12)

    have = _has(log, 'p_torso') and _has(log, 'p_torso_ref')
    if not have or len(t) == 0:
        for ax in axes.ravel():
            _na_text(ax, "torso pose not logged")
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'fig10_torso_6d_tracking.png'), dpi=dpi)
        plt.close(fig)
        return

    p = _to_np2d(log.p_torso, cols=3)
    pr = _to_np2d(log.p_torso_ref, cols=3)
    n = min(len(t), len(p), len(pr))
    t_ = t[:n]
    err = (p[:n] - pr[:n]) * 1000.0
    err_norm = np.linalg.norm(err, axis=1)

    for ax, k, lab in zip(axes[:3], range(3), ['x', 'y', 'z']):
        _phase_shading(ax, t_, phases[:n])
        ax.axhline(0.0, color='black', lw=0.5, alpha=0.4)
        ax.plot(t_, err[:, k], lw=1.3, color=f'C{k}')
        ax.set_ylabel(f'e_p_torso[{lab}] [mm]')
        ax.grid(True, alpha=0.3)
        _ss_tau_markers(ax, t_, phases[:n], log)
        _dock_lines(ax, docks)

    ax = axes[3]
    _phase_shading(ax, t_, phases[:n])
    ax.plot(t_, err_norm, lw=1.4, color='k', label='||e_p_torso||')
    ax.set_ylabel('||e_p_torso|| [mm]')
    ax.grid(True, alpha=0.3)
    _ss_tau_markers(ax, t_, phases[:n], log)
    _dock_lines(ax, docks)
    ax.legend(loc='upper right', fontsize=9)

    # Orientation error — prefer the scalar log; fall back to quat diff.
    ax = axes[4]
    _phase_shading(ax, t_, phases[:n])
    if _has(log, 'e_torso_ori'):
        eo = _to_np(log.e_torso_ori)[:n]
        ax.plot(t_[:len(eo)], eo, lw=1.4, color='C3')
    elif _has(log, 'q_torso') and _has(log, 'q_torso_ref'):
        q_t = list(log.q_torso); q_r = list(log.q_torso_ref)
        m = min(n, len(q_t), len(q_r))
        ori = np.array([_quat_diff_deg(q_t[i], q_r[i]) for i in range(m)])
        ax.plot(t_[:m], ori, lw=1.4, color='C3')
    else:
        _na_text(ax, "no torso orientation data")
    if cfg is not None:
        ax.axhline(5.0, color='red', ls='--', lw=0.8, alpha=0.5,
                   label='5° ref gate')
        ax.legend(loc='upper right', fontsize=9)
    ax.set_ylabel('e_ori_torso [deg]')
    ax.set_xlabel('t [s]')
    ax.grid(True, alpha=0.3)
    _ss_tau_markers(ax, t_, phases[:n], log)
    _dock_lines(ax, docks)

    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig10_torso_6d_tracking.png'), dpi=dpi)
    plt.close(fig)
