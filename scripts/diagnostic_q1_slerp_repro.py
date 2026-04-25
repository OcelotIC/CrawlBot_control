"""Q1 — Step-1 orientation failure diagnosis.

Tests whether the 177° step-1 orientation error in the
mid-waypoint run (results/M7_1pct_3step_v22_t15_midwaypoint/) is a
SLERP hemisphere / quaternion-sign bug or a real kinematic failure.

Hypothesis tested:
  Q1-A: SLERP sub-segments choose different hemispheres → spurious
        180° in axis-angle rep.
  Q1-B: Quaternion sign issue elsewhere (R_mid sign).
  Q1-C: Reference is correct; actual really is 177° off because the
        controller cannot track. No representation bug.

Outputs:
  diagnostic/q1_step1_torso_ori_traces.png
  diagnostic/q1_step1_swing_ori_traces.png
  diagnostic/q1_slerp_repro_torso.png
  diagnostic/q1_slerp_repro_swing.png
"""
from __future__ import annotations

import os
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

SIM_LOG = os.path.join(_root, 'results', 'M7_1pct_3step_v22_t15_midwaypoint',
                       'sim_log.json')
IK_TRACE = os.path.join(_root, 'results', 'M7_1pct_3step_v22_t15_midwaypoint',
                        'ik_trace.json')
OUTDIR = os.path.join(_root, 'diagnostic')


def _quat_wxyz_to_R(q):
    """sim_log stores torso quat as wxyz (Pinocchio convention)."""
    return pin.Quaternion(q[0], q[1], q[2], q[3]).toRotationMatrix()


def _angle_between_quats(qa, qb):
    """Geodesic angle between two unit quaternions (wxyz)."""
    inner = abs(qa[0] * qb[0] + qa[1] * qb[1]
                + qa[2] * qb[2] + qa[3] * qb[3])
    return float(np.degrees(2 * np.arccos(np.clip(inner, -1.0, 1.0))))


def _angle_R_vs_R(R1, R2):
    """||log(R1^T R2)|| in degrees."""
    return float(np.degrees(np.linalg.norm(pin.log3(R1.T @ R2))))


def step1_traces():
    d = json.load(open(SIM_LOG))
    t = np.array(d['t'])
    phase = np.array(d['phase'])
    step_idx = np.array(d['step_idx'])
    mask = (step_idx == 1) & (phase == 'SS')
    t1 = t[mask]
    q_torso_ref = np.array(d['q_torso_ref'])[mask]
    q_torso_act = np.array(d['q_torso'])[mask]
    q_ee_ref = np.array(d['q_ee_ref'])[mask]
    q_ee_act = np.array(d['q_ee'])[mask]

    # Per-sample geodesic distances.
    init_torso_ref = q_torso_ref[0]
    init_swing_ref = q_ee_ref[0]
    n = len(t1)
    err_torso = np.zeros(n)
    err_swing = np.zeros(n)
    cmd_torso = np.zeros(n)   # ref deviation from initial ref
    cmd_swing = np.zeros(n)
    for i in range(n):
        err_torso[i] = _angle_between_quats(q_torso_ref[i], q_torso_act[i])
        err_swing[i] = _angle_between_quats(q_ee_ref[i], q_ee_act[i])
        cmd_torso[i] = _angle_between_quats(init_torso_ref, q_torso_ref[i])
        cmd_swing[i] = _angle_between_quats(init_swing_ref, q_ee_ref[i])

    # Sign-flip indicators.
    def _flip_indices(qs):
        out = []
        for i in range(1, len(qs)):
            ip = float(np.dot(qs[i - 1], qs[i]))
            if ip < 0:
                out.append(i)
        return out
    flips_tref = _flip_indices(q_torso_ref)
    flips_eref = _flip_indices(q_ee_ref)
    flips_tact = _flip_indices(q_torso_act)
    flips_eact = _flip_indices(q_ee_act)

    # ── Plot torso traces ────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax = axes[0]
    ax.plot(t1, cmd_torso, '-', color='#1f77b4', linewidth=2,
            label='|R_ref(t) ⊖ R_ref(0)|  (commanded reorient)')
    ax.plot(t1, err_torso, '-', color='#d62728', linewidth=2,
            label='|R_actual(t) ⊖ R_ref(t)|  (tracking error)')
    for fi in flips_tref:
        ax.axvline(t1[fi], ls=':', color='#1f77b4', alpha=0.4)
    for fi in flips_tact:
        ax.axvline(t1[fi], ls=':', color='#d62728', alpha=0.4)
    ax.set_ylabel('Torso orientation [°]')
    ax.set_title(f'Step 1 — torso orientation, midwaypoint run '
                 f'(q_torso_ref sign-flips: {len(flips_tref)}, '
                 f'q_torso_actual: {len(flips_tact)})')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(-5, 200)
    ax.axhline(180, color='gray', ls='--', linewidth=0.8)

    # Quaternion components — visual confirmation that q_torso_ref
    # is smooth (no sign flips).
    ax = axes[1]
    labels = ['w', 'x', 'y', 'z']
    for k in range(4):
        ax.plot(t1, q_torso_ref[:, k], '-', linewidth=1.5,
                label=f'q_ref[{labels[k]}]')
        ax.plot(t1, q_torso_act[:, k], '--', linewidth=1.0,
                color=ax.lines[-1].get_color(), alpha=0.7,
                label=f'q_act[{labels[k]}]' if k == 0 else None)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('Torso quaternion components')
    ax.set_title('Quaternion components — solid = ref, dashed = actual')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'q1_step1_torso_ori_traces.png'), dpi=130)
    plt.close(fig)

    # ── Plot swing-EE traces ────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax = axes[0]
    ax.plot(t1, cmd_swing, '-', color='#1f77b4', linewidth=2,
            label='|R_ee_ref(t) ⊖ R_ee_ref(0)|  (commanded reorient)')
    ax.plot(t1, err_swing, '-', color='#d62728', linewidth=2,
            label='|R_ee_actual(t) ⊖ R_ee_ref(t)|  (tracking error)')
    for fi in flips_eref:
        ax.axvline(t1[fi], ls=':', color='#1f77b4', alpha=0.4)
    for fi in flips_eact:
        ax.axvline(t1[fi], ls=':', color='#d62728', alpha=0.4)
    ax.set_ylabel('Swing-EE orientation [°]')
    ax.set_title(f'Step 1 — swing-EE orientation, midwaypoint run '
                 f'(q_ee_ref sign-flips: {len(flips_eref)}, '
                 f'q_ee_actual: {len(flips_eact)})')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(-5, 200)
    ax.axhline(180, color='gray', ls='--', linewidth=0.8)
    ax = axes[1]
    for k in range(4):
        ax.plot(t1, q_ee_ref[:, k], '-', linewidth=1.5,
                label=f'q_ref[{labels[k]}]')
        ax.plot(t1, q_ee_act[:, k], '--', linewidth=1.0,
                color=ax.lines[-1].get_color(), alpha=0.7,
                label=f'q_act[{labels[k]}]' if k == 0 else None)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('Swing-EE quaternion components')
    ax.set_title('Quaternion components — solid = ref, dashed = actual')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'q1_step1_swing_ori_traces.png'), dpi=130)
    plt.close(fig)

    return {
        'cmd_torso_max': float(np.max(cmd_torso)),
        'err_torso_max': float(np.max(err_torso)),
        'cmd_swing_max': float(np.max(cmd_swing)),
        'err_swing_max': float(np.max(err_swing)),
        'flips_tref': flips_tref, 'flips_tact': flips_tact,
        'flips_eref': flips_eref, 'flips_eact': flips_eact,
        't1_first_diverge': float(
            t1[np.argmax(err_torso > 30)]
            if (err_torso > 30).any() else float('nan')),
    }


def slerp_repro():
    """Q1.3 — Reproduce the piecewise SLERP path for step 1's torso
    rotation triple. The pin.log3 / pin.exp3 implementation always
    takes the geodesic shorter arc per sub-segment (no quaternion
    sign ambiguity, since log3 gives axis-angle in [0, π])."""
    d = json.load(open(SIM_LOG))
    ik = json.load(open(IK_TRACE))

    t = np.array(d['t'])
    phase = np.array(d['phase'])
    step_idx = np.array(d['step_idx'])
    mask = (step_idx == 1) & (phase == 'SS')
    q_torso_ref = np.array(d['q_torso_ref'])[mask]
    q_ee_ref = np.array(d['q_ee_ref'])[mask]

    # Endpoints of the piecewise SLERP for step 1.
    R_torso_start = _quat_wxyz_to_R(q_torso_ref[0])
    R_torso_end_log_idx = -2  # avoid the final dock-handoff sample
    R_torso_end = _quat_wxyz_to_R(q_torso_ref[R_torso_end_log_idx])

    # mid-waypoint torso orientation: not in sim_log directly,
    # but we can recover it by sampling q_torso_ref at the temporal
    # midpoint of the SS window (since add_phase used t_mid =
    # 0.5 * (t_start + t_end)).
    t1 = t[mask]
    t_mid_step1 = 0.5 * (t1[0] + t1[-1])
    idx_mid = int(np.argmin(np.abs(t1 - t_mid_step1)))
    R_torso_mid_inferred = _quat_wxyz_to_R(q_torso_ref[idx_mid])

    # Sign-of-inner-product cross-check (analogue of "quaternion
    # hemisphere check" the brief Q1.2 mentions).
    qS = q_torso_ref[0]
    qM = q_torso_ref[idx_mid]
    qE = q_torso_ref[R_torso_end_log_idx]
    ip_sm = float(np.dot(qS, qM))
    ip_me = float(np.dot(qM, qE))
    ip_se = float(np.dot(qS, qE))

    # Replay the piecewise-quintic SLERP using the same code path
    # as TorsoPlanner._interpolate_phase: pin.log3(R0^T R1) gives
    # axis-angle in [-π, π]; pin.exp3(s · v) integrates along that
    # geodesic. Always shortest-arc.
    n_samp = 100
    taus = np.linspace(0.0, 1.0, n_samp)
    angles_from_start_torso = np.zeros(n_samp)
    for i, tau in enumerate(taus):
        if tau <= 0.5:
            seg_tau = 2.0 * tau
            R0 = R_torso_start
            R1 = R_torso_mid_inferred
        else:
            seg_tau = 2.0 * (tau - 0.5)
            R0 = R_torso_mid_inferred
            R1 = R_torso_end
        s = (10.0 * seg_tau ** 3 - 15.0 * seg_tau ** 4
             + 6.0 * seg_tau ** 5)
        omega = pin.log3(R0.T @ R1)
        R = R0 @ pin.exp3(s * omega)
        angles_from_start_torso[i] = _angle_R_vs_R(R_torso_start, R)

    # Same for swing.
    R_swing_start = _quat_wxyz_to_R(q_ee_ref[0])
    R_swing_end = _quat_wxyz_to_R(q_ee_ref[R_torso_end_log_idx])
    R_swing_mid = _quat_wxyz_to_R(q_ee_ref[idx_mid])
    qSE_S = q_ee_ref[0]
    qSE_M = q_ee_ref[idx_mid]
    qSE_E = q_ee_ref[R_torso_end_log_idx]
    ip_se_sm = float(np.dot(qSE_S, qSE_M))
    ip_se_me = float(np.dot(qSE_M, qSE_E))
    ip_se_se = float(np.dot(qSE_S, qSE_E))
    angles_from_start_swing = np.zeros(n_samp)
    for i, tau in enumerate(taus):
        if tau <= 0.5:
            seg_tau = 2.0 * tau
            R0 = R_swing_start; R1 = R_swing_mid
        else:
            seg_tau = 2.0 * (tau - 0.5)
            R0 = R_swing_mid; R1 = R_swing_end
        s = (10.0 * seg_tau ** 3 - 15.0 * seg_tau ** 4
             + 6.0 * seg_tau ** 5)
        omega = pin.log3(R0.T @ R1)
        R = R0 @ pin.exp3(s * omega)
        angles_from_start_swing[i] = _angle_R_vs_R(R_swing_start, R)

    # Plot torso replay
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(taus, angles_from_start_torso, '-', linewidth=2,
            color='#1f77b4',
            label='Replayed piecewise SLERP (pin.log3/exp3)')
    ax.axvline(0.5, ls=':', color='gray', alpha=0.5,
               label='t_mid (segment boundary)')
    ax.axhline(180, ls='--', color='red', alpha=0.4,
               label='180° (geodesic upper bound)')
    ax.set_xlabel(r'$\tau$ across step 1 SS')
    ax.set_ylabel('|R_replay(τ) ⊖ R_torso_start| [°]')
    ax.set_title(
        f'Q1.3 piecewise-SLERP replay (torso) — '
        f'inner-products: ⟨S,M⟩={ip_sm:.3f}, ⟨M,E⟩={ip_me:.3f}, '
        f'⟨S,E⟩={ip_se:.3f}\n'
        f'Same-sign across all three? '
        f'{"YES" if np.sign(ip_sm) == np.sign(ip_me) == np.sign(ip_se) else "NO"}'
    )
    ax.set_ylim(0, 360)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'q1_slerp_repro_torso.png'), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(taus, angles_from_start_swing, '-', linewidth=2,
            color='#d62728',
            label='Replayed piecewise SLERP (pin.log3/exp3)')
    ax.axvline(0.5, ls=':', color='gray', alpha=0.5,
               label='t_mid (segment boundary)')
    ax.axhline(180, ls='--', color='red', alpha=0.4,
               label='180° (geodesic upper bound)')
    ax.set_xlabel(r'$\tau$ across step 1 SS')
    ax.set_ylabel('|R_swing_replay(τ) ⊖ R_swing_start| [°]')
    ax.set_title(
        f'Q1.3 piecewise-SLERP replay (swing) — '
        f'inner-products: ⟨S,M⟩={ip_se_sm:.3f}, ⟨M,E⟩={ip_se_me:.3f}, '
        f'⟨S,E⟩={ip_se_se:.3f}\n'
        f'Same-sign across all three? '
        f'{"YES" if np.sign(ip_se_sm) == np.sign(ip_se_me) == np.sign(ip_se_se) else "NO"}'
    )
    ax.set_ylim(0, 360)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'q1_slerp_repro_swing.png'), dpi=130)
    plt.close(fig)

    return {
        'replay_torso_peak': float(np.max(angles_from_start_torso)),
        'replay_swing_peak': float(np.max(angles_from_start_swing)),
        'ip_torso_sm': ip_sm, 'ip_torso_me': ip_me, 'ip_torso_se': ip_se,
        'ip_swing_sm': ip_se_sm, 'ip_swing_me': ip_se_me,
        'ip_swing_se': ip_se_se,
    }


def main():
    print("=== Q1.1 step-1 traces (sim_log refs vs actual) ===")
    s1 = step1_traces()
    for k, v in s1.items():
        print(f"  {k}: {v}")

    print("\n=== Q1.3 piecewise-SLERP replay ===")
    s2 = slerp_repro()
    for k, v in s2.items():
        print(f"  {k}: {v}")

    print(f"\nFigures: {OUTDIR}/q1_*.png")
    json.dump({**s1, **s2}, open(os.path.join(OUTDIR, 'q1_summary.json'),
                                  'w'), indent=2,
              default=lambda o: list(o) if hasattr(o, '__iter__') else str(o))


if __name__ == '__main__':
    main()
