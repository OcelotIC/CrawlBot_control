#!/usr/bin/env python3
"""Diagnostic for the huge torso tracking error.

Runs the M6 setup, intercepts the first QP step of the first SS
phase, and prints:

  (1) r_b_current (live torso position)
      r_b_ref     (mapping(r_com_plan, q_current) at that instant)
      ||r_b_ref - r_b_current||

  (2) cond(J_b)       — torso Jacobian conditioning
      cond(N_b · J_ee) — EE Jacobian projected into torso null space

  (3) joint torque saturation fraction over the full sim
      (read from the already-saved results/M6_baseline_1pct/sim_log.json)

  (4) r_b_ref(t) vs r_b_actual(t) for all 3 axes, saved as
      results/M6_platform_diag/torso_tracking_vs_ref.png
"""
import os
import sys

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import json
from io import StringIO
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pinocchio as pin

from crawlbot.simulation.sim_loop import SimulationLoop
from crawlbot.simulation.config import SimConfig
from crawlbot.solvers.contact_phase import ContactConfig
from crawlbot.core.state_conversions import mujoco_to_pinocchio

URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')
MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')
OUT_DIR = 'results/M6_platform_diag'


def _make_cfg():
    return SimConfig(
        use_m2_stack=True,
        alpha_com_soft=5.0, alpha_passivity=1.0,
        enforce_hw_conservation=True,
        h_max_tight=np.full(3, 5.0),
        w_L_nmpc=1.0, kappa_terminal=1.0,
        aocs_mode='legacy_corrected',
        aocs_use_legacy_corrected=True,
        use_coarse_preplanner=True,
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 80)
    print("  Torso tracking diagnostic")
    print("=" * 80)

    # ── (1) & (2): fresh setup, intercept first SS QP step ──
    old = sys.stdout
    sys.stdout = StringIO()
    try:
        sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=_make_cfg())
        sim.setup(n_steps=1, start_a=2, start_b=2)
        # Walk the plan to the first SS and run the pre-planner / torso setup
        p = sim.plan
        i = 0
        while p.phases[i].phase.value == 'double':
            i += 1
        ss = p.phases[i]
        t_ss_start = p.t_start[i]
        t_ss_end = p.t_end[i]
        sim._setup_torso_for_step(
            t_ss_start, t_ss_end, ss.swing_arm,
            ss.anchor_a_idx, ss.anchor_b_idx, ss.swing_arm, ss.swing_to_idx)
    finally:
        sys.stdout = old

    # Live state at SS entry
    pq, pv = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
    rs = sim.robot.update(pq, pv)

    r_b_current = rs.oMf_torso.translation.copy()
    R_b_current = rs.oMf_torso.rotation.copy()
    r_com_current = rs.r_com.copy()
    v_com_current = rs.v_com.copy()

    # What would the NMPC plan at this instant? Use cref from the
    # pre-planner at t_horizon (what sim_loop does).
    t_horizon = t_ss_start + sim.cfg.nmpc_N * sim.cfg.nmpc_dt
    if sim._coarse_plan is not None:
        tau_rel = t_horizon - sim._coarse_plan_t0
        rp_coarse = sim._coarse_plan.r_com_at(tau_rel)
        vp_coarse = sim._coarse_plan.v_com_at(tau_rel)
    else:
        rp_coarse = r_com_current
        vp_coarse = v_com_current

    # Mapping layer: r_com_ref -> r_b_ref
    r_b_ref, v_b_ref, a_b_ff, _ = sim.mapping.compute(
        r_com_ref=rp_coarse,
        v_com_ref=vp_coarse,
        a_com_ff=np.zeros(3),
        q_current=rs.q,
        dq_current=rs.v,
    )

    delta = r_b_ref - r_b_current
    print()
    print("(1) Initial torso reference vs. current (first QP step of SS phase)")
    print(f"    t_ss_start       = {t_ss_start:.3f} s")
    print(f"    r_com_current    = [{r_com_current[0]:+.4f} {r_com_current[1]:+.4f} {r_com_current[2]:+.4f}] m")
    print(f"    r_com_ref (plan) = [{rp_coarse[0]:+.4f} {rp_coarse[1]:+.4f} {rp_coarse[2]:+.4f}] m")
    print(f"    → ||r_com_ref - r_com_current|| = {np.linalg.norm(rp_coarse - r_com_current)*1000:.1f} mm")
    print()
    print(f"    r_b_current      = [{r_b_current[0]:+.4f} {r_b_current[1]:+.4f} {r_b_current[2]:+.4f}] m")
    print(f"    r_b_ref (mapped) = [{r_b_ref[0]:+.4f} {r_b_ref[1]:+.4f} {r_b_ref[2]:+.4f}] m")
    print(f"    Δ = r_b_ref - r_b_current = [{delta[0]:+.4f} {delta[1]:+.4f} {delta[2]:+.4f}] m")
    print(f"    ‖Δ‖ = {np.linalg.norm(delta)*1000:.1f} mm")
    if np.linalg.norm(delta) > 0.05:
        print(f"    *** INITIAL REFERENCE JUMP > 50 mm — mapping is asking")
        print(f"        for a torso pose that doesn't match the current state.")

    # ── (2) Jacobian conditioning ──
    # Need the full Jacobian of the torso frame and the swing-arm EE.
    model = sim.robot.model
    data = sim.robot.data
    pin.computeJointJacobians(model, data, rs.q)
    pin.updateFramePlacements(model, data)

    # Torso frame Jacobian (6 × nv) in LOCAL_WORLD_ALIGNED frame.
    # The torso frame id on RobotInterface is `frame_torso` (Link_0).
    torso_frame_id = sim.robot.frame_torso
    J_b6 = pin.getFrameJacobian(
        model, data, torso_frame_id, pin.LOCAL_WORLD_ALIGNED)
    J_b = J_b6[:3, :]

    # Swing EE frame Jacobian
    swing_arm = ss.swing_arm
    ee_frame_id = (sim.robot.frame_tool_b if swing_arm == 'b'
                   else sim.robot.frame_tool_a)
    J_ee6 = pin.getFrameJacobian(
        model, data, ee_frame_id, pin.LOCAL_WORLD_ALIGNED)
    J_ee = J_ee6[:3, :]

    # Null-space projector of J_b (pseudo-inverse)
    J_b_pinv = np.linalg.pinv(J_b, rcond=1e-8)
    N_b = np.eye(J_b.shape[1]) - J_b_pinv @ J_b
    J_ee_proj = J_ee @ N_b

    # Condition numbers
    sv_Jb = np.linalg.svd(J_b, compute_uv=False)
    sv_Jee_proj = np.linalg.svd(J_ee_proj, compute_uv=False)
    sv_Jee = np.linalg.svd(J_ee, compute_uv=False)

    def cond_strict(sv, eps=1e-10):
        nonzero = sv[sv > eps]
        if len(nonzero) == 0:
            return float('inf')
        return float(nonzero[0] / nonzero[-1])

    cond_Jb = cond_strict(sv_Jb)
    cond_Jee = cond_strict(sv_Jee)
    cond_Jee_proj = cond_strict(sv_Jee_proj)

    print()
    print("(2) Jacobian conditioning at SS entry")
    print(f"    J_b (3×{J_b.shape[1]}): σ = {sv_Jb}")
    print(f"       cond(J_b)             = {cond_Jb:.2e}")
    print(f"    J_ee (3×{J_ee.shape[1]}): σ = {sv_Jee}")
    print(f"       cond(J_ee)            = {cond_Jee:.2e}")
    print(f"    J_ee · N_b:  σ = {sv_Jee_proj}")
    print(f"       cond(N_b · J_ee)      = {cond_Jee_proj:.2e}")
    if cond_Jb > 1e4:
        print(f"    *** J_b IS ILL-CONDITIONED (>{1e4:.0e})")
    if cond_Jee_proj > 1e4:
        print(f"    *** N_b·J_ee IS ILL-CONDITIONED (>{1e4:.0e}) — EE"
              " task is starved in the torso null space.")

    # ── (3) Joint torque saturation fraction ──
    print()
    print("(3) Joint torque saturation fraction (full sim)")
    log_path = 'results/M6_baseline_1pct/sim_log.json'
    with open(log_path) as f:
        log = json.load(f)
    tau = np.asarray(log['tau'])
    T, nj = tau.shape
    any_sat = np.any(np.abs(tau) > 19.9, axis=1)
    print(f"    Total QP steps logged: {T}")
    print(f"    Steps with ≥1 joint at ±τ_max (20 Nm): "
          f"{int(any_sat.sum())}/{T} ({100*any_sat.mean():.1f}%)")
    phase = np.asarray(log['phase'])
    for ph in np.unique(phase):
        mask = phase == ph
        sat = np.any(np.abs(tau[mask]) > 19.9, axis=1)
        print(f"       phase={ph!r:12s} n={int(mask.sum()):3d}  "
              f"sat={int(sat.sum()):3d} ({100*sat.mean():.1f}%)")
    print(f"    → {'>50% → actuator-limited' if any_sat.mean() > 0.5 else '<50% → not obviously actuator-limited'}")

    # ── (4) r_b_ref(t) vs r_b_actual(t) ──
    print()
    print("(4) Plotting r_b_ref vs r_b_actual over full sim...")
    t_log = np.asarray(log['t'])
    p_tor = np.asarray(log['p_torso'])       # (T, 3) actual
    p_ref = np.asarray(log['p_torso_ref'])   # (T, 3) torso-planner reference
    r_com = np.asarray(log['r_com'])
    r_com_ref = np.asarray(log['r_com_ref'])
    phase = np.asarray(log['phase'])

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axis_labels = ['x', 'y', 'z']
    # Left column: p_torso vs p_torso_ref (torso planner's reference)
    for i, ax in enumerate(axes[:, 0]):
        ax.plot(t_log, p_ref[:, i] * 1000, 'b-', label='torso_planner p_ref', lw=1.5)
        ax.plot(t_log, p_tor[:, i] * 1000, 'r-', label='actual p_torso', lw=1.5)
        ax.set_ylabel(f'{axis_labels[i]} [mm]')
        ax.grid(alpha=0.3)
        # Shade phase regions
        phases = [(t_log[k], phase[k]) for k in range(len(t_log))]
        # quick coloring
        for k in range(1, len(t_log)):
            if phase[k] == 'SS':
                ax.axvspan(t_log[k - 1], t_log[k], alpha=0.08, color='green')
            elif phase[k] == 'EXT':
                ax.axvspan(t_log[k - 1], t_log[k], alpha=0.08, color='orange')
    axes[0, 0].legend(loc='best', fontsize=8)
    axes[0, 0].set_title('p_torso vs. TorsoPlanner p_ref')
    axes[-1, 0].set_xlabel('time [s]')

    # Right column: r_com vs r_com_ref (what the NMPC+QP actually track)
    for i, ax in enumerate(axes[:, 1]):
        ax.plot(t_log, r_com_ref[:, i] * 1000, 'b-', label='r_com_ref (NMPC)', lw=1.5)
        ax.plot(t_log, r_com[:, i] * 1000, 'r-', label='actual r_com', lw=1.5)
        ax.set_ylabel(f'{axis_labels[i]} [mm]')
        ax.grid(alpha=0.3)
        for k in range(1, len(t_log)):
            if phase[k] == 'SS':
                ax.axvspan(t_log[k - 1], t_log[k], alpha=0.08, color='green')
            elif phase[k] == 'EXT':
                ax.axvspan(t_log[k - 1], t_log[k], alpha=0.08, color='orange')
    axes[0, 1].legend(loc='best', fontsize=8)
    axes[0, 1].set_title('r_com vs. NMPC r_com_ref')
    axes[-1, 1].set_xlabel('time [s]')
    fig.suptitle('Torso / CoM tracking — M6 1% single step (SS green, EXT orange)')
    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'torso_tracking_vs_ref.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"    saved: {out}")


if __name__ == "__main__":
    main()
