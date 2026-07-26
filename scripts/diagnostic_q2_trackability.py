"""Q2 — Trackability empirical bound.

Runs a single-step T15-equivalent simulation with the IK output
`q_end` perturbed by α ∈ {0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0}.
Measures closed-loop tracking error metrics as a function of α to
characterize what joint-space displacement the controller can
actually follow.

Method: monkey-patch ``dock_configuration_fixed_rotation`` so its
returned ``q`` is replaced by the linear interpolation
``q_start + α · (q_end_natural − q_start)`` (arm joints only;
torso pose taken from the natural IK output). For α=1.0 the run
should be byte-identical to the unpatched baseline (sanity check).

Outputs:
  diagnostic/q2_trackability_results.json  (per-α metrics)
  diagnostic/q2_tracking_vs_distance.png   (4-panel plot)
"""
from __future__ import annotations

import os
import json
import sys
import time
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import scripts.run_m7_single_step as r_single  # noqa: E402
from crawlbot.core import ik as ik_module  # noqa: E402

OUTDIR = os.path.join(_root, 'diagnostic')
RUN_OUT_BASE = os.path.join(_root, 'Misc', 'runs', 'diagnostic_q2')

# We patch dock_configuration_fixed_rotation; preserve original.
_orig_fixed_rot_ik = ik_module.dock_configuration_fixed_rotation


def _make_patched_ik(alpha: float, q_start_holder: dict):
    """Return a wrapper around dock_configuration_fixed_rotation
    that intercepts the natural q_end and replaces it with the
    α-extrapolation of (q_start → q_end_natural) over the whole
    21-dim Pinocchio configuration:

      torso position (q[:3]):  q_start[:3] + α · (q_natural[:3] - q_start[:3])
      torso quaternion (q[3:7]): R_start · exp3(α · log3(R_start^T R_natural))
      arm joints   (q[7:]):    q_start[7:] + α · (q_natural[7:] - q_start[7:])

    For α=1 this returns the natural q_end exactly (sanity).
    For α∈(0,1) the perturbed q_end is "under-displaced". For α>1
    it extrapolates beyond the natural q_end on each axis.
    """
    def _patched(model, anchor_a, anchor_b, R_torso_fixed,
                 torso_pos=None, q_init=None, max_iter=2000, tol=1e-8):
        q_natural, err, w_yosh, w_sigma_min = _orig_fixed_rot_ik(
            model, anchor_a, anchor_b, R_torso_fixed,
            torso_pos=torso_pos, q_init=q_init,
            max_iter=max_iter, tol=tol,
        )
        if q_init is None:
            q_start_holder.setdefault('warning',
                'q_init was None; using q_natural unchanged')
            return q_natural, err, w_yosh, w_sigma_min
        q_start_local = np.asarray(q_init, dtype=float).copy()
        q_perturbed = q_natural.copy()
        # Torso position: linear extrap by α.
        q_perturbed[:3] = (
            q_start_local[:3]
            + alpha * (q_natural[:3] - q_start_local[:3])
        )
        # Torso quaternion: SLERP-like extrapolation via SO(3) log/exp.
        qS_xyzw = q_start_local[3:7]
        qN_xyzw = q_natural[3:7]
        R_S = pin.Quaternion(qS_xyzw[3], qS_xyzw[0],
                             qS_xyzw[1], qS_xyzw[2]).toRotationMatrix()
        R_N = pin.Quaternion(qN_xyzw[3], qN_xyzw[0],
                             qN_xyzw[1], qN_xyzw[2]).toRotationMatrix()
        omega = pin.log3(R_S.T @ R_N)
        R_P = R_S @ pin.exp3(alpha * omega)
        qP = pin.Quaternion(R_P)
        q_perturbed[3] = qP.x; q_perturbed[4] = qP.y
        q_perturbed[5] = qP.z; q_perturbed[6] = qP.w
        # Arm joints: linear extrap by α.
        q_perturbed[7:] = (
            q_start_local[7:]
            + alpha * (q_natural[7:] - q_start_local[7:])
        )
        # Diagnostics.
        q_start_holder['arm_distance_natural'] = float(
            np.linalg.norm(q_natural[7:] - q_start_local[7:]))
        q_start_holder['arm_distance_perturbed'] = float(
            np.linalg.norm(q_perturbed[7:] - q_start_local[7:]))
        q_start_holder['torso_pos_distance_natural_m'] = float(
            np.linalg.norm(q_natural[:3] - q_start_local[:3]))
        q_start_holder['torso_pos_distance_perturbed_m'] = float(
            np.linalg.norm(q_perturbed[:3] - q_start_local[:3]))
        q_start_holder['torso_rot_distance_natural_deg'] = float(
            np.degrees(float(np.linalg.norm(omega))))
        q_start_holder['torso_rot_distance_perturbed_deg'] = float(
            np.degrees(alpha * float(np.linalg.norm(omega))))
        return q_perturbed, err, w_yosh, w_sigma_min
    return _patched


def _run_one_alpha(alpha: float, out_dir: str):
    """Run a single-step sim with q_end's arm joints scaled by α."""
    holder: dict = {}
    ik_module.dock_configuration_fixed_rotation = _make_patched_ik(alpha, holder)
    # Also patch in sim_loop's namespace (it imported the symbol).
    import crawlbot.simulation.sim_loop as sim_loop_mod
    sim_loop_mod.dock_configuration_fixed_rotation = (
        ik_module.dock_configuration_fixed_rotation)
    try:
        cfg = r_single._make_m7_config()
        # T15-equivalent settings (same as the closed-loop runs we
        # care about). Keep IK fixed-rotation as the default path.
        cfg.preplanner_a_cruise_max = 0.01
        cfg.preplanner_cruise_ramp_frac = 0.2
        cfg.mapping_bypass_in_ss = True
        cfg.swing_early_finish_fraction = 0.80
        cfg.aocs_off_in_ds = True
        cfg.use_trajectory_aware_ik = False  # fixed_rotation only
        # MJCF mutation matches T15 (T11 transient).
        # We don't run the mutation here — keep MJCF in its
        # committed state. The single-step scenario behaves
        # consistently without it for this characterization.
        t0 = time.perf_counter()
        sim, log, metrics = r_single.run_case(
            f'Q2-trackability α={alpha:.2f}',
            out_dir, n_steps=1, config=cfg)
        elapsed = time.perf_counter() - t0
    finally:
        ik_module.dock_configuration_fixed_rotation = _orig_fixed_rot_ik
        sim_loop_mod.dock_configuration_fixed_rotation = _orig_fixed_rot_ik

    # Extract metrics.
    sim_log = json.load(open(os.path.join(out_dir, 'sim_log.json')))
    t_arr = np.array(sim_log['t'])
    e_torso_pos = np.array(sim_log['e_torso_pos'])
    e_ee_pos = np.array(sim_log['e_ee_pos'])
    phase = np.array(sim_log['phase'])
    step_idx = np.array(sim_log['step_idx'])
    ss_mask = (phase == 'SS') & (step_idx == 0)

    e_torso_pos_peak = float(np.max(np.abs(e_torso_pos[ss_mask]))) * 1000.0
    e_swing_pos_peak = float(np.max(np.abs(e_ee_pos[ss_mask]))) * 1000.0
    e_swing_pos_at_dock = float(np.abs(e_ee_pos[ss_mask][-1])) * 1000.0

    docked = bool(sim_log.get('dock_events'))
    aborted = bool(sim_log.get('aborted_steps'))
    if docked:
        ev = sim_log['dock_events'][0]
        outcome = f"DOCKED d={ev['d_mm']:.2f}mm ori={ev['ori_deg']:.2f}°"
    elif aborted:
        ab = sim_log['aborted_steps'][0]
        outcome = (f"ABORT reason={ab.get('reason','?')} "
                   f"d={ab.get('d_mm', float('nan')):.2f}mm "
                   f"ori={ab.get('ori_deg', float('nan')):.2f}°")
    else:
        outcome = "UNKNOWN"

    nmpc = Counter(sim_log.get('nmpc_status_str', []))
    n_total = sum(nmpc.values())
    n_infeas = nmpc.get('Infeasible_Problem_Detected', 0)
    n_maxiter = nmpc.get('Maximum_Iterations_Exceeded', 0)
    nmpc_fail_rate = (n_infeas + n_maxiter) / max(n_total, 1)

    return {
        'alpha': float(alpha),
        'arm_distance_natural_rad': holder.get('arm_distance_natural'),
        'arm_distance_perturbed_rad': holder.get('arm_distance_perturbed'),
        'torso_pos_distance_natural_m': holder.get('torso_pos_distance_natural_m'),
        'torso_pos_distance_perturbed_m': holder.get('torso_pos_distance_perturbed_m'),
        'torso_rot_distance_natural_deg': holder.get('torso_rot_distance_natural_deg'),
        'torso_rot_distance_perturbed_deg': holder.get('torso_rot_distance_perturbed_deg'),
        'e_torso_pos_peak_mm': e_torso_pos_peak,
        'e_swing_pos_peak_mm': e_swing_pos_peak,
        'e_swing_pos_at_dock_mm': e_swing_pos_at_dock,
        'docked': bool(docked),
        'aborted': bool(aborted),
        'outcome': outcome,
        'nmpc_fail_rate': float(nmpc_fail_rate),
        'n_nmpc_ticks': int(n_total),
        'elapsed_s': float(elapsed),
    }


def main():
    alphas = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    results = []
    n_consec_fail = 0
    for alpha in alphas:
        out_dir = os.path.join(RUN_OUT_BASE, f'alpha_{alpha:.2f}')
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n{'='*70}\n  α = {alpha:.2f}\n{'='*70}")
        try:
            r = _run_one_alpha(alpha, out_dir)
            results.append(r)
            print(f"  outcome: {r['outcome']}  "
                  f"e_swing_peak={r['e_swing_pos_peak_mm']:.1f}mm  "
                  f"d_arm={r['arm_distance_perturbed_rad']:.3f} rad  "
                  f"({r['elapsed_s']:.1f}s)")
            if r['aborted']:
                n_consec_fail += 1
            else:
                n_consec_fail = 0
            if n_consec_fail >= 3:
                print(f"\n  [stop] 3 successive failures — terminating sweep.")
                break
        except Exception as e:
            print(f"  Run FAILED: {type(e).__name__}: {e}")
            results.append({
                'alpha': float(alpha),
                'error': f"{type(e).__name__}: {e}",
                'aborted': True,
            })
            n_consec_fail += 1
            if n_consec_fail >= 3:
                break

    out_json = os.path.join(OUTDIR, 'q2_trackability_results.json')
    json.dump(results, open(out_json, 'w'), indent=2,
              default=lambda o: float(o) if hasattr(o, 'item') else str(o))
    print(f"\nWrote {out_json}")

    # 4-panel plot
    valid = [r for r in results if 'error' not in r]
    if not valid:
        print("No valid results — skipping plot.")
        return
    alphas_v = [r['alpha'] for r in valid]
    d_arm = [r['arm_distance_perturbed_rad'] for r in valid]
    docked = [r['docked'] for r in valid]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    metrics = [
        ('e_torso_pos_peak_mm', 'Peak torso pos error [mm]', '#1f77b4'),
        ('e_swing_pos_peak_mm', 'Peak swing-EE pos error [mm]', '#d62728'),
        ('e_swing_pos_at_dock_mm', 'Swing-EE pos error at dock [mm]', '#2ca02c'),
        ('nmpc_fail_rate', 'NMPC infeas+maxiter rate', '#9467bd'),
    ]
    for ax, (key, label, color) in zip(axes.flat, metrics):
        ys = [r[key] for r in valid]
        ax.plot(alphas_v, ys, 'o-', linewidth=2, color=color)
        for x, y, ok in zip(alphas_v, ys, docked):
            ax.plot(x, y, 'o' if ok else 'x', markersize=10,
                    color='green' if ok else 'red',
                    markeredgewidth=2)
        ax.axvline(1.0, ls=':', color='gray', alpha=0.5,
                   label='α=1 (natural q_end)')
        ax.set_xlabel(r'$\alpha$ (perturbation factor)')
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)
    fig.suptitle('Q2 — Closed-loop tracking vs joint-space displacement '
                 '(green o = docked, red x = aborted)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'q2_tracking_vs_distance.png'), dpi=130)
    plt.close(fig)
    print(f"Wrote {OUTDIR}/q2_tracking_vs_distance.png")


if __name__ == '__main__':
    main()
