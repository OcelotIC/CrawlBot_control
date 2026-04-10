#!/usr/bin/env python3
"""Diagnostic: where does the 48° platform rotation at 1% come from?

Runs three configurations from the same post-settling initial state
and reports the structure attitude over time:

  A. LOCK  — Joints frozen at initial config (tau_q = 0, qvel[arms]=0
             after every step), AOCS disabled (tau_w = 0).
             Tests MuJoCo contact/weld baseline drift with no robot
             motion at all. Expected: ~0 rotation.

  B. NOAOCS — Full NMPC+QP controller drives the arm joints, AOCS
              disabled (tau_w = 0).
              Measures the raw uncompensated disturbance generated
              by robot motion through the arms.

  C. AOCS   — Full pipeline, baseline (same as M6 baseline).

Output:
  results/M6_platform_diag/attitude_vs_time.csv
  stdout: peak |angle| per case + interpretation.

Usage:
  MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_platform_rotation.py
"""
import os
import sys

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import csv
from io import StringIO
import numpy as np
import pinocchio as pin

from crawlbot.simulation.sim_loop import SimulationLoop
from crawlbot.simulation.config import SimConfig

URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')
MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')
OUT_DIR = 'results/M6_platform_diag'


def _make_cfg():
    # Baseline M6 config (same as scripts/run_m6_baseline.py).
    return SimConfig(
        use_m2_stack=True,
        alpha_com_soft=5.0,
        alpha_passivity=1.0,
        enforce_hw_conservation=True,
        h_max_tight=np.full(3, 5.0),
        w_L_nmpc=1.0,
        kappa_terminal=1.0,
        aocs_mode='legacy_corrected',
        aocs_use_legacy_corrected=True,
        aocs_use_H_estimator=False,
        use_coarse_preplanner=True,
    )


def _euler_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    """Return ZYX intrinsic Euler angles (deg) from a wxyz quaternion."""
    w, x, y, z = quat_wxyz
    R = pin.Quaternion(w, x, y, z).toRotationMatrix()
    # roll, pitch, yaw (deg)
    sy = float(np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]))
    if sy > 1e-9:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        pitch = float(np.arctan2(-R[2, 0], sy))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = float(np.arctan2(-R[1, 2], R[1, 1]))
        pitch = float(np.arctan2(-R[2, 0], sy))
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def run_case(tag: str, disable_aocs: bool, lock_joints: bool):
    print(f"\n{'=' * 78}")
    print(f"  CASE {tag}  "
          f"(lock_joints={lock_joints}, disable_aocs={disable_aocs})")
    print(f"{'=' * 78}")

    cfg = _make_cfg()
    # Silence setup chatter
    old = sys.stdout
    sys.stdout = StringIO()
    try:
        sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=cfg)
        sim.setup(n_steps=1, start_a=2, start_b=2)
    finally:
        sys.stdout = old

    # Snapshot initial post-settling euler so reported angles are deltas
    q0 = sim.mj_data.qpos[3:7].copy()
    eul0 = _euler_deg(q0)

    # Install diagnostic hooks
    sim._diag_disable_aocs = bool(disable_aocs)
    sim._diag_lock_arm_joints = bool(lock_joints)

    # Run
    old = sys.stdout
    sys.stdout = StringIO()
    try:
        log = sim.run(verbose=False)
    finally:
        sys.stdout = old

    # Extract structure euler over time directly from the log
    times = list(log.t)
    eul_series = np.asarray(log.struct_euler_deg, dtype=float) - eul0
    if eul_series.size == 0:
        return {
            'tag': tag, 't': [], 'roll': [], 'pitch': [], 'yaw': [],
            'peak_abs_deg': 0.0,
        }

    peak = float(np.max(np.abs(eul_series)))
    roll = eul_series[:, 0]
    pitch = eul_series[:, 1]
    yaw = eul_series[:, 2]

    print(f"  Duration       : {times[-1] - times[0]:.1f} s ({len(times)} steps)")
    print(f"  Δ roll  range  : [{roll.min():7.3f}, {roll.max():7.3f}] deg  peak |roll|={np.max(np.abs(roll)):6.3f}")
    print(f"  Δ pitch range  : [{pitch.min():7.3f}, {pitch.max():7.3f}] deg  peak |pitch|={np.max(np.abs(pitch)):6.3f}")
    print(f"  Δ yaw   range  : [{yaw.min():7.3f}, {yaw.max():7.3f}] deg  peak |yaw|={np.max(np.abs(yaw)):6.3f}")
    print(f"  Peak |any axis|: {peak:.3f} deg")
    # Also report omega_s peak
    if hasattr(log, 'omega_s') and len(log.omega_s):
        om = np.asarray(log.omega_s, dtype=float)
        if om.ndim == 2:
            peak_om = float(np.max(np.linalg.norm(om, axis=1)))
        else:
            peak_om = 0.0
        print(f"  Peak |omega_s| : {peak_om:.5f} rad/s")
    return {
        'tag': tag,
        't': times,
        'roll': roll.tolist(),
        'pitch': pitch.tolist(),
        'yaw': yaw.tolist(),
        'peak_abs_deg': peak,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 78)
    print("  Platform rotation diagnostic  (1% mass ratio, single step)")
    print("=" * 78)
    print("  Objective: localize the 48° platform rotation at 1%.")
    print("  Expected at 1%: < 2° even in the worst case.")

    results = []
    results.append(run_case("A_LOCK",   disable_aocs=True,  lock_joints=True))
    results.append(run_case("B_NOAOCS", disable_aocs=True,  lock_joints=False))
    results.append(run_case("C_AOCS",   disable_aocs=False, lock_joints=False))

    # Summary
    print()
    print("=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(f"{'Case':10} {'Peak |roll| deg':>16} {'Peak |pitch| deg':>18} {'Peak |yaw| deg':>16} {'Peak |any| deg':>16}")
    print("-" * 78)
    for r in results:
        if not r['t']:
            print(f"{r['tag']:10} (no data)")
            continue
        roll = np.asarray(r['roll'])
        pitch = np.asarray(r['pitch'])
        yaw = np.asarray(r['yaw'])
        print(f"{r['tag']:10} "
              f"{np.max(np.abs(roll)):16.3f} "
              f"{np.max(np.abs(pitch)):18.3f} "
              f"{np.max(np.abs(yaw)):16.3f} "
              f"{r['peak_abs_deg']:16.3f}")

    # CSV
    csv_path = os.path.join(OUT_DIR, 'attitude_vs_time.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['case', 't', 'roll_deg', 'pitch_deg', 'yaw_deg'])
        for r in results:
            for i in range(len(r['t'])):
                w.writerow([r['tag'], r['t'][i], r['roll'][i], r['pitch'][i], r['yaw'][i]])
    print(f"\nSaved: {csv_path}")

    # Interpretation
    print()
    print("=" * 78)
    print("  INTERPRETATION")
    print("=" * 78)
    peak_A = results[0]['peak_abs_deg']
    peak_B = results[1]['peak_abs_deg']
    peak_C = results[2]['peak_abs_deg']
    if peak_A > 1.0:
        print("  * Case A (locked joints, no AOCS) rotates more than 1° —")
        print("    the bug is in the MuJoCo contact model or the weld")
        print("    constraint, NOT the controller or AOCS.")
    else:
        print(f"  * Case A (locked joints, no AOCS) peak = {peak_A:.3f}° — "
              f"contact model baseline is clean.")
    if peak_B > 1.0:
        print(f"  * Case B (joints active, no AOCS) peak = {peak_B:.2f}° — "
              f"this is the raw uncompensated disturbance.")
    if peak_C > 1.0:
        print(f"  * Case C (AOCS on) peak = {peak_C:.2f}° — "
              f"AOCS attenuation = {100 * (1 - peak_C/max(peak_B, 1e-9)):.0f}% "
              f"relative to Case B.")
    # Compare
    if peak_C > peak_B:
        print("  !! AOCS is AMPLIFYING rotation (C > B). Signs are wrong "
              "or feedforward is broken.")
    elif peak_C > 0.5 * peak_B:
        print("  ~ AOCS reduces rotation by less than 50%. Too weak or "
              "mis-tuned.")
    else:
        print("  + AOCS effective: reduces rotation by >50%.")


if __name__ == "__main__":
    main()
