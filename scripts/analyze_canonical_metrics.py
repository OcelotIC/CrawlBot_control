#!/usr/bin/env python3
"""CANONICAL re-validation metrics (J2 STEP 2/3). Reads ONLY committed artifacts:
  results/j2_figdata/runA_traversal.csv  +  runA_meta.json   (the canonical export)
  <raw run dir>/sim_log.json                                  (q_torso/struct_quat for drift)

Computes the gate + closure numbers the wrong-config validation reported, now on
canonical config. Every value is read back from the artifact (no memory).
  C1 docking (meta dock_events <=5mm)
  C2 torso-pos SS tracking peak/rms (mm)
  C3 ||Hdot_s||_inf on SS (planned) <=5  + full-cycle max + #ticks>5
  C4 attitude theta_s SS/DS/full-run peak (deg) <5
  C5 ||hw||_inf full-run (Nms) <=4.5
  terminal ||hw|| / ||theta_s|| (last tick)
  CoM ||e_com|| SS median/max (mm)
  tau_w: within-settle max + at SS hand-off
  torso cumulative drift vs t=0 (structure- & world-frame, peak+final)  [from sim_log]

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/analyze_canonical_metrics.py \
        results/j2_figdata/runA  results/figA_canon
"""
import csv
import json
import os
import sys
import numpy as np
import pinocchio as pin


def q2R(q):  # sim_log quaternion order is wxyz (MuJoCo) — see dump_closure_curves.py:39
    w, x, y, z = q
    return pin.Quaternion(w, x, y, z).toRotationMatrix()


def geo_deg(R):  # geodesic angle of a rotation matrix
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def fnum(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan


def main(out_prefix, raw_dir):
    csv_path = out_prefix + '_traversal.csv'
    meta_path = out_prefix + '_meta.json'
    rows = list(csv.DictReader(open(csv_path)))
    meta = json.load(open(meta_path))
    n = len(rows)
    phase = np.array([r['phase'] for r in rows])
    is_ss = phase == 'SS'
    is_ds = ~is_ss

    def col(name):
        return np.array([fnum(r[name]) for r in rows])

    Hx, Hy, Hz = col('Hdot_s_x_Nm'), col('Hdot_s_y_Nm'), col('Hdot_s_z_Nm')
    Hinf = np.nanmax(np.abs(np.stack([Hx, Hy, Hz], 1)), axis=1)
    src = np.array([r['Hdot_s_source'] for r in rows])
    hw = np.stack([col('hw_x_Nms'), col('hw_y_Nms'), col('hw_z_Nms')], 1)
    th = np.stack([col('theta_s_x_deg'), col('theta_s_y_deg'), col('theta_s_z_deg')], 1)
    rcom = np.stack([col('rcom_x_m'), col('rcom_y_m'), col('rcom_z_m')], 1)
    rref = np.stack([col('rref_x_m'), col('rref_y_m'), col('rref_z_m')], 1)
    tp = np.stack([col('torso_pos_x_m'), col('torso_pos_y_m'), col('torso_pos_z_m')], 1)
    tpr = np.stack([col('torso_pos_ref_x_m'), col('torso_pos_ref_y_m'), col('torso_pos_ref_z_m')], 1)
    tw = np.stack([col('tauw_x_Nm'), col('tauw_y_Nm'), col('tauw_z_Nm')], 1)
    swing_active = np.array([r['swing_active'] == '1' for r in rows])

    print('=' * 74)
    print('CANONICAL re-validation metrics')
    print(f'  csv : {csv_path}  ({n} ticks)')
    print(f'  meta: {meta_path}')
    print(f'  label: {meta.get("label","")}')
    print(f'  run_config: {json.dumps(meta.get("run_config",{}))}')
    print('=' * 74)

    # ---- C1 docking ----
    de = meta.get('dock_events', [])
    docks = [round(float(e['dock_distance_mm']), 2) for e in de]
    arms = [f"{e.get('arm','?')}{e.get('anchor','')}" for e in de]
    c1 = (len(docks) == 5) and all(d <= 5.0 + 1e-9 for d in docks)
    print(f'C1 docking : {docks} mm  arms={arms}  -> {"PASS" if c1 else "FAIL"} '
          f'(5/5 <=5mm; worst margin {5.0-max(docks):.2f}mm)')

    # ---- C2 torso-pos SS tracking ----
    e_tp_ss = np.linalg.norm((tp - tpr)[is_ss], axis=1) * 1000.0
    print(f'C2 torso-pos SS: peak {np.nanmax(e_tp_ss):.1f} mm  rms {np.sqrt(np.nanmean(e_tp_ss**2)):.1f} mm '
          f'(median {np.nanmedian(e_tp_ss):.1f})')

    # ---- C3 envelope ‖Ḣ_s‖∞ ----
    ss_planned = is_ss & (src == 'planned') & np.isfinite(Hinf)
    hinf_ss = float(np.nanmax(Hinf[ss_planned])) if ss_planned.any() else float('nan')
    finite = np.isfinite(Hinf)
    hinf_full = float(np.nanmax(Hinf[finite]))
    nover = int(np.sum(Hinf[finite] > 5.0 + 1e-6))
    nover_ss = int(np.sum(Hinf[ss_planned] > 5.0 + 1e-6))
    print(f'C3 envelope: ||Hdot_s||inf_SS(planned) = {hinf_ss:.3f}  -> {"PASS" if hinf_ss<=5.0+1e-2 else "FAIL"} (<=5) '
          f'[SS ticks>5: {nover_ss}]')
    print(f'   full-cycle (SS planned + DS realized): max {hinf_full:.3f}  ticks>5: {nover}/{int(finite.sum())}')

    # ---- C4 attitude ----
    thinf = np.nanmax(np.abs(th), axis=1)
    print(f'C4 attitude: peak|theta_s|inf full-run {np.nanmax(thinf):.2f} deg  '
          f'SS {np.nanmax(thinf[is_ss]):.2f}  DS {np.nanmax(thinf[is_ds]):.2f}  '
          f'-> {"PASS" if np.nanmax(thinf)<5.0 else "FAIL"} (<5); first tick {thinf[0]:.2f} (init offset)')

    # ---- C5 h_w ----
    hwinf = np.nanmax(np.abs(hw), axis=1)
    print(f'C5 h_w∞    : full-run {np.nanmax(hwinf):.3f} Nms  SS {np.nanmax(hwinf[is_ss]):.3f}  '
          f'-> {"PASS" if np.nanmax(hwinf)<=4.5 else "FAIL"} (<=4.5)')

    # ---- terminal state (last tick) ----
    print(f'terminal   : ||hw|| {np.linalg.norm(hw[-1]):.3f} Nms  ||theta_s|| {np.linalg.norm(th[-1]):.3f} deg '
          f'(last tick t={rows[-1]["t_s"]}s, phase {phase[-1]})')

    # ---- CoM tracking SS ----
    e_com_ss = np.linalg.norm((rcom - rref)[is_ss], axis=1) * 1000.0
    print(f'CoM e_com SS: median {np.nanmedian(e_com_ss):.1f} mm  max {np.nanmax(e_com_ss):.1f} mm')

    # ---- tau_w within settle vs hand-off ----
    twinf = np.nanmax(np.abs(tw), axis=1)
    print(f'tau_w      : full-run max {np.nanmax(twinf):.3f} Nm  DS max {np.nanmax(twinf[is_ds]):.3f}  '
          f'SS max {np.nanmax(twinf[is_ss]):.3f} (limit 5)')

    # ---- torso cumulative drift vs t=0 (from raw sim_log) ----
    sl = json.load(open(os.path.join(raw_dir, 'sim_log.json')))
    qt = np.asarray(sl['q_torso'], float)
    sq = np.asarray(sl['struct_quat'], float)
    m = len(qt)
    Rt0 = q2R(qt[0])
    Rs0, Rtt0 = q2R(sq[0]), q2R(qt[0])
    Rrel0 = Rs0.T @ Rtt0
    world = np.array([geo_deg(Rt0.T @ q2R(qt[k])) for k in range(m)])
    struct = np.array([geo_deg(Rrel0.T @ (q2R(sq[k]).T @ q2R(qt[k]))) for k in range(m)])
    print(f'torso drift vs t=0 (raw {raw_dir}, {m} ticks):')
    print(f'   structure-frame: peak {struct.max():.3f} deg  final {struct[-1]:.3f} deg')
    print(f'   world-frame    : peak {world.max():.3f} deg  final {world[-1]:.3f} deg')
    print(f'   context theta_s : peak {np.nanmax(thinf):.3f}  final {np.linalg.norm(th[-1]):.3f}')
    print('=' * 74)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
