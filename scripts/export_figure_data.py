#!/usr/bin/env python3
"""Figure-data export (READ-ONLY w.r.t. crawlbot/) — tidy per-tick CSV + meta
JSON from a committed sim_log.json, for building the paper figures off-repo.

Reads results/<run_dir>/sim_log.json (produced by diag_cooperative_arms with
the committed flags) and writes <out>_traversal.csv (one row per tick, SI units
in the header) + <out>_meta.json. NO plotting. NO crawlbot/ change.

Quantities:
  - Hdot_s (envelope, EXACT): Ḣ_s = L̇_com + r_com×m·v̇_com, the origin-referenced
    momentum rate the exact box enforces — computed by centered FD (np.gradient)
    of the logged L_com, v_com, r_com with m = robot subtree mass (71.056 kg).
    Hdot_s_proxy = L̇_com (orbital term omitted) — the robot-CoM proxy.
  - Ltot (conservation): ‖subtree_angmom[0]‖ per axis, recomputed at the 12
    snapshots (initial / release_stepN / dock_stepN / final) by replaying the
    stored qpos/qvel through mj_subtreeVel — the SAME quantity & code path as
    the Fix-A conservation check (audit_fixC_residual.py). It is NOT a per-tick
    logged field (per-tick recompute needs the full 31-DOF state, not stored in
    the per-tick log), so the CSV carries it at the snapshot-nearest ticks and
    meta.conservation_snapshots carries the full labelled series.
  - swing_dist = d_grip_swing = ‖gripper_site − target_anchor‖ via
    _gripper_distance (the same site pair Fix C's dock gate uses); meaningful
    while swing_active (SS phase, an arm swinging).

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/export_figure_data.py \
        --run-dir results/<dir> --out results/j2_figdata/<name> \
        --config '{"qp_envelope_exact": true, ...}'
"""
import argparse
import csv
import json
import os
import numpy as np
import mujoco

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MJCF = os.path.join(ROOT, 'models', 'VISPA_crawling_rwa3.xml')


def robot_subtree_mass(model):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
    return float(model.body_subtreemass[bid])


def ltot_at_snapshots(sl, model):
    """‖subtree_angmom[0]‖ per axis at each (t, qpos, qvel, label) snapshot —
    the exact Fix-A conservation quantity (same mj calls as audit_fixC_residual)."""
    d = mujoco.MjData(model)
    out = []
    for (t, q, v, lbl) in sl.get('snapshots', []):
        d.qpos[:] = np.asarray(q, float)
        d.qvel[:] = np.asarray(v, float)
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        mujoco.mj_comVel(model, d)
        mujoco.mj_subtreeVel(model, d)
        L = d.subtree_angmom[0].copy()
        out.append({'label': str(lbl), 't': float(t),
                    'Ltot_x': float(L[0]), 'Ltot_y': float(L[1]),
                    'Ltot_z': float(L[2]), 'Ltot_norm': float(np.linalg.norm(L))})
    return out


def phase_per_tick(sl):
    """SS / DS_terminal / DS_interstep per tick (DS split via inter_step_settles)."""
    t = np.asarray(sl['t'], float)
    ph = np.asarray(sl['phase'])
    is_mask = np.zeros(len(t), bool)
    for s in (sl.get('inter_step_settles', []) or []):
        is_mask |= (ph == 'DS') & (t > float(s['t_start']) + 1e-6) & (t <= float(s['t_end']) + 1e-6)
    out = []
    for i in range(len(t)):
        if ph[i] == 'SS':
            out.append('SS')
        else:
            out.append('DS_interstep' if is_mask[i] else 'DS_terminal')
    return out


def nearest_tick(t_arr, t):
    return int(np.argmin(np.abs(t_arr - t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--out', required=True, help='output path prefix (no extension)')
    ap.add_argument('--label', default='')
    ap.add_argument('--config', default='{}', help='run flags as JSON (provenance)')
    args = ap.parse_args()

    sl = json.load(open(os.path.join(args.run_dir, 'sim_log.json')))
    model = mujoco.MjModel.from_xml_path(MJCF)
    m_robot = robot_subtree_mass(model)

    t = np.asarray(sl['t'], float)
    n = len(t)
    # The log is MIXED-cadence: SS/_step rows at dt_nmpc (~0.1 s), inter-step DS
    # rows at dt_qp (~0.01 s). The FD below must use the actual t coordinate
    # (np.gradient(·, t)), NOT a scalar dt, or the dense DS rows get a 10x-wrong
    # rate. Report both regimes in meta.
    diffs = np.diff(t) if n > 1 else np.array([0.01])
    dt_median = float(np.median(diffs))
    dt_min = float(diffs.min())

    L_com = np.asarray(sl['L_com'], float)
    v_com = np.asarray(sl['v_com'], float)
    r_com = np.asarray(sl['r_com'], float)
    r_ref = np.asarray(sl['r_com_ref'], float)
    hwp = np.asarray(sl['hw_physical'], float)
    theta = np.asarray(sl['struct_euler_deg'], float)
    tauw = np.asarray(sl['tau_w'], float)
    p_t = np.asarray(sl['p_torso'], float)
    p_tr = np.asarray(sl['p_torso_ref'], float)
    e_to = np.asarray(sl['e_torso_ori'], float)
    dsw = np.asarray(sl['d_grip_swing'], float)
    sarm = np.asarray(sl['swing_arm'])
    sidx = np.asarray(sl['step_idx'], int)
    raw_ph = np.asarray(sl['phase'])
    lam_ref = np.asarray(sl['lambda_ref'], float)   # (n, 12) planned wrench

    # Envelope (figure 1) — the EXACT origin-referenced Ḣ_s the controller plans
    # within: Ḣ_s = Σ_j (r_Cj × f_j + τ_j) from the planned wrench (lambda_ref)
    # and the structure-frame anchors. This is the SAME quantity as the C3 gate
    # metric ‖Ḣ_s‖∞_SS (which reads 5.0 at the binding) — read directly from the
    # postproc CSV so figure 1 matches C3 exactly. It is the NMPC envelope path-
    # constraint quantity (origin-referenced, exact levers), NOT the FD of the
    # realized momentum (which has dock/phase-transition spikes and is not what
    # the envelope enforces). Defined on SS/_step rows; NaN on inter-step DS
    # (NMPC bypassed there ⇒ no planned wrench).
    pp_path = os.path.join(args.run_dir, 'postproc_F3F4.csv')
    Hdot_exact = np.full((n, 3), np.nan)
    with open(pp_path) as f:
        rdr = csv.DictReader(f)
        for i, row in enumerate(rdr):
            if i >= n:
                break
            for j, a in enumerate('xyz'):
                v = row.get(f'Hdot_s_{a}', '')
                try:
                    Hdot_exact[i, j] = float(v)
                except (ValueError, TypeError):
                    Hdot_exact[i, j] = np.nan
    # Proxy (orbital term omitted): lever from the robot CoM instead of O_s ⇒
    # proxy = exact − r_com × Σf (Σf = f1+f2 from the same planned wrench).
    f_sum = lam_ref[:, 0:3] + lam_ref[:, 6:9]
    Hdot_proxy = Hdot_exact - np.cross(r_com, f_sum)

    phase = phase_per_tick(sl)
    swing_active = [(raw_ph[i] == 'SS') and (sarm[i] in ('a', 'b')) for i in range(n)]

    # Ltot at snapshots -> nearest tick (sparse in the CSV; full series in meta).
    snaps = ltot_at_snapshots(sl, model)
    ltot_col = np.full((n, 3), np.nan)
    for s in snaps:
        k = nearest_tick(t, s['t'])
        ltot_col[k] = [s['Ltot_x'], s['Ltot_y'], s['Ltot_z']]

    # ---- write CSV ----
    cols = [
        ('t_s', lambda i: f'{t[i]:.6f}'),
        ('tick', lambda i: str(i)),
        ('phase', lambda i: phase[i]),
        ('step_index', lambda i: str(int(sidx[i]))),
        ('Hdot_s_x_Nm', lambda i: '' if np.isnan(Hdot_exact[i,0]) else f'{Hdot_exact[i,0]:.6e}'),
        ('Hdot_s_y_Nm', lambda i: '' if np.isnan(Hdot_exact[i,1]) else f'{Hdot_exact[i,1]:.6e}'),
        ('Hdot_s_z_Nm', lambda i: '' if np.isnan(Hdot_exact[i,2]) else f'{Hdot_exact[i,2]:.6e}'),
        ('Hdot_s_proxy_x_Nm', lambda i: '' if np.isnan(Hdot_proxy[i,0]) else f'{Hdot_proxy[i,0]:.6e}'),
        ('Hdot_s_proxy_y_Nm', lambda i: '' if np.isnan(Hdot_proxy[i,1]) else f'{Hdot_proxy[i,1]:.6e}'),
        ('Hdot_s_proxy_z_Nm', lambda i: '' if np.isnan(Hdot_proxy[i,2]) else f'{Hdot_proxy[i,2]:.6e}'),
        ('hw_x_Nms', lambda i: f'{hwp[i,0]:.6e}'),
        ('hw_y_Nms', lambda i: f'{hwp[i,1]:.6e}'),
        ('hw_z_Nms', lambda i: f'{hwp[i,2]:.6e}'),
        ('theta_s_x_deg', lambda i: f'{theta[i,0]:.6f}'),
        ('theta_s_y_deg', lambda i: f'{theta[i,1]:.6f}'),
        ('theta_s_z_deg', lambda i: f'{theta[i,2]:.6f}'),
        ('Ltot_x_Nms', lambda i: '' if np.isnan(ltot_col[i,0]) else f'{ltot_col[i,0]:.6e}'),
        ('Ltot_y_Nms', lambda i: '' if np.isnan(ltot_col[i,1]) else f'{ltot_col[i,1]:.6e}'),
        ('Ltot_z_Nms', lambda i: '' if np.isnan(ltot_col[i,2]) else f'{ltot_col[i,2]:.6e}'),
        ('rcom_x_m', lambda i: f'{r_com[i,0]:.6f}'),
        ('rcom_y_m', lambda i: f'{r_com[i,1]:.6f}'),
        ('rcom_z_m', lambda i: f'{r_com[i,2]:.6f}'),
        ('rref_x_m', lambda i: f'{r_ref[i,0]:.6f}'),
        ('rref_y_m', lambda i: f'{r_ref[i,1]:.6f}'),
        ('rref_z_m', lambda i: f'{r_ref[i,2]:.6f}'),
        ('tauw_x_Nm', lambda i: f'{tauw[i,0]:.6e}'),
        ('tauw_y_Nm', lambda i: f'{tauw[i,1]:.6e}'),
        ('tauw_z_Nm', lambda i: f'{tauw[i,2]:.6e}'),
        ('torso_pos_x_m', lambda i: f'{p_t[i,0]:.6f}'),
        ('torso_pos_y_m', lambda i: f'{p_t[i,1]:.6f}'),
        ('torso_pos_z_m', lambda i: f'{p_t[i,2]:.6f}'),
        ('torso_pos_ref_x_m', lambda i: f'{p_tr[i,0]:.6f}'),
        ('torso_pos_ref_y_m', lambda i: f'{p_tr[i,1]:.6f}'),
        ('torso_pos_ref_z_m', lambda i: f'{p_tr[i,2]:.6f}'),
        ('torso_ori_err_deg', lambda i: f'{e_to[i]:.6f}'),
        ('swing_dist_m', lambda i: f'{dsw[i]:.6f}'),
        ('swing_active', lambda i: '1' if swing_active[i] else '0'),
        ('swing_arm', lambda i: str(sarm[i]) if sarm[i] in ('a', 'b') else ''),
    ]
    csv_path = args.out + '_traversal.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([c[0] for c in cols])
        for i in range(n):
            w.writerow([c[1](i) for c in cols])

    # ---- phase segments (contiguous (phase, step) runs) ----
    segs = []
    for i in range(n):
        key = (phase[i], int(sidx[i]))
        if segs and segs[-1]['_key'] == key:
            segs[-1]['tick_end'] = i
        else:
            segs.append({'_key': key, 'phase': phase[i], 'step_index': int(sidx[i]),
                         'tick_start': i, 'tick_end': i})
    for s in segs:
        del s['_key']

    # ---- dock events -> tidy ----
    dock_events = []
    for de in (sl.get('dock_events', []) or []):
        tk = nearest_tick(t, float(de['t']))
        dock_events.append({'step_index': int(de.get('step', -1)), 'tick': tk,
                            't': float(de['t']), 'dock_distance_mm': float(de.get('d_mm', float('nan'))),
                            'arm': de.get('arm', ''), 'anchor': de.get('anchor', -1)})

    for s in snaps:
        s['tick'] = nearest_tick(t, s['t'])

    meta = {
        'label': args.label,
        'run_config': json.loads(args.config),
        'tau_w_max_Nm': 5.0,
        'hw_max_Nms': 5.0,
        'dt_s_median': dt_median,
        'dt_s_min': dt_min,
        'dt_note': ('mixed-cadence log: SS/_step rows ~0.1 s (dt_nmpc), inter-step DS rows ~0.01 s '
                    '(dt_qp). Use the t_s column for the time axis; Hdot_s FD uses the actual t.'),
        'n_ticks': n,
        'n_steps': int(sidx.max()) + 1 if n else 0,
        'robot_mass_kg': m_robot,
        'structure_mass_kg': 7110.0,
        'Ltot_definition': ('subtree_angmom[0] (total system angular momentum about the system CoM, '
                            'world frame), recomputed at snapshots via mj_subtreeVel — identical to the '
                            'Fix-A conservation check (audit_fixC_residual.py). Not per-tick (needs full state).'),
        'Hdot_s_definition': ('exact origin-referenced Ḣ_s = Σ_j (r_Cj×f_j + τ_j) from the planned wrench '
                              '(lambda_ref) + structure-frame anchors — the NMPC envelope-constraint quantity, '
                              'read from postproc_F3F4.csv (identical to the C3 metric ‖Ḣ_s‖∞_SS=5.0 at the '
                              'binding). proxy = exact − r_com×Σf (lever from the robot CoM, orbital omitted; '
                              'the FLAG-2 proxy). Defined on SS/_step rows; NaN (blank) on inter-step DS.'),
        'swing_dist_definition': ('d_grip_swing = ‖gripper_site − target_anchor‖ via _gripper_distance — the '
                                  'same gripper/anchor site pair as Fix C dock gate; meaningful when swing_active=1.'),
        'dock_events': dock_events,
        'phase_segments': segs,
        'conservation_snapshots': snaps,
    }
    meta_path = args.out + '_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'wrote {csv_path}  ({n} rows x {len(cols)} cols)')
    print(f'wrote {meta_path}  ({len(snaps)} conservation snapshots, {len(dock_events)} dock events, '
          f'{len(segs)} phase segments)')
    # quick sanity
    print(f'  m_robot={m_robot:.4f} kg  dt median={dt_median:.4f}s min={dt_min:.4f}s  '
          f'n_steps={meta["n_steps"]}')
    print(f'  max|Hdot_s| per-axis (SS)={np.nanmax(np.abs(Hdot_exact),axis=0)} Nm (cap 5)  '
          f'final‖theta_s‖={np.linalg.norm(theta[-1]):.4f} deg  '
          f'hw range=[{hwp.min():.3f},{hwp.max():.3f}] Nms')
    Ln = [s['Ltot_norm'] for s in snaps]
    print(f'  Ltot‖·‖ snapshots: min={min(Ln):.4e} max={max(Ln):.4e} Nms')


if __name__ == '__main__':
    main()
