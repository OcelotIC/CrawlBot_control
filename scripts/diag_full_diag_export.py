#!/usr/bin/env python3
"""Phase FULL-DIAG-EXPORT — complete per-tick diagnostic export (standard validation export).

READ-ONLY on a committed run's on-disk sim_log.json. NO re-run, NO canonical change.
Reusable on ANY config: `--run-dir results/figC_<tag>`.

Channels (grouped): dock-gate (A), CoM tracking (C), torso tracking (D), momentum
tracking (E), QP quality (F), plus the existing envelope columns. Provenance and
missing-data handling documented in the companion report.

Realized Hdot_s: validated cross(anchor,f)+tau on lambda_qp, all ticks (USERW2-DATA).
Planned Hdot_s: same formula on lambda_ref (NMPC target) — SS ONLY (lambda_ref is NaN
in DS when the NMPC is bypassed). ori_err_deg: recomputed from q_ee (anchor=Identity in
struct frame) — validated vs dock_gate_trace to <=0.13 deg. twist / docked flag: LOGGED
ONLY at the 103 dock_gate_trace evaluation ticks (per-tick twist needs qpos/qvel, which
are NOT logged -> MISSING, marked, not fabricated).

CAVEAT — `nmpc_ok` conflates "not called" with "failed" (CLEANUP-2 finding F3).
The NMPC is only invoked in SS and in the terminal DS settle; it is NOT invoked
during DS_interstep, and those ticks are exported as `nmpc_ok = 0`. On the frozen
canonical run that is 1368 of 2077 ticks, so reading the column as a whole gives a
FALSE 34.1 % success rate. The true rate is 100 % — 508/508 SS + 201/201
DS_terminal = 709/709 solves succeeded. **Always filter by `phase` before
interpreting `nmpc_ok`** (`phase == 'SS' or phase == 'DS_terminal'`).

The encoding is deliberately left as-is: changing the value or adding an
`nmpc_called` column would alter this CSV and therefore require regenerating the
frozen paper baseline `results/j2_adjconv/c25_fulldiag.csv` under a Tier-1
exception (see gate/EXCEPTIONS.md). Deferred until after submission.

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_full_diag_export.py [--run-dir DIR]
"""
import sys
import json
import csv
import numpy as np
import pinocchio as pin
import mujoco
from scripts.export_figure_data import (load_anchors_struct_frame, phase_per_tick, MJCF,
                                        ltot_at_snapshots, nearest_tick)

RUN = 'results/figC_userw2'
if '--run-dir' in sys.argv:
    RUN = sys.argv[sys.argv.index('--run-dir') + 1]
sl = json.load(open(f'{RUN}/sim_log.json'))
model = mujoco.MjModel.from_xml_path(MJCF)

STANCE_ARM = ['a', 'b', 'a', 'b', 'a', 'b']; STANCE_ANCHOR = [2, 3, 3, 4, 4, 5]
SWING_ARM = ['b', 'a', 'b', 'a', 'b', 'a']; SWING_ANCHOR = [3, 3, 4, 4, 5, 5]
WELD_R_MM = 5.0; ORI_TH = 5.0                      # config.py:35 / :42

n = len(sl['t'])
g = lambda k: np.asarray(sl[k], float)
t = g('t'); si = np.asarray(sl['step_idx'], int); ph = np.array(phase_per_tick(sl))
sw = np.asarray(sl['swing_arm'])
lam = g('lambda_qp'); lref = g('lambda_ref')
qee = g('q_ee'); dg = g('d_grip_swing')
aA, aB = load_anchors_struct_frame(model, np.asarray(sl['struct_pos'])[0],
                                   np.asarray(sl['struct_quat'])[0])


def levers(k):
    if STANCE_ARM[k] == 'a':
        return aA[STANCE_ANCHOR[k]], aB[SWING_ANCHOR[k]]
    return aA[SWING_ANCHOR[k]], aB[STANCE_ANCHOR[k]]


def hdot(L, k):
    ra, rb = levers(k)
    return np.cross(ra, L[0:3]) + L[3:6] + np.cross(rb, L[6:9]) + L[9:12]


Hr = np.full((n, 3), np.nan); Hp = np.full((n, 3), np.nan)
ori = np.full(n, np.nan)
for i in range(n):
    k = si[i]
    if 0 <= k <= 5:
        Hr[i] = hdot(lam[i], k)
        if np.all(np.isfinite(lref[i])):
            Hp[i] = hdot(lref[i], k)
    else:
        Hr[i] = 0.0
    w, x, y, z = qee[i]
    try:
        ori[i] = np.degrees(np.linalg.norm(pin.log3(pin.Quaternion(w, x, y, z).toRotationMatrix())))
    except Exception:
        ori[i] = np.nan

# gate-eval sparse: map dock_gate_trace rows -> tick (nearest t within same step)
gate_eval = np.zeros(n, int); twist_ge = np.full(n, np.nan); docked_ge = np.full(n, np.nan)
for row in sl.get('dock_gate_trace', []):
    cand = np.where(si == row['step'])[0]
    if not len(cand):
        continue
    i = cand[int(np.argmin(np.abs(t[cand] - row['t'])))]
    gate_eval[i] = 1; twist_ge[i] = row['twist']; docked_ge[i] = 1 if row['fired'] else 0

# channel arrays
rc = g('r_com'); rcr = g('r_com_ref'); ec = g('e_com'); vc = g('v_com'); vcr = g('v_com_ref')
eul = g('struct_euler_deg'); etp = g('e_torso_pos'); eto = g('e_torso_ori')
pt = g('p_torso'); ptr = g('p_torso_ref'); eep = g('e_ee_pos'); eeo = g('e_ee_ori')
tw = g('tau_w'); hw = g('hw_physical')
qpok = np.asarray(sl['qp_ok']); nmok = np.asarray(sl['nmpc_ok'])
qpt = g('qp_time_ms'); nmt = g('nmpc_time_ms'); nmit = g('nmpc_iterations')
lqn = g('lambda_qp_norm'); lrn = g('lambda_ref_norm'); tmj = g('tau_max_joint')

# ── DRIFT-CLOSURE T2 (append-only diagnostic channels; no control path) ──
# omega_s: structure angular rate = qvel[3:6] in R_s, logged per-tick (2077).
# L_total: total system angular momentum = mj_subtreeVel(subtree_angmom[0]) on the
#   stored (qpos, qvel) SNAPSHOTS ONLY (44 event instants) -> mapped to nearest tick,
#   left blank on the remaining ticks (qpos/qvel not logged per-tick -> not fabricated).
oms = g('omega_s')
ltot_snaps = ltot_at_snapshots(sl, model)
ltot_col = np.full((n, 3), np.nan); ltot_norm = np.full(n, np.nan)
for _s in ltot_snaps:
    _i = nearest_tick(t, _s['t'])
    ltot_col[_i] = [_s['Ltot_x'], _s['Ltot_y'], _s['Ltot_z']]; ltot_norm[_i] = _s['Ltot_norm']


def f6(v): return f'{v:.6e}'
def fb(v): return '' if not np.isfinite(v) else str(int(v))


# ── C2.2: solver-diagnostic columns (OPT-IN, `--solver-diag`) ───────────────
# Appended AFTER the 66 frozen columns; the 66 are untouched either way.
#
# Default OFF, deliberately. `gate/run_gate.py`'s identity check requires EXACT
# header equality (`diff_csv`, R[0] != B[0] -> FAIL "header mismatch"), so
# emitting these unconditionally would make the shared gate fail for every
# caller until `results/j2_adjconv/c25_fulldiag.csv` is regenerated — a frozen
# paper artifact, and not this stream's to regenerate. With the flag off the
# gate's export is byte-for-byte what it was; with it on the caller gets the
# richer CSV. Flipping the default is Idriss's call, together with the
# baseline regeneration it requires.
SOLVER_DIAG = '--solver-diag' in sys.argv
_has_qp_stats = 'qp_n_solves' in sl
if SOLVER_DIAG and not _has_qp_stats:
    print('  [solver-diag] WARNING: this sim_log predates the qp_* channels '
          '(C2.2.1); those columns will be written as the not-measured '
          'sentinel. Re-run the sim to populate them.')
if SOLVER_DIAG:
    _sd = lambda k, d: (np.asarray(sl[k]) if k in sl                # noqa: E731
                        else np.full(n, d))
    qpsms = _sd('qp_solve_ms_sum', 0.0).astype(float)
    qpsmx = _sd('qp_solve_ms_max', 0.0).astype(float)
    qpits = _sd('qp_iter_sum', 0).astype(int)
    qpns = _sd('qp_n_solves', 0).astype(int)
    qpnf = _sd('qp_n_failed', 0).astype(int)
    qpsw = _sd('qp_status_worst', -1).astype(int)
    nmss = (list(sl['nmpc_status_str']) if 'nmpc_status_str' in sl
            else [''] * n)
    # C2.3 — AOCS decomposition, 6 per-axis vectors + the measured flag.
    _AT = ('tau_ff', 'tau_att_p', 'tau_rate_d', 'tau_accel_d',
           'tau_antiwindup', 'tau_w_preclip')
    aocs = {k: (np.asarray(sl[f'aocs_{k}'], float) if f'aocs_{k}' in sl
                else np.zeros((n, 3))) for k in _AT}
    aocs_meas = (np.asarray(sl['aocs_decomp_measured'], int)
                 if 'aocs_decomp_measured' in sl else np.zeros(n, int))

SOLVER_DIAG_COLS = (['qp_solve_ms_sum', 'qp_solve_ms_max', 'qp_iter_sum',
                     'qp_n_solves', 'qp_n_failed', 'qp_status_worst',
                     'nmpc_status_str']
                    # C2.3 — AOCS decomposition. The five terms sum to
                    # aocs_tau_w_preclip by construction; clip(preclip) is the
                    # already-exported tauw_*_Nm, so the saturation is the
                    # difference between the two.
                    + [f'aocs_{k}_{a}_Nm' for k in
                       ('tau_ff', 'tau_att_p', 'tau_rate_d', 'tau_accel_d',
                        'tau_antiwindup', 'tau_w_preclip') for a in 'xyz']
                    + ['aocs_decomp_measured'])

cols = (['t_s', 'phase', 'step_index', 'swing_arm',
         'd_grip_swing_mm', 'ori_err_deg', 'pos_ok', 'ori_ok',
         'gate_eval', 'twist_norm_at_gateeval', 'docked_at_gateeval'] +
        [f'Hdot_s_realized_cont_{a}_Nm' for a in 'xyz'] + ['Hdot_s_realized_norm_Nm'] +
        [f'Hdot_s_planned_ss_{a}_Nm' for a in 'xyz'] + ['Hdot_s_planned_norm_Nm'] +
        [f'tauw_{a}_Nm' for a in 'xyz'] + [f'hw_{a}_Nms' for a in 'xyz'] +
        [f'r_com_{a}_m' for a in 'xyz'] + [f'r_com_ref_{a}_m' for a in 'xyz'] + ['e_com_m'] +
        [f'v_com_{a}_mps' for a in 'xyz'] + [f'v_com_ref_{a}_mps' for a in 'xyz'] +
        [f'theta_s_{a}_deg' for a in 'xyz'] + ['e_torso_pos_m', 'e_torso_ori_deg'] +
        [f'p_torso_{a}_m' for a in 'xyz'] + [f'p_torso_ref_{a}_m' for a in 'xyz'] +
        ['e_ee_pos_mm', 'e_ee_ori_deg',
         'qp_ok', 'nmpc_ok', 'qp_time_ms', 'nmpc_time_ms', 'nmpc_iterations',
         'lambda_qp_norm', 'lambda_ref_norm', 'tau_max_joint_Nm'] +
        [f'omega_s_{a}_radps' for a in 'xyz'] +
        [f'Ltot_{a}_Nms' for a in 'xyz'] + ['Ltot_norm_Nms'] +
        (SOLVER_DIAG_COLS if SOLVER_DIAG else []))

OUT_PREFIX = 'results/j2_adjconv/userw2'
if '--out-prefix' in sys.argv:
    OUT_PREFIX = sys.argv[sys.argv.index('--out-prefix') + 1]
out_csv = f'{OUT_PREFIX}_fulldiag.csv'
with open(out_csv, 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(cols)
    for i in range(n):
        d_mm = np.nan if not np.isfinite(dg[i]) else 1000 * dg[i]
        pos_ok = '' if not np.isfinite(d_mm) else str(int(d_mm < WELD_R_MM))
        ori_ok = '' if not np.isfinite(ori[i]) else str(int(ori[i] < ORI_TH))
        w.writerow([f'{t[i]:.4f}', ph[i], si[i], sw[i],
                    ('' if not np.isfinite(d_mm) else f'{d_mm:.4f}'),
                    ('' if not np.isfinite(ori[i]) else f'{ori[i]:.4f}'), pos_ok, ori_ok,
                    gate_eval[i], fb(twist_ge[i]) if not np.isfinite(twist_ge[i]) else f6(twist_ge[i]),
                    fb(docked_ge[i])] +
                   [f6(Hr[i, j]) for j in range(3)] + [f6(np.linalg.norm(Hr[i]))] +
                   [('' if not np.isfinite(Hp[i, j]) else f6(Hp[i, j])) for j in range(3)] +
                   [('' if not np.all(np.isfinite(Hp[i])) else f6(np.linalg.norm(Hp[i])))] +
                   [f6(tw[i, j]) for j in range(3)] + [f6(hw[i, j]) for j in range(3)] +
                   [f6(rc[i, j]) for j in range(3)] + [f6(rcr[i, j]) for j in range(3)] + [f6(ec[i])] +
                   [f6(vc[i, j]) for j in range(3)] + [f6(vcr[i, j]) for j in range(3)] +
                   [f6(eul[i, j]) for j in range(3)] + [f6(etp[i]), f6(eto[i])] +
                   [f6(pt[i, j]) for j in range(3)] + [f6(ptr[i, j]) for j in range(3)] +
                   [f6(1000 * eep[i]), f6(eeo[i]),
                    int(bool(qpok[i])), int(bool(nmok[i])), f6(qpt[i]), f6(nmt[i]), int(nmit[i]),
                    f6(lqn[i]), ('' if not np.isfinite(lrn[i]) else f6(lrn[i])), f6(tmj[i])] +
                   [f6(oms[i, j]) for j in range(3)] +
                   [('' if not np.isfinite(ltot_col[i, j]) else f6(ltot_col[i, j])) for j in range(3)] +
                   ['' if not np.isfinite(ltot_norm[i]) else f6(ltot_norm[i])] +
                   ([f6(qpsms[i]), f6(qpsmx[i]), int(qpits[i]), int(qpns[i]),
                     int(qpnf[i]), int(qpsw[i]), nmss[i]]
                    + [f6(aocs[k][i, j]) for k in _AT for j in range(3)]
                    + [int(aocs_meas[i])]
                    if SOLVER_DIAG else []))
print(f'wrote {out_csv}  ({n} ticks, {len(cols)} cols'
      f'{" incl. --solver-diag" if SOLVER_DIAG else ""})')

# ── B: per-step at-weld vs min-over-swing ──
dev = {d['step']: d for d in sl.get('dock_events', [])}
print('\n=== B: AT-WELD (first docked) vs MIN-OVER-SWING, all 6 steps ===')
print(' step | weld t | d@weld | d_min_swing | fly-by? | ori@weld | twist@weld')
Brows = []
for k in range(6):
    ssm = (si == k) & (ph == 'SS') & np.isfinite(dg)
    dmin = float(1000 * dg[ssm].min()) if ssm.any() else float('nan')
    e = dev.get(k, {})
    dweld = e.get('d_mm', float('nan')); flyby = dweld - dmin
    Brows.append(dict(step=k, weld_t=e.get('t'), d_weld_mm=dweld, d_min_swing_mm=round(dmin, 3),
                      flyby_gap_mm=round(float(flyby), 3), ori_weld_deg=e.get('ori_deg'),
                      twist_weld=e.get('twist'), arm=e.get('arm'), anchor=e.get('anchor')))
    print(f'  {k}   | {e.get("t"):6}s | {dweld:6.3f} | {dmin:10.3f}  | {flyby:+6.3f} '
          f'| {e.get("ori_deg"):7} | {e.get("twist"):.6f}')

# C2.2.3 / C2.2.4 — pre-planner NLP records and the host that produced the
# timings, into the run metadata. Unconditional (the meta JSON is not
# byte-compared by any gate) and empty on logs predating the channels, so an
# archived run still exports without them rather than failing.
# C2.1 — the per-dock 6-D twist at capture and the thresholds in force now
# ride on the dock_events rows themselves; pass them through verbatim rather
# than re-deriving, and carry the gate trace so a refused capture can be read
# against the same decomposition as an accepted one.
_meta = {'run': RUN, 'csv': out_csv, 'n_ticks': n, 'at_weld_vs_min': Brows,
         'dock_events': sl.get('dock_events', []),
         'dock_gate_trace': sl.get('dock_gate_trace', []),
         'ltot_snapshots': ltot_snaps,
         'preplanner_stats': sl.get('preplanner_stats', []),
         'host': (sl.get('environment') or {}).get('host', {}),
         'library_versions': {
             k: (sl.get('environment') or {}).get(k)
             for k in ('python_version', 'mujoco_version',
                       'pinocchio_version', 'numpy_version')}}
json.dump(_meta, open(f'{OUT_PREFIX}_fulldiag_meta.json', 'w'), indent=2)
print(f'\nwrote {OUT_PREFIX}_fulldiag_meta.json'
      f"  (preplanner_stats: {len(_meta['preplanner_stats'])} records, "
      f"host: {'yes' if _meta['host'] else 'ABSENT — log predates C2.2.4'})")
