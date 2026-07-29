#!/usr/bin/env python3
"""R25 gate exports E1 + E2 — read-only post-processing of the canonical C run.

E1  Neglected-term residuals of the Stage-1 quasi-inertial model (gate M1)
      rho_rot(t)    = ||omega_s x L_robot||   / ||sum_j (r_Cj x f_j + tau_j)||
      rho_origin(t) = ||v_s      x m v_com||  / ||sum_j (r_Cj x f_j + tau_j)||
      L_robot = L_com + r_com x m v_com, all in the structure frame.

E2  Reduced storage-prediction error (gate P1)
      c(k)          = h_w(k) + L_com(k) + r_com(k) x m v_com(k)
      hhat(k+1|k)   = c(k) - L_com(k+1) - r_com(k+1) x m v_com(k+1)
      e_h^(1)(k)    = h_w(k+1) - hhat(k+1|k)

Sources, and why each
---------------------
denominator      c25_fulldiag.csv `Hdot_s_realized_cont_{x,y,z}_Nm`. Built by
                 scripts/diag_full_diag_export.py::hdot() as
                 cross(r_a, f_a) + tau_a + cross(r_b, f_b) + tau_b from the QP
                 contact wrench, levers in the structure frame — the brief's
                 transmitted moment exactly.
omega_s          logged per tick (mj_data.qvel[3:6], structure frame); present
                 in BOTH the CSV and the sim_log. Cross-checked here.
L_com            sim_log ONLY. The 66-column CSV has no L_com channel; its
                 `Ltot_*` is a different quantity (total system angular momentum
                 from mj_subtreeVel) and is populated at 44 snapshot instants
                 only, blank elsewhere. Do not substitute one for the other.
r_com, v_com     CSV and sim_log agree; CSV used, sim_log cross-checked.
h_w              CSV `hw_*_Nms` == sim_log `hw_physical` == rwa_I_w*qvel[6:9],
                 which is the quantity fed to the NMPC as c_simple's h_w_0
                 (sim_loop.py: hw_for_nmpc). The other channel, sim_log `hw`,
                 is a controller-internal value and is NOT used here.
v_s              NOT LOGGED anywhere — neither CSV nor sim_log carries the
                 structure's linear velocity (qvel[0:3]). Derived here by
                 central finite difference of `struct_pos` (= qpos[0:3], logged
                 per tick). The method is VALIDATED in-band: the same difference
                 applied to `struct_quat` reproduces the independently logged
                 `omega_s`, and that agreement is reported so the reader can
                 judge the derivation rather than take it on trust.

Instant convention: the recorder writes end-of-tick values (rs_f, after the QP
sub-loop and mj_forward), so every quantity here is evaluated at the same
instant. E2 is therefore a consistent-instant conservation residual, not a
replay of the solver's internal c_simple (which is read at tick start).

Outputs (results/review_closure/r25/):
    e1_ratios.csv       per-tick rho_rot, rho_origin and their inputs
    e1_summary.json     median / p95 / max + argmax phase & step, both windows
    e2_storage_error.csv per-NMPC-tick one-step error, per axis
    e2_summary.json     median / p95 / max, per axis and inf-norm, vs h_max
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

CSV = 'results/j2_adjconv/c25_fulldiag.csv'
LOG = sys.argv[1] if len(sys.argv) > 1 else 'results/gate_run_scratch/sim_log.json'
OUT = 'results/review_closure/r25'
H_MAX = 5.0                      # [N·m·s] config.py hw_max / h_max_tight
os.makedirs(OUT, exist_ok=True)


# ── inputs ────────────────────────────────────────────────────────────────
def read_csv(path):
    rows = [l.rstrip('\n').split(',') for l in open(path)]
    hdr, data = rows[0], rows[1:]
    col = {name: i for i, name in enumerate(hdr)}

    def num(name):
        i = col[name]
        return np.array([float(r[i]) if r[i] not in ('', 'nan') else np.nan
                         for r in data])

    def txt(name):
        i = col[name]
        return np.array([r[i] for r in data])
    return num, txt, len(data)


num, txt, n_csv = read_csv(CSV)
log = json.load(open(LOG))
n_log = len(log['t'])
assert n_csv == n_log, f'tick mismatch: csv {n_csv} vs log {n_log}'
n = n_csv

t = num('t_s')
phase = txt('phase')
step = num('step_index')

r_com = np.column_stack([num(f'r_com_{a}_m') for a in 'xyz'])
v_com = np.column_stack([num(f'v_com_{a}_mps') for a in 'xyz'])
omega_s = np.column_stack([num(f'omega_s_{a}_radps') for a in 'xyz'])
hw = np.column_stack([num(f'hw_{a}_Nms') for a in 'xyz'])
Hdot = np.column_stack([num(f'Hdot_s_realized_cont_{a}_Nm') for a in 'xyz'])

L_com = np.asarray(log['L_com'], float)              # sim_log only
struct_pos = np.asarray(log['struct_pos'], float)    # qpos[0:3], per tick
struct_quat = np.asarray(log['struct_quat'], float)  # wxyz, per tick

# robot mass: the value the NMPC was actually built with (rs0.total_mass),
# NOT CentroidalNMPCConfig.robot_mass's 90.0 default.
os.environ.setdefault('MUJOCO_GL', 'disabled')
from crawlbot.core.robot_interface import RobotInterface   # noqa: E402
m_robot = float(RobotInterface('models/VISPA_crawling_fixed.urdf',
                               gravity='zero')._total_mass)
# `_total_mass` is set at construction (robot_interface.py:237) and surfaced on
# each RobotState as `total_mass` (:388) — the same value sim_loop.py passes to
# CentroidalNMPCConfig(robot_mass=rs0.total_mass).

# cross-checks against the sim_log copies of CSV channels
xchecks = {
    'r_com': float(np.nanmax(np.abs(r_com - np.asarray(log['r_com'], float)))),
    'v_com': float(np.nanmax(np.abs(v_com - np.asarray(log['v_com'], float)))),
    'omega_s': float(np.nanmax(np.abs(omega_s - np.asarray(log['omega_s'], float)))),
    'hw_vs_hw_physical': float(np.nanmax(np.abs(
        hw - np.asarray(log['hw_physical'], float)))),
}


# ── v_s derivation, and the validation of the method ──────────────────────
def central_diff(x, tt):
    """d/dt by central difference, one-sided at the ends. x: (n,3)."""
    d = np.zeros_like(x)
    d[1:-1] = (x[2:] - x[:-2]) / (tt[2:] - tt[:-2])[:, None]
    d[0] = (x[1] - x[0]) / (tt[1] - tt[0])
    d[-1] = (x[-1] - x[-2]) / (tt[-1] - tt[-2])
    return d


v_s = central_diff(struct_pos, t)

# validate: the SAME operator on the orientation must reproduce the logged
# omega_s. q̇ -> omega via omega = 2 * vec(q̇ ⊗ q*) in the body frame.
qd = central_diff(struct_quat, t)
w, xq, yq, zq = struct_quat.T
qc = np.column_stack([w, -xq, -yq, -zq])          # conjugate


def qmul(a, b):
    aw, ax, ay, az = a.T
    bw, bx, by, bz = b.T
    return np.column_stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw])


omega_fd = 2.0 * qmul(qc, qd)[:, 1:]              # body-frame angular rate
inner = np.arange(1, n - 1)                        # exclude one-sided ends
abs_err = np.linalg.norm(omega_fd[inner] - omega_s[inner], axis=1)
mag = np.linalg.norm(omega_s[inner], axis=1)

# Report the ABSOLUTE agreement as the primary figure. A relative error is
# meaningless where |omega_s| is itself ~1e-9 rad/s: the largest relative
# values in this run are all division by a vanishing denominator, not a
# breakdown of the operator. Relative is therefore reported conditioned on a
# physically meaningful rate.
REL_FLOOR = 1e-3                                   # [rad/s]
cond = mag > REL_FLOOR
fd_validation = {
    'method': 'central difference of struct_quat -> omega, vs logged omega_s',
    'n_compared': int(len(inner)),
    'abs_err_rad_s': {
        'median': float(np.median(abs_err)),
        'p95': float(np.percentile(abs_err, 95)),
        'max': float(np.max(abs_err)),
    },
    'rel_err_where_omega_gt_1e-3_rad_s': {
        'n': int(cond.sum()),
        'median': float(np.median(abs_err[cond] / mag[cond])),
        'p95': float(np.percentile(abs_err[cond] / mag[cond], 95)),
        'max': float(np.max(abs_err[cond] / mag[cond])),
    },
    'note': ('validates the finite-difference operator used to obtain v_s from '
             'struct_pos; v_s itself is not logged. Unconditioned relative '
             'error reaches O(1) only on ticks where |omega_s| < 1e-6 rad/s, '
             'i.e. where the denominator vanishes.'),
}

# Truncation bound on v_s itself: central (2nd order) vs 5-point (4th order).
# Their difference bounds the truncation error of the estimate actually used.
v_s5 = np.zeros_like(struct_pos)
dt_u = float(np.median(np.diff(t)))
v_s5[2:-2] = (-struct_pos[4:] + 8*struct_pos[3:-1]
              - 8*struct_pos[1:-3] + struct_pos[:-4]) / (12.0 * dt_u)
v_s5[:2], v_s5[-2:] = v_s[:2], v_s[-2:]
dv = np.linalg.norm(v_s5 - v_s, axis=1)[2:-2]
vmag = np.linalg.norm(v_s, axis=1)[2:-2]
fd_validation['v_s_truncation_bound'] = {
    'method': '|central(2nd order) - 5point(4th order)| on struct_pos',
    'abs_median_m_s': float(np.median(dv)),
    'abs_p95_m_s': float(np.percentile(dv, 95)),
    'abs_max_m_s': float(np.max(dv)),
    'rel_to_|v_s|_median': float(np.median(dv / np.maximum(vmag, 1e-12))),
    'dt_s': dt_u,
}

# ── E1 ────────────────────────────────────────────────────────────────────
L_robot = L_com + np.cross(r_com, m_robot * v_com)
num_rot = np.linalg.norm(np.cross(omega_s, L_robot), axis=1)
num_org = np.linalg.norm(np.cross(v_s, m_robot * v_com), axis=1)
den = np.linalg.norm(Hdot, axis=1)

# The denominator is identically 0 on ticks with no active contact wrench
# (the exporter sets Hr = 0 outside steps 0..5). A ratio there is undefined,
# not large — excluded and counted rather than clipped.
DEN_FLOOR = 1e-9
valid = np.isfinite(den) & (den > DEN_FLOOR) & np.isfinite(num_rot)
rho_rot = np.where(valid, num_rot / np.where(valid, den, 1.0), np.nan)
rho_org = np.where(valid, num_org / np.where(valid, den, 1.0), np.nan)

ss = np.where(phase == 'SS')[0]
last_ss = int(ss[-1]) if len(ss) else n - 1
windows = {
    'traversal_to_last_SS': np.arange(0, last_ss + 1),
    'full_logged_run': np.arange(0, n),
}


def stats(vals, idx, label):
    sel = idx[valid[idx]]
    v = vals[sel]
    v = v[np.isfinite(v)]
    if not len(v):
        return {'n': 0}
    imax = sel[int(np.nanargmax(vals[sel]))]
    return {
        'n_ticks_in_window': int(len(idx)),
        'n_valid': int(len(v)),
        'n_excluded_zero_denominator': int(len(idx) - len(v)),
        'median': float(np.median(v)),
        'p95': float(np.percentile(v, 95)),
        'max': float(np.max(v)),
        'argmax_t_s': float(t[imax]),
        'argmax_phase': str(phase[imax]),
        'argmax_step_index': int(step[imax]) if np.isfinite(step[imax]) else None,
        'denominator_at_argmax_Nm': float(den[imax]),
        'numerator_at_argmax_Nm': float((num_rot if label == 'rho_rot'
                                         else num_org)[imax]),
    }


e1 = {
    'quantity_definitions': {
        'rho_rot': '||omega_s x L_robot|| / ||Hdot_s_realized_cont||',
        'rho_origin': '||v_s x m v_com|| / ||Hdot_s_realized_cont||',
        'L_robot': 'L_com + r_com x m v_com  (structure frame)',
        'denominator': 'sum_j (r_Cj x f_j + tau_j), CSV Hdot_s_realized_cont_*',
    },
    'robot_mass_kg': m_robot,
    'v_s_provenance': {
        'logged': False,
        'derived_from': 'struct_pos (= mj_data.qpos[0:3]), central difference',
        'validation': fd_validation,
    },
    'denominator_floor': DEN_FLOOR,
    'csv_vs_simlog_max_abs_diff': xchecks,
    'windows': {},
}
for wname, widx in windows.items():
    e1['windows'][wname] = {'rho_rot': stats(rho_rot, widx, 'rho_rot'),
                            'rho_origin': stats(rho_org, widx, 'rho_origin')}

with open(f'{OUT}/e1_ratios.csv', 'w') as f:
    f.write('t_s,phase,step_index,num_rot_Nm,num_origin_Nm,denominator_Nm,'
            'rho_rot,rho_origin,valid\n')
    for i in range(n):
        f.write(f'{t[i]:.6f},{phase[i]},{"" if not np.isfinite(step[i]) else int(step[i])},'
                f'{num_rot[i]:.6e},{num_org[i]:.6e},{den[i]:.6e},'
                f'{rho_rot[i]:.6e},{rho_org[i]:.6e},{int(valid[i])}\n')
json.dump(e1, open(f'{OUT}/e1_summary.json', 'w'), indent=2)


# ── E2 ────────────────────────────────────────────────────────────────────
c_k = hw + L_com + np.cross(r_com, m_robot * v_com)
hhat_next = c_k[:-1] - L_com[1:] - np.cross(r_com[1:], m_robot * v_com[1:])
e_h = hw[1:] - hhat_next                            # (n-1, 3)

nmpc_ok = num('nmpc_ok').astype(bool)
active = nmpc_ok[:-1] & nmpc_ok[1:] & np.all(np.isfinite(e_h), axis=1)


def axstats(v):
    return {'median': float(np.median(v)), 'p95': float(np.percentile(v, 95)),
            'max': float(np.max(v)),
            'max_frac_of_h_max': float(np.max(v) / H_MAX),
            'p95_frac_of_h_max': float(np.percentile(v, 95) / H_MAX),
            'median_frac_of_h_max': float(np.median(v) / H_MAX)}


ea = np.abs(e_h[active])
e2 = {
    'definition': {
        'c(k)': 'h_w(k) + L_com(k) + r_com(k) x m v_com(k)',
        'hhat(k+1|k)': 'c(k) - L_com(k+1) - r_com(k+1) x m v_com(k+1)',
        'e_h^(1)(k)': 'h_w(k+1) - hhat(k+1|k)',
    },
    'horizon_error_e_h_N': {
        'computed': False,
        'reason': ('the NMPC predicted trajectory is not persisted — sim_log '
                   'carries no x_plan / predicted-state channel (only '
                   'preplanner_T_steps and gait_plan), so hhat(k+N|k) cannot be '
                   'evaluated along the solved trajectory from committed data. '
                   'Per the brief, the one-step error is reported instead and no '
                   'horizon export was added.'),
    },
    'h_max_Nms': H_MAX,
    'robot_mass_kg': m_robot,
    'n_pairs_total': int(len(e_h)),
    'n_pairs_nmpc_active': int(active.sum()),
    'per_axis_abs': {a: axstats(ea[:, i]) for i, a in enumerate('xyz')},
    'inf_norm_abs': axstats(np.max(ea, axis=1)),
}
with open(f'{OUT}/e2_storage_error.csv', 'w') as f:
    f.write('t_k_s,t_kp1_s,phase_kp1,e_h_x_Nms,e_h_y_Nms,e_h_z_Nms,'
            'e_h_inf_Nms,nmpc_active\n')
    for i in range(len(e_h)):
        f.write(f'{t[i]:.6f},{t[i+1]:.6f},{phase[i+1]},'
                f'{e_h[i,0]:.6e},{e_h[i,1]:.6e},{e_h[i,2]:.6e},'
                f'{np.max(np.abs(e_h[i])):.6e},{int(active[i])}\n')
json.dump(e2, open(f'{OUT}/e2_summary.json', 'w'), indent=2)

# ── console ───────────────────────────────────────────────────────────────
print(f'ticks {n}   m_robot {m_robot:.4f} kg   log {LOG}')
print(f'csv-vs-simlog max|diff|: ' +
      ', '.join(f'{k}={v:.2e}' for k, v in xchecks.items()))
_a = fd_validation['abs_err_rad_s']; _r = fd_validation['rel_err_where_omega_gt_1e-3_rad_s']
_v = fd_validation['v_s_truncation_bound']
print(f'\nFD operator vs logged omega_s  |abs| median {_a["median"]:.2e}  '
      f'p95 {_a["p95"]:.2e}  max {_a["max"]:.2e} rad/s')
print(f'  rel err where |omega_s|>1e-3: median {_r["median"]:.2e}  '
      f'p95 {_r["p95"]:.2e}  max {_r["max"]:.2e}  (n={_r["n"]})')
print(f'v_s truncation bound (2nd vs 4th order): max {_v["abs_max_m_s"]:.2e} m/s, '
      f'median rel {_v["rel_to_|v_s|_median"]:.2e}')
for wname in windows:
    print(f'\n[E1] {wname}')
    for r in ('rho_rot', 'rho_origin'):
        s = e1['windows'][wname][r]
        print(f'  {r:11s} median {s["median"]:.3e}  p95 {s["p95"]:.3e}  '
              f'max {s["max"]:.3e}  @ t={s["argmax_t_s"]:.2f}s '
              f'{s["argmax_phase"]} step {s["argmax_step_index"]}  '
              f'(valid {s["n_valid"]}/{s["n_ticks_in_window"]})')
print(f'\n[E2] one-step storage error, {int(active.sum())} NMPC-active pairs')
for a in 'xyz':
    s = e2['per_axis_abs'][a]
    print(f'  |e_h,{a}|  median {s["median"]:.3e}  p95 {s["p95"]:.3e}  '
          f'max {s["max"]:.3e} Nms  ({100*s["max_frac_of_h_max"]:.3f} % of h_max)')
s = e2['inf_norm_abs']
print(f'  inf-norm  median {s["median"]:.3e}  p95 {s["p95"]:.3e}  '
      f'max {s["max"]:.3e} Nms  ({100*s["max_frac_of_h_max"]:.3f} % of h_max)')
print(f'\nwrote {OUT}/e1_ratios.csv, e1_summary.json, '
      f'e2_storage_error.csv, e2_summary.json')
