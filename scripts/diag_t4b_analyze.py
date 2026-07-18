#!/usr/bin/env python3
"""T4b — analysis of the 900 s settle: Gate 0 identity, C1-C4 @ 900 s, t_cross,
tail-tau re-fit, and the COMMITTED plotting-grade trace + L_total CSVs.

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_t4b_analyze.py
"""
import json
import csv
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.export_figure_data import MJCF

R2D = 180.0 / np.pi
I_COMP = 2.18e3
T_SPLIT = 64.55
GRID_HZ = 5.0                          # uniform plotting-grade trace rate (>= 2 Hz)
NEW = 'results/figC25_t4b_900s'
CANON = 'results/figC25_addfive'
OUTDIR = 'results/j2_adjconv'

new = json.load(open(f'{NEW}/sim_log.json'))
can = json.load(open(f'{CANON}/sim_log.json'))
model = mujoco.MjModel.from_xml_path(MJCF)

tn = np.asarray(new['t'], float); tc = np.asarray(can['t'], float)
mn = tn <= T_SPLIT; mc = tc <= T_SPLIT
print('=' * 74)
print('T4b — extended 900 s settle')
print('=' * 74)

# ── Gate 0: traversal bit-identity (18 channels) ──────────────────────
CH = ['t', 'struct_pos', 'struct_quat', 'struct_euler_deg', 'omega_s',
      'hw_physical', 'tau_w', 'lambda_qp', 'lambda_ref', 'q_ee', 'e_com',
      'r_com', 'v_com', 'e_torso_pos', 'e_torso_ori', 'p_torso',
      'd_grip_swing', 'step_idx']
identical = (int(mn.sum()) == int(mc.sum()))
maxdiffs = {}
if identical:
    for ch in CH:
        if ch not in new or ch not in can:
            continue
        a = np.asarray(new[ch], float)[mn]; b = np.asarray(can[ch], float)[mc]
        if a.shape != b.shape:
            maxdiffs[ch] = 'SHAPE'; identical = False; continue
        with np.errstate(invalid='ignore'):
            dd = np.abs(a - b); dd = np.where(np.isnan(a) & np.isnan(b), 0.0, dd)
        md = float(np.nanmax(dd)) if dd.size else 0.0
        maxdiffs[ch] = md
        if md != 0.0:
            identical = False
print(f'\n[Gate 0] traversal identity (t<=64.54s): ticks {int(mn.sum())} vs {int(mc.sum())}; '
      f'max|Δ| over 18 ch = {max([v for v in maxdiffs.values() if isinstance(v,float)], default=0.0)}  '
      f'-> {"BIT-IDENTICAL ✓" if identical else "NOT identical ✗"}')

# ── per-tick channels ─────────────────────────────────────────────────
hw = np.asarray(new['hw_physical'], float)
oms = np.asarray(new['omega_s'], float)
eul = np.asarray(new['struct_euler_deg'], float)
tw = np.asarray(new['tau_w'], float)
th = np.linalg.norm(eul, axis=1)
thz = np.abs(eul[:, 2])
tend = tn[-1]
settle = tn > T_SPLIT

# ── C1-C4 at settle end ───────────────────────────────────────────────
hw_end = np.abs(hw[-1]); c1 = bool(np.all(hw_end < 0.05))
oms_norm_end = np.linalg.norm(oms[-1]) * R2D; c2 = bool(oms_norm_end < 1e-3)
th_end = float(th[-1]); c3 = bool(th_end < 0.05)

# ── t_cross: first time |theta_s,z| stays < 0.05 deg (after the peak) ──
above = thz >= 0.05
t_cross = None; settle_time_cross = None
if above.any():
    last_above = np.where(above)[0].max()
    if last_above + 1 < len(tn):
        t_cross = float(tn[last_above + 1])
        settle_time_cross = t_cross - 64.54
crossed = t_cross is not None

# ── tail time-constant re-fit on t > 250 s ────────────────────────────
mfit = tn > 250.0
A = np.polyfit(tn[mfit], np.log(np.abs(eul[mfit, 2]) + 1e-12), 1)
tau_tail = -1.0 / A[0] if A[0] < 0 else float('inf')
t_cross_extrap = (np.log(0.05) - A[1]) / A[0] if A[0] < 0 else None

# ── L_total throughout (dense captures) ───────────────────────────────
dense = json.load(open(f'{NEW}/ltot_dense.json'))
caps = dense['captures']
d = mujoco.MjData(model); Lt = []; Ltt = []
for (tt, q, v) in caps:
    d.qpos[:] = np.asarray(q, float); d.qvel[:] = np.asarray(v, float)
    mujoco.mj_kinematics(model, d); mujoco.mj_comPos(model, d)
    mujoco.mj_comVel(model, d); mujoco.mj_subtreeVel(model, d)
    Lt.append(d.subtree_angmom[0].copy()); Ltt.append(tt)
Lt = np.array(Lt); Ltt = np.array(Ltt)
Lt_absmax = np.abs(Lt).max(axis=0); c4 = bool(np.all(Lt_absmax < 1e-2))

print(f'\n[C1] |h_w| < 0.05  : {hw_end.round(5).tolist()} Nms  -> {"PASS" if c1 else "FAIL"}')
print(f'[C2] |ω_s| < 1e-3  : {oms_norm_end:.3e} deg/s        -> {"PASS" if c2 else "FAIL"}')
print(f'[C3] |θ_s| < 0.05° : {th_end:.5f} (z={np.abs(eul[-1,2]):.5f})   -> {"PASS" if c3 else "FAIL"}')
print(f'[C4] L_total ≈ 0   : max|axis| {Lt_absmax.round(6).tolist()} Nms -> {"PASS" if c4 else "FAIL"}')
print(f'\n[t_cross] first |θ_s,z| < 0.05° (stays): '
      f'{"t=%.2f s (%.1f s of settle past dock)" % (t_cross, settle_time_cross) if crossed else "NO CROSS by 900 s — TAIL FINDING"}')
print(f'[tail τ]  re-fit on t>250s: {tau_tail:.1f} s  (T4 fit ≈ 305 s); '
      f'exp-extrapolated cross ≈ {("%.0f s" % t_cross_extrap) if t_cross_extrap else "n/a"}')

# ── plotting-grade trace CSV (uniform 5 Hz nearest-tick) ──────────────
grid = np.arange(0.0, tend + 1e-9, 1.0 / GRID_HZ)
idx = [int(np.argmin(np.abs(tn - g))) for g in grid]
with open(f'{OUTDIR}/t4b_trace_900s.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['t_s',
                'omega_s_x_dps', 'omega_s_y_dps', 'omega_s_z_dps',
                'hw_x_Nms', 'hw_y_Nms', 'hw_z_Nms',
                'theta_s_x_deg', 'theta_s_y_deg', 'theta_s_z_deg',
                'tau_w_x_Nm', 'tau_w_y_Nm', 'tau_w_z_Nm'])
    for i in idx:
        w.writerow([f'{tn[i]:.3f}',
                    *[f'{oms[i,j]*R2D:.6e}' for j in range(3)],
                    *[f'{hw[i,j]:.6e}' for j in range(3)],
                    *[f'{eul[i,j]:.6e}' for j in range(3)],
                    *[f'{tw[i,j]:.6e}' for j in range(3)]])
print(f'\n[trace] {OUTDIR}/t4b_trace_900s.csv  ({len(idx)} rows @ {GRID_HZ} Hz uniform)')

# ── L_total CSV (committed, 2 Hz dense) ───────────────────────────────
with open(f'{OUTDIR}/t4b_ltot_900s.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['t_s', 'Ltot_x_Nms', 'Ltot_y_Nms', 'Ltot_z_Nms', 'Ltot_norm_Nms'])
    for tt, L in zip(Ltt, Lt):
        w.writerow([f'{tt:.3f}', f'{L[0]:.6e}', f'{L[1]:.6e}', f'{L[2]:.6e}',
                    f'{np.linalg.norm(L):.6e}'])
print(f'[ltot ] {OUTDIR}/t4b_ltot_900s.csv  ({len(Ltt)} rows @ {dense["decimate_s"]}s)')

# ── PNG ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
ax[0].plot(tn, th, 'b-', lw=0.7, label='|θ_s|')
ax[0].plot(tn, np.abs(eul[:, 2]), 'c-', lw=0.6, label='|θ_s,z|')
ax[0].axhline(0.05, color='r', ls='--', lw=0.8, label='0.05° gate')
ax[0].axvline(64.54, color='k', ls=':', lw=0.8)
if crossed:
    ax[0].axvline(t_cross, color='g', ls='-', lw=0.9, label=f't_cross={t_cross:.0f}s')
ax[0].set_ylabel('θ_s [deg]'); ax[0].legend(fontsize=8)
ax[0].set_title('T4b extended 900 s settle — managed 2.5 canonical')
for j, c, lb in zip(range(3), 'rgb', 'xyz'):
    ax[1].plot(tn, hw[:, j], c, lw=0.7, label=f'h_w,{lb}')
ax[1].axhline(0.05, color='k', ls='--', lw=0.4); ax[1].axhline(-0.05, color='k', ls='--', lw=0.4)
ax[1].axvline(64.54, color='k', ls=':', lw=0.8); ax[1].set_ylabel('h_w [Nms]'); ax[1].legend(fontsize=8)
ax[2].plot(tn, np.linalg.norm(oms, axis=1) * R2D, 'm-', lw=0.8)
ax[2].axhline(1e-3, color='r', ls='--', lw=0.8, label='1e-3 deg/s gate')
ax[2].axvline(64.54, color='k', ls=':', lw=0.8); ax[2].set_yscale('log')
ax[2].set_ylabel('|ω_s| [deg/s]'); ax[2].set_xlabel('t [s]'); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{OUTDIR}/t4b_traces.png', dpi=110); plt.close()
print(f'[png  ] {OUTDIR}/t4b_traces.png')

summary = dict(
    settle_seconds=900.0, t_end=float(tend), traversal_bit_identical=identical,
    C1_hw_end=hw_end.round(6).tolist(), C1_pass=c1,
    C2_omega_norm_dps=oms_norm_end, C2_pass=c2,
    C3_theta_end_deg=th_end, C3_theta_end_z=float(np.abs(eul[-1, 2])), C3_pass=c3,
    C4_Ltot_absmax=Lt_absmax.round(6).tolist(), C4_pass=c4,
    t_cross_thetaz_005=t_cross, settle_time_to_cross=settle_time_cross, crossed=crossed,
    tail_tau_refit_s=float(tau_tail), tail_extrap_cross_s=(float(t_cross_extrap) if t_cross_extrap else None),
    all_pass=bool(identical and c1 and c2 and c3 and c4))
json.dump(summary, open(f'{OUTDIR}/t4b_settle900_analysis.json', 'w'), indent=2)
print('\n' + '=' * 74)
print(f'RESULT: Gate0 {"✓" if identical else "✗"} | C1 {"✓" if c1 else "✗"} '
      f'C2 {"✓" if c2 else "✗"} C3 {"✓" if c3 else "✗"} C4 {"✓" if c4 else "✗"} | '
      f'{"ALL PASS ✓" if summary["all_pass"] else "C3 pending/fail — see t_cross"}')
print('=' * 74)
