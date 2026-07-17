#!/usr/bin/env python3
"""DRIFT-CLOSURE T4 — analysis of the extended 450 s settle run.

(1) Traversal identity vs the canonical figC25_addfive for t <= 64.54 s.
(2) Four pass criteria at the settle end.
(3) L_total throughout via mj_subtreeVel on the decimated dense captures.
(4) theta_s(t) / h_w(t) / omega_s(t) traces (CSV + PNG).

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_t4_analyze.py
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
T_SPLIT = 64.55                       # last dock @ 64.54; traversal = t <= this
NEW = 'results/figC25_t4_450s'
CANON = 'results/figC25_addfive'
OUTDIR = 'results/j2_adjconv'

new = json.load(open(f'{NEW}/sim_log.json'))
can = json.load(open(f'{CANON}/sim_log.json'))
model = mujoco.MjModel.from_xml_path(MJCF)

# ── (1) TRAVERSAL IDENTITY, t <= 64.54 ────────────────────────────────
tn = np.asarray(new['t'], float); tc = np.asarray(can['t'], float)
mn = tn <= T_SPLIT; mc = tc <= T_SPLIT
kn = int(mn.sum()); kc = int(mc.sum())
print('=' * 74)
print('DRIFT-CLOSURE T4 — extended 450 s settle')
print('=' * 74)
print(f'\n[1] TRAVERSAL IDENTITY (t <= {T_SPLIT} s)')
print(f'    canonical ticks in window: {kc}   new-run ticks in window: {kn}')
CHANNELS = ['t', 'struct_pos', 'struct_quat', 'struct_euler_deg', 'omega_s',
            'hw_physical', 'tau_w', 'lambda_qp', 'lambda_ref', 'q_ee',
            'e_com', 'r_com', 'v_com', 'e_torso_pos', 'e_torso_ori',
            'p_torso', 'd_grip_swing', 'step_idx']
identical = True
maxdiffs = {}
if kn == kc:
    for ch in CHANNELS:
        if ch not in new or ch not in can:
            continue
        a = np.asarray(new[ch], float)[mn]
        b = np.asarray(can[ch], float)[mc]
        if a.shape != b.shape:
            maxdiffs[ch] = f'SHAPE {a.shape} vs {b.shape}'; identical = False; continue
        with np.errstate(invalid='ignore'):
            d = np.abs(a - b)
            d = np.where(np.isnan(a) & np.isnan(b), 0.0, d)   # NaN==NaN in lambda_ref DS
        md = float(np.nanmax(d)) if d.size else 0.0
        maxdiffs[ch] = md
        if md != 0.0:
            identical = False
else:
    identical = False
    print(f'    !! tick-count mismatch in window ({kn} vs {kc})')
for ch, md in maxdiffs.items():
    flag = '' if (isinstance(md, float) and md == 0.0) else '   <-- NONZERO'
    print(f'      {ch:18}: max|Δ| = {md}{flag}')
print(f'    VERDICT: {"BIT-IDENTICAL ✓" if identical else "NOT bit-identical (see above)"}')

# ── per-tick end-state channels (new run) ─────────────────────────────
hw = np.asarray(new['hw_physical'], float)
oms = np.asarray(new['omega_s'], float)
eul = np.asarray(new['struct_euler_deg'], float)
th = np.linalg.norm(eul, axis=1)
tw = np.asarray(new['tau_w'], float)
tend = tn[-1]
settle = tn > T_SPLIT

# ── (2) FOUR CRITERIA at settle end ───────────────────────────────────
print(f'\n[2] FOUR PASS CRITERIA  (settle end t = {tend:.2f} s)')
hw_end = np.abs(hw[-1]); c1 = bool(np.all(hw_end < 0.05))
oms_end_dps = np.abs(oms[-1]) * R2D; oms_norm_end = np.linalg.norm(oms[-1]) * R2D
c2 = bool(oms_norm_end < 1e-3)
th_end = float(th[-1]); th_end_ax = np.abs(eul[-1]); c3 = bool(th_end < 0.05)
print(f'    C1 |h_w| < 0.05 all axes : {hw_end.round(5).tolist()} Nms   -> {"PASS" if c1 else "FAIL"}')
print(f'    C2 |omega_s| < 1e-3 deg/s: norm={oms_norm_end:.3e} axes={oms_end_dps.round(6).tolist()} '
      f'-> {"PASS" if c2 else "FAIL"}')
print(f'    C3 |theta_s| < 0.05 deg  : norm={th_end:.5f} axes={th_end_ax.round(5).tolist()} '
      f'-> {"PASS" if c3 else "FAIL"}')
# theta_s plateau at 20s-equivalent (entry+20s) for the "from ~0.54deg" context
i20 = int(np.argmin(np.abs(tn - (64.54 + 20.0))))
print(f'       (theta_s at entry+20s = {th[i20]:.4f} deg [canonical plateau]; '
      f'peak-over-settle = {th[settle].max():.4f} deg)')
# transient wheel re-load: peak |h_w| axis during settle vs end
hw_settle_pk = np.abs(hw[settle]).max(axis=0)
i_hwpk = np.where(settle)[0][np.argmax(np.abs(hw[settle]).max(axis=1))]
print(f'    wheel re-load transient  : peak|h_w|_axis over settle = {hw_settle_pk.round(4).tolist()} Nms '
      f'@ t={tn[i_hwpk]:.1f}s  (end {hw_end.round(5).tolist()})')
reload_visible = bool(np.max(hw_settle_pk) > np.max(hw_end) + 1e-6)

# ── (3) L_total THROUGHOUT via dense captures ─────────────────────────
dense = json.load(open(f'{NEW}/ltot_dense.json'))
caps = dense['captures']
d = mujoco.MjData(model)
Lt = []; Lt_t = []
for (tt, q, v) in caps:
    d.qpos[:] = np.asarray(q, float); d.qvel[:] = np.asarray(v, float)
    mujoco.mj_kinematics(model, d); mujoco.mj_comPos(model, d)
    mujoco.mj_comVel(model, d); mujoco.mj_subtreeVel(model, d)
    Lt.append(d.subtree_angmom[0].copy()); Lt_t.append(tt)
Lt = np.array(Lt); Lt_t = np.array(Lt_t)
Lt_absmax = np.abs(Lt).max(axis=0)
c4 = bool(np.all(Lt_absmax < 1e-2))       # numerical-zero band
print(f'\n[3] L_total THROUGHOUT ({len(caps)} dense captures @ {dense["decimate_s"]}s, '
      f't=[{Lt_t.min():.1f},{Lt_t.max():.1f}]s)')
print(f'    C4 L_total ~= 0          : max|L|_axis = {Lt_absmax.round(6).tolist()} Nms  '
      f'max|L|_norm = {np.linalg.norm(Lt,axis=1).max():.3e}  -> {"PASS" if c4 else "FAIL"}')

# ── (4) TRACES: CSV + PNG (downsampled) ───────────────────────────────
ds = max(1, len(tn) // 2000)
tr_idx = np.arange(0, len(tn), ds)
with open(f'{OUTDIR}/t4_traces.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['t_s', 'theta_s_norm_deg', 'theta_s_x_deg', 'theta_s_y_deg', 'theta_s_z_deg',
                'hw_x_Nms', 'hw_y_Nms', 'hw_z_Nms',
                'omega_s_x_dps', 'omega_s_y_dps', 'omega_s_z_dps', 'omega_s_norm_dps'])
    for i in tr_idx:
        w.writerow([f'{tn[i]:.3f}', f'{th[i]:.6e}', *[f'{eul[i,j]:.6e}' for j in range(3)],
                    *[f'{hw[i,j]:.6e}' for j in range(3)],
                    *[f'{oms[i,j]*R2D:.6e}' for j in range(3)], f'{np.linalg.norm(oms[i])*R2D:.6e}'])

fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
ax[0].plot(tn, th, 'b-', lw=0.8); ax[0].axhline(0.05, color='r', ls='--', lw=0.8, label='0.05° gate')
ax[0].axvline(64.54, color='k', ls=':', lw=0.8); ax[0].set_ylabel('|θ_s| [deg]')
ax[0].legend(fontsize=8); ax[0].set_title('T4 extended 450 s settle — managed 2.5 canonical')
for j, c, lb in zip(range(3), 'rgb', ['x', 'y', 'z']):
    ax[1].plot(tn, hw[:, j], c, lw=0.7, label=f'h_w,{lb}')
ax[1].axhline(0.05, color='k', ls='--', lw=0.5); ax[1].axhline(-0.05, color='k', ls='--', lw=0.5)
ax[1].axvline(64.54, color='k', ls=':', lw=0.8); ax[1].set_ylabel('h_w [Nms]'); ax[1].legend(fontsize=8)
ax[2].plot(tn, np.linalg.norm(oms, axis=1) * R2D, 'm-', lw=0.8)
ax[2].axhline(1e-3, color='r', ls='--', lw=0.8, label='1e-3 deg/s gate')
ax[2].axvline(64.54, color='k', ls=':', lw=0.8); ax[2].set_ylabel('|ω_s| [deg/s]')
ax[2].set_yscale('log'); ax[2].set_xlabel('t [s]'); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{OUTDIR}/t4_traces.png', dpi=110); plt.close()
print(f'\n[4] traces -> {OUTDIR}/t4_traces.csv + t4_traces.png')

# ── summary JSON ──────────────────────────────────────────────────────
summary = dict(
    t_end=float(tend), traversal_bit_identical=identical, traversal_maxdiffs=maxdiffs,
    C1_hw_end_Nms=hw_end.round(6).tolist(), C1_pass=c1,
    C2_omega_end_dps=oms_end_dps.round(6).tolist(), C2_omega_norm_dps=oms_norm_end, C2_pass=c2,
    C3_theta_end_deg=th_end, C3_theta_end_axes=th_end_ax.round(6).tolist(), C3_pass=c3,
    theta_at_entry_plus20=float(th[i20]), theta_peak_settle=float(th[settle].max()),
    hw_reload_peak_axis=hw_settle_pk.round(5).tolist(), hw_reload_visible=reload_visible,
    C4_Ltot_absmax_Nms=Lt_absmax.round(6).tolist(), C4_pass=c4,
    n_dense=len(caps),
    all_pass=bool(c1 and c2 and c3 and c4 and identical))
json.dump(summary, open(f'{OUTDIR}/t4_settle450_analysis.json', 'w'), indent=2)
print('\n' + '=' * 74)
print(f'ALL FOUR CRITERIA + TRAVERSAL IDENTITY: '
      f'{"PASS ✓" if summary["all_pass"] else "NOT ALL PASS — see above"}')
print('=' * 74)
