#!/usr/bin/env python3
"""Phase ABL-CMD probe (READ-ONLY, no commit) — resolve which quantity in the raw
figU_rateoff sim_log drives the AOCS wheel command to its ±5 clamp, so we know
whether a faithful pre-clamp `tauw_commanded` reconstruction is even possible.

Compares, per axis over the run:
  - logged clamped tau_w                       (the applied wheel torque)
  - |L_dot|            (realized centroidal momentum rate; ff_term = -L_dot-orbital on SS)
  - |H_dot_est|        (whatever the log stored under this key)
  - reconstructed pre-clamp AOCS command using ff = -L_dot - orbital + feedback

Prints peak per axis and %/count ticks > 5 for each candidate. No write, no commit.
Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_ablcmd_probe.py
"""
import json
import numpy as np

RAW = 'results/figU_rateoff/sim_log.json'
sl = json.load(open(RAW))

print('=== raw log keys (only (N,3) arrays shown) ===')
for k in sorted(sl.keys()):
    v = sl[k]
    try:
        arr = np.asarray(v, float)
    except Exception:
        continue
    if arr.ndim == 2 and arr.shape == (1112, 3):
        print(f'  {k:24s} shape={arr.shape}')


def peak_and_over(name, arr, cap=5.0):
    """arr: (N,3) -> per-axis peak |.|, and count/pct ticks with max_axis|.|>cap."""
    a = np.abs(np.asarray(arr, float))
    pk = a.max(axis=0)
    over = (a.max(axis=1) > cap).sum()
    n = a.shape[0]
    print(f'  {name:34s} peak|.| = [{pk[0]:7.3f} {pk[1]:7.3f} {pk[2]:7.3f}]  '
          f'ticks>5: {over:4d}/{n} = {100*over/n:5.2f}%')
    return pk, over, n


print('\n=== candidate signals (per-axis peak, %ticks any-axis>5) ===')
tau_w = np.asarray(sl['tau_w'], float)
peak_and_over('logged clamped tau_w (applied)', tau_w)

for key in ('L_dot', 'H_dot_est'):
    if key in sl:
        peak_and_over(f'|{key}| (raw log)', sl[key])
        peak_and_over(f'|-{key}| (=candidate ff)', -np.asarray(sl[key], float))

# Reconstruct SS ff = -L_dot - orbital using logged arrays where available.
have = {k: (k in sl) for k in ('L_dot', 'r_com', 'v_com', 'omega_s', 'hw',
                               'struct_euler_deg', 'H_dot_est')}
print('\n=== reconstruction inputs present? ===')
for k, v in have.items():
    print(f'  {k:20s} {v}')

# ---- Full pre-clamp AOCS-command reconstruction (SS kinematic ff) ----
# tau_w = ff + K_hw*hw_err + K_theta*theta_s + K_omega*omega_s + K_d*omega_dot
# ff(SS) = -L_dot - orbital ; orbital = r_com x m*dv_com
K_hw, K_omega, K_d, K_theta = 2.0, 50.0, 25.0, 1.0
TAU_CAP = 5.0
DT = 0.01           # dt_qp
M = 71.06           # robot mass
deg2rad = np.pi / 180.0

L_dot = np.asarray(sl['L_dot'], float)
r_com = np.asarray(sl['r_com'], float)
v_com = np.asarray(sl['v_com'], float)
omega_s = np.asarray(sl['omega_s'], float)
hw = np.asarray(sl['hw'], float)
theta_s = np.asarray(sl['struct_euler_deg'], float) * deg2rad   # approx (euler vs e_R)
tau_w_log = np.asarray(sl['tau_w'], float)

# finite differences (prev = shifted; first tick prev=itself -> 0 rate)
def fd(a):
    d = np.zeros_like(a)
    d[1:] = (a[1:] - a[:-1]) / DT
    return d
dv_com = fd(v_com)
omega_dot = fd(omega_s)
orbital = np.cross(r_com, M * dv_com)

hw_err = np.clip(hw, -TAU_CAP, TAU_CAP) - hw
ff = -L_dot - orbital
pid = K_theta * theta_s + K_omega * omega_s + K_d * omega_dot
tau_cmd = ff + K_hw * hw_err + pid          # PRE-CLAMP reconstruction

print('\n=== term magnitudes (per-axis peak |.|) ===')
peak_and_over('  ff = -L_dot - orbital', ff)
peak_and_over('  K_omega*omega_s (Kw=50)', K_omega * omega_s)
peak_and_over('  K_d*omega_dot   (Kd=25)', K_d * omega_dot)
peak_and_over('  K_theta*theta_s (Kth=1)', K_theta * theta_s)
peak_and_over('  K_hw*hw_err     (Khw=2)', K_hw * hw_err)

print('\n=== PRE-CLAMP reconstruction vs logged CLAMPED ===')
peak_and_over('reconstructed tau_cmd (pre-clamp)', tau_cmd)
peak_and_over('logged clamped tau_w', tau_w_log)
clipped = np.clip(tau_cmd, -TAU_CAP, TAU_CAP)
resid = np.abs(clipped - tau_w_log)
print(f'  faithfulness: |clip(recon) - logged tau_w|  '
      f'median={np.median(resid):.3f}  max={resid.max():.3f}  Nm')
# where logged saturates, does recon exceed 5?
sat = (np.abs(tau_w_log).max(axis=1) >= TAU_CAP - 1e-3)
recon_over_at_sat = (np.abs(tau_cmd[sat]).max(axis=1) > TAU_CAP).mean() if sat.any() else 0.0
print(f'  logged-saturated ticks: {sat.sum()}/{len(sat)} = {100*sat.mean():.2f}%; '
      f'of those, recon>5: {100*recon_over_at_sat:.1f}%')
