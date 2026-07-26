#!/usr/bin/env python3
"""Kinematic-only diagnosis — κ(N_torso·J_ee) as a function of stance-arm extension.

Tests the user's hypothesis directly: the mid-swing singularity is
caused by the stance arm approaching full extension when the step
diagonal is near the arm's reach (v5: 1.0 m diagonal / 1.375 m
neutral reach = 73% extension).

Sweeps the torso position along the straight line from the initial
stance-pose (a=2, b=2) midpoint to a range of test midpoints. For
each test point:
  1. Solve IK with both tools welded at their anchors.
  2. Compute J_torso (6 x nv), J_ee_stance (6 x nv), and
     N_torso · J_ee_stance.
  3. Record κ = σ_max/σ_min of N_t·J_ee, σ_min, the stance arm's
     shoulder→tool distance (reach), and Yoshikawa manipulability.

Plots κ and σ_min vs stance-arm extension fraction. If κ blows up
only near the reach limit, the singularity IS a workspace issue;
if κ blows up in mid-workspace too, something else is going on.

No MJCF variant, no MuJoCo, no sim dynamics — just Pinocchio
kinematics. Takes <10 s.
"""
import os
import sys

_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _root)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pinocchio as pin

from crawlbot.core.robot_interface import RobotInterface
from crawlbot.core.ik import solve_ik, dock_configuration

URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')

rob = RobotInterface(URDF, gravity='zero')
model = rob.model
data = model.createData()

FID_TOOL_A = model.getFrameId('tool_a')
FID_TOOL_B = model.getFrameId('tool_b')
FID_TORSO = model.getFrameId('Link_0')       # matches sim's rs.J_torso
JID_SHOULDER_A = model.getJointId('Joint_1_a')
sl_arm_a_v = rob.arm_a_v_slice

# Anchor pair (a, b). Pick geometry consistent with start_a=2, start_b=2
# in the canonical grid: a at (-0.4, +0.3, 0.025), b at (-0.4, -0.3, 0.025).
# The stance pair during step 0's SS is unchanged: arm a stays welded at
# anchor_3a=(-0.4, 0.3, 0.025). Arm b swings from anchor_3b=(-0.4, -0.3, 0.025)
# to anchor_4b=(+0.4, -0.3, 0.025).
# We sweep b's position along that straight line to emulate the stance
# arm's workspace change as the torso follows: arm a is stance, arm b is
# "fictionally held" at intermediate points along the swing path.
anchor_a = np.array([-0.4, +0.3, 0.025])
anchor_b_start = np.array([-0.4, -0.3, 0.025])
anchor_b_end = np.array([+0.4, -0.3, 0.025])

print("=" * 78)
print("  Kinematic sweep — κ(N_torso·J_ee) vs arm extension")
print("=" * 78)
print(f"  stance A fixed at  : {anchor_a}")
print(f"  swing  B start     : {anchor_b_start}")
print(f"  swing  B end       : {anchor_b_end}")
print(f"  swing distance     : {np.linalg.norm(anchor_b_end - anchor_b_start)*1000:.0f} mm")
print()

# Sweep τ from 0 to 1 (b's position along swing line, no clearance bump
# here — we isolate the straight-line extension effect). Seed each IK
# from the previous τ's solution so the chain stays on a single branch.
taus = np.linspace(0.0, 1.0, 41)
rows = []

# Anchor the chain at τ=0 (both tools welded, symmetric start pose) using
# the project's dock_configuration utility (which itself seeds from
# neutral with torso at anchor midpoint). This should converge robustly.
a_se3 = pin.SE3(np.eye(3), anchor_a)
b0_se3 = pin.SE3(np.eye(3), anchor_b_start)
try:
    q_chain = dock_configuration(model, a_se3, b0_se3)
except RuntimeError as e:
    print(f"  [!] dock_configuration @ τ=0 failed: {e}")
    q_chain = pin.neutral(model)
    q_chain[:3] = 0.5 * (anchor_a + anchor_b_start)

for k, tau in enumerate(taus):
    b_now = anchor_b_start + tau * (anchor_b_end - anchor_b_start)
    tgts = {FID_TOOL_A: pin.SE3(np.eye(3), anchor_a),
            FID_TOOL_B: pin.SE3(np.eye(3), b_now)}

    # Chain: seed from previous solution (or τ=0 anchor if k=0).
    q_init = q_chain.copy()

    # Single-stage solve with many iterations for tight tolerance.
    q, err = solve_ik(model, q_init, tgts,
                      max_iter=2000, tol=1e-8, base_gain=0.5)
    ok = err < 1e-4

    # Keep solution for next seed if reasonable.
    if err < 1e-2:
        q_chain = q

    # Compute Jacobians at q.
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pin.computeJointJacobians(model, data, q)

    # Match sim's convention: LOCAL_WORLD_ALIGNED for both.
    J_torso = pin.getFrameJacobian(
        model, data, FID_TORSO, pin.LOCAL_WORLD_ALIGNED)
    J_ee_stance = pin.getFrameJacobian(
        model, data, FID_TOOL_A, pin.LOCAL_WORLD_ALIGNED)

    # Null-space projector of J_torso (damped).
    JJt = J_torso @ J_torso.T + 1e-6 * np.eye(J_torso.shape[0])
    J_torso_pinv = J_torso.T @ np.linalg.inv(JJt)
    N_t = np.eye(J_torso.shape[1]) - J_torso_pinv @ J_torso
    NJe = J_ee_stance @ N_t
    sig = np.linalg.svd(NJe, compute_uv=False)
    sig_min = float(sig[-1])
    cond = float(sig[0] / sig[-1]) if sig[-1] > 1e-12 else float('inf')

    # Stance-arm reach at this config: shoulder (Joint_1_a origin) → tool_a
    shoulder = data.oMi[JID_SHOULDER_A].translation.copy()
    tool = data.oMf[FID_TOOL_A].translation.copy()
    reach = float(np.linalg.norm(tool - shoulder))

    # Yoshikawa manipulability of the stance arm (at this config)
    Ja = J_ee_stance[:, sl_arm_a_v]
    w_a = float(np.sqrt(max(np.linalg.det(Ja @ Ja.T), 0.0)))

    rows.append(dict(
        tau=float(tau), b_x=float(b_now[0]), ok=ok, err=err,
        q=q.copy(), reach_m=reach, cond_NJe=cond, sig_min_NJe=sig_min,
        w_a=w_a))

# Arm's neutral/max reach for normalization.
pin.forwardKinematics(model, data, pin.neutral(model))
pin.updateFramePlacements(model, data)
shoulder0 = data.oMi[JID_SHOULDER_A].translation.copy()
tool0 = data.oMf[FID_TOOL_A].translation.copy()
reach_neutral = float(np.linalg.norm(tool0 - shoulder0))
print(f"  arm-A neutral reach: {reach_neutral*1000:.1f} mm")

print()
print(f"  {'τ':>6}  {'b_x':>8}  {'reach':>8}  {'reach/neutral':>14}  "
      f"{'w_a(Yosh)':>10}  {'κ(N·Jee)':>12}  {'σmin':>10}  ok")
for r in rows[::4]:   # every 4th sample
    print(f"  {r['tau']:6.3f}  {r['b_x']:+8.3f}  {r['reach_m']*1000:8.1f}  "
          f"{r['reach_m']/reach_neutral:14.3f}  "
          f"{r['w_a']:10.3e}  {r['cond_NJe']:12.2e}  {r['sig_min_NJe']:10.3e}  "
          f"{r['ok']}")

# Report peak κ and the extension fraction where it occurs
kappas = np.array([r['cond_NJe'] for r in rows])
reaches = np.array([r['reach_m'] for r in rows])
ext_frac = reaches / reach_neutral
k_peak = int(np.argmax(kappas))
print()
print(f"  peak κ = {kappas[k_peak]:.2e} at τ={rows[k_peak]['tau']:.3f}, "
      f"reach={reaches[k_peak]*1000:.1f} mm "
      f"(extension fraction {ext_frac[k_peak]:.3f})")

# Plot
os.makedirs('Misc/runs/M7_kinematic_sweep', exist_ok=True)
fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

ax = axes[0]
ax.plot(taus, kappas, 'C0', lw=1.8, label='κ(N_torso · J_ee_stance)')
ax.set_yscale('log')
ax.axhline(100, color='red', ls='--', lw=0.8, alpha=0.6, label='κ=100 (well conditioned)')
ax.axhline(1000, color='orange', ls='--', lw=0.8, alpha=0.6, label='κ=10³ (marginal)')
ax.axhline(1e4, color='red', ls='-.', lw=0.8, alpha=0.7, label='κ=10⁴ (near-singular)')
ax.set_ylabel('condition number')
ax.set_title('κ(N_torso·J_ee_stance) along the swing path  (stance arm A fixed at anchor_3a)')
ax.grid(True, alpha=0.3, which='both')
ax.legend(loc='upper right', fontsize=9)

ax = axes[1]
ax.plot(taus, reaches * 1000, 'C2', lw=1.8, label='stance-arm reach (shoulder→tool)')
ax.axhline(reach_neutral * 1000, color='gray', ls='--', lw=0.8, alpha=0.6,
           label=f'neutral reach ({reach_neutral*1000:.0f} mm)')
# Mark where κ peaks
ax.axvline(taus[k_peak], color='red', ls=':', lw=1, alpha=0.6,
           label=f'κ peak at τ={taus[k_peak]:.2f}')
ax.set_xlabel('τ along swing path  (b from anchor_3b to anchor_4b, straight line)')
ax.set_ylabel('stance-arm reach [mm]')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()
out = 'Misc/runs/M7_kinematic_sweep/kappa_vs_reach.png'
plt.savefig(out, dpi=130)
print(f"\n  saved: {out}")
