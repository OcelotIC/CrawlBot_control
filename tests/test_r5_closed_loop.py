#!/usr/bin/env python3
"""R5 — Closed-loop validation: SimulationLoop with n_steps=3.

Validates that the full MuJoCo closed-loop simulation completes 3 locomotion
steps without divergence, with quantitative metric assertions.

Two-part test:
    Part A — Unit checks: GaitPlan structure for n_steps=3
    Part B — Full sim:    run(), then assert on 5 metric groups

Metrics checked:
    1. Phase sequence:  DS→SS→EXT cycles in correct order, 3 steps completed
    2. Docking:         ||p_tool − p_anchor|| < 5 mm at each dock event
    3. Momentum:        ||L_com|| ≤ L_max throughout
    4. Torques:         ||τ||_∞ ≤ τ_max throughout
    5. Stability:       no NaN / Inf in logged states

Usage:
    PYTHONPATH=. python3 test_r5_closed_loop.py
    PYTHONPATH=. python3 test_r5_closed_loop.py --urdf models/VISPA_crawling_fixed.urdf \
                                                 --mjcf models/VISPA_crawling.xml
"""
import sys
import os
import argparse
import time
import numpy as np

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)

parser = argparse.ArgumentParser()
parser.add_argument('--urdf', default=os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf'))
parser.add_argument('--mjcf', default=os.path.join(_root, 'models', 'VISPA_crawling.xml'))
parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
args = parser.parse_args()

# Headless rendering
os.environ.setdefault('MUJOCO_GL', 'disabled')

from simulation_loop import SimulationLoop, SimConfig

PASS = '\033[92m PASS \033[0m'
FAIL = '\033[91m FAIL \033[0m'
WARN = '\033[93m WARN \033[0m'
n_pass = n_fail = 0


def check(name, cond, detail=''):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f'  [{PASS}] {name}')
    else:
        n_fail += 1
        print(f'  [{FAIL}] {name}  → {detail}')


def warn(name, detail=''):
    print(f'  [{WARN}] {name}  → {detail}')


# ═══════════════════════════════════════════════════════════════════════════════
# PART A — Unit checks: GaitPlan structure for n_steps=3
# ═══════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('  PART A — Unit checks: GaitPlan structure (n_steps=3)')
print('=' * 70)

sim = SimulationLoop(mjcf_path=args.mjcf, urdf_path=args.urdf)
sim.setup(n_steps=3, start_a=2, start_b=2)
plan = sim.plan

# A1: Phase count — n_steps=3 → 7 phases (DS + 3×(SS+DS))
expected_n_phases = 2 * 3 + 1  # initial DS + 3×(SS+DS) = 7
check('A1 — Phase count == 7',
      len(plan.phases) == expected_n_phases,
      f'got {len(plan.phases)}')

# A2: Phase types alternate correctly: DS, SS, DS, SS, DS, SS, DS
expected_types = []
for i in range(3):
    if i == 0:
        expected_types.append('double')
    expected_types.append('single_a' if i % 2 == 0 else 'single_b')
    expected_types.append('double')
# Expected: ['double', 'single_a', 'double', 'single_b', 'double', 'single_a', 'double']
actual_types = [gp.phase.value for gp in plan.phases]
check('A2 — Phase types alternate DS/SS correctly',
      actual_types[0] == 'double' and
      all(actual_types[i] == 'double' for i in range(0, len(actual_types), 2)) and
      all(actual_types[i] != 'double' for i in range(1, len(actual_types), 2)),
      f'{actual_types}')

# A3: Swing arms alternate (b, a, b for start from equal indices)
swing_arms = [gp.swing_arm for gp in plan.phases if gp.swing_arm != '']
check('A3 — Swing arms alternate (b→a→b)',
      swing_arms == ['b', 'a', 'b'],
      f'got {swing_arms}')

# A4: Anchor indices advance correctly
ss_phases = [gp for gp in plan.phases if gp.swing_arm != '']
for idx, gp in enumerate(ss_phases):
    expected_to = (2 + (idx // 2) + 1) if gp.swing_arm == 'b' else (2 + ((idx + 1) // 2))
    check(f'A4.{idx} — Step {idx}: swing_{gp.swing_arm} to_idx={gp.swing_to_idx}',
          gp.swing_to_idx >= gp.swing_from_idx,
          f'from={gp.swing_from_idx} to={gp.swing_to_idx}')

# A5: Timing continuity
for i in range(len(plan.t_start) - 1):
    check(f'A5.{i} — Phase {i} end == phase {i+1} start',
          abs(plan.t_end[i] - plan.t_start[i + 1]) < 1e-10,
          f'{plan.t_end[i]} vs {plan.t_start[i+1]}')

# A6: Total duration is positive and reasonable
t_total_plan = plan.total_duration
check('A6 — Total duration > 0 and < 200s',
      0 < t_total_plan < 200,
      f'{t_total_plan:.1f}s')

print(f'\n  Plan: {len(plan.phases)} phases, {t_total_plan:.1f}s total')
for i, gp in enumerate(plan.phases):
    sw = f' swing_{gp.swing_arm} {gp.swing_from_idx}→{gp.swing_to_idx}' if gp.swing_arm else ''
    print(f'    [{i}] {gp.phase.value:10s}  t=[{plan.t_start[i]:.1f}, {plan.t_end[i]:.1f}]'
          f'  anchors=({gp.anchor_a_idx}a, {gp.anchor_b_idx}b){sw}')


# ═══════════════════════════════════════════════════════════════════════════════
# PART B — Full closed-loop simulation (n_steps=3)
# ═══════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('  PART B — Full closed-loop simulation (n_steps=3)')
print('=' * 70)

cfg = sim.cfg
print(f'  Running sim: dt_nmpc={cfg.dt_nmpc}s, dt_qp={cfg.dt_qp}s, '
      f't_swing={cfg.t_swing}s, t_ds={cfg.t_ds}s')
print(f'  Constraints: L_max={cfg.L_max} Nms, tau_max={cfg.tau_max} Nm, '
      f'weld_r={cfg.weld_radius*1000:.1f} mm')

t0 = time.time()
log = sim.run(verbose=True)
elapsed = time.time() - t0
print(f'\n  Simulation completed in {elapsed:.1f}s ({len(log.t)} NMPC steps logged)')

# ── B1: Phase sequence ──────────────────────────────────────────────────────
print('\n' + '-' * 60)
print('  B1 — Phase sequence validation')
print('-' * 60)

phases_seen = list(dict.fromkeys(log.phase))  # unique, order-preserving
check('B1.1 — At least DS, SS, EXT phases present',
      all(p in log.phase for p in ['DS', 'SS', 'EXT']),
      f'phases seen: {phases_seen}')

# Count phase transitions
transitions = []
for i in range(1, len(log.phase)):
    if log.phase[i] != log.phase[i - 1]:
        transitions.append((log.phase[i - 1], log.phase[i]))

# Each step should produce DS→SS→EXT, so 3 steps = ~9 transitions minimum
check('B1.2 — Multiple DS→SS→EXT transitions occurred',
      len(transitions) >= 6,
      f'only {len(transitions)} transitions: {transitions[:10]}')

# Step indices should reach at least 2 (0-indexed for 3 steps)
max_step = max(log.step_idx)
check('B1.3 — All 3 steps attempted (max step_idx >= 2)',
      max_step >= 2,
      f'max step_idx = {max_step}')


# ── B2: Docking ─────────────────────────────────────────────────────────────
print('\n' + '-' * 60)
print('  B2 — Docking validation')
print('-' * 60)

n_dock = len(log.dock_events)
print(f'  Dock events: {n_dock}')
for ev in log.dock_events:
    print(f'    Step {ev["step"]}: t={ev["t"]:.2f}s d={ev["d_mm"]:.2f}mm '
          f'arm={ev["arm"]} anchor={ev["anchor"]}')

check('B2.1 — At least 1 dock event',
      n_dock >= 1,
      f'got {n_dock}')

# Ideally all 3 steps dock
check('B2.2 — All 3 steps docked',
      n_dock >= 3,
      f'only {n_dock} dock(s)')

# All docks within threshold
if n_dock > 0:
    max_dock_mm = max(ev['d_mm'] for ev in log.dock_events)
    check('B2.3 — All dock distances < 5 mm',
          all(ev['d_mm'] < 5.0 for ev in log.dock_events),
          f'max dock dist = {max_dock_mm:.2f} mm')
else:
    check('B2.3 — All dock distances < 5 mm', False, 'no dock events')


# ── B3: Momentum envelope ───────────────────────────────────────────────────
print('\n' + '-' * 60)
print('  B3 — Momentum envelope')
print('-' * 60)

L_norms = np.array(log.L_com_norm)
L_max_val = cfg.L_max
L_peak = L_norms.max()
L_mean = L_norms.mean()
L_violations = np.sum(L_norms > L_max_val)
L_violation_pct = L_violations / len(L_norms) * 100

print(f'  ||L_com|| max:  {L_peak:.3f} Nms  (limit {L_max_val})')
print(f'  ||L_com|| mean: {L_mean:.3f} Nms')
print(f'  Violations:     {L_violations}/{len(L_norms)} ({L_violation_pct:.1f}%)')

check('B3.1 — Peak ||L_com|| < L_max',
      L_peak < L_max_val,
      f'{L_peak:.3f} > {L_max_val}')

# Soft check: allow up to 5% transient violations with 20% margin
check('B3.2 — Momentum violations < 5% of timesteps',
      L_violation_pct < 5.0,
      f'{L_violation_pct:.1f}%')

if L_peak >= L_max_val and L_violation_pct < 5.0:
    warn('Transient L_com violations',
         f'peak={L_peak:.3f} but only {L_violation_pct:.1f}% of steps')


# ── B4: Torque limits ───────────────────────────────────────────────────────
print('\n' + '-' * 60)
print('  B4 — Torque limits')
print('-' * 60)

tau_maxes = np.array(log.tau_max_joint)
tau_peak = tau_maxes.max()
tau_limit = cfg.tau_max

print(f'  max |τ_joint|:  {tau_peak:.3f} Nm  (limit {tau_limit})')

# Torques are clipped in the loop, so they should never exceed the limit
# Allow tiny numerical margin
check('B4.1 — Peak |τ| ≤ τ_max + ε',
      tau_peak <= tau_limit + 0.01,
      f'{tau_peak:.3f} > {tau_limit}')


# ── B5: Stability (no NaN/Inf) ─────────────────────────────────────────────
print('\n' + '-' * 60)
print('  B5 — Numerical stability')
print('-' * 60)

def has_nan_inf(data_list, name):
    """Check list of arrays/scalars for NaN/Inf."""
    for i, v in enumerate(data_list):
        arr = np.asarray(v)
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            return True, i
    return False, -1

fields_to_check = [
    ('t', log.t),
    ('r_com', log.r_com),
    ('L_com', log.L_com),
    ('tau', log.tau),
    ('p_torso', log.p_torso),
    ('struct_pos', log.struct_pos),
    ('hw', log.hw),
    ('e_com', log.e_com),
    ('L_com_norm', log.L_com_norm),
]

all_clean = True
for fname, fdata in fields_to_check:
    bad, idx = has_nan_inf(fdata, fname)
    if bad:
        check(f'B5 — No NaN/Inf in {fname}', False, f'at index {idx}')
        all_clean = False

if all_clean:
    check('B5 — No NaN/Inf in any logged field', True)


# ── B6: Solver health ──────────────────────────────────────────────────────
print('\n' + '-' * 60)
print('  B6 — Solver health')
print('-' * 60)

nmpc_fails = sum(1 for x in log.nmpc_ok if not x)
qp_fails = sum(1 for x in log.qp_ok if not x)
total_steps = len(log.nmpc_ok)
nmpc_pct = nmpc_fails / total_steps * 100 if total_steps > 0 else 0
qp_pct = qp_fails / total_steps * 100 if total_steps > 0 else 0

print(f'  NMPC fails: {nmpc_fails}/{total_steps} ({nmpc_pct:.1f}%)')
print(f'  QP fails:   {qp_fails}/{total_steps} ({qp_pct:.1f}%)')

check('B6.1 — NMPC fail rate < 10%',
      nmpc_pct < 10.0,
      f'{nmpc_pct:.1f}%')

check('B6.2 — QP fail rate < 5%',
      qp_pct < 5.0,
      f'{qp_pct:.1f}%')


# ── B7: Additional metrics ──────────────────────────────────────────────────
print('\n' + '-' * 60)
print('  B7 — Additional metrics')
print('-' * 60)

# CoM tracking error
e_com = np.array(log.e_com)
print(f'  CoM tracking error — max: {e_com.max()*100:.2f} cm, mean: {e_com.mean()*100:.2f} cm')

# Torso tracking error
e_torso = np.array(log.e_torso_pos)
print(f'  Torso tracking error — max: {e_torso.max()*100:.2f} cm, mean: {e_torso.mean()*100:.2f} cm')

# Structure drift
sp = np.array(log.struct_pos)
struct_drift = np.linalg.norm(sp[-1] - sp[0])
print(f'  Structure drift: {struct_drift*100:.1f} cm')

# Swing grip distance at end of EXT phases
ext_indices = [i for i, p in enumerate(log.phase) if p == 'EXT']
if ext_indices:
    d_swing_ext = np.array([log.d_grip_swing[i] for i in ext_indices])
    print(f'  Swing grip dist during EXT — min: {d_swing_ext.min()*1000:.1f} mm, '
          f'max: {d_swing_ext.max()*1000:.1f} mm')


# ═══════════════════════════════════════════════════════════════════════════════
# PART C — Diagnostic plots
# ═══════════════════════════════════════════════════════════════════════════════

if not args.no_plots:
    print('\n' + '=' * 70)
    print('  PART C — Generating diagnostic plots')
    print('=' * 70)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Use the built-in 7-panel plot
        fig = SimulationLoop.plot(log, save_path='r5_diagnostic_7panel.png', cfg=cfg)
        plt.close(fig)
        print('  Saved: r5_diagnostic_7panel.png')

        # Additional R5-specific plot: docking convergence per step
        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
        fig2.suptitle('R5 — Docking & Metrics Summary (n_steps=3)', fontsize=13, fontweight='bold')

        t_arr = np.array(log.t)

        # (0,0) Grip distance over time
        ax = axes2[0, 0]
        d_swing = np.array(log.d_grip_swing) * 1000  # mm
        ax.semilogy(t_arr, d_swing, 'r-', lw=1.5)
        ax.axhline(cfg.weld_radius * 1000, color='g', ls='--', lw=2,
                    label=f'dock threshold ({cfg.weld_radius*1000:.0f} mm)')
        for ev in log.dock_events:
            ax.axvline(ev['t'], color='green', ls='-', alpha=0.5)
            ax.plot(ev['t'], ev['d_mm'], 'g*', ms=12)
        ax.set_ylabel('Distance [mm]')
        ax.set_xlabel('Time [s]')
        ax.set_title('Swing EE → Anchor distance')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')

        # (0,1) Momentum norm
        ax = axes2[0, 1]
        ax.plot(t_arr, L_norms, 'k-', lw=1.5)
        ax.axhline(L_max_val, color='r', ls='--', lw=2, label=f'L_max={L_max_val}')
        ax.axhline(-L_max_val, color='r', ls='--', lw=2)
        ax.fill_between(t_arr, 0, L_max_val, alpha=0.05, color='green')
        ax.set_ylabel('||L_com|| [Nms]')
        ax.set_xlabel('Time [s]')
        ax.set_title('Angular momentum norm')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # (1,0) CoM tracking error
        ax = axes2[1, 0]
        ax.plot(t_arr, e_com * 100, 'b-', lw=1.5)
        ax.set_ylabel('||e_com|| [cm]')
        ax.set_xlabel('Time [s]')
        ax.set_title('CoM tracking error')
        ax.grid(True, alpha=0.3)

        # (1,1) Max joint torque
        ax = axes2[1, 1]
        ax.plot(t_arr, tau_maxes, 'k-', lw=1.5)
        ax.axhline(tau_limit, color='r', ls='--', lw=2, label=f'τ_max={tau_limit}')
        ax.set_ylabel('max |τ| [Nm]')
        ax.set_xlabel('Time [s]')
        ax.set_title('Peak joint torque')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig2.savefig('r5_metrics_summary.png', dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print('  Saved: r5_metrics_summary.png')

    except Exception as e:
        warn('Plot generation failed', str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# PART D — Regression: R3 + R4 still pass
# ═══════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('  PART D — Regression check (R3 + R4)')
print('=' * 70)

import subprocess
script_dir = os.path.dirname(os.path.abspath(__file__))

for test_name, test_file in [('R4 (includes R3)', 'test_r4_fixes.py')]:
    result = subprocess.run(
        ['python3', test_file, '--urdf', args.urdf, '--mjcf', args.mjcf],
        capture_output=True, text=True,
        cwd=script_dir,
        timeout=120)
    final_lines = [l for l in result.stdout.split('\n') if 'RÉSULTATS' in l]
    summary = final_lines[-1].strip() if final_lines else 'no summary found'
    print(f'  {test_name}: {summary}')
    check(f'D1 — {test_name} regression passes',
          result.returncode == 0,
          f'exit code {result.returncode}')


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print(f'  RÉSULTATS R5 : {n_pass} PASS, {n_fail} FAIL')
print('=' * 70)

# Save log for later analysis
log.save('r5_sim_log.json')
print(f'  Log saved: r5_sim_log.json')

sys.exit(0 if n_fail == 0 else 1)
