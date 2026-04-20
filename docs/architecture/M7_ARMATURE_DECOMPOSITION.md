# M7 — Armature decomposition (Pinocchio consistency + minimum-stabilizing sweep)

**Date:** 2026-04-20
**Scope:** Test whether adding armature to Pinocchio's model closes the
EE orientation drift in A_swing, and identify the minimum armature
value that stabilizes the DS passivity settle.
**Input artefacts:**
- `results/M7_ee_ori_diag/mjcf_vs_urdf_arm_params.md` — documents the
  MJCF-URDF mismatch: 14 arm joints carry `damping=0.05 armature=0.05`
  in the MJCF, URDF/Pinocchio has 0.
- `results/M7_ee_bisection/A_swing/trace_v2.csv` — 16.7° EE ori drift
  on A_swing at current state.
- `results/M7_settle_diag/settle_damping_sweep.md` — prior sweep
  showing `a=0.05` converges, `a=0.01` plateaus; no data for
  `d=0.05, a=0` or intermediate armature-only values.

---

## 1. Intent

Two separate hypotheses, one diagnostic pass, no fix committed.

**H1.** EE orientation drift of 16.7° in A_swing is caused by
Pinocchio's mass matrix not including armature, so the QP plans `q̈`
against `H_pin` while MuJoCo integrates against `H_pin + diag(armature)`.
Adding armature to Pinocchio's model should close the drift.

**H2.** The DS passivity settle requires a minimum armature for
discrete-time stability. Prior sweep used `d=a` variants; this sweep
decouples them to identify which parameter does the stabilizing and at
what minimum value.

Both hypotheses are testable independently and the two results inform
each other: if H1 holds, we know Pinocchio needs armature installed
regardless of value; H2 tells us the value.

No interpretation in deliverables. No fix. No commit.

## 2. Part 1 — Pinocchio armature consistency

### 2.1 Change

In `crawlbot/core/robot_interface.py`, after the Pinocchio model is
loaded, set `robot.model.armature` (length `nv` array) to match the
MJCF:
- 0 on the 6-DOF floating base
- 0 on the 3 RWA spin joints
- 0.05 on each of the 14 arm joints (7 per arm × 2 arms)

Verify the mapping from Pinocchio's `v` ordering to MJCF joints before
assigning — do not assume index ranges. Print `robot.model.armature`
after assignment as a sanity check.

### 2.2 Verification

After assignment, compute `H = pin.crba(model, data, q)` at a nominal
configuration. Check that diagonal entries for arm joints are ≥ 0.05
greater than they were without armature. Report 14 diagonal entries
before and after.

### 2.3 Test run

Rerun A_swing with the modified Pinocchio model. MJCF unchanged
(current `damping=0.05 armature=0.05`). A_swing configuration as used
in the bisection:
- Standalone, no NMPC, no mapping, no AOCS.
- Torso reference constant.
- EE reference from SwingPlanner (quintic + clearance bump + SLERP).
- `swing_arm='b'`, anchor 2→3, T_step = 7.284 s.
- Release arm B weld manually via `sim._deactivate_weld('b', 2)`.

Report:
- `ee_pos_peak_SS [mm]`
- `ee_ori_peak_SS [deg]`
- `ee_ori_at_T_step [deg]`
- `tau_q_peak_SS [Nm]` (all joints)
- `tau_q_peak_ang_Nm` (wrist joints of arm B)
- time of EE ori peak (vs 6.740 s in the prior run)

Output per-tick trace to
`results/M7_armature_decomposition/A_swing_with_pin_armature/trace.csv`
with same columns as `trace_v2.csv`.

### 2.4 Reference

Prior A_swing result (no armature in Pinocchio) for comparison:
- `ee_ori_peak_SS = 16.7362°` at t = 6.740 s
- `ee_ori_at_T_step = 16.3632°`
- `ee_pos_peak_SS = 3.82 mm`
- `tau_q_peak_ang = 0.0046 Nm`

## 3. Part 2 — Minimum armature for passivity settle

Only run this after Part 1 completes.

### 3.1 Setup

MJCF arm joints: `damping = 0` throughout (isolate armature's role).
Pinocchio armature matched to the MJCF value on each variant. All
other MJCF parameters unchanged.

### 3.2 Variants

Seven settle runs, each starting from the same post-setup state
(T_start = 0):

| variant | MJCF armature | MJCF damping | Pinocchio armature |
|---|---|---|---|
| a0_d0 | 0 | 0 | 0 |
| a0p01_d0 | 0.01 | 0 | 0.01 |
| a0p02_d0 | 0.02 | 0 | 0.02 |
| a0p03_d0 | 0.03 | 0 | 0.03 |
| a0p04_d0 | 0.04 | 0 | 0.04 |
| a0p05_d0 | 0.05 | 0 | 0.05 |
| a0_d0p05 | 0 | 0.05 | 0 |

The last variant (`a0_d0p05`) is the mirror test: damping alone,
armature zero, matched on both sides. Tests whether damping alone can
stabilize without armature.

For each variant: run the inter-step DS passivity settle (same code
path as `_run_ds_passivity_loop`, up to 51 steps max, exit on
`T_kinetic < 5.224e-10`). Report per-variant:
- `T_end [J]`
- `exit_reason` (`target_met` vs `plateau`)
- `n_steps_run`
- `T_kinetic(t)` trace saved to per-variant CSV

### 3.3 Output

Consolidated `results/M7_armature_decomposition/settle_sweep_summary.md`
with one table for Part 1 result and one table for Part 2 variants.

## 4. Invariants

- MJCF restored to `damping=0.05, armature=0.05` on script exit. Verify
  by re-reading the file at end of script. Include a try/finally.
- Pinocchio armature assignment reversed on exit (no in-memory leakage
  between variants; rebuild robot model per variant if needed).
- `pytest tests/ -v` passes with the Pinocchio armature field installed.
  If any test assumes `robot.model.armature == 0`, fix the test (it was
  wrong) and report what was changed.
- 192 test baseline (last confirmed count).

## 5. Prohibitions

- No commits during the investigation.
- No changes to `wholebody_qp.py`, `centroidal_nmpc.py`, the planners,
  or `sim_loop.py`. Only `robot_interface.py` is modified for Part 1.
- No new QPConfig or SimConfig flags.
- No interpretation in the deliverables.
- If Part 1 shows `ee_ori_peak_SS` unchanged or worse, stop and report
  before running Part 2 — the hypothesis is wrong and the sweep's
  framing needs reconsideration.

## 6. Deliverables

- `results/M7_armature_decomposition/summary.md` containing both tables.
- `results/M7_armature_decomposition/A_swing_with_pin_armature/trace.csv`
- `results/M7_armature_decomposition/settle_variants/{variant}/trace.csv`
  for each of the seven variants.
- Pinocchio diagonal `H` entries before/after, in the summary.

Send the summary to Idriss. No interpretation. No fix.
