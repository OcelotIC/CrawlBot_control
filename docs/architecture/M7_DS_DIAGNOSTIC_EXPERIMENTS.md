# M7 — Three orthogonal DS-divergence experiments

**Date:** 2026-04-17
**Scope:** Diagnostic decomposition of the 45° post-abort DS divergence
**Input artefacts:**
- `docs/architecture/POST_ABORT_DIVERGENCE.md` (Step 3 analysis)
- `docs/architecture/M7_TECHNICAL_LOG.md` (post-correction)
- `results/M7_1pct_1step_v21/sim_log.json` (reference baseline — do not overwrite)

---

## 1. Intent — read this first

**This is a diagnostic decomposition task, not a fix task.** The goal is to measure how much of the 45° DS divergence is attributable to each of H_DS1, H_DS2, H_DS3 when toggled independently. No architectural change lands from this work. The actual fix is a separate design decision Idriss will make based on the results.

Specifically: **do not add an `if docked` branch anywhere.** The scheduler-level root cause is known; its fix is out of scope.

## 2. Overview — four runs, one summary table

All four runs use the v21 single-step baseline config (1% mass ratio, `start_a=2`, `start_b=2`, 1 step, all v21 fixes active: preplanner cruise-box, planned-δ, 7-DOF arms, manipulability init, task-consistent EE feedforward, `α_wrench=0.01`, `α_com_soft=0`).

The **only** difference between runs is which of three narrowly-scoped overrides is active during the trailing-DS phase **when SS aborted on `dock_timeout`**. No overrides during SS itself — SS must run identically to v21 baseline.

Each override is gated behind an explicit new flag in `SimConfig` so (a) nothing runs without opt-in, (b) every change is traceable per A4.

## 3. SimConfig additions

Add exactly three new flags, all default `False`:

```python
diag_freeze_torso_ref_on_abort: bool = False
# Diagnostic for H_DS2 (POST_ABORT_DIVERGENCE.md).
# When True, skip dock_configuration + set_hold at sim_loop.py:1365-1375
# on trailing-DS entry after dock_timeout, and freeze the TorsoPlanner
# hold target to the actual oMf_torso at the last SS sample.

diag_force_single_contact_on_abort: bool = False
# Diagnostic for H_DS1 (POST_ABORT_DIVERGENCE.md).
# When True, force cc_ds = ContactConfig.from_phase(
#   ContactPhase.SINGLE_A, r_contact_a, r_contact_b) at sim_loop.py:1343
# on trailing-DS entry after dock_timeout, matching the physical state.

diag_disable_passivity_on_abort: bool = False
# Diagnostic for H_DS3 (POST_ABORT_DIVERGENCE.md).
# When True, pass passivity_active=False to the QP during trailing DS
# entered after dock_timeout, overriding the phase=='DS' gate at
# sim_loop.py:1712.
```

No other config changes. No new QPConfig flags. No existing parameter retuning.

## 4. The four runs

### R1_baseline — reproduction
All three flags `False`. Output: `results/M7_abort_diag/R1_baseline/`.
Purpose: catch any pipeline drift or nondeterminism before interpreting deltas. Must match archived v21 on SS metrics exactly.

### R2_freeze_ref — tests H_DS2
`diag_freeze_torso_ref_on_abort=True`, other two `False`.
Output: `results/M7_abort_diag/R2_freeze_ref/`.

### R3_single_contact — tests H_DS1
`diag_force_single_contact_on_abort=True`, other two `False`.
Output: `results/M7_abort_diag/R3_single_contact/`.

### R4_no_passivity — tests H_DS3
`diag_disable_passivity_on_abort=True`, other two `False`.
Output: `results/M7_abort_diag/R4_no_passivity/`.

## 5. Invariants (must hold on every run)

- `torso_ori_peak_deg_SS` equals baseline (0.5334°) within 0.01°.
- `ee_pos_peak_SS`, `ss_end_torso_ori_deg`, `preplanner_T_steps[0]`, and the abort event (`d_mm`, `ori_deg`) match baseline bit-for-bit.
- Overrides apply to trailing DS only, never to SS or to the pre-planner.
- All 191 existing tests pass after the three flags are added (A5).

**If any invariant fails, stop and report. Do not interpret DS numbers.**

## 6. Deliverable — one table

`results/M7_abort_diag/summary.md`:

```
run              | SS peak ori [°] | q_ref jump [°] | DS entry ori [°] | DS peak ori [°] | max|τ_q|_DS [Nm] | τ_q sat frac DS | τ_w sat frac DS | dock?
R1_baseline      |                 |                |                  |                 |                  |                 |                 | no
R2_freeze_ref    |                 |                |                  |                 |                  |                 |                 | no
R3_single_contact|                 |                |                  |                 |                  |                 |                 | no
R4_no_passivity  |                 |                |                  |                 |                  |                 |                 | no
```

Save each run's full `sim_log.json` and the standard diagnostic plot set per M0.

**Do not interpret the numbers. Do not recommend a fix. Do not write "this confirms H_DS2" or similar.** Table goes to Idriss and me; we decide.

## 7. Explicit prohibitions

- Do not fix the root cause. No `if docked` branch.
- Do not touch code outside `crawlbot/simulation/config.py` and `crawlbot/simulation/sim_loop.py`.
- Do not refactor the trailing-DS setup block beyond adding the three flag-gated conditionals.
- Do not combine overrides. Orthogonality is the point.
- Do not retune any parameter. Only the three new flags are added.
- Do not proceed beyond the four runs without explicit validation from Idriss.

## 8. Validation gate

Send the summary table. Do not draw conclusions. Idriss and I will decide:

- whether the decomposition is clean (one dominant symptom → focused fix) or distributed (three contributions → scheduler-level fix required);
- whether additional combinations (R5: H_DS1+H_DS2; R6: all three) are needed;
- what the abort-DS semantics should be.
