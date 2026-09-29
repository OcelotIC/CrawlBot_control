# PHASE R1 — the AOCS modes the paper does not use, retired

**Branch** `refactor/sim-loop-split`. **Decision** Idriss (Cowork 2026-09-29 (6)):
retire what the article does not use; the canonical must stay bit-identical.

---

## 1. What the paper uses

The paper's AOCS is `legacy_pid_numerical` — in the canonical run, in the three
Table 2 envelope-ablation arms and in the published unmanaged run (all pass
`aocs_mode='legacy_pid_numerical'` through `dca.main`). Nothing selects another
mode, and `aocs_off_in_ds` is set by none of them.

## 2. What was retired

| removed | where | why it is safe |
|---|---|---|
| `legacy`, `legacy_corrected`, `legacy_pd_numerical`, `legacy_pd_model`, `legacy_pid_model`, `H_est` branches | `AttitudeController.command` | bodies never executed (coverage: only the `elif` conditions are evaluated, all False) |
| `compute_aocs_command`, `compute_aocs_command_legacy_corrected`, `…_legacy_pd_numerical`, `…_legacy_pd_model`, `…_legacy_pid_model` | `aocs/force_estimator.py` | bodies never executed (only the `def` lines, at import) |
| `MomentumDisturbanceEstimator`, `EstimatorConfig` | `aocs/force_estimator.py`, `sim_loop.setup` | constructed, never updated: `update`, `update_analytical`, `reset` 0 lines; its only effect was the log channels `H_rO`, `H_dot_est`, identically **zero** on all 2077 canonical ticks — now written as zeros directly |
| `aocs_off_in_ds` switch | `AttitudeController.command` | never set by the canonical / Table 2 |
| `AocsHistory.tau_w_prev`, `_struct_I` | `attitude.py`, `sim_loop.setup` | read only by the `*_model` laws |
| config fields `aocs_use_H_estimator`, `aocs_use_legacy_corrected`, `aocs_filter_tau`, `aocs_K_h`, `aocs_hw_target`, `aocs_off_in_ds` | `simulation/config.py` | read only by the removed code |

`aocs_mode` stays as a field; its default becomes `'legacy_pid_numerical'`
(was `'legacy'`) and `AttitudeController` raises on any other value.

**Kept** (Table 2 or scenario-covered): `aocs_use_wrench_ff_in_ds` (the DS
wrench feedforward, used), `aocs_active_in_interstep` (`aocs_off_interstep`),
`interstep_hw_refresh` (`hw_refresh_off`), the gains `aocs_K_*`.

## 3. Proof

**Coverage** (`gate/_run/removal_proof.py`, canonical coverage of the pre-removal
tree + the Table 2 scenarios `table2_rate`, `table2_none`, `table2_u25` frozen on
`36c44bf`): no removed body line executed; the executed lines inside the removed
spans are the `def` lines (import time), the five `elif` conditions (evaluated
False), and the estimator's constructor and `H_rO` / `H_dot` getters (the zero
log channels). Table 2 executes no line the canonical does not.

**Bit-identity**: see the commit — `local_ref check`, `dock_check`, and the 22
remaining gate scenarios vs the `36c44bf` baseline.

## 4. Tests and harness

- `tests/test_aocs_physics.py`: the estimator / H_est / `*_model` tests retired
  with their code; the `pid_numerical` sign test kept; the "K_θ=0 equals PD" test
  rewritten against the written-out formula (it compared to a retired function).
- `tests/test_aocs_orbital.py`: the five orbital-feedforward tests ported to the
  kept law (`legacy_corrected`'s terms are unchanged inside `pid_numerical`, which
  reduces to them exactly with the attitude gains at zero).
- `tests/conftest.py`: the `force_estimator` fixture removed.
- **`scripts/` (the harness)**: `run_m7_single_step._make_m7_config` built its
  `SimConfig` with the removed fields and `aocs_mode='legacy_corrected'` — it
  would have raised; now `aocs_mode='legacy_pid_numerical'`. `dca.main`'s
  mode-override block, its default (`'legacy_corrected'` → `'legacy_pid_numerical'`),
  the dead non-default-mode output-dir branch and the `--aocs_mode` choices were
  reduced to the one mode. The canonical's final config is unchanged (dca
  already overrode those fields).
- `gate/scenarios.py`: the seven per-mode scenarios (`aocs_legacy`,
  `aocs_legacy_corrected`, `aocs_pd_numerical`, `aocs_pd_model`,
  `aocs_pid_model`, `aocs_H_est`, `aocs_off_in_ds`) removed with their frozen
  outputs.

## 5. Revival

`git show <R1 parent>:crawlbot/aocs/force_estimator.py` has all six laws and the
estimator; `crawlbot/control/attitude.py` at the same commit has the selection.
