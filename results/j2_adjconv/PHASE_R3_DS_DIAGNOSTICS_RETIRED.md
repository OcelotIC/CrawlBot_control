# PHASE R3 — post-abort DS diagnostics and the pre-rework DS baseline, retired

**Branch** `refactor/sim-loop-split`. **Decision** Idriss (Cowork 2026-09-29 (6)):
retire `diag_*_on_abort`, and `ds_centroidal_mode=False` / `baseline_ds_rework`
if Table 2 does not use them (it does not: `scripts/run_table2.py` sets neither,
and every arm runs `dca.main`, which sets `ds_centroidal_mode=True`). The
canonical stays bit-identical. Two commits: R3a, R3b.

---

## R3a — the three post-abort DS diagnostic flags

**What they were.** 2026-04-17 hypotheses H_DS1–H_DS3
(`POST_ABORT_DIVERGENCE.md`, `M7_DS_DIAGNOSTIC_EXPERIMENTS.md`, now in `Misc/`)
for the trailing DS entered after a `dock_timeout` abort:
`diag_force_single_contact_on_abort` (force SINGLE_A contact),
`diag_freeze_torso_ref_on_abort` (hold the torso at the current state instead of
the dock IK), `diag_disable_passivity_on_abort` (passivity off). All default
`False`; no paper run sets them.

**Retired.** The three `SimConfig` fields, the `_abort_ds` detector and the three
branches in the trailing DS of `_gait_program`; the gate scenario `abortdiag`
(with its frozen outputs and coverage).

**Proof — coverage** (`gate/_run/snippet_proof.py`, scenario coverage on the
`36c44bf` baseline; `sim_loop.py` line numbers at that commit):

| line | statement | executed by |
|---|---|---|
| L1946 | `cc_ds = ContactConfig.from_phase(SINGLE_A …)` (H_DS1) | `abortdiag` only |
| L2013 | `_pass_override = (False if … else None)` — the non-centroidal branch (H_DS3) | `abortdiag` only |
| L1937, L1945, L1975, L2010 | the detector, the H_DS1 test, `_use_state`, `if cfg.ds_centroidal_mode` | canonical + all scenarios (kept, simplified) |

H_DS2 is an operand of `_use_state` (no line of its own); with
`ds_torso_ref_from_state=True` in every paper run it could not change the value.
The rewritten trailing-DS lines compute the same values:
`_use_state` → `cfg.ds_torso_ref_from_state`, and
`_pass_override` → `True if cfg.ds_centroidal_mode else None`.

**Gates:** see the commit.

**Not retired, noted.** The `else` branch of `if cfg.ds_torso_ref_from_state`
(the dock-configuration IK hold target, L1985) is executed by **no** run — it is
the `ds_torso_ref_from_state=False` path, reachable only through
`baseline_ds_rework` or a bare `SimConfig()`. See R3b.
