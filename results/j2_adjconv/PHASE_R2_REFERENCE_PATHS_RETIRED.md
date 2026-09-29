# PHASE R2 — the torso-reference paths the paper does not use, retired

**Branch** `refactor/sim-loop-split`. **Decision** Idriss (Cowork 2026-09-29 (6)):
retire what the article does not use; the canonical stays bit-identical.
Three commits: R2a (SS paths), R2b (DS mapping), R2c (`use_m2_stack`).

---

## R2a — the non-two-task SS stack, the SS mapping bypass, F-SAT

**What the paper uses.** The two-task SS stack (T-MOM + 6-D torso pose + swing
EE + posture, weighted) in the canonical and all Table 2 arms
(`ss_two_task=True`); no run sets `mapping_bypass_in_ss`.

**Retired.**

| item | where |
|---|---|
| `ss_two_task_mode` switch (default was `False` = legacy) | `SimConfig`, `WholeBodyQPConfig`; `_two_task = not settle_mode` |
| SS use of the δ-mapping (the non-two-task SS path) | `TorsoReferenceShaper.shape`: the mapping branch is now DS-only |
| SS mapping bypass | `mapping_bypass_in_ss` + its branch |
| F-SAT, the SS-only torso-reference rate limiter | its block, `_last_r_b_ref_out`, `_sat_*`, `fsat_jitter_margin` |

**Proof.** Coverage (canonical + all kept scenarios, baseline `36c44bf`): the only
lines executed exclusively by the retired scenarios are F-SAT (9, `legacy_stack`)
and the bypass branch (3, `bypass`). The QP has **no** SS-legacy-exclusive line:
its non-two-task task blocks also run in DS (`settle_mode` ⇒ `_two_task` False),
so they stay and only the switch goes. Canonical outputs that read the retired
state: the F-SAT counters (driver's `sat_stats.txt`) are 0 on the canonical and
are kept as constant properties; `step_log.json` `delta_q` is `None` on all 5080
canonical entries. Bit-identity: see the commit (`local_ref`, `dock_check`, 20
scenarios).

**Behaviour change for non-canonical defaults.** `SimConfig()` /
`WholeBodyQPConfig()` defaulted to the legacy SS stack; the only stack is now the
two-task one. The fast suite is unaffected (no test relied on the legacy stack).

**Harness.** `dca.main(ss_two_task=...)` kept, default `True`, raises on `False`;
`--ss-two-task` kept for compatibility; the `mapping_bypass_in_ss` assignment
removed. Scenarios `bypass` and `legacy_stack` removed with their outputs.

---

## R2b — the DS δ-mapping, the post-dock DS blend, `CoMToTorsoMapping`

**What the paper uses.** In SS the raw TorsoPlanner quintic + SLERP (two-task
stack). In DS, nothing from the mapping: every DS tick the gait sequencer drives
(DWELL, inter-step settle, trailing DS) runs the QP with `settle_mode=True`,
which drops the linear torso task and keeps only the angular row.

**Retired.**

| item | where |
|---|---|
| DS δ-mapping path `r_b = (m/m_b)·r_com_ref − δ(q)/m_b`, F-RATE cache, `_last_mapping_delta*` | `TorsoReferenceShaper.shape` → `shape(tr)` |
| post-dock DS blend (Option A) | `ds_ramp_duration_s`, `_ds_ramp_*`, `_ss_entry_p_torso` |
| events that armed it | `on_ss_entry` / `on_dock` (shaper, controller, 2 call sites in `sim_loop`) |
| the mapping module | `crawlbot/core/com_to_torso_mapping.py` (256 l.), `tests/test_mapping_layer.py`, `docs/crawlbot/core/com_to_torso_mapping.md` |
| the controller's `mapping` constructor argument | `WholeBodyController.__init__` |

**Proof — different from R2a.** These lines **were** executed by the canonical
(in DS), so coverage cannot prove them dead; the proof is inertness.
(1) C2 NaN probe (worktree of `6c0bffc`): the path's linear output replaced by
NaN ⇒ canonical bit-identical, 26 scenarios identical. (2) The removal itself:
`local_ref` PASS (453 193 floats bit-exact), `dock_check` MATCH; 19/20 scenarios
identical vs the `36c44bf` baseline. `CoMToTorsoMapping` owned its own Pinocchio
`Data` (`model.createData()`), so its calls left no shared state behind — which
is why deleting them changes no control float.

**The one difference — telemetry of the retired computation (accepted, option A).**
`dwell`: `step_log.json` `delta_q` / `delta_q_current`, on all 1190 entries.
These entries are written in SS, where the mapping never ran; they carried the
δ(q) cached at the **last DS tick of the preceding DWELL** (one constant value
for the whole run, `[-9.583, -11.104, -11.126]`), i.e. a stale value. They are
now `None`, as on all 5080 canonical entries (the canonical has no DWELL). Every
other file of `dwell` — `sim_log.json` included — and its stdout (20 546 lines)
are bit-identical. Keeping this channel bit-identical would mean keeping
`CoMToTorsoMapping` alive for a telemetry field that only ever logged a stale
value. **Decision (Idriss, Cowork 2026-09-30 (10)): option A** — accept, and
re-freeze the `dwell` scenario reference (only it) at the R2b commit; the other
19 references stay frozen on `36c44bf`. The superseded `dwell` baseline is kept
locally as `scn_dwell_old_36c44bf`.

**Behaviour change for non-canonical callers.** An external `_step(settle_mode=False)`
in DS with `use_m2_stack=True` used to track the mapped linear reference; it now
tracks the planner's. No run in the repo does this.

**Revival.** `git show 4b988ec:crawlbot/core/com_to_torso_mapping.py` (and
`4b988ec:crawlbot/control/torso_reference.py` for the F-RATE / blend logic),
should an active DS with a tracked linear torso task (P3) be wanted.
