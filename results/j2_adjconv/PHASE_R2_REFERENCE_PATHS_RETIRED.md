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
