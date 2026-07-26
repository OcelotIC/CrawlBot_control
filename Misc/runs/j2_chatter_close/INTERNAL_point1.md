# INTERNAL — CC BRIEF (close the chatter fix), POINT 1: inter-config ε-robustness (STOP-GATE 1)

Branch `j2/ds-active-rework` (push, never merge, no PR). **Characterization-only** — no `crawlbot/` change
(the committed default-off Σf code from `c9df233` is untouched; C6 byte-identical preserved). Reproducer:
`scripts/point1_config_sweep.sh` (12 runs) + `scripts/point1_analyze.py` (chatter metrics from `sim_log.json`
alone — the net force Σf=λ[0:3]+λ[6:9] sign-flip is the chatter signature, no `postproc_F3F4.csv` needed).

All runs hold AOCS = **`legacy_pid_numerical`** (the run-A FIGURE config), exact-box, and vary only the
CONFIGURATION. `clean` = arm-a settle Σf-flip < 0.10 AND ‖Σf‖med < 1.0 N.

## DECISIVE OUTPUT

1. **The chatter-cleanliness ε-threshold is ROBUST across configuration.** ε=3 kills the chatter at **every**
   config (arm-a Σf-flip ≤ 0.03, ‖Σf‖ ≤ 0.2 N); ε=1 is partial/insufficient **everywhere** (arm-a flip
   0.41–0.75); ε=0 chatters everywhere. The clean threshold sits in **(1, 3] regardless of reach, anchor
   geometry, or moving-CoM** — it does **not** move above 3. ⇒ **a FIXED ε (≈3 with margin) is robust;
   adaptive-ε is NOT required for cleanliness.**
2. **NEW finding — the fix is traversal-relevant, not cosmetic (and the method-brief "benign" verdict was
   AOCS-mode-specific):** on the figure config (`legacy_pid_numerical`) the chatter **breaks docking** — ε=0/1
   abort at step 2 (dock-timeout 38–53 mm); **ε=3 rescues the full 5/5 traversal.** (The method brief saw
   "benign" only because it used `legacy_corrected`, where the chatter didn't break the dock.)
3. **NEW finding — ε=3 STARVES the run-B moving-CoM DWELL (a settle-DURATION interaction, not cleanliness):**
   ε=3's longer convergence consumes the `dt-ds=2.5 s` DS budget, so the moving-CoM DWELL (needs `>1 s`) never
   runs → the CoM doesn't translate → step-2 fails. The chatter IS clean; the traversal breaks because the
   settle ran too long. **This is the real refinement the fix needs: exit on chatter-free + dock-stable, not on
   the tight kinetic target.**
4. **Reach limit (config, not chatter):** anchor_dx ≥ 0.9 is reach-infeasible (0/5, step-0 abort, any ε). The
   max feasible pitch is dx=0.8; the **largest feasible r_com ≈ 1.84 m** (canonical step-4) is **clean at ε=3.**

---

## Per-config sweep (arm-a settles = the chatterers)

| config | reach | ε=0 | ε=1 | ε=3 | ε=3 clean? |
|---|---|---|---|---|---|
| **c0** (dx0.8, figure) | r_com→1.84 m | flip 0.99, ‖Σf‖7.0 — **abort@2** (2/5) | flip 0.75, ‖Σf‖5.6 — **abort@2** (2/5) | flip 0.02–0.03, ‖Σf‖≤0.2 — **5/5** | **YES** |
| **dx06** (dx0.6, tight geom) | r_com→1.53 m | flip 0.89, ‖Σf‖7.1 — abort@2 (2/5) | flip 0.41–0.45, ‖Σf‖2.7–3.7 — 4/5 | flip 0.01, ‖Σf‖≤0.2 — 4/5† | **YES** |
| **runB** (DWELL moving-CoM) | r_com→1.73 m | flip 0.91–1.00, ‖Σf‖7–8 — 4/5 | flip 0.73 — abort@2 (2/5) | flip 0.02, ‖Σf‖0.15 — **2/5**‡ | **YES** (clean) |
| dx09 / dx10 (wider pitch) | — | 0/5 reach-infeasible | — | 0/5 reach-infeasible | n/a |

† dx06 step-4 abort at ε=1 AND ε=3 = a **non-chatter** geometry issue at the tight 0.6 m pitch (the ε=3 settles
are all clean, flip 0.01). ‡ runB ε=3 step-2 abort = the **DWELL-starvation** below, NOT chatter (settles clean).

**Cleanliness threshold per config:** (1, 3] at c0, dx06, runB — **stable**. ε=3 is clean (flip ≤0.03) at the
largest feasible reach (1.84 m), the tightest geometry (dx 0.6), and with moving-CoM. **ε=1 never suffices.**

## The run-B DWELL starvation — mechanism (confirmed from the run logs)

The chatter-prone holding settle runs in `_run_ds_passivity_loop`; the moving-CoM DWELL runs **after** it in
`_step(settle_mode=True)` for `_dwell_target = gp.duration − dt_ds_elapsed`, **only if `_dwell_target > 1.0`**.

| run-B step-1 settle | settle duration | DWELL? | step-2 dock |
|---|---|---|---|
| ε=0 | plateau, **0.76 s** | `DWELL 1.7 s → CoM +0.200 m` ✓ | docks 4.7 mm |
| ε=3 | target_met, **2.46 s** | **none** (`_dwell_target = 0.04 s`) | **TIMEOUT 41 mm** |

ε=3 makes the settle converge to the tight target `T=2.5e-8` (2.46 s), eating the 2.5 s DS budget so the DWELL
never runs and the CoM never translates. **This is the over-damping the method brief flagged — now shown to be
not merely slow but traversal-breaking in the moving-CoM config.**

## Verdict (STOP-GATE 1) + the fixed-vs-adaptive decision

- **Fixed vs adaptive ε:** the cleanliness threshold is **config-stable** ⇒ **FIXED ε ≈ 3 (with margin) is the
  right answer; no adaptive-ε law is needed.** (POINT 2 will derive this ε from the solver tolerance rather
  than the sweep.)
- **The real refinement is the settle EXIT criterion, not the ε value.** Running the settle to the tight
  kinetic target is (a) the over-damping, and (b) the run-B DWELL starvation. The fix should **exit once
  chatter-free AND dock-stable** (e.g. Σf-flip low + `d_grip_stance` settled) and hand the remaining DS budget
  to the DWELL. This is a small, settle-local code change — but it is a DESIGN FORK that wants your call
  before I wire it (and it feeds POINT 3's bounded-residual check).

**Decision needed (STOP-GATE 1):** confirm the direction —
- (A) keep fixed ε=3 **and** add the early-exit (chatter-free + dock-stable) so the DWELL config works; or
- (B) keep fixed ε=3, treat run-B's DWELL starvation as a separate config issue (only the canonical run-A —
  no long DWELL — is the figure-regen target, where ε=3 docks 5/5 cleanly); or
- (C) other.

Default-off byte-identical (C6) is untouched (no `crawlbot/` change here). Raw run dirs reproducible, not
committed. **No merge, no PR. Reporting STOP-GATE 1, then STOP for direction.**
