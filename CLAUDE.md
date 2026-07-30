# CLAUDE.md — Project Instructions for Claude Code

**This file is read automatically at session start. Follow it.**

---

## Authoritative Documents

Two documents govern all work on this project. They take precedence over any prior context, session history, or assumptions.

1. **Architecture specification:**
   `docs/architecture/brainstorming_reworked_architecture.md`
   — Full mathematical derivation, control architecture, frame conventions, and contribution positioning. This is the ground-truth for what the controller should do.

2. **Implementation plan:**
   `docs/architecture/CLAUDE_CODE_HANDOFF.md`
   — File-level milestone plan, diagnostic suite spec, anti-patterns, environment setup, and pass/fail criteria. This is the ground-truth for how to implement it.

**When in doubt, read these documents. Do not guess. Do not rely on memory from previous sessions.**

---

## Session Startup (MANDATORY)

Every session begins with:

```bash
# 1. Environment setup
bash docs/architecture/setup_env.sh

# 2. Verify
PYTHONPATH=. MUJOCO_GL=disabled python3 -c "import pinocchio; import mujoco; import casadi; print('OK')"

# 3. Run the test suite — GATED (CLEANUP-29). --fast deselects @pytest.mark.slow
PYTHONPATH=. python3 gate/run_suite.py --fast
#    NOT `pytest` directly, and NOT MUJOCO_GL=osmesa: under osmesa this
#    container aborts pytest COLLECTION (PyOpenGL on test_diagnostics), which
#    looks identical to a broken suite. run_suite.py forces MUJOCO_GL=disabled.

# 4. Confirm the documentation and this file's parameter table match the code
PYTHONPATH=. python3 gate/sync_docs.py --check
PYTHONPATH=. python3 gate/verify_params.py
```

Do not skip these steps. Do not start coding before the environment is verified.

---

## Rules (from HANDOFF §0, anti-patterns A1–A8)

1. **Ground-truth is the code, not the paper.** Read the file before editing it. Always `view` before `str_replace`.
2. **Milestone-by-milestone.** Do not proceed to M(n+1) until M(n) passes and Idriss validates.
3. **Every simulation produces diagnostics.** Call `run_diagnostics()` at the end of every sim. "It docked" is not a pass criterion.
4. **No copy-paste model files.** One canonical MJCF. Parametric variations are applied programmatically.
5. **No silent parameter changes.** All tunable parameters live in `SimConfig` with units and justification.
6. **No patching without diagnosis.** Before fixing a bug: state root cause, reference the spec section, predict the quantitative effect. Then fix, run diagnostics, verify.
7. **No regression.** After modifying any core module, re-run `pytest tests/ -v`. Broken tests must be fixed before proceeding.
8. **Show data, not explanations.** When a simulation fails, show the diagnostic plot and point to the problem. Do not write paragraphs rationalizing the result.
9. **Write scripts to disk first**, then run them. No inline heredoc execution.
10. **At-weld dock metric ONLY.** Dock precision is `dock_events` d_mm (distance when the weld fires). Min-over-swing is a fly-by artifact (step-2 lesson: 3.0 mm fly-by vs 4.89 mm at-weld) — never report it as the dock.
11. **References are never FSM-coupled, and exported reference curves must be CONTINUOUS.** A logged reference that jumps at a phase transition is either a control bug or an export artifact — find out which before plotting (torso-export fix `b619ef4`/`b37b528`: logging-only, control byte-identical, proven by full re-run diff).
12. **One variable at a time for weight changes.** The COPRIORITY campaign asserted the wrong lever four times (torso, momentum ×2, hw-slack) before single-variable elimination found `alpha_torque`. Test each weight; never attribute by plausibility.
13. **Every delivery states: commit hash + artifact path + key numbers.** A result without its commit and its JSON/CSV path is not reproducible and does not count.
14. **Torque-min ≳ 5× the accel-reg floor** (feasibility gate). At torque:floor = 1:1 the SS redundancy resolution degrades to a step-0 dock timeout (PHASE_COPRIORITY_1000 Addendum 5). When raising the regularizer floor for conditioning, raise `alpha_torque` with it.

15. **Every change to `crawlbot/` updates its document, in the same commit.** `docs/crawlbot/<pkg>/<module>.md` is the reference for anyone reading this repo cold. The measured half (header, API table, code map) is regenerated — never hand-edited — by `gate/sync_docs.py`; the prose half (maths, design rationale, traps) is yours to update. **This is enforced, not requested:** `gate/sync_docs.py --check` exits non-zero when a symbol was added, removed or moved without the document following, and it runs as part of the pre-commit routine below. A doc that lags the code is how `docs/api/` died — it ended up describing a `dynamics` module that does not exist.
---

## Commands

```bash
# Run the tests — via the gate, which forces MUJOCO_GL=disabled and applies the
# pass criterion (0 failed, 0 errors, 0 XPASS). Do not call pytest directly.
PYTHONPATH=. python3 gate/run_suite.py --fast    # ~23 s
PYTHONPATH=. python3 gate/run_suite.py           # ~89 s, before merging

# Run a simulation script
MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/<script>.py

# Run diagnostics on a simulation log
MUJOCO_GL=disabled PYTHONPATH=. python3 -c "
from crawlbot.diagnostics import run_diagnostics
import json
log = json.load(open('results/<log>.json'))
run_diagnostics(log, 'results/<output_dir>/')
"

# pip install
pip install <package> --break-system-packages
```

### Mandatory routine after ANY change to `crawlbot/` (Rule 15)

```bash
# 1. regenerate the measured half of the docs (header, API table, code map)
PYTHONPATH=. python3 gate/sync_docs.py

# 2. update the prose half by hand where the behaviour or the maths changed
#    docs/crawlbot/<pkg>/<module>.md

# 3. verify — all three must pass before committing
PYTHONPATH=. python3 gate/sync_docs.py --check    # docs match the code
PYTHONPATH=. python3 gate/verify_docs.py          # every file:line + symbol resolves
PYTHONPATH=. python3 gate/link_audit.py           # no path citation broken
PYTHONPATH=. python3 gate/verify_params.py        # THIS table's file:line + values

# 4. and the canonical invariant, as always
MUJOCO_GL=disabled PYTHONPATH=. python3 gate/run_gate.py
MUJOCO_GL=disabled PYTHONPATH=. python3 gate/dock_check.py results/gate_run_scratch/sim_log.json

# 5. the component suite — the half run_gate.py structurally cannot do
PYTHONPATH=. python3 gate/run_suite.py --fast    # per-commit  (~25 s)
PYTHONPATH=. python3 gate/run_suite.py           # BEFORE MERGING (~90 s)
```

**Both gates, not either.** `run_gate.py` proves the canonical run still
reproduces byte-identically — it is indifferent to whether the behaviour is
*correct*, because a wrong controller reproduces just as faithfully and
byte-identity then locks the bug in as the new baseline. `run_suite.py` is the
only thing that can say a **changed** configuration is still physically valid
(dynamics residual, passivity, CoM-Jacobian mass form, J̇ assembly, quaternion
conventions). One protects the past, the other protects a change.

A test marked `xfail(strict=True)` that starts passing **fails** the suite gate
on purpose. Do not delete the marker to make it green — find out what changed,
and write it down.

If a removal or rename changed line numbers, step 1 fixes every `file:line`
link automatically — that is why the links are generated rather than typed.

**Coverage annotations** (`canonical? = yes / not exercised`) come from
`gate/_run/cov/cov.json`, which stores EXECUTED LINE NUMBERS. So it goes stale
in two ways, and the second is worse: (a) changing which paths execute makes
the column describe the previous architecture; (b) **changing line numbers at
all — even a pure comment insertion — silently mis-attributes coverage to the
wrong symbols.** Case (b) was observed on this branch: after a field was added
to `centroidal_nmpc.py`, `sync_docs` annotated `get_shifted_fallback` and
`get_full_trajectory` as **`yes`** when neither executes at all. `--check`
does NOT catch this. Regenerate cov.json whenever a `crawlbot/` file's line
numbering moves, not only when behaviour changes. Regenerate it
with `bash gate/_run/cov_replay.sh` before step 1, otherwise the column reflects
the previous architecture.

---

## Current Milestone

Update this line as work progresses:

> **For current-state questions, `docs/architecture/STACK_OVERVIEW.md` is the code-ground-truth reference.** This milestone block is a short pointer, kept in sync with it.

**→ Active:** **repo-hygiene chantier** on branch `cleanup-nmpc` (CLEANUP-0 … CLEANUP-30) — 43 commits ahead of `main`, 0 behind. Reports: `results/j2_adjconv/PHASE_CLEANUP_*.md`, ledger `CLEANUP_CARRYOVER.md`, overview `PHASE_CLEANUP_OVERVIEW.md`.

**The science is already on `main`:** **FROZEN 2.5 CANONICAL** — τ_w,max = 2.5 N·m (controller + plant) + Add-5 WQP weights, frozen at `32aefaf`, default-cap alignment `ec41cd9`, torso-export continuity fix `b619ef4`/`b37b528`. Docks **6/6** (worst at-weld 4.99 mm / margin 0.01 mm). **PR #29 merged 2026-07-14**; no PR is open. See "Canonical Results" below.

**Two gates, and both must pass — they check different things:**
`gate/run_gate.py` proves the canonical run reproduces **byte-identically** (2077 rows × 132 928 fields). It is indifferent to correctness: a wrong controller reproduces just as faithfully, and byte-identity then locks the bug in as the new baseline. **It protects the past.** `gate/run_suite.py` is the only thing that can say a **changed** configuration is still physically valid — dynamics residual, passivity, CoM-Jacobian mass form, J̇ assembly, quaternion conventions. **It protects a change.**

**Completed (freeze campaign):** weight-tuning isolation (COPRIORITY Addenda 1–8 → dock-lever hierarchy: torque ≥ 5× floor gate, EE ≥ 1000 gross reach, torso = fine lever, momentum/hw-slack inert); NMPC-PLAN-SATURATION + U-PLAN-CHECK; CANONICAL-2p5 freeze with plant-cap proof; TORSO-REF-AUDIT + export continuity fix.

**Completed (cleanup chantier):** `crawlbot/` −3281/+644 across 19 files with the canonical byte-identical at every stage; 2098 files of research sediment moved to `Misc/`; 33 per-module documents under `docs/crawlbot/` with a generated-and-enforced half (`gate/sync_docs.py --check`); seven checkers in `gate/` (run_gate, run_suite, dock_check, sync_docs, verify_docs, verify_params, verify_roots, link_audit), each proven to bite on an injected fault before being trusted; the test suite taken from 12 problems to 0 and **gated**.

**Two-task SS stack (current architecture):** T-MOM linear + 6-D torso-pose + swing-EE + posture, all weighted, NO null-space projection, `weight_ratio=1` ⇒ **α magnitudes ARE the hierarchy** (nominal priority integers inert). In two-task SS the torso task is fed the **raw TorsoPlanner quintic+SLERP — the CoM→torso δ-mapping is NOT used in SS** (`sim_loop.py:2573-2576`); DS still uses the mapping. Superseded: the cooperative split, strict-P1, planned-δ, and the handoff-era "torso-ori blocker".

---

## Key Parameters (single source of truth: SimConfig)

**FROZEN 2.5 CANONICAL** (freeze commit `32aefaf`; default-cap alignment `ec41cd9`). Verify against the cited file:lines, not this table, when in doubt.

| Parameter | Value | Unit | Reference |
|-----------|-------|------|-----------|
| Robot mass | ~71 | kg | spec §0.4 |
| Arm DOFs | 7 per arm (14 total) | — | spec §4.9, 7-DOF upgrade |
| nq / nv / nu | 21 / 20 / 14 (Pinocchio) | — | 7-DOF model |
| nq / nv / nu | 31 / 29 / 17 (MuJoCo+RWA) | — | 7-DOF + 3 wheels |
| Free DOFs in SS | 14 | — | 20 - 6 weld |
| hw_max | ±5 (**unchanged by design**) | Nms | `config.py:83-84`, spec §4.6 |
| **tau_w_max** | **2.5** — enforced 3×: controller `config.py:80`, AOCS clip `config.py:84`, **plant** MJCF wheel `ctrlrange ±2.5` (`VISPA_crawling_rwa3.xml:324-326`) | Nm | freeze `32aefaf` (was 5); dca.main default aligned `ec41cd9` |
| tau_max | 20 | Nm | `config.py:45` |
| nmpc_period | 0.1 | s | `config.py:37` |
| dt_qp | 0.01 | s | `config.py:38` |
| NMPC horizon N | **20** (8 → 15 → 20; 2.0 s lookahead). **Now a clean knob** — F1 decoupled the reference sampling from N. ⚠ T_step is solved per step (**2.775…7.900 s** measured, NOT the 6 s default), so the horizon is **72 % of the shortest step**, margin 0.775 s; it would exceed it at **N ≥ 28**, where the constant-contact assumption breaks | — | `config.py:285`, spec §5.1 |
| **nmpc_per_stage_refs** | **True** — NLP carries N+1 reference blocks ⇒ trajectory tracker, not setpoint regulator (NMPC_AUDIT F1) | — | `crawlbot/simulation/config.py` `nmpc_per_stage_refs` |
| NMPC state dim | 9 | — | spec §5.1 (B2) |
| NMPC control dim | 12 | — | spec §5.1 |
| weight_ratio | 1.0 — **α magnitudes ARE the hierarchy** (two-task weighted stack, no null-space projection; priority integers inert) | — | `wholebody_qp.py:153` |
| **α torso-pose** | **2000** | — | `config.py:362` (Add-5) |
| **α swing-EE** | **1000** (dock lever; needs ≥ ~1000) | — | `config.py:341` (Add-5) |
| **α momentum (T-MOM)** | **400** (near-inert on Ḣ_s — NMPC owns the envelope) | — | `config.py:349` (Add-5) |
| **w hw-slack** | **800** (slacks active only if the hw box is violated) | — | `wholebody_qp.py:218` (Add-5) |
| **α posture** | **20** | — | `config.py:342` |
| **α torque-min** | **5** (must stay ≳ 5× accel-reg floor — Rule 14) | — | `sim_loop.py:973` (QP-construction literal) |
| **α wrench-track** | **1.0** | — | `config.py:343` (Add-5; was 0.01 pre-freeze) |
| **α accel-reg** | **1.0** (regularizer floor) | — | `sim_loop.py:973` |
| ε (Tikhonov) | 1e-6 (inert: λ_min(H_LS)=1 ≫ ε) | — | `hierarchical_qp.py:98` default |
| **κ_SS(H)** | ≈ 7.5e3 (530× below the pre-freeze canonical 3.6e6) | — | `canonical2p5_result.json` |
| ~~α_com_soft~~ | **field REMOVED** (CLEANUP-6), not merely 0 | — | The soft-CoM residual task is gone; the QP has no direct CoM feedback path. Do not re-add a config field for it without re-adding the task |
| CoM shaping | a_cruise_max=**0.0** (off) | m/s² | `coarse_preplanner.py:99` — pre-planner cruise-accel cap disabled |
| Torso reference (SS) | **raw TorsoPlanner quintic+SLERP — NO δ-mapping in two-task SS** (`sim_loop.py:2573-2576`); DS still uses δ(q_current)+F-SAT | — | TORSO-REF-AUDIT; per-step reference re-anchored each SS |
| CoM-z standoff | −0.35 m (on) | m | Dock-IK + init pin crawl height (PR #17) |

---

## Canonical Results (frozen 2.5 canonical — source of truth: `results/j2_adjconv/canonical2p5_result.json` + `c25_fulldiag.csv` / `u25_fulldiag.csv`)

6-step traversal, C = managed (τ_w,max 2.5 everywhere) vs U = management OFF (NMPC envelope + WQP box + AOCS clip lifted; **plant cap ±2.5 still active**):

| metric | C (managed) | U (unmanaged) |
|---|---|---|
| at-weld docks [mm] | **6/6** — 4.02 / 4.89 / **4.99** / 4.97 / 4.95 / 4.62 (worst margin **0.01**) | 6/6 |
| planned Ḣ_s per-axis peak | capped at **2.500 on ALL six steps** (y and z pin; 58 % of SS ticks at cap) | up to **10.88** (4.4× envelope) |
| realized Ḣ_s per-axis peak | **2.500** (WQP box ACTIVE on the realized wrench at the peak step) | 3.451 |
| θ_s peak | **0.540°** | 1.194° (2.2×) |
| h_w peak (axis / norm) | 4.10 / 4.24 | 4.55 / 5.08 |
| e_com peak | 0.154 m | 0.150 m |
| applied τ_w (measured, `actuator_force`) | ≤ 2.500 (demand ≤ 2.5 by management) | ≤ 2.500 while the controller demands up to **26.9** — the actuator saturates |
| κ_SS / qp_fail | 7.48e3 / 0 | 7.71e3 / 0 |

The **5 mm dock gate is the docking-mechanism capture radius** — the 0.01 mm worst-case margin is the accepted operating point at the tightened budget. Torso tracking: boundary residual **18–27 mm** steady-state (98.6 mm on the initial step); the mid-swing excursion (~150 mm) is real free-floating recoil against the momentum envelope, not a tracking bug (TORSO-REF-AUDIT). Exported torso reference is continuous across SS→DS→SS (terminal-hold logging fix, control byte-identical).

---

## Known Issues

- **⚠ BRANCH `claude/com-gain-semantics-audit-j0u6yr` DIVERGES FROM THE FROZEN CANONICAL BY DESIGN.** COM-GAIN-AUDIT Phase 3 fixed a rank-one CoM feedback defect in the QP (`as_gain_matrix`, `wholebody_qp.py:70-126`), so **`gate/run_gate.py` FAILS artifact identity** (row 1, `hw_x_Nms`) on this branch. This is the intended A/B, not a regression. Docks stay **6/6** under 5 mm (worst margin 0.01 → **0.02 mm**, i.e. better); θ_s 0.540 → 0.539°; **e_com peak unchanged at 0.154 m**. `gate/run_suite.py` PASSES. **Whether to re-freeze the canonical artifacts at the corrected law is an open decision** — see `results/j2_adjconv/COM_GAIN_PHASE3_FIX.md` §6. Do not "fix" the gate by reverting the controller, and do not re-freeze without Idriss's call.
- **RENAMED: `dt_nmpc` → `nmpc_period`, `nmpc_dt` → `nmpc_pred_dt`.** The old pair were near-anagrams for genuinely different quantities (control period vs prediction step), which is *how* F3 happened: code using the wrong one read perfectly AND computed the right answer, because the two values were always equal. The F3 arithmetic fix repaired the instances; the rename removes the class. Pure rename, **proven inert** — canonical byte-identical, 0 differing fields across 125 888 physics fields. `tests/test_nmpc_qp_consistency.py::TestTimescaleNaming` fails if either old name returns. **`dt_qp` keeps its name** (nothing collides with it), so the timescales now read: `dt_qp` 0.01 s inner loop, `nmpc_period` 0.1 s outer loop, `nmpc_pred_dt` 0.1 s NLP knot spacing.
- **NMPC F3 FIXED — `nmpc_pred_dt` (prediction step) and `nmpc_period` (control period) are now genuinely independent.** They are different quantities (standard MPC), but three paths advanced the plan by ONE prediction knot per CONTROL period: the QP plan interpolation (`sim_loop.py`), `get_shifted_fallback`, and `shift_warm_start`. Correct only while both were 0.1 — which they always had been, so nothing caught it; setting `nmpc_pred_dt=0.05` silently dilated the QP's CoM reference **2×**. Fixed: the interpolation indexes the plan by elapsed time in **integer** knot units (`u = qs / _qp_per_knot`; the algebraically-equal `(qs·dt_qp)/nmpc_pred_dt` is 1 ULP off and breaks bit-identity), and the two shifts advance `round(control_period/dt)` knots via the new `CentroidalNMPCConfig.control_period`. A non-integer `nmpc_pred_dt/dt_qp` is now **rejected** rather than allowed to drift. Proven inert: **0 differing fields across 125 888 physics fields** vs the pre-F3 run at the same config (only `qp_time_ms`/`nmpc_time_ms` differ — the two columns `run_gate.py:66` already excludes). Report: `results/j2_adjconv/NMPC_F3_TIMING.md`.
- **⚠ Compare runs with the gate's exclusion list, not `cmp`.** `qp_time_ms` and `nmpc_time_ms` are wall-clock and nondeterministic; `gate/run_gate.py:66` excludes them. A plain byte-diff of two fulldiag CSVs reports ~1278 differing lines for a physically identical pair.
- **With F3 fixed, `nmpc_pred_dt=0.05` at 10 Hz is WORSE, not better**: θ_s **0.494 → 0.774°** once the dilation is removed (docks still 6/6). The pre-fix result was flattering that configuration. Keep `nmpc_pred_dt=0.1`, `N=20`. (Not a single-variable isolation — the pre-fix run also predates F1.)
- **NMPC F1 FIXED + horizon N=20** (`config.py:285`, `265`). The NLP used to carry ONE reference block shared by every stage, so the reference was a constant **setpoint** and `sim_loop` had to sample it in the future — which made `nmpc_N` two knobs. `nmpc_per_stage_refs=True` gives N+1 blocks, one per knot, so the NMPC **tracks the pre-planner's trajectory** and N is now a clean horizon knob. Refactor proven inert first (broadcast reference ⇒ Δcost = Δsolution = **0.000e+00**, `scripts/audit_nmpc_f1_equivalence.py`; `F1off_N15` reproduces the committed N=15 exactly). Rule-12 ladder: θ_s **0.554 → 0.511 → 0.455°** (vs frozen 0.540, **−16 %**), h_w peak norm 4.172 → 4.087 Nms, docks **6/6** (worst margin 0.05 mm), qp_fail 0, and at N=20 **639/639 solves fully converged** (zero acceptable-tol exits), max 57.1 ms against the 100 ms period. ⚠ `e_com` is **not comparable across the flag** — the exported `r_com_ref` is the horizon-end setpoint under F1-off and the current-time reference under F1-on. Report: `results/j2_adjconv/NMPC_F1_PER_STAGE_REFS.md`.
- **⚠ NMPC solve times are comparable only WITHIN one session.** Same configuration, different sessions: median 30.05 vs 38.36 ms; maxima span 67–95 ms for configurations whose medians differ by 1 ms. Always run an in-session control before attributing a solve-time delta to a code change — an F2 draft claimed "+25 % cost" that way, and the in-session control showed the box is if anything marginally *faster*.
- **RETRACTED: the N=15 "real-time violation" was machine contention.** `NMPC_HORIZON_N15.md` §3 recorded max 117.9 ms and 1 solve over the 100 ms period. Re-running that exact configuration gives **max 43.6 ms, 0 over budget** with byte-identical trajectory metrics. No configuration measured exceeds its control period on a quiet machine. Treat solve-time outliers on this container as noise; only medians and trends are evidence.
- **⚠ `docs/crawlbot/solvers/centroidal_nmpc.md` §3 asserted the OPPOSITE of the code** — it claimed the canonical sets `enforce_hw_conservation=True` with `ng_path=17`/`ng_term=6`, presented as "measured on the real run". Measured now: **False**, `ng_path=11`, `ng_term=0`. Corrected in place. A stale doc that states a measurement is worse than one that states nothing.
- **The QP CoM task differences the NMPC plan, not the tracking error** — `sim_loop.py:2549` feeds it `rp_interp` (NMPC plan, re-anchored to the measured CoM every 100 ms ⇒ `|e_r|` ≤ 1.147 mm, identically 0 on 508/5080 SS ticks), while the *logged* reference `cref_r` goes to the NMPC (`:2243`, exported `:2969`) and carries **61.5 mm median / 153.7 mm max** SS error. So no CoM gain change in the QP can improve `e_com`; **81 %** of that error is anisotropic (⊥[1,1,1]), x-dominated. Structural, not yet addressed.
- **`gate/_run/cov_replay.sh` does not exist** although the Rule-15 routine below prescribes it — `gate/_run/` is gitignored in full (`.gitignore:109`). Without `gate/_run/cov/cov.json`, `sync_docs.py --check` false-positives on **all 25** modules and a blanket `sync_docs.py` **erases** the committed coverage annotations. Regenerate with `pip install coverage --break-system-packages` then `MUJOCO_GL=disabled PYTHONPATH=. python3 -m coverage run --source=crawlbot --data-file=gate/_run/cov/.coverage gate/replay_canonical.py && python3 -m coverage json --data-file=gate/_run/cov/.coverage -o gate/_run/cov/cov.json` (~5 min). `coverage` is **not** in `setup_env.sh`.
- **Suite state: 205 passed, 0 failed, 0 errors, 0 skipped, 1 xfail — and GATED** (`gate/run_suite.py`; 206 tests incl. the 6 new `TestGainSemantics`). The "210 passed" figure previously recorded here was stale: the pre-Phase-3 tree had **200** tests (CLEANUP-29). The one xfail, `test_coarse_preplanner::test_far_infeasible_under_tight_rate`, is `strict=True` with its reasoning in the marker: the envelope-semantics question at cap 2.5 is open (see Remaining Work), and if the far case ever goes infeasible again the test **fails** rather than turning green. Getting from 12 problems to 0 retired nothing: 6 tests ported to the two-task API (`PHASE_CLEANUP_28`), 7 repaired (`PHASE_CLEANUP_29` §1), 2 genuinely-dead retired, 1 marked honestly.
- **`MUJOCO_GL=osmesa` aborts pytest collection** in the current container: `tests/test_diagnostics.py` → `PyOpenGL AttributeError: 'NoneType' object has no attribute 'glGetError'`. Environment, not code. `gate/run_suite.py` forces `MUJOCO_GL=disabled` for this reason — use it rather than calling `pytest` directly.
- **The orphaned manipulability-IK path is RETIRED (CLEANUP-30).** `manipulability_config_trajectory`, `manipulability_config_mid_waypoint`, `check_path_feasibility`, `precompute_torso_map` + 4 stranded helpers — 695 lines, 47 % of `ik.py` — had zero callers and 0 lines executed by the canonical replay. Removal proven inert: artifact identity byte-exact, all six docks delta +0.0000. `sim_loop` uses **`manipulability_config`** (IK 2, `sim_loop.py:307`), a different and live function. Suite 743 s → **89 s**. Reasoning and revival path: `PHASE_CLEANUP_30_IK_OPTION_B_RETIRED.md`; §7–§9 of `IK_FORMULATION.md` still derive it under a RETIRED banner. **Consequence:** there is now no interior path-feasibility guard at all — `check_path_feasibility` was the only one and was already disconnected.
- **`crawlbot/diagnostics/` is KEPT** — live consumer (`scripts/run_m7_single_step.py`) and mandated by Rule 3. The real defect is that the **canonical run does not honour Rule 3**: the gate exports via `scripts/diag_full_diag_export.py`, never `run_diagnostics()` (`CARRYOVER` §A6).
- **`gate/link_audit.py` cannot see computed paths.** It audits citations in prose; the CLEANUP-21 miss that disabled 7 tests for six passes was `os.path.join(_root, 'diagnostic', ...)` in Python. `tests/fixtures/` is now the convention for test data.
- **CoM-reference export snaps to the measured CoM at SS→DS entry** — logging convention (`_log_ds_tick` logs e_com=0 with ref:=measured, `sim_loop.py:1020-1023`); reviewer-reported magnitude ~76 mm (not repo-verified — the fulldiag CSV has no CoM-ref channel). Decision pending whether to apply the same terminal-hold fix as the torso export.
- **Fig-3 conservation quantity ‖L_total‖ is NOT in the fulldiag export** (verified: no `Ltot` column in `c25_fulldiag.csv`; it exists in the `export_figure_data.py` traversal CSVs). Dedicated export pending.
- `dca.main` sets `cfg.ds_centroidal_mode=True` (`diag_cooperative_arms.py:347`) — the locked config runs centroidal DS everywhere including the trailing settle; flags keyed on it cannot discriminate the DWELL (see TORSO-REF-EXPORT-FIX).

---

## Remaining Work

**Repo-side:**
- Fig-3 export: add the ‖L_total‖ / conservation channel to the fulldiag exporter.
- Decide/apply the CoM-reference export terminal-hold (same class as the torso fix) if Idriss rules the SS→DS snap non-publishable.
- Re-examine `test_far_infeasible_under_tight_rate` semantics at cap 2.5. Now `xfail(strict=True)` so it cannot be forgotten: if the far case ever goes infeasible again the suite gate **fails** rather than turning green.
- `CARRYOVER` §A6 — make the canonical run honour Rule 3 (it exports via `scripts/diag_full_diag_export.py`, never `run_diagnostics()`). Do it in a pass of its own, where the gate is the only thing that changes.
- `CARRYOVER` §A7 — `solve_ik_waypoints`, 118 lines with zero callers; same class as the four retired in CLEANUP-30.
- `planning/sequence_loader.py` — 107 statements of documented user-facing feature (`sim.setup(sequence_path=…)`) verified by **neither** gate. The largest genuinely-unverified block left.
- Delete `Misc/` when its ~120 inbound citations have been swept (it is 2098 of the branch's 2235 changed files, all moves).
- **This file's prose `file.py:NNN` citations are checked by nothing.** `verify_params.py` covers only the parameter table; `link_audit.py` checks that a *path* resolves, not that a *line number* still says what the text claims. Three refs had silently drifted by the time CLEANUP-30 measured them (`sim_loop.py:2581→2573`, `1038→1020`, `dca:352→347`). Extending `verify_params.py` to the prose refs is the fix.

**Paper-side (Overleaf — external to this repo; state per Idriss's review notes, NOT repo-verifiable):**
- Propagate θ_s = 0.54° (new canonical) everywhere the paper states 0.59°/old values (abstract, §VII-C, Fig. 5 caption).
- §V-C rewrite: drop the never-done "h_max sized within the singularity-free sub-region" promise; adopted framing = demonstration budget (power-constrained, small-CMG class; Lappas/Leve/Gurrisi refs already in refs_master.bib).
- Annotation #91 (RF): resolved by the small-CMG reference (4-unit MiniCMG-class pyramid ≈ 9 N·m·s envelope covers the ‖h_w‖∞ worst case) — integrate.
- RF Q3: quaternion/twist notation consistency pass (flagged 6×).
- Swap `master.tex` → `sec_III_IV_v3` (§V momentum-conservation rewrite, compiled clean, awaiting activation).
- Broken artifacts: Fig ??, "Draft for review" footnote, ablation placeholder.
- §II differentiation table vs Lutze[2023]/Rognant[2025] (sourced).
- Torso narrative framing decision (per-step regulator; boundary residual 18–27 mm steady-state, 98.6 mm initial step — repo-verified from `c25_fulldiag.csv`; mid-swing recoil = free-floating momentum conservation, shown in Fig 6).
- Submission housekeeping: `ieeeaccess.cls` swap, IEEE AI declaration, final figure set (done for Figs 1, 2, 4, 5, 6, 7 + planned-ablation from the continuous CSVs; Fig 3 pending its export).

**Separate track (do NOT mix into the journal-paper TODO):** IROS 2026 Space Robotics Workshop paper (sim-to-real VIO), deadline ~Aug 14 AoE.

---

## Do Not

- Do not create new MJCF files without explicit justification
- Do not import from root-level shim files (use `crawlbot.*`)
- Do not proceed past a failing metric by arguing it doesn't matter
- Do not use `pinocchio>=2.7` — this project uses `pin==3.9.0`
- Do not run simulations without setting `MUJOCO_GL`. **Prefer `disabled`** — it is what both gates use, and `osmesa` currently aborts pytest *collection* in this container (PyOpenGL, via `test_diagnostics`). Use `osmesa` only when you actually need rendering
- Do not assume quaternion conventions — verify in `state_conversions.py` (Pinocchio: xyzw, MuJoCo: wxyz)
- Do not use `weight_ratio > 1` in the QP — at `weight_ratio = 1` the α magnitudes ARE the hierarchy (two-task weighted stack; nominal priority integers are inert)
- Do not freeze references or add threshold-based switches to handle trajectory coordination failures — fix the trajectory synchronization instead
- Do not implement a three-phase state machine (DS/SS/EXT) — the architecture is two-phase (DS/SS) per spec §7.1
- Do not activate welds on position alone — require both `d < 5mm AND ori < 5°`
- Do not use α_wrench > 1 — wrench regularization at 100 consumed 20% of QP budget and blocked torso/EE authority
- Do not route the SS torso reference through the δ-mapping in two-task mode — SS uses the raw TorsoPlanner quintic (mapping is explicitly excluded, `sim_loop.py:2573-2576`); the mapping (δ(q_current)+F-SAT) remains a DS-only path
- Do not assume standalone component tests guarantee closed-loop success — always run the cascade bisection (A/B/C/D) to isolate integration failures
- Do not generate trajectory acceleration profiles without checking actuator feasibility — quintic on 591mm torso displacement saturates 20 Nm joints
