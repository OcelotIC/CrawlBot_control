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
PYTHONPATH=. MUJOCO_GL=osmesa python3 -c "import pinocchio; import mujoco; import casadi; print('OK')"

# 3. Run existing tests
PYTHONPATH=. MUJOCO_GL=osmesa python3 -m pytest tests/ -x -q --tb=short
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

---

## Commands

```bash
# Run all tests
PYTHONPATH=. MUJOCO_GL=osmesa python3 -m pytest tests/ -v

# Run a simulation script
MUJOCO_GL=osmesa PYTHONPATH=. python3 scripts/<script>.py

# Run diagnostics on a simulation log
MUJOCO_GL=osmesa PYTHONPATH=. python3 -c "
from crawlbot.diagnostics import run_diagnostics
import json
log = json.load(open('results/<log>.json'))
run_diagnostics(log, 'results/<output_dir>/')
"

# pip install
pip install <package> --break-system-packages
```

---

## Current Milestone

Update this line as work progresses:

> **For current-state questions, `docs/architecture/STACK_OVERVIEW.md` is the code-ground-truth reference.** This milestone block is a short pointer, kept in sync with it.

**→ Active:** **FROZEN 2.5 CANONICAL** — τ_w,max = 2.5 N·m (controller + plant) + Add-5 WQP weights, frozen at commit `32aefaf`, default-cap alignment `ec41cd9`, torso-export continuity fix `b619ef4`/`b37b528`. Docks **6/6** end-to-end (6-step traversal, worst at-weld 4.99 mm / margin 0.01 mm). **PR #29** (`j2/ds-active-rework` → `main`) carries the freeze + the full measurement campaign; Idriss reviews/merges via GitHub UI. See "Canonical Results" below and the phase reports under `results/j2_adjconv/PHASE_*.md`.

**Completed (this campaign):** weight-tuning isolation (COPRIORITY Addenda 1–8 → dock-lever hierarchy: torque ≥ 5× floor gate, EE ≥ 1000 gross reach, torso = fine lever, momentum/hw-slack inert); NMPC-PLAN-SATURATION (planned Ḣ_s saturation is NMPC-level, weight-independent) + U-PLAN-CHECK (rate-off plan exceeds envelope 2–4× — the constraint genuinely binds); CANONICAL-2p5 freeze with plant-cap proof (applied wheel torque measured ≤ 2.500 while U demands 26.9); TORSO-REF-AUDIT + export continuity fix (control byte-identical).

**Two-task SS stack (current architecture):** T-MOM linear + 6-D torso-pose + swing-EE + posture, all weighted, NO null-space projection, `weight_ratio=1` ⇒ **α magnitudes ARE the hierarchy** (nominal priority integers inert). In two-task SS the torso task is fed the **raw TorsoPlanner quintic+SLERP — the CoM→torso δ-mapping is NOT used in SS** (`sim_loop.py:2865-2870`); DS still uses the mapping. Superseded: the cooperative split, strict-P1, planned-δ, and the handoff-era "torso-ori blocker".

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
| hw_max | ±5 (**unchanged by design**) | Nms | `config.py:71-72`, spec §4.6 |
| **tau_w_max** | **2.5** — enforced 3×: controller `config.py:80`, AOCS clip `config.py:84`, **plant** MJCF wheel `ctrlrange ±2.5` (`VISPA_crawling_rwa3.xml:324-326`) | Nm | freeze `32aefaf` (was 5); dca.main default aligned `ec41cd9` |
| tau_max | 20 | Nm | `config.py:32` |
| dt_nmpc | 0.1 | s | `config.py:24` |
| dt_qp | 0.01 | s | `config.py:25` |
| NMPC horizon N | 8 | — | spec §5.1 |
| NMPC state dim | 9 | — | spec §5.1 (B2) |
| NMPC control dim | 12 | — | spec §5.1 |
| weight_ratio | 1.0 — **α magnitudes ARE the hierarchy** (two-task weighted stack, no null-space projection; priority integers inert) | — | `wholebody_qp.py:75` |
| **α torso-pose** | **2000** | — | `config.py:351` (Add-5) |
| **α swing-EE** | **1000** (dock lever; needs ≥ ~1000) | — | `config.py:319` (Add-5) |
| **α momentum (T-MOM)** | **400** (near-inert on Ḣ_s — NMPC owns the envelope) | — | `config.py:336` (Add-5) |
| **w hw-slack** | **800** (slacks active only if the hw box is violated) | — | `wholebody_qp.py:181` (Add-5) |
| **α posture** | **20** | — | `config.py:320` |
| **α torque-min** | **5** (must stay ≳ 5× accel-reg floor — Rule 14) | — | `sim_loop.py:1145` (QP-construction literal) |
| **α wrench-track** | **1.0** | — | `config.py:321` (Add-5; was 0.01 pre-freeze) |
| **α accel-reg** | **1.0** (regularizer floor) | — | `sim_loop.py:1145` |
| ε (Tikhonov) | 1e-6 (inert: λ_min(H_LS)=1 ≫ ε) | — | `hierarchical_qp.py:97` default |
| **κ_SS(H)** | ≈ 7.5e3 (530× below the pre-freeze canonical 3.6e6) | — | `canonical2p5_result.json` |
| α_com_soft | 0.0 | — | Soft-CoM residual disabled (QP has no direct CoM feedback) |
| CoM shaping | a_cruise_max=**0.0** (off) | — | Pre-planner cruise-accel cap disabled |
| Torso reference (SS) | **raw TorsoPlanner quintic+SLERP — NO δ-mapping in two-task SS** (`sim_loop.py:2865-2870`); DS still uses δ(q_current)+F-SAT | — | TORSO-REF-AUDIT; per-step reference re-anchored each SS |
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

- **2 pre-existing pytest failures** (identical set before/after the freeze — zero new): `test_far_infeasible_under_tight_rate` (its "tight rate" semantics should be re-examined now that the canonical cap IS 2.5 — follow-up) and `test_E7_t15_step2_dock_under_fk_mode`.
- **CoM-reference export snaps to the measured CoM at SS→DS entry** — logging convention (`_log_ds_tick` logs e_com=0 with ref:=measured, `sim_loop.py:1038-1041`); reviewer-reported magnitude ~76 mm (not repo-verified — the fulldiag CSV has no CoM-ref channel). Decision pending whether to apply the same terminal-hold fix as the torso export.
- **Fig-3 conservation quantity ‖L_total‖ is NOT in the fulldiag export** (verified: no `Ltot` column in `c25_fulldiag.csv`; it exists in the `export_figure_data.py` traversal CSVs). Dedicated export pending.
- `dca.main` sets `cfg.ds_centroidal_mode=True` (`diag_cooperative_arms.py:352`) — the locked config runs centroidal DS everywhere including the trailing settle; flags keyed on it cannot discriminate the DWELL (see TORSO-REF-EXPORT-FIX).

---

## Remaining Work

**Repo-side:**
- Fig-3 export: add the ‖L_total‖ / conservation channel to the fulldiag exporter.
- Decide/apply the CoM-reference export terminal-hold (same class as the torso fix) if Idriss rules the SS→DS snap non-publishable.
- Re-examine `test_far_infeasible_under_tight_rate` semantics at cap 2.5.

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
- Do not run simulations without `MUJOCO_GL=osmesa` (or `disabled` if rendering unavailable)
- Do not assume quaternion conventions — verify in `state_conversions.py` (Pinocchio: xyzw, MuJoCo: wxyz)
- Do not use `weight_ratio > 1` in the QP — at `weight_ratio = 1` the α magnitudes ARE the hierarchy (two-task weighted stack; nominal priority integers are inert)
- Do not freeze references or add threshold-based switches to handle trajectory coordination failures — fix the trajectory synchronization instead
- Do not implement a three-phase state machine (DS/SS/EXT) — the architecture is two-phase (DS/SS) per spec §7.1
- Do not activate welds on position alone — require both `d < 5mm AND ori < 5°`
- Do not use α_wrench > 1 — wrench regularization at 100 consumed 20% of QP budget and blocked torso/EE authority
- Do not route the SS torso reference through the δ-mapping in two-task mode — SS uses the raw TorsoPlanner quintic (mapping is explicitly excluded, `sim_loop.py:2865-2870`); the mapping (δ(q_current)+F-SAT) remains a DS-only path
- Do not assume standalone component tests guarantee closed-loop success — always run the cascade bisection (A/B/C/D) to isolate integration failures
- Do not generate trajectory acceleration profiles without checking actuator feasibility — quintic on 591mm torso displacement saturates 20 Nm joints
