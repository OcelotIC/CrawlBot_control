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

**→ Active:** Cooperative-arms traversal **docks steps 0–3**; **step 4 dock-timeout** is the open blocker (PR #17 merged: F-SAT rate-cap rescale + constant CoM-z standoff).

**Completed (current):** M-1…M6; M7 two-phase state machine; 7-DOF arms; cooperative-arms QP stack (torso-ang P1 + torso-lin/EE co-equal P2, brainstorming §4.4); AOCS legacy_corrected (orbital term on); dock gate (d<5mm AND ori<5°); α_wrench=0.01; **F-SAT rate-cap rescale**; **constant CoM-z standoff (−0.35)**; swing-reference logging clamp.

**Note — reverted/superseded since the handoff doc:** "planned-δ mapping" was **reverted to δ(q_current) + F-SAT** (commit 50a9e52, 2026-05-12); CoM-shaping `a_cruise_max` is **0 (off)**. The "torso-ori 45° last blocker" was partly a units artifact and is no longer observed (e_torso_ori ~1–3° in the docking run).

**Open:** step 4 dock-timeout; loop-free-mapping angular drift (committed OFF — the spec §3 constrained-dynamic-singularity, §6 mitigation never implemented); δ-source / F-SAT removal.

---

## Key Parameters (single source of truth: SimConfig)

| Parameter | Value | Unit | Reference |
|-----------|-------|------|-----------|
| Robot mass | ~71 | kg | spec §0.4 |
| Arm DOFs | 7 per arm (14 total) | — | spec §4.9, 7-DOF upgrade |
| nq / nv / nu | 21 / 20 / 14 (Pinocchio) | — | 7-DOF model |
| nq / nv / nu | 31 / 29 / 17 (MuJoCo+RWA) | — | 7-DOF + 3 wheels |
| Free DOFs in SS | 14 | — | 20 - 6 weld |
| hw_max | ±5 | Nms | spec §4.6 |
| tau_w_max | 5 | Nm | spec §5.1 |
| tau_max | 20 | Nm | SimConfig |
| dt_nmpc | 0.1 | s | spec §0.5 |
| dt_qp | 0.01 | s | spec §0.5 |
| NMPC horizon N | 8 | — | spec §5.1 |
| NMPC state dim | 9 | — | spec §5.1 (B2) |
| NMPC control dim | 12 | — | spec §5.1 |
| weight_ratio | 1.0 | — | Tasks use face-value weights + null-space projection |
| α_wrench | 0.01 | — | Pure regularization, not a competing objective |
| α_com_soft | 0.0 | — | Soft-CoM residual disabled (QP has no direct CoM feedback) |
| CoM shaping | a_cruise_max=**0.0** (off) | — | Pre-planner cruise-accel cap disabled |
| Mapping mode (SS) | **δ(q_current) + F-SAT** rate clamp | — | World-frame δ at current config (NOT planned-δ — reverted 50a9e52); F-SAT band-aids the resulting jitter. Loop-free δ_local exists but OFF |
| CoM-z standoff | −0.35 m (on) | m | Dock-IK + init pin crawl height (PR #17) |

---

## Do Not

- Do not create new MJCF files without explicit justification
- Do not import from root-level shim files (use `crawlbot.*`)
- Do not proceed past a failing metric by arguing it doesn't matter
- Do not use `pinocchio>=2.7` — this project uses `pin==3.9.0`
- Do not run simulations without `MUJOCO_GL=osmesa` (or `disabled` if rendering unavailable)
- Do not assume quaternion conventions — verify in `state_conversions.py` (Pinocchio: xyzw, MuJoCo: wxyz)
- Do not use `weight_ratio > 1` in the QP — task isolation comes from null-space projection, not weight penalties
- Do not freeze references or add threshold-based switches to handle trajectory coordination failures — fix the trajectory synchronization instead
- Do not implement a three-phase state machine (DS/SS/EXT) — the architecture is two-phase (DS/SS) per spec §7.1
- Do not activate welds on position alone — require both `d < 5mm AND ori < 5°`
- Do not use α_wrench > 1 — wrench regularization at 100 consumed 20% of QP budget and blocked torso/EE authority
- Do not use live δ(q_current) in the mapping during SS — use planned δ(q_planned) to avoid feedback jitter
- Do not assume standalone component tests guarantee closed-loop success — always run the cascade bisection (A/B/C/D) to isolate integration failures
- Do not generate trajectory acceleration profiles without checking actuator feasibility — quintic on 591mm torso displacement saturates 20 Nm joints
