# M7 Technical Log — Correction Task

**Date:** 2026-04-17
**Scope:** Surgical edits to `docs/architecture/M7_TECHNICAL_LOG.md`
**Input artefacts:**
- `results/archive_rediagnostic.md` (from Step 2 of post-processing)
- `docs/architecture/POST_ABORT_DIVERGENCE.md` (from Step 3)

---

## 1. Context

The archive re-diagnostic produced in Step 2 falsifies the central claim of the current log: "torso orientation error has been 29–45° across all versions since v12". Per-phase metrics show SS orientation peak is 0.53 ± 0.01° across v17, v19, v20, v21 — below the standalone floor and well inside the 5° dock threshold. The 29–45° figures were global-max artefacts dominated by post-abort divergence in the DS hold phase after `dock_timeout`. That is a separate failure mode, orthogonal to SS tracking.

The log must be corrected. This is **not a rewrite**. The v12–v21 version history, the weight/configuration fix table, the trajectory-shaping reversal table, and the architectural insights in §6 are preserved. Only passages making claims contradicted by the new per-phase data get amended.

## 2. Required changes

### 2.1 Add §0 — Correction Notice (new, top of file)

Insert immediately after the date/branch/commit header. Two paragraphs:

- **P1:** State that the log as previously written drew conclusions from global-max metrics; that per-phase analysis (2026-04-17 post-processing) overturned the central claim; that sections have been amended where factually incorrect. Reference `results/archive_rediagnostic.md`.
- **P2:** One-sentence corrected picture — SS orientation is 0.5° across all versions; the remaining real problem is EE position tracking in SS (24 mm standalone → 162 mm closed-loop peak, 25–41 mm at abort); plus a separate post-abort DS divergence issue.

### 2.2 Amend §10 (v20/v21 plans and results)

Keep the v21 result table. Add below it a new subsection **"Per-phase re-diagnostic (2026-04-17)"** containing the full Step 2 table verbatim. Follow with one paragraph: SS orientation was never the problem; the original v20/v21 rationale (CoM shaping to free actuator budget for EE) remains valid for position tracking. The architectural work was sound; its characterization was not.

### 2.3 Rename and rewrite §11

**Old title:** "Current State (v21) — Orientation Is The Last Blocker"
**New title:** "Current State (v21) — Position Inflation in SS Is the Remaining Blocker"

New structure, three subsections:

**What's solved** — keep the existing bullet list (QP task stack, actuator budget, momentum management, singularity, AOCS exonerated). Append one bullet confirming SS orientation is at the standalone floor across all versions (0.5°).

**What's broken (revised)** — two items:

1. EE position peak 162 mm / closest approach 25–41 mm in SS across v17–v21, with 24 mm standalone as achievable floor. ~6.7× closed-loop inflation.
2. Post-abort DS divergence: single scheduler-level root cause with three downstream symptoms. Root cause: main loop at `sim_loop.py:1337-1338` advances `step_idx += 1; i += 2` unconditionally after SS, without gating on dock success. The trailing DS is a scheduled plan entry, not a post-dock consequence. When SS aborts, trailing DS runs with arm B not welded in MuJoCo, but the plan treats it as double-support. Symptoms:
   - H_DS1: `cc_ds = DOUBLE` at `sim_loop.py:1343` — QP builds Jacobians for both tools at 1597-1598 against a single-weld physical state.
   - H_DS2: `dock_configuration(anchor_a, anchor_b, q_init=pq)` at 1365-1375 assumes both tools at anchors; `set_hold(...)` produces the 3.4° jump in `q_torso_ref` at the SS→DS boundary.
   - H_DS3: `passivity_active=True` at line 1712 is gated on `phase=='DS'`, a plan-string check, not a physical-contact check.

   Cite `docs/architecture/POST_ABORT_DIVERGENCE.md`.

**Next investigation (revised)** — a single `if docked` branch at the scheduler level is needed before trailing-DS setup, plus an architectural decision (to be specified by Idriss) on abort-DS semantics: freeze swing arm / attempt retraction / stop. The three orthogonal experimental tests in `POST_ABORT_DIVERGENCE.md` are retained as **diagnostic decomposition** only — they quantify how much of the 45° is attributable to each symptom, useful for scientific characterization but not the candidate fix.

For position: apply the same bisection structure originally proposed for orientation (cases A/B/C/D) to `ee_pos_peak_SS` as the primary metric.

### 2.4 Strike §7F ("Torso orientation management")

Replace the section body with a one-line crossed-out note:

> ~~Struck 2026-04-17: SS orientation is at the standalone floor. No management needed.~~

Keep §7A–E and §7G–H intact.

### 2.5 §2 ("The Core Problem — Dynamic Coupling During SS") — NO CHANGE

This section's position diagnosis (mapping 3.6× inflation, null-space drift 3.4×) was correct. The per-phase data confirms position is the locus.

### 2.6 Amend §5 item 3 ("Remaining Unknowns")

Current text attributes "31° torso orientation error" to "pure arm-reaction disturbance". Replace with: this figure was a global-max artefact; SS orientation is tracked to the standalone floor.

## 3. Style

- No apology, no editorializing. Factual corrections only.
- Preserve every numerical value that was correct.
- Every amended paragraph ends with `(amended 2026-04-17)`.
- Git commit message states exactly what changed, references `results/archive_rediagnostic.md`.

## 4. Pass criteria

- Log reads consistently end-to-end. No residual references to "45° orientation blocker" or "60× cascade inflation".
- Step 2 table appears verbatim in §10.
- §11 "Next investigation" points at EE position bisection plus the scheduler-level fix for DS — not orientation bisection.
- 191/191 tests still pass (documentation change, but run pytest per A5).

## 5. Explicit prohibitions

- Do not touch `brainstorming_reworked_architecture.md`.
- Do not modify `CLAUDE_CODE_HANDOFF.md`.
- Do not modify any source file outside `docs/`.
- Do not start any Track 1 or Track 2 work.
- Do not commit without sending the amended §0, §10 addendum, and rewritten §11 to Idriss first.
