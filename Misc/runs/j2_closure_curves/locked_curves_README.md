# Closure curves — LOCKED run (tick-by-tick, for visual validation BEFORE the PR)

Focused validation curves for **STOP-GATES C & D** — NOT the full 39-column figure re-export (that runs after
the PR). Three signals, tick-by-tick, so the curves can be plotted and validated visually.

## Config (the one C/D were measured on)
- **Figure config:** `legacy_pid_numerical`, exact-box (`--qp-envelope-exact`).
- **Locked fix:** `--interstep-settle-alpha-wrench 3` (ε=3, solver-derived) + `--interstep-settle-epsilon-v 5e-3`
  (ε_v=5 mm/s, dock-derived).
- **Commit:** `ed1c200` (j2/ds-active-rework). Run B (moving-CoM DWELL) = future work, not dumped.
- **Reproduce:** `diag_cooperative_arms.py --qp-envelope-exact --aocs_mode legacy_pid_numerical
  --interstep-settle-alpha-wrench 3 --interstep-settle-epsilon-v 5e-3 --out-dir locked_run` →
  `SSMOM_RUN_DIR=locked_run postprocess_results_figs.py` → `dump_closure_curves.py locked_run locked_curves.csv`.

## CSV — `locked_curves.csv` (1184 ticks)
Columns: `tick, t_s, phase, step_index, Hdot_s_{x,y,z}, Hdot_s_source, tau_w_{x,y,z}, dq_joint_norm, T_kin,
L_com_norm, hw_norm, settle_exit`.
- `Hdot_s` = exact origin-referenced Σ_j(r_Cj×f_j+τ_j) in the structure frame; **PLANNED** (from `lambda_ref`)
  on SS ticks, **REALIZED** (from `lambda_qp`) on inter-step DS ticks — tagged by `Hdot_s_source`.
- `tau_w` = AOCS-commanded wheel torque (per axis). `±5 N·m` is the envelope cap (the C3 reference line).
- `settle_exit = 1` marks the last tick of each inter-step settle (the exit-residual sample for Signal 2).

## Phase-segment tick ranges (for shading SS vs DS vs settle)
| segment | ticks | t [s] | phase | step |
|---|---|---|---|---|
| initial setup DS | 0–9 | 0.01–0.10 | DS | −1 |
| SS swing 0 | 10–37 | 0.11–2.81 | SS | 0 |
| **DS settle 0** | 38–179 | 2.92–4.33 | DS | 0 (exit @179) |
| SS swing 1 | 180–262 | 4.34–12.54 | SS | 1 |
| **DS settle 1** | 263–463 | 12.65–14.65 | DS | 1 (exit @463) |
| SS swing 2 | 464–572 | 14.66–25.46 | SS | 2 |
| **DS settle 2** | 573–705 | 25.57–26.89 | DS | 2 (exit @705) |
| SS swing 3 | 706–790 | 26.90–35.30 | SS | 3 |
| **DS settle 3** | 791–942 | 35.41–36.92 | DS | 3 (exit @942) |
| SS swing 4 | 943–983 | 36.93–40.93 | SS | 4 |
| terminal DS | 984–1183 | 41.03–60.93 | DS | 4 |

Arm of each dock (preceding the settle): step0→**b**(anchor3), step1→**a**(3), step2→**b**(4),
step3→**a**(4), step4→**b**(5). The **arm-a settles** (the chatterers in the diagnosis) are **DS settle 1**
(t 12.65–14.65) and **DS settle 3** (t 35.41–36.92).

## Docks (C1): 5/5 — **[1.86, 4.79, 4.90, 4.87, 4.67] mm** (all < 5 mm gate).

---

## What the three signals show (how to read the plots)

### Signal 1 — chatter is dead (`Hdot_s_{x,y,z}` + `tau_w_{x,y,z}`, full cycle)
- In the **arm-a DS settles** (settles 1, 3) the dominant-axis Ḣ_s **sign-flip fraction is 0.014 / 0.008**
  (≈0; the diagnosis baseline was ≈0.9). The period-2 ±5 bang-bang is **gone** — Ḣ_s is smooth.
- **Note for the plot:** the first **13–19 ticks of each DS block ride the −5 cap** — but this is the
  **post-dock IMPACT transient**, not chatter: it is **contiguous from the block start**, **same-sign**, and
  decays **smoothly and monotonically** (e.g. settle 0 z-axis −5.00 → −4.26 over ~0.2 s). The settle median
  |Ḣ_s| is **0.24–0.52 N·m**. Likewise `tau_w` shows a **single** Δτ_w≈7–9 spike at each SS→DS transition
  tick (the contact/AOCS hand-off), **not** the sustained tick-to-tick ±10 slamming of the chatter. Shade the
  first ~0.2 s of each DS block as the impact transient.

### Signal 2 — residual bounded (`dq_joint_norm`, `T_kin` within each settle)
- Every inter-step settle **decays cleanly to its exit** — `T_kin`: 7.0e-2→6.1e-7 (settle 0), 1.2e-2→6.1e-7
  (1), 6.1e-2→6.2e-7 (2), 1.1e-2→6.2e-7 (3); `dq_joint_norm` → ~0.000. **No pumping** (the chatter baseline
  *rose* ×6.7 within the settle).
- The settle-exit residual (`settle_exit=1` rows) does **not** grow across the arm-a settles — bounded
  step-to-step. (Exit ‖dq_joint‖ ≈ 0.05–0.07 rad/s, decreasing arm-a settle 1→3.)

### Signal 3 — conservation holds throughout (`L_com_norm`, `hw_norm`)
- `‖L_com‖` (total angular momentum about the system CoM) stays small **across the whole run** — median
  0.056, **final 0.0000**; the max 1.507 is a transient *during the swings* (expected), not a DS drift. The
  bounded settle residual injects **no secular momentum** mid-traversal. `‖hw‖` max 3.997 Nms (< 4.5).

---

**Characterization-only** — no `crawlbot/` change, no merge, no PR. After visual validation, the sequence
resumes: PR (#26) → figure-data regeneration → full figures.
