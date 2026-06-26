# INTERNAL — DS Inter-Step Audit (read-only diagnostic)

**Target:** commit `21cec74`, branch `feat/ss-centroidal-momentum-task`.
**Run:** `results/p3b_gate_w24000_kp3/` — canonical two-task 5-step traversal, working point
`ss_alpha_mom=5000, alpha_torso_pose=24000, ss_Kp_torso=3, ss_Kd_torso=2.5, ss_alpha_ee=3000,
ss_alpha_posture=20`, AOCS `--aocs_mode legacy_pid_numerical --K_omega 50`.
**Mode:** READ-ONLY (no source/config/MJCF edits; no re-runs). Evidence from committed `21cec74`
source + existing run artefacts only.

**Status: PHASE 1 COMPLETE (variant pinned). Phases 2 (coherence) and 3 (behaviour metrics) are
PENDING** — paused at the Phase-1 stop-gate per the brief; the variant gates all downstream
interpretation. (Committed at the user's explicit request; the brief had specified no-commit.)

---

## PHASE 1 — Inter-step DS variant: **PASSIVE-SETTLE** (`_run_ds_passivity_loop`)

The inter-step DS (the DS windows *between* locomotion steps) commands **no motion**: it is a
hold-in-place, joint-velocity-damping settle with the NMPC bypassed and the AOCS wheels stood down.
It is **not** an active DS variant (no `active_ds_torso_advance`, no centroidal-control inter-step).

### Code path (`21cec74`)
| element | evidence |
|---|---|
| inter-step DS dispatch | `sim_loop.py:1792–1834` — each inter-step DS calls `_run_ds_passivity_loop(...)` |
| settle engine | `sim_loop.py:601` `_run_ds_passivity_loop` — M2 QP, energy-based exit (`T_kin < T_settle`) |
| **NMPC** | **OFF / bypassed** in inter-step DS — `sim_loop.py:1794–1795`; loop calls `self.qp_ss.solve(...)` directly (`:735`) |
| QP solve flags | `settle_mode=True, passivity_active=True` (`:755–756`); `ds_centroidal_active` defaults `False` (`wholebody_qp.py:345`, not passed) |
| **QP task stack** | **joint-velocity-damping cost ONLY** — added at `wholebody_qp.py:1053` (`settle_mode AND NOT(ds_centroidal_mode AND ds_centroidal_active)` = True). Centroidal-DS tasks **not** added (`:1065`, requires `ds_centroidal_active`). |
| tasks dropped by `settle_mode` | CoM (`:677`), torso-pose (`:689`), swing-EE (`:737/:847`), posture (`:989`, `_posture_in_ds=False`), two-task stack (`:630`) |
| references | **hold-in-place**: `r_com_ref=rs.r_com` (current), `p_torso_ref/R_torso_ref = current torso pose`, all velocities & feed-forwards zero (`:738–754`) |
| dissipation | passivity inequality `dq_jᵀτ_q + 2α·T ≤ 0` (`passivity_active=True`); `τ_q` (joints) settles the joint kinetic energy |

### `ds_centroidal_mode` is True but does NOT touch the inter-step settle
`cfg.ds_centroidal_mode = True` in this run (`diag_cooperative_arms.py:317`; `baseline_ds_rework`
not set). It engages only:
- the optional centroidal **DWELL** (`sim_loop.py:1867`), gated on `_dwell_target > 1.0`;
- the **terminal** settle after the last dock (`sim_loop.py:2270`, `ds_centroidal_active=cfg.ds_centroidal_mode`).

The run log shows **no "DWELL" line** for any inter-step DS — all five settle via the short
energy-based loop, so the centroidal dwell never fired inter-step:

| DS window | t_start [s] | duration | n_steps | exit |
|---|---|---|---|---|
| initial (pre-step 0) | 0.00 | 0.110 s | 11 | target_met |
| after step 0 | 3.31 | 1.720 s | 172 | plateau |
| after step 1 | 13.43 | 0.510 s | 51 | plateau |
| after step 2 | 17.24 | 1.020 s | 102 | plateau |
| after step 3 | 26.36 | 0.510 s | 51 | plateau |
| **terminal** (post-dock) | 30.37 | **20.0 s** | — | centroidal-DS (§5.6 context, out of coherence scope) |

So `ds_centroidal_active=True` applies only to the 20 s terminal settle; every inter-step DS is
pure passive-settle.

### Provisional A/B/C implication (NOT presuming the favourable outcome)
With no commanded motion (hold-in-place references), joint-velocity-damping, NMPC off, and the
wheels stood down, the inter-step DS disturbance `d_DS ≈ settling reaction only` → the **A/C**
regime is structurally indicated, and **issue B (active momentum injection) is not wired** in the
inter-step DS. This is a *structural* read from the code path; it must still be **confirmed
numerically in Phases 2/3**: `τ_w ≈ 0` throughout each inter-step DS, genuinely two-weld every
tick, `h_w` frozen, and `θ̈_s,DS ≈ 0` in the DS body (a structured/sustained `θ̈_s` would contradict
this and reopen B). Those measurements are pending.

---

## Phases 2–3 — PENDING
- **Phase 2 (coherence):** gating (`τ_w≈0`, two-weld, pin actual `K_θ`/`K_d`) + Q1 injection /
  Q2 compounding & anchor drift / Q3 `h_w` freeze.
- **Phase 3 (behaviour):** `θ̈_s,DS` (body vs weld boundary), `Δθ_DS`, the per-step `Δθ_k` series,
  total net irreversible vs `Σ Δθ_k`, SS→DS transition, terminal-settle context; the §6
  inconsistency check (0.085° vs ~0.4°/step) and §7 flags.
