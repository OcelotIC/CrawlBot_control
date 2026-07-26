# INTERNAL — DS Inter-Step Audit (read-only diagnostic)

**Target:** commit `21cec74`, branch `feat/ss-centroidal-momentum-task`.
**Run:** `Misc/runs/p3b_gate_w24000_kp3/` — canonical two-task 5-step traversal, working point
`ss_alpha_mom=5000, alpha_torso_pose=24000, ss_Kp_torso=3, ss_Kd_torso=2.5, ss_alpha_ee=3000,
ss_alpha_posture=20`, AOCS `--aocs_mode legacy_pid_numerical --K_omega 50`.
**Mode:** READ-ONLY (no source/config/MJCF edits; no re-runs). Evidence from committed `21cec74`
source + existing run artefacts only. Analysis: `Misc/scripts/audit_ds_phase2.py`, `audit_ds_phase3.py`.
**Status: COMPLETE (Phases 1–3).** No design decision is made here — only the evidence.

---

## Executive summary / provisional A/B/C read (NOT presuming the favourable outcome)

The inter-step DS on the two-task run is **passive-settle** (NMPC off, joint-velocity-damping,
hold-in-place references, passivity inequality; Phase 1). The coherence question resolves to
**A/C, not B**:
- **No active injection (Q1):** θ̈_s in the DS body is small and decaying (≤0.043 rad/s²), the
  settling reaction — not a structured/sustained dump.
- **The DS deposits a small, bounded, passive attitude increment** (mean intra-window 0.039°,
  ~17 % of the SS swing contribution). The within-gait same-sign accumulation on the z-axis is the
  **expected** consequence of a 6 s step period ≪ 288 s settling time (no inter-step AOCS recovery
  at this cadence), **not** evidence of issue B — B would require the DS-specific deposit to be
  active/structured, which Q1 rules out.
- **One flag (Q3), benign:** h_w is not bit-exactly frozen — it drifts ~0.7–5.9 mNms per window
  (≈0.1 % of operating h_w) via a rigid-body free-wheel/base kinematic coupling (τ_w applied = 0
  verified). The M7_T12 "h_w → 125 % of box" pathology does **not** occur.

**Residual judgement for Idriss:** the evidence supports treating inter-step DS as a benign passive
settle (A/C). The only open characterisation is the bounded weak-B deposit (0.039°/window) — small
and passive, but real; whether to surface it in the paper as a bounded-drift caveat is a narrative
call, not a correctness one. Two narrative constraints fall out: (i) the "does not accumulate"
claim is only sourceable in the **bounded** form (final 0.151° over 5 steps + 20 s settle), not
asymptotically (settle is mid-decay); (ii) the attitude performance is **architectural** (clean
two-task → small disturbance), **not** a K_θ effect.

---

## Phase 1 — Inter-step DS variant: **PASSIVE-SETTLE** (`_run_ds_passivity_loop`)

| element | evidence |
|---|---|
| dispatch | `sim_loop.py:1792–1834` → `_run_ds_passivity_loop` (`:601`), energy-based exit |
| **NMPC** | **OFF / bypassed** (`:1794–1795`); loop calls `self.qp_ss.solve(...)` directly (`:735`) |
| **QP task stack** | **joint-velocity-damping cost ONLY** (`wholebody_qp.py:1053`; `ds_centroidal_active` defaults False, `:345`). CoM/torso/EE/posture/two-task all dropped by `settle_mode` (`:677/:689/:737/:847/:989/:630`) |
| references | hold-in-place: `r_com_ref/p_torso_ref/R_torso_ref = current`, all velocities & FF = 0 (`:738–754`) |
| dissipation | passivity inequality `dq_jᵀτ_q + 2α·T ≤ 0` (`passivity_active=True`); τ_q settles joint KE |
| wheels | `ctrl[wheels] = 0.0` (`:765`) — τ_w applied is zero in inter-step DS |

`cfg.ds_centroidal_mode = True` (`diag:317`) but engages only the **terminal** settle (`:2270`) and
the optional dwell (`:1867`, gated on `_dwell_target>1.0` — **never fired** inter-step; no "DWELL"
in the log). All 5 inter-step DS settle via the short energy loop (initial 0.11 s `target_met`;
inter-step 0.51–1.72 s `plateau`). No active DS variant is wired inter-step.

## Phase 2 — Coherence

**Pinned gains (actual, this run):** K_θ=1.0, K_ω=50.0, K_d=25.0, K_hw=2.0
(`config.py:67/87/88/89`; K_ω via `--K_omega 50` = default). **I_s** = [1777, 1493, 597] kg·m²
(x,y,z). **2 % settling time** t_s = 8(I_s+K_d)/K_ω = **[288, 243, 100] s** → the +20 s terminal
settle is mid-decay on the heavy x-axis (~7 % of t_s).

*Methodology:* segmented on the authoritative per-tick `phase` label (an earlier pass keyed off
`inter_step_settles` timestamps, which are on a different clock and pulled one SS→DS boundary tick
per window, yielding false flags; `swing_arm` in DS rows is the just-landed-arm *label*
[`sim_loop:1804`], not a weld-state). Corrected results:

- **Gating:** τ_w = **exactly 0** on all 5 inter-step DS windows (SS = 5.0; terminal = 1.91, AOCS
  on). **Two-weld** confirmed every tick (stance anchor pair set; `d_grip_stance` ≤ 0.029 mm).
  POST_ABORT single-weld mode ruled out.
- **Q1 (active injection?) — NO.** θ̈_s body peak 0.003–0.043 rad/s² (median ≤0.002), decaying
  settling reaction, not structured/sustained → A/C.
- **Q2 (compounding / anchor).** Per-window Δθ_s bounded; anchor compliance `d_grip_stance` ≤
  0.029 mm → M7_T12 anchor concern absent. (See Phase-3 item 6 for nominal-vs-actual.)
- **Q3 (h_w freeze) — ⚑ FLAG (benign).** h_w drifts 0.7–5.9 mNms/window despite τ_w applied = 0
  (`:765`). Cause: free wheels (zero motor torque, frictionless — mixed-sign Δω rules out damping)
  kinematically track the residual base rotation; drift = I_w·Δω_body (magnitude matched). ≈0.1 %
  of operating h_w; stays ~1.2 Nms in DS, far from the 5 Nms box → **M7_T12 box pathology absent**.

## Phase 3 — Behaviour metrics (§VII)

*(θ_s per-axis = log3(R_init^T R_now); matches the F3F4 norm to 5e-10°.)*

**Item 1 — per-step SS-vs-DS attitude decomposition** (deg, [x,y,z]|norm):

| step k | ΔθSS_k (swing) | ΔθDS_k (inter-step settle) |
|---|---|---|
| 0 | [−0.064,+0.087,−0.108]\|**0.153** | [+0.030,−0.020,+0.025]\|**0.044** |
| 1 | [+0.042,+0.150,−0.220]\|**0.270** | [+0.007,−0.012,+0.019]\|**0.024** |
| 2 | [−0.079,+0.021,−0.099]\|**0.128** | [+0.022,−0.030,+0.046]\|**0.059** |
| 3 | [+0.028,+0.036,−0.069]\|**0.083** | [+0.009,−0.013,+0.026]\|**0.030** |
| 4 | [−0.074,+0.002,−0.153]\|**0.170** | [+0.065,−0.254,+0.369]\|**0.453** (DS = terminal) |
| **Σ** | SS = **0.728** | inter-step DS = **0.153** |

The inter-step DS deposit (Σ 0.153°, mean 0.039°/window) is **~4–6× smaller than the SS swing per
step** (~17 % of the SS attitude budget). SS is the dominant attitude driver; the DS is a minor,
passive contributor.

**Item 2 — Δθ_k net-per-cycle series** (θ_s at end of consecutive DS windows; [x,y,z]|norm):
Δθ_0 = …\|0.112, Δθ_1 = \|0.247, Δθ_2 = \|0.077, Δθ_3 = \|0.058, Δθ_4 = \|0.345 (→ terminal).
Per-axis sign over cycles 0–3 (excl terminal): **z monotone negative (−,−,−,−)**; x,y mixed.
**Interpretation (NOT B):** with a ~6 s step period ≪ 288 s settling time, the AOCS recovers almost
nothing *between* fast steps (active K_θ recovery is at the terminal settle, not inter-step), so a
same-sign within-gait accumulation on z is **expected and benign** — it is read as B only if the
DS-specific deposit is structured/active, which Q1 has ruled out (passive).

**Item 3 — Σ Δθ_k vs θ_s,final (CONFOUNDED — report only, not validation):** θ_s,final = 0.151°
(norm) after the terminal settle; Σ Δθ_k telescopes to 0.151°. Because the 20 s terminal settle ≪
288 s t_s, θ_s,final is **not asymptotic** → this cross-check is confounded and is not used as
validation. Defensible bounded statement: *over 5 steps + 20 s settle, final attitude 0.151°,
bounded.*

**Item 4 / §5.5 — SS→DS boundary (transition physics, NOT controller error, NOT d_DS):**

| step | ref-jump pos [mm] | ref-jump ori [°] | τ_w drop [N·m] | θ̈_s spike (guard) | θ̈_s body |
|---|---|---|---|---|---|
| 0 | 125.5 | 0.000 | 0.984 | 0.0011 | 0.0003 |
| 1 | 660.3 | 0.000 | 0.131 | 0.0012 | 0.0009 |
| 2 | 135.4 | 0.000 | 1.157 | 0.0026 | 0.0003 |
| 3 | 628.7 | 0.000 | 0.156 | 0.0021 | 0.0019 |
| 4 | 148.9 | 0.164 | 1.056 | 0.0021 | 0.0006 |

The torso position reference jumps 125–660 mm at the weld (planner quintic → DS hold), but the
**torso-pose task is DROPPED in settle_mode**, so this is a task-stack switch + weld impact, not
discontinuous-reference tracking. The resulting θ̈_s spike is negligible (≤0.003 rad/s²) — a mild
transition, not a controller error. (DS torso error is measured against the held DS-entry pose per
the established invariant, not this discontinuous reference.)

**Item 5 — τ_w saturation duty in SS** (frac ticks |τ_w|∞ = 5.0, i.e. K_θ clamped / open-loop):
step0 3.1 %, step1 0 %, **step2 42.4 %**, step3 0 %, **step4 48.6 %**; all-SS 12.1 %. During the
binding swings (2,4) the AOCS is saturated ~half the time → **K_θ is clamped there**; active
attitude restoring happens mostly at the terminal settle (τ_w mostly unsaturated). Supports item 8.

**Item 6 / Q2(b) — nominal-vs-actual anchor:** the 0.029 mm `d_grip_stance` is weld *compliance*,
not the scheduler-nominal-vs-actual anchor offset due to attitude drift (the true M7_T12 concern).
The per-step nominal-vs-actual weld offset is **not extracted** from the existing artefacts (would
need the scheduler's nominal anchor frame per step vs the realised weld pose). It is **negligible
for 5 steps regardless**: the inter-step attitude deposit (Σ 0.153°) at a ~1 m lever ≈ **2.6 mm**,
bounded and non-compounding over the traversal.

**§5.6 — Terminal settle (context):** 20 s centroidal-DS settle, AOCS **on** (τ_w max 1.91 N·m),
K_θ performs the slow attitude restoring (terminal Δθ = 0.453° recovery). θ_s,final = 0.151° — a
mid-decay value (t_s = 288 s), **not a hard lock**.

**Item 8 — K_θ=1 finding (narrative constraint):** K_θ=1 is modest and is **clamped** ~42–49 % of
the binding swings (item 5). The attitude advantage of the two-task run (θ_s peak 0.62° vs baseline
1.88°) is **architectural** — clean two-task momentum tracking produces a *small disturbance*, so
the platform barely tilts — **not** a K_θ effect. K_θ=1 provides only slow terminal restoring; it
does **not** "close" the irreversible drift. **The paper must not claim otherwise.**

## §6 — Inconsistency reconciliation (finalised on the two-task numbers)

The handoff's 0.085° "does not accumulate" and CAMPAIGN §11's ~0.4°/step "accumulates" are
reconciled — **(a) different quantities, (b) different configs**:
- **Quantity:** intra-DS-window Δθ_s here = **0.039°** mean (0.024–0.059°), the analogue of the
  0.085° baseline figure (and *smaller*, consistent with the cleaner two-task entry). §11's 0.4° is
  the **per-step net** (a different measure; here the per-cycle Δθ_k is 0.06–0.25°).
- **Config:** this run = `legacy_pid_numerical`, K_θ=1, clean two-task (no mapping cascade). §11 =
  `legacy_pd_numerical`, **K_θ=0**, + the δ-mapping cascade.
**Confirmed:** on the two-task run the intra-DS drift is small (0.039°) and the per-step net is
bounded (no monotone blow-up); θ_s,final = 0.151°. The "does not accumulate" claim is safe **only in
the bounded form** (5 steps + 20 s settle → 0.151°); a long-duration asymptotic claim is **not
sourceable** from this run (settle is mid-decay vs t_s = 288 s — amendment 2).

## §7 — Flags

1. **Q3 h_w drift (benign, characterised):** h_w not bit-exactly constant in inter-step DS (0.7–5.9
   mNms/window) despite τ_w applied = 0 — a rigid-body free-wheel/base kinematic coupling, ≈0.1 % of
   operating h_w, NOT an active injection / leak / τ_w≠0. M7_T12 box pathology absent. *No action;
   refines the "h_w frozen" statement to "frozen up to the wheel-base coupling."*
2. **No other flags:** τ_w = 0 (inter-step), two-weld throughout, no active injection (Q1), no
   anchor compounding, Δθ_k bounded/non-monotone. The AOCS terms match the paper's intended
   description (**PD on attitude + accel feedback + feedforward + desaturation, no integral** —
   verified prior chunk: `legacy_pid_numerical` has no integral term; "pid" is a code-identifier
   misnomer).
3. **Narrative caveats (not defects):** (a) "does not accumulate" is bounded-only, not asymptotic
   (amendment 2); (b) attitude performance is architectural, not K_θ (item 8); (c) the small-Δθ_DS
   "benign" conclusion is **scoped to the short passive inter-step DS** (0.51–1.72 s) and does not
   transfer to a long or active DS (amendment 4).
