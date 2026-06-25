# SS Centroidal-Momentum Task — Formulation & Validation Plan

**Status:** Phase 0 — formulation (no code)
**Target branch:** `feat/ss-centroidal-momentum-task` (off `main` @ `dcda974` — the commit that added this memo; supersedes the `63bea1f` referenced in §4, which predates this memo's own commit)
**Companion memo:** `DS_REWORK_CENTROIDAL_2026-06.md` (this work unifies SS with the DS architecture established there)
**Decision record:** L_com rows EXCLUDED in v1 (prudent variant, Idriss arbitration 2026-06-12). TorsoPlanner-linear retention decided by A/B comparison (Phase 2/3).
**Paper impact:** gated. If the Phase-3 gate passes, the IEEE Access manuscript is regenerated on this architecture (new audit required). If it fails, the paper ships on `5bca42c` unchanged and this work becomes revision material.

---

## 1. Problem

The SS-phase NMPC→QP channel is currently a **position-level CoM→torso mapping**:

```
r_b_ref(t) = r_com*(t) − (1/m) · R_b(t) · s(q(t))        [CoMToTorsoMapping]
```

with the mass first-moment offset `δ = R_b·s` evaluated at the **measured current configuration**. The reference is a function of the state it commands → algebraic loop (reference → tracking → q → reference) closed at 100 Hz through the QP dynamics. Symptoms and band-aids on `main`:

- jitter on the mapped torso reference → **F-SAT rate clamp** active in the canonical config (band-aid, `CLAUDE.md` Open);
- the alternative "planned-δ" was **reverted** (`50a9e52`): δ(q_plan) requires a full configuration plan consistent with the centroidal plan, which the reduced NMPC does not produce — by design of the reduction;
- internal contradiction in `CLAUDE.md` (guardrail line says "do not use live δ(q_current)"; the canonical mode IS δ(q_current)+F-SAT).

During DS, the same family of pathologies (no centroidal objective in the QP) was diagnosed and fixed by the centroidal-DS architecture (DS_REWORK memo §3–4). **SS still runs the legacy mapping.** This work replaces it.

## 2. Solution — momentum-level task via the Centroidal Momentum Matrix

Standard terrestrial momentum-based WBC (Orin–Goswami–Lee 2013; Wensing–Orin; Herzog et al. 2016; Kuindersma et al. 2016; Koolen et al. 2016): do not map positions; track momentum. With the Centroidal Momentum Matrix `A_G(q) ∈ R^{6×nv}`:

```
h_G = [ m·v_com ; L_com ] = A_G(q) · q̇
```

the WBC task is acceleration-level:

```
A_G · q̈ + Ȧ_G · q̇ = ḣ_G_des                                    (T-MOM)
```

**Key structural property:** the configuration dependence (the old δ) lives in the task matrix `A_G(q)`, evaluated at current q like any task Jacobian (same status as the EE task Jacobian). No reference depends on the controlled state → the algebraic loop disappears by construction; F-SAT becomes unnecessary on this channel.

### 2.1 v1 scope — LINEAR rows only (prudent variant)

Track the 3 **linear** rows of (T-MOM):

```
[A_G]_lin · q̈ + [Ȧ_G]_lin · q̇ = m · a_com_des
a_com_des = a_ff* + Kp_com (r_com* − r̂_com) + Kd_com (v_com* − v̂_com)
```

- `a_ff*` = NMPC feedforward (existing `a_com_ff` output), `r_com*`, `v_com*` = NMPC optimal plan at the current tick.
- The P feedback on `r_com` closes the position drift inherent to an acceleration-level task.
- **L_com rows are EXCLUDED in v1.** Rationale: torso angular velocity dominates L_com; the strict-P1 torso-orientation null-space projection amputates the angular rows of A_G in a way only quantifiable empirically. Risk of P1↔L_com conflict is the main unknown; v1 avoids it. Extension "v2 = +L_com rows at moderate weight" is pre-registered as follow-up, motivated by envelope fidelity (realized Ḣ_s closer to planned Ḣ_s), but is OUT OF SCOPE of the gate.

### 2.2 Task hierarchy in SS (v1)

| Priority | Task | Status vs canonical |
|---|---|---|
| hard constraint | stance weld (bilateral, 6D) | unchanged |
| hard constraint | dynamics (VI-D.2), torque/wrench/q̈ boxes | unchanged |
| P1 (strict, null-space proj.) | torso orientation (SLERP ref) | unchanged |
| P2 (weighted) | **centroidal linear momentum (T-MOM lin, α_mom)** | **replaces torso-linear channel** |
| P2 (weighted) | swing EE (α_ee = 3000) | unchanged |
| P2 (weak, A/B) | torso-linear regularisation (TorsoPlanner quintic ref, α_tl_weak) | **A/B variant — see §3** |
| P3 | posture (α_posture = 20) | unchanged |
| P4 | wrench regularisation toward λ* (α_wrench = 1e-2) | unchanged |

Notes:
- SS has nc = 1 (single 6D weld): the grasp map is rank 6, no internal-stress null space → the DS internal-stress regularisation is NOT needed in SS (it remains gated to nc=2 as on `main`).
- DOF budget in SS: nv = 20 − 6 (weld) = 14 free; P1 consumes 3 → 11 residual for momentum-lin (3) + EE (6) [+ weak torso-lin (3) in variant B] — weighted arbitration as in the current 1:6 scheme.
- Initial weights: α_mom = 500 (slot of the replaced torso-linear task), Kp_com/Kd_com seeded from existing `ss_Kp_com`/`ss_Kd_com` config entries if present. **Phase-1 check:** existing plumbing `ss_alpha_com`/`ss_Kp_com` in `_build_qp` may already implement a CoM task (canonical α = 0 / disabled per the 5bca42c audit: "no α_com task"). Determine whether that path is A_G-based or torso-proxy-based; reuse only if A_G-based and sound, otherwise implement cleanly and leave the legacy path untouched.
- Reference frames: A_G (Pinocchio ccrba/dccrba) is expressed at the CoM, world-aligned; NMPC references live in the rotating structure frame R_s. At ω_s ~ 1 °/s transport terms are negligible but MUST be stated and consistently handled (same treatment as the H-estimator `include_transport` logic). Phase-1 deliverable: a short frame-consistency note in the implementation log.

### 2.3 What this changes for the architecture narrative

- NMPC → QP channel becomes: optimal centroidal plan (r*, v*, a_ff*) → momentum task (T-MOM) → QP. CoMToTorsoMapping and F-SAT become unused in SS (code retained, channel selected by flag).
- Unification: DS already runs a centroidal objective (DS_REWORK §4); with this work the QP has a centroidal task across the whole cycle. Single coherent story.

## 3. TorsoPlanner-linear A/B comparison (Idriss decision: compare)

The SLERP **orientation** reference stays at P1 regardless. The question is the **linear** quintic reference:

- **Variant A — removed:** no torso-linear task in SS. CoM placement is fully delegated to T-MOM; torso position is an outcome (weld + P1 + posture resolve the remaining DOFs).
- **Variant B — weak regularisation:** TorsoPlanner quintic retained as a weak P2 task (α_tl_weak ≈ 50, i.e. one order below α_mom), expressing a torso-posture preference without fighting the momentum task.

Hypotheses to discriminate: A is cleaner but may let the torso wander within the redundancy during the swing (analogue of the DS §3.4 pathology, though SS has only 14−3−3−6 = 2 weakly-constrained DOFs in variant A with posture at P3); B guards against wander but reintroduces a position-level reference (benign: it is a fixed planner output, not state-dependent — no algebraic loop). Both variants run Phase 2; both survivors run Phase 3; selection by gate metrics, tie-break = simpler (A).

## 4. Validation plan and gate

Methodology per repo standard: standalone validation before integration; cascade bisection on failure; root-cause before any fix; no patchwork (no weight inflation to mask a structural issue).

### Phase 1 — baseline + implementation (stop-gate at end)
1. **Re-establish the baseline on `main` HEAD (`63bea1f`):** run the canonical 5-step scenario (`diag_cooperative_arms.py`, `legacy_pid_numerical`, K_ω=50, τ_w,max=5). Sanity-compare against the `5bca42c` artifacts (`postproc_F3F4.csv` metrics): 5/5 docks, attitude peak/final, h_w peak, tracking RMS. Any unexplained deviation vs 5bca42c STOPS the phase (main has the DS rework merged; deviations must be attributable to it and to it only).
2. Implement T-MOM (linear rows) behind `cfg.ss_centroidal_momentum_task: bool = False` (+ `ss_alpha_mom`, `ss_alpha_tl_weak`, Kp/Kd entries). Flag OFF must be **bit-identical** to the Phase-1.1 baseline.
3. Log run metadata in every result dir: `git rev-parse HEAD`, dirty/clean state, full config dump.
4. STOP. Report: baseline comparison table, bit-identical proof, implementation notes, frame-consistency note.

### Phase 2.0 — standalone task validation (unit, no swing / no NMPC / no MuJoCo) (stop-gate)
Purpose: prove the T-MOM linear task is correctly formulated and realized BEFORE
exposing it to swing dynamics. A failure here is a formulation bug, isolated from
any integration effect. Harness: the existing pure-Pinocchio path in
`tests/test_reworked_qp.py` (`_make_m2_qp` / `_solve_qp_step` / `_integrate`),
extended to enable `cooperative_arms_mode=True` + `ss_centroidal_momentum_task=True`,
with `p_ee_ref=None` (EE task off) and a frozen nc=1 contact (`single_contact_config`
/ `dock_state`). NMPC bypassed (`lambda_ref=0`); CoM references injected directly
via the existing `r_com_ref/v_com_ref/a_com_ff` kwargs.

Tests to add (Z1 analytical driver + Z2 realized-vs-commanded comparator):
1. **Static hold.** r_com_ref = r_com(q0), v*=0, a_ff*=0 → assert ‖q̈‖≈0 and CoM
   drift below tolerance over N steps. Catches a wrong PD sign or a reference that
   depends on the controlled state.
2. **Pure tracking, per axis.** Impose an analytical r_com_ref(t) (jerk-limited
   step and sinusoid, one axis at a time, via the existing `septic()` shape
   retargeted to CoM) → assert realized r_com tracks to tolerance AND the realized
   task row `J_com·q̈ + J̇_com·q̇` reproduces the commanded `a_com_des` (checks that
   Ȧ_G·q̇ is correctly assembled, not just A_G).
3. **Mass-scalar sanity.** Verify the m factor folding into α_mom is handled as
   intended: the effective CoM-task gain must not be off by a factor m≈71.
4. **Variant B weak-reference coexistence.** With ss_alpha_tl_weak active, confirm
   the weak torso-linear reference does not fight CoM tracking (test-2 tolerance
   still met).

Frame note: the QP is world-frame-native (J_com is the world CoM Jacobian); no
world↔R_s transport term lives in the QP. Transport consistency is therefore NOT
a Phase-2.0 concern — it is deferred to Phase 2.1, where R_s NMPC references meet
the world-frame QP.

GATE to Phase 2.1: all four tests pass. A failure stops here with a cascade-
bisection diagnosis; no swing run until the task is proven in isolation.

### Phase 2.1 — single-step swing, variants A/B
Scenario: canonical step 1 in isolation, variants A and B, flag ON.
Metrics vs baseline single step: torso jitter (reference and realized), F-SAT activity on the legacy channel = N/A (channel off) — instead log T-MOM residual; CoM tracking error (r̂_com vs r*); EE docking distance/orientation at capture; realized Ḣ_s vs NMPC-planned Ḣ_s (fidelity, informative only in v1); QP solve time (<5 ms budget); joint-torque margins.
STOP. Report per variant; kill a variant only with a diagnosed cause.

### Phase 3 — canonical 5-step + GATE
Scenario: canonical 5-step traversal, surviving variant(s).

**GATE criteria (all required, defined before running):**
1. 5/5 docks, **per-step capture margin ≥ canonical baseline margin** (baseline: 1.86/4.94/4.96/4.77/5.00 mm against the 5 mm gate);
2. torso tracking ≤ baseline (orientation RMS ≤ 0.68°, position peak ≤ 17.6 mm — or justified equivalence);
3. envelope: planned ‖Ḣ_s‖∞ ≤ 5 N·m at all knots (constraint unchanged), realized τ_w saturation ≤ 2.95 % of ticks;
4. platform attitude peak ≤ ~1.9°, final ≤ ~1.65° (parity with baseline);
5. h_w peak ≤ baseline + margin (3.38 N·m·s reference);
6. flag OFF bit-identical to Phase-1 baseline (re-verified at the Phase-3 commit).

**Gate PASS →** select variant (tie-break A), freeze commit, trigger paper-regeneration track: full paper-vs-code audit on the new commit (5bca42c-audit standard), regenerate Section VII from new CSVs, rewrite VI-C/VI-D and Fig. 1 (mapping/F-SAT removed from the description; momentum task described; DS section updated to centroidal-DS), THEN run F-ABL on this commit.
**Gate FAIL →** paper ships on `5bca42c` as planned; this branch continues post-submission with the failure diagnosis as input. No "almost passing, one more week" middle path: a miss is a documented diagnosis, then the submission proceeds.

## 5. Explicitly out of scope (this branch)
- L_com rows in T-MOM (v2, pre-registered follow-up).
- F-SAT / CoMToTorsoMapping code removal (only channel deselection; removal is post-submission hygiene).
- DS-side changes (centroidal-DS is already on main and is taken as-is).
- DS→SS post-dwell launch transient (NEXT_SESSION Tier 1 — separate thread; NOTE: if Phase-2/3 diagnosis lands on reference discontinuities at SS start, coordinate with that thread before fixing).
- Multi-traversal scenarios, mass ratios ≠ 1 %.

## 6. Plotting rules (binding — no phase report without its plot set)

Every phase report MUST include a standard diagnostic plot set. Numbers alone are not sufficient for review; the curves are how results are appreciated and anomalies spotted.

**General rules:**
- All plots generated by a committed script (`scripts/plot_ssmom_phaseN.py` or extension of `postprocess_results_figs.py`) reading the run's logged data — never hand-made, always re-runnable.
- Saved as PNG into the run's result directory; the report embeds or links them.
- **Overlay convention:** baseline (Phase-1.1 main-HEAD run) in grey/dashed, candidate variant(s) in color, same axes — every comparison plot superposes, never side-by-side panels only.
- SS phases shaded (same convention as paper Figs. 2–5); per-axis curves labeled x/y/z; constraint limits as dashed horizontals where applicable.
- Identical y-scales between variants A and B for the same quantity.

**Phase-1.1 (baseline re-establishment) plot set:** the four paper-style figures regenerated on main HEAD (torso tracking ori+pos; swing-EE ori+distance; platform attitude |θ_s|; Ḣ_s and τ_w per axis with ±5 N·m), overlaid on the 5bca42c canonical curves from `postproc_F3F4.csv`. Purpose: any main-vs-5bca42c deviation must be *visible*, not just tabulated.

**Phase-2 (single step) plot set, per variant, overlaid on baseline:**
1. CoM tracking: r̂_com vs r*_com per axis, and the error;
2. T-MOM residual ‖[A_G]_lin q̈ + [Ȧ_G]_lin q̇ − m·a_com_des‖ over the step;
3. torso position per axis (realized vs reference where one exists) — the jitter plot; include the baseline's mapped-reference jitter for contrast;
4. swing-EE distance + orientation error to target (docking approach);
5. realized Ḣ_s vs NMPC-planned Ḣ_s per axis (fidelity, informative in v1);
6. joint torques vs ±20 N·m bounds; QP solve time histogram.

**Phase-3 (5-step, gate) plot set:** the full paper figure set (Figs. 2–5 equivalents) for the candidate, each overlaid on baseline, plus a per-step docking-margin bar chart (candidate vs baseline vs the 5 mm gate) and the h_w per-axis trajectory vs ±5 N·m·s. Every gate criterion must be readable off at least one plot.

## 7. Claude Code session rules (binding for every phase)
- Work only on `feat/ss-centroidal-momentum-task`; verify with `git branch --show-current` before first commit; push, never merge.
- Stop-gates between phases are mandatory; findings are relayed to the review session before the next phase begins.
- No commit before the phase's verification step passes; commit messages reference this memo.
- Read-only with respect to `results/` baselines from prior commits; new runs go to new directories with metadata per Phase-1.3.
