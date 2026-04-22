# M7 T12 — DS1 divergence at 14% mass ratio: mechanism memo

**Status.** Internal technical memo. Not for publication.
**Date.** 2026-04-22.
**Run under analysis.** `results/M7_14pct_1step_v22_with_swing_hold/` (T12).
**Predecessor.** T11 closed 2026-04-20, `results/M7_1pct_1step_v22_with_swing_hold/`.
**Purpose.** Freeze the T12 investigation state in a single document:
observations, mechanism hypothesis, evidence, confirmed vs. conjectured
boundaries, candidate fix, and open items. To be consulted if the
investigation is revisited (paper discussion, reviewer response, co-author
onboarding, revival after a gap).

---

## 1. Context

T12 was a single-parameter generalization test: rerun the T11 configuration
with `mass_ratio` changed from 0.01 to 0.14, everything else held. The
goal was to determine whether the T11 closure (armature installation,
damping removal, `mapping_bypass_in_ss = True`, `swing_early_finish_fraction
= 0.80`) generalizes across mass ratio before committing to multi-step
scenarios (T15, T16).

The `mass_ratio` knob is not a runtime `SimConfig` field; it is realized
as the floating-structure body `mass` and principal `fullinertia` in
`models/VISPA_crawling_rwa3.xml`. In T12, structure mass was scaled by
1/14 (7110 → 507.86 kg) and principal inertias by the same factor
(preserving shape ratios 1.00 : 2.50 : 2.98). The MJCF was mutated
transiently and byte-exactly restored, mirroring the existing damping /
armature mutation pattern from T11.

## 2. Observation

| Phase | Metric | T11 (1%) | T12 (14%) |
|---|---|---|---|
| SS | torso_pos_peak | 36.0 mm | 34.8 mm |
| SS | torso_ori_peak | 1.05° | 1.12° |
| SS | ee_pos_peak | 32.0 mm | 32.8 mm |
| SS | ee_ori_peak | 9.37° | 9.34° |
| Dock | t_dock | 6.01 s | 6.01 s |
| Dock | dock_d | 2.70 mm | 0.86 mm |
| Dock | dock_ori | 0.06° | 0.10° |
| Dock | activation path | kinematic | kinematic |
| DS1 | \|h_w\| at t_dock | — | 0.24 Nm·s |
| DS1 | \|h_w\| at end-DS1 (t=25.91 s) | — | 6.28 Nm·s |
| DS1 | (hw_x, hw_y, hw_z) at end-DS1 | — | (−3.98, −4.55, −1.70) Nm·s |
| DS1 | torso_ori at end-DS1 | — | 9.92° |
| DS1 | structure attitude at end-DS1 | — | (+33.0°, +13.7°, +18.3°) |
| DS1 | ee_pos at end-DS1 | — | 0.007 mm |
| DS1 | ee_ori at end-DS1 | — | 0.063° |
| NMPC | failures over horizon | 0 | 1 (k=65, t=6.61 s, status=1) |
| QP | failures over horizon | 0 | 0 |
| NMPC | cost before / after k=65 | — | 265 → 73 |

The SS metrics and the dock event generalize from T11 to T12 with no
material change. DS1 behavior is qualitatively different: `|h_w|` grows
monotonically from 0.24 to 6.28 Nm·s, structure attitude grows to tens of
degrees, torso orientation error crosses the 5° gate at t ≈ 16 s. The
end-effector is kinematically clamped by the weld constraint throughout
DS1 (`ee_pos` and `ee_ori` errors under 0.1 mm / 0.1°).

The `|h_w|` trajectory during DS1 fits a piecewise-linear model with
breakpoint at t ≈ 16.4 s (R² = 0.9997, RMS = 29 µNm·s) with slopes
b₁ = +0.37 Nm·s/s before and b₂ = +0.26 Nm·s/s after. A single-segment
linear fit gives R² = 0.992 and b = +0.317 Nm·s/s. The exponential fit is
worse than the constant-mean baseline (R² = −0.82). Growth is linear,
not exponential.

## 3. Mechanism

### 3.1 Hypothesis

At the SS→DS transition (weld activation, t = 6.01 s), the NMPC torso
*position* reference steps discontinuously; the *orientation* reference
is continuous. The NMPC tracks the position step under the welded
kinematic chain, where arm joint motion reacts at the anchors, producing
sustained torque on the structure body. Structure angular velocity ramps
linearly, attitude grows quadratically; torso attitude — defined in
structure frame — drifts with the rotating structure. The AOCS observes
`h_w` growing (internal angular momentum conservation: structure body
gains momentum, wheels gain the opposite), commands counter-torques
saturated at ±5 Nm partial-duty, `|h_w|` grows linearly at ~0.32 Nm·s/s
until it reaches the ±5 Nm·s per-component box.

### 3.2 Evidence

Three converging lines, each from a different diagnostic:

**D1 — torso attitude reference is flat across SS→DS.** The logged
`q_torso_ref` components change by 0.0000°, 0.0000°, 0.0000° across the
SS→DS1 tick boundary (k = 58 → 59). Inspection of the torso planner
confirms the quintic reference has already saturated at its end-pose by
τ ≈ 5.82 s under `swing_early_finish_fraction = 0.80`, so both the last
SS tick and the first DS tick read the same saturated end-pose. The
~9.9° torso orientation error accumulating through DS1 is therefore a
*drift of the actual away from a held reference*, not tracking of a
reference step. See `figs_post/D1_reference_zoom.png` and
`D1_reference_full.png`.

**Q3 — the code path produces a position-only reference step.** Per
`sim_loop.py:1697–1725`, under `mapping_bypass_in_ss = True`:
- SS: `p_ref = self._ss_entry_p_torso` (frozen at SS-entry pose).
- DS: `p_ref = self.mapping.compute(q_current)` (live mapping at post-impact q).
`R_ref = torso_planner.reference_at(t)` in both phases. The position
reference discontinuously jumps at weld; the orientation reference does
not. The position step magnitude equals the full planned torso excursion
accrued in SS (~591 mm on the 14% case), because `_ss_entry_p_torso` is
held from SS-entry ~6 s earlier. This matches the discontinuities
visible on `e_p_torso` components in Fig 10 of the T12 tracking plots.

**D3 — `|h_w|` grows linearly, consistent with sustained reaction torque.**
R² = 0.992 for a single-segment linear fit, R² = 0.9997 piecewise. Linear
growth is the signature of an approximately constant commanded wheel
torque — inconsistent with a one-off impulse, consistent with a
persistent reference-vs-state mismatch driving continuous NMPC reaction.
Structure attitude grows quadratically over the same window, consistent
with constant angular acceleration from constant applied torque. See
`figs_post/D3_hw_growth.png`.

### 3.3 Why the mass ratio matters

The mechanism exists in principle at T11 (1%) as well: the position
reference stepped by a similar magnitude at SS→DS. At 1%, the structure
was 14× heavier and the reaction torque from a given arm-joint motion
produced ~1/14 of the angular acceleration. `h_w` at T11 end-DS1 was
noted in prior logs at ~2.7 Nm·s after one step, which is at 54% of the
box but does not saturate. At 14%, the same reaction produces 14× more
structure rotation, which couples back through the NMPC cost (torso
attitude error drives more aggressive arm commands) and through the
AOCS (more momentum to dump), pushing the accumulation into saturation
range within one step.

### 3.4 Role of `mapping_bypass_in_ss`

The flag was installed in T11 to close an EE-position inflation from the
SS-phase mapping loop. It works for its intended purpose (SS metrics at
14% are clean). The side effect, unobserved at 1% because the
downstream consequences were mild, is that the SS-frozen position
reference diverges from the mapping output over the full SS duration,
producing the large position-reference step at SS→DS. At 14% this
amplifies into the DS1 divergence above.

The flag was a closure on a T11 problem that produced a new problem at
T12 through a different mechanism. Removing the flag (Option B in §5)
re-opens the T11 problem; keeping it requires addressing the SS→DS
discontinuity differently.

## 4. Confirmed vs. conjectured

### Confirmed from T12 data

- SS metrics and dock event generalize across mass ratio under the T11
  configuration. The T11 fixes are not compromised by the 14% ratio.
- A discontinuous position-reference step exists at SS→DS under
  `mapping_bypass_in_ss = True`, traceable to `sim_loop.py:1697–1725`.
- `|h_w|` grows linearly through DS1 at ~0.32 Nm·s/s, reaching 6.28 Nm·s
  at end-DS1 (125% of per-component box on `hy`).
- Structure attitude grows quadratically over DS1 to tens of degrees.
- No attitude-threshold regime switches exist in the control code; the
  piecewise-linear breakpoint at t ≈ 16.4 s is continuous-dynamics
  coincidence, not a thresholded branch.
- `tau_w` logging at 10 Hz aliases the 100 Hz AOCS command; the observed
  factor-20 discrepancy between `∫τ_w dt` and `Δh_w` is a logging
  artefact, not a physics inconsistency. The underlying wheel-torque
  command and `h_w` response are consistent in sign and order of
  magnitude.

### Conjectured, not closed by current data

- **Generalization to N > 1 steps.** The mechanism is identified from a
  single-step run. Multi-step behavior introduces coupling not tested
  here: wheel-momentum carry-over between steps, structure-drift
  compounding (nominal scheduler anchor positions may not match actual
  anchor positions after several steps of cumulative drift), tracking
  error propagation through impact projections. The mechanism may
  dominate at N steps or be superseded by secondary mechanisms.
- **Completeness of the mechanism story.** The observations are
  consistent with the hypothesis, but single-step data cannot rule out
  co-active mechanisms of smaller magnitude. For example, the
  warm-start reset at weld (`sim_loop.py:1344`) could contribute to the
  NMPC cost regime change at k = 65 independently of the
  position-reference step.
- **Quantitative attribution.** The factor-20 logging aliasing prevents
  a direct closure of the momentum budget `Δh_w = ∫τ_w dt` from T12
  data. The mechanism is consistent qualitatively (signs agree, shapes
  agree, order-of-magnitude consistent); a quantitative closure
  requires rerun with AOCS-cadence logging or an `∫τ_w dt` accumulator.
- **The NMPC cost regime change at k = 65.** Cost drops from ~265 to
  ~73 at the status=1 solve, then stays in the ~70 band. The cause —
  warm-start reset effect, a different solution branch, active-set
  change — is not identifiable from logged data (IPOPT return string,
  iteration count, constraint violations are not persisted). The
  coincidence with DS1 mechanics is established; the causal structure
  is not.

## 5. Candidate fix (Option A) and alternatives

Three options for addressing the SS→DS position-reference step were
considered.

**Option A — Smooth the position-reference release.** Ramp `p_ref` from
`_ss_entry_p_torso` (last SS value) to `mapping.compute(q_current)`
(first DS value) over a post-dock window of duration T_ramp, using a
quintic shape function for C² continuity. The ramp magnitude is set by
the code; T_ramp is a new parameter. Starting point: T_ramp = 2 s
(~10% of DS1 budget). Localized to the phase-transition handler
(`sim_loop.py:1324–1361`) and the reference-selection branch
(`sim_loop.py:1697–1725`).

**Option B — Remove `mapping_bypass_in_ss`.** Let the mapping evolve
the position reference in SS, so that SS-exit and DS-entry references
differ only by the small jump due to contact-set change, not the full
swing excursion. Re-opens the T11 EE-position inflation issue that the
flag was introduced to close. Rejected unless T11 mechanism can be
closed differently first.

**Option C — Hold position reference at `_ss_entry_p_torso` through DS1
settle, then release to mapping.** Keeps SS behavior unchanged but
forces NMPC to track a stale pose for the entire DS1 duration. The
welded kinematic chain makes the frozen torso reference generally
inconsistent with the welded configuration; NMPC would drive sustained
reaction at the welds in steady state. A different form of the present
failure, not a fix.

**Selected: Option A.** Rationale: minimum-surgery; addresses the
specific trigger identified by the mechanism analysis; does not reopen
the T11 closure; introduces one new parameter (T_ramp) with a bounded
tuning range.

### 5.1 What Option A does not address

- Warm-start reset at weld. The reset remains; if the ramp makes the
  reference continuous, the reset may be unnecessary, but this is a
  separate investigation.
- Impact-projection velocity jump at weld. The inelastic projection onto
  the constraint null-space remains; it acts on `q̇`, not on the
  reference.
- Wheel-momentum carry-over across steps. Single-step Option A does not
  address how `h_w` is dumped between steps. Relevant for T15 onward.
- Structure drift across multiple steps. Option A does not address
  cumulative attitude drift; if T15 shows anchor position errors from
  accumulated drift, a separate fix is needed.

### 5.2 Pass criteria for T12-fix validation

- `|torso_ori|` at end-DS1 < 2° (T12 current: 9.9°).
- `|h_w|` at end-DS1 < 2 Nm·s per component (T12 current: up to 4.55 Nm·s).
- `|e_p_torso|` at end-DS1 within T11 envelope (< 5 mm).
- SS metrics unchanged from current T12 values within 10%.
- Zero NMPC failures (T12 current: 1).
- Dock event: `dock_d` < 3 mm, `dock_ori` < 0.5°, kinematic activation.

## 6. Open items

Items logged but not resolved in the T12 investigation. Listed for
future revisitation.

1. **`tau_w` logging at AOCS cadence.** Current 10 Hz log of the last
   AOCS sub-tick command aliases a 100 Hz bang-bang-saturated signal.
   An `∫τ_w dt` accumulator inside the AOCS sub-tick loop would close
   the momentum budget check. Not blocking T12-fix or T15.

2. **Piecewise-linear breakpoint at t ≈ 16.4 s.** Coincident with
   torso_ori crossing 5°; no corresponding code branch exists.
   Continuous-dynamics explanation candidates: (a) saturation-pattern
   shift as `h_w` approaches ±5 Nm·s box; (b) structure inertia-tensor
   principal-axes rotation changing effective gain. Not investigated
   further. Expected to disappear with Option A (if `|h_w|` no longer
   approaches the box, neither candidate applies).

3. **Warm-start reset at weld.** `sim_loop.py:1344` drops the NMPC
   warm-start at every weld activation. With Option A making the
   reference continuous, the reset may be unnecessary or even
   counterproductive. Revisit if T12-fix or T15 show unexpected
   first-DS-tick behavior.

4. **NMPC cost regime change at k = 65.** Cost drop 265 → 73 at
   status=1 solve, persisting post-recovery. Causal structure not
   identifiable from logged data. Expected to be affected by Option A
   (reference discontinuity was the likely driver); verifiable if
   T12-fix re-runs without a cost regime change.

5. **Scheduler advance semantics at dock.** The prior open item
   "freeze / retract / stop" from the session prologue was partially
   subsumed by this investigation: under Option A, the scheduler still
   advances at dock, but the reference is ramped rather than stepped.
   The architectural question of whether a settle-complete gate should
   be added before scheduler advance remains open. Relevant if T15
   shows between-step DS1 quality degradation.

6. **Post-abort DS divergence (H_DS1 in the session prologue).** The
   scheduler-advance mechanism was originally hypothesized as the H_DS1
   driver. T12 shows the dominant DS1 divergence is upstream of the
   scheduler, in the mapping-bypass release. Whether the original
   abort-semantics issue is a separate mechanism or a symptom of the
   same one is not determined by T12 data. Relevant if a dock failure
   occurs at T12-fix, T15, or T16.

7. **Logging of IPOPT return string, iteration count, per-constraint
   violations.** Currently reduced to a 3-code enum in `SimLog`.
   Adding the full solver stats would enable future investigations to
   diagnose NMPC solution-branch changes without inference. Not
   blocking T12-fix.

8. **Phase-dependent reference-source switching is a design weakness.**
   The current architecture uses the CoM→torso mapping in DS but
   bypasses it in SS (frozen SS-entry pose). T11 evidence showed the
   SS-phase mapping output was incompatible with swing kinematics (EE
   position inflation, closed by installing the bypass). T12 evidence
   shows the bypass itself generates a large SS→DS position-reference
   step whose downstream consequences (structure rotation, `h_w`
   saturation) are the dominant DS1 failure mode. Both problems are
   manifestations of the same underlying issue: the mapping does not
   produce a torso reference that is smooth and kinematically feasible
   across SS and DS. The Option A ramp is a tactical smoothing of one
   symptom; it does not eliminate the phase-dependent source. A
   follow-up investigation should address whether the mapping can be
   reformulated to be phase-invariant — for instance by taking phase
   as a continuous parameter, by producing a reference consistent with
   the upcoming weld configuration during SS, or by generating the
   torso reference directly from the planner with CoM tracking as a
   cost term rather than as a reference-generation step. Not blocking
   T12-fix, T15, or T16 under Option A; relevant for a v2 architecture
   and for paper §V exposition.

## 7. Paper implications (noted, not developed)

- **§V (mechanism / architecture).** The mapping-bypass release and the
  ramp smoothing are implementation choices, not central contributions.
  They should be documented in §VI (parameters and implementation) or
  in a dedicated subsection of §V.B on DS-phase reference handling. The
  paper's central claim remains momentum-aware NMPC preventing RWA
  saturation during locomotion; the ramp is a supporting design choice.
- **§VI (parameters).** `T_ramp` joins the list of parameters requiring
  justification. A sensitivity sweep at T12 with T_ramp ∈ [0.5, 1, 2,
  4] s would support a defensible range.
- **§VII (simulation results).** The T12-single-step-with-fix result is
  a pre-requisite, not the headline. The headline remains T15 (3-step,
  1%) and T16 (3-step, 14%) if they close. DS1 divergence analysis at
  T12 does not need to appear in §VII; it is internal to the
  investigation.
- **§IV (related work, Rognant et al.).** Not affected by T12. The
  offline-planning-context critique stands as calibrated previously.
- **Acta Astronautica vs. robotics venues.** The single-step sensitivity
  analysis across mass ratio (T11, T12) is useful evidence for the
  robotics-venue bar. Acta Astronautica submission remains the primary
  target; the additional data does not change the venue choice.

## 8. Next actions

1. Draft implementation spec for Claude Code: Option A in
   `sim_loop.py`, with T_ramp = 2 s as a new `SimConfig` field.
2. Rerun T12 at 14% with Option A. Pass criteria in §5.2.
3. Conditional on T12-fix passing: T15 (3-step, 1%).
4. Conditional on T15 passing: T16 (3-step, 14%).
5. Deferred: items in §6 as they become relevant.

---

**Artefacts referenced in this memo:**
- `results/M7_14pct_1step_v22_with_swing_hold/T12_report.md`
- `results/M7_14pct_1step_v22_with_swing_hold/T12_post_diagnostic.md`
- `results/M7_14pct_1step_v22_with_swing_hold/T12_post2_code_inspection.md`
- `results/M7_14pct_1step_v22_with_swing_hold/figs_post/D1_reference_zoom.png`
- `results/M7_14pct_1step_v22_with_swing_hold/figs_post/D1_reference_full.png`
- `results/M7_14pct_1step_v22_with_swing_hold/figs_post/D3_hw_growth.png`
- `results/M7_14pct_1step_v22_with_swing_hold/figs/Fig_9_ee_tracking.png`
- `results/M7_14pct_1step_v22_with_swing_hold/figs/Fig_10_torso_tracking.png`
- `results/M7_1pct_1step_v22_with_swing_hold/` (T11 reference)
