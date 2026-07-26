# AOCS architectural concern — Mode B transport term gap

**Status.** Open architectural item. Not blocking T15 / T16.
**Date.** 2026-04-22.
**Context.** T12 closure via `aocs_off_in_ds = True` (commit `bcd3d7c`).
**Purpose.** Record the architectural concern that the T12 fix
routes around rather than resolves, and define the conditions under
which the underlying fix becomes non-deferrable.

---

## 1. The concern

The T12 closure disables AOCS during DS phases via `aocs_off_in_ds`.
Mode B (`compute_aocs_command_legacy_corrected`) remains active
during SS. The underlying bug identified in §9.4 / §10.5 of
`M7_T12_MEMO.md` — a missing frame-rotation transport term in
Mode B's feedforward — is not fixed by this closure. The controller
continues to ship with the bug; the fix exploits the fact that the
bug's dynamical consequences are small in the regime where Mode B
still runs.

This is a route-around, not a repair. The concern is that the
reasons the bug is quiet in SS may not hold across all scenarios
we care about, and the diagnostic coverage for detecting the bug
waking up is currently indirect.

## 2. Why Mode B works in SS despite the bug

Mode B commands `τ_w = −L̇_com_est − r_com × m·v̇_com_est + K_hw · (clip(h_w) − h_w)`.
The feedforward terms are derived in the structure body frame. The
correct inertial derivative would include a transport term
`ω_s × H_{r/O}` that is present in Mode C (`force_estimator.py:166`)
and absent in Mode B.

The transport term contributes a wheel-torque bias proportional to
`ω_s`. In SS, three factors keep it quiescent:

1. **Small ω_s.** When Mode B is running correctly in SS, structure
   attitude stays well controlled and ω_s remains small (at
   T11/T12 mass ratios, ω_s peaks at order 10⁻² rad/s during
   swing). The bias scales with ω_s, so in well-behaved SS it is
   small.

2. **Dominant real signal.** The intended feedforward terms
   (`−L̇_com − r_com × m·v̇_com`) are large in SS because the arm
   is moving and producing real centroidal momentum rate. The
   transport bias, being small, is drowned out by the real signal.

3. **Mass-ratio favorability in single-step.** Even if the bias
   contributes some wheel torque, the resulting structure
   rotation per unit wheel torque scales as `1/I_struct`. At 1%
   (M1) the structure is 14× heavier than at 14% (M14), so the
   same bias wheel torque produces ~1/14 the rotation.

The bug becomes dangerous when any of these three protections
weaken.

## 3. Where the protections could weaken

### 3.1 Multi-step structure drift (highest-probability concern)

After DS1 in T12 with `aocs_off_in_ds=True`, structure attitude
drifts ballistically by ~2.8° over 20 s (D4 ballistic prediction,
confirmed). At the start of step 2's SS, ω_s is not zero — it's
the ballistic-drift value, ~2.3 mrad/s at 14%.

Mode B activates at SS entry with `ω_s(k=SS_entry) ≠ 0`. The
transport bias is proportional to ω_s, so SS step 2 begins with
a feedforward already biased from the first tick. The bias does
not grow unbounded during SS because the real signal dominates,
but it seeds the arm-motion controller with an incorrect
reference.

Expected magnitude at 14% × 3 steps: cumulative ballistic drift
reaches ~8.4° by end of step 3. If this feeds back through NMPC
tracking, SS metrics could degrade step-over-step in a way not
visible at 1% or at single-step 14%.

### 3.2 Mass-ratio amplification in multi-step

Even if the bias per step is small, the `1/I_struct` amplifier
compounds across steps. T16 (3 steps × 14%) is the product of
both risk factors: each step contributes ballistic drift that
seeds the next step's SS, and the amplifier is at its highest.

### 3.3 Wheel-momentum accumulation into the deadband feedback

The feedback term `K_hw · (clip(h_w) − h_w)` is zero when h_w is
inside ±5 Nm·s. Per-swing h_w deposit in T12 was ~0.24 Nm·s. At
14% × 3 steps the cumulative h_w could approach 0.7–1 Nm·s — still
inside the box, but not by much. If a 14%-scenario variant pushed
h_w outside the box, the deadband feedback would activate, and
its coupling with the buggy feedforward has not been
characterized.

### 3.4 Scenarios where ω_s is not small by design

The mission involves pointing maneuvers that intentionally slew
the structure. Slewing means ω_s ≠ 0 by design. In a pointing-
maneuver-during-assembly regime, Mode B's transport bias would
be a first-order effect, not a residual. This is outside the
current simulation scope but is in scope for the paper's mission
claims.

## 4. What we've done about it

### 4.1 T12 closure (short-term)

`aocs_off_in_ds = True` disables Mode B in the regime where the
bug was dominating the failure. This is verified correct at
single-step 14%: structure rotation dropped from 33° to 2.8°
(ballistic), h_w from 6.28 to 0.22 Nm·s, zero NMPC failures.

### 4.2 Solver instrumentation (permanent)

`SimLog` now carries `nmpc_status_str` and `nmpc_iterations`
(commit `bcd3d7c`). This makes any future NMPC-side failure
diagnosable without new instrumentation cycles. It does not
directly monitor the AOCS transport term; it monitors downstream
NMPC health, which is the first place an AOCS issue would
typically show up as a secondary effect.

### 4.3 Disposition in the paper

Per §10.4 of `M7_T12_MEMO.md`: document the phase gate as a
design choice, note the transport-term fix as v2 future work.
This is defensible for Acta Astronautica but understates the
issue for a robotics-venue readership.

## 5. What we have not done

1. **No direct AOCS-side instrumentation.** The transport-term
   magnitude `|ω_s × H_{r/O}|` is not logged per tick. We cannot
   verify post-hoc whether the bias stays small in SS during
   T15 / T16 without recomputing from other logged fields (which
   is possible but adds cross-checking work per run).

2. **No mass-ratio sensitivity analysis on the bias.** We have
   not run a sweep of mass ratios with Mode B active and ω_s
   forced nonzero to characterize the bias magnitude vs. ω_s
   and I_struct. T11/T12 are two points; a third would let us
   extrapolate toward the threshold where the bias becomes
   first-order.

3. **No transport-term fix in Mode B.** Adding
   `+ ω_s × H_{r/O}` to `compute_aocs_command_legacy_corrected`
   is ~2–3 lines of code. Deferred because the route-around
   closes T12 and the fix would perturb SS behavior in ways that
   require re-validation across T11 (currently closed) and T12
   (currently closed under the route-around).

## 6. Tripwires: when the concern upgrades to action

The route-around ships until one of the following is observed.
Each would indicate that the bug's quiescence assumption is
breaking down in a scenario we care about.

### Tripwire A — SS metric degradation across steps

T15 or T16 shows monotonically worsening SS metrics (`torso_pos_peak`,
`torso_ori_peak`, `ee_pos_peak`, `ee_ori_peak`) step 1 → step 2 →
step 3, beyond the 10% envelope established by T11/T12 single-step
metrics. Mechanism candidate: §3.1 step-over-step seeding of the
feedforward bias.

**Action if triggered:** pause T15/T16. Add the transport-term fix
to Mode B. Re-validate T11, T12 (both mass ratios, single-step) to
confirm the fix does not regress; then re-run T15/T16.

### Tripwire B — h_w approaches the box in SS

SS h_w reaches |h_w_i| > 3 Nm·s on any axis during any SS phase of
T15 or T16. Mechanism candidate: the transport bias contributing
to h_w beyond the real signal, approaching a regime where the
deadband feedback would activate.

**Action if triggered:** pause. Instrument `|ω_s × H_{r/O}|` and
rerun to quantify the bias contribution. If bias accounts for
> 20% of the commanded wheel torque, fix Mode B before proceeding.

### Tripwire C — NMPC failures in SS

Any `nmpc_status != 0` event during SS of T15/T16 beyond the known
single-dock-tick max_iter event. Mechanism candidate: feedforward
bias driving NMPC into a state region it cannot linearize around.

**Action if triggered:** read `nmpc_status_str` (available from the
new logging) to characterize the failure. If the failure mode is
a new one (not `Maximum_Iterations_Exceeded`), treat as
investigation-level issue.

### Tripwire D — Paper-level claim requires pointing maneuver

When the paper's §VII simulation scenarios expand to include
explicit pointing-maneuver-during-assembly cases (per §4 bullet 3,
the mission context requires this), Mode B's transport-term fix
becomes non-deferrable regardless of T15/T16 behavior.

**Action if triggered:** fix Mode B before running any
pointing-maneuver scenarios.

## 7. Proposed lightweight monitoring for T15 / T16

Before running T15, add one-line instrumentation to the AOCS block
at `sim_loop.py:~1949`:

```python
# AOCS diagnostic: log the transport-term magnitude even when the
# term is not actively applied by Mode B, for post-hoc monitoring
# of the route-around (see AOCS_CONCERN memo §7).
if self.has_rwa:
    omega_s_log = self.mj_data.qvel[3:6]
    H_rO_log = rs.L_com + np.cross(rs.r_com, m_robot * rs.v_com)
    transport_log = np.cross(omega_s_log, H_rO_log)
    log.transport_term_mag.append(float(np.linalg.norm(transport_log)))
```

(Exact placement and variable names subject to code conventions;
`transport_term_mag: list` field added to `SimLog`.)

This lets us plot `|ω_s × H_{r/O}|` per tick for any future run
and verify it stays below some threshold (e.g., 10% of commanded
τ_w) across SS phases. If the ratio starts rising step-over-step,
tripwire A is evidenced with data.

**Decision:** add this instrumentation before T15, or defer.
Recommendation: add. Cost is near-zero, value is a permanent
monitoring channel that catches the tripwire conditions
automatically rather than requiring post-hoc reconstruction.

## 8. Summary

- T12 closure ships a route-around, not a fix.
- The underlying Mode B bug is quiescent in SS at tested mass
  ratios (1%, 14%) in single-step. Multi-step has not been
  tested.
- Three mechanisms could wake the bug up: step-over-step
  ω_s seeding, mass-ratio amplification stacking with step
  coupling, deadband feedback activation near h_w = 5 Nm·s.
- Four tripwires define when the bug upgrades from deferred to
  blocking.
- Recommended next action: add `|ω_s × H_{r/O}|` logging
  before T15, then proceed with T15 as planned.
- If T15 and T16 show quiescent transport-term magnitudes and
  stable SS metrics, the route-around is empirically justified
  for the current paper scope. If either tripwire fires, fix
  Mode B before completing the validation campaign.

---

**Cross-references:**
- `Misc/reports/architecture/M7_T12_MEMO.md` §§9–10 (T12 closure, mechanism).
- Commit `bcd3d7c` (T12 closure bundle).
- `Misc/runs/M7_14pct_1step_v22_with_swing_hold_optA/T12_post3_d5_cost_reconstruction.md`
  (D5 cost analysis).
- `force_estimator.py:166` (Mode C transport term).
- `force_estimator.py:286` (Mode B, transport term absent).
