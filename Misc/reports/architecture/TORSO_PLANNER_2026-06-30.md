# TorsoPlanner — as-it-runs (2026-06-30, orientation-reference fix IMPLEMENTED)

Authoritative description of `crawlbot/planning/torso_planner.py` as it runs today, including the
**orientation-reference fix** (the torso orientation reference is now sourced from a single stored
`R_flat = R_torso(t=0)`, held globally and FSM-independently, instead of being re-sampled from live torso state
at every phase entry).

**Supersedes** for the torso-reference question the pre-J2 `Misc/runs/j2_audit/INTERNAL_j2_torsoplanner.md`
(`ae0673e`, DS-mobile framing). `Misc/reports/architecture/SS_TORSO_GUIDANCE_AUDIT_2026-06.md` covers only the
position/CoM reference (δ_com); `docs/architecture/IK_FORMULATION.md` (2026-06-23) does not document the
torso orientation-hold choice. Root-cause diagnosis: `Misc/runs/j2_canonical_revalidation/ORI_CHAIN_CONTINUITY_DIAG.md`.

---

## 1. What the planner is (machinery + accessors)
A **neutral, FSM-agnostic** trajectory/hold container in the **structure body frame**. It stores a time-windowed
phase list and a single static hold; it has **no** notion of DS/SS/EXT, stance, or step. Three query accessors,
each called per tick by `sim_loop`:
- `reference_at(t) → TorsoReference(p, R, v, a)` (`:439`) — 6-DoF torso pose + twist + accel. **`R` is the
  orientation reference.**
- `com_reference_at(t) → ComReference(r_com, v_com)` (`:450`) — position/CoM reference (`r_com = p + R·δ_com`).
- `l_com_reference_at(t) → (3,)` (`:475`) — NMPC angular-momentum feedforward.

Reference construction:
- **`add_phase(...)`** (`:195`) appends a time window with `(p_start, R_start, p_end, R_end, δ_com_*)`;
  `reference_at` inside the window **SLERPs** `R = R_start·exp3(s·log3(R_startᵀR_end))` (`_interpolate_phase`,
  `:630`). With `R_start == R_end` this is a **constant** `R`, ω = α = 0.
- **`set_hold(p, R, r_com)`** (`:125`) is the **degenerate static case** — `reference_at` outside all phases
  returns this hold (`_hold_reference`, `:532`). Used for DS.
- **FK mode** (`reference_source='joint_space_fk'`, `_add_phase_fk`/`_reference_at_fk`): `R` from FK on a
  smoothed geodesic `q(τ)`. **Dormant** in the canonical run (`reference_source` default `'task_space'`,
  `config.py:509`).

The planner **never re-anchors orientation itself** — `R_start`/`R_end`/`_hold_R` are caller inputs.

## 2. How `sim_loop` drives it (SS-trajectory vs DS-hold)
- **SS**: `_setup_torso_for_step` (`:1382`) clears phases, sets a hold, and adds one swing phase over
  `[t_ss_start, t_ss_start+T_step]`. The dock-IK (`dock_configuration_fixed_rotation`, `:1485`) solves the end
  config `q_end` with the torso orientation **pinned** to a fixed `R`, so `R_t1 == R` (the SS orientation SLERP
  span was **≡ 0** even pre-fix — the dock-IK pin made `R_t1 = R_t0`; the SS reference has always been a
  constant orientation hold, never an advancing dock-to-dock SLERP).
- **DS** (inter-step settle, DWELL, terminal): `set_hold(...)` with a static pose (`:2125`, `:2505/2517/2524`,
  and the DWELL `set_from_waypoints` `:2117`). The QP tracks `R_ref` per tick (`:2581`); the inter-step settle
  (`_run_ds_passivity_loop`, `:601`) runs before the next `_setup_torso_for_step`.

## 3. The orientation-reference fix (implemented)
**Defect (CASE B, `618ddcf`):** pre-fix, `sim_loop` sourced the orientation from the **live** torso state
`R_t0 = rs_s.oMf_torso.rotation` (`:1411`) at every SS entry, feeding it to the dock-IK pin, the hold, and the
swing end. Within each inter-step DS the torso drifts ~0.11–0.19° off the fixed hold; the next SS entry
**re-captured that drifted current**, jumping the reference by the drift and re-zeroing `e_torso_ori` — the
sawtooth. The realized torso was physically continuous (≤0.013°/tick); only the **reference** jumped
(`ORI_CHAIN_CONTINUITY_DIAG.md`: per-seam reference jump 0.14–0.21°, plus a one-time 5.16° init).

**Fix:** capture `R_flat = R_torso(t=0)` ONCE at the start of `run()` (structure frame; `:1973`) and source it
for the orientation at **every** site, instead of live `R_t0`:
- the initial seed hold (`run()` start), so even the initial DS references `R_flat`;
- the SS dock-IK pin `R_torso_fixed = self._R_torso_flat` (`:1485`) ⇒ `R_t1 = R_flat`;
- the SS hold `set_hold(p_t0, self._R_torso_flat, …)` (`:1685`);
- the SS swing phase `add_phase(…, self._R_torso_flat, …, R_t1, …)` (`:1753`) — `R_start = R_end = R_flat`, so
  the SS orientation reference is the **constant** global `R_flat`;
- all DS holds (DWELL `:2117/2125`, terminal `:2505/2517/2524`).
Position/CoM held-setpoint logic (`p_t0`, `r_com`) is **unchanged** — orientation target only.

**Why `R_flat = R_torso(t=0)`, not analytic identity:** the torso is mounted with a yaw offset
(rpy ≈ 0, 0, **−5.16°** in the structure frame); forcing identity would snap that offset at tick 0. `R_flat`
holds the real mounting orientation, imposing no frame convention. (The one-time pre-fix 5.16° step-0 jump WAS
this offset vs the pre-capture identity reference; the seed removes it.)

**Why this is continuous by construction:** a reference sourced from a single stored value never jumps,
regardless of in-DS tracking drift. The torso (which may drift sub-degree off `R_flat` during a swing) is
driven back to `R_flat` by the QP tracking the constant reference — the swing **closes** the residual drift
instead of the next phase **accepting** it.

> **Note (deviation from the brief's literal sub-instruction, flagged):** the brief suggested `add_phase`
> `R_start = current realized, R_end = R_flat` (a SLERP that closes the drift, span ≠ 0). That reintroduces a
> residual-drift seam at DS→SS (reference jumps `R_flat`→current) and keeps the `e_torso_ori` sawtooth. To meet
> the brief's **primary** goal ("a reference that jumps is bad; zero seam"), `R_start = R_flat` was used
> instead (reference = `R_flat` everywhere, span ≡ 0, seam ≡ 0); the QP closes the residual drift by tracking
> the constant `R_flat`. Switchable to the literal SLERP variant on request.

## 4. Results (canonical 6-step, fixed vs pre-fix)
| metric | pre-fix | **fixed** |
|---|---|---|
| per DS→SS reference jump | 0.14–0.21° (+5.16° init) | **0.000° (all 6 seams)** |
| `e_torso_ori` across seams | sawtooth (resets to ~0) | **continuous** (= true drift vs `R_flat`) |
| torso drift vs `R_flat` final | ~0.72° (permanent, uncorrected) | **0.038° world / 0.112° struct** (returns to `R_flat`) |
| torso drift vs `R_flat` peak | ~0.72° | 0.524° world / 0.579° struct (transient, swing) |
| docks (6) | [4.94,4.45,4.94,4.65,4.84,4.89] | **[4.94,4.41,4.88,4.42,4.76,4.92]** (6/6 ≤5mm) |
| envelope ‖Ḣ_s‖∞ SS ticks>5 | 0 | **0** |
| QP feasibility | feasible | **0 QP fails / 1080** |
| h_w∞ | 4.930 | **4.885** |

⇒ The orientation hold is now global and FSM-independent; the sawtooth is gone; the torso returns to `R_flat`
(non-accumulating); no regression (docks/envelope/feasibility/h_w unchanged-or-better). C5 (h_w∞ > 4.5) remains
the separate pre-existing open item.

## 5. Reorientation freedom vs the wrench null space (clarification, per the audit)
At a docked (welded double-stance) config the **motion** reorientation freedom is `null([Jc; J_com]) = 5-D`
(= 8 weld-internal motions − 3 CoM-transport = ~3 torso-orientation + ~2 arm-posture;
`Misc/runs/j2_audit/INTERNAL_j2_torsoplanner.md` Q4). **`P_int = I − G⁺G` (`wholebody_qp.py:1170`) is NOT this
space** — it is the 12-D contact-**wrench** internal-stress null space (a force object, the hyperstatic wrench
redundancy resolved by the internal-stress regularizer). The orientation hold uses the **torso-orientation**
portion of the 5-D motion freedom (the planner outputs a torso-orientation reference; it has no arm-posture
reference output).

## 6. Stale debt (unchanged by this fix; flagged for later cleanup)
- **Dead code** `_trapezoidal_params` (`:547`) — never called (`_profile_params`→`_quintic_params`); its
  comments cite the reverted "planned-δ mapping (v19)".
- **Stale comments** in `_profile_params` (`:606`) reference CoM cruise-shaping that is OFF (`a_cruise_max=0`).
- **Legacy torso-only `l_com_reference_at`** (`:475`) — ~20% limb-error formula, active in the canonical
  (task_space) run (FK exact-momentum path dormant); retire if FK becomes canonical.
