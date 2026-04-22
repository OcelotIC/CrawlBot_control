# T12-post-2 — Code-only inspection of three open mechanisms

Scope: read-only inspection of the crawlbot control/sim code to characterise
three mechanisms left open by T12-post. No runs, no edits, no commits.
All line numbers refer to the checkout at the current HEAD.

---

## Q1 — `log.tau_w` unit/scaling path and saturation

### Q1.1 Source of `tau_w` written to the log

The T12 run uses the legacy-corrected AOCS path (`cfg.aocs_use_legacy_corrected`
= True, `cfg.aocs_mode != 'H_est'`). The command is produced by:

- `crawlbot/aocs/force_estimator.py:286` — `compute_aocs_command_legacy_corrected(...)`
- Formula assembled at `force_estimator.py:370`:

  ```
  tau_w = -L_dot_est - (r_com × m · v_com_dot) + K_hw · hw_error
  ```

  where `L_dot_est` is a finite-difference derivative of `L_com` at the AOCS
  cadence (`dt_qp = 0.01 s`), `K_hw` is the momentum-saturation feedback gain,
  and the "orbital" term `r×mv̇` compensates the translational contribution.

- Clip at `force_estimator.py:371`:

  ```
  return np.clip(tau_w, -tau_w_max, +tau_w_max)      # tau_w_max = 5.0 Nm
  ```

  Units are **Nm** throughout; the function returns a saturated torque
  command directly in SI units.

### Q1.2 Where it is applied to MuJoCo

`crawlbot/simulation/sim_loop.py:1902–1950` — AOCS sub-tick block. The
command is written to MuJoCo at `sim_loop.py:1948`:

```
self.mj_data.ctrl[self.robot.n_joints:self.robot.n_joints+3] = tau_w_cmd
```

Immediately after, `self._tau_w_last = tau_w_cmd.copy()` caches the value
for logging.

### Q1.3 MuJoCo actuator scaling

`models/VISPA_crawling_rwa3.xml:320–322`:

```
<motor name="act_rw_x" joint="rw_x" gear="1" ctrlrange="-5 5"/>
<motor name="act_rw_y" joint="rw_y" gear="1" ctrlrange="-5 5"/>
<motor name="act_rw_z" joint="rw_z" gear="1" ctrlrange="-5 5"/>
```

`gear="1"` means the applied joint torque equals `ctrl` exactly. `ctrlrange`
matches `tau_w_max = 5.0 Nm` (`crawlbot/simulation/config.py:62`), i.e. there
is **no further scaling** between the Python-side saturation and the wheel.

Wheel joints themselves (`VISPA_crawling_rwa3.xml:117/125/133`) carry
`damping="1e-4"` and `armature="0.01"`; the effective rotor inertia is
`I_w = 0.01 kg·m²` (consistent with `rwa_I_w` in config and the comment at
`VISPA_crawling_rwa3.xml:108`: `h_w_max = I_w · ω_max = 0.01 · 500 = 5 Nms`).

### Q1.4 What the log actually stores

`crawlbot/simulation/sim_loop.py:2066–2073` (log-tick branch, every
`n_qp_per_nmpc = 10` physics steps):

```
log.tau_w.append(self._tau_w_last.copy())
```

So `log.tau_w[k]` is the **last AOCS command of the 10-sub-tick window
ending at log tick k**. It is a snapshot, not a window average.

### Q1.5 Saturation accounting

Saturation happens once, inside `compute_aocs_command_legacy_corrected` at
`force_estimator.py:371` (`np.clip(..., ±5.0)`). Nothing downstream clips
again; MuJoCo will silently clamp to `ctrlrange` if a value outside ±5 ever
reached `ctrl`, but it never does because the Python side already clipped.

### Q1.6 Diagnostic divergence scale factor — raw numbers

From `T12_post_diagnostic.md` (D3):

| quantity               | value              |
|------------------------|--------------------|
| Δh_w over DS1          | ≈ 1.0 Nms          |
| ∫ log.tau_w dt, DS1    | ≈ 20 Nms           |
| sign-discrepancy ratio | ≈ factor 20        |

The integrated logged torque is ≈ 20× the observed wheel-momentum change.

### Q1.7 Hypothesis (factual statement — no interpretation beyond mechanism)

There is **no unit mismatch or scaling factor in the code path**: the Python
controller outputs Nm, clips at ±5 Nm, and MuJoCo applies Nm directly
(`gear=1`). The factor-20 discrepancy between `∫ log.tau_w dt` and the
measured `Δh_w` is therefore **not a transformation error**. It is
consistent with a **sampling/aliasing artefact** in how `tau_w` is logged
relative to how it is applied:

- AOCS runs at 100 Hz (`dt_qp = 0.01 s`), so 10 torque commands are applied
  per log tick.
- `log.tau_w` at 10 Hz records only the **last** sub-tick command (not a
  mean). With a rapidly oscillating, frequently-saturated signal — and
  `L_dot_est` is a centred/finite-difference derivative of `L_com`, which
  amplifies per-step noise by 1/dt_qp = 100 — the last-sample aliased series
  does **not** integrate to the true time-integral of the applied torque.
- An oscillating saturated command of peak ±5 Nm whose mean is ≈ ±0.5 Nm
  will log as typical magnitude ≈ 5 Nm (saturated end of window) and
  integrate to 10 · 5 · dt = 0.5 Nms per 0.1 s, producing an apparent
  ∫τ dt that is ≈ 10–20× the true momentum change.

Factor ≈ 10–20 matches `n_qp_per_nmpc = 10` and the partial-duty saturation
pattern: this is consistent with aliasing of a bang-bang-like τ_w, not a
scaling bug.

No code change is warranted on this evidence alone; the fix (if one is
wanted) would be to log `tau_w_mean_over_window` or to accumulate
`∫τ_w dt` inside the AOCS sub-tick loop.

---

## Q2 — Attitude thresholds (grep for 5° / 9° regime switches)

### Q2.1 Enumerated literals near 5° / 9°

Grep over `crawlbot/`, `scripts/`, `models/`:

| location                                            | literal            | role                                                                 |
|-----------------------------------------------------|--------------------|----------------------------------------------------------------------|
| `crawlbot/simulation/config.py:40`                  | `5.0` deg          | `dock_ori_threshold_deg` — **dock gate**, binary                     |
| `crawlbot/simulation/config.py:44`                  | `5.0` N            | `gmo_F_threshold` — contact-force threshold, **not an angle**         |
| `crawlbot/simulation/config.py:53-54`               | `±5.0` Nms         | `hw_min / hw_max` — wheel momentum saturation, not an angle           |
| `crawlbot/simulation/config.py:62,67`               | `5.0` Nm           | `tau_w_max`, `aocs_tau_w_max` — wheel torque saturation, not an angle |
| `models/VISPA_crawling_rwa3.xml:320-322`            | `"-5 5"`           | motor `ctrlrange` — Nm, not degrees                                  |

No literal `9.0` (deg or otherwise) appears in any control-loop path.
The only `9`-adjacent constants are physical parameters unrelated to angle.

### Q2.2 Use of `dock_ori_threshold_deg`

`crawlbot/simulation/sim_loop.py` — one and only one use site in the dock
gate: both the position criterion (`d < 5 mm`) **and** the orientation
criterion (`ori_err < 5°`) must hold simultaneously for weld activation to
fire (`_activate_weld`, `sim_loop.py:1324–1361`). This is a **one-shot
binary event** (weld flag latches true), not a regime switch or gain
schedule.

### Q2.3 Explicit absence of gain scheduling

Three comments in `sim_loop.py` **explicitly** state that M7 removed any
approach/EXT regime or angle-based scheduling:

- `sim_loop.py:308-310` — "M7 single-QP architecture: no gain scheduling,
  no approach thresholds"
- `sim_loop.py:1198-1200` — same note at the SS-entry block
- `sim_loop.py:1641-1642` — same note at the sub-tick loop

There is a single `qp_ss` instance used throughout SS; the earlier EXT /
approach QP variants are removed from the code path.

### Q2.4 Factual conclusion

No attitude threshold triggers controller mode changes. The only attitude
threshold present is the 5° dock-gate criterion, which fires a one-shot
weld event (SS → DS transition). There is no 9° threshold anywhere in the
code. Any apparent regime change around 5°–9° attitude error in T12 logs
must therefore originate from continuous dynamics (trajectory saturation,
NMPC warm-start reset at weld, or impact-projection velocity jump), not
from a thresholded code branch.

---

## Q3 — `mapping_bypass_in_ss` release path and SS → DS reference discontinuity

### Q3.1 Flag declaration and docstring

`crawlbot/simulation/config.py:227-235`:

```
mapping_bypass_in_ss: bool = False
"""Diagnostic override: if True, in SS the torso linear reference is
frozen at the SS-entry value instead of being driven by the CoM→torso
mapping. DS uses the mapping unconditionally. Purpose: isolate whether
SS-phase instability comes from the mapping loop itself."""
```

Position reference is the **only** quantity affected; orientation
reference comes from `torso_planner.reference_at(t)` in both SS and DS
(see Q3.3).

### Q3.2 SS-entry capture

`crawlbot/simulation/sim_loop.py:935-939`, at the SS-entry branch:

```
p_t0, R_t0 = <torso pose at SS entry>
self._ss_entry_p_torso = p_t0.copy()
self._ss_entry_R_torso = R_t0.copy()
```

Only the **position** `_ss_entry_p_torso` is consumed by the bypass logic;
`_ss_entry_R_torso` is retained for diagnostics but not used to override
the orientation reference.

### Q3.3 Reference selection per phase

`crawlbot/simulation/sim_loop.py:1697–1725`, inside the per-tick task
assembly:

```
# Orientation: identical in both phases
R_ref = tr.R                          # from torso_planner.reference_at(t)

# Position: phase-dependent
if phase == 'SS' and cfg.mapping_bypass_in_ss:
    p_ref = self._ss_entry_p_torso    # frozen at SS entry
else:
    p_ref = self.mapping.compute(     # r_b_ref = ratio·r_com_ref - δ/m_b
        r_com_ref=tr.r_com, ..., q_current=q_mj
    )
```

So with `mapping_bypass_in_ss = True`:

- **SS**: position = frozen `_ss_entry_p_torso`; orientation = planner.
- **DS (after weld)**: position = `mapping.compute(q_current)`; orientation = planner.

### Q3.4 Transition event sequence

`sim_loop.py:1324–1361` — `_activate_weld(...)` handler, invoked the tick
the dock gate latches true:

1. `mujoco.mj_forward(...)` — consolidate kinematics after weld.
2. `self.nmpc.reset_warm_start()` — NMPC warm-start dropped.
3. Inelastic impact: velocity projected via `Λ⁻¹` onto the constraint
   null-space.
4. `self._weld_active = True`; subsequent ticks take the DS branch at
   `sim_loop.py:1697–1725`, i.e. position flips from
   `_ss_entry_p_torso` to `mapping.compute(q_current)`.

Trailing DS planning is appended by
`torso_planner.set_hold(p_end, R_end, r_com_end)` at `sim_loop.py:1410–1431`,
where `p_end / R_end / r_com_end` are the post-dock IK equilibrium pose.

### Q3.5 Reference discontinuity at SS → DS (factual statement)

- **Position reference**: discontinuous. The last SS tick returns
  `_ss_entry_p_torso` (the pose at *SS entry*, ≈ 6 s earlier). The first
  DS tick returns `mapping.compute(q_current)` evaluated at the
  post-impact `q`. These differ by the full planned torso excursion
  accrued in SS (≈ 591 mm on the M7 14 % case per T12 D1), so the
  closed-loop position-error input to the QP **steps** at weld.
- **Orientation reference**: continuous in construction. `R_ref` comes
  from `torso_planner.reference_at(t)` in both phases. The quintic
  TorsoPlanner phase with `early_finish_fraction = 0.80` finishes at
  τ = 5.82 s and the dock occurs at t ≈ 6.01 s, so both the last SS tick
  and the first DS tick read `R_ref ≈ R_end` from the already-saturated
  quintic. There is no programmed orientation-reference step at the
  transition.
- **Warm-start**: NMPC warm-start is explicitly reset at weld
  (`sim_loop.py:1344`), so the first DS NMPC solve starts from a cold
  initial guess even if the reference happened to be smooth.
- **Velocity**: impact-projection at `sim_loop.py:1355` instantaneously
  modifies `qd` via `Λ⁻¹`, independent of the reference.

The only reference-side discontinuity introduced by
`mapping_bypass_in_ss` at SS → DS is in **position**; orientation is
continuous through the transition.

---

## Files inspected (read-only)

- `crawlbot/aocs/force_estimator.py` (lines 243, 286–371)
- `crawlbot/simulation/sim_loop.py` (lines 308–310, 935–939, 1198–1200,
  1324–1361, 1410–1431, 1641–1642, 1697–1725, 1902–1950, 2066–2073)
- `crawlbot/simulation/config.py` (lines 40–67, 227–235)
- `crawlbot/core/com_to_torso_mapping.py` (lines 133–170)
- `crawlbot/planning/torso_planner.py` (lines 206–213)
- `models/VISPA_crawling_rwa3.xml` (lines 108, 117/125/133, 320–322)

No source files were modified. No simulations were executed. No commits
or pushes were made.
