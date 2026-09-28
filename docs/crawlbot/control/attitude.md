# `crawlbot.control.attitude`

**File**: [`crawlbot/control/attitude.py`](../../../crawlbot/control/attitude.py) — **306 lines** — canonical coverage **78 %**

**The AOCS: reaction-wheel torque for the free-floating platform.** Extraction
3a of the `sim_loop.py` split (branch `refactor/sim-loop-split`). The control
*laws* stay in [`aocs/force_estimator.py`](../aocs/force_estimator.md); this
module is the controller block that selects the law, assembles its inputs
from measurements, and returns a torque.

---

## 1. Contract

In: platform gyro ω_s, attitude quaternion and wheel momentum h_w (read through
[`SensorSuite`](../simulation/sensors.md)); the robot's centroidal state `rs`;
the QP contact wrench λ_qp; the per-tick carry (previous L_com, v_com, ω_s,
τ_w). Out: τ_w, clipped to `cfg.aocs_tau_w_max`. It never writes MuJoCo — the
caller applies τ_w through [`MujocoPlant`](../simulation/plant.md).

| entry point | loop | law |
|---|---|---|
| `command()` | NMPC-tracked ticks (SS, DWELL, trailing DS) | every `cfg.aocs_mode` branch |
| `command_interstep()` | inter-step DS passivity loop | `legacy_pid_numerical` + DS wrench FF |

Canonical mode: `legacy_pid_numerical`,

```
τ_w = ff + K_hw·(clip(h_w) − h_w) + K_θ·θ_s + K_ω·ω_s + K_d·(ω_s − ω_s,prev)/dt
θ_s = ½ vee(R_errᵀ − R_err),   R_err = R_initᵀ R_now          (Lee–McClamroch)
ff  = −L̇_com − r_com × m·v̇_com                    (SS: FD on centroidal state)
ff  = −Σ_i (r_Ci × f_i + τ_i) from λ_qp           (DS: wrench feedforward)
```

## 2. Why the move is byte-identical

Both blocks were moved by **text slicing** (re-indented, scope variables renamed
with each rename asserted), not retyped — including the six `aocs_mode` branches
the canonical never executes, which the bit-identity gate cannot see. Verified
with `gate/local_ref.py check` + `gate/dock_check.py`.

## 3. The per-NMPC-tick history restart — measured, fix available, OFF

**Legacy behaviour (canonical).** The QP carry is re-created every NMPC tick
(0.1 s). On QP sub-step 0 the AOCS therefore sees `ω_s,prev = 0` — a
numerical ω̇_s of `ω_s/dt`, i.e. `K_d·ω_s/dt = 2500·ω_s` N·m with `K_d = 25` —
and, in SS, `L_com,prev = L_com`, `v_com,prev = v_com` (the current state), so
the FD feedforward `−L̇_com − r×m·v̇_com` is zero on that sub-step. The DS
settle path is unaffected: it seeds its own history from the entry ω_s.

**Measured** (`scripts/diag_aocs_kick.py`, canonical C run, instrumentation
proven inert — `sim_log.json` bit-identical to the reference; one-step
counterfactual validated by an exact 0.0 match on the 6 381 sub-steps 1–9):

| sub-step 0 (709 ticks) | legacy | with the previous tick's history |
|---|---:|---:|
| \|K_d·Δω_s/dt\|∞ median / max | 1.18 / 4.47 N·m | 0.007 / 0.12 N·m |
| \|τ_w − τ_w,true-history\|∞ median / max | 1.36 / 3.91 N·m | — |
| wheel commands at the 2.5 N·m cap | 103 | 20 |

83 of the run's 368 saturated wheel commands are this artefact.

**Fix — `cfg.aocs_carry_across_nmpc_ticks` (default False).** The controller
keeps the last control tick's `(ω_s, L_com, v_com, τ_w)` on BOTH paths (NMPC
sub-step and DS settle) and `begin_tracking` seeds sub-step 0 with it, i.e. the
same sampling as sub-steps 1–9. Off ⇒ byte-identical to the frozen canonical.

**Closed loop with the fix ON — not adoptable as is:**

| | legacy (canonical) | fix ON |
|---|---:|---:|
| at-weld d [mm] | 4.016 / 4.888 / 4.990 / 4.973 / 4.954 / 4.624 | 4.017 / 4.891 / **4.997** / **4.996** / 4.953 / 4.587 |
| θ_s peak | 0.540° | **0.873°** (+62 %) |
| h_w peak axis / norm [N·m·s] | 4.10 / 4.24 | 3.85 / 3.94 |
| ω_s peak | 1.79 mrad/s | 1.53 mrad/s |
| saturated wheel commands | 368 | 287 |

The kick is gone (sub-step-0 K_d term median 1.18 → 0.007 N·m), but θ_s grows
by 62 % and steps 3–4 dock 3–4 µm inside the 5 mm capture radius. **The
canonical AOCS gains (K_θ = 1, K_ω = 50, K_d = 25) were tuned with the artefact
in the loop**; θ_s = 0.54° depends on it. Adopting the fix means re-tuning
(one gain at a time, Rule 12) and re-freezing the canonical — a decision, not a
default. Full record: `results/j2_adjconv/PHASE_AOCS_CARRY.md`.

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`AttitudeController`** |  |  | [L32](../../../crawlbot/control/attitude.py#L32) |
| `.command` | `(phase, rs, lambda_qp_sol, cc_nmpc, stance_anchors, L_co...)` | **yes** | [L46](../../../crawlbot/control/attitude.py#L46) |
| `.command_interstep` | `(rs, cc_ds, lambda_qp_sol, omega_s_prev)` | **yes** | [L235](../../../crawlbot/control/attitude.py#L235) |

## Code map

| unit | source |
|---|---|
| `class AttitudeController` | [L32-305](../../../crawlbot/control/attitude.py#L32-L305) |
| `AttitudeController.command` | [L46-233](../../../crawlbot/control/attitude.py#L46-L233) |
| `AttitudeController.command_interstep` | [L235-305](../../../crawlbot/control/attitude.py#L235-L305) |

---

## See also

- the laws: [`aocs/force_estimator.md`](../aocs/force_estimator.md)
- package overview: [`control.md`](control.md)
