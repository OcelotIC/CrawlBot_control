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

## 3. Trap — kept as found, flagged for decision

**The ω̇_s carry is reset every NMPC tick.** In `SimulationLoop._step`,
`_omega_s_last` (→ `omega_s_prev`) and `tau_w_last` (→ `tau_w_prev`) are
initialised to **zero at every call of `_step`**, i.e. every 0.1 s. On the first
QP sub-step of each NMPC tick the numerical derivative is therefore
`(ω_s − 0)/dt`, and with the canonical `K_d = 25`, `dt = 0.01` the damping term
is `2500·ω_s` N·m — a 10 Hz kick that saturates the 2.5 N·m cap for
|ω_s| ≳ 1 mrad/s. The DS passivity loop, by contrast, seeds its history from the
entry ω_s (`command_interstep` receives it correctly). This refactor preserves
the behaviour (no behaviour change, Rule 6); whether it is intended is a
separate, measured decision.

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
