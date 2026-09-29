# `crawlbot.control.attitude`

**File**: [`crawlbot/control/attitude.py`](../../../crawlbot/control/attitude.py) — **281 lines** — canonical coverage **92 %**

**The AOCS: reaction-wheel torque for the free-floating platform.** Extraction
3a of the `sim_loop.py` split (branch `refactor/sim-loop-split`). The control
*laws* stay in [`aocs/force_estimator.py`](../aocs/force_estimator.md); this
module is the controller block that selects the law, assembles its inputs
from measurements, and returns a torque.

---

## 1. Contract

In: platform gyro ω_s, attitude quaternion and wheel momentum h_w (read through
[`SensorSuite`](../simulation/sensors.md)); the robot's centroidal state `rs`;
the QP contact wrench λ_qp. Out: τ_w, clipped to `cfg.aocs_tau_w_max`. The
previous tick's values its finite differences need are the AOCS's **own
state** (`AocsHistory`, below), not an argument. It never writes MuJoCo — the
caller applies τ_w through [`MujocoPlant`](../simulation/plant.md).

| entry point | loop | law |
|---|---|---|
| `command()` | NMPC-tracked ticks (SS, DWELL, trailing DS) | `legacy_pid_numerical` + FD (SS) or wrench (DS) feedforward |
| `command_interstep()` | inter-step DS passivity loop | `legacy_pid_numerical` + DS wrench FF |

Canonical mode: `legacy_pid_numerical`,

```
τ_w = ff + K_hw·(clip(h_w) − h_w) + K_θ·θ_s + K_ω·ω_s + K_d·(ω_s − ω_s,prev)/dt
θ_s = ½ vee(R_errᵀ − R_err),   R_err = R_initᵀ R_now          (Lee–McClamroch)
ff  = −L̇_com − r_com × m·v̇_com                    (SS: FD on centroidal state)
ff  = −Σ_i (r_Ci × f_i + τ_i) from λ_qp           (DS: wrench feedforward)
```

## 1b. The history belongs to the AOCS (A0)

`AocsHistory` = (ω_s,prev, τ_w,prev, L_com,prev, v_com,prev). It used to live in
the controller's per-NMPC-tick `QPCarry` (and, for the settle, in the loop's DS
state), which is why the restart defect of §3 was invisible from here. Now:

| call | effect on the history |
|---|---|
| `reset_for_nmpc_tick(rs)` | ω,τ := 0; L,v := `rs` — **the known defect, in one place** |
| `reset_for_settle()` | ω := entry ω_s (τ, L, v unused on that path) |
| `command(...)` | reads the previous tick, then records this one (after the `disable_aocs` override) |
| `command_interstep(...)` | reads ω_s,prev, then records this tick's ω_s |

The `disable_aocs` diagnostic override moved here with it (it must act before
the history records τ_w). Frozen on synthetic inputs by
`tests/test_attitude_controller.py`; a fix of §3 changes
`test_nmpc_tick_reset_is_the_known_defect` on purpose.

## 1c. M0 instrumentation (logging only)

With `cfg.log_hifreq_all`, `command()` also evaluates the same pure law with the
clip lifted and keeps it in `last_tau_w_preclip` (never applied); the loop
records every plant step in `SimulationLoop.hifreq_trace` — τ_w commanded
(post-clip, what is applied), the pre-clip value, qs, phase, h_w, ω_s. The
10 Hz log only ever records sub-step `qs=9`, which cannot see the `qs=0` kick of
§3. Default off; proven inert (canonical bit-identical with the trace on).
Measurement script: `scripts/diag_m0_aocs_hifreq.py`.

## 2. Why the move is byte-identical

Both blocks were moved by **text slicing** (re-indented, scope variables renamed
with each rename asserted), not retyped. Verified with `gate/local_ref.py check`
+ `gate/dock_check.py`.

**R1 — one law left.** `command()` used to select among six `aocs_mode`
branches (plus `aocs_off_in_ds`); only `legacy_pid_numerical` is used by the
canonical and by the paper's Table 2, so the other branches, their laws and the
H_{r/O} estimator were retired. `AttitudeController` now raises on any other
`aocs_mode`. The removal is proven inert by coverage (no removed line executed
by the canonical or the Table 2 scenarios) and by bit-identity
(`results/j2_adjconv/PHASE_R1_AOCS_MODES_RETIRED.md`).

## 3. Trap — kept as found, flagged for decision

**The ω̇_s history is reset every NMPC tick** (`reset_for_nmpc_tick`, called
by `WholeBodyController.begin_tracking`): ω_s,prev and τ_w,prev restart at
**zero** every 0.1 s, and L_com,prev / v_com,prev at the current state (so the
SS FD feedforward is 0 on that sub-step). On the first
QP sub-step of each NMPC tick the numerical derivative is therefore
`(ω_s − 0)/dt`, and with the canonical `K_d = 25`, `dt = 0.01` the damping term
is `2500·ω_s` N·m — a 10 Hz kick that saturates the 2.5 N·m cap for
|ω_s| ≳ 1 mrad/s. The DS passivity loop, by contrast, seeds its history from the
entry ω_s (`reset_for_settle`). This refactor preserves
the behaviour (no behaviour change, Rule 6); whether it is intended is a
separate, measured decision.

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`AocsHistory`** *(dataclass)* |  |  | [L34](../../../crawlbot/control/attitude.py#L34) |
|   `omega_s_prev` | `` | _field_ | [L38](../../../crawlbot/control/attitude.py#L38) |
|   `L_com_prev` | `` | _field_ | [L39](../../../crawlbot/control/attitude.py#L39) |
|   `v_com_prev` | `` | _field_ | [L40](../../../crawlbot/control/attitude.py#L40) |
| **`AttitudeController`** |  |  | [L43](../../../crawlbot/control/attitude.py#L43) |
| `.reset_for_nmpc_tick` | `(rs)` | **yes** | [L71](../../../crawlbot/control/attitude.py#L71) |
| `.reset_for_settle` | `()` | **yes** | [L85](../../../crawlbot/control/attitude.py#L85) |
| `.command` | `(phase, rs, lambda_qp_sol, cc_nmpc, stance_anchors)` | **yes** | [L93](../../../crawlbot/control/attitude.py#L93) |
| `.command_interstep` | `(rs, cc_ds, lambda_qp_sol)` | **yes** | [L205](../../../crawlbot/control/attitude.py#L205) |

## Code map

| unit | source |
|---|---|
| `class AocsHistory` | [L34-40](../../../crawlbot/control/attitude.py#L34-L40) |
| `class AttitudeController` | [L43-280](../../../crawlbot/control/attitude.py#L43-L280) |
| `AttitudeController.reset_for_nmpc_tick` | [L71-83](../../../crawlbot/control/attitude.py#L71-L83) |
| `AttitudeController.reset_for_settle` | [L85-91](../../../crawlbot/control/attitude.py#L85-L91) |
| `AttitudeController.command` | [L93-203](../../../crawlbot/control/attitude.py#L93-L203) |
| `AttitudeController.command_interstep` | [L205-280](../../../crawlbot/control/attitude.py#L205-L280) |

---

## See also

- the laws: [`aocs/force_estimator.md`](../aocs/force_estimator.md)
- package overview: [`control.md`](control.md)
