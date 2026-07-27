# `crawlbot.simulation.tick_logging`

The per-tick recorders, split out of `sim_loop.py` in CLEANUP-32.

---

## 1. Why this is a separate module

`SimulationLoop` is the controller; this is its telemetry. They are separated
because **they fail differently**: a mistake here corrupts a plot, a mistake
there corrupts the robot. Anyone chasing a control question can now skip 400
lines of `log.*.append`, and anyone chasing a missing channel knows which file to
open without grepping.

⚠ Not to be confused with **`logging.py`**, which defines `SimLog` — the
*container* these write into. This module holds the *writers*. Two files, two
jobs, unfortunately adjacent names; the distinction is container vs. writer.

## 2. What makes the split safe to verify

Both recorders are **pure with respect to control**: they read `self`, the
MuJoCo / Pinocchio state and their arguments, and write only to `log`.

Two calls do mutate — `mujoco.mj_forward` and `robot.update` — but both only
recompute derived quantities from unchanged `qpos` / `qvel`, which is why the
blocks were moved verbatim and in order rather than tidied on the way.

That property is what lets `gate/run_gate.py` settle any change here: if the
exported artifact is byte-identical over 2077 rows x 132 928 fields, the control
path provably did not move. The split was verified exactly that way, with all six
docks at delta +0.0000.

## 3. `TickState` — the boundary record

`_step` ends by handing over everything its recorder needs. Passing that
positionally would mean a **29-argument** signature — the same debt as
`WholeBodyQP.solve()`'s 40 parameters (`CLEANUP_CARRYOVER` A1). One record
instead, built once at the boundary.

Membership is **measured, not chosen**: a field is present iff the recorder reads
it before writing it. That test matters — `L_dot_est` and `R_err` look like
inputs, since `_step` assigns both, but the recorder *recomputes* them from
`rs_f`. Five names fell out of the naive set this way. `cfg` is excluded because
it is only ever `self.cfg`; `log` because it is the destination, not tick state.

Field names come from **where each value is logged**, not from `_step`'s
abbreviations:

| in `_step` | field | logged as |
|---|---|---|
| `lr` | `lambda_ref` | `log.lambda_ref` |
| `vp` | `v_com_ref` | `log.v_com_ref` |
| `cref_r` | `r_com_ref` | `log.r_com_ref` |
| `t_qp_start` | `t_qp_ms` | `log.qp_time_ms` — a result, not a stopwatch reading |

Groups: step context (12 fields), stage-1 NMPC outcome (8), stage-2 QP outcome
(6), torso reference actually tracked (1).

`p_torso_ref_used` is bound only when the QP sub-loop ran. The original code
discovered that by catching **`NameError`** inside the logging block; as a field
defaulting to `None` it is an ordinary `is None` test — identical behaviour, no
longer leaning on Python scoping to express control flow.

## 4. The mixin, and the three calls it does not own

`TickLoggingMixin` is mixed into `SimulationLoop`, so `self.*` resolves against
the finished class exactly as before the move. Three calls resolve there rather
than here:

`_get_ee_data`, `_gripper_distance`, `_swing_query_time`

They are geometry queries owned by the loop. Duplicating them to make this module
standalone would trade a little coupling for a lot of drift risk, which is a bad
trade — the CLEANUP-21 fixture miss is what that failure looks like.

## 5. DS and SS are not symmetric, and one asymmetry is real

`_log_ds_tick` (double support) predates `_log_ss_tick` by the whole chantier —
the SS recorder was inlined in `_step` until CLEANUP-31 lifted it out. That was
drift, not design, and the two are now peers.

One genuine difference remains: `_log_ds_tick` logs the **CoM reference as the
measured CoM**, so `e_com` is 0 by construction during DS. That is a logging
convention, not a control fact, and it produces a visible snap in the exported
reference at SS→DS entry. Decision pending on whether to apply the same
terminal-hold fix the torso export received (CLAUDE.md Known Issues,
`CLEANUP_CARRYOVER` A4).

A second asymmetry is **not** real and was closed by C2.2.1. Both recorders
solve QPs — `_step` ten per tick, `_run_ds_passivity_loop` one — but only the
SS side ever recorded anything about them, and even there `qp_ok` came from a
`try/except` that cannot fire (see `solvers/hierarchical_qp.md` §4). The DS
recorder simply hardcoded `log.qp_ok.append(True)`. Both now record through the
same accumulator, so the `qp_*` columns cover **all** 2077 canonical ticks and
all 8458 solves. The hardcoded `qp_ok = True` line stays only because `qp_ok`
is one of the 66 frozen columns; the comment there points at its replacement.

## 6. `QPStatAccumulator` — why the reduction lives here

The CSV is one row per logged tick, but a tick is not one QP solve: SS and
DS-terminal ticks each run `n_qp_per_nmpc` (=10) solves inside the WBC block,
the inter-step settle runs one. Per-solve records therefore have to be reduced
before they can be a column, and the reduction has to be **identical** in both
recorders or the column means different things in different phases — exactly
the drift §5 describes. One class, used by both, is the whole point.

It also fixes the status ordering in one place:

| code | meaning |
|---|---|
| `-1` | not measured — no solve was offered. Never an outcome |
| `0` | the backend reported success (`stats()['success']`) |
| `1` | the backend reported **not** success — the outcome `qp_ok` structurally cannot see, because `error_on_fail: False` means an infeasible QP returns instead of raising |
| `2` | the solve raised |

"Worst" is `max` over the tick, with `-1` ranking below `0`.

`add_raised()` is the path for a solve that threw before returning an info
object; it is **not exercised** on the canonical, which has zero QP failures
across all 8458 solves — the first measurement of that fact rather than an
artifact of a channel that could not say otherwise.

The accumulator is telemetry: nothing reads it back, and the canonical replay
is byte-identical on all 66 frozen columns with it in place.

---

## Code map

| unit | source |
|---|---|
| `class QPStatAccumulator` | [L47-112](../../../crawlbot/simulation/tick_logging.py#L47-L112) |
| `QPStatAccumulator.add` | [L81-100](../../../crawlbot/simulation/tick_logging.py#L81-L100) |
| `QPStatAccumulator.add_raised` | [L102-104](../../../crawlbot/simulation/tick_logging.py#L102-L104) |
| `QPStatAccumulator.as_dict` | [L106-112](../../../crawlbot/simulation/tick_logging.py#L106-L112) |
| `class TickState` | [L124-199](../../../crawlbot/simulation/tick_logging.py#L124-L199) |
| `class TickLoggingMixin` | [L202-641](../../../crawlbot/simulation/tick_logging.py#L202-L641) |
| `TickLoggingMixin._log_ds_tick` | [L205-430](../../../crawlbot/simulation/tick_logging.py#L205-L430) |
| `TickLoggingMixin._log_ss_tick` | [L432-641](../../../crawlbot/simulation/tick_logging.py#L432-L641) |

---

## See also

- the controller: [`sim_loop.md`](sim_loop.md)
- the container these write into: [`logging.md`](logging.md)
- package overview: [`simulation.md`](simulation.md)
