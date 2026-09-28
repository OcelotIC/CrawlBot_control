# `crawlbot.simulation`

**The closed loop and its tuning surface.**

| file | lines | canonical coverage | document |
|---|---:|---:|---|
| `sim_loop.py` | 2431 | — | [sim_loop.md](sim_loop.md) |
| `plant.py` | 199 | — | [plant.md](plant.md) |
| `sensors.py` | 128 | — | [sensors.md](sensors.md) |
| `config.py` | 507 | **100 %** | [config.md](config.md) |
| `logging.py` | 269 | 93 % | [logging.md](logging.md) |
| `plotting.py` | 154 | 2 % | [plotting.md](plotting.md) |

## Role

`sim_loop` is the orchestrator: the gait sequencer (DS/SS, docking IK ->
pre-planner -> planners -> swing -> settle), weld activation under the 5 mm /
5 deg gate, the single simulation loop `_drive`, and the log. The control law
itself is in [`control/`](../control/control.md).

`plant.py` is the MuJoCo side of the loop — the only writer of simulator state
(step, welds, dock impact, actuators). `sensors.py` is its read side — one
method per measurement channel; the control path touches MuJoCo only through
these two.

`config.py` is the single tuning surface — rule 5 of the project. `logging.py`
produces the `sim_log.json` that every downstream analysis reads.

## Three things to know before reading a log

1. **`nmpc_ok = 0` means "not called"**, not "failed" — 1368 of 2077 ticks.
2. **`H_rO`, `H_dot_est` and `gmo_contact_state` carry no signal** (the objects
   are constructed and logged but never updated).
3. **Dock precision is the at-weld value** in `dock_events`, never the minimum
   over the swing — the two differ by 40 % on step 2.

## Main debt

The `refactor/sim-loop-split` chantier cut `sim_loop.py` into plant / sensors /
controller / orchestrator and made `_drive` the single simulation loop (see
[sim_loop.md](sim_loop.md) §1). `WholeBodyQP.solve()` still takes 40 parameters,
but only `control/controller.py` calls it now (`CLEANUP_CARRYOVER` A1).
