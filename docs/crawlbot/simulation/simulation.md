# `crawlbot.simulation`

**The closed loop and its tuning surface.**

| file | lines | canonical coverage | document |
|---|---:|---:|---|
| `sim_loop.py` | 3387 | 83 % | [sim_loop.md](sim_loop.md) |
| `config.py` | 507 | **100 %** | [config.md](config.md) |
| `logging.py` | 269 | 93 % | [logging.md](logging.md) |
| `plotting.py` | 154 | 2 % | [plotting.md](plotting.md) |

## Role

`sim_loop` is where the architecture actually happens: the DS/SS state machine,
the per-step sequence (docking IK -> pre-planner -> planners -> swing -> settle),
weld activation under the 5 mm / 5 deg gate, the AOCS command, and the log.

`config.py` is the single tuning surface — rule 5 of the project. `logging.py`
produces the `sim_log.json` that every downstream analysis reads.

## Three things to know before reading a log

1. **`nmpc_ok = 0` means "not called"**, not "failed" — 1368 of 2077 ticks.
2. **`H_rO`, `H_dot_est` and `gmo_contact_state` carry no signal** (the objects
   are constructed and logged but never updated).
3. **Dock precision is the at-weld value** in `dock_events`, never the minimum
   over the swing — the two differ by 40 % on step 2.

## Main debt

`_step()` is 1013 lines and `WholeBodyQP.solve()` takes 40 parameters. Both
decompositions are identified, scoped in `CLEANUP_CARRYOVER` section A, and
deliberately not done yet — each needs its own coupling measurement first.
