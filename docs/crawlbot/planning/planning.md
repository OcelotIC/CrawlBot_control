# `crawlbot.planning`

**Reference generation.**

| file | lines | canonical coverage | document |
|---|---:|---:|---|
| `coarse_preplanner.py` | 540 | 81 % | [coarse_preplanner.md](coarse_preplanner.md) |
| `torso_planner.py` | 480 | 81 % | [torso_planner.md](torso_planner.md) |
| `contact_scheduler.py` | 350 | 87 % | [contact_scheduler.md](contact_scheduler.md) |
| `swing_planner.py` | 337 | **95 %** | [swing_planner.md](swing_planner.md) |
| `sequence_loader.py` | 254 | 0 % — kept | [sequence_loader.md](sequence_loader.md) |
| `locomotion_planner.py` | 205 | 17 % — kept | [locomotion_planner.md](locomotion_planner.md) |

## How a step is produced

```
ContactScheduler.plan_traversal()       DS/SS/DS skeleton (SS durations = 0)
        |
CoarsePrePlanner.solve()                T_step + feasible CoM trajectory
        |
GaitPlan.set_step_duration(idx, T_step) install duration, cascade the timeline
        |
TorsoPlanner.add_phase()  +  SwingPlanner over the same horizon
        |
                                        references for the QP
```

**The key point: step duration is computed, not chosen.** Move the CoM by `d` in
time `T` and you generate transverse momentum of order `m*d/T`. For a given reach
and a given wheel capacity there is therefore a minimum feasible `T`, specific to
that step's geometry. The pre-planner computes it; everything downstream is built
over the horizon it returns.

That is also why the torso and swing phases are installed *after* the solve and
share one horizon — synchronising them by construction rather than patching a
mismatch later.

## Package audited end to end

CLEANUP-16 through 19: 3258 -> 2175 lines (-33 %), `constrained_geodesic.py`
(470 lines) deleted, the swing phase-override mechanism removed.

Two files are kept despite near-zero coverage, for reasons that are not about
the canonical run — see their documents.
