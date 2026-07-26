# `Misc/tests/` — retired test modules

Test modules that exercise features deliberately removed from `crawlbot/`.
They are kept rather than deleted so the intent behind the original checks
stays readable, but they are **not** part of the suite and are expected to
fail.

| module | tests | why retired |
|---|---:|---|
| `test_fk_reference_consistency.py` | 9 | imports `constrained_geodesic`,
deleted in CLEANUP-17; 8 of 9 tests need it via the `smoothed` fixture, and
the 9th validates an FK-mode run, a path removed from `sim_loop` in
CLEANUP-15. Full reasoning: `results/j2_adjconv/PHASE_CLEANUP_26_*.md`. |
| `test_trajectory_aware_ik.py` | 4 | |
| `test_ik_anomaly_regression.py` | 4 | |
| `test_mid_waypoint_reshape.py` | 3 | the three above exercise the **Option-B
manipulability-IK path** — `manipulability_config_trajectory`,
`manipulability_config_mid_waypoint`, `check_path_feasibility`,
`precompute_torso_map` — retired from `crawlbot/core/ik.py` in CLEANUP-30 with
zero callers and 0 lines executed by the canonical replay. Full reasoning and
the measurements: `results/j2_adjconv/PHASE_CLEANUP_30_*.md`. |

Its data fixture went with it: `Misc/runs/M7_1pct_3step_v22_t15_fk/`.

The Option-B trio took `Misc/tests/fixtures/step2_ss_entry_fixture.npz` with
them — they were its only consumers. **Note:** the 4th test of
`test_mid_waypoint_reshape.py`, `test_torso_planner_piecewise_continuous`, did
**not** retire; it exercises the live `TorsoPlanner` and was moved into
`tests/test_planners_6d.py`.

⚠ These modules are expected to fail on import: the functions they call no
longer exist. That is the point — they document what the retired path checked.
Reviving any of them means reviving its subject from git history first
(`git show d61e1a0:crawlbot/core/ik.py` is the last revision that has them).
