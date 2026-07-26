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

Its data fixture went with it: `Misc/runs/M7_1pct_3step_v22_t15_fk/`.
