# PHASE CLEANUP-26 — the two unblocking fixes

Two items, both chosen because they unblock other people rather than because
they clean anything: the repository could not be *installed* on a fresh
container, and the test suite could not be *run* end to end.

Both were long-standing. Neither was hard. That is the point.

---

## 1. `setup_env.sh` now installs a working pinocchio

**Top of the carry-over ledger (§C1) for the entire chantier, and never done.**

`setup_env.sh` pinned `pin==3.9.0` but not its cmeel ABI dependencies. Those are
unpinned upstream, so a fresh resolve pulls `cmeel-urdfdom 6.x` /
`cmeel-tinyxml2 11.x`, and `import pinocchio` then dies on a missing shared
object. **Nothing in the repository runs** — not a test, not the gate, not the
canonical replay.

It has been fixed by hand in every session for months and never committed, which
means every new container repeated the same discovery.

### The fix, and why these exact majors

```bash
$PIP "cmeel-urdfdom~=4.0" "cmeel-tinyxml2~=10.0"
```

Not guessed from the versions that happened to be installed — confirmed against
the linker's own record for pin 3.9.0's compiled extension:

```
liburdfdom_sensor.so.4.0  =>  .../cmeel.prefix/lib/.../liburdfdom_sensor.so.4.0
liburdfdom_model.so.4.0   =>  .../liburdfdom_model.so.4.0
liburdfdom_world.so.4.0   =>  .../liburdfdom_world.so.4.0
libtinyxml2.so.10         =>  .../libtinyxml2.so.10
```

`~=4.0` and `~=10.0` are exactly the majors those sonames require. And
`pip --dry-run` resolves them to the versions currently working (4.0.1 / 10.0.0).

The pins were also added to the **error hint** the script prints when its own
import check fails — that message is where a reader lands when this breaks, so it
is the one place the fix most needs to be visible.

---

## 2. The test suite collects end to end again

`tests/test_fk_reference_consistency.py` imported six symbols from
`crawlbot/planning/constrained_geodesic.py`, deleted in CLEANUP-17. That is a
pytest **collection error**, not a test failure: it aborts the entire run. The
suite has therefore been un-runnable end to end since CLEANUP-17, and nobody
noticed because the suite is not gated.

### Retired, not repaired — and the measurement says so

| what | finding |
|---|---|
| 8 of its 9 tests | take the `smoothed` fixture, which calls `smoothed_constrained_geodesic` → the deleted module |
| the 9th (`test_E7_t15_step2_dock_under_fk_mode`) | validates a sim_log from an **FK-mode** run — a path removed from `sim_loop` in CLEANUP-15 |
| its generator script | `run_m7_v22_1pct_3step_t15_fk.py`, already retired under `Misc/scripts/` |

So the whole module tests a feature that was deliberately removed. Repairing it
would have meant **restoring the feature**. Retiring it is the coherent action.

```
tests/test_fk_reference_consistency.py  ->  Misc/tests/
results/M7_1pct_3step_v22_t15_fk/       ->  Misc/runs/
```

The fixture directory follows because this module was its **only** consumer —
leaving it would contradict the CLEANUP-21 rule that `results/` holds only
load-bearing artifacts. 21 citations rewritten; `Misc/tests/README.md` records
why the module is there and that it is expected to fail.

### Result

```
before:  228 tests collected, 1 error   -> Interrupted, whole run aborted
after :  228 tests collected            -> no error
```

`results/` is now down to three tracked directories: `j2_adjconv`, `j2_figdata`,
`hero_render`.

---

## 3. What this does *not* fix

The 12 pre-existing suite problems are untouched — the collection error was the
one that stopped the run, not one of the failures. After this pass:

```
9 failed, 196 passed, 4 skipped, 3 errors
```

Same set as measured in CLEANUP-25, now reachable **without** an `--ignore` flag.
The 3 errors are `test_mid_waypoint_reshape`'s `FileNotFoundError` — fixtures for
another path removed in CLEANUP-15, so the same class as §2 and probably the same
verdict, but it deserves its own look rather than being swept along.

The standing recommendation still holds: either the suite becomes a gate and gets
driven to green, or it is explicitly labelled advisory. The current middle state
is exactly what let it drift from a documented "2 failures" to a measured 12
without anyone noticing.

---

## 4. Verification

```
pytest collection      228 tests, 0 errors   (was: 1 error, run aborted)
pytest full            9 failed, 196 passed, 4 skipped, 3 errors — unchanged set
ldd on pin 3.9.0       liburdfdom_*.so.4.0, libtinyxml2.so.10 — pins match
pip --dry-run          cmeel-urdfdom 4.0.1, cmeel-tinyxml2 10.0.0
bash -n setup_env.sh   syntax OK

gate/sync_docs.py --check     in sync
gate/verify_params.py         14/14 rows OK
gate/verify_docs.py           33 documents OK
gate/link_audit.py            0 BROKEN BY MOVE
```

`crawlbot/` was **not** touched by this pass, so the canonical run cannot have
moved; the four structural checkers cover what did change.
