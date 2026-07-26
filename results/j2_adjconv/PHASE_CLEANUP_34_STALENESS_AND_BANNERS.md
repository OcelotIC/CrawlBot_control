# PHASE CLEANUP-34 — close the staleness hole, then the free readability wins

Plan order 1 → 3 → 2: finish what CLEANUP-33 started, take the zero-risk
readability items, then open the PR.

The pass is worth reading for one reason: **the safeguards built in step 1 caught
a real regression in step 3, within the same session.**

---

## 1. The staleness hole, everywhere it existed

CLEANUP-33 fixed `replay_canonical.py` and `run_gate.py`. Auditing the rest:

| checker | exposure | verdict |
|---|---|---|
| `cov_replay.sh` | wraps the replay under `coverage` with `set -euo pipefail` | **already safe** once the replay exits non-zero |
| `run_suite.py` | already fails when pytest writes no junit | **already safe** |
| **`dock_check.py`** | takes a log path; the 6/6 count catches a *truncated* log, nothing caught a stale *complete* one | **HOLE — fixed here** |

### `dock_check.py`

Now prints the log's age unconditionally, and refuses a log that predates its
companion verdict by more than one gate run:

```
log: results/gate_run_scratch/sim_log.json   ticks=2077
     written 21:53:36 (0.1 min ago)
```

The budget comes from `seconds_total` in the gitignored timed verdict (the
tracked one omits timings to stay byte-stable), falling back to 10 minutes.

⚠ The first version of this check was **wrong and I caught it by running it**: I
compared the log against the verdict's mtime and failed if the log was older.
But the verdict is written at the *end* of a gate run and the log *during* it, so
the log is always older — the check failed every legitimate run. The invariant is
not "newer than the verdict", it is "within one gate run of it". Fixed, then
proven in both directions:

```
fresh log  (0.1 min gap)   -> exit 0, CANONICAL RESULTS: MATCH frozen 2.5
stale log  (44.3 min gap)  -> exit 1, *** STALE LOG ***
```

### The duration note in `run_gate.py`

The CLEANUP-33 false PASS ran in **80.4 s** against an observed 127–142 s band.
That was visible on screen and nobody was looking at it, so the gate now says it:

```
[1] canonical replay + export : replay rc=1 (78.6s), export rc=None
    ^ NOTE: 78.6s is below the observed band (100-200s for a full 6-step
      traversal). Verify the replay actually ran to completion.
```

**Reported, not enforced.** Wall-clock is machine-dependent and a hard floor
would fail honest runs on faster hardware. Freshness and identity do the
enforcing; this exists to be noticed.

---

## 2. The readability items

### Banners — `_step` had **1** across 851 lines

Four phase banners, placed at the boundaries the coupling profile identified
rather than wherever felt right:

```
PHASE 0  — read state and references          coupling 0, accumulating
STAGE 1  — centroidal NMPC (dt 0.1 s, once)   contact cfg, solve, fallback
STAGE 2  — whole-body QP sub-loop (dt 0.01 s) the 615-line `for qs`
HAND OFF — telemetry                          TickState -> _log_ss_tick
```

Each banner says what the block does *and* what a reader should know about it —
that the NMPC fallback is canonical-unreached by design, that the QP loop keeps
~25 live locals so no single cut extracts it.

### A module map in the docstring

`sim_loop.py` is ~3000 lines and four methods are most of it. The map routes by
question rather than by structure — *"where is X logged?" → NOT HERE,
tick_logging.py* — and names the two structural debts so a reader does not
rediscover them.

### `_planned_arm_config` — and the mistake that followed

37 lines, zero callers repo-wide. Removed.

Its four `_step_*` attributes appeared to strand: an AST scan of
`SimulationLoop` showed `_planned_arm_config` was their **only reader**, so I
removed the eight assignment sites too.

**That broke the canonical replay.**

```
[replay] main() raised: AttributeError: 'SimulationLoop' object has no
         attribute '_step_q_start'
```

`scripts/diag_cooperative_arms.py:510-511` reads `sim._step_q_start` and
`sim._step_q_end` to build a per-step q-log — and `dca.main()` **is** the
canonical driver. The orphan scan was scoped to one class in one file; the reader
lived in `scripts/`. Same instrument-too-narrow error the chantier keeps logging,
and note that a repo-wide grep is exactly what I *did* do for the method name and
did *not* do for the attributes.

Restored, with a comment at the declaration saying who actually reads them so the
next scan does not repeat it. The method stays deleted — its zero-caller status
was verified repo-wide.

### The safeguards earned their keep immediately

The regression was caught **because step 1 landed first**:

```
[1] replay rc=1 (78.6s)
    ^ NOTE: 78.6s is below the observed band
[2] artifact identity : FAIL  (replay exited 1)
VERDICT: FAIL
```

Before CLEANUP-33, that same break would have produced `rc=0`, a stale log, and
a **PASS**. The hole was open for the entire chantier; the first regression after
closing it was mine, one hour later.

---

## 3. Verification

```
gate/run_gate.py     VERDICT: PASS   replay rc=0 (128.0 s)
                     artifact identity PASS — 2077 rows × 132 928 fields
gate/dock_check.py   log written 0.1 min ago
                     docks 6/6 — every step delta +0.0000, MATCH frozen 2.5
gate/run_suite.py    PASS — 200 tests, 199 passed, 1 xfail, 93 s
sync_docs --check / verify_docs / verify_roots / link_audit    OK
verify_params        bit again: sim_loop.py:921 -> :951 after the banners
                     shifted the QP-construction literals. 15/15 rows OK.
```

`sim_loop.py` 3010 → 3027 lines: −54 from the dead method, +71 of banners and
map. Longer, and considerably easier to read — which is the trade this pass was
for.

---

## 4. What this changes about the method

Three faults in two passes, all of the same family:

| pass | claim | why it was wrong |
|---|---|---|
| 32 | "the split is inert, gate PASS" | gate validated a stale artifact |
| 33 | "log must be newer than the verdict" | verdict is written *after* the log |
| 34 | "these four attributes strand" | scan covered one file; reader was in `scripts/` |

The pattern is not carelessness about the answer — each was measured. It is
**scope**: a stale-data check that never considered absent data, a freshness rule
that never considered write order, an orphan scan that never considered other
directories. The measurement was right and the frame was too small.

The practical rule this yields, and it is cheap: **before trusting a scan, state
what it does not look at.** For an orphan scan that is other files, other
directories, `getattr`, and string-keyed access. For a gate, it is a missing run.
