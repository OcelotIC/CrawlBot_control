# `gate/` — the CLEANUP chantier reproduction & consistency gate

**The gate exists before the first broom stroke.** No cleanup change lands on the
chantier without this gate green before *and* after. It is roadmap item 1 of
`docs/architecture/PORT_SYNTHESIS.md`: convert silent regression into an error.

## Run it

```bash
PYTHONPATH=. python3 gate/run_gate.py
```

One command → four checks → one verdict (`PASS`/`FAIL`, exit 0/1) plus a
machine-readable `gate/last_verdict.json`.

## The four checks

1. **Canonical replay** — re-runs the frozen *managed* (C) scenario into a
   scratch dir (`gate/replay_canonical.py` → `results/gate_run_scratch/`), then
   exports the 66-column fulldiag CSV from it. No committed artifact is touched.
2. **Artifact identity** — field-by-field vs the committed baseline
   `results/j2_adjconv/c25_fulldiag.csv`. Byte-identical on all reproducible
   columns = PASS; the two wall-clock timing columns are excluded (see
   `EXCEPTIONS.md`). Reports the first mismatching row/column otherwise.
3. **Two-model consistency** — loads the MJCF plant (`models/VISPA_crawling_rwa3.xml`)
   and the URDF controller model (`models/VISPA_crawling_fixed.urdf`) and diffs
   the BIN A-a2 hand-duplicated quantities (per-link composite mass / COM /
   principal inertia — no-joint child bodies like `tool_a` are lumped into their
   parent link so MuJoCo's separate tool body matches Pinocchio's absorbed fixed
   joint — plus total mass, joint order + limits, tool/torso frame placements)
   and the BIN B-b1 naming contract. Any disagreement FAILs with the quantity
   named.
4. **Environment pin** — compares the live versions against
   `gate/environment.lock`; WARNs on mismatch (bit-identity is meaningless on an
   unpinned stack). Creates the lock from the live env on first run.

Overall verdict = PASS iff checks 1+2+3 pass. Check 4 is advisory.

## Reading the physical result — `gate/dock_check.py`

The gate's verdict is a *hash* statement. It entails identical docks, but never
shows them, and "the CSV is byte-identical" is not the sentence a reviewer wants.
After a replay:

```bash
MUJOCO_GL=disabled PYTHONPATH=. python3 gate/dock_check.py \
    results/gate_run_scratch/sim_log.json
```

Prints the six **at-weld** `dock_events` d_mm against the frozen 2.5 table with
per-step margin to the 5 mm capture radius, plus θ_s / h_w / e_com peaks and the
QP-failure count. Exits non-zero on divergence.

Rule 10 applies: at-weld only. It deliberately does **not** compute a
min-over-swing distance, which is a fly-by artifact.

## Host-local bit identity — `gate/local_ref.py`

`run_gate.py` compares a 7-significant-digit CSV against a baseline committed
from another host, and that baseline does not reproduce everywhere (Python
3.11.15 → 3.11.16 flips `tauw_x_Nm` at 1e-15). For a no-behaviour-change
refactor the claim needed is stronger and host-local: **every file** the replay
writes, bit-identical to a reference produced on this host from the
pre-refactor code. JSON floats compare by `float.hex()`; wall-clock fields
(`*_ms`, the per-step solve-time summary) and the export's own path are the
only exclusions, and each is printed when used.

```bash
PYTHONPATH=. python3 gate/local_ref.py run r0a && \
PYTHONPATH=. python3 gate/local_ref.py run r0b && \
PYTHONPATH=. python3 gate/local_ref.py diff r0a r0b   # self-consistency first
PYTHONPATH=. python3 gate/local_ref.py freeze r0a      # -> gate/_run/local_ref/ref
PYTHONPATH=. python3 gate/local_ref.py check           # after each commit
```

The reference lives under `gate/_run/` (git-ignored): it is valid only on the
host that produced it, and its `_manifest.json` records the commit.

## Non-canonical branches — `gate/scenarios.py`

The canonical replay takes only one path through most branches, so neither
`run_gate.py` nor `local_ref.py` can say a change left the others alone. This
harness forces each one in a shortened traversal and replays it on a FROZEN
baseline tree and on the working tree, then compares every output file
bit-for-bit (`local_ref.py`'s comparator) and stdout line for line with
wall-clock numbers masked.

| group | scenarios | forces |
|---|---|---|
| gait sequencer | `timeout`, `stop`, `skip`, `skipstop`, `dwell` | dock TIMEOUT (continue / stop), pre-planner SKIP (continue / stop), DWELL. (`abortdiag` left with the `diag_*_on_abort` overrides it covered, R3a.) |
| AOCS | `aocs_off_interstep`, `hw_refresh_off` | the inter-step settle with the AOCS off, and with the entry-frozen h_w (`interstep_hw_refresh=False`). (The per-`aocs_mode` scenarios left with the modes, R1.) |
| torso reference | `ff_compress` | CoM-reference time compression (`torso_early_finish_fraction` 0.7). (`bypass` and `legacy_stack` left with the paths they covered, R2a.) |
| diagnostic hooks | `diag_pure_pd`, `diag_freeze_ref`, `diag_disable_aocs`, `diag_lock_arm_joints` | the four `DiagHooks` switches |
| paper Table 2 | `table2_rate`, `table2_none`, `table2_u25` | the envelope-ablation arms (`rate`: NMPC storage box off; `none`: NMPC rate cap + storage box off, QP box and AOCS clip held — `SimConfig.nmpc_tau_w_max=inf`, ported from `e37b9ca`) and the published unmanaged run u25 (`tau_w_max=1e6`). `full` = the canonical; station keeping = the canonical's trailing DS, longer. Full-length arms: `scripts/run_table2.py`, see `results/j2_adjconv/TABLE2_REPRODUCTION.md` |
| solver failures | `nmpc_fail`, `nmpc_unsuccessful`, `qp_fail`, `qp_fail_track` | NMPC call raising (no-previous-solve and shifted fallbacks); NMPC returning success=False without raising (status 2 infeasible, 1 max-iter); QP call raising in the settle (`qp_fail`, joint-damping fallback) and in tracking (`qp_fail_track`, zero-torque QP-FAIL) |

Failures are injected by the harness (the Nth `solve` call raises), never by
changing the code under test.

```bash
git worktree add /tmp/old_tree <baseline commit>
PYTHONPATH=. python3 gate/scenarios.py run /tmp/old_tree old --cov   # once
PYTHONPATH=. python3 gate/scenarios.py run . new                     # every commit
PYTHONPATH=. python3 gate/scenarios.py diff        # exit 0 iff all identical
PYTHONPATH=. python3 gate/scenarios.py coverage    # what each scenario covers
```

The "old" outputs carry a `_manifest.json` with their commit and remain the
reference for every later commit that is bit-identical to it: after each
commit only the "new" side is re-run. `--cov` records coverage;
`coverage` then lists, per scenario, the lines it executes that the canonical
replay does not (`gate/_run/cov/cov.json`, same commit), grouped by function —
so a scenario that silently stopped taking its branch shows up as "none".

**Bite tests** (2026-09-29, baseline `36c44bf`, run with `--new-side`): a
`+1e-12` on the output of `compute_aocs_command_legacy_pd_model` fails
`aocs_pd_model` and nothing else (the difference reaches `sim_log.json`,
2.4e-14 on L_com by tick 10); a `+1e-12` on the logging-only `c_ref` of the
`_step2_diag_log` entry fails every scenario with an SS phase (23/25 — not
`skip` / `skipstop`) through `step_log.json` alone, all control files identical.

The harness seeds `sim._step_q_start` / `_step_q_end` so the canonical driver
survives a step-0 pre-planner failure (a driver defect, see CLAUDE.md Known
Issues); the seed is identical on both sides and reaches no control path.

## Provenance

- Baseline reference point: **commit `bfd5509`** (main HEAD at founding; the first
  mainline commit carrying the paper's complete artifact set: 66-col fulldiag CSVs,
  `t4b_trace_900s.csv`, the T4/T4b scripts). The canonical config was frozen earlier
  at `32aefaf` (a documented ancestor). No git tag is used — this managed remote does
  not sync tags, and the base commit on `main` is a stable anchor by hash.
- Acceptance standard and the exception ledger: `gate/EXCEPTIONS.md`.
- Founding session report: `results/j2_adjconv/PHASE_CLEANUP_0.md`.

## Files

| path | role | committed |
|---|---|---|
| `run_gate.py` | orchestrator (4 checks, verdict) | yes |
| `replay_canonical.py` | isolated managed-scenario replay | yes |
| `dock_check.py` | headline canonical numbers from a replay log | yes |
| `local_ref.py` | host-local bit-identity reference for refactors | yes |
| `scenarios.py` | differential replay of non-canonical branches, old tree vs new | yes |
| `environment.lock` | pinned stack (founding baseline) | yes |
| `EXCEPTIONS.md` | acceptance policy + sign-off ledger | yes |
| `last_verdict.json` | latest run output (founding = baseline) | yes |
| `_run/` | scratch (re-export CSV) | git-ignored |
| `results/gate_run_scratch/` | scratch (replay sim_log) | git-ignored |

## Structural checks — `link_audit.py` and `verify_roots.py`

Added by the CLEANUP-20/21/22 restructure; run them after moving anything.

```bash
PYTHONPATH=. python3 gate/link_audit.py     # exits 1 if a move broke a citation
PYTHONPATH=. python3 gate/verify_roots.py   # exits 1 if a script's _root is wrong
```

`link_audit` resolves every repo-relative path cited in tracked `.md`/`.py` and
splits the failures into BROKEN BY MOVE (a move made the citation stale — the
only actionable class), DELETED, and DANGLING (never existed). It skips
gitignored paths and prose ellipses, and requires an *unambiguous* relocation
before calling something broken: without that, every citation of a common
basename like `sim_log.json` reads as a false alarm.

`verify_roots` evaluates each script's `_root` expression with `__file__` bound
to its real path and asserts it equals the repo root. This matters because
`_root` feeds both `sys.path` *and* path construction (URDF, MJCF, OUT_DIR): a
wrong `sys.path` fails loudly, a wrong `OUT_DIR` silently writes elsewhere.

## Documentation check — `verify_docs.py`

```bash
PYTHONPATH=. python3 gate/verify_docs.py    # exits 1 on an unresolved reference
```

Checks every claim in `docs/crawlbot/` that can be checked mechanically: each
`file:line` reference points inside a file that is long enough, and each
`Class.method` symbol is still defined in `crawlbot/`.

It exists because this repo has already lost one per-package documentation set
to silent rot (`docs/api/`, now under `Misc/reports/`, carrying a SUPERSEDED
banner and describing a module that does not exist). It earned its keep on the
first run: the new docs had inherited `sim_loop.py:2865-2870` from CLAUDE.md,
stale by ~280 lines after the chantier shrank `sim_loop`. The real site is
`sim_loop.py:2581-2584`; CLAUDE.md and CLEANUP_CARRYOVER were corrected too.

It does **not** check numeric values (weights, gains, thresholds) — those are
still verified by hand against CLAUDE.md.

## Documentation sync — `sync_docs.py`

```bash
PYTHONPATH=. python3 gate/sync_docs.py           # regenerate the measured blocks
PYTHONPATH=. python3 gate/sync_docs.py --check   # exit 1 if the docs lag the code
```

`docs/crawlbot/<pkg>/<module>.md` is half generated and half hand-written.
`sync_docs` owns the generated half — the header line (line count, canonical
coverage), the **Public API** table and the **Code map**, each entry carrying a
line-anchored link into the source — and never touches the prose.

`--check` is what makes CLAUDE.md rule 15 enforceable rather than aspirational:
it fails when a symbol is added, removed or moved without the document
following. Verified by injecting a public function into `contact_phase.py` and
observing the check go from exit 0 to exit 1.

Because every `file:line` link is generated, a refactor that shifts line numbers
is repaired by re-running the tool — links are never typed by hand, which is how
they went stale in CLAUDE.md before this existed.

Coverage annotations come from `gate/_run/cov/cov.json`; regenerate it with
`bash gate/_run/cov_replay.sh` after changing which paths execute.

## Canonical parameter check — `verify_params.py`

```bash
PYTHONPATH=. python3 gate/verify_params.py    # exit 1 on any mismatch
```

CLAUDE.md's "Key Parameters" table is the single source of truth for canonical
values, and every row cites a `file.py:LINE`. Nothing checked those citations, so
when the chantier shrank `config.py` (610 → 507) and `wholebody_qp.py`
(1385 → 950), **8 of 14 references drifted 20–50 lines** while the values stayed
correct — a pointer landing on `ss_Kp_torso = 6.0` while claiming to show
`alpha_torso_pose = 2000`.

Per row it checks that the file exists and is long enough, that the cited line
**declares the parameter the row names**, and that the value matches
**numerically** — so `2e3` in the source satisfies `2000` in the table. A string
comparison would have raised five false positives on scientific notation.

This closes the gap the documentation pass explicitly left open: paths, lines and
symbols were verified, numeric values were not.
