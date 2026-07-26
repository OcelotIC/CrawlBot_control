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
