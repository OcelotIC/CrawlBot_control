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

## Provenance

- Baseline reference point: **`paper-2p5-base` = commit `bfd5509`** (main HEAD at
  founding; the first mainline commit carrying the paper's complete artifact set:
  66-col fulldiag CSVs, `t4b_trace_900s.csv`, the T4/T4b scripts). The canonical
  config was frozen earlier at `32aefaf` (a documented ancestor, untagged).
  *Note:* this managed remote does not sync git tags; `paper-2p5-base` lives as a
  local annotated tag + this documented equivalence to `bfd5509` (permanent on
  `main`), pushable from a developer machine.
- Acceptance standard and the exception ledger: `gate/EXCEPTIONS.md`.
- Founding session report: `results/j2_adjconv/PHASE_CLEANUP_0.md`.

## Files

| path | role | committed |
|---|---|---|
| `run_gate.py` | orchestrator (4 checks, verdict) | yes |
| `replay_canonical.py` | isolated managed-scenario replay | yes |
| `environment.lock` | pinned stack (founding baseline) | yes |
| `EXCEPTIONS.md` | acceptance policy + sign-off ledger | yes |
| `last_verdict.json` | latest run output (founding = baseline) | yes |
| `_run/` | scratch (re-export CSV) | git-ignored |
| `results/gate_run_scratch/` | scratch (replay sim_log) | git-ignored |
