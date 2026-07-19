# Gate acceptance policy & exceptions

The CLEANUP chantier accepts changes on a **two-tier** standard.

## Tier-0 — bit-identity (default for every PR)

`gate/run_gate.py` must report **PASS**: the re-run canonical fulldiag CSV is
**byte-identical** to the committed baseline `results/j2_adjconv/c25_fulldiag.csv`
(the `paper-2p5-base` / `bfd5509` base), and the two-model consistency check
passes. This is the default and requires no sign-off.

### Definitional exclusion (not an exception)

Two columns are excluded from the byte-comparison because they are
**nondeterministic instrumentation**, not scientific results:

| column | why excluded |
|---|---|
| `qp_time_ms` | wall-clock QP solve time, measured per tick during the run |
| `nmpc_time_ms` | wall-clock NMPC solve time, measured per tick during the run |

These vary run-to-run on any machine and carry no physical meaning. All other
**64** columns (kinematics, momenta, wrenches, gate flags, solver status,
`nmpc_iterations`, …) are byte-compared. Excluding wall-clock timings is *not* a
metric-equivalence exception and needs no sign-off — a timing-only difference is
still a Tier-0 PASS.

## Tier-1 — metric-equivalence exception (needs Idriss's explicit sign-off)

A legitimate rewrite may reorder floating-point operations (e.g. a refactor that
changes summation order) and break bit-identity while preserving the result to
tolerance. Such a change is acceptable **only** with a documented justification
and Idriss's explicit sign-off, recorded in the table below.

Each row must state: the PR, the affected quantities/columns, the numerical
tolerance applied, and the one-line justification. Absent a matching row, any
byte-level difference on a compared column is a **FAIL**.

| date | PR | affected quantities | tolerance | justification | signed-off |
|---|---|---|---|---|---|
| _(none yet — Tier-0 is clean at founding)_ | | | | | |

## Environment

Bit-identity is only meaningful on a pinned stack. `gate/environment.lock`
records the exact interpreter/library versions of the founding baseline. The
gate **WARNs** on mismatch today (a later revision may promote this to FAIL);
reproduce the lock's versions before trusting a Tier-0 PASS.
