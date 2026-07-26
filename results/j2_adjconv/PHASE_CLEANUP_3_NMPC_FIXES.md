# PHASE CLEANUP-3 — NMPC audit fixes F2, F5, F7, F8, F9, F10

Acts on the CLEANUP-2 audit in the order Idriss approved. **Six of the seven planned items
landed; F3 is blocked on a governance decision (below), and F1/Tier B was not touched.**

All six are **bit-identity-safe by construction** — verified: the gate returns byte-identical
on the canonical, and pytest is unchanged.

## F2 — latent bug: failed solve no longer poisons the warm start

`crawlbot/solvers/nmpc_solver.py`, `solve()`. The `_w0_prev` / `_lam_g0_prev` / `_lam_x0_prev`
stores were unconditional, executed before `info.success` was known, so an infeasible
iterate (and its duals) became the next solve's initial guess. The `info` block was moved
ahead of the store and the store gated on `info.success`; on failure the last **successful**
warm start is retained, which is what `CentroidalNMPC`'s M5 shifted-trajectory fallback
expects.

Verified by forcing `Infeasible_Problem_Detected`:

| | before fix | after fix |
|---|---|---|
| `_w0_prev is None` after failed solve | `False` (failed iterate kept) | **`True`** |
| `_lam_g0_prev is None` | `False` | **`True`** |

Inert on the canonical because all 709 invoked solves succeed — which is exactly why this
was dormant rather than visible.

## F5 — six silent canonical parameters hoisted into `SimConfig` (Rule 5)

`nmpc_Wr=100.0`, `nmpc_Wu_f=0.01`, `nmpc_Wu_tau=0.001`, `nmpc_Qf_r=1000.0`,
`nmpc_Qf_v=100.0`, `nmpc_Qf_L=10.0` added to `config.py` and wired at
`sim_loop.py:388-...`. Values are the previous `CentroidalNMPCConfig` defaults **verbatim** —
each pair asserted IDENTICAL (including the `* np.ones(3)` expansion for the vector weights),
so the plant is unchanged.

## F7 — no more `inf` inside NLP constraint expressions

`build()` now emits the Ḣ_s group and the linear-momentum row **only when their bound is
finite**, instead of building `H_dot_s - inf`. `ng_path` became
`4 + (6 if tau_w finite) + (1 if p_max finite)`.

| config | `ng_path` |
|---|---|
| canonical (τ_w=2.5, p_max=50) | **11 — unchanged** |
| bare defaults (all `inf`) | **4** (SOC only; previously 11 rows, 7 of them constant `−inf`) |

State bounds were left alone: `±inf` on a *variable* is the correct way to say unbounded and
causes IPOPT no trouble — only constraint *expressions* were the problem.

## F8 — three stale doc claims corrected

- module header: "20 Hz … N=20, dt=0.05s" → 10 Hz / 0.8 s (N=8, dt=0.1) on the canonical,
  plus an explicit warning that the dataclass defaults are **not** canonical;
- `L_ref` "stub=0" → documented as the **live** `TorsoPlanner.l_com_reference_at(t_mid)`
  reference used in both stage and terminal cost (header and the inline comment at the
  stage-cost parameter unpack);
- `robot_mass = 90.0` flagged as a placeholder against the real 71.056 kg.

## F9 — single source for the parameter vector

New `_assemble_params()` builds `p = [r_ref, v_ref, r_C1, r_C2, c_simple, L_ref]`; `solve()`
and `get_full_trajectory()` both call it and no longer `np.concatenate` independently.
Verified: neither method still assembles the vector itself.

## F10 — `NMPCSolver` owns its decision-vector layout

New `NMPCSolver.apply_control_bounds_all_stages(u_min, u_max)` rewrites the per-stage control
bounds of an already-built solver. `CentroidalNMPC._apply_contact_bounds` now delegates to it
instead of re-deriving `u_start = nx + k*(nx+nu)` and writing `_lbw`/`_ubw` directly.
Re-verified after the change (SINGLE_A): all 8 `U_k` slots pin contact B to `[0,0]`, contact A
keeps `±300` force / `±8` torque, interleaved state slots untouched (`L_com` still `±10`).

## Verification

| check | before this phase | after |
|---|---|---|
| gate verdict | PASS | **PASS** (141.9 s) |
| artifact identity | byte-identical, 2077 × 132 928 | **byte-identical, 2077 × 132 928** |
| two-model consistency | PASS | PASS |
| environment pin | PASS | PASS |
| `pytest tests/` | 2 failed / 219 passed | **2 failed / 219 passed** (same two pre-existing) |

Net: `centroidal_nmpc.py` 681 → 701, `nmpc_solver.py` 616 → 649 (net +53; the audit fixes add
an accessor, a shared helper and documentation, so this phase is not a line-count reduction).

## F3 — BLOCKED, needs a Tier-1 decision

The fix (a sentinel for "NMPC not called", or a companion `nmpc_called` column) **necessarily
changes the fulldiag CSV**: a changed value fails the gate's field comparison, and a new column
fails its header comparison. Either way the committed baseline
`results/j2_adjconv/c25_fulldiag.csv` — the frozen paper artifact — would have to be
regenerated. Per `gate/EXCEPTIONS.md` that is a **Tier-1 metric-equivalence exception requiring
Idriss's explicit sign-off**, logged in the exception ledger.

Options:
1. **Sign the Tier-1 exception** and regenerate the baseline (the underlying physics is
   untouched; only a diagnostic channel's encoding changes).
2. **Defer** — document the semantics in `INTERNAL_figdata.md` and the exporter docstring so
   nobody plots `nmpc_ok` naively, and leave the artifact frozen through submission.

Recommendation: **option 2 until the paper is submitted**, then option 1. The risk F3 guards
against is a wrong number in a figure, and a documented caveat neutralises that without
touching the frozen artifact.

## Still open

F1 / Tier B (the ~85-line M3 h_w block, plus the dead `hw_for_nmpc` computation at
`sim_loop.py:2704-2707`) — untouched, still needs a ruling on
`tests/test_nmpc_conservation.py`, ~16 legacy scripts and the published B2 mechanism.
Carried over from CLEANUP-1: `setup_env.sh` cmeel ABI pins, and the 5 test-written PNGs that
dirty the tree on every `pytest` run.
