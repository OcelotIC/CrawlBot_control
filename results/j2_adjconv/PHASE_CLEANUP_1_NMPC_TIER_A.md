# PHASE CLEANUP-1 — centroidal NMPC, Tier A (zero-risk dead code)

First cleanup landing of the chantier. Branch `cleanup-nmpc` off `cleanup` (base `bfd5509`).
Scope approved by Idriss: **Tier A only** — code with zero callers anywhere in the repo,
including tests. Tiers B/C/D deliberately untouched.

## What was removed — `crawlbot/solvers/nmpc_solver.py` (657 → 616 lines, −41)

| removed | old lines | evidence |
|---|---|---|
| `set_dynamics()` — discrete-time dynamics setter | 139–147 | 0 callers repo-wide; only `set_continuous_dynamics()` (RK4) is ever used |
| `sqpmethod` / qpOASES branch in `_get_default_solver_options()` | 611–617 | 0 references to `sqpmethod`; `solver_name` is `'ipopt'` everywhere |
| `n_decision_vars`, `n_constraints` properties | 623–635 | 0 references |
| `u_guess` / `x_guess` params of `solve()` + their `_build_initial_guess` plumbing | 399–400, 547–565 | 0 callers pass them; both production call sites (`centroidal_nmpc.py:443,600`) use keyword args only |
| dead imports `List`, `warnings` | 44–45 | present only on the import line |

Docstrings referencing the removed features were corrected in the same pass (module
header "Discrete or continuous"/"qpOASES", class `solver_name` doc, and the `build()`
error message that named `set_dynamics()`). `_build_initial_guess` keeps identical
behaviour: the `None`-defaulted branches became the unconditional cold-start path.

**Behaviourally inert by construction** — nothing removed was reachable from any caller.

## Verification (this container; env matches `gate/environment.lock` exactly)

| check | before edit | after edit |
|---|---|---|
| `gate/run_gate.py` verdict | **PASS** (152.8 s) | **PASS** (143.3 s) |
| artifact identity vs `c25_fulldiag.csv` | byte-identical, 2077 rows × 132 928 fields | **byte-identical, 2077 rows × 132 928 fields** |
| two-model consistency | PASS (15 links, 14 joints, 71.056 kg) | PASS |
| environment pin | PASS | PASS |
| `pytest tests/` | 2 failed / 219 passed | **2 failed / 219 passed** (same two: `test_far_infeasible_under_tight_rate`, `test_E7_t15_step2_dock_under_fk_mode`) |

The **before**-edit gate run was executed deliberately, so that bit-identity was proven in
this rebuilt container *before* any change — making the after-result attributable.

## Findings recorded, NOT acted on (out of scope for this PR)

1. **`setup_env.sh` yields a broken pinocchio on a fresh container.** It pins `pin==3.9.0`
   but not its cmeel ABI deps, so pip resolves `cmeel-urdfdom 6.0.0` / `cmeel-tinyxml2 11.0.0`
   while pin's binary needs `liburdfdom_*.so.4.0` / `libtinyxml2.so.10`. Fixed locally with
   `cmeel-urdfdom~=4.0` + `cmeel-tinyxml2~=10.0`. **Recommended next PR** — nothing in the repo
   runs without it.
2. **The test suite dirties five tracked files on every run** —
   `Misc/runs/M2_tests/{t10_passivity,t7_tracking}.png`, `Misc/runs/M3_tests/t4_hw_bounds.png`,
   `Misc/runs/phase2_0_tmom/{t_mom_sine_x,t_mom_step_x}.png` are rewritten by
   `test_nmpc_conservation.py` / `test_reworked_qp.py`, and matplotlib's encoding differs
   byte-wise run to run (±1 kB on identical plots). The repo can therefore never be verified
   clean after `pytest`, which undercuts the gate's bit-identity discipline. Fix = gitignore
   them or point the tests at a scratch dir.
3. **Five canonical NMPC cost weights live outside `SimConfig`** — `Wr=100`, `Wu_f=0.01`,
   `Wu_tau=0.001`, `Qf_r=1000`, `Qf_v=100` are `CentroidalNMPCConfig` defaults that `sim_loop`
   never overrides, i.e. silent canonical parameters (Rule 5). Tier D.
4. **Stale docs in `centroidal_nmpc.py`** (not dead code): header claims N=20/dt=0.05 (actual
   canonical N=8/dt=0.1), `robot_mass=90.0` (actual 71.056), and lines 33/202 call `L_ref` a
   "stub=0" although `sim_loop:2711` feeds a live `TorsoPlanner.l_com_reference_at(t_mid)`.
5. **`_apply_contact_bounds` reaches into `self._nmpc._lbw/_ubw`** (private state of the
   generic solver) — a coupling smell worth an accessor.

## Not touched

Tier B (the ~85-line M3 hw-conservation block: config 96–106, path box 291–300, terminal
309–326, `compute_c_simple` 510–543). It is dead on the canonical path
(`enforce_hw_conservation=False` ⇒ `ng_path=11`, no terminal constraints) but **live** in
`tests/test_nmpc_conservation.py` and ~16 legacy M4–M7 scripts, and is the published B2
Option-B mechanism. Tier C (`get_full_trajectory`, 5 test users) also untouched.
