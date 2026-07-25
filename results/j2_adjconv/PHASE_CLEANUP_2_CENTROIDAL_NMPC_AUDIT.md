# PHASE CLEANUP-2 — `centroidal_nmpc.py` audit (READ-ONLY)

Audit of `crawlbot/solvers/centroidal_nmpc.py` (681 lines). **No code changed.** Every
claim below was verified by execution, not by reading. Probes run on `cleanup-nmpc`
(env matches `gate/environment.lock`); canonical numbers read from the committed
`results/j2_adjconv/c25_fulldiag.csv` (2077 ticks).

Physics core is **correct**: `v̇=Σf/m` (no gravity, orbit), `L̇_com=Σ[(r_Cj−r_com)×f_j+τ_j]`
(lever from robot CoM), `Ḣ_s=Σ[r_Cj×f_j+τ_j]` (lever from structure CoM = origin of R_s).
`NX/NU=9/12` matches CLAUDE.md. The findings are about dead paths, silent parameters and
one latent solver-state bug — not about the model.

---

## ⛔ F1 IS RETRACTED — see CLEANUP-4. The h_w path is **LIVE** on the canonical.

**This finding was wrong.** The canonical config is not bare `SimConfig()`: `dca.main` takes
`cfg = r_single._make_m7_config()` (`diag_cooperative_arms.py:268`), and
`run_m7_single_step.py:49` sets **`enforce_hw_conservation=True`** with
`h_max_tight=[5,5,5]`, `kappa_terminal=1.0`. Instrumenting `build()` during a real run
reports `enforce_hw=True, ng_path=17, ng_term=6` — the M3 RWA box **and** its terminal
constraint are active, and line-coverage of a full canonical replay confirms the block
executes.

The error: the probe below proved the *conditional* behaviour correctly (h_w is ignored when
the flag is off), but the flag's canonical value was inferred from the dataclass default
instead of traced through `dca`. Consequently `compute_c_simple`, the `hw_current` argument,
`h_max_tight`, `kappa_terminal`, and `sim_loop.py`'s `hw_for_nmpc` computation are **all
live** — and "Tier B" is **not** a removal candidate at all: it is live canonical code
implementing the published B2 Option-B mechanism. The original text is kept below for the
record.

### (retracted) F1 — original claim

`enforce_hw_conservation=False` canonically ⇒ `c_simple` (`p[12:15]`) is referenced by **no**
expression in the NLP. Proven behaviourally:

| `enforce_hw` | `max‖solution(h_w=0) − solution(h_w=±4.9)‖` | reading |
|---|---|---|
| **False (canonical)** | **0.000e+00** | h_w input is **ignored entirely** |
| True | 1.982e-01 | h_w genuinely drives the solution |

Dead surface this implies:
- `hw_current` argument of `solve()` (line 370) and `get_full_trajectory()` (line 569);
- `compute_c_simple()` (**510–543**) — executed on *every* solve (lines 427, 589), result unused;
- 3 of the 18 parameter slots carried into every NLP evaluation;
- **and it reaches outside this file**: `sim_loop.py:2704–2707` computes `hw_for_nmpc` from
  the RWA wheel velocities on every SS tick purely to feed this dead input.

This is the data-path half of the Tier-B M3 block. Note the block is still **live in
`tests/test_nmpc_conservation.py` and ~16 legacy M4–M7 scripts**, and is the published B2
Option-B mechanism — so removal remains a deliberate decision, not a freebie.

## F2 — A failed solve's iterate is retained as the next warm start *(latent bug)*

`CentroidalNMPC.solve()` guards its own cache correctly (`_last_x_opt` untouched on failure,
`shift_warm_start()` only on success — lines 449–455). But `NMPCSolver.solve()` stores
`_w0_prev`, `_lam_g0_prev`, `_lam_x0_prev` **unconditionally**, before `info.success` is
known. Verified by forcing `Infeasible_Problem_Detected`:

```
solve success                   : False (Infeasible_Problem_Detected)
CentroidalNMPC._last_x_opt None : True    <- cache correctly NOT updated
NMPCSolver._w0_prev None        : False   <- infeasible iterate KEPT as next warm start
NMPCSolver._lam_g0_prev None    : False
```

So the next `solve(warm_start=True)` starts from an infeasible point (and its duals).
**Dormant on the canonical** (F3: 709/709 solves succeed), but this is precisely the path
that governs a failure cascade — and the M5 `get_shifted_fallback()` machinery exists
because failures were once real. Fix belongs in `nmpc_solver.py`, gated on `info.success`.

## F3 — `nmpc_ok` conflates "not called" with "failed" *(reporting risk)*

| phase | ticks | `nmpc_ok=1` |
|---|---|---|
| `SS` | 508 | 508 (100 %) |
| `DS_terminal` | 201 | 201 (100 %) |
| `DS_interstep` | 1368 | **0 — NMPC is not invoked in this phase** |

Whole-column reading gives a false **34.1 %** success rate; the true rate is **100 %
(709/709)**. Anyone plotting `nmpc_ok` straight from the fulldiag CSV would report a
two-thirds failure rate for a solver that never failed. Needs a sentinel (e.g. `NaN`/`-1`)
for "not called", or a companion `nmpc_called` column.

## F4 — The linear-momentum constraint never binds

Peak `‖m·v_com‖ = 9.567` kg·m/s against `p_max = 50` → **19.1 % of the bound, never active**.
It costs one constraint row at every knot of every solve. Keep it as a safety bound if you
like, but it does no work at the canonical operating point and should not be described as
shaping the trajectory.

## F5 — Six silent canonical parameters (Rule 5)

`sim_loop.py:388–398` overrides 12 fields but **never** these, so the `CentroidalNMPCConfig`
defaults *are* canonical values living outside `SimConfig`:

`Wr=100`, `Wu_f=0.01`, `Wu_tau=0.001`, `Qf_r=1000`, `Qf_v=100`, `Qf_L=10.0` (lines 79–84, 105).

## F6 — Defaults diverge from canonical badly enough to be a trap

| field | default | canonical | ratio |
|---|---|---|---|
| `robot_mass` | 90.0 | 71.056 | different robot |
| `N` / `dt` | 20 / 0.05 | 8 / 0.1 | 10 Hz, 0.8 s — not 20 Hz, 1 s |
| `f_max` | 3000.0 | 300.0 | **10×** |
| `tau_max` | 300.0 | 8.0 | **37×** |
| `L_max`, `tau_w_max`, `p_max` | `inf`, `inf`, `inf` | 10.0, **2.5**, 50.0 | unconstrained |

`CentroidalNMPCConfig()` on its own yields a heavier robot with no momentum envelope at all
— the exact opposite of the paper's contribution. Only `sim_loop` makes it correct.

## F7 — `inf` leaks into NLP expressions at the defaults

With `tau_w_max=inf` the path constraint becomes `vertcat(H_dot_s − inf, −H_dot_s − inf)`,
i.e. a constant `−inf` row handed to IPOPT (same for `p_max`). Confirmed symbolically:
`p_max=inf → (sq(e)-inf)`. Harmless canonically (all three are finite) but fragile;
guarding with `np.isfinite` and simply omitting the row would be cleaner.

## F8 — Stale documentation (correct, don't delete)

- Line 10: "20 Hz … N=20, dt=0.05s" → actual **N=8, dt=0.1 (10 Hz, 0.8 s horizon)**.
- Lines 33 and 202: `L_ref` called a "stub=0" — **false**; `sim_loop.py:2711` feeds a live
  `torso_planner.l_com_reference_at(t_mid)`, and `L_ref` is used in both stage and terminal cost.
- Line 72: `robot_mass = 90.0` in a docstringed default vs the real 71.056 kg.

## F9 — `solve()` and `get_full_trajectory()` duplicate the parameter assembly

Both independently build the identical 6-term vector (`r_ref, v_ref, r_C1, r_C2, c_simple,
L_ref`) at lines 434–440 and 592–598, each calling `compute_c_simple` first. Verified both
contain the duplicate logic. Any change to `NP` must be made twice — a silent-divergence risk.

## F10 — `_apply_contact_bounds` mutates another class's private state

Lines 655–669 write `self._nmpc._lbw` / `._ubw` directly, re-deriving the decision-vector
layout (`u_start = nx + k*(nx+nu)`) that belongs to `NMPCSolver`. **The arithmetic is
correct** — verified on all 8 `U_k` slots for `SINGLE_A`: contact B pinned to `[0,0]`,
contact A at `[±300]` force / `[±8]` torque, and the interleaved state slots untouched
(`L_com` still `±10`). The risk is coupling, not correctness: a layout change inside
`NMPCSolver` would silently corrupt these bounds with no error. Wants an accessor
(`NMPCSolver.set_control_bounds_at_all_stages(...)`).

---

## Suggested ordering if we act

1. **F2** — real bug, small, self-contained (`nmpc_solver.py`, gate on `info.success`).
2. **F3** — cheap, prevents a wrong number reaching the paper.
3. **F5 + F8** — hoist the six weights into `SimConfig`; fix the three stale claims.
4. **F10 → F9** — accessor, then de-duplicate the param assembly.
5. **F1 / Tier B** — biggest deletion, but needs your ruling on the tests + legacy scripts
   + the published B2 mechanism. **F7** rides along with whatever touches the constraints.

Findings carried over from CLEANUP-1 and still open: `setup_env.sh` cmeel ABI pins
(nothing runs on a fresh container without them) and the 5 test-written PNGs that dirty
the tree on every `pytest`.
