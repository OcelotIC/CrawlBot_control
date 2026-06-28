# INTERNAL — Non-co-integration audit: h_w is a parameter everywhere / a decision variable nowhere; the c_simple(k) → c_curr template

**Read-only map on `ae0673e`, branch `j2/ds-active-rework`. No `crawlbot/` change, no implementation.**
Reproducer `scripts/audit_no_cointegration.py` (20/20, code-anchor + runtime structural proof). Two goals:
(1) back the paper's decentralization/frugality argument with an evidence table; (2) template the upcoming
`c_curr` (the QP's per-tick h_w bound input) on the existing `c_simple(k)` frozen-parameter pathway.

---

## DECISIVE OUTPUT

1. **The frugality / no-co-integration argument is a FACT.** There are **two optimizers + one closed-form
   law**, never three optimizations: NMPC (state `[r_com,v_com,L_com]`, controls = contact wrench) + whole-body
   QP (`qdd_t, qdd, λ, τ_q, slack`) + AOCS (closed-form PID law, no decision variables). The two decision
   vectors are **disjoint** and **neither contains h_w or τ_w**. Repo-wide there is **no** optimization
   variable named `h_w`/`τ_w`/`wheel` (programmatic scan: 0 matches), and **no joint solver** co-optimizes
   `[τ_q, τ_w]`. **h_w and τ_w are a frozen parameter / I-O quantity everywhere and a decision variable
   nowhere.** Nothing contradicts the framing.
2. **`c_curr` is the same parametric mechanism as `c_simple(k)`.** h_w enters the NMPC only as the frozen
   parameter `c_simple(k)` (read once per MPC step from wheel telemetry, bound as a CasADi *parameter*, used
   inside the envelope box) and enters the QP only as the frozen parameter `hw_current` (a numeric `solve()`
   argument forming the box-constraint RHS). Refreshing `hw_current` per tick (= passing a fresh `c_curr`) is
   a **parameter refresh, frozen at solve time — NOT a co-solve**; it never puts h_w in a decision vector. It
   is architecturally identical to how `c_simple(k)` is refreshed per MPC step — and the `_step` QP **already
   does** this per-tick refresh. The "refresh breaks decentralization" worry is **resolved: it does not.**

---

## Q1 — the optimization problems and their decision vectors

| subsystem | kind | decision vector | h_w in it? | τ_w in it? | evidence |
|---|---|---|---|---|---|
| **NMPC** (`centroidal_nmpc.py`) | optimization (IPOPT) | `w = {X_k=[r_com,v_com,L_com] (NX=9), U_k=[f1,τ1,f2,τ2] (NU=12)}` over the horizon | **NO** | **NO** | NX=9 `:123` (“hw removed, AOCS independent”), NU=12 `:124`; `w.append(Xk)/append(Uk)` only, `nmpc_solver.py:310/325/343` |
| **Whole-body QP** (`wholebody_qp.py`) | optimization (QP) | `z = [qdd_t(6), qdd(nq), λ(6·nc), τ_q(nq), slack_hw(6)]` | **NO** | **NO** | `_n_vars` `:251`; `_compute_indices` layout `:1311`; runtime keys = `{qdd_t,qdd,lambda,tau,slack_hw_up,slack_hw_lo}` |
| **AOCS** (`force_estimator.py`) | **closed-form law** (no solver) | — (none) | — | — | no `opti.`/`cp.Variable`/`nlpsol`/`solve_qp`/`minimize` anywhere (the grep “opti” hits are the word *optional*); `τ_w = ff + K_hw·hw_error + pid` `:594` |

**Q1(d) — DECISIVE:** the two solver decision vectors `{X,U}` and `{qdd_t,qdd,λ,τ_q,slack}` are disjoint, and
**neither contains h_w or τ_w**. A repo-wide scan for any `(opti.variable|cp.Variable|.variable|SX.sym)(…h_w|tau_w|wheel…)`
returns **0 matches**. The AOCS is a closed-form law (trivially co-integrates nothing). **⇒ there is no
monolithic / joint `[τ_q, τ_w]` optimization, and no optimization whose decision vector contains h_w or τ_w.**

## Q2 — the status table (the paper deliverable)

Status of each shared quantity in each subsystem (DV = decision variable; PARAM = frozen parameter;
FB-IN = state-feedback input; OUT = computed output; — = absent), with code anchors:

| quantity | NMPC | Whole-body QP | AOCS |
|---|---|---|---|
| **h_w** (wheel momentum) | **PARAM** — via `c_simple(k)=h_w₀+L₀+r₀×m·v₀`, p[12:15] (`centroidal_nmpc.py:125,427`); box `h_w(k)=c_simple−L_com(k)−orbital` (`:297`) | **PARAM** — `hw_current`, a `solve()` arg (`wholebody_qp.py:322`) forming the box RHS `b=hw_max−hw_current` (`:511,516`) | **FB-IN** — desaturation term `+K_hw·(clip(h_w)−h_w)` (`force_estimator.py:372`) |
| **λ** (contact wrench) | **DV** (control `U_k`, NU=12) (`:124`) | **DV** (`idx['lambda']`) (`:1322`) | **PARAM-IN** — wrench FF `−Σ(r_Ci×f_i+τ_i)` from `lambda_qp_sol` (`sim_loop.py` wrench-FF block) |
| **τ_w** (wheel torque) | — (absent) | — (absent; enters only as the scalar **bound** τ_w_max on \|M_env·λ\|, `:551`) | **OUT** — the law's result `τ_w=ff+K_hw·hw_error+pid`, clipped (`force_estimator.py:594,595`) |

**Decision-variable row for h_w: nowhere. Decision-variable row for τ_w: nowhere.** h_w is a frozen parameter
in both optimizers and a feedback input to the law; τ_w is only ever the law's output and a scalar bound. λ is
the only quantity that is a decision variable in both solvers — and it is a *parameter input* to the AOCS,
not a co-solved variable. **This is the "frugal parametric coupling, no co-integration" evidence.**

## Q3 — the c_simple(k) parameter pathway (the template)

| step | what happens | code |
|---|---|---|
| **read** | per MPC step, h_w read fresh from wheel state `qvel[6:9]` → `hw_for_nmpc` | `sim_loop.py:2626` |
| **form** | `c_simple = compute_c_simple(r,v,L,hw) = h_w₀ + L_com₀ + r_com₀×m·v_com₀` (once per solve) | `centroidal_nmpc.py:427,510-543` |
| **declare** | the parameter vector is `p = ca.SX.sym('p', NP=18)`, c_simple = p[12:15] (a CasADi **parameter**, not `opti.variable`) | `centroidal_nmpc.py:125`; `nmpc_solver.py:242` |
| **bind** | at solve: `P = concat([x0, params])` passed to the NLP as `'p': ca.DM(P)` — **frozen numeric value**, separate from the decision vector (`x0/lbx/ubx`) | `nmpc_solver.py:439,451` |
| **use** | inside the envelope box at every knot + terminal: `h_w(k)=c_simple − L_com(k) − r_com(k)×m·v_com(k)`, bounded `∈[−h′,h′]` — `c_simple` frozen, `L_com(k)` the decision STATE | `centroidal_nmpc.py:297-300,323` |

So h_w is reconstructed *inside a constraint* from a **frozen parameter** (`c_simple`) and a **decision state**
(`L_com`); h_w itself is never a free variable. **The constant is held over the whole horizon solve and
refreshed only between MPC steps.** This is the exact pattern `c_curr` must follow.

## Q4 — the QP h_w bound and how c_curr fits (the worry, resolved)

- **The bound:** the momentum-safety box `h_min ≤ hw_current − dt·M_λ·λ ≤ h_max` (with slacks), C5-relevant.
  RHS `b_mom_upper = hw_max − hw_current`, `b_mom_lower = hw_current − hw_min` (`wholebody_qp.py:511,516`).
  The constraint acts on **λ** (a decision var); `hw_current` is the frozen RHS. (τ_w_max enters separately as
  the scalar bound on `|M_env·λ|`, `:544-555` — also a λ-constraint, τ_w not a variable.)
- **Already a parameter:** `hw_current` is a `solve()` argument (`:322`), passed in as a numeric array — frozen
  at solve, never computed/optimized inside the QP. h_w is reconstructed as `hw_current(frozen) − dt·M_λ·λ`,
  exactly mirroring the NMPC's `c_simple(frozen) − L_com − orbital`.
- **DECISIVE — refresh = parameter mechanism, not co-solve (YES):** updating `hw_current` per tick (= passing
  a fresh `c_curr` each tick) only changes the **value of a frozen parameter read once per solve**. The QP
  does **not** co-optimize it; h_w/τ_w never enter the decision vector (Q1/Q2). This is **architecturally
  identical** to refreshing `c_simple(k)` per MPC step. The forbidden thing — h_w in the decision vector —
  does **not** happen, before or after the refresh. Concretely, **the `_step` QP already refreshes its
  `hw_current` per tick** (one-tick-lagged: `sim_loop.py:3416` reads `qvel[6:9]` → feeds `:3002`); only the
  inter-step loop freezes it at loop entry (`:701`). So the refresh is already in production and is not a
  coupling.
- **What c_curr would change (next brief — NOT implemented here):** pass the **current** wheel momentum as the
  frozen `hw_current` parameter on each inter-step QP solve (replacing the entry-frozen value at
  `sim_loop.py:701/753`), mirroring `c_simple(k)`'s per-step refresh. A one-line parameter-source change; no
  new decision variable, no co-solve, no change to the QP structure.

---

## Flags / divergences vs prior audit facts

- **Consistent with the AOCS-FF audit:** re-confirms τ_w is **not** a QP decision variable (it appears only as
  the scalar bound τ_w_max on `|M_env·λ|`), and that the AOCS is closed-form (no solver). No divergence.
- **Consistent with the envelope / Piste-A audits:** the QP envelope box is the `|M_env·λ| ≤ τ_w_max`
  constraint (proxy `M_λ` default, exact `M_exact` under `qp_envelope_exact`) — a constraint on λ, not on τ_w;
  unchanged here. The NMPC 9-D-centroidal / h_w-as-`c_simple(k)` invariant holds exactly.
- **No contradiction found.** If any of h_w/τ_w had appeared as a decision variable, or a joint `[τ_q, τ_w]`
  solver existed, it would break the framing — none does (0 matches, 20/20 checks).

## Reproduce
```
MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_no_cointegration.py
```
Code anchors: `centroidal_nmpc.py:123-125` (NX/NU/NP), `:297,427,510` (c_simple); `nmpc_solver.py:310/325`
(decision vector), `:439,451` (parameter binding); `wholebody_qp.py:251,1311` (decision vector), `:511,516`
(hw_current RHS), `:322` (solve arg), `:551` (τ_w_max bound); `force_estimator.py:594` (closed-form τ_w);
`sim_loop.py:2626` (NMPC hw read), `:701` (inter-step entry-frozen), `:3416→3002` (`_step` per-tick refresh).
No `crawlbot/` change ⇒ no regression run needed.

**STOP after the report.** Map only — no design, no implementation. The c_curr implementation brief follows.
No merge, no PR.
