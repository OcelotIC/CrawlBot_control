# COM_GAIN_STATIC_TRACE — Phase 0 of the CoM Gain Semantics Audit

**Brief:** CoM Gain Semantics Audit and Controlled Fix (Idriss Chelikh)
**Phase:** 0 — static execution trace. **No code changed.**
**Audit branch:** `claude/com-gain-semantics-audit-j0u6yr`
**Inspected states:** canonical freeze `32aefaf`, current `main`, `WholeBodyQPConfig()` bare defaults
**Repo HEAD at audit time:** `eecbf94` (`sim_loop readability`)
**Session gate:** `gate/run_suite.py --fast` → **PASS** (199 tests, 198 passed, 0 failed, 0 errors, 1 xfail, 32 s)

---

## 0. Result in one line

`sim_loop._build_qp` hands `WholeBodyQPConfig` a **(3,3) matrix** for `Kp_com`/`Kd_com`;
`WholeBodyQP._com_task_rows` assumes a **(3,) vector** and applies `np.diag` a second
time, which *extracts* the diagonal. The subsequent `vector @ vector` collapses to a
**scalar**, and NumPy broadcasts it across all three Cartesian components. The
canonical CoM feedback law is therefore rank-one, not diagonal.

---

## 1. Canonical entry point

The frozen paper runs are generated through `scripts/diag_cooperative_arms.py::main`
(`dca.main`). The gate's reproduction of the **managed (C)** case is the executable
statement of the canonical configuration:

| item | value |
|---|---|
| Entry script (gate) | `gate/replay_canonical.py` |
| Underlying entry point | `scripts/diag_cooperative_arms.py::main` (`dca.main`) |
| Provenance of kwargs | verbatim from `Misc/scripts/diag_canonical2p5_run.py::_run('C', 2.5, …)` |
| Config constructor | `dca._make_m7_config()` → `SimConfig`, then per-kwarg mutation in `main` |
| QP constructor | `sim_loop.SimulationLoop._build_qp` (`sim_loop.py:934`), called at `sim_loop.py:469` |
| Extra patch in the canonical | `HierarchicalQP._solve_weighted` regularization pinned to `1e-6` (`gate/replay_canonical.py:22-33`) |

C-run kwargs (`gate/replay_canonical.py:36-47`):

```
legacy=False, alpha_torso_lin=0.0, anchor_dx=0.8, mass_ratio=0.01,
aocs_mode='legacy_pid_numerical', settle_seconds=20.0,
K_theta=1.0, K_omega=50.0, tau_w_max=2.5,
n_steps=6, ss_two_task=True, ss_alpha_mom=400.0,
alpha_torso_pose=2000.0, ss_alpha_ee=1000.0, ss_alpha_posture=2e1,
ss_alpha_wrench=1.0, ss_kp_torso=3.0, ss_kd_torso=2.5,
qp_envelope_exact=True,
interstep_settle_alpha_wrench=3.0, interstep_settle_epsilon_v=5e-3,
```

### 1.1 CoM gains are NOT overridden anywhere

`ss_Kp_com` / `ss_Kd_com` appear **nowhere** in `C_KWARGS`, nowhere in
`dca.main`, and nowhere in `dca._make_m7_config`. A repo-wide grep
(`grep -rn "Kp_com\|Kd_com" --include=*.py`, `Misc/` excluded) returns exactly
**nine** sites, listed in §2. The canonical therefore runs the `SimConfig`
defaults:

| symbol | file:line | value | Python type |
|---|---|---|---|
| `SimConfig.ss_Kp_com` | `crawlbot/simulation/config.py:349` | `3.0` | `float` |
| `SimConfig.ss_Kd_com` | `crawlbot/simulation/config.py:350` | `3.0` | `float` |

`kp = 3.0` happens to coincide with the brief's chosen sentinel — convenient, and
not a coincidence to correct for: the numbers in §3 are the production numbers.

### 1.2 Values and flags relevant to activation

| symbol | file:line | canonical value | source |
|---|---|---|---|
| `ss_two_task_mode` | set at `diag_cooperative_arms.py:445` | **`True`** | `ss_two_task=True` in `C_KWARGS` |
| `ds_centroidal_mode` | set at `diag_cooperative_arms.py:347` | **`True`** (unconditional; `:357` flips it off only under `--baseline_ds_rework`, not set here) | `dca.main` |
| `ss_alpha_mom` | `config.py:290`, overridden by kwarg | **`400.0`** | `C_KWARGS` |
| `ds_alpha_com` | `config.py:344` | **`100.0`** (default, not overridden) | `SimConfig` |
| `settle_mode` | `solve()` kwarg, `wholebody_qp.py:306` | `False` in SS, `True` in DS | `sim_loop` per phase |

### 1.3 Is `_com_task_rows` active?

`_com_task_rows` is called **unconditionally**, outside every mode branch, at
`wholebody_qp.py:409-411`. So `a_com_des` is *always computed*, on every tick, in
every phase. What is conditional is whether the resulting rows are **added as a
task**:

| phase | gate | file:line | added? | weight |
|---|---|---|---|---|
| **SS** | `_two_task = cfg.ss_two_task_mode and not settle_mode` → `True` | `wholebody_qp.py:422`, added `:425` | **YES** | `ss_alpha_mom = 400`, `priority=2` |
| **inter-step DS** | `settle_mode and cfg.ds_centroidal_mode and ds_centroidal_active` | `wholebody_qp.py:514`, added `:517` | **YES when `ds_centroidal_active`** | `ds_alpha_com = 100`, `priority=1` |
| **terminal DS** | same gate | `wholebody_qp.py:514` | **YES** — CLAUDE.md *Known Issues*: `dca.main` sets `ds_centroidal_mode=True` so centroidal DS runs everywhere *including the trailing settle* | `ds_alpha_com = 100`, `priority=1` |

Both consumers share the **same** `b_com` object built at `:409`. One defect,
two load paths.

---

## 2. Every `Kp_com` / `Kd_com` site in the repo (`Misc/` excluded)

| # | file:line | expression | role |
|---|---|---|---|
| 1 | `crawlbot/simulation/config.py:349` | `ss_Kp_com: float = 3.0` | scalar source |
| 2 | `crawlbot/simulation/config.py:350` | `ss_Kd_com: float = 3.0` | scalar source |
| 3 | `crawlbot/simulation/sim_loop.py:472` | `cfg.ss_Kp_com, cfg.ss_Kd_com` | passed positionally as `kpc, kdc` |
| 4 | **`crawlbot/simulation/sim_loop.py:957`** | **`Kp_com=np.diag([kpc]*3), Kd_com=np.diag([kdc]*3)`** | **DEFECT SITE 1 — builds a (3,3) matrix** |
| 5 | `crawlbot/solvers/wholebody_qp.py:162` | `Kp_com: np.ndarray = field(default_factory=lambda: 100.0*np.ones(3))` | dataclass default — a **(3,) vector**, i.e. the *opposite* contract |
| 6 | `crawlbot/solvers/wholebody_qp.py:163` | `Kd_com: … 20.0*np.ones(3)` | same |
| 7 | **`crawlbot/solvers/wholebody_qp.py:902-903`** | **`Kp_com_mat = np.diag(cfg.Kp_com)`** | **DEFECT SITE 2 — second `np.diag` extracts the diagonal** |
| 8 | `crawlbot/solvers/wholebody_qp.py:905-906` | `Kp_com_mat @ (r_com_ref - r_com_actual)` | collapses to a scalar |
| 9 | `tests/test_reworked_qp.py:568-569` | `np.diag(cfg.Kp_com) @ (r_com_ref - rs.r_com)` | test **mirrors** the production expression — see §5 |

**The contract is stated in two places and they disagree.** `wholebody_qp.py:162`
declares a vector; `sim_loop.py:957` supplies a matrix. Nothing validates the shape,
because `np.diag` accepts both and silently means something different for each.

---

## 3. Source trace table — proportional channel (K_p)

Canonical values, `kpc = 3.0`. Runtime types/shapes are the **measured** ones from
Phase 1 (`COM_GAIN_EXECUTABLE_PROOF.md`), not inferred.

| Stage | File:line | Expression | Runtime type | Runtime shape |
|---|---|---|---|---|
| Config scalar | `config.py:349` | `ss_Kp_com` → `kpc` | `float` | scalar (`3.0`) |
| QP construction | `sim_loop.py:957` | `np.diag([kpc]*3)` | `np.ndarray` `float64` | **`(3,3)`** — `[[3,0,0],[0,3,0],[0,0,3]]` |
| Dataclass storage | `wholebody_qp.py:162` (field) | `cfg.Kp_com` | `np.ndarray` `float64` | **`(3,3)`** (stored verbatim; no validation) |
| QP helper | `wholebody_qp.py:902` | `np.diag(cfg.Kp_com)` | `np.ndarray` `float64` | **`(3,)`** — `[3.0, 3.0, 3.0]` ← **diagonal EXTRACTED** |
| Error product | `wholebody_qp.py:905` | `Kp_com_mat @ e_r` | **`np.float64`** | **`()`** — a **scalar** = `3·(eₓ+e_y+e_z)` |
| Final addition | `wholebody_qp.py:904` | `a_com_ff + <scalar> + <scalar>` | `np.ndarray` | `(3,)` — scalar **broadcast** to all 3 axes |

## 3b. Source trace table — derivative channel (K_d)

Identical defect, identical line pair. `kdc = 3.0`.

| Stage | File:line | Expression | Runtime type | Runtime shape |
|---|---|---|---|---|
| Config scalar | `config.py:350` | `ss_Kd_com` → `kdc` | `float` | scalar (`3.0`) |
| QP construction | `sim_loop.py:957` | `np.diag([kdc]*3)` | `np.ndarray` | **`(3,3)`** |
| Dataclass storage | `wholebody_qp.py:163` (field) | `cfg.Kd_com` | `np.ndarray` | **`(3,3)`** |
| QP helper | `wholebody_qp.py:903` | `np.diag(cfg.Kd_com)` | `np.ndarray` | **`(3,)`** — `[3.0, 3.0, 3.0]` |
| Error product | `wholebody_qp.py:906` | `Kd_com_mat @ e_v` | **`np.float64`** | **`()`** — scalar = `3·(e_vx+e_vy+e_vz)` |
| Final addition | `wholebody_qp.py:904` | broadcast into `a_com_des` | `np.ndarray` | `(3,)` |

**Effective law as executed:**

```
a_com_des = a_com_ff  +  k_p · 𝟙𝟙ᵀ e_r  +  k_d · 𝟙𝟙ᵀ e_v          (k_p = k_d = 3)
```

not

```
a_com_des = a_com_ff  +  K_p e_r  +  K_d e_v,     K_p = diag(k_px, k_py, k_pz)
```

`𝟙𝟙ᵀ` is rank one: eigenvalue 3 on `𝟙`, eigenvalue **0** on the entire 2-D
differential subspace orthogonal to `𝟙`. Consequences, all verified executably in
Phase 1: only the common mode `eₓ+e_y+e_z` is corrected; two Cartesian error modes
are invisible; equal-and-opposite errors cancel exactly; a single-axis error
commands the same correction on all three axes; and the common mode receives `3k_p`,
not `k_p`.

---

## 4. Three code states compared

| State | `Kp_com` supplied | shape stored | `np.diag(cfg.Kp_com)` | law executed |
|---|---|---|---|---|
| **Canonical `32aefaf`** | `np.diag([kpc]*3)` (`sim_loop.py:1151` @ that commit) | `(3,3)` | `(3,)` | **RANK-ONE sum-and-broadcast** |
| **Current `main`** | `np.diag([kpc]*3)` (`sim_loop.py:957`) | `(3,3)` | `(3,)` | **RANK-ONE sum-and-broadcast** |
| **`WholeBodyQPConfig()` bare defaults** | `100.0*np.ones(3)` (`wholebody_qp.py:162`) | `(3,)` | `(3,3)` | **CLASSICAL DIAGONAL** |

Verification that `32aefaf` is semantically identical to `main` on this point:

```
$ git show 32aefaf:crawlbot/simulation/sim_loop.py | grep -n "Kp_com"
1151:            Kp_com=np.diag([kpc]*3), Kd_com=np.diag([kdc]*3),

$ git show 32aefaf:crawlbot/solvers/wholebody_qp.py | grep -n "Kp_com"
184:    Kp_com: np.ndarray = field(default_factory=lambda: 100.0 * np.ones(3))
659:        Kp_com_mat = np.diag(cfg.Kp_com)
662:                     + Kp_com_mat @ (r_com_ref - r_com_actual)
```

Same construction, same double-`np.diag`, only the line numbers moved (CLEANUP
refactors). **The frozen paper run and current `main` execute the same rank-one
law.** No behavioural drift between them on this path; the audit's conclusion
applies to both.

---

## 5. Why no test caught it — and why no sibling task is affected

### 5.1 The tests exercise the *other* contract

`tests/test_reworked_qp.py:92-129` builds its own `WholeBodyQPConfig` and **never
passes `Kp_com`/`Kd_com`**. Confirmed: `grep -rn "Kp_com=" tests/` returns nothing.
So every T-MOM test runs on the `(3,)` dataclass defaults — i.e. on the **correct
diagonal law**, which production never uses.

The comparator `_com_task_probe` (`tests/test_reworked_qp.py:566-570`) then
recomputes `a_com_des` with the *same expression as production*,
`np.diag(cfg.Kp_com) @ (…)`. Because it re-reads the same `cfg` object, it is a
**mirror**: it confirms the QP solves whatever task it was handed, and is
structurally incapable of detecting that the gain semantics are wrong. Its
docstring says as much — "Agreement confirms the task is solved" — it was never
claiming to check the gain.

Net effect: **the suite validates the diagonal law; the canonical executes the
rank-one law; nothing compares the two.** That is the blind spot, and it is a
missing shape contract, not a missing assertion.

### 5.2 Every sibling gain in the same file uses the vector contract correctly

The defect is confined to CoM. Same function, same `np.diag` idiom, correct inputs:

| task | supplied by `sim_loop._build_qp` | shape | `np.diag(...)` | correct? |
|---|---|---|---|---|
| torso 6-D | `Kp_torso=np.array([kpt]*6)` (`sim_loop.py:964`) | `(6,)` | `(6,6)` at `wholebody_qp.py:428` | **yes** |
| torso angular (DS) | same object | `(6,)` | `np.diag(cfg.Kp_torso)[3:,3:]` → `(3,3)` at `:530-531` | **yes** |
| swing EE 6-D | `Kp_ee=kpe*np.ones(3)`, `Kp_ee_ang=kpe_ang*np.ones(3)` (`:966-967`) | `(3,)`+`(3,)` | `np.diag(concat)` → `(6,6)` at `:452-453` | **yes** |
| posture | `Kp_posture=1.0` (`:968`) | scalar | used as a scalar, no `np.diag` | **yes** |
| **CoM** | **`Kp_com=np.diag([kpc]*3)`** (`:957`) | **`(3,3)`** | **`(3,)` at `:902`** | **NO** |

`Kp_com` is the **only** gain on line 957–967 wrapped in `np.diag` at the call
site. Read against its five neighbours, it is unambiguously a slip, not a design
choice — there is no plausible reading in which one gain was deliberately given the
matrix contract while the code that consumes it assumes vectors for all six.

---

## 6. Root cause (Phase 0 conclusion)

1. `SimConfig` stores CoM gains as **scalars** (`config.py:349-350`).
2. `sim_loop._build_qp` promotes them to a **(3,3) diagonal matrix**
   (`sim_loop.py:957`) — a contract that matches no other gain it passes.
3. `WholeBodyQPConfig` **declares** the field as a `(3,)` vector
   (`wholebody_qp.py:162-163`) and performs **no shape validation**.
4. `_com_task_rows` applies `np.diag` a second time (`wholebody_qp.py:902-903`).
   On a matrix input this **extracts** the diagonal → `(3,)`.
5. `(3,) @ (3,)` is an inner product → **`np.float64` scalar**
   (`wholebody_qp.py:905-906`).
6. `a_com_ff + scalar` **broadcasts** across all three components
   (`wholebody_qp.py:904`), yielding `k_p 𝟙𝟙ᵀ e` per component.

Both channels (`K_p`, `K_d`) and both consumers (SS T-MOM at `α=400`, DS centroidal
CoM at `α=100`) are affected. The hypothesis in the brief §10 is confirmed at every
step.

---

## 7. Status and next step

Phase 0 complete. **No code was modified.**

- Phase 1 (executable proof at the production helper): **done** →
  `COM_GAIN_EXECUTABLE_PROOF.md`
- Phase 2 (activation + load-bearing evidence from the canonical run): **done** →
  `COM_GAIN_ACTIVATION_EVIDENCE.md`
- Phase 3 (A/B fix): **BLOCKED on Idriss's approval**, per brief §6 human stop
  gate and §3 frozen rule "no behavioural code change before the Phase-2 human
  checkpoint".
