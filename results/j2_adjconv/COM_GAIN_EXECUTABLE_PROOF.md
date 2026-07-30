# COM_GAIN_EXECUTABLE_PROOF — Phase 1 of the CoM Gain Semantics Audit

**Brief:** CoM Gain Semantics Audit and Controlled Fix (Idriss Chelikh)
**Phase:** 1 — minimal executable semantics test. **No production code changed.**
**Audit branch:** `claude/com-gain-semantics-audit-j0u6yr`, repo HEAD `eecbf94`
**Target exercised:** `crawlbot.solvers.wholebody_qp.WholeBodyQP._com_task_rows` — the
production helper itself, not a re-implementation.
**Script:** written to the session scratchpad (deliberately **not committed**; it is a
throwaway probe, and the permanent version of these assertions belongs in
`tests/` as part of Phase 3 §7.3).
**Environment:** NumPy **2.3.5**, Python 3.11, `pin==3.9.0`, `mujoco 3.11.0`,
`MUJOCO_GL=disabled`

---

## 1. Test construction

Per brief §5, with the production helper called directly:

| quantity | value | consequence |
|---|---|---|
| `a_com_ff` | `0` | feedback is the only term in `a_com_des` |
| `J_com` | `I₃` on the torso-linear acceleration block, zero elsewhere | `A_com` well-formed |
| `Jdot_dq_com` | `0` | `b_com = a_com_des` exactly |
| `dq_robot` | `0` | `v_com_actual = J_com @ dq_robot = 0` |
| `v_com_ref` | `0` (except in the K_d probe) | `e_v = 0`, isolates the K_p channel |
| `r_com` | `0`, `r_com_ref = e` | `e_r = e`, the sentinel |
| `k_p = k_d` | **`3.0`** | not an arbitrary sentinel — this is the canonical `SimConfig.ss_Kp_com` (`config.py:349`) |

`nq = 14` (canonical 7-DOF × 2 arms), `nc_max = 2` ⇒ `A_com` is `(3, 52)`.
Measured: `SimConfig.ss_Kp_com = 3.0` (`float`), `SimConfig.ss_Kd_com = 3.0` (`float`).

---

## 2. Shape trace, measured

### STATE A — canonical & current `main` (`sim_loop.py:957`, `np.diag([3.0]*3)`)

```
input gain object      : [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]
input gain shape       : (3, 3)
stored cfg.Kp_com      : shape (3, 3)  dtype float64
post-np.diag           : shape (3,)    value [3.0, 3.0, 3.0]      <-- EXTRACTED
(np.diag(Kp)) @ e_x    : type float64  shape ()   value 3.0        <-- SCALAR
A_com shape            : (3, 52)
```

### STATE B — intended vector contract (`np.full(3, 3.0)`)

```
input gain shape       : (3,)
stored cfg.Kp_com      : shape (3,)
post-np.diag           : shape (3, 3)  value [[3,0,0],[0,3,0],[0,0,3]]
(np.diag(Kp)) @ e_x    : type ndarray  shape (3,)  value [3. 0. 0.]
```

### STATE C — `WholeBodyQPConfig()` bare defaults (`wholebody_qp.py:162`)

```
default Kp_com         : shape (3,)  value [100.0, 100.0, 100.0]
post-np.diag           : shape (3, 3)
```

---

## 3. Sentinel results — `a_com_des` (= `b_com`, since `Jdot_dq_com = 0`)

### STATE A — canonical & current `main`

| error `e` | measured `a_com_des` | brief §5.2 rank-one prediction | match |
|---|---|---|---|
| `eₓ = [1,0,0]` | `[3. 3. 3.]` | `[3,3,3]` | ✔ |
| `e_y = [0,1,0]` | `[3. 3. 3.]` | `[3,3,3]` | ✔ |
| `e_z = [0,0,1]` | `[3. 3. 3.]` | `[3,3,3]` | ✔ |
| `e_cancel = [1,−1,0]` | `[0. 0. 0.]` | `[0,0,0]` | ✔ |
| `e_common = [1,1,1]` | `[9. 9. 9.]` | `[9,9,9]` | ✔ |

**VERDICT: RANK-ONE SUM-AND-BROADCAST — `k_p 𝟙𝟙ᵀ e`.** All five sentinels match the
brief's rank-one table exactly; none matches the diagonal table.

### STATE B — intended vector contract

| error `e` | measured `a_com_des` | brief §5.2 diagonal prediction | match |
|---|---|---|---|
| `eₓ` | `[3. 0. 0.]` | `[3,0,0]` | ✔ |
| `e_y` | `[0. 3. 0.]` | `[0,3,0]` | ✔ |
| `e_z` | `[0. 0. 3.]` | `[0,0,3]` | ✔ |
| `e_cancel` | `[ 3. -3.  0.]` | `[3,−3,0]` | ✔ |
| `e_common` | `[3. 3. 3.]` | `[3,3,3]` | ✔ |

**VERDICT: CLASSICAL DIAGONAL — `K_p e`.**

### STATE C — bare defaults

`a_com_des` for `eₓ` = `[100. 0. 0.]` with `k_p = 100` ⇒ **CLASSICAL DIAGONAL**.

---

## 4. Structural assertions (all passed)

```
[OK] STATE A is rank-one: e_x,e_y,e_z -> [3,3,3]; e_cancel -> 0; e_common -> [9,9,9]
[OK] STATE B is diagonal: e_x -> [3,0,0]; e_cancel -> [3,-3,0]
[OK] STATE C (bare defaults) is diagonal
[OK] STATE A annihilates every differential direction (e . 1 = 0 -> a_com_des = 0):
     rank <= 1 confirmed
[OK] STATE A common-mode gain is 3*kp = 9.0 per component, not kp = 3.0
[OK] the Kd channel has the SAME defect (v_com_ref=[1,-1,0] -> a_com_des = 0)

ALL ASSERTIONS PASSED
```

The rank-one claim is proven by construction rather than by three examples: three
independent differential directions — `[1,−1,0]`, `[0,1,−1]`, `[2,−1,−1]`, spanning
the whole 2-D subspace `{e : e·𝟙 = 0}` — each produce `a_com_des = 0` exactly. The
feedback operator's null space therefore has dimension ≥ 2, so its rank is ≤ 1.
Combined with `e_common → [9,9,9] = 3k_p·𝟙`, the operator is exactly `k_p 𝟙𝟙ᵀ`.

The derivative channel was probed separately, through `v_com_ref` rather than
`r_com_ref`: `v_com_ref = [1,−1,0]` with everything else zero also yields
`a_com_des = 0`. Both channels carry the defect.

---

## 5. Confidence and scope

- The three states were exercised through **one and the same production function**,
  `WholeBodyQP._com_task_rows`, differing only in the gain object handed to
  `WholeBodyQPConfig`. No behaviour was re-implemented, so there is no
  transcription risk between probe and production.
- `A_com` is `(3, 52)` in all states — the Jacobian assembly is unaffected. The
  defect is confined to `b_com`, i.e. to the desired-acceleration right-hand side.
- Scalar-broadcast is silent in NumPy 2.3.5: no warning, no error, correct output
  *shape* `(3,)`. Nothing short of a value check could have surfaced it, which is
  why source review alone had to be backed by this proof.

**Phase 1 conclusion: the canonical and current-`main` construction executes the
rank-one sum-and-broadcast law. The intended diagonal law is executed only by the
`WholeBodyQPConfig` defaults, which production never uses and the test suite always
uses.**

Next: `COM_GAIN_ACTIVATION_EVIDENCE.md` (Phase 2) — whether this law is actually
assembled into the canonical QP and whether it is load-bearing.
