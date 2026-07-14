# Phase ADJ-CONV — sense of the torso/structure translation p + numerical adjoint-transport verification

For the §V rewrite `L_robot^struct = Ad^{-T}(g)·(Π_ω H_r q̇_r)`. Convention/verification pass on the
committed canonical run (runfix @`5ab2c91`, branch `j2/ds-active-rework`). READ + numerical check;
**no `crawlbot/` change, no new sim.** Script: `scripts/diag_adjconv_verify.py`; numbers:
`adjconv_verify.json`.

## VOLET 1 — sense of p (READ)
The controller does **not** store a matrix literally named `T_struct_torso`; the torso-in-structure
pose is built inside the MuJoCo→Pinocchio conversion and IS the Pinocchio floating-base placement.

- **`crawlbot/core/state_conversions.py:86-88`** (`mujoco_to_pinocchio`):
  ```
  dp      = p_t - p_s                 # torso minus structure, in world
  p_local = R_s.T @ dp                # -> assigned to pin_q[0:3] at :98
  R_local = R_s.T @ R_t
  ```
- **p = `R_s^T (p_torso − p_struct)` = position of the torso (Pinocchio base origin) measured FROM the
  structure origin, expressed in R_s.** The structure origin = structure CoM = origin of R_s
  (`crawlbot/aocs/force_estimator.py:69-70`). ⇒ the stored pose is **`T_struct_torso`** (torso pose
  in the structure frame); **p points structure→torso**. It is **NOT** the inverse `T_torso_struct`.
- Velocity is also referred to the structure (`:89-90`, subtracts `v_s`, `ω_s×dp`), so the Pinocchio
  momentum built from this state IS the "robot momentum seen from the platform."
- **How `Π_ω H_r q̇_r` is computed:** the controller takes the robot **centroidal** momentum via
  `pin.computeCentroidalMomentum` — **`crawlbot/core/robot_interface.py:312-314`**:
  `h_centroidal = data.hg.vector; L_com = h_centroidal[3:6]` (angular part, at the **robot CoM**, in
  R_s; components R_s-aligned = mixed, since the Pinocchio model is expressed in R_s). The paper's
  `H_t·ẋ_t + H_tq·q̇` reduced at the **torso** frame is the *equivalent* momentum at a different
  reduction point; the code instead reduces at the robot CoM and shifts to the structure CoM by
  `+ r_com × P` (V-C.2). Both are verified equivalent below.

## VOLET 2 — numerical verification (MEASURE, the decisive check)
Per-tick full config is **not** logged; the run's **42 full-state snapshots** (`[t, qpos(31),
qvel(29), label]`, spanning initial / release / frame / dock across all 6 steps) are the achievable
full-state set. The transport is kinematic (holds at every config) so 42 diverse configs is rigorous.
Source: `figC_qpcond/sim_log.json` snapshots (identity-confirmed == committed runfix @`5ab2c91`,
ABL-HDOT-2 Step 0). Method, per snapshot: convert with the code's own `mujoco_to_pinocchio`;
`g = data.oMi[1]` (Pinocchio base placement = the stored `struct_M_torso`); **RHS** = independent
per-body momentum about the structure origin `Σ_i oMi.act(Y_i v_i)`; momentum at torso =
`g.actInv(RHS)`; **LHS** = the `[L;P]` force adjoint `L=R·L+p̂R·P, P=R·P` with the stored `g`.

| check | max\|dev\| over 42 snaps | result |
|---|---|---|
| **LHS = Ad^{-T}(g)·(mom@torso), stored g,  vs RHS (per-body)** | **3.6e-15** | **MATCH** |
| `g.act` (Pinocchio force adjoint) vs RHS | 3.6e-15 | explicit `[L;P]` formula == SE(3) transport |
| **V-C.2** `L_com + r_com×P` vs RHS | 3.6e-15 | **MATCH** (validates existing V-C.2 term) |
| **INVERSE sense** (`g^{-1}`, as if `T_torso_struct`) vs RHS | **20.4** ( = 2\|p×P\|, ≤19.2) | **OPPOSITE/OFFSET** |

- `p = R_s^T(p_torso − p_struct)` reproduced from raw `qpos` at every snapshot (assert `<1e-9`).
  `|p| ∈ [0.876, 2.369] m` (grows monotonically as the robot crawls away from the structure CoM);
  mean `p = [+1.175, −0.701, −0.327]`.

**VERDICT: MATCH.** The adjoint with the code's stored `g` reproduces the independent robot angular
momentum about the structure CoM to machine precision. **p's sense is correct as stored
(`T_struct_torso`, p = torso-from-structure); no inversion / no `g^{-1}` is needed.** The inverse
sense would be wrong by exactly `2|p×P|` (up to 19.2 N·m·s), confirming the sense is load-bearing and
the stored one is the correct pairing for `Ad^{-T}`.

## SIGN CROSS-CHECK vs V-C.2
Both reductions reproduce the same `L_struct` (each vs the per-body RHS at 3.6e-15):
`L_struct = L_torso + p̂·P` (adjoint) `= L_com + r_com×P` (V-C.2). The adjoint's angular-row
moment-arm term is **`+ p̂R·P = + p×P`** (R = I in the R_s-mixed convention); V-C.2's is
**`+ r_com×P`**. **Same `+` sign convention** — both are `+(lever measured from the structure
origin)×P`. The levers differ (`p` localizes the **torso** origin, `|p|` 0.88–2.37 m; `r_com`
localizes the **robot CoM**), so the two isolated terms are not numerically equal — but they sum
consistently to the same `L_struct`, and the transport **sign** agrees with V-C.2. Confirmed.

## Deliverable summary
- **Stored pose + p sense:** `state_conversions.py:86-88,98` → `T_struct_torso`, p = R_s^T(p_torso −
  p_struct) = torso-from-structure. Momentum input: `robot_interface.py:312-314`.
- **LHS vs RHS:** MATCH (3.6e-15); inverse sense OPPOSITE (20.4 = 2|p×P|).
- **Corrected g sense:** **none needed** — stored sense is correct.
- **Sign vs V-C.2:** same `+(lever-from-structure-origin)×P` convention.
- **Caveat:** verified at 42 full-state snapshots (per-tick full config not logged); kinematic
  identity ⇒ representative-config verification is conclusive.
