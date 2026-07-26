# M7 — EE angular task vs torso null space (A_swing, t = 3.6 s)

Source matrices: `/home/user/CrawlBot_control/Misc/runs/M7_ee_ori_diag/A_swing_t3p6.npz` (dumped by `Misc/scripts/bisect_qp_cascade.py --case A_swing`).
Numerical-rank tolerance: `tol = 1e-08` (applied uniformly).
Damped pseudo-inverse: `rcond = 1e-8` (matches `wholebody_qp.py:582,620`).

**Tick:** `t_rel = 3.600000 s` of A_swing SS (closest QP tick at or after the 3.6 s target).

**Shapes:** `J_ee = (6, 20)`, `J_torso = (6, 20)`, `J_ee_ang = J_ee[3:6, :] = (3, 20)`, `N_torso = (20, 20)`.

## (1) `rank(J_ee[3:6, :])`

`rank = 3` (out of max 3).

Singular values of `J_ee[3:6, :]`:
`[+2.000641, +1.898704, +1.546725]`

## (2) `rank(J_ee[3:6, :] @ N_torso)`

`rank = 3` (out of max 3).

For reference: `rank(J_torso) = 6` (out of max 6); `rank(N_torso) = 14` (= n − rank(J_torso) = 20 − 6 = 14).

## (3) Singular values of `J_ee[3:6, :] @ N_torso`

`[+1.732791, +1.614025, +1.179983]`

## (4) Row-space intersection between `J_ee[3:6, :]` and `J_torso`

Stacked-rank test:

- `rank(J_ee_ang)        = 3`
- `rank(J_torso)         = 6`
- `rank([J_ee_ang; J_torso]) = 9`
- `dim(intersection)     = rank(J_ee_ang) + rank(J_torso) − rank(stack) = 0`

Null-space deficiency check (must agree with the stacked-rank test):

- `rank(J_ee_ang)            = 3`
- `rank(J_ee_ang @ N_torso)  = 3`
- `dim(intersection)         = rank(J_ee_ang) − rank(J_ee_ang @ N_torso) = 0`

Per-row decomposition into the row spaces of `J_torso` and `N_torso`:

| row | ‖row‖ | ‖proj onto row(J_torso)‖ / ‖row‖ | ‖proj onto null(J_torso)‖ / ‖row‖ |
|---|---|---|---|
| J_ee[3, :] | 1.812801 | 0.551632 | 0.834087 |
| J_ee[4, :] | 1.813644 | 0.551376 | 0.834257 |
| J_ee[5, :] | 1.850526 | 0.540387 | 0.841417 |
