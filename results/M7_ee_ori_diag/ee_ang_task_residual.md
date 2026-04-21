# M7 — EE angular task residual (A_swing, t = 3.6 s)

Source: `/home/user/CrawlBot_control/results/M7_ee_ori_diag/A_swing_t3p6.npz`. All values at the closest QP tick to t = 3.6 s, observed at `t_rel = 3.600000 s`.
Reconstructions follow `crawlbot/solvers/wholebody_qp.py`:
`a_torso_des` from lines 540-555, `a_ee_des` from 596-610,
`v17` task-consistent FF from 619-621 (`rcond = 1e-8`).

## (1) `J_ee_ang_dot_dq = (J̇_ee · q̇)[3:6]`

`[+0.064511, +0.015427, +0.042209]`  [rad/s²]

## (2) `omega_ee_actual = (J_ee · q̇)[3:6]`

`[-0.001759, -0.011801, +0.072144]`  [rad/s]

For reference, `omega_ee_ref = v_ee_ref[3:6] = [-0.000000, -0.000000, -0.000000]` rad/s.

## (3) `a_ee_des[3:6]` reconstruction

Components fed into the QP at this tick:

- SwingPlanner `a_ee_ff[3:6]` (alpha):  `[-0.000000, -0.000000, -0.000000]`
- v17 task-consistent FF `J_ee · J_torso^+ · a_torso_des` (angular rows):
  `[-0.001278, -0.018463, -0.005805]`
- `a_torso_des = Kp_t·e_6d_t − Kd_t·(J_torso·q̇)` (constant ref, A_swing):
  `[+0.060295, +0.029026, +0.010919, -0.001278, -0.018463, -0.005805]`
- Orientation error `e_ori_ee = log3(R_ee^T R_ee_ref)` [rad]: `[+0.012085, -0.020306, -0.078384]`
- `omega_ref − omega_actual` [rad/s]: `[+0.001759, +0.011801, -0.072144]`
- `Kp_ee_ang = [+6.000000, +6.000000, +6.000000]`,  `Kd_ee_ang = [+4.500000, +4.500000, +4.500000]`

**`a_ee_des[3:6] = [+0.079147, -0.087198, -0.800759]`** [rad/s²].

Full 6D for completeness: `a_ee_des = [+0.046119, +0.009995, +0.033817, +0.079147, -0.087198, -0.800759]`.

## (4) Standalone least-squares — angular EE task alone

Solve `min_x ‖J_ee[3:6,:] · x + J_ee_ang_dot_dq − a_ee_des[3:6]‖²` over `x ∈ R²⁰`, no other constraints.

Per `np.linalg.lstsq(A, b, rcond=None)` with `A = J_ee[3:6,:]`, `b = a_ee_des[3:6] − J_ee_ang_dot_dq`:

`qdd_lsq` (length 20):
`[-0.000000, +0.000000, +0.000000, +0.191923, -0.166417, -0.088269, +0.000000, +0.000000, +0.000000, +0.000000, +0.000000, +0.000000, +0.000000, +0.085842, -0.094270, -0.251409, -0.063010, -0.045833, +0.018944, -0.256891]`

Residual `A·qdd_lsq − b`: `[-0.000000, -0.000000, -0.000000]`

**‖residual‖ = 1.907880e-16**

## (5) Actual QP solution plugged back in

Reconstruction of the angular EE task acceleration the QP realises with its own `qdd`:

`J_ee[3:6, :] · qdd_qp + J_ee_ang_dot_dq` = `[+0.080440, -0.088815, -0.804938]` rad/s²

Difference from the desired `a_ee_des[3:6]`:

`diff = [+0.001293, -0.001617, -0.004179]`

**`‖diff‖ = 4.664224e-03` rad/s²**

## Auxiliary numerics (for traceability)

`qdd_t_qp` (base): `[+0.004277, +0.006057, +0.000004, +0.007661, -0.019581, +0.008527]`

`qdd_qp` (joints, 14): `[+0.025304, +0.005645, -0.008485, +0.006330, -0.006371, -0.005604, -0.002250, -0.038401, -0.023647, -0.022409, +0.228547, +0.091374, -0.148823, -0.927661]`

`Kp_torso` = `[+6.000000, +6.000000, +6.000000, +6.000000, +6.000000, +6.000000]`,  `Kd_torso` = `[+5.000000, +5.000000, +5.000000, +5.000000, +5.000000, +5.000000]`

`v_ee_ref` = `[+0.205866, +0.000000, -0.000470, -0.000000, -0.000000, -0.000000]` (linear; angular)
