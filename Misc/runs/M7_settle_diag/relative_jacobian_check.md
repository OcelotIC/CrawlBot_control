# M7 — relative-Jacobian check (d0_a0, k=10)

t = 0.100 s. Deterministic replay of the first 11 ticks of the d0_a0 settle; MJCF mutated to (damping=0, armature=0) transiently and restored on exit.

## 1. `J_rel_pin` = `J(gripper_*)` − `J(anchor_2*)`

The URDF (`models/VISPA_crawling_fixed.urdf`) roots `Link_0` to `world` with a fixed joint; the MJCF body `structure` (a second free-joint body in MuJoCo carrying the anchor sites) is **not present in the URDF**. In Pinocchio's kinematic tree anchor `anchor_2*` would be a world-fixed point, so `J_pin(anchor_2*) ≡ 0` by construction. Therefore in Pinocchio `J_rel == J_abs` rigorously.

Shape of `J_rel_pin_A` = `J(gripper_a)` − 0  : `(6, 20)`

First row of `J_rel_pin_A` (same as first row of `J(gripper_a)`):

`[+7.939228e-01, +4.737014e-01, +3.811740e-01, +1.329862e-01, +4.417339e-01, -8.259501e-01, +5.167005e-01, -2.078246e-01, -2.609293e-02, -8.641878e-01, -3.763593e-03, -3.243725e-01, +0.000000e+00, +0.000000e+00, +0.000000e+00, +0.000000e+00, +0.000000e+00, +0.000000e+00, +0.000000e+00, +0.000000e+00]`

First row of `J_rel_pin_B`:

`[+7.939228e-01, +4.737014e-01, +3.811740e-01, -2.287873e-01, +7.613281e-01, -4.696090e-01, +0.000000e+00, +0.000000e+00, +0.000000e+00, +0.000000e+00, +0.000000e+00, +0.000000e+00, +0.000000e+00, -1.543612e-01, -8.082923e-01, +4.813424e-01, -6.110794e-01, -1.597951e-01, -2.201749e-01, +0.000000e+00]`

## 2. `P_weld_rel_pin_A`  (relative Jacobian, arm A)

`P_rel_pin_A = (J_rel_pin_A · v_pin)^T · λ_qp[0:6] = +1.186413e+02` W

For cross-check, absolute: `P_abs_pin_A = +1.186413e+02` W (equal by construction; `J_anchor_pin = 0`).

## 3. `P_weld_rel_pin_B` + total

`P_rel_pin_B = (J_rel_pin_B · v_pin)^T · λ_qp[6:12] = +1.541386e+02` W

**Total Pinocchio relative-Jacobian P_weld = +2.727799e+02 W** (absolute-Jacobian total = +2.727799e+02 W).

## 4. Anchor and gripper site velocities in MuJoCo world

Read from `mj_data.site_xvelp` (linear) and `mj_data.site_xvelr` (angular), populated by `mj_forward`. World frame.

| site | linear [m/s] | angular [rad/s] | ‖lin‖ [m/s] | ‖ang‖ [rad/s] |
|---|---|---|---|---|
| `anchor_2a` | `[+1.9852e-05, +3.1073e-04, -1.4664e-04]` | `[+3.8985e-04, -1.5546e-04, -1.5066e-04]` | `3.4416e-04` | `4.4592e-04` |
| `gripper_a` | `[+4.3300e-03, +4.0459e-03, -1.2375e-03]` | `[-1.9679e-01, +3.4196e-01, +5.2248e+00]` | `6.0539e-03` | `5.2397e+00` |
| `anchor_2b` | `[-7.0543e-05, +3.1073e-04, -3.8055e-04]` | `[+3.8985e-04, -1.5546e-04, -1.5066e-04]` | `4.9633e-04` | `4.4592e-04` |
| `gripper_b` | `[-7.6419e-04, +6.3509e-03, -6.8528e-03]` | `[-7.2110e-01, +1.9197e-02, +5.6451e+00]` | `9.3744e-03` | `5.6910e+00` |

Structure free-joint `mj_qvel[0:6]` (lin + ang in world): `[-2.1459e-05, +1.3968e-04, -7.7040e-05, +3.8985e-04, -1.5546e-04, -1.5066e-04]`, ‖lin‖ = `1.6096e-04` m/s, ‖ang‖ = `4.4592e-04` rad/s.

Per-contact RELATIVE velocities `v_gripper − v_anchor` (world frame, what an idealised weld residual would see):

- A: `lin = [+4.3101e-03, +3.7352e-03, -1.0909e-03]`, `ang = [-1.9718e-01, +3.4212e-01, +5.2249e+00]`, `‖lin‖ = 5.8068e-03`, `‖ang‖ = 5.2398e+00`
- B: `lin = [-6.9365e-04, +6.0402e-03, -6.4722e-03]`, `ang = [-7.2149e-01, +1.9353e-02, +5.6452e+00]`, `‖lin‖ = 8.8800e-03`, `‖ang‖ = 5.6912e+00`

## 5. MuJoCo `eq_weld` frame convention

From the MuJoCo reference documentation (equality constraints, `<weld>`), an `mjEQ_WELD` produces 6 `efc` rows whose residual is the relative pose error between `site1` (body1-attached) and `site2` (body2-attached). Internally MuJoCo computes the relative body Jacobian `J_rel = J(body1) − J(body2)` and writes it into `efc_J` rows. The `efc_force` entries are the Lagrange multipliers enforcing those 6 scalar residuals.

Key implication recorded, no interpretation: MuJoCo `efc_J` is intrinsically **relative** (includes motion of both connected bodies); Pinocchio's `J_c` here is **absolute with anchor=world** because the `structure` body exists only in the MJCF. The `mj_struct_v` block reported above is exactly the degrees of freedom present in MuJoCo but absent in the URDF.
