# M7 — Jacobian/velocity convention audit (d0_a0, k=10)

t = 0.1000 s. MJCF mutated to (damping=0, armature=0) transiently and restored on exit.

## 1. Pinocchio side — convention (source inspection)

- `crawlbot/core/robot_interface.py:310-313` — `J_tool_a/b = pin.getFrameJacobian(model, data, fid, **pin.LOCAL_WORLD_ALIGNED**)`. Each tool Jacobian is (6, nv). Rows are `[v_lin(3), v_ang(3)]` (linear-first, axes aligned with Pinocchio's working "world" = the structure frame).
- `get_contact_jacobians(active_A, active_B)` (`robot_interface.py:421-443`) `np.vstack`s the two tool Jacobians in `[A, B]` order → shape `(12, 20)`.
- `crawlbot/core/state_conversions.py:43-104` — `mujoco_to_pinocchio` converts MuJoCo world-frame free-joint velocities to Pinocchio structure-frame, storing `pin_v[0:3] = v_torso_local_linear`, `pin_v[3:6] = omega_torso_local_angular`. Linear-first, structure-frame axes, at torso frame.
- λ layout (`crawlbot/solvers/wholebody_qp.py:340-355`): the dynamics equation `H·q̈ + C = B·τ + J_c^T·λ` uses the stacked `J_contacts` as-is; therefore λ has the same row ordering as `J_contacts` — `[force(3), torque(3)]` per contact, structure-frame axes.

## 2. MuJoCo side — convention (model + data)

`mj_model.nv = 29`, `mj_model.nq = 31`, `mj_model.neq = 12`. At k=10, `mj_data.nefc = 12`.

Column (v-space) assignment from `mj_model.jnt_dofadr` and `jnt_type`:

| jid | name | type | dof_start | dof_count |
|---|---|---|---|---|
| 0 | `structure_free` | free(6) | 0 | 6 |
| 1 | `rw_x` | hinge(1) | 6 | 1 |
| 2 | `rw_y` | hinge(1) | 7 | 1 |
| 3 | `rw_z` | hinge(1) | 8 | 1 |
| 4 | `root` | free(6) | 9 | 6 |
| 5 | `Joint_1_a` | hinge(1) | 15 | 1 |
| 6 | `Joint_2_a` | hinge(1) | 16 | 1 |
| 7 | `Joint_swivel_a` | hinge(1) | 17 | 1 |
| 8 | `Joint_3_a` | hinge(1) | 18 | 1 |
| 9 | `Joint_4_a` | hinge(1) | 19 | 1 |
| 10 | `Joint_5_a` | hinge(1) | 20 | 1 |
| 11 | `Joint_6_a` | hinge(1) | 21 | 1 |
| 12 | `Joint_1_b` | hinge(1) | 22 | 1 |
| 13 | `Joint_2_b` | hinge(1) | 23 | 1 |
| 14 | `Joint_swivel_b` | hinge(1) | 24 | 1 |
| 15 | `Joint_3_b` | hinge(1) | 25 | 1 |
| 16 | `Joint_4_b` | hinge(1) | 26 | 1 |
| 17 | `Joint_5_b` | hinge(1) | 27 | 1 |
| 18 | `Joint_6_b` | hinge(1) | 28 | 1 |

For `mjJNT_FREE`, MuJoCo documents qvel ordering as `[lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]` in the world frame (linear-first).

Equality-constraint types (for each of the 12 `<weld>` elements):

| eid | name | type |
|---|---|---|
| 0 | `grip_a_to_1a` | weld(6) |
| 1 | `grip_a_to_2a` | weld(6) |
| 2 | `grip_a_to_3a` | weld(6) |
| 3 | `grip_a_to_4a` | weld(6) |
| 4 | `grip_a_to_5a` | weld(6) |
| 5 | `grip_a_to_6a` | weld(6) |
| 6 | `grip_b_to_1b` | weld(6) |
| 7 | `grip_b_to_2b` | weld(6) |
| 8 | `grip_b_to_3b` | weld(6) |
| 9 | `grip_b_to_4b` | weld(6) |
| 10 | `grip_b_to_5b` | weld(6) |
| 11 | `grip_b_to_6b` | weld(6) |

For `mjEQ_WELD`, MuJoCo generates 6 `efc` rows per weld. Per MuJoCo source/docs, the efc row ordering for a weld is **[rotation(3), translation(3)]** — angular first, then positional. `efc_force` units per row are the Lagrange multipliers matching the corresponding `efc_J` row (in the frame MuJoCo used internally; see the cross-convention sweep below).

## 3. Eight P_weld variants

For each side, four swap variants of the inner product `(J · v) · λ`:
- **a (as-is)**: the 6-tuples enter in native ordering.
- **b (swap v base lin/ang)**: for Pinocchio, swap `v_pin[0:3] ↔ v_pin[3:6]` (torso free-joint block). For MuJoCo, swap BOTH `qvel[0:3]↔[3:6]` (structure free-joint) and `qvel[9:12]↔[12:15]` (torso free-joint).
- **c (swap λ per contact)**: within each 6-tuple of `λ_qp` (Pin) or `efc_force` (MJ), swap the first 3 components with the last 3.
- **d (both swapped)**: apply both b and c.

| variant | Pin P_weld [W] | MuJoCo P_weld [W] | rel agreement |
|---|---|---|---|
| a (as-is) | `+2.727799e+02` | `+2.152831e+02` | `2.356e-01` |
| b (swap v base lin/ang) | `+2.756183e+02` | `+2.204857e+02` | `2.223e-01` |
| c (swap λ per contact) | `-4.367460e+01` | `+1.943297e+02` | `2.000e+00` |
| d (both swapped) | `-4.430046e+01` | `+1.956284e+02` | `2.000e+00` |

Best cross-side match (smallest relative disagreement):

- Variant `b (swap v base lin/ang)` — Pin = `+2.756183e+02` W, MJ = `+2.204857e+02` W.

## 4. Kinetic energy cross-check

MuJoCo convention (per `mjdata.h`): `mj_data.energy = [potential, kinetic]`. `mj_energyPos` and `mj_energyVel` are called explicitly before reading.

- `T_pin = ½ v_pin^T H_pin v_pin` = `2.226965e-01` J
- `mj_data.energy[1]` (pre-step, kinetic)  = `2.261085e-01` J
- `mj_data.energy[1]` (post-step, kinetic) = `2.210675e-01` J
- `mj_data.energy[0]` (pre-step, potential) = `0.000000e+00` J
- `mj_data.energy[0]` (post-step, potential)= `0.000000e+00` J
- manual `0.5·qvel·M·qvel` (pre-step)  = `2.261085e-01` J
- manual `0.5·qvel·M·qvel` (post-step) = `2.210675e-01` J
- relative diff `|T_pin − T_mj_kin_pre| / max(|T_pin|, |T_mj_kin_pre|)` = `1.509e-02`

## 5. MJCF restoration

Verified on exit: `damping="0.05"`, `armature="0.05"`.
