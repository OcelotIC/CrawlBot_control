# M7 dock-activation velocity audit (T11 closed run)

Log: `Misc/runs/M7_1pct_1step_v21_armature_fix_damping_off_mapping_off/sim_log.json`. Log cadence `dt = 0.100 s`. Dock event at `t = 7.310 s` (log index `k_dock = 72`). Last SS tick: `k = 71` at `t = 7.210 s`.

## Method

No new simulation. All velocities are finite-differences of logged fields, then combined via the rigid-body formula `v_site = v_body_cm + ω × (R_body · r_off_body)` for anchor_3b (body = `structure`, site offset `[0.4, -0.3, 0.025]` from MJCF line 101). Gripper_b velocities are finite-differences of `p_ee` (linear) and `q_ee` (angular via axis-angle log of `R_k^T · R_{k+1}`). Structure angular velocity is the logged `omega_s` (MuJoCo `qvel[3:6]` for the structure free joint, world frame). Central differences are used interior-of-trace; one-sided at boundaries.

## 1. At dock activation (t = 7.31 s, k = 72)

Approach direction `(p_anchor − p_gripper)` (world): length `1800.257` mm, unit vector `[-0.0007047109453259853, -9.613730974376311e-05, -0.9999997470700186]`.

- `‖v_rel_lin‖` = **`1.976899e-02`** m/s
- `‖v_rel_ang‖` = **`6.109928e-03`** rad/s
- Component of `v_rel_lin` along approach direction (positive = gripper closing on anchor): **`-8.169872e-03`** m/s

Vector components (world):

- `v_rel_lin` = `[0.017899655455676386, -0.001967524385346153, 0.008157448864051843]` m/s
- `v_rel_ang` = `[-0.002555185980785923, -0.005544608373607391, 0.0002440519080358464]` rad/s

## 2. Profile over last 1 s of SS (t ∈ [6.3, 7.3] s)

| k | t [s] | ‖v_rel_lin‖ [m/s] | ‖v_rel_ang‖ [rad/s] |
|---|---|---|---|
| 62 | 6.310 | 8.652646e-02 | 3.344770e-04 |
| 63 | 6.410 | 7.778492e-02 | 3.034625e-04 |
| 64 | 6.510 | 6.913819e-02 | 2.773978e-04 |
| 65 | 6.610 | 6.065165e-02 | 2.571445e-04 |
| 66 | 6.710 | 5.239319e-02 | 2.432543e-04 |
| 67 | 6.810 | 4.443345e-02 | 2.358067e-04 |
| 68 | 6.910 | 3.684603e-02 | 2.343384e-04 |
| 69 | 7.010 | 2.970772e-02 | 2.379934e-04 |
| 70 | 7.110 | 2.309922e-02 | 2.458310e-04 |
| 71 | 7.210 | 2.848505e-02 | 1.032825e-02 |

## 3. Structure body velocity at dock tick

- `v_struct_lin` (world, finite-diff of `struct_pos`): `[-0.0001330318690809392, 1.0248815645632402e-05, -8.003548391477544e-05]` m/s, `‖·‖ = 1.555898e-04` m/s
- `ω_struct` (world, logged `omega_s`): `[8.361124763731386e-06, 0.00029607119738609577, -5.533282878362277e-05]` rad/s, `‖·‖ = 3.013134e-04` rad/s
- For context, `struct_pos[k_dock]` = `[-0.0013079912514822535, 1.922592584476207e-05, -1.8002660565162456]`, `‖struct_quat‖` ≈ 1 (`1.000000`).

## 4. SwingPlanner reference velocity (from logged `p_ee_ref` / `q_ee_ref`)

`T_step = 7.284 s` (from `preplanner_T_steps[0]`). Nearest log tick ≤ T_step: `k = 71` (t = 7.210 s). Dock tick: `k = 72` (t = 7.310 s).

| tick | t [s] | ‖v_ref_lin‖ [m/s] | ‖v_ref_ang‖ [rad/s] | ‖v_rel_actual_lin‖ [m/s] | ‖v_rel_actual_ang‖ [rad/s] |
|---|---|---|---|---|---|
| k_ss_last = 71 | 7.210 | 1.909995e-02 | 0.000000e+00 | 2.848505e-02 | 1.032825e-02 |
| k_dock = 72 | 7.310 | 1.385337e-02 | 0.000000e+00 | 1.976899e-02 | 6.109928e-03 |

## 5. Post-dock residual relative motion (t ∈ [7.31, 7.51] s)

| k | t [s] | ‖v_rel_lin‖ [m/s] | ‖v_rel_ang‖ [rad/s] |
|---|---|---|---|
| 72 | 7.310 | 1.976899e-02 | 6.109928e-03 |
| 73 | 7.410 | 1.161186e-03 | 1.678186e-02 |
| 74 | 7.510 | 7.829625e-05 | 6.190236e-03 |

**`efc_force` on the newly activated weld is not in the committed log** — `SimLog` does not capture MuJoCo `efc_*` arrays. Only the relative-velocity signal above is available from the committed data.

## 6. Post-T_step velocity decay

`T_step = 7.284 s` (`preplanner_T_steps[0]`). `d_grip_swing` pulled from the committed log and converted to mm; `v_rel_lin` computed as in sections 1–5 (finite-diff + rigid-body anchor).

| label | k | t [s] | d [mm] | ‖v_rel_lin‖ [m/s] | v_rel_lin along approach [m/s] |
|---|---|---|---|---|---|
| (1) last SS, t ≤ T_step | 71 | 7.210 | 3.892 | 2.848505e-02 | -1.062881e-02 |
| (2) first post-T_step, t > T_step | 72 | 7.310 | 2399.840 | 1.976899e-02 | -8.169872e-03 |
| (3) first tick w/ t ≥ T_step+0.5 AND d<5mm | — | — | — | — | — |
| (4) first tick w/ d<5mm | 71 | 7.210 | 3.892 | 2.848505e-02 | -1.062881e-02 |

**Note on the `d` column.** `d` above is pulled verbatim from `log.d_grip_swing`, which `sim_loop` computes as |gripper − scheduled-target-anchor|. During SS the scheduled target is anchor index 3 (`anchor_4b`, the dock target), so row (1) is the real gripper-to-dock distance. During the trailing DS branch (`sim_loop.py:1383-1388`), `_step` is called with `target_anchor = 0`, so `log.d_grip_swing` at k ≥ 72 reports distance to anchor index 0 (`anchor_1b` at body-local `(-2.0, -0.3, 0.025)`), which is ~2.4 m from the welded gripper — this explains the 2399.840 mm reading at row (2). It does not affect rows (1), (3), (4)'s interpretation of dock proximity — but it means the "d < 5 mm" mask is satisfied only at k=71 (SS with scheduled target = dock target).

**Note on item (3).** The dock actually activates at `t = 7.310 s`, so the post-dock phase enters DS at that tick. The weld is therefore **active** for every tick with `t ≥ T_step + 0.5` in the committed log. No counterfactual (dock gate delayed 500 ms with weld still inactive) is simulable from the committed log — it would require a new simulation with the dock gate logic altered. Row (3) reads "—" because the `log.d_grip_swing < 5 mm` mask is never True for `t ≥ T_step + 0.5` (see note above on the `d` column definition during DS).
