# Step-2 QP isolation test — report

**Branch:** `claude/fk-bypass-aware-tuning`

**Test scenario:** `start_a=3, start_b=3, n_steps=1` — system initialised at (3, 3) docked configuration; swing arm 'b' commanded to anchor 4. **No prior drift from steps 0/1.**

## Side-by-side metrics

| metric | A: bypass=ON (current default) | B: bypass=OFF (FK ref to QP) |
|---|---|---|
| docked | False | False |
| d_grip_swing_min_mm | 20.144 | 20.144 |
| d_grip_swing_final_mm | 20.297 | 20.297 |
| e_torso_pos_peak_mm | 101.771 | 101.771 |
| e_torso_ori_peak_deg | 7.940 | 7.940 |
| e_ee_pos_peak_mm | 105.682 | 105.682 |
| e_ee_ori_peak_deg | 2.019 | 2.019 |
| tau_max_joint_peak_Nm | 4.200 | 4.200 |
| tau_w_peak_Nm | 5.498 | 5.498 |
| hw_peak_Nms | 2.980 | 2.980 |
| nmpc_infeas_count | 0 | 0 |
| T_phase_s | 10.100 | 10.100 |

## Verdicts

- **A_bypass_on**: FAILED at d_min=20.1 mm (dock_timeout)
- **B_bypass_off**: FAILED at d_min=20.1 mm (dock_timeout)

## Figures

- `A_bypass_on/fig_qp_isolation_tracking.png` — torso/EE refs vs actual + tracking errors
- `A_bypass_on/fig_qp_isolation_actuators.png` — joint torque, AOCS torque, AOCS momentum
- `B_bypass_off/fig_qp_isolation_tracking.png` — torso/EE refs vs actual + tracking errors
- `B_bypass_off/fig_qp_isolation_actuators.png` — joint torque, AOCS torque, AOCS momentum

## Standard diagnostic figures (per variant)

- `{A_bypass_on,B_bypass_off}/fig{1..10}_*.png` — full diagnostic suite
