# Step-2 isolation with extended SS margin — report

**Branch:** `claude/fk-bypass-aware-tuning`

**Test scenario:** `start_a=3, start_b=3, n_steps=1, t_ss_margin=5.0s` — clean (3,3) start, swing arm goes to anchor 4, with **5 extra seconds of grace beyond T_step** to let the QP's exponential approach converge.

## Headline metric

- d_grip_swing_min: **10.52 mm**
- d_grip_swing_final: 10.52 mm
- T_phase: 14.10 s
- Verdict: **FAILED at d_min=10.5 mm (dock_timeout)**

## Full metrics

| metric | value |
|---|---|
| docked | False |
| d_grip_swing_min_mm | 10.517 |
| d_grip_swing_final_mm | 10.517 |
| e_torso_pos_peak_mm | 101.771 |
| e_torso_ori_peak_deg | 7.940 |
| e_ee_pos_peak_mm | 105.682 |
| e_ee_ori_peak_deg | 2.019 |
| tau_max_joint_peak_Nm | 4.200 |
| tau_w_peak_Nm | 5.498 |
| hw_peak_Nms | 2.980 |
| nmpc_infeas_count | 0 |
| T_phase_s | 14.100 |

## Figures

- `long_hold/fig_qp_isolation_tracking.png` — torso/EE refs vs actual
- `long_hold/fig_qp_isolation_actuators.png` — joint, AOCS torque, AOCS momentum
- `long_hold/fig{1..10}_*.png` — full diagnostic suite
