# Phase FULL-DIAG-EXPORT — complete per-tick diagnostic export (userw2, committed 3871de4)

**Branch** `j2/ds-active-rework` · export-only from the committed run, NO re-run, NO canonical change ·
pushed, never merged. Artifacts: `results/j2_adjconv/userw2_fulldiag.csv` (1860 ticks, 59 cols),
`userw2_fulldiag_meta.json`; script `scripts/diag_full_diag_export.py` (reusable: `--run-dir results/figC_<tag>`).

**Headline finding (fly-by vs weld):** the per-step "dock" I have been quoting is the **min-over-swing**
`d_grip` — a *fly-by* closest approach. For **step 2 that fly-by (3.006 mm) did NOT weld**; the gate fired
later at **4.890 mm** (the swing flew through 3.006 mm too fast — twist gate — and welded when it slowed).
So by **weld distance** userw2's worst-of-6 is **4.890 mm (step 2), margin 0.110** — not 4.593 mm / 0.407
(which is the fly-by metric on step 1). Both are true under their own definition; the weld is the physical dock.

## B — AT-WELD (first `docked=True`) vs MIN-OVER-SWING, all 6 steps
From `dock_events` (weld instant) and per-tick `d_grip_swing` (fly-by min):

| step | weld t [s] | d @ weld [mm] | d min-swing [mm] | fly-by gap [mm] | ori @ weld [°] | twist @ weld |
|---|---|---|---|---|---|---|
| 0 | 2.91 | 2.570 | 2.565 | +0.005 | 0.17 | 0.0266 |
| 1 | 14.41 | 4.590 | 4.593 | −0.003 | 0.04 | 0.0071 |
| **2** | 26.96 | **4.890** | **3.006** | **+1.884** | 0.03 | 0.0047 |
| 3 | 38.08 | 4.390 | 4.391 | −0.001 | 0.04 | 0.0068 |
| 4 | 44.44 | 2.490 | 2.485 | +0.005 | 0.19 | 0.0329 |
| 5 | 55.08 | 4.490 | 4.494 | −0.004 | 0.04 | 0.0075 |

- Steps 0,1,3,4,5: weld ≈ closest approach (|gap| ≤ 0.005 mm) — they welded where they got closest.
- **Step 2 is a fly-by** (3.006 → welds at 4.890): consistent with the same phenomenon DOCK-CAUSE saw on
  canonical step 4 (transient dip that doesn't weld). All ori/twist at weld are tiny (ori ≤0.19°, twist
  ≤0.033 ≪ 0.05 gate) — the docks are clean when they fire; the position at weld is the discriminator.
- **weld-distance docks:** `2.57 / 4.59 / 4.89 / 4.39 / 2.49 / 4.49` → worst 4.890 (step 2), margin 0.110.

## CSV columns (59), grouped
`t_s, phase, step_index, swing_arm` ·
**A gate:** `d_grip_swing_mm, ori_err_deg, pos_ok, ori_ok, gate_eval, twist_norm_at_gateeval,
docked_at_gateeval` ·
**E momentum:** `Hdot_s_realized_cont_{xyz}_Nm, Hdot_s_realized_norm_Nm, Hdot_s_planned_ss_{xyz}_Nm,
Hdot_s_planned_norm_Nm` · `tauw_{xyz}_Nm, hw_{xyz}_Nms` ·
**C CoM:** `r_com_{xyz}_m, r_com_ref_{xyz}_m, e_com_m, v_com_{xyz}_mps, v_com_ref_{xyz}_mps` ·
**D torso:** `theta_s_{xyz}_deg, e_torso_pos_m, e_torso_ori_deg, p_torso_{xyz}_m, p_torso_ref_{xyz}_m` ·
EE: `e_ee_pos_mm, e_ee_ori_deg` ·
**F QP:** `qp_ok, nmpc_ok, qp_time_ms, nmpc_time_ms, nmpc_iterations, lambda_qp_norm, lambda_ref_norm,
tau_max_joint_Nm`.

## Availability report (per channel: present / recomputed / sparse / MISSING + file:line)

| channel | status | source / file:line |
|---|---|---|
| d_grip_swing (pos input) | **logged/tick** | sim_loop.py:3546 (SS), :1014 (DS) — `_gripper_distance` :1238 |
| ori_err_deg (gate cond 2) | **recomputed/tick** | from `q_ee` (anchor=I in struct frame), validated vs `dock_gate_trace` to ≤0.13° (mean 0.02°); gate fn `_gripper_ori_err_deg` :1261 |
| pos_ok, ori_ok | **recomputed/tick** | `d<weld_radius(5mm)` config.py:35; `ori<dock_ori_threshold_deg(5°)` config.py:42 |
| twist_norm (gate cond 3) | **SPARSE** (103 gate-eval ticks) | `dock_gate_trace` :1333 (`_weld_relative_twist` :1276). NOT per-tick |
| twist_lin / twist_ang split | **MISSING** | not logged anywhere; needs qpos/qvel → instrumented re-run |
| per-tick twist / vel_ok / docked | **MISSING per-tick** (present only at 103 gate-eval + 6 weld ticks) | needs qpos/qvel (`mj_jacSite`); `qpos`/`qvel` NOT logged. Re-run to add per-tick twist |
| dock weld instant (d/ori/twist@weld) | **logged** (6) | `dock_events` (weld) — used for table B |
| r_com, r_com_ref, e_com | **logged/tick** | :3566 / :3570 / :3571 (SS). ⚠ In **DS** r_com_ref = held-current r_com and e_com=0 (:1040-1041) — the NMPC target is meaningful only in SS |
| v_com, v_com_ref | **logged/tick** | :3657 / :3658 (SS); DS v_com_ref=0 (:1043) |
| e_torso_pos, e_torso_ori | **logged/tick** | :3541 / :3545 (SS), :1004 / :1006 (DS) |
| p_torso, p_torso_ref, q_torso(_ref) | **logged/tick** | :3539-3545, :3618-3623 |
| torso task definition | **read** | 6-D torso-pose on `J_torso` vs TorsoPlanner quintic+SLERP ref (`p_torso_ref`, `R_torso_ref`), wholebody_qp.py:698 (`_two_task`) |
| torso-pose task RESIDUAL @ QP output | **MISSING** | `‖J_torso·q̈ − a_torso_des‖` needs the QP q̈ (not logged). State-level error `e_torso_*` IS logged (above) |
| Hdot_s realized (E) | **recomputed/tick** | `cross(anchor,f)+τ` on `lambda_qp` (export_figure_data.py:190 method); validated (USERW2-DATA: SS peaks match, 0 overshoot) |
| Hdot_s planned/NMPC (E) | **recomputed, SS only** | same formula on `lambda_ref`; `lambda_ref` is finite in SS (‖·‖ mean 4.90) but **NaN in DS** (NMPC bypassed) → planned Ḣ_s blank in DS |
| momentum task RESIDUAL @ QP output | **MISSING** | `‖A_com·q̈ − b_com‖` needs QP q̈ (not logged) |
| tauw, hw(_physical), theta_s | **logged/tick** | :3580 / :3579 / :3607 |
| qp_ok, nmpc_ok | **logged/tick** | :3609 / :3608 |
| qp_time_ms, nmpc_time_ms | **logged/tick** | :3612 / :3611 |
| nmpc_iterations (NMPC/IPOPT) | **logged/tick** | :3671 (SS), :1121 (DS=0, bypassed) — this is **IPOPT**, not the WBC QP |
| QP nWSR / iterations (qpOASES) | **MISSING** | `_solve_qp_raw` returns only success/exitflag/cost (hierarchical_qp.py:487-491); nWSR not captured → instrumented re-run |
| total weighted task residual (QP) | **MISSING** | QP cost/residual not logged; proxies logged: `lambda_qp_norm` (:3680), `tau_max_joint` (:3604), `qp_time_ms` |

## Missing-data summary (what each needs)
- **Recomputable from the committed log (done here):** realized Ḣ_s, planned Ḣ_s (SS), ori_err_deg,
  pos_ok/ori_ok. No re-run needed.
- **Present but sparse:** twist_norm + docked flag at the 103 `dock_gate_trace` evaluation ticks and 6
  `dock_events` weld ticks — sufficient for table B and weld-instant analysis, NOT for a per-tick twist curve.
- **Needs an instrumented re-run (log the field):** per-tick 6-D twist (norm + lin/ang split) and per-tick
  `docked`/`vel_ok` (require `qpos`/`qvel` or logging `_weld_relative_twist` every tick); QP `nWSR`/iterations
  and the total/per-task weighted QP residual (require capturing them in `_solve_qp_raw` / the task loop).
  These are cheap to add for the next run but are **not present in 3871de4** — not fabricated here.

## Note for the standing userw2 result
The "worst 4.593 mm, margin 0.407" figure (and every prior-phase dock number) is the **fly-by (min-swing)**
metric. By **weld distance** userw2's worst is **4.890 mm (step 2), margin 0.110** — still all-6 welded and
under the 5 mm gate, but the ≥0.2 mm margin is met on the fly-by metric, not the weld metric. Worth deciding
which metric the paper reports (this fly-by/weld gap likely exists in the canonical run too — DOCK-CAUSE saw
it on step 4).

NO canonical change. `crawlbot/` untouched. Raw run (`figC_userw2`) gitignored. **STOP — plot the full set.**
