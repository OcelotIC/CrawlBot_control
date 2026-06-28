# INTERNAL — figure-data export for the paper (per-tick CSV + meta JSON)

**Export only — NO `crawlbot/` change** (reuses the committed flags via `diag_cooperative_arms`). Branch
`j2/ds-active-rework` (pushed, never merged). Reproducer `scripts/export_figure_data.py` (+ driver
`scripts/run_figdata.sh`). Two runs exported as tidy CSV + metadata JSON; **no plots** (figures built
downstream from these files). Cross-checks against prior audit numbers below.

## Deliverables
```
results/j2_figdata/runA_traversal.csv   773 rows × 38 cols   + runA_meta.json
results/j2_figdata/runB_traversal.csv   869 rows × 38 cols   + runB_meta.json
```
One row per tick, SI units in the header. The log is **mixed-cadence**: SS/_step rows ~0.1 s (dt_nmpc),
inter-step DS rows ~0.01 s (dt_qp) — use the `t_s` column for the time axis (not a constant dt).

## The two run configs (committed flags; no code change)
- **RUN A — latest features (correct-physics, AOCS-everywhere, c_curr-on):**
  `--qp-envelope-exact` (exact box) with `aocs_active_in_interstep=True` and `interstep_hw_refresh=True`
  (both committed defaults), `--aocs_mode legacy_pid_numerical`, canonical 5-step. Figures 1/2/3/5/6/7 +
  general CoM tracking. *(= the ccurr_exact_on configuration.)*
- **RUN B — DWELL with moving CoM:** RUN A + `--ds-mobile-com-magnitude 0.20 --dt-ds 2.5`. Exposes the
  DS-mobile CoM lag dormant in run A (figure 4-mobile).

Full flag sets are in each `*_meta.json` `run_config`.

## Quantity identities (so figures match the audits)

- **Ltot (conservation, figure 3) = `mj_data.subtree_angmom[0]`** — total system angular momentum about the
  system CoM (world frame), recomputed at the **12 snapshots** (`initial`, `release_step{0..4}`,
  `dock_step{0..4}`, `final`) by replaying the stored `qpos/qvel` through `mj_kinematics/comPos/comVel/
  subtreeVel`. **This is the identical field and code path as the Fix-A conservation check**
  (`audit_fixC_residual.py` / `audit_fixA_leak.py` / `plot_dock_leak_part3.py`), so the conservation /
  ~0.08 %-leak claim stays consistent. It is **NOT a per-tick logged field** (per-tick recompute needs the
  full 31-DOF state, which the per-tick log does not store) — the full labelled series is in
  `meta.conservation_snapshots`, and the CSV `Ltot_*` columns carry it at the snapshot-nearest ticks (blank
  elsewhere). Figure 3 is built from the 12-point series.
- **Hdot_s (envelope, figure 1) = the exact origin-referenced Ḣ_s = Σⱼ (r_Cⱼ × fⱼ + τⱼ)** from the planned
  wrench (`lambda_ref`) + structure-frame anchors — the NMPC envelope-constraint quantity, read from
  `postproc_F3F4.csv` so it is **identical to the C3 gate metric** ‖Ḣ_s‖∞_SS. (NOT a finite-difference of
  the realized momentum — that has dock/phase-transition spikes and is not what the envelope enforces.)
  Defined on SS/_step rows; **blank on inter-step DS** (NMPC bypassed ⇒ no planned wrench). `Hdot_s_proxy_*`
  = exact − r_com×Σf (lever from the robot CoM, orbital omitted) — the FLAG-2 proxy, for the exact-vs-proxy
  overlay.
- **swing_dist = `d_grip_swing` = ‖gripper_site − target_anchor‖** (Euclidean, structure frame) via the
  `_gripper_distance` helper — the **same gripper/anchor site pair as Fix C's dock gate**. Meaningful only
  while `swing_active=1` (SS phase, an arm swinging — `swing_arm` ∈ {a,b}); its per-step minimum ≈ that
  step's dock distance. (In DS it carries a non-meaningful sentinel; mask by `swing_active`.)
- **torso (figure 6):** `torso_pos_*` = realized `p_torso`; `torso_pos_ref_*` = `p_torso_ref`;
  `torso_ori_err_deg` = `e_torso_ori` (geodesic). **Caveat:** `p_torso_ref` is a (0,0,0) sentinel during the
  initial DS only (`step_index = -1`, the 10 pre-first-SS ticks) → a spurious ~0.90 m "error" there.
  **Restrict figure 6 / the pos-error to `step_index ≥ 0` (or SS rows)**, where the peak is **28.0 mm**
  (matching C2). The torso reference is otherwise near-constant (held orientation, canonical) — expected.
- **hw = `hw_physical`** (the physical wheel momentum `rwa_I_w·qvel[6:9]`, the C5 quantity). **tauw = `tau_w`**
  (commanded, post-clip). **theta_s = `struct_euler_deg`**. **phase** ∈ {SS, DS_terminal, DS_interstep}
  (DS split via `inter_step_settles` windows).

## Sanity values (cross-check vs prior audits)

| quantity | RUN A | RUN B | matches |
|---|---|---|---|
| rows × cols | 773 × 38 | 869 × 38 | one row/tick |
| n_steps / docks | 5 / 5 | 5 / 5 | full traversal |
| **Hdot_s SS peak per-axis [Nm]** (cap 5) | **[3.30, 5.00, 5.00]** | [4.33, 5.00, 5.00] | **= C3 gate [3.3, 5.0, 5.0]** (run A) |
| final ‖θ_s‖ [deg] | 0.105 | 0.100 | C4 final ~0.05–0.11 |
| hw range [Nms] | [−4.949, 2.284] | [−4.765, 2.083] | **\|hw\|max 4.949 = C5** (run A exact box) |
| Ltot ‖·‖: initial / dock-max / final [Nms] | 0 / 2.51e-3 / 2.99e-3 | 0 / 2.53e-3 / 3.16e-3 | **= ccurr_exact_on residual 0.002987** (run A) |
| min swing_dist per step [mm] | {4.94, 4.44, 4.99, 4.64, 2.99} | {4.94, 4.14, 4.89, 4.18, 4.18} | ≈ dock distances (C1) |
| torso pos-peak (step≥0) [mm] | **28.0** | ~28 | **= C2 pos-peak 28.0 mm** (exact box) |
| CoM lag max ‖rcom−rref‖ [m] | — (static DWELL) | **0.1996** | **≈ moving-CoM mag 0.20** (Piste-A magtest) |

Run A reproduces the exact-box working point exactly (C3 [3.3,5,5], C5 4.949, residual 0.002987, C2 28 mm) —
i.e. the figure data is the same run the gate scored. Run B's CoM lag (0.1996 m ≈ the 0.20 m command) is the
DS-mobile lag for figure 4-mobile.

## Column list (CSV header carries units)
`t_s, tick, phase, step_index | Hdot_s_{x,y,z}_Nm, Hdot_s_proxy_{x,y,z}_Nm | hw_{x,y,z}_Nms,
theta_s_{x,y,z}_deg | Ltot_{x,y,z}_Nms | rcom_{x,y,z}_m, rref_{x,y,z}_m | tauw_{x,y,z}_Nm |
torso_pos_{x,y,z}_m, torso_pos_ref_{x,y,z}_m, torso_ori_err_deg | swing_dist_m, swing_active, swing_arm`.
`meta.json`: `tau_w_max_Nm`/`hw_max_Nms` (5/5), `dt_s_median`/`dt_s_min`+`dt_note`, `n_ticks`/`n_steps`,
`robot_mass_kg` (71.056), `structure_mass_kg` (7110), `dock_events`, `phase_segments`,
`conservation_snapshots`, `run_config`, and the three `*_definition` strings.

## Reproduce
```
bash scripts/run_figdata.sh   # runs A & B (committed flags) → postproc → export both
```
Raw per-run dirs (`results/figA`, `results/figB`) are reproducible from the driver and not committed; the
deliverable CSV/JSON in `results/j2_figdata/` are.

**STOP after the report.** Data files only — no plotting, no figure generation. No merge, no PR.
