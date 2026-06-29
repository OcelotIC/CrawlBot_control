# INTERNAL — figure-data export for the paper (per-tick CSV + meta JSON) — FIXED

**Export only — NO `crawlbot/` change** (reuses the committed flags; FIX 1/2 are post-hoc from logged fields).
Branch `j2/ds-active-rework` (pushed, never merged). Reproducer `scripts/export_figure_data.py` (+ driver
`scripts/run_figdata.sh`). Two runs as tidy CSV + meta JSON; **no plots**. This revision fixes the reference
columns (FIX 1) and fills the inter-step envelope gap (FIX 2).

## Deliverables
```
results/j2_figdata/runA_traversal.csv   773 rows × 39 cols   + runA_meta.json
results/j2_figdata/runB_traversal.csv   869 rows × 39 cols   + runB_meta.json
```
(+1 column vs the first export: `Hdot_s_source` ∈ {planned, realized}.) One row per tick, SI units in the
header. Mixed-cadence log (SS/_step ~0.1 s, inter-step DS ~0.01 s) — use the `t_s` column for the time axis.

## The two run configs (committed flags; no code change)
- **RUN A — latest features:** `--qp-envelope-exact` + `aocs_active_in_interstep=True` +
  `interstep_hw_refresh=True` (defaults) + `--aocs_mode legacy_pid_numerical`, canonical 5-step.
- **RUN B — DWELL moving CoM:** RUN A + `--ds-mobile-com-magnitude 0.20 --dt-ds 2.5`.

---

## FIX 1 — reference columns (torso held-setpoint; CoM left as-is)

**Torso (`torso_pos_ref_*`) — fixed.** Now the **held setpoint**: in DS, the realized torso pose **frozen at
the entry of each contiguous DS block** (the docked pose it holds); in SS, the logged NMPC/QP setpoint
(unchanged). This removes all three artifacts of the first export — the `(0,0,0)` sentinel on the initial
`step=-1` ticks, the DS_interstep phase-lag (previous step's setpoint), and the DS_terminal dock-IK offset.
Result — `‖torso − ref‖` is now the **true tracking error** everywhere:

| | first export (artifact) | fixed |
|---|---|---|
| torso pos-peak SS [mm] | 28.0 (real) | **28.0** (= C2, unchanged) |
| torso pos-peak DS [mm] | **902** (sentinel) / 144 (offset) | **8.8 (run A) / 13.7 (run B)** — true hold drift |

**CoM (`rref_*`) — left as logged; it was already correct.** Diverging from the brief's premise: the CoM
reference has **no** sentinel or phase-lag. The code shows `r_com_ref` logs `cref_r`, the NMPC **current-tick**
reference, and `e_com = ‖r_com − cref_r‖` — so `‖rcom − rref‖` is the *true* CoM tracking error. Concretely:
inter-step DS logs `r_com_ref = rs.r_com` (the QP holds in place ⇒ error ≈ 0); the run-B DWELL logs the
**moving** reference (must be preserved); SS logs the real lag. Verified: run A CoM lag peak **95 mm** (the
real SS swing lag = `e_com`, figure-4 content), run B CoM lag peak **199.6 mm** (≈ the 0.20 m moving-CoM
command — the DS-mobile lag, **preserved**). Overriding it would have destroyed run B's figure. **Only the
torso reference needed fixing.**

How the per-tick reference is sourced (stated for the figures): torso → held-entry realized in DS / logged
setpoint in SS; CoM → logged `cref_r` everywhere (held-realized in inter-step DS, moving in run-B DWELL, real
NMPC ref in SS).

## FIX 2 — inter-step envelope gap filled (realized Ḣ_s)

`Hdot_s_*` is now **defined on every tick** (no blank), via a new `Hdot_s_source` column:
- **SS / DS_terminal — `planned`:** the exact origin-referenced `Ḣ_s = Σⱼ(r_Cⱼ×fⱼ+τⱼ)` from the planned
  wrench (`lambda_ref`), read from `postproc_F3F4.csv` = the C3 metric ‖Ḣ_s‖∞_SS (=5.0 at the binding).
  Unchanged.
- **DS_interstep — `realized`:** the **same** exact formula/anchors but from the **settle-QP wrench
  `lambda_qp`** (the NMPC is bypassed there, so `lambda_ref` is absent). `lambda_qp` is already logged per
  inter-step tick (mean‖·‖ 9.6, max 41.5) — **no `crawlbot/` change needed**; anchors loaded in R_s0 exactly
  as the postproc helper, stance indices read from `postproc_F3F4.csv`.
- `proxy = exact − r_com×Σf` (robot-CoM lever, orbital omitted), Σf from the same wrench as the exact value
  at that tick.

**Does each component stay within ±τ_w_max=5 on the inter-step ticks too? — YES, on every axis, both runs.**

| run | planned SS \|Ḣ_s\|∞ [Nm] | realized inter-step \|Ḣ_s\|∞ [Nm] | FULL-CYCLE \|Ḣ_s\|∞ | inter-step ticks over 5 |
|---|---|---|---|---|
| A | [3.305, 5.000, 5.000] | **[5.000, 5.000, 5.000]** | [5.000, 5.000, 5.000] | **0** |
| B | [3.381, 4.806, 5.000] | **[5.000, 5.000, 5.000]** | [5.000, 5.000, 5.000] | **0** |

The realized inter-step Ḣ_s **binds at exactly 5.0** on all axes (the post-dock impact drives it to the cap)
and **never exceeds it** — i.e. with the exact box active in the inter-step settle QP (+ c_curr), the
origin-referenced envelope is respected through the post-dock settle, so the figure can show ‖Ḣ_s‖∞ ≤ 5
across the **full SS→DS→SS cycle**, not just SS. (Contrast: a proxy-box run leaves the *exact* inter-step
Ḣ_s uncapped — up to ~57 N·m — because the proxy caps only the robot-CoM-lever quantity.)

## FIX 3 — continuity sanity

`hw_*`, `theta_s_*`, `tau_w_*` have **0 blanks / no sentinels** in both runs (continuous per-tick), confirmed.
All first-export columns retained (38 → 39 with `Hdot_s_source`).

## Sanity cross-checks (vs prior audits) — re-confirmed

| quantity | RUN A | RUN B | matches |
|---|---|---|---|
| rows × cols | 773 × 39 | 869 × 39 | one row/tick |
| Ḣ_s full-cycle \|·\|∞ per-axis [Nm] | [5.0, 5.0, 5.0] (0 over) | [5.0, 5.0, 5.0] (0 over) | envelope respected end-to-end |
| Ḣ_s planned SS \|·\|∞ | [3.305, 5.0, 5.0] | [3.381, 4.806, 5.0] | = C3 gate [3.3, 5.0, 5.0] (run A) |
| **torso pos-peak SS [mm]** | **28.0** | 22.3 | **= C2 (28.0, run A exact box)** |
| torso hold-peak DS [mm] | 8.8 | 13.7 | true hold drift (was 902/144 artifact) |
| final ‖θ_s‖ [deg] | 0.105 | 0.100 | C4 |
| hw range [Nms] (\|·\|max) | [−4.949, 2.284] (4.949) | [−4.765, 2.083] | **\|hw\|max 4.949 = C5** (run A) |
| Ltot ‖·‖ final [Nms] | 2.99e-3 | 3.16e-3 | **= ccurr_exact_on residual 0.002987** (run A) |
| CoM lag peak [mm] | 95.1 (real SS lag = e_com) | **199.6** | ≈ moving-CoM 0.20 m (run B, preserved) |
| min swing_dist/step [mm] | {4.9,4.4,5.0,4.6,3.0} | {4.9,4.1,4.9,4.2,4.2} | ≈ dock distances (C1) |

All audit cross-checks hold (C2 28.0 mm with the fixed ref, C5 4.949, C3 [3.3,5,5], residual 0.002987,
run-B CoM lag ≈ 0.20 m).

## Quantity identities (unchanged from the first export, re-stated)
- **Ltot = `subtree_angmom[0]`** about the system CoM, recomputed at the 12 snapshots (mj_subtreeVel) —
  identical to the Fix-A conservation check; full series in `meta.conservation_snapshots`, CSV carries it at
  the snapshot-nearest ticks.
- **swing_dist = `d_grip_swing`** via `_gripper_distance` (the Fix C dock-gate gripper/anchor site pair);
  meaningful when `swing_active=1`.
- `hw = hw_physical`, `tauw = tau_w` (commanded, post-clip), `theta_s = struct_euler_deg`.

## Column list (39)
`t_s, tick, phase, step_index | Hdot_s_{x,y,z}_Nm, Hdot_s_proxy_{x,y,z}_Nm, Hdot_s_source | hw_{x,y,z}_Nms,
theta_s_{x,y,z}_deg | Ltot_{x,y,z}_Nms | rcom_{x,y,z}_m, rref_{x,y,z}_m | tauw_{x,y,z}_Nm |
torso_pos_{x,y,z}_m, torso_pos_ref_{x,y,z}_m, torso_ori_err_deg | swing_dist_m, swing_active, swing_arm`.

## Reproduce
```
bash scripts/run_figdata.sh   # runs A & B (committed flags) → postproc → export both
```
Raw per-run dirs (`results/figA`, `results/figB`) reproducible from the driver, not committed; the deliverable
CSV/JSON in `results/j2_figdata/` are.

**STOP after the report.** Data files only — no plotting, no `crawlbot/` change. No merge, no PR.
