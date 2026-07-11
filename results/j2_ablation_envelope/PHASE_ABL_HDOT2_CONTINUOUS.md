# Phase ABL-HDOT-2 (STOP-GATE 2) — continuous REALIZED Ḣ_s over SS+DS

The committed `Hdot_s` was planned-on-SS / realized-on-DS (inhomogeneous wrench source, seam
artifacts up to 4.37 N·m). This builds the **homogeneous realized** Ḣ_s = Σ_j(r_Cj×f_j+τ_j) from
`λ_qp` on **every** tick, reproducible from committed data. Scripts: `extract_lambda_qp.py` (Step 1),
`build_continuous_hdot.py` + `diag_ablhdot2_split.py` (Step 2). No `crawlbot/` change, no new sim;
gitignored logs used once (Idriss-authorized) only to mint the committed `λ_qp` CSVs.

## Step 0 — identity (STOP-GATE 1, commit `83257e5`)
`figC_qpcond/sim_log.json` (1080) ≡ committed `runfix` and `figU_rateoff/sim_log.json` (1112) ≡
committed `runU_rateoff`: every overlapping quantity (tauw, hw, theta_s, rcom, torso_pos,
torso_ori_err, swing_dist) matches tick-for-tick to **max |dev| ≈ 5e-7** (CSV rounding floor);
`step_index`, `phase` exact. Full table: `PHASE_ABL_HDOT2_STEP0_IDENTITY.md`.

## Step 1 — committed `λ_qp` (commit `2a79c97`)
`ablation_data/lambda_qp_{C,U}.csv` (`λ_qp[12]=[f_a,τ_a,f_b,τ_b]`, stance_a/b_idx, phase,
step_index, t_s) + `lambda_qp_meta.json` (struct pose₀ for anchors). Stance validated to reproduce
committed DS Ḣ_s: C←`figA_canon_fixed/postproc_F3F4.csv` (5e-7), U←`figU_rateoff/postproc_F3F4.csv`.

## Step 2 — the reduction (`build_continuous_hdot.py`)
Same form as **`export_figure_data.py:190-191`**, DS-only guard (`:186`) removed, applied on all
ticks: `cont[i] = cross(aA[sa], f_a)+τ_a + cross(aB[sb], f_b)+τ_b`, structure-frame anchors
(`load_anchors_struct_frame`), origin-referenced, `+Σ` sign (matches committed `Hdot_s_realized`;
your −Σ is its negation). On SS the swing arm's **full wrench (f AND τ) is exactly 0** (verified),
so the same two-arm formula is automatically single-contact. New columns added to
`ablation_{C,U}_timeseries.csv`: **`Hdot_s_realized_cont_{x,y,z}_Nm`**.

### VERIFY
- **Continuity:** 0 empty cells (C 1080, U 1112). **Homogeneity:** `λ_qp` on ALL ticks, no `λ_ref`.
- **DS cross-check:** `cont` == committed `Hdot_s_realized` on DS to **1.3e-6 (C) / 4.7e-6 (U)** —
  the continuous signal is a faithful extension of the committed DS values.
- **Seam value-continuity** (the homogeneity trap): the continuous signal **joins in value** — the
  old planned→realized artifact is gone. At the worst seam **t≈13.5 (DS→SS)** the old jump **4.37**
  becomes **1.99** (`contL=[0.22,0.02,0.48]`→`contR=[0.67,0.79,2.46]`): a **real** ramp as the
  swing demand turns on (realized both sides), not a source switch. Other seams join to ≤0.72.
  Remaining jumps are physical (contact state changing), distinguishable from the artifact because
  both sides now use `λ_qp`.

### THE SPLIT — swing vs weld vs settle (per axis, `W_weld=2` ticks/side)
| category | C peak x/y/z | U peak x/y/z | >5? |
|---|---|---|---|
| **SS-swing** (excl. weld) | 1.33 / 3.37 / **5.00** | 1.42 / 3.51 / **7.48** | C caps, **U over** |
| weld window (±2 of SS↔DS) | 3.36 / 5.00 / 5.00 | 4.08 / **15.99** / 14.09 | both |
| DS-settle (excl. weld) | 3.32 / 5.00 / 5.00 | 3.94 / 13.17 / **31.15** | both |
| overall | 5.00 (settle) | **31.15** (settle, t=31.06) | — |

**U's swing over-5 is REAL, not a weld transient** — robust to the exclusion width:

| W (ticks/side excluded) | 0 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|
| U SS-swing peak_z | 7.48 | 7.48 | 7.48 | 7.48 | **6.52** |
| C SS-swing peak_z | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 |

The U peak (7.48) is at tick 684 (t=28.22), **6 ticks from the nearest boundary**, a smooth
mid-swing bump (5.15→7.48→5.28). Excluding up to 10 ticks around every weld still leaves 6.52 > 5.
C sits at a flat **5.00** plateau (envelope constraint binding).

## Decision
**Over-5 occurs in SWING (excl. weld) on U ⇒ real locomotion over-demand ⇒ the continuous realized
Ḣ_s IS the ablation figure.** Without rate management the realized structure reaction-torque rate
exceeds the ±5 envelope during swing locomotion (z=7.48, mid-swing, robust), and far more at the
docking weld/settle (15.99 / 31.15). With management (C) it is held at exactly 5.00 throughout. This
is a genuine actuator over-demand, not a contact-model transient. Plot `Hdot_s_realized_cont_*`,
C vs U, ±5, welds annotated.

## Artifacts
`ablation_data/ablation_{C,U}_timeseries.csv` (+`Hdot_s_realized_cont_*`),
`ablation_data/lambda_qp_{C,U}.csv`, `lambda_qp_meta.json`, `continuous_hdot_report.json`. Manifest
updated. Reproducible from committed data (no gitignored log needed henceforth).
