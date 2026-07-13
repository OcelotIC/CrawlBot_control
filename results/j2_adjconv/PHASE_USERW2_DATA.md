# Phase USERW2-DATA — userw2 timeseries export + envelope analysis (the cap IS active — in DS-settle)

**Branch** `j2/ds-active-rework` · export from the committed run, NO new run, NO canonical change · pushed,
never merged. Artifacts: `results/j2_adjconv/userw2_timeseries.csv` (1860 ticks), `userw2_envelope_analysis.json`;
script `scripts/diag_userw2_export.py`.

**Correction up front:** the premise "userw2's realized |Ḣ_s| ≤ 2.50 never reaches the cap → constraint
inactive → ablation at risk" is **SS-only**. The homogeneous realized Ḣ_s (λ_qp on *all* ticks) peaks at
**5.000 Nm — AT the cap — during the DS inter-step settles** (125 ticks, steps 0/2/4). The momentum-rate
box is **NOT inactive**; it binds in the settle phase. What changed vs canonical is that the **SS swing**
no longer saturates (≤2.50, headroom), not that the cap is unused.

## Provenance
Committed run **3871de4** (`userweights_result2.json`). The on-disk `results/figC_userw2/sim_log.json`
(1860 ticks) is the same run: SS docks match the committed result exactly
(`2.5652/4.5933/3.0056/4.3915/2.4854/4.4941`). `lambda_qp` logged on all ticks.

## Method — homogeneous realized Ḣ_s (validated canonical formula, applied on ALL ticks)
Same origin-referenced formula as the validated canonical export (`export_figure_data.py:184-192`, FIX 2):
`Ḣ_s = r_Ca×f_a + τ_a + r_Cb×f_b + τ_b` from realized `lambda_qp` + structure-frame anchors — but on **every**
tick (SS and DS), not planned-on-SS. Per-tick weld levers from the confirmed leapfrog (validated against the
force-activity blocks: in SS exactly one arm's contact force is active, the swing arm's is 0.00): SS ⇒ stance
arm at `STANCE_ANCHOR[k]` (swing λ=0, lever moot); DS ⇒ both arms at `STANCE_ANCHOR[k]`/`SWING_ANCHOR[k]`.
- **Validation:** homogeneous SS per-step peak |Ḣ_s| = `[2.024, 2.05, 2.163, 1.537, 2.496, 1.634]` — matches
  the committed `hdot_s_pk` to 3 dp. And **0/1860 ticks exceed the ±5 box** (never overshoots) ⇒ the 5.000 is
  genuine box binding, not an FD/impact artifact. (`qfrc_constraint_torque` is all-zero here — the H-estimator
  was off — so no independent MuJoCo cross-check; the box-enforcement check substitutes.)

## CSV (`userw2_timeseries.csv`, 1860 ticks, 17 cols)
`t_s, phase, step_index, Hdot_s_realized_cont_{x,y,z}_Nm, Hdot_s_realized_norm_Nm, tauw_{x,y,z}_Nm,
hw_{x,y,z}_Nms (physical), theta_s_{x,y,z}_deg (struct euler), d_grip_swing_mm`. `phase` is granular
(`SS`/`DS_interstep`/`DS_terminal`).

## Analysis

### Q1 — where is the peak
Global realized |Ḣ_s| peak = **5.000 Nm**, tick 1388, **t=44.45 s, DS_interstep of step 4, axis z** (that
tick has y AND z at ±5.00, norm 7.20). Per step (max abs component):

| step | SS-swing peak (x/y/z) | SS max | DS-settle peak (x/y/z) | DS max | dock [mm] | \|Ḣ_s\| at dock |
|---|---|---|---|---|---|---|
| 0 | 0.80/1.45/2.02 | 2.02 | 1.76/4.47/**5.00** | **5.00** | 2.565 | 0.187 |
| 1 | 0.25/1.40/2.05 | 2.05 | 1.49/2.00/0.92 | 2.00 | 4.593 | 0.075 |
| 2 | 0.61/1.29/2.16 | 2.16 | 2.02/**5.00**/**5.00** | **5.00** | 3.006 | 0.704 |
| 3 | 0.17/1.37/1.54 | 1.54 | 1.54/2.07/4.71 | 4.71 | 4.391 | 0.201 |
| 4 | 0.56/1.86/**2.50** | **2.50** | 1.51/**5.00**/**5.00** | **5.00** | 2.485 | 0.965 |
| 5 | 0.31/1.63/1.02 | 1.63 | 0.40/0.83/0.71 | 0.83 | 4.494 | 0.254 |

The SS-swing max is 2.50 (step 4, axis z); the DS-settle reaches the 5.00 cap on the **even steps (0/2/4)** —
the post-b-dock settles where both arms load heaviest.

### Q2 — is the box active / plateaued?
**ACTIVE.** 125/1860 ticks have an Ḣ_s axis ≥ 4.95 — **all 125 in DS-settle, 0 in SS-swing**. SS-swing peak
2.496 (headroom, ≈50 % of the box); DS-settle peak 5.000 (at cap). So the momentum-rate box does not just
"oscillate freely below 2.5" — it **binds during the inter-step settles**, then the swing runs with headroom.
(Contrast — two other limits: the **AOCS wheel-torque clamp** `|τ_w|=5` is active 110/1860 ticks; **hw** peaks
4.124 Nms, inside the ±5 envelope, C5 margin 0.88.)

### Q3 — dock-limiting factor: tracking, not momentum
At every step's dock instant, |Ḣ_s| = **0.075–0.965 Nm** — far below both the per-step SS peak (≤2.5) and the
5.0 cap. The momentum demand peaks *mid-swing* (≤2.5) and *in settle* (5.0), but is near-zero at closest
approach. ⇒ **all six docks are tracking-limited, with large momentum headroom at the dock** — not
momentum-demand-limited. This is exactly why the userw2 rebalance (momentum-task weight 5000→400) opened the
docks: it relaxed the SS-swing momentum demand (canonical saturated the swing at 5.0; userw2 ≤2.5), freeing
EE tracking, while the hard box still catches the settles.

### Q4 — τ_w,max = 5 provenance (file:line)
- `config.py:80` `tau_w_max: float = 5.0  # |Ḣ_s,i| ≤ τ_w_max [Nm] — wheel-torque rate cap (NMPC)` — the QP
  momentum-rate box (`wholebody_qp.py:564-580`, per-axis ±5) and the NMPC cap.
- `config.py:84` `aocs_tau_w_max: float = 5.0  # Max wheel torque [Nm]` — the AOCS wheel-torque actuator clamp
  (what realized `τ_w` saturates against).
- `config.py:72` `hw_max = +5.0 Nms` (`:71` hw_min −5.0) — physical wheel-momentum envelope.
- CLI: `--tau_w_max` (default 5.0, `diag_cooperative_arms.py:243`) sets **both** `cfg.tau_w_max` and
  `cfg.aocs_tau_w_max` (`:302-303`).
- Origin: paper spec values (per `CLAUDE.md`: τ_w,max spec §5.1, hw_max ±5 spec §4.6) — a wheel-actuator
  design/spec choice (±5 Nm torque, ±5 Nms momentum), not tuned per-run.

## Verdict for the cap decision
- **The constraint is NOT truly inactive.** The momentum-rate box binds during the DS inter-step settles
  (5.000 Nm, 125 ticks, steps 0/2/4) — the ablation demonstration is **not dead**; the cap still constrains
  realized momentum-rate, just in the settle phase rather than the swing.
- **The SS swing has headroom** (≤2.50, ~50 % of the box) — that headroom is *what freed the docks*
  (tracking-limited, momentum-relaxed swing).
- **Lowering the cap to revive swing saturation is a direct trade-off:** dropping τ_w,max toward ~2.5 would
  make the SWING bind again (reviving the canonical-style swing saturation) — but it would remove the exact
  headroom the userw2 docks depend on, so it would likely **degrade the docks back toward the floor**. The
  settles already saturate at 5.0 regardless. So the cap is a lever that trades swing-side saturation
  (ablation clarity) against dock margin — Idriss's call. Curves are in the CSV.

NO canonical change. `crawlbot/` untouched. Raw run (`figC_userw2`) gitignored. **STOP — plot & decide on the cap.**
