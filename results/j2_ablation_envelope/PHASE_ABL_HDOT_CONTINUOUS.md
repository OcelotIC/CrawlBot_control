# Phase ABL-HDOT — continuous realized Ḣ_s: STOP-GATE (missing committed source)

**Verdict: the continuous realized Ḣ_s cannot be built from committed data.** The SS
half needs the realized contact wrench `λ_qp`, which is **not in any committed artifact** —
it exists only in the gitignored raw `sim_log.json`. Per the VERIFY instruction ("if SS
cannot be computed from committed wrenches, STOP and report exactly which source is missing;
do NOT reconstruct from gitignored logs, do NOT substitute the clamped tauw"), no column was
produced. READ-ONLY; no `crawlbot/` change, no new sim. Evidence:
`scripts/diag_ablhdot_seam.py` (committed-CSV seam/homogeneity check).

## The physical quantity is well-posed and the code already computes it — on DS only
`export_figure_data.py`:
- **`:142`** `lam_qp = np.asarray(sl['lambda_qp'], float)` — the (n,12) **realized** contact
  wrench `[f_a, τ_a, f_b, τ_b]`. **This is the wrench source.** `sl` is the raw sim_log JSON.
- **`:184`** `anchors_a, anchors_b = load_anchors_struct_frame(model, struct_pos[0], struct_quat[0])`
  — structure-frame contact levers `r_Ci`.
- **`:185-192`** the realized reduction, **guarded to DS only**:
  ```
  for i in range(n):
      if phase[i] in ('DS_interstep','DS_terminal') and stance valid:
          L = lam_qp[i]
          Hdot_exact[i] = cross(anchors_a[sa], L[0:3]) + L[3:6]
                        + cross(anchors_b[sb], L[6:9]) + L[9:12]      # = Σ_j(r_Cj×f_j+τ_j)
          Hdot_source[i] = 'realized'
  ```
- **`:156-166`** on SS, `Hdot_exact` is instead the **PLANNED** value read from
  `postproc_F3F4.csv` (`Hdot_s_{a}`, from `λ_ref`).

⇒ To make it continuous (realized on SS too) I would apply the **same** `:190-191` reduction
with `lam_qp` on SS ticks. The **only** missing input is `sl['lambda_qp']` on SS ticks.

## Homogeneity — the trap, answered
- **Formula IS homogeneous.** SS-planned (`:166`) and DS-realized (`:190-191`) use the SAME
  definition `Σ_j(r_Cj×f_j+τ_j)`, SAME structure-frame anchors, SAME origin reference. So the
  SS↔DS pieces are NOT computed by two different formulations (it is **not** the centroidal
  `−L̇_com−orbital` of `force_estimator.py:585-588` on SS — that form is used by the AOCS, not
  by this export).
- **Wrench source is INHOMOGENEOUS.** SS = **planned** `λ_ref`; DS = **realized** `λ_qp`. This
  is exactly the disjoint half-signal the reviewer rejects. A homogeneous *realized* signal
  needs `λ_qp` on both — missing on SS.
- **Sign convention:** the committed column stores `+Σ_j(r_Cj×f_j+τ_j)` (structure Ḣ_s from
  contacts); your stated quantity `Ḣ_s,realized = −Σ_i(r_Ci×f_i+τ_i)` is its negation. Consistent
  within the column — a labeling choice, not a discontinuity.

## Seam evidence (committed data, `scripts/diag_ablhdot_seam.py`)
Both runs: tagged `Hdot_s` finite everywhere (0 empty cells) but **SS 100% planned / DS 100%
realized** (U 336/776; C 328/752). At the SS↔DS seams the value jumps because the wrench source
changes, e.g. (U):

| tick | left (phase/src) | right (phase/src) | \|Ḣ\|_L | \|Ḣ\|_R | jump_∞ |
|---|---|---|---|---|---|
| 304→305 | DS_interstep/realized | SS/planned | [0.22,0.02,0.48] | [0.92,1.41,**4.85**] | **4.37** |
| 9→10 | DS_interstep/realized | SS/planned | [0,0,0] | [0.46,1.44,2.47] | 2.47 |
| 41→42 | SS/planned | DS_interstep/realized | [0.15,0.47,0.50] | [0.34,0.77,0.32] | 0.48 |

These seam jumps (up to **4.37 N·m**) are a **planned→realized wrench-source change**, not a
clean physical weld transient — so the committed column cannot even serve the value-continuity
check you asked for (its seams are contaminated by the source switch).

## The missing source — exactly what and where
| need | committed? | where it actually is |
|---|---|---|
| `λ_qp` realized wrench (SS ticks) | **NO** — no CSV column; no committed postproc; no committed sim_log for C/U | gitignored raw sim_log only |
| `λ_qp` (DS ticks) | already reduced into committed `Hdot_s` (DS, tagged realized) | — |
| structure-frame anchors `r_Ci` | derivable (model + `struct_pos/quat[0]`) | — (not the blocker) |
| per-tick stance idx | in `postproc_F3F4.csv` (not committed for C/U) | run_dir postproc |

The raw logs **exist on disk but are gitignored** (`git check-ignore` positive):
- **C (runfix, 1080 ticks):** `results/figC_qpcond/sim_log.json` (1080 ticks, has `lambda_qp`) — tick
  count matches the committed `runfix_traversal.csv`; **bit-identity to the committed C run must
  be confirmed** before use.
- **U (1112 ticks):** `results/figU_rateoff/sim_log.json` (1112 ticks, has `lambda_qp`).
- No committed `sim_log.json` exists for either run (all committed sim_logs are old M-milestone /
  diag dirs). Committed CSVs carry no `lambda`/contact/wrench/`r_C` columns.

## This is a governance gate, not data loss — options (awaiting Idriss)
1. **(Recommended) Authorize the realized wrench as source.** Best form: extract a small committed
   CSV `λ_qp[12] + stance_a/b_idx` per tick from the two raw logs, so the continuous Ḣ_s becomes
   **reproducible from committed data permanently**. Then remove the DS-only guard
   (`export_figure_data.py:186`) so `:190-191` runs on every tick → homogeneous realized Ḣ_s over
   SS+DS, with the swing-peak vs weld-peak split and the seam value-continuity check. (First
   confirm `figC_qpcond` ≡ committed `runfix`.)
2. One-time GO to read the gitignored logs directly and emit the column (less clean; not
   self-contained for future reviewers).
3. Fresh instrumented run that writes `λ_qp` into the tidy CSV — **forbidden** here (no new sim).

**Recommendation: option 1.** The signal is fully computable, the reduction code already exists
(`:190-191`), and the data is on disk — only the "committed-only / no gitignored logs" rule
blocks it, and only you can lift it.
