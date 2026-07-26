# INTERNAL — figure-pipeline inventory (read-only ground truth, before freezing the plot convention)

Branch `j2/ds-active-rework`. **Inventory only — no `crawlbot/` or script change.** Establishes what the figure
pipeline ACTUALLY is today, so a single plot convention (`vispa_plot_style.py`, not yet created) can be aligned
to the real code. Answers the five brief questions.

## TL;DR
- The only **plot stage that runs today** is the diagnostic plotter `crawlbot/diagnostics/plots.py`
  (`run_diagnostics()`) — it reads a `SimLog` and writes the **PNG** set `fig1_tracking … fig10` directly (no
  CSV intermediary). It is **PNG-only, Agg, no rcParams/PDF/IEEE/fonttype** — diagnostic-grade, not paper-grade.
- The **publication-quality** plotter `Misc/scripts/run_r7_figures.py` exists but is **R7-era and superseded** —
  it plots Lutze-vs-MPC single/3-step from old logs (`sim_lutze_log.json`, `r6_multistep_log.json`), **not** the
  J2 cooperative-arms run, and its outputs are **not committed**.
- `postprocess_results_figs.py` is a **data postprocessor, NOT a plotter** (writes `postproc_F3F4.csv`, no
  matplotlib). Not superseded for its data role; **nothing draws figures from it.**
- **There is NO current paper fig1–fig7 PLOT stage for the J2 results.** The paper **data** stage
  (`export_figure_data.py` → tidy CSV) exists; **no script consumes that CSV to plot.** `vispa_plot_style.py`
  must be authored fresh — there is no current paper plotter to copy.
- **All 701 committed `fig*.png` are pre-J2** (M0–M7 / diagnostic_q2 / qp_tracking dirs). **No figure has been
  regenerated on the J2/locked config.** The committed figures are fully stale.

---

## 1. What generates fig1–fig7 today (data stage vs plot stage)
- **Plotter (combined data+plot):** `crawlbot/diagnostics/plots.py`, entry `run_diagnostics(log, output_dir,
  dpi)`. Reads the in-memory `SimLog` and writes `fig1_tracking.png, fig2_momentum_aocs.png,
  fig3_com_momentum.png, fig4_energy_passivity.png, fig5_nmpc_health.png, fig6_contact_wrenches.png,
  fig7_joints.png, fig8_snapshots_grid.png` (+ `fig9_ee_6d_tracking`, `fig10_torso_6d_tracking`). **No CSV
  intermediary** — it plots straight from the log arrays.
- **Invocation:** at the end of a sim (CLAUDE.md: "Call `run_diagnostics()` at the end of every sim"), and
  standalone (`from crawlbot.diagnostics import run_diagnostics; run_diagnostics(json.load(...), out)`).
  **No Makefile / no CI step** (no `Makefile` in the repo).
- **`Misc/scripts/run_r7_figures.py`** — "R7 — Generate publication-quality figures for the VISPA paper":
  `fig1_single_step_comparison.pdf`, `fig2_multistep_locomotion.pdf`, `fig3_momentum_comparison.pdf`. Requires
  `sim_torso6d_log.json` / `sim_lutze_log.json` / `r6_multistep_log.json` (R6/R7 Lutze-vs-MPC logs) — **not the
  J2 traversal.** Superseded.

## 2. Is `postprocess_results_figs.py` used or superseded?
- **Still used — as a DATA postprocessor, not a plotter.** It writes `postproc_F3F4.csv` (per-tick stance
  anchor indices + **planned** Ḣ_s), `postproc_metrics.json` (the C1–C5 gate metrics), and
  `postproc_stance_sanity.txt`. It imports **no matplotlib** and draws nothing.
- **Consumers of `postproc_F3F4.csv`:** `gate_phase3.py` (gate metrics), `export_figure_data.py` (paper CSV),
  `audit_chatter.py` / `audit_ss_orbital.py` / `audit_ds_phase2/3.py` (the closure analyses), and my
  `dump_closure_curves.py`. **None of these draw the paper figures** — they are all data/analysis stages.
- So nothing "draws figures from it"; it is the data root that the (not-yet-existing) paper plotter would sit on.

## 3. How is Ḣ_s handled during the inter-step DS? (the soundness question)
- **`postprocess_results_figs.py` (planned):** `Hdot_s[k] = cross(r_C1, lambda_ref[0:3]) + … ` from
  `lambda_ref`. In the inter-step DS the **NMPC is bypassed ⇒ `lambda_ref` is the NaN sentinel ⇒ Ḣ_s is NaN**
  in DS. So this convention is **PLANNED, SS-only; blank in DS.** (Confirms the brief's reading.)
- **`export_figure_data.py` (the CURRENT paper-DATA convention):** reads that planned Ḣ_s
  (`Hdot_source='planned'` where finite, else blank — line 175), then **FIX 2 (lines 177–186) fills the DS gap
  with the REALIZED Ḣ_s from `lambda_qp` + anchors**, tagging `Hdot_source='realized'`. So the paper **data**
  already exports a **full-cycle Ḣ_s: planned on SS, realized in DS** — defined on every tick.
- **`crawlbot/diagnostics/plots.py`:** has **no explicit Ḣ_s figure.** fig2 plots the **realized** `log.tau_w`
  (full cycle, incl. DS) with `±tau_w_max` red dashed reference lines; fig6 plots the realized contact wrenches.
- **Net for the fig1 decision:** there is no committed fig1 to inspect. The **data is already realized-in-DS**
  (export FIX 2) and the closure dump proved the realized DS Ḣ_s is chatter-free and ≪5. The
  planned-NaN-in-DS convention survives **only** in `postproc_F3F4.csv` (the C3 SS metric); it is **not** what
  the paper export uses. So the consistent, already-available choice for fig1 is **realized full-cycle Ḣ_s.**

## 4. Where the visual style is defined (quoted)
**Diagnostic — `crawlbot/diagnostics/plots.py` (the only style that runs today):**
- `matplotlib.use('Agg')`; **PNG only** (`fig.savefig(out/'figN_*.png', dpi=dpi)`); **no PDF, no rcParams, no
  `pdf.fonttype`, no IEEE column dims.**
- Phase shading: `_PHASE_COLORS = {'DS': 'blue', 'SS': 'orange'}`; `ax.axvspan(start, t[i], alpha=0.06,
  color=color, lw=0)`.
- Dock markers: `ax.axvline(ev['t'], color='green', ls='--', lw=0.8, alpha=0.6)`.
- Envelope refs: `ax.axhline(±tau_w_max, color='red', ls='--', lw=0.8)` (also ±hw_max). Per-axis x/y/z series
  use matplotlib defaults (no fixed x/y/z color map). Figure sizes `figsize=(12,10)` / `(12,8)` (screen, not
  column).

**Paper (superseded) — `Misc/scripts/run_r7_figures.py`:**
- `plt.rcParams.update({'font.family':'serif','font.size':9,'axes.labelsize':10,'lines.linewidth':1.2,
  'axes.grid':True,'grid.alpha':0.3,'figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight',
  'savefig.pad_inches':0.05})`. **No `pdf.fonttype=42`** (TrueType embedding NOT set; no `usetex`).
- Colors are **per-controller**, not per-axis: `COL_MPC='#d62728'`, `COL_LUTZE='#1f77b4'`,
  `COL_MULTI='#2ca02c'`, `COL_LIMIT='#333333'`. Saves `.pdf` then `.png` (`savefig(path.replace('.pdf','.png'))`).
  **No SS/DS phase shading.**
- ⇒ Neither file is a paper-grade per-axis/phase-shaded convention with embedded fonts. `vispa_plot_style.py`
  must be authored; the nearest reusable bits are run_r7's serif/dpi-300/bbox-tight rcParams (but it lacks
  `pdf.fonttype=42`, the per-axis color map, and phase shading).

## 5. Which run/CSV the committed fig1–fig7 are built from (staleness)
- **701 committed `fig*.png`, ALL in pre-J2 dirs:** `M0_baseline`, `M3_tests/standalone_diag`,
  `M4/M5/M6_baseline_*`, the many `M7_1pct_1step_v2…v22` / `M7_*_t15_*` / `M7_step2_isolation_*` /
  `M7_abort_diag` dirs, plus `diagnostic_q2/q2b` and `qp_tracking_test*`. These were produced by
  `plots.py` at M-milestone time, on the **pre-J2 plant** (before AOCS-in-DS, c_curr, the chatter fix, and the
  settle-exit criterion).
- **No committed paper PDFs** (run_r7's outputs are not in git). **No `fig*.png` under any j2 / locked /
  chatter / cD_ / pA_ dir** — i.e. **zero figures regenerated on the J2/locked config.**
- The J2 results exist today **only as data + prose:** the figure-data CSVs (`results/j2_figdata/runA*,runB*`),
  the closure curves (`Misc/runs/j2_closure_curves/locked_curves.csv`), and the INTERNAL reports — **never
  plotted, never committed as figures.** ⇒ the committed figures are **fully stale**; nothing on the locked
  (chatter-free, realized-DS-Ḣ_s) config has been drawn.

---

**Conclusion for the convention freeze:** align `vispa_plot_style.py` to a *new* paper plotter that consumes
the `export_figure_data.py` tidy CSV (which already carries realized full-cycle Ḣ_s + the held-setpoint refs).
The diagnostic `plots.py` defines the only live conventions (DS=blue/SS=orange shading α=0.06, ±cap red dashed,
green dock lines) but is PNG/screen-grade; run_r7's rcParams are the nearest paper baseline but R7-content and
sans `pdf.fonttype=42`. Decide fig1's DS segment on **realized** Ḣ_s — it is already what the export produces
and what the closure proof validated. **No change made. Read-only.**
