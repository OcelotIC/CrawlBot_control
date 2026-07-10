# Phase ABL-CMD — the pre-clamp "commanded" wheel torque: STOP-GATE finding

**Verdict: the requested `tauw_commanded` column is NOT produced.** The overshoot *direction*
is confirmed, but (a) the pre-clamp command is not logged, (b) it cannot be faithfully
reconstructed from committed data, and (c) the hypothesized peak (~6.3 = planned |Ḣ_s|) and
mechanism are wrong — the true pre-clamp command peaks near **31 N·m** on DS and is driven by
the realized wrench / kinematic feedforward, not the planned envelope. Per the VERIFY
instruction ("do not force it"), no column was fabricated. READ-ONLY; no `crawlbot/` change,
no new sim. Evidence: `scripts/diag_ablcmd_probe.py` (raw-log probe), `scripts/diag_ablcmd_phase.py`
(committed-CSV phase split).

## (1) The pre-clamp command is not logged
- `force_estimator.py:594` computes `tau_w = ff_term + K_hw·hw_err + pid_term`, then
  **`:595` `return np.clip(tau_w, ±tau_w_max=5)`** — the pre-clamp value is a local, never returned.
- `sim_loop.py:1059-1060` logs `tau_w_applied` = the **clamped** return (comment `:1053`).
- The committed tidy CSVs carry only `tauw_*` (AOCS-clamped ±5) and **none** of the control-law
  inputs (`omega_s`, `L_dot`, `theta_s` absent — verified against the header). ⇒ no reconstruction
  from committed data.

## (2) Direction confirmed, but peak ≈ 31 (not 6.3) and mechanism differs
Applied `tauw` saturates at **exactly ±5.000** ⇒ the pre-clamp command exceeded 5. Saturated-tick
split (committed CSVs):

| run | source (commit) | sat ticks | on SS | on DS | planned‑\|Ḣ_s\| peak | realized‑\|Ḣ_s\| peak |
|---|---|---|---|---|---|---|
| **U** rate-off | `runU_rateoff_traversal.csv` @`be76c9c` | 72/1112 (6.47%) | 31 | 41 | **6.270** (41>5, all SS) | **31.149** (41>5, all DS) |
| **C** canonical | `runfix_traversal.csv` @`5ab2c91` | 40/1080 (3.70%) | 20 | 20 | 5.000 (0>5) | 5.000 (0>5) |

- **DS feedforward IS `−Ḣ_s_realized`** (`force_estimator.py:590-592`, `−Σ(r_Ci×f_i+τ_i)` from
  `λ_qp`, identical to the committed realized column). On U's 41 saturated DS ticks
  `|Ḣ_s_realized| ∈ [5.05, 31.15]` — pre-clamp command **~31 N·m**, an order of magnitude past ±5.
- **SS feedforward is kinematic `−L_dot − orbital`** (`:585-588`). Realized `−L_dot` peaks at only
  **1.758**; `H_dot_est` in the raw log is **all zeros**.
- **The hypothesis mechanism is disproved:** on U's 31 saturated SS ticks the committed **planned**
  |Ḣ_s| ranges `[2.21, 6.27]` — including ticks where planned is only **2.21** while the wheel
  clamps at 5.000. The SS wheel command is therefore not the planned |Ḣ_s|; it is the realized
  `−L_dot−orbital` + `K_d·ω̇_s` terms.

## (3) The 6.27 is a different, already-committed signal
Planned |Ḣ_s| peak 6.270 = the **pre-planner rate-cap envelope** (`coarse_preplanner.py:342-343`,
`|L̇|≤tau_w_max=5`) — the *planning-time demand* U disables (`tau_w_max=1e6`). It is not the AOCS
wheel command. Already exported as `Hdot_s_planned_*` in `ablation_data/` @`606311e`.

## (4) Why no faithful column exists
- **DS half** reconstructs from committed data as `−Ḣ_s_realized` — but peaks **31.15**, not 6.3.
- **SS half** needs `−L_dot−orbital`, which is **not committed**; the only source
  (`results/figU_rateoff/sim_log.json`) is **gitignored** (`git check-ignore` confirmed) — violates
  "committed runs only" — and a raw-log reconstruction is **unfaithful**: the orbital
  `r_com×m·dv_com` + `K_d·ω̇_s` FD terms over-predict saturation 5× (**31.6%** of ticks vs true
  **6.47%**), max per-tick residual = a full **10 N·m** clamp band.

## (5) Honest with/without contrast (already in committed data — no new column needed)
- **Demand:** `Hdot_s_planned_*` — **U 6.27 (12.2% SS>5) vs C 5.00 (0%>5)**. Label = *planned
  reaction-torque-rate demand vs the ±5 rate cap*, not "commanded wheel torque."
- **Clamp response:** committed clamped `tauw_*` — flat-topped ±5, **U 6.47% vs C 3.70% saturated**.

## Options (awaiting Idriss)
1. **(Recommended)** Figure = committed `Hdot_s_planned` (demand, 6.27>5 on U) + committed clamped
   `tauw` (±5 flat-top). Faithful, honestly labeled, no new column.
2. DS-only `tauw_commanded = −Ḣ_s_realized` from the committed realized column (faithful; peaks 31;
   DS-only).
3. Fresh instrumented run logging pre-clamp `tau_w` directly (one log line before
   `force_estimator.py:595`) — currently forbidden by "no new sim / no `crawlbot/` change."
