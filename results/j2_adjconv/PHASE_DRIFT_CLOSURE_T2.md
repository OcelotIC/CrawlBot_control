# Phase DRIFT-CLOSURE — SUB-PHASE A (T2): logging-only export of ω_s + L_total

**Scope:** additive diagnostic export only. The canonical freeze `32aefaf` is untouched (it is an
ancestor of this branch); no controller, config, or model file changed; **no simulation was re-run** —
the two new channels are computed by post-processing the already-committed on-disk `sim_log.json` of the
canonical managed (C) and unmanaged (U) runs.

**Branch:** `claude/lucid-gates-rsigzt` (drift work stream, isolated from PR #29's `j2/ds-active-rework`).
**Gate:** STOP after T2. SUB-PHASE B (T4) is blocked on this passing.

---

## 1. The two channels

| channel | meaning | source | availability |
|---|---|---|---|
| `omega_s_{x,y,z}_radps` | structure angular rate in R_s | `mj_data.qvel[3:6]` logged **per tick** | all 2077 ticks |
| `Ltot_{x,y,z}_Nms` + `Ltot_norm_Nms` | total system angular momentum `subtree_angmom[0]` | `mj_subtreeVel` on stored `(qpos,qvel)` **snapshots** | 44 event instants → mapped to nearest tick, blank elsewhere |

- `omega_s` provenance: `crawlbot/simulation/logging.py:158`; appended at `sim_loop.py:1109` (DS) and
  `sim_loop.py:3687` (SS), both `= self.mj_data.qvel[3:6].copy()`.
- `L_total` provenance: `scripts/export_figure_data.py:66-82` `ltot_at_snapshots()` — sets `d.qpos/qvel`
  from each snapshot, runs `mj_kinematics/comPos/comVel/subtreeVel`, returns `subtree_angmom[0]`. This is
  the exact Fix-A conservation quantity used elsewhere in the paper pipeline. It is **not** available
  per-tick because `qpos/qvel` are only stored at the 44 event snapshots (dock/release/frame/initial/final);
  the remaining ticks are left blank (not fabricated).

**Exporter edit** (`scripts/diag_full_diag_export.py`, append-only): import `ltot_at_snapshots, nearest_tick`
(`:27`); compute the channels (`:94-104`); append 7 columns after `tau_max_joint_Nm` (`:124-125`); append
7 fields per row (`:150-152`); add `ltot_snapshots` to the meta JSON (`:174-175`). 23 insertions, 0 pre-existing
line rewritten except the 4 that were extended.

---

## 2. Proof of control neutrality (standard of `b37b528`)

Two independent arguments, both satisfied:

**(a) Structural — stronger than `b37b528`.** `b37b528` was an *in-loop* logging change and needed a control
re-run diff to prove byte-identity. Here the edit is confined to a **post-processing script that never executes
in the simulation/control loop**. The C and U `sim_log.json` files it reads were produced by the frozen `32aefaf`
run *before this edit existed*; they are physically incapable of being perturbed by it.

**(b) Empirical — pre-existing 59 columns byte-identical.** Re-exporting C from the *same* committed
`figC25_addfive/sim_log.json` and comparing the first 59 fields of every row (via `csv.reader`, line-ending safe)
against the committed CSV at git HEAD:

```
rows compared   : 2078 (header + 2077)
mismatched rows : 0            → pre-existing 59 columns BYTE-IDENTICAL ✓
appended columns: omega_s_{x,y,z}_radps, Ltot_{x,y,z}_Nms, Ltot_norm_Nms
meta JSON       : run / n_ticks / at_weld_vs_min identical; only key added = ltot_snapshots
```

> **Caveat on `git diff --stat`:** it reports every CSV row as changed (≈4156 lines for C). That is expected —
> each row *gained* 7 fields, so each line is longer and git diffs it whole. Neutrality is a **field-level**
> property (first 59 fields unchanged), proven above; it is not visible in git's line-level stat.

---

## 3. The three checks (canonical managed run, `results/j2_adjconv/c25_fulldiag.csv`)

`I_comp = 2.18×10³ kg·m²` (composite structure+robot about the system CoM, given). DS_terminal = 201 ticks,
t ∈ [64.540, 84.540] s; entry = `dock_step5` snapshot @ 64.540 s.

### CHECK 1 — max |L_total| per axis over the whole run (44 snapshots)

| axis | max |L_total| [Nms] | at |
|---|---|---|
| x | **1.247×10⁻³** | t=84.54 s (final) |
| y | 6.085×10⁻⁴ | t=44.88 s |
| z | 6.428×10⁻⁴ | t=18.20 s |
| ‖·‖ | **1.473×10⁻³** | — |

Total angular momentum is at the **10⁻³ Nms numerical-noise floor on every axis** across the entire 6-step
traversal + settle — the system is momentum-conserving, as required (the platform starts from rest,
`|ω_s(t=0)|≈1×10⁻¹³`).

### CHECK 2 — ω_s,z vs the conservation prediction −h_w,z / I_comp through DS_terminal

| quantity | value |
|---|---|
| ω_s,z range | [+6.85×10⁻³, +4.18×10⁻²] deg/s |
| −h_w,z/I_comp range | [+6.76×10⁻³, +3.36×10⁻²] deg/s |
| ratio ω_z / pred | mean **0.984**, median **1.013**, std 0.099, min 0.850, max 1.331 (n=201) |
| implied I_comp = −h_w,z/ω_s,z | median **2152 kg·m²** (10–90 %: 2136–2465) vs given 2180 (**1.3 %**) |

The structure spin is explained by the wheel-momentum store via `ω_s,z ≈ −h_w,z/I_comp` to within the
docking transient; the implied composite inertia (2152) reproduces the given 2180 to 1.3 %.

**Time evolution (the conservation relation tightening as the settle progresses):**

| t [s] | ω_s,z [deg/s] | −h_w,z/I [deg/s] | ratio | ‖L_total‖ [Nms] |
|---|---|---|---|---|
| 64.54 (entry) | +0.04182 | +0.03358 | 1.245 | 1.26×10⁻³ |
| 66.54 | +0.02893 | +0.03298 | 0.877 | 1.26×10⁻³ |
| 69.54 | +0.02338 | +0.02626 | 0.891 | 1.26×10⁻³ |
| 74.54 | +0.01628 | +0.01723 | 0.945 | 1.26×10⁻³ |
| 79.54 | +0.01097 | +0.01080 | 1.016 | 1.47×10⁻³ |
| 84.54 (end) | +0.00685 | +0.00676 | 1.013 | 1.47×10⁻³ |

Both ω_s,z and −h_w,z/I decay together toward zero; ratio → 1.01 by the settle end.

### CHECK 3 — ω_s at DS_terminal entry vs predicted 0.0336 deg/s

At the entry tick (t=64.540 s, first DS_terminal tick = nearest `dock_step5`):

```
omega_s = (+2.451e-03, -2.562e-02, +4.182e-02) deg/s   |omega_s| = 4.911e-02 deg/s
h_w,z   = -1.2775 Nms  ->  -h_w,z / I_comp = +0.03358 deg/s   (target 0.0336)
```

The prediction **0.0336 deg/s is reproduced exactly** from the logged entry `h_w,z = −1.2775` and
`I_comp = 2180`. The **measured** ω_s,z at entry is **0.0418 deg/s — 1.24× the prediction**: the entry is the
fresh post-weld transient (multi-axis: a −0.0256 deg/s y-component is also present from the dock impulse), and
it relaxes onto the conservation relation over the settle (Check-2 table: ratio 1.245 → 1.013). This modest
entry excess is honest and expected — it is exactly the residual the T4 extension is designed to bleed to zero.

---

## 4. Verdict

All three checks are consistent:
1. **L_total ≤ 1.5×10⁻³ Nms on all axes, whole run** — momentum conserved (numerical zero).
2. **ω_s,z ≈ −h_w,z/I_comp** through DS_terminal (ratio mean 0.984 / median 1.013; implied I_comp 2152 vs 2180).
3. **Entry ω_s,z = 0.0418 deg/s** vs the 0.0336 deg/s conservation prediction — same sign/axis, 1.24× (docking
   transient), converging to the relation across the settle.

**Note carried to T4:** the 20 s terminal settle is *not* converged — θ_s,z climbs 0.174° → **0.535°** over the
20 s (residual +ω_s,z still integrating), and ω_s,z is still 0.0068 deg/s at the cutoff. The 0.54° plateau is the
angle accumulated at the 20 s truncation, not steady state. This is precisely what SUB-PHASE B (T4, 450 s settle)
is built to drive down (target |ω_s|<1×10⁻³ deg/s, |θ_s|<0.05°, with the transient wheel re-load visible).

---

## 5. Deliverables

| artifact | change |
|---|---|
| `scripts/diag_full_diag_export.py` | +7 export columns + meta key (append-only, `:27/94-104/124-125/150-152/174-175`) |
| `results/j2_adjconv/c25_fulldiag.csv` | 59 → 66 cols (managed run; first 59 byte-identical) |
| `results/j2_adjconv/u25_fulldiag.csv` | 59 → 66 cols (unmanaged run; first 59 byte-identical) |
| `results/j2_adjconv/{c25,u25}_fulldiag_meta.json` | + `ltot_snapshots` (44 L_total values each) |

**STOP** — gate before SUB-PHASE B (T4).
