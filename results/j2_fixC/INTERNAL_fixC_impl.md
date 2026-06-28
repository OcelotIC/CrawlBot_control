# INTERNAL — Fix C (J2 #1): 6-D weld-relative dock gate — implementation + characterization

**Implementation brief, then characterize — NO success threshold.** Raw numbers below; the performance
judgement (which ε, what residual is "good", the residual/timeout/C1–C5 trade) is for Idriss + reviewing
Claude. Branch `j2/ds-active-rework` (pushed, never merged). Base `ae0673e`.

**Headline (raw, no verdict):** the dock is **pose-gated** at the canonical working point — ‖Jc·v⁻‖ when
`d` first crosses 5 mm is ~0.006, below any ε_twist ≥ 0.02, so over [0.02, 0.20] the gate is inert
(residual fixed at 0.003977). **Tightening ε_twist into the binding regime (≤ 0.006) defers the weld into
the convergence-hold window**, where the gripper settles to **both lower twist and lower `d`**, dropping the
residual ~4× (0.003977 → 0.001005 at ε_twist=0.005), **still docking 5/5, all C1–C5 PASS**. Tightening the
position tolerance `weld_radius` instead **times out** below ~5 mm (the gripper's achievable dock distance
is ~4.5 mm). **This is contrary to the brief's prior expectation** (gap-couple/ε_pos-dominated): the data
shows **ε_twist is the effective, feasible residual knob; ε_pos is floor-limited.**

---

## Part 1 — Implementation (committed `027ab17`; tooling `d72d62a`)

The dock-gate velocity criterion is upgraded from the **linear EE speed** to the **6-D weld-relative twist**:
```
docked = (d < weld_radius) ∧ (ori_err_deg < dock_ori_threshold_deg) ∧ (‖Jc·v⁻‖ < dock_twist_max)
```
evaluated (`mj_forward` first) **before** `_activate_weld`, at **both** gate sites (main SS loop + the
convergence-hold loop).

- **`sim_loop.py`** — `_weld_relative_twist(arm, anchor_idx)` **reuses the Fix-A relative-site weld
  Jacobian** `J = [jpg − jpa ; jrg − jra]` (`mj_jacSite` for the gripper/anchor sites over **all** `nv`),
  `twist = J @ qvel` — **not rebuilt**, same construction as the impact block. `_dock_gate(...)` factors the
  predicate into one place; `twist_norm` is always computed (logged) even on the legacy path. Legacy linear
  gate kept behind `cfg.dock_use_6d_twist=False` for A/B. Dock-timeout path unchanged (clean timeout if ε
  never reached).
- **`config.py`** — `dock_use_6d_twist=True` (canonical), `dock_twist_max=0.05` (ε_twist; documented as a
  sweep starting point, **not** tuned). ε_pos = `weld_radius` (0.005), ε_ori = `dock_ori_threshold_deg` (5°)
  reused.
- **`logging.py`** — `dock_gate_trace` (per-eval ‖Jc·v⁻‖ + d + ori + fired) for the Part-2 trajectory;
  `twist` added to `dock_events`.
- **`diag_cooperative_arms.py`** — `--dock-twist-max`, `--dock-gate-linear`, `--weld-radius` CLI for sweeps.

**Regression (`pytest tests/`):** **220 passed, 1 failed.** The single failure
(`test_E7_t15_step2_dock_under_fk_mode`) is **pre-existing and identical on clean `ae0673e`** (verified by
stashing the diff: same FK-mode `preplanner_infeasible` + `dock_timeout`, `d=694 mm` — unrelated to the
gate). **No new breakage.** `test_reworked_qp`: 8 passed. **C6 OFF byte-identical** to the Fix-A baseline in
every run below (flag-OFF determinism intact).

## Part 2 — Does (b) the swing planner drive the gate?

**Yes — (b) drives ‖Jc·v⁻‖ down to a floor of ~0.0037**, so the gate fires through the driver (not by
timeout) for every ε_twist ≥ ~0.004. Near-dock ‖Jc·v⁻‖ trajectory (last gate evals before firing; `*`=fired):

| ε_twist | step-0 ‖Jc·v⁻‖ trajectory (→ fire) | twist@fire | d@fire |
|---|---|---|---|
| 0.007 (≈loose) | 0.0033 0.0050 0.0059 0.0061 **0.0060\*** | 0.0060 | 4.94 mm |
| 0.006 | 0.0033 0.0050 0.0059 0.0061 0.0060 **0.0058\*** | 0.0058 | 4.76 mm |
| 0.005 | 0.0050 0.0059 0.0061 0.0060 0.0058 0.0055 0.0053 **0.0050\*** | 0.0050 | 4.53 mm |
| 0.004 | 0.0058 0.0055 0.0053 0.0050 0.0046 0.0043 0.0040 **0.0037\*** | 0.0037 | 4.73 mm |

‖Jc·v⁻‖ **rises** to ~0.006 as the gripper closes the last mm (it is still moving relative to the anchor at
the mm scale), then the convergence-hold window settles it back down. So a *looser* gate fires early at the
~0.006 peak; a *tighter* gate waits through the hold until the twist relaxes to its ε. **The floor (~0.0037)
is what (b) achieves with the current swing planner + hold** — below it the gate would time out (an
active-nulling driver to push lower is a later brief, explicitly out of scope here). **No under-drive flag
for ε_twist ≥ 0.004.**

## Part 3 — Characterization (raw; measure, do not judge)

### 3.1  residual(ε_twist), weld_radius = 0.005 (fixed). 5-step traversal, dock momentum residual = final-snapshot ‖subtree_angmom[0]‖

| ε_twist | residual [N·m·s] | docks | timeouts | per-dock d [mm] | per-dock twist@fire |
|---|---|---|---|---|---|
| linear gate (A/B) | 0.003977 | 5 | 0 | 4.94 4.51 4.91 4.61 4.84 | 0.0060 0.0040 0.0060 0.0042 0.0058 |
| 0.02 … 0.20 | 0.003977 | 5 | 0 | (identical — pose-gated, twist slack) | |
| 0.007 | 0.003977 | 5 | 0 | 4.94 4.51 4.91 4.61 4.84 | 0.0060 0.0040 0.0060 0.0042 0.0058 |
| 0.006 | 0.003236 | 5 | 0 | 4.76 4.50 4.69 4.65 4.80 | 0.0058 0.0039 0.0058 0.0042 0.0057 |
| 0.005 | **0.001005** | 5 | 0 | 4.53 4.45 4.35 4.73 4.17 | 0.0050 0.0039 0.0048 0.0042 0.0048 |
| 0.004 | 0.001656 | 5 | 0 | 4.73 4.37 4.37 4.35 4.00 | 0.0037 0.0039 0.0039 0.0039 0.0038 |
| < ~0.0037 | — | (would time out: below the (b) twist floor) | | | |

Monotone-ish drop from 0.007→0.005 (0.003977 → 0.001005, **~4×**), then a small uptick at 0.004 (0.001656);
the 5 docks each shift firing-tick independently as ε tightens, so the total residual is not perfectly
monotone in ε_twist. (Loose ε ≥ 0.02 reproduces the pose-gated 0.003977 exactly.)

### 3.2  residual(ε_pos) = residual(weld_radius), ε_twist loose (0.05)

| weld_radius | residual | docks | timeouts | note |
|---|---|---|---|---|
| 0.005 (default) | 0.003977 | 5 | 0 | natural dock d ~4.5–4.9 mm |
| 0.004 | — | **0** | **1** | step-0 dock-timeout (t=10.91 s); gripper never reaches d<4 mm |
| 0.003 | — | 0 | 1 | step-0 dock-timeout |
| 0.002 | — | 0 | 1 | step-0 dock-timeout |

The gripper's **achievable dock distance is ~4.5 mm** (swing terminal + hold). `weld_radius ≤ 0.004` is below
that floor ⇒ the gate never fires ⇒ step-0 times out and the traversal aborts (0 docks). **ε_pos is not a
usable lever below ~5 mm.**

### 3.3  Twist-vs-pose split (what drives the 0.0040 residual)

- **The binding twist gate improves BOTH twist and `d` at the weld instant.** Going ε_twist 0.007→0.005, the
  weld defers into the hold window and the dock state moves from (twist 0.006, d 4.8 mm) to (twist 0.0048,
  d 4.4 mm); residual 0.003977 → 0.001005. The two co-improve (the hold settles both), so the residual
  reduction is **not** cleanly separable into "impact part" vs "gap-couple part" — but **ε_twist is the lever
  that accesses the low-residual regime**, because it buys settle time before the weld.
- **ε_pos cannot access it directly:** demanding a smaller `d` (weld_radius↓) just removes the only firing
  tick → timeout. The position floor (~4.5 mm) is set by the swing approach, not the gate.
- **Contrary to the prior expectation** (Fix A made the impact map momentum-consistent ⇒ residual expected
  gap-couple-dominated / ε_pos-sensitive): the data shows the residual **is** reducible, and the lever is
  **ε_twist** (deferring the weld to a lower-twist-and-lower-`d` state), **not** ε_pos (floor-limited). The
  remaining ~0.001 floor at ε_twist=0.005 is the dock state the hold can reach with the current driver.

### 3.4  Gate firing behaviour

| regime | outcome |
|---|---|
| ε_twist ∈ [0.004, 0.20], weld_radius=0.005 | **5/5 dock, 0 timeouts.** Dock time shifts +0.4–0.8 s later as ε_twist tightens (deferred into hold). |
| ε_twist < ~0.0037 | expected timeout (below the (b)-achievable twist floor) — not run |
| weld_radius ≤ 0.004 (twist loose) | **0 docks, step-0 timeout** (position floor) |

### 3.5  Effect on C1–C5 (raw, no pass/fail editorialising — gate prints PASS/values)

All **docking** runs (ε_twist 0.004–0.20) pass C1–C5; the only monotone trend is C5 (h_w∞) easing slightly as
ε_twist tightens:

| ε_twist | C1 docking (d mm) | C3 ‖Ḣ_s‖∞ SS | C5 h_w∞ | C6 OFF |
|---|---|---|---|---|
| 0.007 | PASS [4.94,4.51,4.91,4.61,4.84] | 5.00 ≤5 | 4.373 ≤4.5 | BIT-IDENTICAL |
| 0.006 | PASS [4.76,4.50,4.69,4.65,4.80] | 5.00 | 4.344 | BIT-IDENTICAL |
| 0.005 | PASS [4.53,4.45,4.35,4.73,4.17] | 5.00 | 4.283 | BIT-IDENTICAL |
| 0.004 | PASS [4.73,4.37,4.37,4.35,4.00] | 5.00 | 4.248 | BIT-IDENTICAL |

(C2 torso-track PASS, C4 attitude PASS in all; identical to the loose-ε values. ε_pos timeout runs have no
docking traversal to gate.) C3's realized ‖Ḣ_s‖∞_SS = 5.00 is the AOCS envelope (unchanged by Fix C — the
planned constraint is untouched). **No C1–C5 regression from the 6-D gate at any docking ε_twist.**

---

## Flags / divergences vs the J2 audit facts

1. **Contrary to the prior expectation** (Fix-C audit / brief): the residual is **not** ε_pos-dominated — it
   is reduced by **ε_twist** (deferring the weld into the settle window) and is **floor-limited** in ε_pos
   (timeout below ~5 mm). Reported plainly, not forced.
2. The audit's "(c) is a guard, not a driver" holds and is now quantified: the gate **defers** the weld; the
   actual twist reduction is done by **(b)** the swing planner + the hold window (floor ~0.0037). The gate
   cannot drive the twist itself — confirmed.
3. (b)'s swing-planner terminal reference being 6-D-zero (audit) is consistent: the realized ‖Jc·v⁻‖ relaxes
   to ~0.0037 in the hold, not exactly 0 (tracking + structure motion), matching the audit's
   objective-vs-guarantee point.

## Reproduce

```
# implementation regression
MUJOCO_GL=disabled PYTHONPATH=. python3 -m pytest tests/ -q
# sweeps (canonical working point; each run ~140 s)
bash scripts/run_fixC_sweep.sh    # ε_twist ∈ {linear, 0.02,0.05,0.10,0.20}
bash scripts/run_fixC_sweep2.sh   # ε_twist ∈ {0.004..0.007} + weld_radius ∈ {0.002,0.003,0.004}
# residual / twist-pose / firing for any run dir(s)
MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_fixC_residual.py results/<dir> ...
```
Supporting logs: `sweep1_residual.log`, `sweep1_gate_C1-C5.log`, `sweep2_residual.log`,
`sweep2_gate_C1-C5.log` (this dir). Raw per-run sim dirs are reproducible from the scripts and are not
committed (bulk).

**STOP — doc-first.** No success threshold applied; the ε choice and the residual/timeout/C1–C5 trade are
yours to decide on these numbers. No merge, no PR.
