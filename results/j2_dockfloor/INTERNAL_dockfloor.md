# INTERNAL — Dock-floor passivity audit: is the ~4.5 mm floor passivity-limited or kinematic? (`ae0673e`)

**Implementation-light characterization — NO success threshold.** Raw numbers; interpretation for Idriss +
reviewing Claude. Branch `j2/ds-active-rework` (pushed, never merged). Dock-only (no CoM translation,
`ds_mobile_com_magnitude=0`).

**VERDICT — the ~4.5 mm dock floor is KINEMATIC, not passivity-limited.** Decisively: the dock fires **during
the passivity-OFF swing**, before the convergence-hold window is ever reached, and the close needs **no
positive joint work** (`dqⱼᵀτ_q ≤ 0` throughout). Forcing passivity back ON in the hold window changes
nothing — off and strict are **byte-identical**. There is no passivity to relax (the whole SS close is
already passivity-off), so no relaxation can bring the gripper below ~4.5 mm. The floor is the swing-terminal
+ anchor **geometry**, addressable only by the approach, not by passivity / Piste A.

---

## Background framing — and a premise correction

The brief's premise ("under strict passivity the dock floors at ~4.5 mm") rests on the SS-hold comment
(`sim_loop.py:2144-2150`): a past test with `passivity_hold=True` "prevented the arm from doing the positive
work to close the last few mm." **But the dock close is NOT under strict passivity.** All of SS runs
passivity-OFF: `passivity_active = use_m2_stack and (phase=='DS' or passivity_hold)` (`:2857`), and the SS
swing + hold pass `passivity_hold=False`. So the ~4.5 mm C1 baseline `d=[4.94,4.51,4.79,4.64,4.85]` is the
**passivity-OFF** dock, not a strict-passivity floor. This audit tests the premise by forcing passivity back
ON in the hold window and relaxing it by a constant budget.

## Method (implementation-light, default-off — committed `d3e5b0f`)

Knobs in the SS dock-close hold window (`sim_loop.py:2047-2071`): `dock_hold_passivity_on` (force passivity
ON), `passivity_W_budget` (relax the RHS to `dqⱼᵀτ_q + 2α T_kin ≤ W_budget`), `log_dock_work` (per-SS-tick
`dqⱼᵀτ_q` + `d` + passivity trace). Dock gate held fixed (measure the *achievable* distance, not a gate
change). All default-off ⇒ byte-identical. Sweep: off (current) / strict (ON, W=0) / budgeted (ON,
W∈{0.5,2,10}). Canonical working point, 5-step traversal.

## Results (raw)

### Decisive: dock distance + positive work, off vs strict vs budgeted

| level | worst dock [mm] | per-step d [mm] | close-window `dqⱼᵀτ_q` max [W] | positive-work frac | passivity active in close | docks/timeouts |
|---|---|---|---|---|---|---|
| off (current) | **4.94** | [4.94,4.51,4.91,4.61,4.84] | −0.0009 | 0.00 | **False** | 5 / 0 |
| strict (ON, W=0) | **4.94** | [4.94,4.51,4.91,4.61,4.84] | −0.0009 | 0.00 | **False** | 5 / 0 |
| budgeted W=0.5 | **4.94** | [4.94,4.51,4.91,4.61,4.84] | −0.0009 | 0.00 | **False** | 5 / 0 |
| budgeted W=2.0 | **4.94** | [4.94,4.51,4.91,4.61,4.84] | −0.0009 | 0.00 | **False** | 5 / 0 |
| budgeted W=10 | **4.94** | [4.94,4.51,4.91,4.61,4.84] | −0.0009 | 0.00 | **False** | 5 / 0 |

**All five levels are identical** (4.940 mm worst, same per-step vector). The decisive evidence is in the
`passivity active in close = False` and `positive-work frac = 0.00` columns for **every** level: every
close-window tick (d<10 mm, 386 ticks) is a **swing** tick with passivity OFF, and the arm does **no**
positive work (`dqⱼᵀτ_q ≤ −0.0009 W` — the swing is decelerating, v→0 at τ=1). So **the dock fires during
the swing, never reaching the hold window** — which is why forcing passivity ON there (strict) and relaxing
it by any budget (0.5/2/10) all have **zero** effect on the dock. (The budget *does* touch the DS settle —
see the negligible residual/`h_w` shifts below — just not the dock.)

### Dock twist ‖Jc·v⁻‖ at the gate
Per step `[0.0060, 0.0040, 0.0060, 0.0042, 0.0058]` — identical off/strict (the close is unchanged), and
consistent with the Fix-C audit (the swing terminal drives the twist to ~0.006).

### Residual (post-dock subtree_angmom), traversal-final
off **0.003977**, strict **0.003977**, W0.5/W2/W10 **0.003978**. Identical to 6 digits between off and
strict (the dock is unchanged); the budgeted runs differ by **+1e-6** — the W-budget relaxes the **DS-settle**
passivity slightly, not the dock. The closer/relaxed dock does **not** change the conservation residual.

### C1–C5 (raw, no verdict) — all five levels
- **C1 docking: PASS** (all levels), `wp_d=[4.94,4.51,4.91,4.61,4.84]` (identical) — worst margin = the
  baseline's worst, so it passes (cf. the α m=0.10 run which tripped C1 at 4.99 mm; here no dock exceeds
  4.94 mm). **C2/C3/C4: PASS** (unchanged across levels).
- **C5 h_w∞: PASS** — off/strict **4.373**, budgeted **4.374** (≤4.5; the +0.001 is the DS-settle budget,
  not the dock).
- **C6 OFF determinism: BIT-IDENTICAL** at every level (`worst|delta|=0`); `test_reworked_qp` 8 passed.

### Feasibility
No timeouts, no QP/NMPC infeasibility at any level (5/5 dock).

## Decisive output

**Does relaxing passivity bring the gripper closer than the strict ~4.5 mm floor? — NO.** The floor is
**KINEMATIC**: the gripper reaches ~4.5 mm during the **passivity-off swing**, the close requires **no
positive work**, and the convergence-hold window (the only surface the passivity knob touches) is **never
reached**. There is no passivity acting on the close to relax — it is already fully off — so no relaxation,
and thus no Piste A, can improve the dock distance. **The tight dock margin is an approach-geometry problem
(swing terminal + anchor), separate from passivity.**

### Realistic best dock (geometric floor, passivity fully off) — for the paper
With passivity fully off (the current and most-relaxed state), the swing-terminal + anchor geometry brings
the gripper to **~4.5–4.9 mm** (per-step `[4.94, 4.51, 4.91, 4.61, 4.84]`, worst **4.94 mm**, best 4.51 mm).
**A ~5 mm dock tolerance (`weld_radius`) is the realistic expectation**; the Fix-C audit's `weld_radius<4 mm
→ timeout` is the other side of the same geometric floor. Improving dock margin requires a better *approach*
(swing-terminal geometry / a closer pre-dock pose), not passivity.

## Flags / divergences vs the Fix-C / α / passivity audit facts

1. **Premise correction (vs this brief + the α/passivity framing):** the dock close is **not** under strict
   passivity — all SS is passivity-off, and in the canonical config the dock fires **during the swing**, so
   the hold-window passivity (the SS-hold comment's subject) is **never even reached**. The "passivity blocks
   the last-mm close" observation came from a configuration where the hold window ran; the current canonical
   dock does not hit it.
2. **Consistent with Fix-C:** the dock is twist-clean (~0.006) and pose-gated at ~4.5 mm; Fix-C showed
   `weld_radius<4 mm → timeout` — same geometric floor, now attributed to the swing approach, not passivity.
3. **Consistent with α (J2 #2):** there the conflict was passivity-dominated because the **CoM was
   translating** (demanding work in DS); here, **docking only** (no translation), the close needs no positive
   work and passivity is uninvolved. The two are not contradictory: passivity bites when the reference
   demands work (moving CoM), not when the arm coasts to a static dock target.

## Reproduce
```
bash scripts/run_dockfloor_sweep.sh   # off / strict / W{0.5,2,10}; n=5; dock-only
MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_dockfloor.py off=results/df_off strict=results/df_strict ...
```
Supporting: `results/j2_dockfloor/dock.log`, `residual.log`, `gate_C1-C5.log`. Raw per-run dirs reproducible
from the script, not committed.

**Regression (`pytest tests/`):** defaults are dormant and **byte-identical** — `W_budget+0.0 ≡ W`,
`dock_hold_passivity_on=False`, `log_dock_work=False` — and **C6 is BIT-IDENTICAL in every sweep run above**
(direct evidence the flag-OFF path is unchanged). Full-suite count: **220 passed, 1 failed** — the single
failure is the pre-existing FK test `test_E7_t15_step2_dock_under_fk_mode` (identical on clean `ae0673e`).
No new failures.

**STOP after the report.** No success threshold; Piste A design follows the digest. No merge, no PR.
