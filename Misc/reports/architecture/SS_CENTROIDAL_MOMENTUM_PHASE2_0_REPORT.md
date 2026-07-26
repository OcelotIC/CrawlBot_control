# SS Centroidal-Momentum Task — Phase 2.0 Report

**Standalone task validation (unit; no swing / no NMPC / no MuJoCo)**

- **Branch:** `feat/ss-centroidal-momentum-task`
- **Built on:** `f04e79f` (memo patch: Phase 2.0 inserted into §4)
- **Date:** 2026-06-25
- **Governing doc:** `SS_CENTROIDAL_MOMENTUM_TASK_2026-06.md` §4 Phase 2.0

---

## 1. Verdict — GATE PASS

All four Phase-2.0 tests pass; T7–T10 (existing standalone-QP tests) still pass
(no regression). **The T-MOM linear task is correctly formulated and realized.**
The two initial failures were **harness-default bugs, not formulation bugs**, found
by cascade bisection and fixed in the test harness only.

```
tests/test_reworked_qp.py ........  8 passed   (T7,T8,T9,T10 + TestPhase20TMOM x4)
```

---

## 2. What was added (test code only — shipping `crawlbot/` untouched)

| Item | Location |
|------|----------|
| **Z1** analytical CoM driver (`_septic`, `_com_step_reference`, `_com_sine_reference`) | `tests/test_reworked_qp.py` |
| **Z2** comparator (`_com_task_probe`: realized `J_com·q̈+J̇_com·q̇` vs `a_com_des`) | same |
| Independent J̇_com·q̇ (Ȧ_G·q̇) finite-difference check (`_jdot_com_fd_check`) | same |
| Plot helper (`_plot_tmom_tracking`) → `Misc/runs/phase2_0_tmom/` | same |
| `TestPhase20TMOM` (4 tests) | same |
| Harness extension: `_make_m2_qp(cooperative, ss_mom, ss_alpha_mom, ss_alpha_tl_weak, alpha_torso_ang, alpha_wrench)` — defaults preserve legacy M2 | same |
| Harness fix: `_solve_qp_step` now passes `r_com=rs.r_com` to `qp.solve` | same |

Harness setup: `cooperative_arms_mode=True` + `ss_centroidal_momentum_task=True`,
`p_ee_ref=None` (EE task off), frozen nc=1 `SINGLE_A` (`dock_state`), `lambda_ref=0`
(NMPC bypassed), `alpha_com_soft=0` (T-MOM is the sole CoM driver), CoM references
injected via `r_com_ref/v_com_ref/a_com_ff`.

---

## 3. Cascade-bisection diagnosis (both failures were harness, not formulation)

Initial run: all 4 failed (static-hold diverged 28 mm; tracking 438 mm; mass ratio 2.06).
Bisection isolated **two latent defects in the pre-existing `_solve_qp_step` helper**,
both harmless when CoM is a weak P4 soft task (T7/T8/T10) but fatal when CoM is the
strong P2 T-MOM task:

1. **Missing `r_com=` kwarg.** `_solve_qp_step` never forwarded the actual CoM, so
   `solve()` used its default `r_com_actual=0` (`wholebody_qp.py:600`). The CoM PD term
   became `Kp_com·r_com_ref ≈ 100·[1.03,−0.12,−1.21] = [103,−12,−121] m/s²` — a spurious
   ~10² m/s² command (confirmed arithmetically: `r_com0=[1.025,−0.116,−1.211]`). Fix:
   pass `r_com=rs.r_com`.
2. **`alpha_wrench=1e2` with no NMPC.** With `lambda_ref=0` the wrench task degenerates
   to "minimise contact force", but with one EE welded the contact force is the *only*
   means of CoM acceleration (net external force = contact force = m·a_com). Sweep
   confirmed suppression: realized a_com_x rose 0.00003→0.0099 and |f_contact| 0.002→0.729
   as α_wrench dropped 100→0.01. Fix: `alpha_wrench=1e-2` (same correction documented in
   `Misc/scripts/test_qp_tracking.py:147`).

**Formulation evidence (after the harness fixes):**
- Static hold exact: `q̈`=1.6e-11, drift=0, task-row residual=2.8e-11.
- J̇_com·q̇ assembly correct: FD rel error **1.5e-7** (rules out an Ȧ_G·q̇ bug).
- Realized/commanded CoM accel **converges monotonically to unity** as task authority
  rises (ratio 0.198→0.662→0.853 for ss_α_mom 5e2→5e3→3e4), with |f_contact|→m·a — i.e.
  consistent **CoM-Jacobian (A_G/m) form, no fixed m≈71 or 1/71 offset**.

---

## 4. Gate basis and achieved results (run at SHIPPING `ss_alpha_mom=500`)

**Gate (per the review decision): TASK-INTRINSIC only —** (i) formulation correctness and
(ii) authority monotonicity. Per-step tracking error and accel-residual are **REPORTED as
characterization, NOT pass/fail thresholds** (their representative value is only meaningful
under swing — Phase 2.1). Tests 2 & 4 keep a generous **divergence/sign guard** (50 mm),
not a fidelity gate.

| # | Metric | Role | Criterion | Achieved | Status |
|---|--------|------|-----------|----------|--------|
| 1 | peak \|q̈\| | **gate** (formulation) | < 1e-2 | **1.6e-11** | ✓ |
| 1 | peak CoM drift | **gate** | < 0.2 mm | **0.0000 mm** | ✓ |
| 1 | task-row residual (a_com_des reproduced at rest) | **gate** | < 1e-6 | **2.8e-11** | ✓ |
| 2 | J̇_com·q̇ FD rel error (Ȧ_G·q̇ assembly) | **gate** (formulation) | < 1e-3 | **1.5e-7** | ✓ |
| 3 | mass ratio monotone-with-authority, top∈[0.6,1.4], all∈(0.05,2.0) | **gate** (monotonicity) | see §3 | **[0.198,0.662,0.853]** | ✓ |
| 2 | peak CoM tracking (worst of 6 axis×profile) | *reported* | (char.) | **1.45 mm** (axes 0.82–1.45) | — |
| 2/4 | divergence guard | guard | < 50 mm | 1.45 / 1.37 mm | ✓ |
| 4 | Variant-B vs Variant-A tracking | *reported* + coexist | B ≤ 1.5·A | A 1.32 / B 1.37 mm (×1.04) | ✓ |

**Authority characterization (reported — feeds the Phase-2.1 weight sweep):** CoM tracking
tightens and the realized/commanded accel ratio rises monotonically toward unity as
`ss_alpha_mom` is raised (α_wrench=1e-2 throughout):

| ss_alpha_mom | step track [mm] | sine track [mm] | accel ratio (rest, ‖f‖→m·a) |
|---|---|---|---|
| 500 (shipping) | 4.93 | 4.35 | 0.198 |
| 2000 | 1.55 | 2.17 | — |
| 5000 | 0.83 | 1.60 | 0.662 |
| 10000 | 0.59 | 1.40 | — |
| 30000 | 0.45 | 1.26 | 0.853 |

(The §4 per-axis 0.82–1.45 mm row uses the *moderate* 10 mm/6 mm references; this table uses
the 30 mm/15 mm aggressive references to expose the authority knob. No wall — a continuous
weight knob, confirming the limitation is authority, not formulation.)

Plots: `Misc/runs/phase2_0_tmom/t_mom_step_x.png`, `t_mom_sine_x.png`.

---

## 5. Honest caveat (context for Phase 2.1 — not a defect)

At the **shipping weight** the per-step *full* CoM-accel row `J_com·q̈+J̇_com·q̇` does **not**
equal `a_com_des` away from rest (residual ≈ the commanded magnitude). This is **correct
behaviour**, two effects: (a) the T-MOM task is *projected* (it drives `A_com·N_torso·z`,
not `A_com·z`); (b) it sits at P2 below the P1 torso-angular hold, which (under
`weight_ratio=1`, equal weights) rightly refuses to rotate the torso to chase an
instantaneous CoM accel spike. **Position still tracks** (≤1.45 mm) because feedback
integrates around it, and the accel row is reproduced exactly at static hold. The
authority sweep (§3) shows fidelity rises with weight. **Implication for Phase 2.1:** the
linear task's authority vs the torso-angular P1 hold is a weight-balance question to watch
in the coordinated swing scenario — but it is not a Phase-2.0 formulation concern.

---

## 6. Bit-identical-OFF re-check

The shipping flag `ss_centroidal_momentum_task` defaults **OFF**. All changes are
**test-only** (`tests/test_reworked_qp.py`) + docs + new result plots; **`git diff` shows
zero `crawlbot/` changes**, so the flag-OFF control path is bit-identical to the Phase-1
baseline **by construction**.

---

## 7. Gate

**Phase 2.0 GATE: PASS** on task-intrinsic grounds — (i) formulation correctness (static
hold exact, Ȧ_G·q̇ FD 1.5e-7, mass-factor with no m-offset) and (ii) authority monotonicity
(realized/commanded rises monotonically toward unity with `ss_alpha_mom`). Per-step tracking
and accel-residual are reported characterization, not thresholds (review decision). Per the
stop-gate, **Phase 2.1 has NOT been started** at the time of this report.

## 8. Carry-forwards to Phase 2.1 (decisions recorded; do not act on here)

1. **`ss_alpha_mom` is NOT frozen at 500.** Phase 2.1 will **sweep** it (500 / 5000 / 30000).
   Rationale: 500 (= `alpha_torso_ang` under `weight_ratio=1`) balanced torso-linear vs EE,
   but T-MOM replaces torso-linear with a CoM-*acceleration* task of different nature, so the
   1:6 balance is reconsidered under swing, not inherited.
2. **Ratio question (open for Phase 2.1 — do not pre-decide architecture).** The standalone
   shows that at equal P1/P2 weights the strict-P1 torso-angular hold (via null-space
   projection) rightly refuses to rotate the torso to chase an instantaneous CoM-accel spike,
   so per-step CoM-accel fidelity is partial at the shipping weight (position still tracks via
   feedback). This is a **weight-balance** property, expected to relax under swing (torso not
   frozen; P1 constrains orientation only, swing-arm columns add coordination DOF). **Only if**
   the Phase-2.1 sweep fails to restore CoM authority (inadequate at all weights, or
   torso-angular degrades unacceptably as α_mom rises) does the strict-P1-vs-strong-weighted-P2
   **architecture** question arise — to be raised then, not now.
