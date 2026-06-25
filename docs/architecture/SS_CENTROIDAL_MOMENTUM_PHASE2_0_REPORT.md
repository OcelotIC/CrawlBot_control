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
| Plot helper (`_plot_tmom_tracking`) → `results/phase2_0_tmom/` | same |
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
   `scripts/test_qp_tracking.py:147`).

**Formulation evidence (after the harness fixes):**
- Static hold exact: `q̈`=1.6e-11, drift=0, task-row residual=2.8e-11.
- J̇_com·q̇ assembly correct: FD rel error **1.5e-7** (rules out an Ȧ_G·q̇ bug).
- Realized/commanded CoM accel **converges monotonically to unity** as task authority
  rises (ratio 0.198→0.662→0.853 for ss_α_mom 5e2→5e3→3e4), with |f_contact|→m·a — i.e.
  consistent **CoM-Jacobian (A_G/m) form, no fixed m≈71 or 1/71 offset**.

---

## 4. Achieved results vs tolerances (run at SHIPPING `ss_alpha_mom=500`)

| Test | Metric | Tolerance | Achieved | Status |
|------|--------|-----------|----------|--------|
| 1 Static hold | peak \|q̈\| | < 1e-2 | **1.6e-11** | ✓ |
| 1 | peak CoM drift | < 0.2 mm | **0.0000 mm** | ✓ |
| 1 | task-row residual (a_com_des reproduced at rest) | < 1e-6 | **2.8e-11** | ✓ |
| 2 | J̇_com·q̇ FD rel error (Ȧ_G·q̇ assembly) | < 1e-3 | **1.5e-7** | ✓ |
| 2 | peak CoM tracking (worst of 6 axis×profile) | < 2.5 mm | **1.45 mm** (x/y/z 0.82–1.45) | ✓ |
| 3 | mass ratio: monotone & top∈[0.6,1.4], all∈(0.05,2.0) | — | **[0.198, 0.662, 0.853]** | ✓ |
| 4 | Variant-B CoM tracking (ss_α_tl_weak=50) | < 2.5 mm | **1.37 mm** | ✓ |

**Tolerance rationale (stated per brief):**
- *Hold 0.2 mm / 1e-2*: from exact rest + ref=current + zero gravity the equilibrium is
  `q̈=τ=0`; any motion is a wrong PD sign or state-dependent reference.
- *Track 2.5 mm*: closed-loop tracking of a **moderate** jerk-limited reference (10 mm
  septic step / 6 mm sinusoid — peak demand within the linear task's null-space authority
  at the shipping weight). Every gross formulation error (sign→divergence, mass-factor→
  metres-or-zero, missing J̇→drift) is ≫1 cm; 2.5 mm catches them while absorbing PD lag +
  Euler error at dt=2 ms.
- *J̇ rel 1e-3*: independent finite-difference, weight-free.
- *Mass sweep*: a mass-scalar bug offsets the ratio by a **fixed** ~71 or ~1/71
  regardless of weight; correct form converges toward unity — so the test checks the
  *trajectory* of the ratio, not a single point (no reweighting-to-pass).

Plots: `results/phase2_0_tmom/t_mom_step_x.png`, `t_mom_sine_x.png` (commanded vs
realized CoM + task residual over time).

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

**Phase 2.0 GATE: PASS.** Per the stop-gate, **Phase 2.1 (single-step swing) has NOT been
started** — awaiting review.
