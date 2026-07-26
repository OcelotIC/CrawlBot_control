# INTERNAL — Fix-C mini-audit: can "weld-relative pose+twist → 0" be a terminal NMPC constraint? (`ae0673e`)

**One decisive question.** Fix C must make the dock arrive with **zero weld-relative twist** (`Jc·v⁻ → 0`,
kills the impact) and **zero weld-relative pose error** (kills the residual angular gap-couple, the 0.0040
N·m·s). The design favors a **terminal NMPC constraint**, but the NMPC state is only `[r_com, v_com, L_com]`.
Can the constraint live there, or must it attach to (b) the swing planner / (c) the dock gate? READ-ONLY,
branch `j2/ds-active-rework`. Reproducer `Misc/scripts/audit_fixC.py`: **13/13**.

**VERDICT — (a) is NOT formulable; route Fix C to (c) as the hard guard, backed by (b) as the driver.**

| insertion point | can carry Fix-C pose+twist? | hard or objective | why |
|---|---|---|---|
| **(a) terminal NMPC constraint** | **NO** | — | the gripper twist/pose is **not a function of the NMPC state** (the centroidal reduction is many-to-one; proven below) |
| **(b) swing-planner terminal** | **YES** | **objective** (feedforward ref) | terminal ref is already `v_ee=0 ∧ ω_ee=0` (6-D), pose → anchor; shapes the approach, cannot *guarantee* `Jc·v⁻=0` |
| **(c) dock gate** | **YES** | **hard** (boolean guard) | reuse `Jc·v⁻` (Fix-A relative-site Jacobian) + `d` + `ori`; guards the weld, does **not** actively drive the twist down |

**Recommendation:** Fix C = a **hard gate** `‖Jc·v⁻‖<ε` **and** a 6-D pose criterion at (c) (the binding
"no-impact" guarantee — don't weld until the twist is null), **driven** by (b) (the swing reference that
brings `Jc·v⁻` down so the gate fires without a dock-timeout). The two are complementary: (c) without (b)
risks never firing; (b) without (c) cannot guarantee. **(a) is out** — and that is the useful result.

---

## Q1 — Where do the gripper pose/velocity live in the pipeline?

- **Weld-relative twist `Jc·v⁻`** — computed in the **Fix-A impact block** (`sim_loop.py:2170-2179`): for each
  active weld it builds the **relative-site Jacobian** `J = [jpg − jpa ; jrg − jra]` (gripper-site minus
  anchor-site, linear+angular, via `mj_jacSite` over **all** `nv` DOFs) and `v_pre = J @ v_minus`. That
  `v_pre` **is** `Jc·v⁻` — the 6-D weld-relative twist. (This is the same relative-site relation the J1
  Lemma-2 work used to reconstruct the wrench — Q4: reuse it.)
- **Weld-relative pose** — position via `_gripper_distance` (`:1106`, `‖site_xpos[grip] − site_xpos[anch]‖`)
  and orientation via `_gripper_ori_err_deg` (`:1129`, angle between gripper `R` and anchor `R`). Together =
  the 6-D weld-relative pose (the full SE(3) error is in `site_xpos`/`site_xmat`).
- **Swing reference** (`swing_planner.py`) — a feedforward Cartesian gripper reference
  `(p_ee, v_ee, R_ee, ω_ee)` in the structure frame; the WBC/QP tracks it.
- **Contact Jacobians** in the QP (`get_contact_jacobians` → `J_tool_a/b`, `robot_interface.py:433`) — the
  Pinocchio tool Jacobians (6×nv), used for the welded-EE acceleration constraint.

So the Fix-C quantities are **already present** at dock time — none need to be rebuilt.

## Q2 — How do they relate to the NMPC state `[r_com, v_com, L_com]`? (DECISIVE)

**They are decoupled. The gripper twist/pose is NOT a function of the NMPC state.**

The NMPC state is purely the **centroidal reduction**: `NX=9`, `x=[r_com, v_com, L_com]`
(`centroidal_nmpc.py:123`) — **no gripper pose, no gripper velocity, no joint configuration**. The anchor
positions `r_C1, r_C2` enter only as **parameters** `p[6:12]` (`NP=18`, `:125`) — the *target* sites, not the
gripper, and the NMPC cannot drive them. The map from the full velocity `v ∈ ℝ^{nv}` to `[v_com, L_com]` is
the centroidal momentum matrix `A_G` (6×nv), which is **many-to-one**.

**Proof (real robot, `models/VISPA_crawling_fixed.urdf`, `nv=20`):** `A_G` has rank 6, so `null(A_G)` is
**14-dimensional**. A perturbation `δv ∈ null(A_G)`:
```
‖A_G·δv‖   = 5.5e-16   (the NMPC state [v_com, L_com] is UNCHANGED)
‖J_tool·δv‖ = 1.01      (the gripper twist CHANGES)
```
⇒ a 14-D family of motions leaves `[r_com, v_com, L_com]` **exactly fixed** while moving the gripper. Hence
`Jc·v⁻` cannot be written as `g(x_NMPC)`; **a terminal constraint `c(x,p) ≤ 0` on the weld-relative twist
(or pose) is not expressible in the 9-D NMPC.** Same argument for the pose (gripper SE(3) ≠ a function of the
CoM position + aggregate momentum).

## Q3 — Which of the three insertion points can carry the constraint?

**(a) Terminal NMPC constraint** (`centroidal_nmpc.py:309-326`, the `terminal_constraints(x,p)` / `c_simple`
tightening) — **NOT formulable.** That block constrains `hw_N = c_simple − L_com − r_com×m·v_com ∈
[−h_terminal, h_terminal]` — a function of the **centroidal state + a parameter**. By Q2 the gripper
twist/pose is *not* such a function, so it cannot be added here. (You would have to enlarge the NMPC state to
carry the gripper DOFs — out of scope, and contrary to the 9-D centroidal design.)

**(b) Swing-planner terminal** (`swing_planner.py`) — **CAN carry it, as an OBJECTIVE.** The terminal
reference is **already** `v_ee=0 ∧ a_ee=0` (quintic timing: `_quintic_dot(1)=0`, `_quintic_ddot(1)=0`,
`:419-427`) **and** `ω_ee=0` (delayed-cosine SLERP with zero terminal rate, `:321`), with `p_ee→anchor`,
`R_ee→anchor`. So the planner's terminal is **already a 6-D weld-relative-zero reference** — not "just linear
`v_ee`". But it is a **feedforward reference the WBC tracks**, i.e. an **objective**, not a hard constraint:
it *shapes* the approach but cannot **guarantee** the actual `Jc·v⁻=0` at dock (tracking error + structure
motion remain). Fix-C role here: **the driver** that brings `Jc·v⁻` down.

**(c) Dock gate** (`sim_loop.py:2010-2031`) — **CAN carry it, as a HARD GATE.** Today
`docked = pos_ok ∧ ori_ok ∧ vel_ok` (`:2031`), where `vel_ok = _gripper_speed < dock_vel_max` — and
`_gripper_speed` is the **LINEAR** EE speed only (`:1114`, "linear part"), **not** the 6-D weld-relative
twist. Fix-C here = replace `_gripper_speed` with **`‖Jc·v⁻‖ < ε`** (reuse the Fix-A relative-site Jacobian)
and keep/tighten the pose criterion (`d` + `ori` are the 6-D weld-relative pose). This is a **hard boolean
precondition** evaluated before `_activate_weld`, so it **guarantees** the weld never engages with nonzero
twist. Caveat: a gate is a **guard, not a driver** — if the approach never achieves `‖Jc·v⁻‖<ε`, the gate
simply never fires (dock-timeout). Hence it must be **paired with (b)**.

## Q4 — Are `Jc·v⁻` and the weld-relative pose already computed? (reuse)

**Yes — both, no rebuild needed.**
- `Jc·v⁻`: the **Fix-A relative-site weld Jacobian** `J=[jpg−jpa; jrg−jra]` and `v_pre = J @ v_minus`
  (`sim_loop.py:2170-2179`) — the exact 6-D weld-relative twist, over all `nv` DOFs. Same Jacobian the J1
  Lemma-2 audit (`scripts/audit_lemma2.py`) used to reconstruct the weld wrench. ⇒ Fix C's twist criterion is
  a `np.linalg.norm(v_pre)` on a quantity already built at dock.
- weld-relative pose: `_gripper_distance` (`:1106`) + `_gripper_ori_err_deg` (`:1129`), or the raw
  `site_xpos`/`site_xmat` differences. ⇒ the pose criterion reuses existing metrics.

---

## Divergence vs the J2-audit facts

The J2 cartography (Fix-C surface) said: *"the swing planner already drives v_ee=0 (and accel 0) at terminal
τ=1, and the dock gate requires v_ee<0.01 m/s; **but both gate the gripper's linear EE speed, not the 6-D
weld-relative constraint twist Jc·v⁻**."* Refinements on `ae0673e`:

1. **The swing planner terminal is already 6-D, not linear-only.** Its reference outputs **both** `v_ee` and
   `ω_ee`, both → 0 at τ=1 (verified: quintic terminal derivatives 0; SLERP zero terminal rate). So "not just
   linear `v_ee`" is **accurate for the dock GATE** (which checks `_gripper_speed`, linear only) but **not**
   for the planner *reference* (already 6-D-zero). The real gap at (b) is **objective-vs-guarantee**, not
   "missing angular".
2. **The dock gate's velocity check is confirmed linear-only** (`_gripper_speed`, `:1114`) — matches the J2
   flag. ✔
3. **New, decisive (not in the prior audit):** the terminal-NMPC option (a) is **structurally impossible**,
   proven by `null(A_G)` (14-D) moving the gripper at fixed centroidal state. The prior audit listed (a) as
   "the framing's preferred home; cleanest" — this audit **overturns that**: (a) cannot carry the constraint.
4. Everything else (Jc·v⁻ via the Fix-A relative-site Jacobian; pose via `d`/`ori`) **confirmed**.

## Reproducer

`Misc/scripts/audit_fixC.py` — READ-ONLY. Source anchors (Q1/Q4 reuse points, NMPC state/params, swing terminal,
dock gate) + the **real-robot many-to-one demo**: builds `RobotInterface` from the canonical URDF, computes
`A_G` and `J_tool`, and shows a `δv ∈ null(A_G)` leaves `[v_com,L_com]` fixed (‖A_G·δv‖≈5e-16) while moving
the gripper (‖J_tool·δv‖≈1.0).
```
MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/audit_fixC.py
→ VERDICT: 13/13 checks confirmed.  (a) NOT formulable; route to (c) guard + (b) driver.  (exit 0)
```

**STOP — doc-first.** The Fix-C implementation brief follows the digest. No `crawlbot/` change, no `main`
write, no PR, no implementation.
