# Phase QP-STACK-AUDIT — read-only code audit of the SS Whole-Body QP (§VI-D)

**Read-only code audit — no code/config change, no simulation run.** Every claim carries `file:line` + the
verbatim line. Audited on the **canonical path** (`reference_source='task_space'`, `use_mid_waypoint_reshape=False`,
`use_path_feasibility_check=False`, `ss_two_task_mode=True`, frozen `32aefaf`+`ec41cd9`) — i.e. what the C/U runs
actually execute, verified: the runners never set those flags and `config.py` defaults them off.

---

## Q1 — SWING END-EFFECTOR TASK

**a) Where built.** `crawlbot/solvers/wholebody_qp.py:700-716`, inside the `if _two_task:` block.
- `wholebody_qp.py:700` — `            # (3) Swing-EE 6-D task (direct, no projection).`
- `wholebody_qp.py:701` — `            if J_ee is not None and p_ee_ref is not None:`

**b) 6-D (pose), not 3-D.** The task error stacks a 3-D position residual and a 3-D log-map orientation residual,
and the task Jacobian block has **6** rows:
- `wholebody_qp.py:707` — `                e6e = np.concatenate([p_ee_ref - p_ea, pin.log3(R_ea.T @ R_er)])`
- `wholebody_qp.py:713` — `                A_ee2 = np.zeros((6, n))`

**c) Frame.** `J_ee` is the Jacobian of the Pinocchio frame **`"tool_a"` / `"tool_b"`**, `LOCAL_WORLD_ALIGNED`.
- `crawlbot/simulation/sim_loop.py:3715` — `            return rs.J_tool_b, rs.Jdot_dq_tool_b, rs.oMf_tool_b`
- `crawlbot/simulation/sim_loop.py:3717` — `            return rs.J_tool_a, rs.Jdot_dq_tool_a, rs.oMf_tool_a`
- `crawlbot/core/robot_interface.py:204` — `        self.frame_tool_a = self.model.getFrameId("tool_a")`
- `crawlbot/core/robot_interface.py:322` — `        J_tool_a = pin.getFrameJacobian(` … `:323` — `            model, data, fid_a, pin.LOCAL_WORLD_ALIGNED).copy()`

So the frame name in the model is **`tool_a` / `tool_b`** (the gripper/tool frame), not the last link.

**d) Weight = 1000.** The task weight is `cfg.alpha_ee`:
- `wholebody_qp.py:716` — `                qp.add_task(A_ee2, a_e_des - jdq_ee, cfg.alpha_ee, priority=2)`

`alpha_ee` is fed from `cfg.ss_alpha_ee` (sim_loop `_build_qp` maps the 3rd positional arg → `alpha_ee`), whose
frozen value is 1000:
- `crawlbot/simulation/config.py:319` — `    ss_alpha_ee: float = 1e3       # CANONICAL-2p5 / Add-5 freeze (was 3e3)`

**e) Dropped in DS (not a stale reference).** The entire swing-EE block is inside `if _two_task:`, and `_two_task`
is false whenever `settle_mode` (DS) is true, so the task is **not added to the cost** in DS:
- `wholebody_qp.py:678` — `        _two_task = cfg.ss_two_task_mode and not settle_mode`

(The legacy non-two-task EE path is likewise settle-gated — `wholebody_qp.py:896` … `and not settle_mode`.)

---

## Q2 — SWING REFERENCE GENERATION

**a) Generator.** `crawlbot/planning/swing_planner.py:500` — `    def reference_at(self, t: float) -> SwingReference:`
(the scheduler-driven path; on the canonical there is **no** phase override registered — see the parenthetical
under (d)). Called from `sim_loop.py:3018` — `                sr = self.swing_planner.reference_at(`.

**b) POSITION — quintic to the FULL anchor, no start delay.**
- Profile: quintic `s(τ)=10τ³−15τ⁴+6τ⁵` (`swing_planner.py:18`, evaluated at `:556` — `        s = self._quintic(tau)`).
- Applied to the **full** start→end displacement:
  `swing_planner.py:552` — `        dp = p_end - p_start`
  `swing_planner.py:558` — `        p_ee = p_start + dp * s + self.clearance * n * bump`
  ⇒ **target fraction = 1.0** (reaches the anchor), NOT the torso's `κ_f = 0.70`.
- Start delay: **none** for position — the quintic runs from `τ=0`. (The torso's `κ_d = 0.20`-style delay applies
  only to *orientation*, see (e).)
- Duration: `T_eff = T · early_finish_fraction` — `swing_planner.py:539` — `        T_eff = T * self.early_finish_fraction`,
  with `swing_early_finish_fraction` frozen at **1.0** (`config.py:545`), so `T_eff = T_step`.

**c) LIFT / clearance arc — PRESENT.** There **is** an out-of-plane arc; the schematic figure is correct.
- `swing_planner.py:558` — `        p_ee = p_start + dp * s + self.clearance * n * bump`
- normal direction `n = self.away_normal` (`swing_planner.py:553` — `        n = self.away_normal`)
- bell profile `bump` peaking at mid-swing (`swing_planner.py:557` — `        bump = self._bump(tau)`; `bump=sin²(πτ)`, `:19`)
- magnitude `clearance = cfg.swing_clearance = 0.03 m` (`config.py:409` — `    swing_clearance: float = 0.03  # [m]`).

**d) ORIENTATION — SLERP (delayed-cosine), NOT held constant; and the target is a FIXED identity, not read from the anchor.**
It is an interpolation:
- `swing_planner.py:581` — `        dR = self._R_start.T @ self._R_end`
- `swing_planner.py:583` — `        R_ee = self._R_start @ pin.exp3(sigma_r * omega_total)`

`self._R_start` = the swing tool's orientation captured at release:
- `sim_loop.py:2236-2237` — `                    self.swing_planner.set_swing_orientation(` / `                        oMf_release.rotation)`
- `swing_planner.py:158` — `        self._R_start = R_start.copy()`

`self._R_end` = **`np.eye(3)`**, set once at construction and **never reassigned on the canonical path** (grep of
`crawlbot/` confirms the only writes to `_R_end` are the init and, inside `add_phase`, into a phase-override dict
that is not installed here):
- `swing_planner.py:95` — `        self._R_end: np.ndarray = np.eye(3)`

  > **DIVERGENCE (flag).** The paper's "the target anchor's orientation, held CONSTANT for the whole swing" is
  > wrong on **both** counts: (i) it is a delayed-cosine **SLERP** from the release orientation, not a constant;
  > (ii) the SLERP target is the fixed **structure-frame identity** (`_R_end = eye(3)`, the constructor default),
  > **not** a per-step read of the target anchor's orientation. The two happen to coincide numerically only
  > because the anchors are ≈ axis-aligned in the structure frame (docks achieve `ori ≤ 0.38°` at cap 2.5), which
  > is why docking still works — but the code neither holds constant nor reads the anchor. Recommend the paper say:
  > "the swing tool orientation SLERPs (delayed-cosine timing) from its release orientation to the fixed
  > structure-frame reference `I`, which coincides with the axis-aligned anchor frame."

  (The mid-waypoint / FK paths at `sim_loop.py:1776` and `:1794` *do* pass real anchor orientations into
  `add_phase` → `_override_reference_at`, but neither fires on the canonical: `reference_source='task_space'`
  (`config.py:523`) and `use_mid_waypoint_reshape=False` / `use_path_feasibility_check=False` (`config.py:498-499`),
  none overridden by the runners.)

**e) Separate time profiles.** Position uses the quintic `s(τ)`; orientation uses a **separate** delayed-cosine
`σ_r(τ, τ_d)` that stays 0 until `τ = rotation_delay_ratio` (frozen 0.2) then rises:
- `swing_planner.py:576` — `        tau_d = self.rotation_delay_ratio`
- `swing_planner.py:577` — `        sigma_r = self._delayed_cosine(tau, tau_d)`
- `swing_planner.py:84` — `        rotation_delay_ratio: float = 0.2,`

---

## Q3 — TORQUE-MINIMISATION TERM

**a) Form = `‖τ_q‖²`** (plain squared norm of the joint torques; the residual target is zero, uniform scalar weight):
- `wholebody_qp.py:1265` — `        A_torque = np.zeros((nq, n))`
- `wholebody_qp.py:1266` — `        A_torque[:, idx['tau'][0]: idx['tau'][1]] = np.eye(nq)`
- `wholebody_qp.py:1267` — `        b_torque = np.zeros(nq)`
- `wholebody_qp.py:1270` — `        qp.add_task(A_torque, b_torque, cfg.alpha_torque, priority=5)`

`A·z − b = τ_q − 0` ⇒ cost `= α_τ · ‖τ_q‖²`. It is **not** `‖τ_q − τ_ref‖²` and **not** per-joint weighted
(the only weighting is the scalar `α_τ`). This confirms the paper's inferred form.

**b) `alpha_tau = 5`** — a QP-construction literal in `sim_loop._build_qp` (not a `config.py` field):
- `crawlbot/simulation/sim_loop.py:1145` — `            alpha_torque=5e0, alpha_reg=1e0,`

---

## Q4 — INACTIVE CONTACT IN SINGLE SUPPORT

**a) NMPC — box-constrained to zero (not removed, not free).** The control vector is always `NU = 12`
(`centroidal_nmpc.py:124` — `    NU = 12    # [f1(3), τ1(3), f2(3), τ2(3)]`); the inactive contact's 6 entries are
clamped by equal lower/upper bounds `u_min = u_max = 0`:
- `centroidal_nmpc.py:634` — `        Inactive contacts are zeroed: u_min = u_max = 0.`
- `centroidal_nmpc.py:638` — `        u_min = np.zeros(self.NU)`
- `centroidal_nmpc.py:639` — `        u_max = np.zeros(self.NU)`
- `centroidal_nmpc.py:642` — `        if contact_config.active_contacts[0]:` (only an *active* contact's block is opened to ±f_max/±τ_max; `:648` the same for contact B)

So the released arm's `λ` stays in the decision vector but is pinned to **zero via bounds**.

**b) WQP — identical mechanism.** The `λ` block is retained (size `6·nc_max = 12`); the inactive contact's
6-block is bounded to `[0, 0]`:
- `wholebody_qp.py:620` — `        # Contact wrench bounds (zero for inactive contacts)`
- `wholebody_qp.py:623` — `            if contact_config.active_contacts[j]:` (active → ±f_max / ±τ_contact_max)
- `wholebody_qp.py:628` — `            else:`
- `wholebody_qp.py:629` — `                lb[s: s + 6] = 0.0`
- `wholebody_qp.py:630` — `                ub[s: s + 6] = 0.0`

⇒ The paper's "the released arm's wrench is held at zero in SS" is **correct**, and the exact mechanism is
box bounds `= 0` (both stages), not structural removal.

---

## Q5 — THE FULL WQP COST STACK (SS canonical, code order)

`weight_ratio = 1` (`wholebody_qp.py:75`) ⇒ all priorities collapse to a **flat weighted sum**; the `priority=`
integers below are inert labels. Every weighted least-squares term actually assembled in the SS (`_two_task`,
non-settle) path:

| # | term | form | weight | value | file:line |
|---|---|---|---|---|---|
| 1 | momentum (T-MOM linear) | CoM-Jacobian rows → `m·a_com_des` | `ss_alpha_mom` | **400** | `wholebody_qp.py:681` |
| 2 | torso-pose (6-D) | `‖[p_ref−p; log3(RᵀR_ref)]‖²` on `J_torso` | `alpha_torso_pose` | **2000** | `wholebody_qp.py:698-699` |
| 3 | swing-EE (6-D) | `‖[p_ref−p; log3]‖²` on `J_ee` | `alpha_ee` | **1000** | `wholebody_qp.py:716` |
| 4 | posture | `‖q̈ − (Kp(q₀−q) − Kd·q̇)‖²` | `alpha_posture` | **20** | `wholebody_qp.py:720-722` |
| 5 | wrench-track | `‖λ − λ_ref‖²` (`b_wrench = lambda_ref`, `wholebody_qp.py:1144`) | `alpha_wrench` | **1** | `wholebody_qp.py:1189` |
| 6 | torque-min | `‖τ_q‖²` | `alpha_torque` | **5** | `wholebody_qp.py:1270` |
| 7 | accel-reg | `‖[q̈_t; q̈]‖²` | `alpha_reg` | **1** | `wholebody_qp.py:1278` |
| 8 | **hw-slack** | `‖s_up‖² + ‖s_lo‖²` (momentum-box slacks) | `w_hw_slack` | **800** | `wholebody_qp.py:1290` |
| + | Tikhonov | `+ ε·I` on the Hessian | `regularization` | **1e-6** | `hierarchical_qp.py:98`, `:275` |

Verbatim anchors for the two the paper does not fully pin down:
- `wholebody_qp.py:1290` — `            qp.add_task(A_slack, b_slack, cfg.w_hw_slack, priority=1)`
- `wholebody_qp.py:181` — `    w_hw_slack: float = 8e2       # Quadratic penalty on hw slack (CANONICAL-2p5 / Add-5 freeze; was 1e4)`

> **DIVERGENCE (flag).** The paper lists **seven** weighted terms (momentum 400, torso 2000, swing-EE 1000,
> posture 20, wrench-reg 1, torque-min 5, accel-reg 1) + Tikhonov. The code assembles **eight**: it also carries
> the **hw-slack penalty `w_hw_slack = 800`** (`wholebody_qp.py:1290`), the largest weight after torso. It is a
> genuine quadratic cost on the momentum-safety-box slack variables `s_up, s_lo` (their bounds open to `[0, ∞)`
> only when the `h_w` box is violated, `wholebody_qp.py:634-643`; otherwise pinned to 0 and inert). The paper's
> §VI-D cost should either add this eighth term or state explicitly that it is omitted because the slacks are
> zero on the canonical run (`h_w` never breaches the box — CANONICAL-2p5 shows peak `‖h_w‖ = 4.24 < 5`).

**Terms present in the source but NOT active on the SS canonical (so correctly absent from the paper's cost):**
- net-force `Σf` penalty — `settle_mode`-only, `wholebody_qp.py:1198-1204`.
- reaction null-space — `alpha_reaction = 0.0` (`wholebody_qp.py:84`), disabled.
- internal-stress `λ` reg — `alpha_lambda_int = 0.0`, DS-only.
- soft-CoM residual — `alpha_com_soft = 0.0`.

---

## Summary of divergences to fix in the paper

1. **Swing orientation (Q2d):** not "held constant at the anchor orientation" — it is a **delayed-cosine SLERP**
   from the release orientation to a **fixed structure-frame identity** (`_R_end = eye(3)`, never read from the
   anchor; coincides with the axis-aligned anchors numerically).
2. **Cost term count (Q5):** the stack has **eight** weighted terms, not seven — the **hw-slack penalty (800)**
   is missing from the paper's list (inert on the canonical because the `h_w` box is never breached).

Everything else the paper (as described in the brief) states is **confirmed** by the code: swing-EE is 6-D
(Q1), on frame `tool_a/b`, weight 1000, dropped in DS; position is a full-target quintic with a real clearance
arc (Q2b/c); torque-min is `‖τ_q‖²` with `α = 5` (Q3); the inactive contact is zeroed via box bounds in both the
NMPC and the WQP (Q4).

*Read-only audit — no code, config, or model file modified; no simulation run.*
