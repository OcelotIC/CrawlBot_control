# Phase DOCK-CAUSE — dock error is a WBC EE-tracking residual, not a reference-gap

**Branch** `j2/ds-active-rework` · read + measure only, NO code change · pushed, never merged.
Data: `results/j2_adjconv/dockcause_decomp.json`; script `Misc/scripts/diag_dockcause_decompose.py`
(READ-ONLY on the committed per-step baselines `results/figC_sw_s{k}_x1/sim_log.json` — the ×1 sweep
runs from TSTEP-DIAG-ALL, each truncated to step k terminal, so step k's dock == the full-run dock).

Context (from TSTEP-DIAG-ALL): dock error is a tracking residual, decoupled from the Ḣ_s cap
(step 5 worst dock yet Ḣ_s headroom; corr(dock, standoff) = −0.11). The swing EE lives in the WBC QP
(full kinematics), not the centroidal NMPC. This phase localises the residual **inside the WBC**.

---

## 1. WBC QP task structure (canonical SS docking path = `ss_two_task_mode`)

The canonical run (`--ss-two-task`) takes the `_two_task` branch of `WholeBodyQP.solve`
(`crawlbot/solvers/wholebody_qp.py:678-722`). **All tracking tasks are weighted least-squares at the
SAME priority 2, with NO null-space projection and NO terminal tightening** (the `weight_ratio=1`
comment at :676-677: "the α's set the hierarchy directly"). SS weights/gains are built at
`sim_loop.py:436-442` (`qp_ss = _build_qp(... cfg.ss_alpha_ee, ... cfg.ss_Kp_ee ...)`), sourced from
`config.py` and the sweep CANON overrides (`--ss-alpha-mom 5000 --alpha-torso-pose 24000 --ss-kp-torso 3`):

| Task | Jacobian | weight α (canonical) | Kp | prio | null-space | file:line |
|---|---|---|---|---|---|---|
| Momentum (linear CMM) | J_com | **5 000** (`ss_alpha_mom`) | 3 | 2 | none | wholebody_qp.py:681 |
| Torso-pose 6-D | J_torso | **24 000** (`alpha_torso_pose`) | 3 | 2 | none | wholebody_qp.py:698 |
| **Swing-EE 6-D** | J_ee | **3 000** (`ss_alpha_ee`) | 10 lin / 6 ang | 2 | none | wholebody_qp.py:716 |
| Posture | I(nq) | 20 (`ss_alpha_posture`) | 1 | 3 | none | wholebody_qp.py:720 |
| Wrench-track | I(λ) | 0.01 | — | 4 | — | wholebody_qp.py:1189 |
| Torque-min / accel-reg | I | 1 / 0.01 | — | 5 / 6 | — | wholebody_qp.py:1270,1278 |

**An EE task IS present** and it is **soft** (weighted, not a hard/strict priority-1 task). It is
**outweighed 8:1 by torso-pose (24 000) and 1.67:1 by momentum (5 000) at the same priority 2**, and its
own linear stiffness is low (`Kp_ee=10`). There is **no terminal tightening**: `α_ee` and `Kp_ee` are
constant across every swing tick — no ramp of EE weight/gain on the dock approach. (Under
`ss_two_task_mode` the whole M2 null-space machinery at :866-951 is bypassed — `not _two_task` gates it
off — so the EE task gets no geometric protection either.)

---

## 2. Swing-EE reference terminal vs anchor — reference reaches the anchor

`SwingPlanner.reference_at` (`swing_planner.py:558`): `p_ee = p_start + Δp·s(τ) + clearance·n̂·bump(τ)`,
with `Δp = p_end − p_start` and `p_end = scheduler.anchors_{a,b}[swing_to_idx]` (the dock anchor, struct
frame). At τ=1: `s(1)=1`, `bump(1)=0` ⇒ **`p_ee(τ=1) = p_end = anchor` exactly**. The SS convergence hold
pins τ at 1 (`_swing_query_time` clamp, `sim_loop.py:2599-2600`), so at the dock instant the reference
sits on the anchor. **Measured planned-ref → anchor gap at the dock instant (per step):**

| step (swing@anch) | 0 (b@3) | 1 (a@3) | 2 (b@4) | 3 (a@4) | 4 (b@5) | 5 (a@5) |
|---|---|---|---|---|---|---|
| ref-gap [mm] | 0.0039 | 0.0004 | 0.0036 | 0.0005 | **4.582** | 0.0008 |

Five of six steps: ref-gap **< 4 µm** — the planned reference is on the anchor. **Cause A (reference ends
short) is REJECTED** — the planned terminal equals the anchor by construction (`swing_planner.py:558`),
not ~5 mm short.

Step 4's 4.58 mm is NOT a short terminal — it is a **sampling artifact**: the dock GATE takes the *min*
`d_grip_swing` over the step, and step 4's min lands **mid-approach** (at 82 % of SS), where the reference
has not yet reached the anchor. At step 4's *last* SS tick (converged hold) the ref-gap collapses to
**0.0032 mm** — same as the others (see §4).

---

## 3. Dock-error decomposition — 100 % WBC residual

Exact vector identity (holds to machine precision, `ident` < 0.01 µm all steps):
`(p_ee − anchor) = (p_ee − p_ee_ref) + (p_ee_ref − anchor)`, i.e. **dock = WBC-residual + ref-gap**.
`p_ee`=realised EE, `p_ee_ref`=swing ref, both logged in the Pinocchio world = structure frame; anchors
static in that frame. Cross-check: `dock_pin = ‖p_ee − anchor‖` equals the physical gate metric
`d_grip_swing` to **0.0000 mm** every step ⇒ the Pinocchio tool frame IS the MuJoCo gripper site (no
hidden tool/gripper offset), so the decomposition is exact and gate-faithful.

| step | dock (gate) [mm] | WBC residual [mm] | ref-gap [mm] | dominant term |
|---|---|---|---|---|
| 0 (b@3) | 4.940 | **4.939** | 0.0039 | WBC residual (100 %) |
| 1 (a@3) | 4.405 | **4.405** | 0.0004 | WBC residual (100 %) |
| 2 (b@4) | 4.904 | **4.902** | 0.0036 | WBC residual (100 %) |
| 3 (a@4) | 4.436 | **4.436** | 0.0005 | WBC residual (100 %) |
| 4 (b@5) | 4.045\* | 4.724 | 4.582 | WBC residual (vectors partially cancel at the transient) |
| 5 (a@5) | 5.000 | **5.000** | 0.0008 | WBC residual (100 %) |

\*transient min — see §4. Consistency: recomputed WBC residual = logged `e_ee_pos` (`sim_loop.py:3556`)
to < 6 µm on steps 0-3,5 (they use the same tick); step 4 differs because `e_ee_pos` samples a different
sub-time and the EE is moving fast there.

**Verdict: on all five converged steps the dock error is 100 % WBC EE-tracking residual and 0 %
reference gap.** The QP leaves the swing EE **4.4–5.0 mm short of its own reference**, which is sitting on
the anchor.

---

## 4. Step-4 anomaly resolved — its "best dock" is a transient, not a better closure

`dock-min tick vs LAST-SS tick` (ref-gap ≈ 0 ⇒ converged terminal hold):

| step | dock-min at (frac of SS) | d@min [mm] | ref-gap@min [mm] | d@last-SS [mm] | ref-gap@last [mm] | verdict |
|---|---|---|---|---|---|---|
| 0 | 1.00 | 4.940 | 0.004 | 4.940 | 0.004 | terminal hold |
| 1 | 1.00 | 4.405 | 0.000 | 4.405 | 0.000 | terminal hold |
| 2 | 1.00 | 4.904 | 0.004 | 4.904 | 0.004 | terminal hold |
| 3 | 1.00 | 4.436 | 0.001 | 4.436 | 0.001 | terminal hold |
| **4** | **0.82** | **4.045** | **4.582** | **4.758** | 0.003 | **mid-approach transient** |
| 5 | 1.00 | 5.000 | 0.001 | 5.000 | 0.001 | terminal hold |

Step 4's gate value (4.045) is the min of a transient dip as the EE arcs through the anchor
neighbourhood before the hold settles; its **converged** residual is **4.758 mm** — squarely in the
swing-B cluster (steps 0/2: 4.94/4.90). This **explains the TSTEP-DIAG-ALL anomaly** (step 4 "best docker"
that broke the standoff hypothesis): it wasn't a better closure, it was a sampling artifact, which is why
TSTEP saw it "revert into the swing-A/B cluster as T_step grows."

---

## 5. Arm-config (posture) dependence — residual tracks reach, not standoff

True J_ee conditioning needs per-tick `q` (not logged); **reach = ‖p_ee − p_torso‖** (both logged) is the
in-scope arm-extension proxy. Across the 6 steps:

| | s0 | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|
| WBC residual [mm] | 4.939 | 4.405 | 4.902 | 4.436 | 4.724 | 5.000 |
| reach [m] | 0.593 | 1.262 | 0.547 | 1.164 | 0.506 | 0.971 |
| standoff [m] | 0.709 | 0.678 | 1.010 | 1.131 | 1.631 | 1.812 |

- **corr(WBC residual, reach) = −0.684** — longer arm extension → smaller residual (the long-reach 'a'
  steps 1,3 dock best, 4.41/4.44; the folded-short 'b' steps 0,2,4 dock worst, 4.90–4.94).
- corr(WBC residual, standoff) = **+0.356**; corr(dock, standoff) = **−0.112** — **reproduces TSTEP's
  −0.11**. Standoff explains ~1 % of the dock variance.

So the residual is **posture-dependent** (moderate reach coupling, |r|≈0.68) and **standoff-decoupled** —
exactly the −0.11 decorrelation TSTEP flagged: the driver is arm configuration, not standoff. The reach
law is not single-variable clean (step 5, mid-reach 0.97, is the worst docker 5.00), consistent with
TSTEP's per-step idiosyncrasy — but the posture channel dominates the standoff channel 2:1 in correlation.

---

## Interpretation — this is **Cause B** (WBC under-weights / never tightens the EE task)

- **Cause A rejected:** the planned swing-EE reference reaches the anchor (ref-gap < 4 µm on 5/6 steps;
  step 4's gap is a transient-sampling artifact, converging to 3 µm). The fix is NOT in the swing reference.
- **Cause B confirmed:** the reference reaches the anchor, and the WBC leaves the entire 4.4–5.0 mm as a
  tracking residual. Mechanism: the swing-EE task is a **soft P2 weighted-LS task (α=3000)**, outweighed
  **8:1 by torso-pose (24 000)** and 1.67:1 by momentum (5 000) at the same priority, with **no null-space
  protection and no terminal tightening** — so the weighted-LS equilibrium trades ~5 mm of EE closure to
  the higher-weighted torso/momentum tasks. The residual's posture dependence (−0.68 on reach) is the
  fingerprint of a Jacobian-limited weighted-LS compromise, not a momentum-cap or standoff effect.
- **Not cause C** in the strict sense (a terminal EE task *does* exist) — but it shares C's remedy: the EE
  task needs **terminal authority on the dock approach**.

**Candidate levers (NOT applied — for cross-check/decision):** raise `ss_alpha_ee` and/or `Kp_ee` on the
terminal approach (terminal tightening), or restore a strict/null-space EE priority near dock, so the EE
task is not outbid 8:1 by torso-pose in the final mm. These are WBC-side; the swing reference is correct.

---

## Deliverable (STOP-GATE)
1. **QP task list + priorities** (§1): EE task present, **soft** (P2 weighted α=3000), **no terminal
   tightening**, outweighed 8:1 by torso-pose + 1.67:1 by momentum, no null-space. file:line given.
2. **Planned-ref → anchor gap** (§2): < 4 µm on 5/6 steps (ref on anchor); step 4 = transient artifact.
   **Cause A rejected.**
3. **WBC EE residual at dock** (§3): 4.4–5.0 mm every step.
4. **Decomposition** (§3-4): **dock = 100 % WBC residual + 0 % ref-gap** (exact vector identity, gate-faithful).
5. **Arm-config correlation** (§5): corr(residual, reach) = **−0.684** (posture-driven); corr(dock,
   standoff) = **−0.112** (reproduces −0.11). Residual is posture-dependent, standoff-decoupled.

**Verdict: Cause B — the dock error is a WBC swing-EE tracking residual left because the EE task is
under-weighted (8:1 vs torso-pose) and never terminally tightened; the swing reference is correct.**
NO canonical change. Default gain/scale remain off. Raw runs (`figC_sw_*`) gitignored. **STOP for cross-check.**
