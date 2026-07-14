# Phase TORSO-REF-AUDIT — the torso task is full 6-DoF (x tracked); mid-swing lag is REAL; the DS backward reset is a LOGGING artifact

**Branch** `j2/ds-active-rework` · static code audit + inspection of the frozen `32aefaf` run's committed CSV.
**NO code change, NO new run.** Push-only, never merge.

## Answers (A)–(E)

| Q | Answer | Key file:line |
|---|---|---|
| **(A)** does the task track x-position? | **YES — full 6-DoF** (position xyz + orientation), option (a) | `wholebody_qp.py:689-699` |
| **(B)** continuous or per-step reference? | **PER-STEP quintic, re-anchored each step** (cleared + re-added at every SS start) | `sim_loop.py:1685-1757` |
| **(C)** logged ref true or artifact? | **SS rows: TRUE** (logs the QP-used value). **DS rows: ARTIFACT** (planner hold = p_t0, never consumed by any QP) | `sim_loop.py:3527-3531` vs `:950-951` |
| **(D)** torso task active in DS? | **NO** — interstep DS runs `settle_mode=True`; every torso task is gated off | `sim_loop.py:787`, `wholebody_qp.py:678,737` |
| **(E)** dynamically naive reference? | **YES** — geometric quintic p_t0→p_t1, no momentum/recoil term; dynamics-awareness lives in the NMPC side-channels, not in the torso reference | `torso_planner.py:195+, 547+`, docstring `:4-15` |

---

## 1. WHAT the torso-pose task constrains — full 6-DoF, x included

Two-task SS stack (`_two_task`, `wholebody_qp.py:678-699`):
- Error: `e6 = concat([p_torso_ref − p_t,  log3(R_t.T @ R_rt)])` — **3 position components (x,y,z) + 3 orientation** (`:689`).
- Desired accel: `a_t_des = a_ft + Kp_t@e6 + Kd_t@(v_rt − v_t_act)` (`:692`); gains are **uniform across all 6 dims** (`sim_loop.py:1155-1156`: `Kp_torso=[kpt]*6`, kpt = `ss_Kp_torso` = 3.0).
- Rows: `A_tp` = **all 6 rows of J_torso** (`:695-697`), weight `alpha_torso_pose` (frozen 2000), `qp.add_task(..., priority=2)` (`:698-699`).

**⇒ x-position IS a task target at the top weight.** The mid-swing x divergence is a genuine tracking residual of an active task — not "x unconstrained". (The diagnostic SS-entry-freeze branch `mapping_bypass_in_ss` exists at `sim_loop.py:2857-2864` but its default is **False**, `config.py:588` — inactive in the frozen run.)

## 2. WHERE the reference comes from — per-step TorsoPlanner quintic, raw (no mapping) in two-task SS

Producer: `TorsoPlanner` (`crawlbot/planning/torso_planner.py:55`), configured per step by `_setup_torso_for_step` (`sim_loop.py:1392`, called at the DS→SS transition, `:2186`):
1. `clear_phases()` (`:1685`) — the previous step's segment is discarded;
2. `set_hold(p_t0, R_flat, r_com0)` (`:1688`) — the **hold pose = the step's START pose p_t0**;
3. `add_phase(t_ss_start, t_ss_start+T_step, p_t0, R_flat, p_t1, R_t1, delta_com_*, …)` (`:1754-1757`) — one quintic segment from the current pose to the dock-IK target; orientation reference is the **constant global R_flat** (`:1749-1753`, zero DS↔SS seam).

**⇒ the reference is a PER-STEP segment, re-anchored at the live torso pose each step** — continuous across steps only to the extent tracking converged (p_t0(k+1) = actual pose ≈ p_t1(k) − residual).

Consumption in the frozen two-task SS: the mapping branch is **explicitly excluded** — condition `not (cfg.ss_two_task_mode and phase == 'SS')` (`sim_loop.py:2865-2867`, comment `:2868-2870`: "in SS the torso-pose task is fed the RAW TorsoPlanner quintic+SLERP … NO CoMToTorsoMapping δ"). So `p_torso_ref_used = tr.p` (the else-branch, `:2976-2978`), with the planner query capped at `ss_end − 1e-3` (`:2852-2853`) so the post-T_step margin window returns the quintic's **terminal** pose instead of falling through to the hold (see §4 — the cap is a workaround for exactly the hold-at-p_t0 behavior that pollutes the DS log rows).

## 3. Logged p_torso_ref: TRUE in SS, ARTIFACT in DS

- **SS rows** (`_step` logger): `p_torso_ref_log = p_torso_ref_used` (`sim_loop.py:3527-3531`) — the **exact quantity the QP tracked at the last sub-step**. The comment at `:3519-3526` documents that this was deliberately fixed to log the QP-used reference rather than the geometric planner value. In the frozen two-task config the two coincide anyway (§2). **⇒ SS logged reference = the controller's true reference.**
- **DS rows** (`_log_ds_tick`): `tref = torso_planner.reference_at(t_abs)` (`:950-951`) — a **logging-only** planner query (the method's docstring `:934-940` guarantees it "does not call qp.solve … side-effect-free w.r.t. control"). During interstep DS, `t_abs` is past the (only) phase's `t_end`, so `reference_at` falls through (`torso_planner.py:447-448`) to `_hold_reference()` (`:532-545`) which returns `_hold_p` = **p_t0 of the just-finished step** (`sim_loop.py:1688`). **⇒ the DS-row p_torso_ref is one full step BEHIND — and no QP ever consumes it.**

### Quantified from the frozen run (`c25_fulldiag.csv`, committed)

| step | x-err @ SS end | max mid-swing x-err | DS-entry ref_x jump | SS ref x-span |
|---|---|---|---|---|
| 0 | 97.3 mm | 127.0 mm | −113.0 mm | +113 mm |
| 1 | 26.8 mm | 51.0 mm | **−589.5 mm** | +590 mm |
| 2 | 21.7 mm | **156.9 mm** | −157.1 mm | +157 mm |
| 3 | 22.1 mm | 42.4 mm | **−617.3 mm** | +617 mm |
| 4 | 17.3 mm | 148.5 mm | −151.4 mm | +151 mm |
| 5 | 20.8 mm | 42.3 mm | **−604.1 mm** | +605 mm |

The DS-entry jump equals **−(that step's SS reference span) to the millimeter** in every step, and the first-DS logged ref_x (0.8288, step 2) equals the step's SS-**start** reference (0.8289): the logged DS value **is** p_t0 — the `set_hold`/`_hold_reference` fall-through, verbatim. The "~616 mm spurious errors" are the arm-b steps' full torso stride (590–617 mm). **Pure export/logging artifact.**

## 4. DS: no torso task is active

The interstep DS solve passes `settle_mode=True` (`sim_loop.py:787`) and `p_torso_ref = rs.oMf_torso.translation` — the **current pose as its own reference** (`:783-786`, zero error placeholder). In the QP:
- two-task torso-pose: `_two_task = cfg.ss_two_task_mode and not settle_mode` (`wholebody_qp.py:678`) → **OFF**;
- legacy torso 6D: `torso_task_active` requires `not settle_mode` (`:737`) → **OFF**;
- CoM task (`:725`) and EE task (`:895`, memo ref `:785`) also settle-gated → **OFF**;
- `ds_centroidal_mode` default False (`wholebody_qp.py:99`) → the settle stack is joint-velocity damping + passivity + wrench/torque/reg only.

**⇒ during DS the torso is not tracked at all; the backward "reference" exists only in the log rows (§3).**

## 5. Dynamic consistency — the reference is geometrically naive by design

The position reference is a pure **geometric quintic** p_t0 → p_t1 with half-cosine/trapezoidal timing (`torso_planner.py:195+` add_phase, `:547+` `_trapezoidal_params`; optional mid-waypoint and FK-on-smoothed-q modes). **No term anywhere in the reference models the free-floating momentum coupling** (the recoil). The dynamics-awareness in the system lives in side-channels:
- the **CoM reference derived from the torso quintic** (`r_com(t) = p_torso(t) + R·δ_com(s)`, docstring `:8-15`, arm-reconfiguration via interpolated δ_com) feeds the NMPC, which then plans momentum-feasible wrenches under the Ḣ_s envelope;
- the **L_com_ref feedforward** for the NMPC momentum cost (`:475-528`, torso-only approximation, ~20% limb error absorbed by NMPC feedback).

But the torso-task reference itself assumes the base can advance monotonically along the quintic. Under a hard momentum budget (the frozen ±2.5, where the envelope binds on every step — CANONICAL-2p5 STEP 4), the free-floating base must recoil against the swing; the realized x therefore lags/oscillates up to **~157 mm mid-swing and reconverges to ~17–27 mm at the boundary**, where the quintic terminates at the reachable dock-IK pose. The planner's own docstring states the geometric intent (`:4-6`: "the torso advances during swing using the stance arm as an inverted manipulator"). **⇒ (b) mid-swing divergence is REAL closed-loop tracking behavior — the unmodeled recoil — not an artifact;** what matters for docking is the boundary residual (DOCK-CAUSE), which stays ~20 mm.

## Bottom line for the three observations

- **(a) step-boundary match ~20 mm** — real: the 6-DoF task (weight 2000) reconverges the torso onto the quintic's terminal (dock-IK) pose; 17–27 mm boundary residual (97 mm on the initial step).
- **(b) ~150 mm mid-swing divergence** — real tracking excursion: geometric reference vs free-floating recoil under the bound momentum envelope. Not a bug; a modeling choice (naive reference + feedback absorbs the difference).
- **(c) backward p_torso_ref reset at SS→DS (~590–617 mm)** — **logging artifact**: `_log_ds_tick` queries the planner outside its phase and gets the `set_hold(p_t0)` fall-through; the DS controller neither uses that value nor tracks the torso at all (settle mode). Candidate cosmetic fix (NOT applied — out of scope): have `_log_ds_tick` log the phase's terminal pose (or NaN) instead of the hold, or update `set_hold` to hold p_t1 after phase completion; the SS-side already works around this exact behavior via the `ss_end − 1e-3` query cap (`sim_loop.py:2847-2853`).

`crawlbot/` untouched (audit only). **STOP for cross-check.**
