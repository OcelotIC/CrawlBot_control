# INTERNAL — AOCS feed-forward audit: how `_step` builds the AOCS, and what inter-step re-activation needs

**Read-only map on `ae0673e`, branch `j2/ds-active-rework`. No `crawlbot/` change, no implementation, no
sim run.** Reproducer `Misc/scripts/audit_aocs_ff.py` (22/22). Raw map; the inter-step re-activation brief
follows. Companion to the AOCS-during-DS audit (which found `_run_ds_passivity_loop` hardcodes the wheels to
0.0).

---

## DECISIVE OUTPUT (the four questions)

1. **FF quantity:** the AOCS feed-forward is **EXACT Ḣ_s everywhere — no proxy in the AOCS.** SS sources it
   by FD on state (`-(L̇_com + r_com×m·v̇_com)`, orbital IN); DWELL/terminal DS sources the **same** exact,
   origin-referenced Ḣ_s from the QP wrench as `-Σ(r_Ci×f_i + τ_i)`. The proxy/exact split the C5/FLAG-2
   finding refers to is a **different** mechanism — the QP envelope *box* (`qp_envelope_exact`, default
   proxy) — not the AOCS command.
2. **Command structure:** `τ_w = ff_term + K_hw·hw_error + (K_θ·θ_s + K_ω·ω_s + K_d·ω̇_s)`, clipped to
   ±5 Nm. FF (above) + desaturation + attitude-PID. Canonical mode `legacy_pid_numerical`.
3. **Inter-step re-activation:** **YES — the same `compute_aocs_command_legacy_pid_numerical` call is makeable
   inside `_run_ds_passivity_loop` from in-loop values; no input lacks a valid analogue.** Use the **DS wrench
   FF** from `lambda_qp_sol` (matches DWELL DS; FD-FF would be wrong — blind to the welded couple). Only one
   loop-local is new: `ω_s_prev`.
4. **Difficulties:** causality is already correct (QP solves before the wheel write); AOCS↔passivity coupling
   is benign (different actuators, no within-tick conflict, cross-tick trajectory change only); **h_w
   re-evolves during the previously-frozen settles ⇒ C5 must be re-gated.**

---

## Q1 — the feed-forward quantity: EXACT, not proxy (decisive)

The canonical `legacy_pid_numerical` (`force_estimator.py:514-595`) has **two FF sources**, selected by
whether the caller passes `tau_struct_ff`:

**SS (`tau_struct_ff is None`, `force_estimator.py:585-588`):** FD on the robot's centroidal momentum —
```python
L_dot_est = (L_com - L_com_prev) / dt
dv_com_est = (v_com - v_com_prev) / dt
orbital    = np.cross(r_com, robot_mass * dv_com_est)   # the M4 orbital term
ff_term    = -L_dot_est - orbital            # = -(L̇_com + r_com×m·v̇_com) = -Ḣ_s   ← EXACT (orbital IN)
```
This is the **exact origin-referenced Ḣ_s**, not the proxy `L̇_com`. The orbital term `r_com×m·v̇_com` is the
M4 correction (`force_estimator.py:369`, `legacy_corrected` docstring §5.8).

**DWELL/terminal DS (`tau_struct_ff` provided, `force_estimator.py:589-592`):** direct wrench couple, built
in `_step` (`sim_loop.py:3126-3136`) from the **QP wrench** `lambda_qp_sol` and the anchor levers:
```python
_lam = lambda_qp_sol                          # the QP contact wrench (world frame)  — sim_loop.py:3128
_ff -= np.cross(_r_C[_ci], _f) + _tq          # τ_w_FF = -Σ_i (r_Ci × f_i + τ_i)      — sim_loop.py:3135
ff_term = tau_struct_ff                       #                                        — force_estimator.py:592
```
The reason DS switches source (docstring `force_estimator.py:582-584`): *in DS the welded loop carries
internal stress that contributes a couple (r_CA−r_CB)×f on the structure **invisible to L_com**.* So the
FD-on-L_com FF would be wrong in DS; the wrench couple captures the welded-loop stress the FD misses.

**Both forms target the SAME exact origin-referenced Ḣ_s** — SS via FD on state, DS via the wrench. The DS
wrench FF uses **absolute levers `r_Ci`** (struct-frame anchor positions, from O_s). Numeric identity
(reproducer): with absolute levers vs robot-CoM levers,
`Σ r_Ci×f_i − Σ(r_Ci−r_com)×f_i = r_com×Σf_i` (residual **6.4e-16**) — i.e. the absolute-lever moment is the
exact one, and a robot-CoM-lever proxy would differ by exactly the orbital/transport term `r_com×Σf`. **The
DS wrench FF is on the EXACT side of the same axis as FLAG-2's exact box.**

**Decisive sub-answer to the C5 connection:** *SS does **not** use the proxy.* The AOCS FF is exact in both
SS and DS. The proxy that under-estimates wheel load by 0.33 N·m (C5/FLAG-2) lives in a **separate**
mechanism — the QP momentum-rate **box** `|M_λ·λ| ≤ τ_w_max` inside `wholebody_qp`, gated by
`config.qp_envelope_exact` (**default False = proxy**, `config.py:319`). There is therefore a pre-existing
exact-vs-proxy split in the system, but it is **between** the AOCS FF (exact) and the QP box (proxy default),
**not within the AOCS.** Re-activating the inter-step AOCS via the exact wrench FF introduces **no new**
proxy/exact split and is consistent with both SS and DWELL DS. The QP-box reconciliation (FLAG 2) is
orthogonal and stays where it is.

## Q2 — the full AOCS command structure

`compute_aocs_command_legacy_pid_numerical` (`force_estimator.py:514-595`):

```
τ_w = ff_term  +  K_hw·hw_error  +  (K_θ·θ_s + K_ω·ω_s + K_d·ω̇_s)        clipped to ±tau_w_max
      └ FF ┘     └ desaturation ┘   └──────── attitude PID ────────┘
```

| part | term | what it regulates | source / sign |
|---|---|---|---|
| **feed-forward** | `ff_term` (Q1) | cancels the robot's Ḣ_s disturbance on the structure | exact Ḣ_s (FD in SS / wrench in DS) |
| **desaturation** | `+K_hw·hw_error`, `hw_error = clip(h_w, ±5) − h_w` | pulls wheel momentum back into ±hw_max | `force_estimator.py:372`, `K_hw=2.0` |
| **attitude P** | `+K_θ·θ_s` | drives structure back to its initial orientation (θ_s=0) | `K_θ=1.0`; θ_s = vee(R_init.T·R_now)ₐ (`sim_loop.py:3225-3229`) |
| **rate D** | `+K_ω·ω_s` | damps structure angular velocity | `K_ω=50.0` |
| **accel D** | `+K_d·ω̇_s`, `ω̇_s=(ω_s−ω_s_prev)/dt` | extra damping (numerical ω̇) | `K_d=25.0` |
| **saturation** | `np.clip(τ_w, ±tau_w_max)` | wheel-torque limit | `aocs_tau_w_max = 5.0 Nm` (`force_estimator.py:376`) |

All signs are `+` on the feedback terms (Newton-Euler about the structure CoM: `I_s·ω̇_s = -Ḣ_s − τ_w`, so a
braking torque must add positive — see `compute_aocs_command` docstring `force_estimator.py:256-259`).

**Modes available (6 command fns):** `compute_aocs_command` (H_est — full-H estimator FF, `-K_h` desat,
`-K_ω` damping — note the H_est sign convention differs and is untested under non-zero ω_s),
`legacy_corrected` (FF+desat only), `legacy_pd_{numerical,model}` (add PD on ω_s), `legacy_pid_{numerical,
model}` (add the θ_s attitude P). **Canonical = `legacy_pid_numerical`** (the `_numerical`/`_model` split is
only how ω̇_s is sourced — FD vs Newton-Euler model). The `else` legacy branch (`sim_loop.py:3264-3271`) is
the only place a bare proxy `L_dot` FF (no orbital) appears, and it is **not** the canonical mode.

## Q3 — what the inter-step loop has vs needs

`_run_ds_passivity_loop` (`sim_loop.py:601-793`) runs at dt_qp, recomputes `rs` each iteration, and solves
the same `qp_ss`. For the canonical (numerical, wrench-FF) AOCS:

| AOCS input | in the loop? | where |
|---|---|---|
| `lambda_qp_sol` (wrench FF force input) | **PRESENT** — captured each iter | `sim_loop.py:735` |
| `rs.L_com / v_com / r_com` | **PRESENT** — `rs = robot.update(...)` each iter | `sim_loop.py:710` |
| anchor levers `r_Ci`, contact mask | **PRESENT** — `cc_ds.r_contact_A/B`, `cc_ds.active_contacts` | `contact_phase.py:36-38`; `cc_ds` = arg |
| `ω_s` (qvel[3:6]), `h_w` (qvel[6:9]·I_w) | **PRESENT** — read MuJoCo state | trivial recompute |
| `θ_s` (needs R_init, I_s) | **PRESENT** — `self._struct_quat_init`, `self._struct_I` (instance attrs) | `sim_loop.py:222, :217` |
| gains, tau_w_max | **PRESENT** — `cfg.*` | config |
| **`ω_s_prev`** (for `K_d·ω̇_s`) | **NEEDS LOCAL** — `_omega_s_last` is local to `_step` (`sim_loop.py:2625`), invisible here | track loop-local (init from entry qvel, update each iter) |

The FD-FF history `_L_com_qp_prev / _v_com_qp_prev` are **not needed** — they feed only the
`tau_struct_ff is None` (FD-FF) branch, and the loop should use the **wrench FF** (it is welded DS, so the
FD-FF would be blind to the loop's own internal-stress couple — same reason DWELL DS uses the wrench FF).

**Answer:** the **same** `compute_aocs_command_legacy_pid_numerical(...)` call can be made inside the loop,
fed from in-loop values, with **only `ω_s_prev` added as a loop-local**. **No input lacks a valid inter-step
analogue.** The fix replaces `ctrl[n_j:n_j+3] = 0.0` (`sim_loop.py:765`) with this call. (Minor bookkeeping
note, pre-existing and orthogonal: the `_step` FD-history locals are re-initialized at the top of each
`_step`, so they are not continuous across the loop boundary regardless — not affected by this change.)

## Q4 — anticipated difficulties

1. **Causality — already correct.** Order in the loop: `qp.solve` (→`lambda_qp_sol`) at `sim_loop.py:735` →
   wheel write at `:765` → `mj_step` at `:766`. The wrench the FF needs is computed **before** the wheel
   command — exactly the ordering `_step` already uses. The AOCS slots into the spot the `= 0.0` zeroing
   currently occupies.
2. **AOCS ↔ passivity interaction — benign.** They act on **different actuators**: the passivity inequality
   `dqⱼᵀτ_q + 2α·T_kin ≤ W_budget` constrains the **joint** torques τ_q; the AOCS drives the **wheels**,
   which are **not a decision variable** of the settle QP. So **within a tick there is no conflict** — the
   QP is solved before τ_w is known, and τ_w cannot change that tick's QP solution. **Across ticks**, τ_w
   integrates into the structure's motion, so the next tick's state (and the wrench the next QP sees) differ
   ⇒ the settle **trajectory** changes with the AOCS on. This is benign for the passivity *guarantee* (the
   inequality is re-imposed on the current state every tick — joint-KE dissipation is still enforced), but
   it is **not a no-op**: the realized λ/h_w/dock impact along the settle will differ and must be
   re-characterized.
3. **h_w accumulation — interacts with C5.** With the AOCS off, the wheels were frozen for **383 ticks /
   3.83 s** (≈46 % of DS time) across 5 settles, so h_w did **not** evolve there. Re-activating drives the
   wheels during those windows ⇒ h_w now evolves in the settles ⇒ the whole-traversal h_w trajectory
   changes. C5 (h_w∞ ≤ 4.5 budget) is **already near the cap** (~4.18 with the proxy box, ~4.88–4.93 with
   the exact box). The desaturation term `+K_hw·hw_error` actively pulls h_w toward the box, but the FF +
   attitude terms load it; **net direction is empirical — C5 must be re-gated after re-activation.**

---

## Flags / divergences vs prior audit facts

1. **AOCS-during-DS audit (companion):** confirms its feasibility claim and **sharpens it** — the AOCS is not
   merely "self-contained," its DS feed-forward input is specifically `lambda_qp_sol` (the QP wrench), which
   the inter-step loop already captures at `sim_loop.py:735`. So re-activation is the wrench-FF call from
   in-loop values; only `ω_s_prev` is new.
2. **FLAG-2 / C5 (the load-bearing reconciliation):** the C5/FLAG-2 proxy is the **QP envelope box**
   (`qp_envelope_exact`, default proxy) — **not** the AOCS FF, which is exact (FD-with-orbital in SS, absolute
   levers in DS). So **there is no exact-in-DS / proxy-in-SS mismatch to fix inside the AOCS** (the premise's
   alternative is refuted): re-activating the inter-step FF via the exact wrench couple is consistent with
   both SS and DWELL DS. The QP-box exact-vs-proxy decision (FLAG 2) is independent and unaffected.
3. **Piste A:** the inter-step loop is exactly where the passivity inequality runs (`passivity_active=True`,
   `sim_loop.py:756`); re-activating the AOCS there is the wheel-side complement to Piste A's joint-side work
   budget — they constrain different actuators and do not compete within a tick (Q4.2).

## Reproduce
```
MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/audit_aocs_ff.py
```
Code anchors: `force_estimator.py:514-595` (legacy_pid_numerical), `:585-592` (the two FF branches), `:369`
(orbital), `:376` (clip); `sim_loop.py:3126-3136` (DS wrench FF build), `:3225-3229` (θ_s), `:3275` (`_step`
applies τ_w), `:735/:765/:766` (loop causality), `:2625` (`_omega_s_last` local), `:217/:222` (struct attrs);
`config.py:319` (`qp_envelope_exact` proxy default), `:127` (`aocs_off_in_ds`); `diag_cooperative_arms.py:307`
(`aocs_use_wrench_ff_in_ds=True`). No `crawlbot/` change ⇒ no regression run needed.

**STOP after the report.** Map only — no design, no implementation. No merge, no PR.
