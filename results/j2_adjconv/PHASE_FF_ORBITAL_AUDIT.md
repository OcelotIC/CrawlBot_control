# Phase FF-ORBITAL-AUDIT — read-only audit of the AOCS feedforward (§VI-E) + structure inertia (HANDOFF T1)

**Read-only. No code/config change, no run.** Every claim carries `file:line` + verbatim line. Canonical AOCS
mode = `legacy_pid_numerical` (`aocs_mode`, set by the runner); the DS_terminal settle runs through `_step`
with `settle_mode=True`, so its AOCS command is assembled at `sim_loop.py:3380-3423`.

---

## Q1 — THE FEEDFORWARD TERM

The canonical law is `compute_aocs_command_legacy_pid_numerical`. Total command:
- `crawlbot/aocs/force_estimator.py:594` — `    tau_w = ff_term + K_hw * hw_error + pid_term`
- `crawlbot/aocs/force_estimator.py:595` — `    return np.clip(tau_w, -tau_w_max, tau_w_max)`

The feedforward `ff_term` has **two branches** (`force_estimator.py:579-592`):

**SS branch** (`tau_struct_ff is None`) — compensates the **FULL Ḣ (spin + orbital)**, not only L̇_com:
- `force_estimator.py:585` — `        L_dot_est = (L_com - L_com_prev) / dt`
- `force_estimator.py:586` — `        dv_com_est = (v_com - v_com_prev) / dt`
- `force_estimator.py:587` — `        orbital = np.cross(r_com, robot_mass * dv_com_est)`
- `force_estimator.py:588` — `        ff_term = -L_dot_est - orbital`

**DS branch** (`tau_struct_ff` provided — the canonical DS/settle path) — the **exact contact-wrench couple**:
- `force_estimator.py:592` — `        ff_term = tau_struct_ff`
- assembled by the caller from `λ_qp`: `sim_loop.py:3315` — `                        _ff -= np.cross(_r_C[_ci], _f) + _tq`
  (`sim_loop.py:3302-3305` gates this on `aocs_use_wrench_ff_in_ds and phase == 'DS' and aocs_mode in legacy_pid_*`).

**Answer: (a) the full planned/estimated Ḣ_s, NOT (b) only L̇_com.** In SS the FF is `−L̇_com − r_com×m·a_com`
(`:588`) — the orbital term is explicit. In DS it is `−Σ_i (r_Ci×f_i + τ_i)` (`:592`, `sim_loop.py:3315`), the
exact reaction couple. The §VIII dominant term `r_com × Σf_j` equals `r_com × m·a_com` (Newton: net contact
force `Σf_j = m·a_com`), which **is** the SS `orbital` term at `:587`.

> **VERDICT — §VI-E is a TEXT DEFECT, not a controller gap.** The code does **not** use "only L̇_com"; both
> branches carry the orbital/reaction term. The paper should state the FF compensates the full centroidal
> momentum rate about O (SS: `−L̇_com − r_com×m·a_com`; DS: `−Σ(r_Ci×f_i+τ_i)` from the contact wrench).

**Additional finding (beyond Q1-Q5; bears on HANDOFF §2 H0 premise — reported, not interpreted).** The
"bias-to-zero" wheel-momentum term in this canonical law is an **anti-windup on box violation, not a
proportional desaturation toward zero**:
- `force_estimator.py:575` — `    hw_error = np.clip(hw_current, hw_min, hw_max) - hw_current`

With `hw_min/hw_max = ±5` (passed as `cfg.hw_min/hw_max`, `sim_loop.py:3419-3421` region), `hw_error = 0`
whenever `|h_w| ≤ 5`. During the settle `h_w,z = −1.28 ∈ ±5`, so `K_hw·hw_error = 0` there — this term does
**not** actively unload `h_w`. (The HANDOFF §2 attributes the `h_w` unloading to "the bias-to-zero wheel-momentum
feedback K_hw"; the canonical `legacy_pid_numerical` has no such proportional term, only the saturation clamp.
The `H_est`/`legacy_corrected` variants use a different `−K_h·(hw − hw_target)` form, `force_estimator.py:286`,
but they are not the canonical mode.) Flagged for the drift work stream; no judgment on H0 here.

---

## Q2 — SOURCE OF THE COMPENSATED QUANTITY

**Reconstructed from MEASURED state (finite difference) — not taken from the NMPC plan.**

SS branch — finite differences of the robot's measured centroidal state:
- `force_estimator.py:585-587` (above): `L_dot_est`, `dv_com_est` are backward differences of `L_com`, `v_com`.
- Fed from the live Pinocchio state at the call site: `sim_loop.py:3414` — `                            L_com=rs.L_com, L_com_prev=_L_com_qp_prev,` and `:3415` — `                            r_com=rs.r_com, v_com=rs.v_com,` (`rs` = `self.robot.update(...)`, measured, not planned).

DS branch — from the **WQP contact-wrench solution** `λ_qp`:
- `sim_loop.py:3308` — `                    _lam = np.asarray(lambda_qp_sol, dtype=float).ravel()`
- `sim_loop.py:3315` — `                        _ff -= np.cross(_r_C[_ci], _f) + _tq`

**Which Ḣ column of the NMPC plan?** — **absent.** The AOCS feedforward never reads the NMPC plan (`lambda_ref`
/ planned `Ḣ_s`); it uses measured-state finite differences (SS) or the realized QP wrench (DS).

---

## Q3 — SIGN & FRAME

**Frame: the structure body frame.** `L_com`, `v_com`, `r_com` are Pinocchio quantities, and the project
convention is Pinocchio world = structure frame `R_s`; so `ff_term` (`:588`) is expressed in the structure body
frame. The DS wrench-FF mixes frames but is treated as equivalent at small angle:
- `sim_loop.py:3298-3300` — `                # r_Ci taken in struct body frame; λ_qp in world frame — equivalent` … `                # at small structure-frame rotation` (the `<5°` transient regime).

**Sign: minus the robot angular-momentum-rate about O** (so the wheels absorb the reaction, driving
`Ḣ_structure → 0`):
- `force_estimator.py:588` — `        ff_term = -L_dot_est - orbital`
- rationale, `force_estimator.py:545-548` — `    Sign on K_θ is positive (same derivation as K_ω, K_d): Newton-Euler` / `    about structure CoM with τ_w on wheels giving -τ_w reaction on the` / `    structure. …`
- overall assembly `force_estimator.py:594` — `    tau_w = ff_term + K_hw * hw_error + pid_term` (FF, anti-windup, and PID add).

---

## Q4 — IS THE FF SHARE LOGGED SEPARATELY?

**No — requires export.** The function returns only the clipped **total**:
- `force_estimator.py:595` — `    return np.clip(tau_w, -tau_w_max, tau_w_max)`

`ff_term`, `pid_term`, and `K_hw·hw_error` are local and not returned. The sim logs only the total wheel torque:
- `crawlbot/simulation/logging.py:124` — `    tau_w: list = field(default_factory=list)` (the only τ_w channel; plus `tau_w_ss_hifreq:` `:132`, also the total at 100 Hz)
- `sim_loop.py:1063` — `        log.tau_w.append((np.zeros(3) if tau_w_applied is None` … (DS logger, total)
- `sim_loop.py:3603` — `            log.tau_w.append(tau_w_last.copy())` (SS logger, total)

(`sim_loop.py:3597` — `        log.L_dot.append(L_dot_est.copy())` logs the spin-rate `L̇_com` as a diagnostic, but
that is **not** the FF share: the SS FF is `−L̇_com − orbital` and the DS FF is `tau_struct_ff`, neither of which
is stored.) **The feedforward contribution to τ_w is not separable from the committed logs — requires export.
Not estimated.**

---

## Q5 — STRUCTURE INERTIA (HANDOFF T1)

From the MJCF, the `structure` body inertial:
- `models/VISPA_crawling_rwa3.xml:82` — `      <inertial pos="0 0 0" mass="7110"`
- `models/VISPA_crawling_rwa3.xml:83` — `                fullinertia="597 1493 1777 0 0 0"/>`

`fullinertia = "Ixx Iyy Izz Ixy Ixz Iyz"` ⇒ diagonal **[I_xx, I_yy, I_zz] = [597, 1493, 1777] kg·m²**,
off-diagonals 0. **I_zz = 1777 kg·m².**

Code mirror (structure body principal inertia, with an identical fallback literal):
- `sim_loop.py:224-225` — `        self._struct_I = self.mj_model.body_inertia[sid].copy() \` / `            if sid >= 0 else np.array([597.0, 1493.0, 1777.0])`

Reported values (no interpretation, per instruction):
| axis | inertia [kg·m²] |
|---|---|
| I_xx | 597 |
| I_yy | 1493 |
| **I_zz** | **1777** |

The HANDOFF's finite-difference-implied value is `I_s,zz ≈ 2.2·10³ kg·m²` with a ±15% pass band (1870–2530);
the model's **structure-body** `I_zz = 1777`. Factual caveat (not a pass/fail call): `1777` is the bare
structure body about its own CoM; during the docked settle the ~71 kg robot is welded to the structure, so the
**effective** inertia the terminal rotation experiences includes the robot's own inertia + parallel-axis term
about the system z-axis and is not 1777 — a composite about the system CoM would be needed to compare against
the FD estimate. Reporting the model number only, as instructed.

---

## Summary

| Q | answer |
|---|---|
| Q1 | FF compensates the **full Ḣ about O** — SS `−L̇_com − r_com×m·a_com` (`force_estimator.py:588`), DS `−Σ(r_Ci×f_i+τ_i)` (`:592`). §VI-E "only L̇_com" is a **text defect**, not a controller gap. |
| Q2 | Reconstructed by **finite difference on measured state** (SS) / from **`λ_qp`** (DS). The NMPC plan's Ḣ is **not** used — no Ḣ column read. |
| Q3 | **Structure body frame**; sign `= −(robot Ḣ about O)` (`:588`), so wheels null the structure reaction. |
| Q4 | **Not logged separately — requires export** (`force_estimator.py:595` returns only the clipped total; `logging.py:124` stores only `tau_w`). |
| Q5 | MJCF `:82-83` `fullinertia="597 1493 1777"` ⇒ **I_zz = 1777 kg·m²** (diagonal 597 / 1493 / 1777); structure body alone. |

*Read-only audit — no code, config, or model file modified; no simulation run.*
