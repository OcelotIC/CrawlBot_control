# C2.1 + C2.3 — dock twist and AOCS decomposition

**Review-Closure Bloc 2, phases C2.1 and C2.3, run as one round.** Logging-only.
Both are persistence of quantities the code already computes and discards, so
they share one canonical replay and one neutrality proof.

---

## Header (mandatory)

| item | value |
|---|---|
| commit worked on | branch `claude/review-closure-bloc-2-uwu1x7`, parent **`7d2a218`** (C2.2) |
| date | 2026-07-27 |
| python / mujoco / pinocchio | 3.11.15 / 3.10.0 / 3.9.0 |
| casadi + ipopt / numpy / scipy | 3.7.2 + IPOPT-MUMPS / 2.3.5 / 1.17.1 |
| env vs `gate/environment.lock` | exact match |
| host | Intel Xeon @ 2.10 GHz, 4 logical CPUs, 15 GiB, Linux 6.18.5 |

**Artifacts**: `results/review_closure/c2/c25_c2_fulldiag.csv` (92 cols),
`c25_c2_fulldiag_meta.json`, `c2_neutrality_verdict.json`.

---

## 1. Neutrality — PASS, both routes

| route | result |
|---|---|
| **flag OFF** — `gate/run_gate.py`, unmodified | **PASS**, 2077 rows × **132 928 fields byte-identical**, replay rc=0 (197.7 s) |
| **flag ON** — `c2_neutrality_check.py` | **PASS**, 66 baseline columns byte-identical + **26 appended** (7 from C2.2 + 19 new) |
| `gate/dock_check.py` | 6/6 at-weld, **every step delta +0.0000**, θ_s 0.540°, h_w 4.102/4.243, e_com 0.154, qp_fail 0 — `MATCH frozen 2.5` |
| `gate/run_suite.py --fast` | PASS (199 tests, 198 passed, 0 failed, 1 xfail) |

The identity check on the AOCS decomposition also passes independently:
`max |Σ terms − tau_w_preclip| = 9.88e-07` across all 2077 ticks — that residual
is the CSV's 6-significant-digit formatting, not a discrepancy in the sum.

---

## 2. C2.1 — 6-D relative twist at capture

`_weld_relative_twist` already returned a 6-vector; `_dock_gate` collapsed it to
a norm and threw the components away, so a refused capture could report "twist
too high" without saying whether the excess was linear or angular.

**Frame.** The raw quantity is `[j_pos(gripper) − j_pos(anchor); j_rot(g) − j_rot(a)] · qvel`
from `mj_jacSite`, i.e. MuJoCo's **global** frame. `_twist_components` rotates by
`R_sᵀ` into the **structure frame**, matching every other reported quantity
(θ_s, h_w, the anchor grid). **The norm is rotation-invariant**, so the quantity
the gate actually thresholds is untouched — that is what makes this
control-neutral rather than merely control-neutral-in-practice.

**Per-dock twist at capture** (structure frame; `eps_twist = 0.05` on all six):

| step | ‖twist‖ | linear ‖·‖ | angular ‖·‖ | linear share |
|---:|---:|---:|---:|---:|
| 0 | 0.020058 | 0.014830 | 0.013510 | 74 % |
| 1 | 0.007735 | 0.007290 | 0.002590 | 94 % |
| 2 | 0.005781 | 0.005360 | 0.002165 | 93 % |
| 3 | 0.007170 | 0.006730 | 0.002487 | 94 % |
| 4 | 0.004990 | 0.004650 | 0.001818 | 93 % |
| 5 | 0.007020 | 0.006550 | 0.002516 | 93 % |

**Capture is linear-dominated** — 93–94 % of the twist norm is translational on
five of six docks. Step 0 is the outlier at 74 %, with an angular component 5×
the others; it is also the first step, released from the initial IK pose rather
than from a settled post-dock state.

Every `dock_events` row now also carries the criteria in force
(`eps_pos_m`, `eps_ori_deg`, `eps_twist`, `gate_uses_6d_twist`,
`dock_check_delay_s`), and the full `dock_gate_trace` is passed through to the
meta JSON. This matters because C1.6 found that `dock_twist_max` — documented
in-source as *"NOT a tuned value"* — is what sets the worst reported dock and
~70 % of the managed-vs-unmanaged traversal difference. **A dock can no longer
be read without the gate that produced it.** This closes C3.5's dependency.

---

## 3. C2.3 — AOCS torque decomposition

Five contributions and the pre-clip total, per axis, per tick. All 2077 ticks
measured, **zero sentinel rows** (the inter-step settle runs the canonical AOCS
every tick, so it is a real measurement there too).

| term | peak/axis [N·m] | RMS [N·m] |
|---|---:|---:|
| `tau_ff` (feedforward) | **2.6537** | **0.8142** |
| `tau_accel_d` (K_d·ω̇_s) | 1.3823 | 0.0319 |
| `tau_rate_d` (K_ω·ω_s) | 0.0893 | 0.0210 |
| `tau_att_p` (K_θ·θ_s) | 0.0093 | 0.0024 |
| `tau_antiwindup` | **0.0000** | **0.0000** |

Three results worth carrying into C3.1:

1. **The AOCS is feedforward, to a first approximation.** `tau_ff` carries
   RMS 0.814 N·m against 0.032 for the largest feedback term — a **25:1**
   feedforward-to-feedback ratio in RMS. The attitude-P term peaks at
   **9.3 mN·m**, 0.35 % of the feedforward peak. That is consistent by
   construction (`K_θ = 1.0 N·m/rad` against θ_s ≤ 0.0094 rad = 0.540°), but it
   is now measured rather than inferred, and it bears on how the paper describes
   the attitude controller's role.
2. **The anti-windup term is exactly zero on every tick** — not "small", not
   "≈0", but identically 0.0000 for all 2077 ticks × 3 axes. C1.4 asserted this
   from `|h_w|_∞ = 4.1019 < 5`; it is now a direct per-tick measurement.
3. **Saturation is real and is not where you would expect.** Peak demand is
   **3.899 N·m against the ±2.5 cap** — and it occurs in `DS_interstep`, not in
   SS. `DS_terminal` never saturates at all (peak demand 0.705 N·m).

---

## 4. ⚠ The paper's 4.1 % clip fraction does not reproduce

The brief's C3.1 asks to relate the decomposition to "the 4.1 % clip fraction"
the paper cites. Measured from `tau_w_preclip` against the ±2.5 cap, under both
natural definitions:

| definition | value |
|---|---:|
| ticks with **any** axis saturating | 95 / 2077 = **4.57 %** |
| **axis-samples** saturating | 99 / 6231 = **1.59 %** |
| *paper as cited in the brief* | *4.1 %* |

Neither matches. 4.57 % is close but not equal; 1.59 % is far. Per axis:
x 0.14 % (peak 2.508), y 1.01 % (peak **3.899**), z 3.61 % (peak 2.738). Per
phase: SS 3.94 %, DS_interstep **5.48 %**, DS_terminal 0.00 %.

I cannot tell from the repository which definition produced 4.1 %, or against
which run — the number appears in no committed artifact. **Flagged, not
resolved**: C3.1 should either recover the original definition or restate the
figure from these channels, and the paper should say which convention it uses.
The same applies to the "368/51448 plant clamps" the brief mentions — 51 448 is
neither 2077 nor 6231 nor 2077×10, so it counts something at a third cadence
(plausibly per-QP-substep × axes), and that convention is likewise unrecorded.

---

## 5. Files changed

| file | change |
|---|---|
| `crawlbot/aocs/force_estimator.py` | five terms named separately and summed; optional `decomposition` out-dict; return value unchanged |
| `crawlbot/simulation/sim_loop.py` | `_twist_components`, `_dock_thresholds`; `_dock_gate` keeps the 6-vector; dock events enriched (both sites); `aocs_decomp` threaded through `_step` and the inter-step loop |
| `crawlbot/simulation/logging.py` | 7 `aocs_*` list fields with the sentinel documented |
| `crawlbot/simulation/tick_logging.py` | `_log_aocs_decomposition` shared by both recorders; `TickState.aocs_decomposition` |
| `scripts/diag_full_diag_export.py` | 19 more `--solver-diag` columns; `dock_events` + `dock_gate_trace` into the meta |
| `docs/crawlbot/{aocs/force_estimator,simulation/logging,simulation/tick_logging,simulation/sim_loop}.md` | Rule 15 |
| `CLAUDE.md` | α torque-min / accel-reg anchor `:957 → :971` |

One latent bug was caught and fixed while wiring: `_aocs_decomp` was initially
bound inside the `aocs_active_in_interstep` branch, which would have raised
`NameError` at the recorder on the flag-OFF and no-RWA paths. It is now bound
before the branches, where an empty dict is exactly the documented sentinel.

---

## STOP

C2.1 and C2.3 complete; neutrality PASS by both routes with all six docks at
delta +0.0000. **C3.5 and C3.1 are now unblocked.**

Two items for Idriss:

1. **The 4.1 % clip fraction does not reproduce** under either natural
   definition (§4). Needs the original convention, or a restatement.
2. Unchanged from C2.2: flipping `--solver-diag` on by default requires
   regenerating `c25_fulldiag.csv` or teaching `gate/run_gate.py` to tolerate
   appended columns.
