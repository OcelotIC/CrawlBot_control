# INTERNAL — AOCS-during-DS audit: is the structure attitude controller active throughout ALL of DS?

**Read-only map on `ae0673e`, branch `j2/ds-active-rework`. No `crawlbot/` change, no sim run** (the
empirical tell reuses the committed `Misc/runs/fixA_gate/sim_log.json`). Reproducer `scripts/audit_aocs_ds.py`
(7/7). Raw facts; the step-4 framing is for Idriss + reviewing Claude.

---

## DECISIVE OUTPUT — **NO. The AOCS invariant is VIOLATED.**

The structure is free-floating at all times, so the AOCS (tau_w → wheels/RWA) must run every tick. It does
**not**. There are two DS code paths and they disagree:

| DS path | who drives it | AOCS each tick? | where |
|---|---|---|---|
| **terminal / DWELL DS** | `_step(phase='DS', settle_mode, ds_centroidal_active)` — NMPC-driven | **YES** — `tau_w_cmd` applied to wheels | `sim_loop.py:3275` |
| **inter-step DS** | `_run_ds_passivity_loop` — **NMPC bypassed**, passivity-constrained energy dissipation | **NO** — wheels **HARDCODED to 0.0** | `sim_loop.py:765` |

**Exactly where:** the inter-step settle loop solves the joint QP (`qp.solve(..., passivity_active=True)`,
`sim_loop.py:735`) then, before `mj_step`, explicitly zeros the wheel actuators:

```python
# sim_loop.py:763-766  (_run_ds_passivity_loop)
self.mj_data.ctrl[:n_j] = tau
if self.has_rwa:
    self.mj_data.ctrl[n_j:n_j + 3] = 0.0      # ← the violation: wheels OFF every iteration
mujoco.mj_step(self.mj_model, self.mj_data)
```

The logger records the truth: `log.tau_w.append(np.zeros(3))` (`sim_loop.py:930`), with the comment
(`L927-928`) *"RWA physical — wheels are OFF in this loop (ctrl[n_j:n_j+3]=0). tau_w = zeros is the actual
commanded value."* So for the entire inter-step settle, **the structure attitude is open-loop / uncontrolled.**

By contrast `_step` runs the AOCS every tick in **both** SS and terminal/DWELL DS. There *is* a DS opt-out
(`if cfg.aocs_off_in_ds and phase=='DS': tau_w_cmd = np.zeros(3)`, `sim_loop.py:3138`) but it
**defaults False** (`config.py:127`) and is not set in canonical ⇒ `_step` DS keeps the AOCS on. The hole is
*only* the inter-step loop.

---

## For how long — empirical (Misc/runs/fixA_gate, full 5-step traversal)

`tau_w` magnitude and tick counts, split by path (dt=0.01 s):

| path | ticks | wall time | \|tau_w\| mean | \|tau_w\| max |
|---|---|---|---|---|
| SS (AOCS on) | 265 | 2.65 s | 2.389 | 6.365 Nm |
| DS terminal/DWELL (AOCS on) | 200 | 2.00 s | 0.100 | 2.121 Nm |
| **DS inter-step (`_run_ds_passivity_loop`)** | **383** | **3.83 s** | **0.000** | **0.000 Nm** |

- The uncontrolled window is **383 ticks = 3.83 s** — **~46 % of all DS time** (383 / 583) and ~30 % of the
  whole traversal — split across **5 inter-step settles** of 0.10 / 1.72 / 0.50 / 1.01 / 0.50 s. tau_w is
  **identically zero** for every one of those ticks (max 0.0 confirms the hardcode, not just "small").
- The AOCS is doing real work where it *is* on: SS mean 2.39 / max 6.36 Nm, terminal-DS mean 0.10 / max
  2.12 Nm. It is not a dormant controller — it is suspended precisely during the inter-step dissipation.

## Does θ_s drift during the uncontrolled window? — **YES, small but non-zero & uncorrected**

Per-settle structure-attitude excursion Δθ_s = (max − min) of `struct_euler_deg` within each inter-step settle:

| settle | ticks | duration | Δθ_s [x, y, z]° | max |
|---|---|---|---|---|
| 1 | 10 | 0.10 s | [0.0000, 0.0000, 0.0000] | 0.0000° |
| 2 | 172 | 1.72 s | [0.0319, 0.0179, 0.0238] | **0.0319°** |
| 3 | 50 | 0.50 s | [0.0071, 0.0113, 0.0189] | 0.0189° |
| 4 | 101 | 1.01 s | [0.0223, 0.0276, **0.0459**] | **0.0459°** |
| 5 | 50 | 0.50 s | [0.0081, 0.0130, 0.0266] | 0.0266° |

- **Worst per-settle drift 0.0459°** (settle 4, z-axis). The drift scales with settle duration — the longer
  the loop runs open-loop, the more θ_s wanders, exactly as an uncontrolled free-floater would.
- **Why small here:** the structure is heavy (~7111 kg incl. RWA vs the 71 kg robot subtree) and the
  inter-step settle is, by construction, a low-velocity energy-dissipation phase — so the disturbance torque
  the arms inject is small and the attitude integrates slowly. **Small is not zero, and it is uncorrected**
  (no wheel torque opposes it). It also accumulates: 5 settles, no AOCS to null any of it between steps.

---

## Is re-activation feasible? — YES, the AOCS is self-contained (no NMPC dependency)

The `_step` AOCS block (`sim_loop.py:3100-3276`) reads only **state + the QP wrench**: `rs.L_com`, `omega_s`,
`hw_phys`, and `lambda_qp_sol` (and `cc_nmpc`/`r_com` for the feed-forward `tau_struct_ff_aocs`). It does
**not** consume the NMPC plan (`rp` / `lr` / `vp`). Verified by scanning the block: `uses_state=True`,
`uses_nmpc_output=False`.

This matters because the inter-step loop's whole reason to exist is that **it bypasses the NMPC**. Since the
AOCS does not need the NMPC plan — and the inter-step loop already has the inputs the AOCS needs in scope
(`rs` is recomputed each iteration; `lambda_qp_sol` is captured at `sim_loop.py:735`) — **re-activating the
AOCS in `_run_ds_passivity_loop` does not require the NMPC.** It is the same `compute_aocs_command*` call the
`_step` path already makes, fed from values already present in the loop. The bypass is not the obstacle; the
hardcoded `ctrl[n_j:n_j+3] = 0.0` is.

(The one feed-forward term `tau_struct_ff_aocs` is built from the NMPC contact plan in `_step`; in the
inter-step loop the analogous wrench is `lambda_qp_sol` from the settle QP, or it can be dropped to the
feedback-only AOCS modes — `legacy_pid_numerical` / `H_est` — which need no plan at all.)

---

## Bottom line / flags for step 4

1. **The invariant is structurally violated in `_run_ds_passivity_loop` (`sim_loop.py:765`).** Wheels are
   commanded to zero for **3.83 s / 383 ticks across 5 inter-step settles (~46 % of DS time)**; the structure
   attitude is open-loop there and **θ_s drifts up to 0.0459° per settle, uncorrected and accumulating.**
2. **It is not violated in `_step`** — SS and terminal/DWELL DS run the AOCS every tick (`aocs_off_in_ds`
   defaults False). So this is a *targeted* hole in one path, not a global AOCS-off.
3. **AOCS-reactivation in the inter-step loop is the core requirement of step 4, and it is feasible:** the
   AOCS is self-contained (state + QP wrench, no NMPC), and those inputs already exist inside the loop. The
   fix is to replace the `= 0.0` hardcode with the `compute_aocs_command*` call, not to re-architect the
   bypass.
4. **Magnitude caveat (honest):** at the canonical working point the drift is sub-0.05° per settle because
   the structure is heavy and the settle is low-velocity — so this is a *correctness / invariant* bug, not
   (yet) a demonstrated traversal-failure driver. Whether it matters dynamically should be judged at the
   larger-disturbance / CoM-mobile DS regimes step 4 targets, where the arms move more during DS.

## Reproduce
```
MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_aocs_ds.py
```
Code anchors: `sim_loop.py:765` (violation), `:930` (tau_w=0 log + comment `:927`), `:3275` (`_step`
applies AOCS), `:3138` (`aocs_off_in_ds` gate), `config.py:127` (default False). Empirical tell:
`Misc/runs/fixA_gate/sim_log.json` (committed). No `crawlbot/` change ⇒ no regression run needed.

**STOP after the report.** No design, no implementation — map only. No merge, no PR.
