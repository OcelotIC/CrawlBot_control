# PHASE_AOCS_CARRY — the AOCS history restart every NMPC tick

**Branch** `fix/aocs-carry` (from `refactor/sim-loop-split` @ `36c44bf`).
**Status** measured; fix implemented behind `cfg.aocs_carry_across_nmpc_ticks`,
**default OFF** — the canonical is unchanged (bit-identical, `gate/local_ref.py`).
Adopting it is a decision for Idriss: it moves the frozen numbers.

---

## 1. Root cause

`WholeBodyController.begin_tracking` (formerly the locals at the top of
`SimulationLoop._step`'s QP sub-loop) re-creates the QP carry at every NMPC
tick. On QP sub-step `qs = 0`:

- `omega_s_prev = 0` ⇒ the numerical `ω̇_s = (ω_s − 0)/dt`, and the AOCS law's
  `K_d·ω̇_s` term becomes `K_d·ω_s/dt = 2500·ω_s` N·m (`K_d = 25`, `dt = 0.01`);
- `L_com_prev = L_com`, `v_com_prev = v_com` (the current state) ⇒ in SS the FD
  feedforward `−L̇_com − r_com × m·v̇_com` is zero on that sub-step.

Spec reference: §4 *AOCS Controller (Corrected)* — both terms are finite
differences of signals sampled at dt_qp and presuppose the previous sample.
The inter-step DS settle is unaffected: it seeds its own ω_s history from the
entry value.

## 2. Measurement (legacy, canonical C run) — read-only

`scripts/diag_aocs_kick.py` wraps the AOCS law and re-evaluates it, as a side
call never applied, with the true previous-tick history. Inertness: the
instrumented run's `sim_log.json` is bit-identical to the host-local reference
(400 437 floats). Counterfactual validity: on the 6 381 sub-steps 1–9, where the
carry is already correct, it reproduces the applied command exactly (Δ = 0.0).

| sub-step 0, 709 ticks | legacy | previous-tick history |
|---|---:|---:|
| \|K_d·Δω_s/dt\|∞ median / max | 1.18 / 4.47 N·m | 0.007 / 0.12 N·m |
| \|τ_w − τ_w,true\|∞ median / p95 / max | 1.36 / 3.04 / 3.91 N·m | — |
| — of which SS (508) median | 1.80 N·m | — |
| — of which DS dwell / trailing (201) median \|τ_w\| | 0.76 N·m | 0.04 N·m |
| commands at the 2.5 N·m cap | 103 | 20 |

83 of the run's 368 saturated wheel commands (23 %) are the artefact. In the
trailing DS the AOCS receives a 0.3–1.3 N·m pulse at 10 Hz where the correct
command is ≈ 0 (figure: `results/aocs_kick/aocs_kick.png`).

## 3. Fix

`SimConfig.aocs_carry_across_nmpc_ticks` (`crawlbot/simulation/config.py`).
The controller records the last control tick's `(ω_s, L_com, v_com, τ_w)` on
both paths (`track` and `settle`); when the flag is on, `begin_tracking` seeds
sub-step 0 from it. OFF: the history is recorded but never read — byte-identical.

## 4. Closed loop with the fix ON

`scripts/diag_aocs_kick.py --carry-fix` → `results/aocs_kick_fix/`;
`gate/dock_check.py` on its log:

| | legacy (canonical) | fix ON |
|---|---:|---:|
| at-weld d [mm] (dock_gate_trace, 3 dp) | 4.016 / 4.888 / 4.990 / 4.973 / 4.954 / 4.624 | 4.017 / 4.891 / **4.997** / **4.996** / 4.953 / 4.587 |
| worst margin to 5 mm | 0.010 mm | **0.003 mm** |
| θ_s peak | 0.540° | **0.873°** (+62 %) |
| h_w peak axis / norm [N·m·s] | 4.10 / 4.24 | 3.85 / 3.94 |
| ω_s peak | 1.79 mrad/s | 1.53 mrad/s |
| e_com peak | 0.154 m | 0.154 m |
| qp_fail | 0 | 0 |
| saturated wheel commands | 368 | 287 |
| sub-step-0 K_d term, median | 1.18 N·m | 0.007 N·m |

`dock_check` reports a gate breach at 2 dp (steps 3–4 print 5.00 mm); at 3 dp
both docks fired inside the radius, by 3 and 4 µm.

## 5. Reading

The fix does what it says — the kick is gone and saturation drops by 81
commands — but the attitude gets worse, not better: θ_s peak +62 % with a
*lower* ω_s peak. The canonical gains (K_θ = 1, K_ω = 50, K_d = 25) were tuned
with the 10 Hz kick in the loop, and θ_s = 0.54° depends on it.

Options, none taken here:

1. **Keep OFF** (status quo); the published numbers stand, the artefact is
   documented.
2. **Adopt and re-tune** — one AOCS gain at a time (Rule 12), then re-freeze
   the canonical and propagate θ_s to the paper.
3. **Adopt as is** — not recommended: dock margin 3 µm, θ_s 0.87°.

## 6. Reproduce

```bash
MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_aocs_kick.py              # §2
MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_aocs_kick.py --carry-fix  # §4
MUJOCO_GL=disabled PYTHONPATH=. python3 gate/dock_check.py results/aocs_kick_fix/run/sim_log.json
PYTHONPATH=. python3 gate/local_ref.py check     # flag OFF: bit-identical
```

Artifacts: `results/aocs_kick/` and `results/aocs_kick_fix/`
(`aocs_kick_summary.json`, `aocs_kick_ticks.csv`, `aocs_kick.png`).
