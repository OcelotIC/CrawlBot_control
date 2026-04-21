# M7 — Armature decomposition, Part 1 summary

Diagnostic only, per `docs/architecture/M7_ARMATURE_DECOMPOSITION.md`.
**Part 1 only** — Part 2 has not been run; stop-and-report gate is this
summary. No commits. No fix.

## Pinocchio change applied

`crawlbot/core/robot_interface.py:189-201` (after the v-space slice
detection): `model.armature` assigned to a length-20 array with 0 on
the 6-DOF floating base and 0.05 on each of the 14 arm joints
(`slices['joints_v']`), followed by `data = model.createData()`. The
free-flyer base column block receives zero armature.

## Pinocchio-side `diag(H)` before/after, at neutral `q`

All units kg·m² (since free-flyer linear DOFs show 71.056 kg here in the
first three diagonal entries, and the arm entries are inertias about
their joint axes). Computed as `np.diag(pin.crba(model, data,
pin.neutral(model)))`.

```
base(6)  WITHOUT armature: [71.056000, 71.056000, 71.056000, 36.043721, 16.147468, 40.133717]
base(6)  WITH armature:    [71.056000, 71.056000, 71.056000, 36.043721, 16.147468, 40.133717]
Δ base(6):                 [0, 0, 0, 0, 0, 0]

arms(14) WITHOUT armature: [0.133541, 10.185757, 0.033028, 2.856509, 0.023069, 0.179369, 0.001100,
                            0.133541, 10.185757, 0.033028, 2.856509, 0.023069, 0.179369, 0.001100]
arms(14) WITH armature:    [0.183541, 10.235757, 0.083028, 2.906509, 0.073069, 0.229369, 0.051100,
                            0.183541, 10.235757, 0.083028, 2.906509, 0.073069, 0.229369, 0.051100]
Δ arms(14):                [0.050000]×14
```

`model.armature` after install = `[0]×6 ∪ [0.05]×14`.

## `pytest tests/ -v` with armature installed

`192 passed, 1 warning in 27.79s`. No test fixes required — all 192
tests that passed before the change pass after it.

## Part 1 — A_swing with Pinocchio armature installed

Same A_swing configuration as in the EE-position bisection: standalone
QP, torso reference constant, EE reference from `SwingPlanner`
(quintic + clearance bump + SLERP), `swing_arm='b'`, anchor 2→3,
`T_step = 7.284 s`, `sim._deactivate_weld('b', 2)` before the loop.
MJCF unchanged (`damping=0.05`, `armature=0.05`).

| metric | prior A_swing (no Pin armature) | Part 1 (Pin armature 0.05) |
|---|---|---|
| `ee_pos_peak_SS` [mm] | 3.8180 | **3.7782** |
| `ee_ori_peak_SS` [deg] | 16.7362 | **0.8798** |
| time of EE ori peak [s] | 6.740 | **5.420** |
| `ee_ori_at_T_step` [deg] | 16.3632 | **0.4059** |
| `ee_pos_at_T_step` [mm] | 3.0830 | 3.7537 |
| `tau_q_peak_SS` (all 14) [Nm] | 0.6000 | 0.6077 |
| `tau_q_peak_ang` (wrist B) [Nm] | 0.0046 | 0.0512 |
| qpos checksum | 5.3040941901 | 5.3040941901 |
| `T_step` [s] | 7.284331 | 7.284331 |
| NMPC calls, mapping calls | 0, 0 | 0, 0 |

Per-tick trace: `results/M7_armature_decomposition/A_swing_with_pin_armature/trace.csv`
(729 SS-window rows; columns `t_s, e_ee_pos_mm, e_ee_ori_deg,
tau_q_max_Nm, tau_q_max_ang_Nm`, identical schema to
`results/M7_ee_bisection/A_swing/trace_v2.csv`).

## Stop-and-report gate

Per the task file §5: "If Part 1 shows `ee_ori_peak_SS` unchanged or
worse, stop and report before running Part 2."

Observation: `ee_ori_peak_SS` went from **16.7362°** to **0.8798°**
(factor of 19.0 reduction). `ee_ori_at_T_step` went from 16.3632° to
0.4059° (factor of 40.3 reduction). `ee_pos_peak_SS` remained
essentially unchanged (3.82 → 3.78 mm). `tau_q_peak_ang` increased
from 0.0046 to 0.0512 Nm.

Part 2 has not been executed. Awaiting Idriss's call.


---

# M7 — Armature decomposition, Part 2 summary

Per `docs/architecture/M7_ARMATURE_DECOMPOSITION.md` §3. Seven inter-step DS passivity-settle variants; each decouples MJCF damping / MJCF armature / Pinocchio armature. For each variant the MJCF is mutated transiently (restored byte-exactly on exit) and the installed Pinocchio `model.armature` is overridden per-variant after `sim.setup()`. All variants start from the post-setup state with `T_start ≈ 0`.

Settle threshold `T_settle = 0.5·ε²·λmin(H)` is recomputed at entry per variant (it depends on the mass matrix, which includes armature). Reported below with each row.

| variant | MJCF damping | MJCF armature | Pin armature | T_start [J] | T_end [J] | T_settle [J] | exit_reason | n_steps_run |
|---|---|---|---|---|---|---|---|---|
| `a0_d0` | 0.0 | 0.0 | 0.0 | 0.000e+00 | 1.914e-01 | 5.224e-10 | plateau | 51 |
| `a0p01_d0` | 0.0 | 0.01 | 0.01 | 0.000e+00 | 1.504e-01 | 5.522e-09 | plateau | 51 |
| `a0p02_d0` | 0.0 | 0.02 | 0.02 | 0.000e+00 | 1.160e-01 | 1.052e-08 | plateau | 51 |
| `a0p03_d0` | 0.0 | 0.03 | 0.03 | 0.000e+00 | 2.134e-09 | 1.552e-08 | target_met | 11 |
| `a0p04_d0` | 0.0 | 0.04 | 0.04 | 0.000e+00 | 1.556e-10 | 2.052e-08 | target_met | 11 |
| `a0p05_d0` | 0.0 | 0.05 | 0.05 | 0.000e+00 | 1.530e-11 | 2.552e-08 | target_met | 11 |
| `a0_d0p05` | 0.05 | 0.0 | 0.0 | 0.000e+00 | 1.887e-01 | 5.224e-10 | plateau | 51 |

Per-variant `T_kinetic(t)` traces:

- `a0_d0`: `results/M7_armature_decomposition/settle_variants/a0_d0/trace.csv`
- `a0p01_d0`: `results/M7_armature_decomposition/settle_variants/a0p01_d0/trace.csv`
- `a0p02_d0`: `results/M7_armature_decomposition/settle_variants/a0p02_d0/trace.csv`
- `a0p03_d0`: `results/M7_armature_decomposition/settle_variants/a0p03_d0/trace.csv`
- `a0p04_d0`: `results/M7_armature_decomposition/settle_variants/a0p04_d0/trace.csv`
- `a0p05_d0`: `results/M7_armature_decomposition/settle_variants/a0p05_d0/trace.csv`
- `a0_d0p05`: `results/M7_armature_decomposition/settle_variants/a0_d0p05/trace.csv`

## MJCF restoration

`_verify_restored_byte_equal(original)` returned `True` at script exit — MJCF is byte-identical to the text captured at script entry (`damping="0.05"`, `armature="0.05"` on the `robot_joint` default).
