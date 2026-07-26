# M7 — DS settle weld solref sweep (damping=0, armature=0)

Three 51-step DS settles, identical robot config (`damping=0, armature=0` via transient MJCF mutation; restored on script exit). Only difference across rows: the `<weld solref="...">` attribute applied to every `grip_*_to_*` weld in the MJCF.

## Consolidated results

| variant | solref | n_steps | T_end [J] | exit | mean(P_weld) [W] | max(P_weld) [W] | mean(‖J_c·q̇‖) [m/s, rad/s] | max(‖J_c·q̇‖) | mean(lin) / mean(ang) | max(lin) / max(ang) |
|---|---|---|---|---|---|---|---|---|---|---|
| solref_0p003_1 | `"0.003 1"` | 51 | 1.9142e-01 | plateau | +2.3033e+02 | +2.9413e+02 | 6.7808e+00 | 8.6465e+00 | 1.162e-02 / 6.781e+00 | 1.930e-02 / 8.646e+00 |
| solref_0p001_1 | `"0.001 1"` | 51 | 1.9142e-01 | plateau | +2.3033e+02 | +2.9413e+02 | 6.7808e+00 | 8.6465e+00 | 1.162e-02 / 6.781e+00 | 1.930e-02 / 8.646e+00 |
| solref_direct_stiff | `"-1e6 -1e3"` | 51 | 8.2407e+41 | plateau | -3.9226e+03 | +8.1933e+01 | 1.9385e+20 | 2.1539e+20 | 1.939e+20 / 3.381e+17 | 2.154e+20 / 3.757e+17 |

## MJCF mutation protocol

Per variant, the MJCF file is edited in place: the `<default class="robot_joint">`'s `damping` / `armature` attributes are set to 0, and every `<weld>` element's `solref` attribute is replaced with the variant value. The script records the branch-HEAD state of the MJCF at entry and restores it via `try/finally`. A re-read at exit verifies `damping=0.05`, `armature=0.05`, and `solref="0.003 1"` on every weld.

## Per-tick traces

- `solref_0p003_1`: `Misc/runs/M7_settle_diag/weld_sweep/solref_0p003_1/trace.csv`
- `solref_0p001_1`: `Misc/runs/M7_settle_diag/weld_sweep/solref_0p001_1/trace.csv`
- `solref_direct_stiff`: `Misc/runs/M7_settle_diag/weld_sweep/solref_direct_stiff/trace.csv`

Trace columns per tick: `k, t_s, T_full_J, P_weld_W, tau_max_Nm, Jcq_norm_full, Jcq_norm_lin, Jcq_norm_ang, Jcq_A_lin_{x,y,z}, Jcq_A_ang_{x,y,z}, Jcq_B_lin_{x,y,z}, Jcq_B_ang_{x,y,z}`. Per-contact norms are pooled (lin = concat(A_lin, B_lin); ang = concat(A_ang, B_ang)) for the summary columns.

## MJCF restoration check

`_verify_mjcf_restored()` passed at script exit: `damping="0.05"` and `armature="0.05"` on the `robot_joint` default, and `solref="0.003 1"` is the unique value across all twelve `<weld>` elements (`grip_a_to_1a` … `grip_b_to_6b`).
