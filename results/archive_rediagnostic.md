# M7 archive re-diagnostic — per-phase metrics

Re-computed from archived `sim_log.json` using the per-phase
refactor of `crawlbot/diagnostics/metrics.py`. No re-simulation.
Metrics are raw `np.max(np.abs(·))` over the corresponding
phase mask; `q_ref jump` is the geodesic angle between
`q_torso_ref[i_ss_last]` and `q_torso_ref[i_ds_first]`.

| version | SS peak ori [°] | DS peak ori [°] | ori at SS end [°] | q_ref jump [°] | EE pos peak SS [mm] | EE pos peak DS [mm] | abort? | abort d_mm | abort ori_deg |
|---|---|---|---|---|---|---|---|---|---|
| v17 | 0.5348 | 179.5354 | 0.1431 | 2.9440 | 153.91 | 3811.24 | yes | 24.88 | 6.30 |
| v19 | 0.5432 | 171.9536 | 0.2022 | 3.3173 | 165.37 | 2314.60 | yes | 40.78 | 6.97 |
| v20 | 0.5432 | 118.0349 | 0.2022 | 3.3173 | 165.37 | 2512.07 | yes | 40.78 | 6.97 |
| v21 | 0.5334 | 45.4672 | 0.1990 | 3.4162 | 162.38 | 933.31 | yes | 40.84 | 6.97 |
