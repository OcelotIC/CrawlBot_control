# T12-post-3 D5 — NMPC stage-0 cost reconstruction at k=58..70

## Method

The persisted script
`scripts/diagnostics/t12_post3_d5_cost_reconstruction.py` loads
`sim_log.json` from two runs (T12 unfixed and T12 Option A), reads
the per-tick quantities `t`, `nmpc_cost`, `nmpc_status`, `r_com`,
`r_com_ref`, `v_com`, `v_com_ref`, `L_com`, `L_com_ref`,
`lambda_ref` (12-vector of contact wrench references, two 6D
slots), `nmpc_time_ms`, and `hw_physical`, and reconstructs the
NMPC stage-0 cost from the hard-coded weights in
`CentroidalNMPCConfig` defaults (with `Wv` overridden by
`SimConfig.nmpc_Wv = 10.0`, same as default):

```
Wr = 100 · I₃,   Wv = 10 · I₃,   w_L = 1
Wu_f = 0.01,   Wu_tau = 0.001
Qf_r = 1000 · I₃,   Qf_v = 100 · I₃,   Qf_L = 10
N_horizon = 8
```

Per-tick reconstruction:

```
c_tracking    = eᵀ_r · Wr · e_r + eᵀ_v · Wv · e_v     (e = state − ref)
c_L_track     = w_L · ‖e_L‖²
c_wrench_reg  = Wu_f · (‖f₁‖² + ‖f₂‖²) + Wu_tau · (‖τ₁‖² + ‖τ₂‖²)
st0_recon     = c_tracking + c_L_track + c_wrench_reg
```

`st0_recon` is the stage-0 contribution only (the reported
`nmpc_cost` field is the full summed cost over all N=8 stages +
terminal and is therefore larger in magnitude). The reconstruction
is used to attribute the per-stage change in the full cost to its
components; it is not expected to equal the reported total.

## Source-data fingerprints

| run | path | `sim_log.json` sha256 |
|-----|------|------------------------|
| T12 unfixed | `results/M7_14pct_1step_v22_with_swing_hold/sim_log.json` | `de8f7920a69820d9978a8563cb66fc77b14b3623e1cb439a8e1c9ea1518f927c` |
| T12 Option A | `results/M7_14pct_1step_v22_with_swing_hold_optA/sim_log.json` | `6b7602bfc336d33309865d512c58eb380dbd5c4a1476f14d88993cedddabfee1` |

## T12 unfixed — per-tick reconstructed cost terms (k=58..70)

All values produced by running the script on 2026-04-22.

| k | t [s] | `nmpc_cost` (total) | st0_recon | tracking | L-track | wrench_reg | status |
|---|-------|----------------------|-----------|----------|---------|------------|--------|
| 58 | 5.91 | 280.6299 | 25.0352 | 18.7212 | 0.0000 | 6.3140  | 0 |
| 59 | 6.01 | 264.9805 | 32.7494 | 20.1214 | 0.0000 | 12.6280 | 0 |
| 60 | 6.11 | 264.4014 | 32.8241 | 20.1961 | 0.0000 | 12.6280 | 0 |
| 61 | 6.21 | 265.2605 | 32.8775 | 20.2495 | 0.0000 | 12.6280 | 0 |
| 62 | 6.31 | 265.9869 | 32.9108 | 20.2828 | 0.0000 | 12.6280 | 0 |
| 63 | 6.41 | 266.4010 | 32.9256 | 20.2976 | 0.0000 | 12.6280 | 0 |
| 64 | 6.51 | 266.5745 | 32.9272 | 20.2992 | 0.0000 | 12.6280 | 0 |
| 65 | 6.61 |  72.9978 | 16.3103 |  5.4052 | 0.0000 | 10.9051 | 1 |
| 66 | 6.71 |  70.7472 |  9.1246 |  5.2442 | 0.0000 |  3.8804 | 0 |
| 67 | 6.81 |  70.8666 |  9.1411 |  5.2526 | 0.0000 |  3.8885 | 0 |
| 70 | 7.11 |  71.0492 |  9.1539 |  5.2715 | 0.0000 |  3.8824 | 0 |

## T12 unfixed — reference quantities around k=65

| k | t [s] | r_com | r_com_ref | L_com | L_com_ref |
|---|-------|-------|-----------|-------|-----------|
| 58 | 5.91 | (+0.049, −0.015, −0.679) | (+0.109, −0.002, −0.251) | (+0.071, +0.008, +0.062) | (+0.071, +0.008, +0.062) |
| 59 | 6.01 | (+0.049, −0.014, −0.679) | (+0.118, −0.003, −0.236) | (+0.082, −0.099, +0.009) | (+0.082, −0.099, +0.009) |
| 60 | 6.11 | (+0.049, −0.014, −0.679) | (+0.119, −0.003, −0.235) | (+0.125, −0.130, −0.004) | (+0.125, −0.130, −0.004) |
| 64 | 6.51 | (+0.050, −0.013, −0.679) | (+0.119, −0.003, −0.234) | (+0.114, −0.113, −0.007) | (+0.114, −0.113, −0.007) |
| 65 | 6.61 | (+0.050, −0.013, −0.679) | (−0.178, −0.016, −0.667) | (+0.111, −0.110, −0.008) | (+0.111, −0.110, −0.008) |
| 66 | 6.71 | (+0.051, −0.012, −0.679) | (−0.178, −0.016, −0.667) | (+0.109, −0.106, −0.007) | (+0.109, −0.106, −0.007) |
| 67 | 6.81 | (+0.051, −0.012, −0.679) | (−0.178, −0.016, −0.667) | (+0.106, −0.103, −0.004) | (+0.106, −0.103, −0.004) |

## T12 unfixed — constraint-activity proxies (`|lambda_ref|` components)

| k | t [s] | \|f₁\| | \|τ₁\| | \|f₂\| | \|τ₂\| | \|λ\| |
|---|-------|--------|--------|--------|--------|-------|
| 58 | 5.91 | 25.0000 | 8.0000 |  0.0000 | 0.0000 | 26.2488 |
| 59 | 6.01 | 25.0000 | 8.0000 | 25.0000 | 8.0000 | 37.1214 |
| 60 | 6.11 | 25.0000 | 8.0000 | 25.0000 | 8.0000 | 37.1214 |
| 61 | 6.21 | 25.0000 | 8.0000 | 25.0000 | 8.0000 | 37.1214 |
| 62 | 6.31 | 25.0000 | 8.0000 | 25.0000 | 8.0000 | 37.1214 |
| 63 | 6.41 | 25.0000 | 8.0000 | 25.0000 | 8.0000 | 37.1214 |
| 64 | 6.51 | 25.0000 | 8.0000 | 25.0000 | 8.0000 | 37.1214 |
| 65 | 6.61 | 23.2113 | 8.0000 | 23.2151 | 8.0000 | 34.7233 |
| 66 | 6.71 | 13.6710 | 8.0000 | 13.7238 | 8.0000 | 22.4330 |
| 67 | 6.81 | 13.6857 | 8.0000 | 13.7386 | 8.0000 | 22.4510 |
| 70 | 7.11 | 13.6742 | 8.0000 | 13.7278 | 8.0000 | 22.4374 |

## T12 unfixed — NMPC solve time around k=65

| k | t [s] | `nmpc_time_ms` | status |
|---|-------|-----------------|--------|
| 58 | 5.91 |  23.33 | 0 |
| 59 | 6.01 |  25.63 | 0 |
| 60 | 6.11 |  35.14 | 0 |
| 61 | 6.21 |  34.31 | 0 |
| 62 | 6.31 |  34.03 | 0 |
| 63 | 6.41 |  35.36 | 0 |
| 64 | 6.51 |  35.82 | 0 |
| 65 | 6.61 | 218.10 | 1 |
| 66 | 6.71 |  93.50 | 0 |
| 67 | 6.81 |  41.23 | 0 |
| 70 | 7.11 |  41.62 | 0 |

## T12 unfixed — `h_w` (component-wise box ±5 N·m·s) around k=65

| k | t [s] | h_x | h_y | h_z | \|h\| |
|---|-------|------|------|------|--------|
| 58 | 5.91 | −0.1250 | +0.2017 | −0.0457 | 0.2416 |
| 59 | 6.01 | +0.0050 | +0.2017 | −0.0481 | 0.2074 |
| 60 | 6.11 | +0.0242 | +0.1766 | −0.0610 | 0.1884 |
| 64 | 6.51 | −0.0436 | +0.0764 | −0.1608 | 0.1833 |
| 65 | 6.61 | −0.0615 | +0.0513 | −0.1857 | 0.2022 |
| 66 | 6.71 | −0.0756 | +0.0263 | −0.2106 | 0.2253 |
| 67 | 6.81 | −0.0929 | +0.0013 | −0.2355 | 0.2532 |

## T12 Option A — same tables

The T12 Option A run produces bit-identical per-tick values for all
of the above tables over k=58..70 (verified to 4 decimal places by
the script's second pass). The `r_com_ref.z` jump at k=65 from
−0.234 → −0.667, the tracking-cost drop 20.2992 → 5.4052, the
wrench-regularizer drop 12.6280 → 10.9051 → 3.8804, and the |f|
unbinding from 25 to ≈ 13.67 are identical across both runs.
Option A's smoothing of the torso position reference therefore has
no effect on the k=65 cost regime change. The source of that
regime change is the `r_com_ref.z` advance at the k=64→k=65
planner waypoint boundary.
