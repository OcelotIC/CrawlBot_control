# T15-post-4 — Singularity analysis of step 2 swing trajectory

Read-only diagnostic. Arm-B end-effector Jacobian (6 × 7
submatrix of the full LOCAL_WORLD_ALIGNED frame Jacobian at the
`tool_b` frame) computed at every `physics_trace.pkl` sample
across the T15 bug1fix_vel run, SVD / conditioning / manipulability
tabulated per sample, per step. No simulation, no source edits.

**Script.** `scripts/diagnostics/t15_post4_singularity.py`
(standalone: reads run outputs + Pinocchio). Execution time 0.22 s
on 130 samples. No numerical warnings.

**Full raw output.** `results/M7_1pct_3step_v22_t15_bug1fix_vel/T15_post4_singularity_output.txt`.

---

## §1 Configuration echo

| Item | Value |
|---|---|
| Pinocchio version | 3.9.0 |
| URDF | `models/VISPA_crawling_fixed.urdf` |
| `model.nq / model.nv` | 21 / 20 |
| `arm_b_q_slice` | `slice(14, 21)` — q indices 14..20 |
| `arm_b_v_slice` | `slice(13, 20)` — v indices 13..19 |
| `frame_tool_b` | 36 (`model.frames[36].name = 'tool_b'`) |
| Jacobian convention | `pin.LOCAL_WORLD_ALIGNED` (matches `v_ee_b` logging in `sim_loop.py`: `J_tool_b @ v`) |
| Arm B joint columns in full `J_6×nv` | `[13, 14, 15, 16, 17, 18, 19]` |
| Data source | `results/M7_1pct_3step_v22_t15_bug1fix_vel/physics_trace.pkl` (130 SS samples) |

For each sample, computed:
- `J_6×7 = J_tool_b[:, armB_v]` (LOCAL_WORLD_ALIGNED, 6 × 7)
- SVD → six singular values `σ₁ ≥ σ₂ ≥ … ≥ σ₆`
- Split translational (`J_3×7 = J_6×7[:3, :]`) and rotational (`J_3×7 = J_6×7[3:, :]`) with separate SVDs
- Manipulability: `√det(J Jᵀ)` for the full 6×7, translational 3×7, and rotational 3×7 sub-Jacobians
- Singular direction of the smallest singular value (`U[:, −1]` of each SVD)

---

## §2 Per-step Jacobian conditioning summary (arm B EE, 6 × 7)

| Step | N samples | min σ_full | min σ_trans | min σ_rot | max cond_full | min manip_trans | max manip_trans |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 30 | **1.829e-01** | 2.330e-01 | 1.003 | **11.80** | 2.445e-01 | 4.723e-01 |
| 1 | 32 | **1.844e-01** | 2.337e-01 | 1.005 | **11.70** | 2.449e-01 | 2.757e-01 |
| 2 | 68 | **7.231e-04** | 1.737e-01 | 0.985 | **2651.73** | 1.883e-01 | 2.702e-01 |

Step 2's full (6 × 7) Jacobian has a minimum singular value **~250×
smaller** than steps 0 or 1 (7.2e-04 vs 0.18 – 0.18). The maximum
condition number is **~225×** that of steps 0 / 1 (2652 vs 11.7 –
11.8). Translational and rotational sub-Jacobians taken alone
remain conditioned comparably across all three steps (min σ_trans
0.17 – 0.23; min σ_rot ≈ 1.0 everywhere). **The near-rank-deficiency
in step 2 lives in a mixed translational-rotational mode of the
full 6 × 7 Jacobian.**

---

## §3 Step 2 SS conditioning over time

### 3.1 Coarse (~0.5 s spacing, from §B of raw output)

Excerpt — σ_min_full crosses below 1e-1 between t = 17.79 s and
t = 18.19 s (one sampling interval):

| t (s) | σ_min_full | σ_min_trans | σ_min_rot | cond_full | manip_full | manip_trans |
|---:|---:|---:|---:|---:|---:|---:|
| 14.99 | 2.226e-01 | 2.589e-01 | 1.066 | 9.59 | 1.912e-01 | 2.693e-01 |
| 15.59 | 2.232e-01 | 2.595e-01 | 1.068 | 9.56 | 1.923e-01 | 2.702e-01 |
| 16.19 | 2.213e-01 | 2.597e-01 | 1.068 | 9.63 | 1.914e-01 | 2.669e-01 |
| 16.59 | 2.141e-01 | 2.574e-01 | 1.061 | 9.96 | 1.850e-01 | 2.582e-01 |
| 16.99 | 1.971e-01 | 2.509e-01 | 1.043 | 10.83 | 1.687e-01 | 2.425e-01 |
| 17.39 | 1.644e-01 | 2.398e-01 | 1.012 | 13.01 | 1.362e-01 | 2.218e-01 |
| 17.79 | 1.044e-01 | 2.282e-01 | 0.985 | 20.45 | 8.094e-02 | 2.093e-01 |
| **18.19** | **2.226e-03** | 2.150e-01 | 1.219 | **859.10** | **2.023e-03** | 2.586e-01 |
| 18.59 | 1.055e-02 | 2.234e-01 | 1.248 | 180.58 | 9.670e-03 | 2.638e-01 |
| 18.99 | 1.447e-02 | 1.737e-01 | 1.196 | 136.03 | 1.214e-02 | 1.883e-01 |
| 19.39 | 3.455e-03 | 2.011e-01 | 1.208 | 551.39 | 3.052e-03 | 2.292e-01 |
| 20.99 | 2.967e-03 | 1.773e-01 | 1.238 | 649.54 | 2.588e-03 | 1.980e-01 |
| 21.99 | 5.195e-03 | 1.885e-01 | 1.189 | 366.43 | 4.408e-03 | 2.078e-01 |
| 22.99 | 2.005e-03 | 1.862e-01 | 1.235 | 953.25 | 1.756e-03 | 2.085e-01 |
| **23.79** | **1.160e-03** | 1.860e-01 | 1.242 | **1649.82** | **1.023e-03** | 2.081e-01 |
| **24.19** | **1.176e-03** | 1.863e-01 | 1.243 | **1626.74** | **1.040e-03** | 2.085e-01 |
| 25.79 | 1.521e-03 | 1.873e-01 | 1.246 | 1257.22 | 1.352e-03 | 2.093e-01 |
| 26.99 | 3.449e-03 | 1.912e-01 | 1.238 | 552.94 | 3.072e-03 | 2.126e-01 |
| 27.79 | 3.365e-03 | 1.916e-01 | 1.239 | 566.71 | 3.005e-03 | 2.127e-01 |
| **28.19** | **1.511e-03** | 1.851e-01 | 1.258 | **1270.22** | **1.359e-03** | 2.060e-01 |

σ_min_full drops from **0.22 at t = 14.99 s to 2.2e-03 at t = 18.19 s** (100×
drop in one 3-s window), then stays in the range 1e-3 – 2e-2 for
the remainder of step 2 SS. σ_min_trans and σ_min_rot remain in
~0.18 – 0.26 and ~1.0 – 1.25 respectively across the whole step.

### 3.2 Zoom around flailing events (every physics_trace sample, ~0.2 s spacing)

Velocity-report §3 flagged three "flailing" events (Σ q̇² > 5) at
t ≈ 18 s, 22 s, 27 s. Conditioning at those exact ticks:

**Flailing #1 (t ≈ 18 s):**

| t (s) | σ_min_full | σ_min_trans | σ_min_rot | cond_full | manip_trans |
|---:|---:|---:|---:|---:|---:|
| 17.59 | 1.391e-01 | 2.334e-01 | 0.995 | 15.40 | 2.125e-01 |
| 17.79 | 1.044e-01 | 2.282e-01 | 0.985 | 20.45 | 2.093e-01 |
| 17.99 | 5.741e-02 | 2.255e-01 | 1.008 | 36.63 | 2.241e-01 |
| **18.19** | **2.226e-03** | 2.150e-01 | 1.219 | **859.10** | 2.586e-01 |
| 18.39 | 1.877e-02 | 1.767e-01 | 1.229 | 104.76 | 1.934e-01 |
| 18.59 | 1.055e-02 | 2.234e-01 | 1.248 | 180.58 | 2.638e-01 |
| 18.79 | 7.681e-03 | 2.172e-01 | 1.266 | 248.17 | 2.556e-01 |
| 18.99 | 1.447e-02 | 1.737e-01 | 1.196 | 136.03 | 1.883e-01 |
| 19.19 | 7.840e-03 | 2.127e-01 | 1.196 | 242.16 | 2.433e-01 |
| 19.39 | 3.455e-03 | 2.011e-01 | 1.208 | 551.39 | 2.292e-01 |
| 19.59 | 8.473e-03 | 1.761e-01 | 1.225 | 229.22 | 1.970e-01 |

σ_min_full drops from 0.139 (t=17.59) to 2.2e-03 (t=18.19) —
~60× in 0.6 s.

**Flailing #2 (t ≈ 22 s):**

| t (s) | σ_min_full | σ_min_trans | σ_min_rot | cond_full | manip_trans |
|---:|---:|---:|---:|---:|---:|
| 21.59 | 4.743e-03 | 1.758e-01 | 1.224 | 407.65 | 1.965e-01 |
| 21.79 | 5.195e-03 | 1.885e-01 | 1.189 | 366.43 | 2.078e-01 |
| 21.99 | 5.202e-03 | 1.747e-01 | 1.215 | 372.25 | 1.957e-01 |
| 22.19 | 4.755e-03 | 1.889e-01 | 1.200 | 400.63 | 2.099e-01 |
| 22.39 | 4.380e-03 | 1.750e-01 | 1.225 | 441.40 | 1.964e-01 |
| 22.59 | 3.293e-03 | 1.875e-01 | 1.221 | 579.60 | 2.097e-01 |

σ_min_full stays 3e-03 – 5e-03; cond_full 370 – 580. Persistent
near-singularity through this window.

**Flailing #3 (t ≈ 27 s):**

| t (s) | σ_min_full | σ_min_trans | σ_min_rot | cond_full | manip_trans |
|---:|---:|---:|---:|---:|---:|
| 26.59 | 2.380e-03 | 1.892e-01 | 1.244 | 802.42 | 2.108e-01 |
| 26.79 | 2.411e-03 | 1.823e-01 | 1.251 | 797.72 | 2.036e-01 |
| 26.99 | 3.449e-03 | 1.912e-01 | 1.238 | 552.94 | 2.126e-01 |
| 27.19 | 3.455e-03 | 1.817e-01 | 1.247 | 557.52 | 2.026e-01 |
| 27.39 | 4.435e-03 | 1.930e-01 | 1.231 | 429.39 | 2.141e-01 |
| 27.59 | 3.644e-03 | 1.823e-01 | 1.247 | 528.71 | 2.030e-01 |

Similar sustained near-singularity.

---

## §4 Singularity-direction vs reference-velocity alignment

For each step 2 SS `physics_trace` sample: reference-velocity unit
vector `v̂_ref` = finite-difference of `log.p_ee_ref`; translational
left-singular-vector for smallest σ `ŝ_min = U_trans[:, −1]`. Report
`|cos θ| = |v̂_ref · ŝ_min|`.

### 4.1 Step 2 SS — representative samples

| t (s) | \|v_ref\| (m/s) | `v̂_ref` | `ŝ_min_trans` | \|cos θ\| |
|---:|---:|---|---|---:|
| 14.99 | 0.002 | (+0.626, 0, −0.780) | (+0.552, −0.160, +0.819) | 0.293 |
| 16.99 | 0.133 | (+0.996, 0, −0.092) | (+0.601, −0.153, +0.785) | 0.526 * |
| 17.79 | 0.181 | (+0.999, 0, −0.044) | (+0.662, −0.107, +0.742) | 0.629 * |
| 18.19 | 0.194 | (+1.000, 0, −0.022) | (+0.628, +0.052, +0.776) | 0.611 * |
| 18.59 | 0.199 | (+1.000, 0, −0.001) | (+0.689, +0.021, +0.725) | 0.688 * |
| 19.39 | 0.182 | (+0.999, 0, +0.042) | (+0.657, +0.038, +0.753) | 0.688 * |
| 19.79 | 0.162 | (+0.998, 0, +0.064) | (+0.692, +0.036, +0.721) | 0.737 * |
| 20.99 | 0.073 | (+0.988, 0, +0.157) | (+0.663, −0.062, +0.746) | 0.772 * |
| 21.39 | 0.042 | (+0.976, 0, +0.217) | (+0.688, −0.009, +0.725) | 0.829 * |
| 22.19 | 0.003 | (+0.703, 0, +0.712) | (+0.703, +0.007, +0.711) | **1.000** * |
| 22.59+ | 0.000 | — | — | — (ref saturated) |

`*` = `|cos θ| > 0.5` threshold for "singularity-aligned".

### 4.2 Alignment counts per step

| Step | Samples | `|cos|>0.5` and `|v_ref|>1e-3` | max `|cos|` in active-ref samples |
|---:|---:|---:|---:|
| 0 | 30 | 13 | 0.987 |
| 1 | 32 | 22 | 0.985 |
| 2 | 68 | 27 | **1.000** |

The alignment count and max `|cos|` are similar across all three
steps — the reference-velocity / translational-singular-direction
alignment **does not distinguish step 2 from steps 0 / 1**.

---

## §5 Verdict

**SINGULARITY CONFIRMED — step 2 passes through a near-rank-deficient
whole-arm-B Jacobian (6 × 7) configuration that is absent in
steps 0 and 1.**

Specific findings:

1. **Step 2 min σ_full = 7.23e-04**, max cond_full = 2652.
   Step 0 min σ_full = 0.183, max cond_full = 11.8.
   Step 1 min σ_full = 0.184, max cond_full = 11.7.
   **Step 2's full Jacobian is ~250× more ill-conditioned than
   steps 0 and 1.**

2. The rank deficiency is in a **mixed translational + rotational
   mode**, not in pure translation or pure rotation: step 2's
   `σ_min_trans` (0.17) is only ~25 % lower than step 0 / 1's
   (0.23), and `σ_min_rot` (0.98) is comparable to step 0 / 1's
   (1.00). The 250× drop lives in the 6-D SVD of `J_6×7`, not in
   its 3-D sub-blocks.

3. **Temporal coincidence with flailing events (velocity-report
   §3):**
   - Flailing #1 at t ≈ 18.1 s coincides with σ_min_full crossing
     from 0.14 (t = 17.59 s) down to 2.23e-03 (t = 18.19 s) — a
     60× drop in 0.6 s.
   - Flailing #2 at t ≈ 22 s coincides with σ_min_full sustained
     at 3e-03 – 5e-03 (cond_full 370 – 580) across [21.6, 22.6] s.
   - Flailing #3 at t ≈ 27 s coincides with σ_min_full sustained
     at 2e-03 – 4e-03 (cond_full 430 – 800) across [26.6, 27.6] s.

4. The reference-velocity vs translational-singular-direction
   alignment (`|cos(v̂_ref, ŝ_min_trans)|`) reaches 1.000 in step 2
   but also reaches 0.987 and 0.985 in steps 0 and 1. **Alignment
   alone does not distinguish the steps**; the distinguishing
   factor is the *magnitude* of the smallest singular value in
   step 2 (250× smaller), which amplifies the pseudo-inverse
   gain regardless of alignment direction.

### 5.1 Consequences consistent with observed behaviour

A Jacobian pseudo-inverse (explicit or implicit via QP) converts
task-space velocity command `ṗ_ref` into joint-space command
`q̇ = J⁺ ṗ_ref ≈ V Σ⁻¹ Uᵀ ṗ_ref`. When `σ_min ≈ 10⁻³`, the
contribution of the smallest-σ mode is amplified by ~10³. A 0.2
m/s reference projected onto that mode yields ~200 rad/s of
joint-velocity demand in principle, clipped by whichever
regulariser the QP uses but still an order of magnitude larger
than baseline. The observed peak of 7.66 rad/s on J1_b
(velocity-report §1) and the Σ q̇² spikes up to 43.7 rad²/s² are
consistent with this regime.

### 5.2 Steps 0 / 1 do not exhibit the condition

`max cond_full ≈ 12` for steps 0 / 1 vs 2652 for step 2. No
physics_trace sample in steps 0 or 1 has σ_min_full below 0.18.
The singularity is a property of the step-2 starting configuration
(whole-body state at t = 14.99 s reported in `T15_post2_step2_ss.md`
§Q4) and the trajectory the swing planner commands from it — not a
property of the reference shape (identical across all three
steps per `T15_post2_step2_ss.md` §Q2.3).

---

*End of T15-post-4 singularity analysis. No fix proposal in this
report.*
