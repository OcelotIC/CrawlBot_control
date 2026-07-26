# T15-post-2 — Velocity analysis (instrumented rerun)

Rerun of T15 on `claude/t15-bug1-fix` at HEAD = `a752b4e`
(velocity instrumentation landed at `e680164`, Option Z fix at
`7c8f01a`). Same controller behaviour as the prior bug1fix run
(2/3 docks, step 2 `dock_timeout` at t = 28.49 s, d = 374 mm) —
this run adds the 8 new velocity fields to `SimLog` to
characterise the **velocity-side** signature of step 2's failure.

**Output:** `Misc/runs/M7_1pct_3step_v22_t15_bug1fix_vel/`
(`sim_log.json`, `metrics.csv`, `physics_trace.pkl`, 10 plots).

**Run identity check** vs prior bug1fix run:

| Outcome | prior bug1fix | bug1fix_vel | Δ |
|---|---|---|---|
| Step 0 dock | t=6.01 s, d=3.82 mm, ori=0.08° | t=6.01 s, d=3.8 mm, ori=0.08° | identical |
| Step 1 dock | t=13.02 s, d=4.84 mm, ori=0.22° | t=13.02 s, d=4.8 mm, ori=0.22° | identical |
| Step 2 abort | d=374.35 mm, ori=9.84° | d=374.4 mm, ori=9.8° | identical |

Controller is deterministic; only `SimLog` fields differ.

---

## §1 Per-step SS joint-velocity peaks (swing arm)

Peak `|q̇|` per joint of the **swinging** arm within the SS
window of each step.

### Step 0 SS (swing arm = b, N = 59 ticks, 0.11–5.91 s)

| joint | peak \|q̇\| (rad/s) | rms (rad/s) |
|---|---:|---:|
| J1_b | 0.138 | 0.079 |
| J2_b | 0.409 | 0.252 |
| J_swivel_b | 0.182 | 0.091 |
| J3_b | 0.405 | 0.221 |
| J4_b | 0.295 | 0.159 |
| J5_b | 0.254 | 0.131 |
| J6_b | 0.157 | 0.094 |

### Step 1 SS (swing arm = a, N = 65 ticks, 6.52–12.92 s)

| joint | peak \|q̇\| (rad/s) | rms (rad/s) |
|---|---:|---:|
| J1_a | 1.206 | 0.514 |
| J2_a | 0.346 | 0.163 |
| J_swivel_a | 0.220 | 0.094 |
| J3_a | 0.423 | 0.225 |
| J4_a | 0.260 | 0.136 |
| J5_a | 1.216 | 0.534 |
| J6_a | 0.254 | 0.107 |

### Step 2 SS (swing arm = b, N = 135 ticks, 14.99–28.39 s)

| joint | peak \|q̇\| (rad/s) | rms (rad/s) | × step 0 peak |
|---|---:|---:|---:|
| J1_b | **7.664** | 2.881 | 55× |
| J2_b | 0.640 | 0.145 | 1.6× |
| J_swivel_b | **5.150** | 1.574 | 28× |
| J3_b | **3.286** | 0.982 | 8.1× |
| J4_b | **2.861** | 1.279 | 9.7× |
| J5_b | 1.303 | 0.440 | 5.1× |
| J6_b | 1.021 | 0.187 | 6.5× |

Peaks on J1_b, J_swivel_b, J3_b, and J4_b exceed 2.8 rad/s and
are 5–55× larger than their step-0 counterparts.

### Stance-arm comparison

| Step | Stance arm peak \|q̇\| (rad/s) |
|---|---:|
| 0 | 0.035 |
| 1 | 0.165 |
| 2 | **0.629** |

The stance arm also sees an order-of-magnitude increase in joint
rate at step 2 vs step 0.

---

## §2 EE and torso velocities (swing arm)

| metric | Step 0 SS | Step 1 SS | Step 2 SS |
|---|---:|---:|---:|
| \|v_ee\| peak (m/s) | 0.260 | 0.239 | 0.183 |
| \|v_ee\| mean (m/s) | 0.138 | 0.126 | 0.053 |
| \|ω_ee\| peak (rad/s) | 0.005 | 0.057 | **0.180** |
| \|ω_ee\| mean (rad/s) | (not reported) | — | — |
| \|v_torso\| peak (m/s) | 0.019 | 0.049 | **0.122** |
| \|v_torso\| mean (m/s) | 0.013 | 0.025 | 0.040 |
| \|ω_torso\| peak (rad/s) | 0.010 | 0.019 | **0.166** |
| \|ω_torso\| mean (rad/s) | 0.006 | 0.012 | 0.053 |

Step 2 swing EE has the **lowest** linear-velocity peak but the
**highest** angular-velocity peak among the three steps. Torso
linear and angular velocities are both elevated ~8–16× vs step
0. The swing arm is spinning its EE orientation (ω_ee peak 0.18
rad/s) while the EE translates more slowly than in steps 0 or 1.

---

## §3 Joint-vs-EE decoupling in step 2 SS

Sum-of-squares of arm-B joint velocities through step 2 SS:

| t (s) | Σ q̇_i² (rad²/s²) | dominant joint |
|---:|---:|---|
| 14.99 | 0.000 | J2_b |
| 15.99 | 0.005 | J2_b |
| 16.99 | 0.190 | J3_b |
| **17.99** | **21.275** | **J1_b** |
| **18.99** | **43.677** | **J1_b** |
| 19.99 | 0.450 | J1_b |
| 20.99 | 0.484 | J1_b |
| **21.99** | **35.358** | **J1_b** |
| 22.99 | 2.183 | J1_b |
| 23.99 | 0.687 | J1_b |
| 24.99 | 0.864 | J1_b |
| 25.99 | 0.965 | J1_b |
| **26.99** | **9.548** | **J1_b** |
| 27.99 | 0.054 | J1_b |

Three distinct joint-velocity "flailing" events: at t ≈ 18–19 s,
t ≈ 22 s, and t ≈ 27 s. J1_b is the dominant joint in all three.

Per-joint peak `|q̇|` with the same-tick EE / torso context:

| joint | peak \|q̇\| (rad/s) | t (s) | \|v_ee_b\| (m/s) | \|v_ref\| (m/s) | \|ω_ee_b\| (rad/s) | \|v_torso\| (m/s) |
|---|---:|---:|---:|---:|---:|---:|
| J1_b | 7.664 | 18.09 | 0.110 | 0.192 | 0.112 | 0.122 |
| J_swivel_b | 5.150 | 18.09 | 0.110 | 0.192 | 0.112 | 0.122 |
| J3_b | 3.286 | 18.09 | 0.110 | 0.192 | 0.112 | 0.122 |
| J4_b | 2.861 | 19.29 | 0.055 | 0.186 | 0.180 | 0.096 |
| J5_b | 1.303 | 18.09 | 0.110 | 0.192 | 0.112 | 0.122 |
| J6_b | 1.021 | 18.09 | 0.110 | 0.192 | 0.112 | 0.122 |
| J2_b | 0.640 | 17.99 | 0.181 | 0.189 | 0.016 | 0.044 |

At t = 18.09 s, six of seven swing-arm-B joints hit their peak
velocity simultaneously. `|v_ee_b|` at that instant is 0.110 m/s
— **57 % of the planned reference velocity `|v_ref| = 0.192 m/s`**.
Very high joint rates (sum 21–44 rad²/s²) producing only
moderate / sub-reference EE motion.

---

## §4 Actual-vs-reference EE speed matching through step 2 SS

`|v_ref|` estimated as a finite-difference of `log.p_ee_ref`.
Reference goes to zero after the quintic saturates (t ≳ 22.5 s).

| t (s) | \|v_ee_b\| actual (m/s) | \|v_ref\| planned (m/s) | actual / ref |
|---:|---:|---:|---:|
| 14.99 | 0.005 | 0.002 | 2.59 (start, both ~0) |
| 15.49 | 0.019 | 0.021 | 0.91 |
| 15.99 | 0.051 | 0.054 | 0.94 |
| 16.49 | 0.092 | 0.094 | 0.98 |
| 16.99 | 0.132 | 0.133 | 0.99 |
| 17.49 | 0.165 | 0.166 | 1.00 |
| **17.99** | **0.181** | **0.189** | **0.96** |
| **18.49** | **0.104** | **0.199** | **0.52** |
| **18.99** | **0.156** | **0.195** | **0.80** |
| 19.49 | 0.122 | 0.178 | 0.68 |
| 19.99 | 0.080 | 0.150 | 0.54 |
| 20.49 | 0.048 | 0.113 | 0.42 |
| 20.99 | 0.027 | 0.073 | 0.36 |
| 21.49 | 0.023 | 0.035 | 0.65 |
| 21.99 | 0.022 | 0.009 | 2.56 (ref decelerating to 0) |
| 22.49 – 28.39 | 0.01–0.03 | ~0.000 | n/a (ref saturated at `anchors_b[4]`) |

From SS entry up to t = 17.49 s, actual EE velocity tracks the
reference within 2 %. **At t = 18.49 s the tracking ratio halves
(1.00 → 0.52)** and never fully recovers — the ratio oscillates
between 0.36 and 0.80 through the rest of the planned-swing
window (t = 18.49 – 21.49 s). After the reference saturates at
`anchors_b[4]` (t ≳ 22.5 s), actual EE continues to wander at
0.02–0.03 m/s instead of converging to zero at the target.

---

## §5 Temporal alignment with T15-post-2 observations

| Observation | T15-post-2 source | time in step 2 SS |
|---|---|---:|
| Joint flailing event #1 | §3 this report | t ≈ 18.09 s |
| EE position-error takes off (0 → 115 mm) | Q1.1 (T15-post-2) | t ≈ 18.99 s |
| EE orientation error takes off (0 → 4°) | Q1.1 (T15-post-2) | t ≈ 18.99 s |
| ω_ee peak (0.18 rad/s on J4_b tick) | §2 / §3 this report | t ≈ 19.29 s |
| Joint flailing event #2 | §3 this report | t ≈ 21.99 s |
| Reference saturates at `anchors_b[4]` | Q2.1 (T15-post-2) | t ≈ 22.52 s |
| EE position-error peak (432.8 mm) | Q1.2 (T15-post-2) | t ≈ 22.99 s |
| Joint flailing event #3 | §3 this report | t ≈ 26.99 s |
| Abort (`dock_timeout`) | T15_report.md §2.4 | t = 28.49 s |
| `sig_min_NJe` minimum (0.049, torso-null-projected stance J) | T15-post-2 Q3.3 | t = 28.39 s |
| `cond_NJe` maximum (38.4) | T15-post-2 Q3.3 | t = 28.39 s |

Joint-velocity flailing events precede both the EE position-error
takeoff (by ~0.9 s) and the EE orientation-error takeoff (by
~0.9 s). The first flailing event at t ≈ 18 s coincides with
`|v_ee| / |v_ref|` dropping from 1.00 to 0.52.

---

## §6 Summary of velocity signature (numbers only)

| fact | value |
|---|---|
| Step 2 SS peak arm-B joint rate | **7.664 rad/s** on J1_b at t = 18.09 s |
| Ratio to step-0 SS peak (J2_b, 0.409 rad/s) | **18.7×** |
| Ratio of step 2 peak to its step-0 counterpart (J1_b, 0.138 rad/s) | **55×** |
| Step 2 SS peak EE linear speed | 0.183 m/s (at t = 17.99 s) |
| Step 2 SS peak EE angular speed | 0.180 rad/s (at t = 19.29 s) |
| Ratio of EE linear peak step-2 / step-0 | 0.70× (**lower**) |
| Ratio of EE angular peak step-2 / step-0 | **37×** |
| Step 2 SS peak torso linear speed | 0.122 m/s |
| Step 2 SS peak torso angular speed | 0.166 rad/s (**17× step 0**) |
| Step 2 SS tick where `|v_ee|/|v_ref|` first drops below 0.9 | t = 18.49 s (ratio = 0.52) |
| Joint-velocity flailing events (Σ q̇² > 5) | 3 (t ≈ 18, 22, 27 s); J1_b dominant |
| Peak Σ q̇² during flailing | 43.7 rad²/s² at t = 18.99 s |

*(End of velocity analysis. No interpretation or fix proposal in
this report.)*
