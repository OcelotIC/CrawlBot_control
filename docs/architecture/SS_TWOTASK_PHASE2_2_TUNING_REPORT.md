# SS two-task weight×gain tuning (Phase 2.2) — torso arrival vs τ_w-envelope

Two-axis sweep on the **canonical single step** (real NMPC + MuJoCo, 1 step), mom weight
fixed at the validated **5000**. Goal: reduce torso arrival residual from the balanced
working point's 11 % toward ~2–3 % of the torso's own travel (125.5 mm) **without** pushing
τ_w-saturation to the TP-dominant level (~24 %). Analysis `scripts/report_phase2_2_tuning.py`
(reuses the Phase-2.1 metric definitions verbatim: arrival vs geometric `p_t1`, τ_w-sat @100 Hz).

---

## TL;DR

1. **~2–3 % arrival is NOT reachable at low saturation.** The only point at ~3 % arrival is
   TP-dom `0.5k:30k` (3.3 %), and it costs **23.8 % τ_w-sat**. Softening *its* gain (Kp 6→3)
   only trades arrival straight back (→ 5.1 % arrival, 15.7 % sat). The lower-left corner of the
   arrival-vs-saturation plane — **tight arrival AND low saturation** — is **empty**. The two
   levers (weight, gain) trade along a *single* Pareto frontier; neither breaks through it.
   **The trade-off is fundamental** (this is a valid finding, per the brief).

2. **But the current balanced default `5k:5k` is itself Pareto-dominated.** Two new points beat
   it on both axes at once: `5k:12k Kp3` (arrival 10.7 % at **0.0 %** sat) and `5k:20k Kp3`
   (arrival **8.2 %** at **1.5 %** sat). I.e. raising the torso-pose weight *and* softening the
   gain tightens arrival from 11.4 %→8.2 % at essentially the same (negligible) saturation as
   today. A pure weight sweep would have missed this — it only appears in the 2-axis view.

3. **Gain decouples saturation from weight.** At fixed torso-pose weight (20k), Kp 6→4→3 cuts
   τ_w-sat 11.9 %→7.5 %→1.5 % while arrival only loosens 4.7 %→6.6 %→8.2 %. Weight is the
   *arrival* dial; Kp is the *saturation* dial. The 24 % TP-dom saturation is therefore partly
   gain-aggressiveness, not pure weight — but softening the gain costs arrival, so it cannot
   deliver the 2–3 % target cheaply.

**The choice of working point is the review session's call — NOT auto-selected here** (see §5).

---

## 1. Method

| lever | parameter | values | role |
|---|---|---|---|
| weight | `alpha_torso_pose` (mom fixed 5000) | 5k, 8k, 12k, 20k | task **priority** |
| gain | `ss_Kp_torso` / `ss_Kd_torso` | Kp ∈ {6,4,3}, Kd = 5·Kp/6 (5, 3.33, 2.5) | task **aggressiveness** |

The 6-D torso-pose task commands `a_t_des = a_ff + Kp·e6 + Kd·(v_ref − v_act)`
(`wholebody_qp.py:636`), so Kp sets the stiffness of the *commanded* torso acceleration. Kd is
scaled with Kp to hold the damping character roughly fixed. Gains exposed via the new
`--ss-kp-torso` / `--ss-kd-torso` diag flags (defaults 6.0/5.0 = config; no-flag bit-identical,
§6). Arrival measured vs the **geometric** torso target `p_t1` = FK-torso @ planned `q_end` =
`[0.290, −0.761, −0.336] m`, `|p_t1 − p_t0| = 125.5 mm`. Anchors `bal 5k:5k` and `TP-dom
0.5k:30k` carried from Phase-2.1. **All 8 new runs DOCK** (4.89–4.98 mm < 5 mm gate); the
EE-yield bound (M-dom) does not appear because mom stays at 5000.

## 2. Pareto table (sorted by arrival, tightest first)

| point | mom | tp | Kp | arrival mm | arrival % | τ_w-sat % | jerk_rms | torso_acc_pk | dock mm |
|---|---|---|---|---|---|---|---|---|---|
| TP-dom .5k:30k | 500 | 30000 | 6 | 4.16 | **3.32** | **23.79** | 0.127 | 0.095 | 4.97 |
| 5k:20k | 5000 | 20000 | 6 | 5.90 | 4.70 | 11.94 | 0.113 | 0.093 | 4.92 |
| TPdom Kp3 | 500 | 30000 | 3 | 6.38 | 5.09 | 15.67 | 0.114 | 0.090 | 4.89 |
| 5k:20k Kp4 | 5000 | 20000 | 4 | 8.30 | 6.61 | 7.50 | 0.100 | 0.087 | 4.89 |
| 5k:12k | 5000 | 12000 | 6 | 8.89 | 7.09 | 10.31 | 0.105 | 0.092 | 4.98 |
| 5k:20k Kp3 | 5000 | 20000 | 3 | 10.25 | 8.17 | 1.52 | 0.089 | 0.080 | 4.91 |
| 5k:8k | 5000 | 8000 | 6 | 11.73 | 9.35 | 9.12 | 0.098 | 0.091 | 4.93 |
| 5k:12k Kp4 | 5000 | 12000 | 4 | 12.23 | 9.75 | 2.35 | 0.088 | 0.082 | 4.97 |
| 5k:12k Kp3 | 5000 | 12000 | 3 | 13.48 | 10.74 | **0.00** | 0.073 | 0.072 | 4.97 |
| bal 5k:5k | 5000 | 5000 | 6 | 14.27 | 11.37 | 1.32 | 0.090 | 0.089 | 4.95 |

Full metrics (CoM-track, h_w∞, Ḣ_s, τ_peak, QP p50/p99): `Misc/runs/phase2_2_tuning_report/tuning_metrics.md`.

## 3. The Pareto frontier (non-dominated set, minimise BOTH arrival and saturation)

| # | point | τ_w-sat % | arrival % | note |
|---|---|---|---|---|
| 1 | **5k:12k Kp3** | 0.00 | 10.74 | zero-saturation endpoint |
| 2 | **5k:20k Kp3** | 1.52 | 8.17 | envelope-clean knee — *dominates balanced* |
| 3 | **5k:20k Kp4** | 7.50 | 6.61 | mid compromise |
| 4 | **5k:20k** (Kp6) | 11.94 | 4.70 | tight arrival, moderate sat |
| 5 | **TP-dom 0.5k:30k** | 23.79 | 3.32 | tightest arrival, heavy sat |

`bal 5k:5k` (1.32 %, 11.37 %) is **off** the frontier — dominated by points 1 and 2. Four of the
five frontier points are the **torso-pose=20k** family with the gain as the slider, plus the
zero-sat 12k endpoint and the TP-dom tight-arrival endpoint.

## 4. Plots (`Misc/runs/phase2_2_tuning_report/`)

- **`pareto_arrival_vs_sat.png`** — THE trade-off. The 2–3 % target band (green) is reached only
  by TP-dom at the far right (24 % sat); the low-saturation/tight-arrival corner is empty.
- **`two_levers.png`** — weight = arrival dial (left, monotone down); Kp = saturation dial (right,
  the Kp=3 line sits near 0 % at every weight). Visual proof of the decoupling.
- **`best_candidate_trajectory.png`** — `5k:20k Kp4` (the script's sat<10 % best-arrival pick):
  torso x/y/z position tracking is **smooth and monotone, no jitter/overshoot**, arriving near
  `p_t1`. The bottom τ_w panel shows the *cost* honestly — wheel torque chatters against the 5 N·m
  limit ~7.5 % of samples, vs the smooth baseline hump. Torso-trajectory **aggressiveness is low**
  (jerk_rms is among the smallest in the grid); the binding cost is wheel saturation, not torso
  jitter.

## 5. Decision — candidate working points (REVIEW DECIDES)

Per the brief, the working point is **not** selected here; the choice trades paper-narrative
torso precision against the τ_w envelope. The frontier offers three archetypes:

- **A — envelope-clean (recommended floor): `5k:20k Kp3`** → arrival **8.2 %** (10.3 mm), sat
  **1.5 %**. Strictly better than today's balanced default (11.4 % at 1.3 %) — tighter arrival at
  the same negligible saturation. If the envelope must stay pristine, this dominates balanced and
  should replace it regardless of the rest of the decision. (For *zero* saturation, `5k:12k Kp3`:
  10.7 % at 0.0 %.)
- **B — mid compromise: `5k:20k Kp4`** (6.6 % / 7.5 %) or **`5k:20k` Kp6** (4.7 % / 11.9 %). The
  brief's "5–6 % arrival at low-ish saturation" lives here.
- **C — tight arrival: TP-dom `0.5k:30k`** (3.3 % / 23.8 %). The brief's "2–3 % at higher
  saturation". The *only* route to ~3 %, at the full envelope cost.

There is no point at ≤3 % arrival with <10 % saturation. If the paper needs ≤3 % torso arrival,
it must accept heavy τ_w-saturation (and should frame the 24 % as execution cost, not plan
binding — see §6); otherwise B or A is the honest sweet spot.

## 6. Envelope fidelity, smoothness, provenance

- **Envelope is KEPT (execution cost, not plan binding):** realized Ḣ_s ≈ planned across the whole
  grid (e.g. `Hdot_real_pk` vs `Hdot_plan_pk`: 0.64/0.64 at 5k:20k, 0.59/0.59 at TP-dom, 0.96/0.96
  at balanced). The τ_w-saturation is the QP/wheels working to realise the commanded torso accel,
  **not** the NMPC plan demanding excess momentum. Softening Kp lowers that execution cost.
- **Smoothness:** torso jerk_rms 0.073–0.127 and accel_pk 0.072–0.095 m/s² across the grid — all
  small; **softer gain → lower jerk** (Kp3 ≈ 0.073–0.089 < Kp6 ≈ 0.09–0.11 < TP-dom 0.127). No
  point shows harsh torso acceleration. `tau_peak` 7.7–9.7 N·m (< 20). QP p99 ≤ 104 ms.
- **No default changed; no architecture change.** Gains exposed as diag flags only. **No-flag
  bit-identical:** a `--ss-two-task --n-steps 1` run with no weight/gain flags is byte-identical
  to `20e6031` (docks 4.95 = 4.95 mm, Δ = 0) — the gain-flag defaults (6.0/5.0) equal
  `_make_m7_config()` = SimConfig, verified MATCH. Out of scope (per brief): no 5-step, no gate,
  no new default committed — Phase 3 runs at the working point the review picks.
