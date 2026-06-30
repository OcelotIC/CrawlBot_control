# INTERNAL — distributions (not maxima) for C5 h_w and C2 torso (canonical)

Branch `j2/ds-active-rework`. Read-only analysis of the committed 5-step canonical export
(`results/j2_figdata/runA_traversal.csv`, commit `a24db03`) + a supplementary 6-step run
(`run7step_traversal.csv`, this commit) to qualify the C5 `h_w∞ = 4.930` and C2 `23.0 mm` **maxima** —
is each an isolated transient or a sustained/accumulating regime? `h_w∞` = `max(|hw_x|,|hw_y|,|hw_z|)` per
tick (the C5-gated quantity). C2 = `‖torso_pos − ref‖` (Euclidean, mm) on SS rows. **No `crawlbot/` change, no merge.**

## 1. C5 h_w — an ISOLATED per-swing transient, not sustained
From `runA_traversal.csv` (905 ticks):
- **Occurrences:** `‖h_w‖∞ > 4.5` = **6 ticks (0.66 %)**; `> 4.7` = 5; `> 4.9` = **2 ticks (0.22 %)** — **all 6 in SS, all in step 4**, zero in any DS.
- **Peak shape:** the max 4.930 (tick 688, t=29.34 s, SS, step 4; per-axis hw=[−0.70, 2.28, **−4.93**]) is a **smooth ~1 s hump**: 2.84 → 4.93 → 3.22 over ±10 ticks. Not a plateau, not chatter — one swing's wheel loading that then unloads.
- **Distribution `‖h_w‖∞`:** full-run median **1.11**, p95 2.93, p99 4.21, max 4.93; SS-only median 2.17, p99 4.84; DS-only median 0.89, max 1.71. ⇒ the 4.93 is the p100 tail, not the operating level.

## 2. The per-swing PEAK scales with crawl distance — but ONLY on the b-arm, and concavely
Per-step `‖h_w‖∞` max, by **swing arm** (step k's swing arm = the arm that docks at step k, from `dock_events`):

| step | swing | \|r_com\|@peak | h_w_peak | torso-pos SS peak |
|---|---|---|---|---|
| 0 | b3 | 0.686 m | **2.359** | 9.44 mm |
| 1 | a3 | 0.826 m | 2.934 | 11.72 mm |
| 2 | b4 | 1.074 m | **4.015** | 18.78 mm |
| 3 | a4 | 1.383 m | 2.970 | 11.98 mm |
| 4 | b5 | 1.728 m | **4.930** | 23.05 mm |
| 5 | a5 | 1.991 m | 2.909 (6-step) | 11.43 mm (6-step) |

- **b-arm swings (0/2/4): 2.36 → 4.02 → 4.93** — grow with crawl distance. Linear fit `h_w ≈ 2.36·|r_com| + 1.03`
  (slope matches the SS orbital scaling ≈ 2.14·|r_com|).
- **But the growth is CONCAVE / decelerating:** increments **+1.66, +0.92**; the 3rd point (4.93 @ 1.73 m) sits
  **below** the linear extrapolation (5.10), i.e. it is bending toward an asymptote near ~5, not blowing through
  it. The linear fit would cross 5.0 at |r_com|=1.69 m; the actual b5 at 1.73 m is still 4.93 (< 5).
- **a-arm swings (1/3/5): 2.93, 2.97, 2.91** — **flat**, no r_com dependence (a-arm swing loads the wheels about a
  benign axis; r_com more than doubled 0.83→1.99 m with no change). The 6-step run's new **a5 @ 1.99 m = 2.909**
  confirms the a-arm is bounded at larger crawl distance.
- **Baseline does NOT accumulate:** per-step median `[0.28, 1.02, 1.12, 1.38, 0.81, 0.92]` — no growing floor; the
  wheels unload between swings (DS median 0.89). Only the b-swing *peak* grows, not a secular drift.

## 3. C2 torso — same story, same transient
SS `‖e_torso_pos‖`: median **7.4–7.8 mm**, p95 ~18 mm, p99 ~22.5 mm, **max 23.05 mm** at tick 691 (t=29.64 s,
SS, step 4) — the **same step-4 b-swing** as the h_w peak. Per-step SS max tracks h_w exactly: b-swings
[9.44, 18.78, 23.05] grow, a-swings [11.72, 11.98, 11.43] flat. The 23 mm is a p100 transient, not loose tracking
throughout (median ~8 mm).

## 4. Could a longer traversal cross ±5? — UNCONFIRMED (planner caps at 6 swings without a code change)
The decisive test is another b-swing beyond b5 (b6, at |r_com| ≈ 2.2–2.5 m). The gait planner caps the
hardcoded `start=(2,2)` traversal at **6 swings** (`n_steps = len(seq.swing_targets)`, `sim_loop.py:253`; ending
a5) — reaching b6 needs a different `start_a/start_b` (`sim_loop.py:494`, a code change the brief forbids). So:
- the b-swing peak **grows with crawl distance** (real, orbital-tied) **but concavely, toward ~5** — at b5 it is
  **4.93 = 98.6 % of the ±5 hardware budget**;
- whether a further b-swing **crosses ±5** is **not confirmed** here. The concave trend suggests near-saturation,
  but the margin (1.4 %) is razor-thin.

## Verdict (for the C5-blocker decision)
The C5 4.930 is **neither** a misleading "one isolated tick" **nor** a sustained 98.6 %-of-hardware regime:
- It is an **isolated per-swing transient** (0.66 % of ticks, ~1 s hump, unloads; operating median ~1–2).
  "The system rides 98.6 % of hardware" **overstates** it as a steady state.
- **But** the b-swing peak **scales with crawl distance** (orbital ∝ r_com), reaching 98.6 % at r_com=1.73 m and
  bending concavely toward ~5. This is a **real, geometry-driven limitation** on the b-arm, not a numerical fluke.
- ⇒ C5 is a **bounded-but-thin-margin** item: on the validated 5–6-step traversal it stays < 5 (max 4.93), the
  growth is decelerating, but the per-swing peak does grow and the margin is 1.4 %. A longer traversal (or the
  largest-r_com b-swing) should be confirmed before claiming a hard ±5 guarantee; or add wheel-desaturation /
  retune the SS momentum weight to widen the margin. **Not a sustained-regime blocker; not a non-issue either.**

Reproduce: `MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/dist_hw_torso.py <csv>` over the committed
`results/j2_figdata/runA_traversal.csv` (5-step) or `results/j2_canonical_revalidation/run7step_traversal.csv`
(6-step). The b-swing scaling fit is an inline numpy polyfit over the per-step peaks above. **No code change, no merge.**
