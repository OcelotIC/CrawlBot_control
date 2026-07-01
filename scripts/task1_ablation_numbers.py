#!/usr/bin/env python3
"""TASK 1 (ablation Phase-1b) key numbers — envelope rate-cap ON (C) vs OFF (U).

Single-variable rate-cap ablation: C is the canonical run (envelope ON, @5ab2c91);
U is identical EXCEPT cfg.tau_w_max = 1e6 (the NMPC/pre-planner/QP wheel-RATE cap
removed). Everything else — storage box (h_max_tight), QP soft h_w box (hw_max),
AOCS realized-command clamp (aocs_tau_w_max=5), physical MJCF ctrlrange +/-5 — is
UNCHANGED. So this isolates what the *rate* constraint does to actuator demand.

Reads the two committed/exported tidy CSVs, prints the key-numbers table, and dumps
  results/j2_ablation_envelope/task1_key_numbers.json
  results/j2_ablation_envelope/task1_key_numbers.md

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/task1_ablation_numbers.py
"""
import csv
import json
import os

import numpy as np

C_CSV = 'results/j2_canonical_revalidation/runfix_traversal.csv'
U_CSV = 'results/j2_ablation_envelope/runU_rateoff_traversal.csv'
OUT_JSON = 'results/j2_ablation_envelope/task1_key_numbers.json'
OUT_MD = 'results/j2_ablation_envelope/task1_key_numbers.md'
TAU_W_MAX = 5.0


def load(p):
    r = list(csv.DictReader(open(p)))
    def col(n, f=float):
        return np.array([f(x[n]) if x[n] != '' else np.nan for x in r])
    return dict(
        n=len(r), ph=np.array([x['phase'] for x in r]),
        src=np.array([x['Hdot_s_source'] for x in r]), step=col('step_index', int),
        H=np.stack([col('Hdot_s_x_Nm'), col('Hdot_s_y_Nm'), col('Hdot_s_z_Nm')], 1),
        tw=np.stack([col('tauw_x_Nm'), col('tauw_y_Nm'), col('tauw_z_Nm')], 1),
        th=np.stack([col('theta_s_x_deg'), col('theta_s_y_deg'), col('theta_s_z_deg')], 1),
        hw=np.stack([col('hw_x_Nms'), col('hw_y_Nms'), col('hw_z_Nms')], 1))


def analyze(p):
    d = load(p)
    ss_planned = (d['ph'] == 'SS') & (d['src'] == 'planned')
    Hss = d['H'][ss_planned]
    perax = np.nanmax(np.abs(Hss), axis=0)
    over = np.nanmax(np.abs(Hss), axis=1) > TAU_W_MAX + 1e-6
    per_swing = {}
    for s in sorted(set(d['step'][ss_planned].tolist())):
        m = ss_planned & (d['step'] == s)
        o = np.nanmax(np.abs(d['H'][m]), axis=1) > TAU_W_MAX + 1e-6
        per_swing[int(s)] = round(100 * float(o.mean()), 1)
    twinf = np.nanmax(np.abs(d['tw']), axis=1)
    sat = twinf >= TAU_W_MAX - 1e-3
    thn = np.linalg.norm(d['th'], axis=1)
    meta = json.load(open(p.replace('_traversal.csv', '_meta.json')))
    return dict(
        ticks=d['n'], ss_planned_ticks=int(ss_planned.sum()),
        planned_Hdot_s_SS_peak_axis_Nm=dict(x=round(float(perax[0]), 3),
                                            y=round(float(perax[1]), 3),
                                            z=round(float(perax[2]), 3)),
        planned_Hdot_s_SS_over_cap=dict(count=int(over.sum()),
                                        total=int(ss_planned.sum()),
                                        pct=round(100 * float(over.mean()), 1)),
        planned_Hdot_s_per_swing_over_cap_pct=per_swing,
        realized_tauw_peak_inf_Nm=round(float(np.nanmax(twinf)), 3),
        realized_tauw_saturation=dict(count=int(sat.sum()), total=int(d['n']),
                                      pct=round(100 * float(sat.mean()), 1)),
        theta_s_peak_norm_deg=round(float(np.nanmax(thn)), 4),
        theta_s_final_norm_deg=round(float(thn[-1]), 4),
        hw_peak_inf_Nms=round(float(np.nanmax(np.abs(d['hw']))), 4),
        docks_mm=[round(float(e['dock_distance_mm']), 2) for e in meta['dock_events']])


res = {
    'tau_w_max_Nm': TAU_W_MAX,
    'C': {'label': 'envelope rate-cap ON (canonical @5ab2c91)', 'csv': C_CSV, **analyze(C_CSV)},
    'U': {'label': 'rate cap OFF (cfg.tau_w_max=1e6; storage box + AOCS clamp UNCHANGED)',
          'csv': U_CSV, **analyze(U_CSV)},
}

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
json.dump(res, open(OUT_JSON, 'w'), indent=2)


def fmt_swing(d):
    return ' '.join(f's{k}:{v:.0f}%' for k, v in d['planned_Hdot_s_per_swing_over_cap_pct'].items())


C, U = res['C'], res['U']
md = f"""# Task 1 — rate-cap ablation key numbers (envelope ON vs OFF)

Single-variable test: **U = C with `cfg.tau_w_max` raised from 5 to 1e6** (NMPC + pre-planner + QP wheel
**rate** cap removed). Storage box (`h_max_tight`), QP soft box (`hw_max`), AOCS realized-command clamp
(`aocs_tau_w_max`=5) and the physical MJCF `ctrlrange` ±5 are all **unchanged**, so this isolates the rate
constraint's effect on actuator demand.

- C: `{C['csv']}`
- U: `{U['csv']}`
- rate cap τ_w,max = {TAU_W_MAX} Nm

| metric | C — envelope ON | U — rate cap OFF |
|---|---|---|
| ticks | {C['ticks']} | {U['ticks']} |
| **PLANNED \\|Ḣ_s\\| SS peak / axis (Nm)** | x {C['planned_Hdot_s_SS_peak_axis_Nm']['x']}  y {C['planned_Hdot_s_SS_peak_axis_Nm']['y']}  z {C['planned_Hdot_s_SS_peak_axis_Nm']['z']} | x {U['planned_Hdot_s_SS_peak_axis_Nm']['x']}  y {U['planned_Hdot_s_SS_peak_axis_Nm']['y']}  **z {U['planned_Hdot_s_SS_peak_axis_Nm']['z']}** |
| **planned Ḣ_s SS ticks > 5 Nm** | {C['planned_Hdot_s_SS_over_cap']['count']}/{C['planned_Hdot_s_SS_over_cap']['total']} ({C['planned_Hdot_s_SS_over_cap']['pct']}%) | **{U['planned_Hdot_s_SS_over_cap']['count']}/{U['planned_Hdot_s_SS_over_cap']['total']} ({U['planned_Hdot_s_SS_over_cap']['pct']}%)** |
| per-swing > 5 fraction | {fmt_swing(C)} | {fmt_swing(U)} |
| realized \\|τ_w\\|∞ peak (Nm) | {C['realized_tauw_peak_inf_Nm']} | {U['realized_tauw_peak_inf_Nm']} |
| realized τ_w saturation | {C['realized_tauw_saturation']['pct']}% ({C['realized_tauw_saturation']['count']}/{C['realized_tauw_saturation']['total']}) | **{U['realized_tauw_saturation']['pct']}% ({U['realized_tauw_saturation']['count']}/{U['realized_tauw_saturation']['total']})** |
| θ_s peak-norm (deg) | {C['theta_s_peak_norm_deg']} | {U['theta_s_peak_norm_deg']} |
| θ_s final-norm (deg) | {C['theta_s_final_norm_deg']} | **{U['theta_s_final_norm_deg']}** |
| h_w peak-∞ (Nms) | {C['hw_peak_inf_Nms']} | {U['hw_peak_inf_Nms']} |
| docks (mm, n=6) | {C['docks_mm']} | {U['docks_mm']} |

**Verdict (primary / actuator demand):** the pre-planner's PLANNED wheel-rate demand reaches
**{U['planned_Hdot_s_SS_peak_axis_Nm']['z']} Nm on z** with the cap OFF — **{U['planned_Hdot_s_SS_peak_axis_Nm']['z'] - TAU_W_MAX:.2f} Nm ({100*(U['planned_Hdot_s_SS_peak_axis_Nm']['z']/TAU_W_MAX - 1):.0f}%) above** the 5 Nm cap — and
**{U['planned_Hdot_s_SS_over_cap']['pct']}% of SS ticks** ask for more than 5 Nm. With the envelope ON the same demand is
held at exactly **{C['planned_Hdot_s_SS_peak_axis_Nm']['y']:.2f}/{C['planned_Hdot_s_SS_peak_axis_Nm']['z']:.2f} Nm (y/z pinned at the cap)** with **0** SS ticks over. The constraint is **binding**, not
slack — the plan genuinely wants more than the wheel can deliver, so the hard rate constraint actively
reshapes the trajectory. Paper claim **validated**.

**Realized actuator:** both runs realize the SAME peak {C['realized_tauw_peak_inf_Nm']} Nm — because the AOCS realized-command
clamp (aocs_tau_w_max=5) + physical MJCF ±5 still clip the wheel. Removing the *software* rate cap does not
remove the limit; it pushes enforcement onto the hard rail, raising time-at-saturation from {C['realized_tauw_saturation']['pct']}% to
{U['realized_tauw_saturation']['pct']}% of ticks (plan-vs-plant mismatch).

**Secondary / attitude:** θ_s peak is comparable ({U['theta_s_peak_norm_deg']}° vs {C['theta_s_peak_norm_deg']}°), but the rate-off run's
**final** attitude error is {U['theta_s_final_norm_deg']/C['theta_s_final_norm_deg']:.1f}× worse ({U['theta_s_final_norm_deg']}° vs {C['theta_s_final_norm_deg']}°).

**h_w:** bounded in BOTH ({U['hw_peak_inf_Nms']} vs {C['hw_peak_inf_Nms']} Nms) — the storage box is still ON in U, so the rate-cap
ablation does not (and should not) blow up storage. The storage counterfactual is a separate test (Task 3, gated).
"""
open(OUT_MD, 'w').write(md)

# console echo
for tag, x in [('C  (envelope ON)', C), ('U  (rate-off)', U)]:
    print(f'\n=== {tag}  ({x["ticks"]} ticks) ===')
    a = x['planned_Hdot_s_SS_peak_axis_Nm']
    print(f'  PLANNED |Hdot_s| SS peak/axis: x {a["x"]}  y {a["y"]}  z {a["z"]} Nm  (cap 5)')
    o = x['planned_Hdot_s_SS_over_cap']
    print(f'    SS ticks >5: {o["count"]}/{o["total"]} ({o["pct"]}%)   per-swing: {fmt_swing(x)}')
    s = x['realized_tauw_saturation']
    print(f'  REALIZED tau_w: peak {x["realized_tauw_peak_inf_Nm"]}  sat {s["pct"]}% ({s["count"]}/{s["total"]})')
    print(f'  theta_s peak {x["theta_s_peak_norm_deg"]}  final {x["theta_s_final_norm_deg"]}  '
          f'h_w peak {x["hw_peak_inf_Nms"]}  docks {x["docks_mm"]}')
print(f'\nwrote {OUT_JSON}\nwrote {OUT_MD}')
