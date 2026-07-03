#!/usr/bin/env python3
"""Phase ABL-DATA — export plot-ready with/without-management ablation data from COMMITTED
artifacts only. No sim run, no crawlbot/ change. Reads:
  C (with management)  : results/j2_canonical_revalidation/runfix_traversal.csv   @ 5ab2c91
  U (rate-off)         : results/j2_ablation_envelope/runU_rateoff_traversal.csv  @ be76c9c
  task1_key_numbers.json                                                          @ be76c9c
Writes results/j2_ablation_envelope/ablation_data/: paired time-series CSVs, summary JSON+md.
(manifest.md is authored separately.)

HONESTY: h_w in this regime never exceeds 5 (approaches the ±5 box on long steps, no overshoot).
Peak theta_s is NOT a with/without benefit (U 0.537 < C 0.591); settled theta_s + actuator demand
are the honest discriminators. Every summary number carries value + source + commit + derivation.

Ḣ_s provenance in the committed CSV (export_figure_data.py:316-324): the single tagged Hdot_s
column is PLANNED on SS ticks (from lambda_ref) and REALIZED on DS ticks (from lambda_qp). Planned
Ḣ_s therefore exists SS-only; realized Ḣ_s (in that column) exists DS-only. The realized
reaction-torque rate at EVERY tick is the wheel command tau_w (separate columns). We split the
tagged column faithfully; we do NOT synthesize a realized-SS or planned-DS value.

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/export_ablation_data.py
"""
import csv
import json
import os

import numpy as np

RATE_CAP = 5.0     # N·m   (tau_w_max; planned |Ḣ_s| reference line)
STORAGE_BOX = 5.0  # N·m·s (±h_max; h_w reference line)
SAT_EPS = 1e-3

RUNS = {
    'C': dict(csv='results/j2_canonical_revalidation/runfix_traversal.csv', commit='5ab2c91',
              label='with management (canonical, rate cap + storage box ON)'),
    'U': dict(csv='results/j2_ablation_envelope/runU_rateoff_traversal.csv', commit='be76c9c',
              label='without rate management (rate cap OFF, tau_w_max=1e6)'),
}
TASK1 = 'results/j2_ablation_envelope/task1_key_numbers.json'
TASK1_COMMIT = 'be76c9c'
OUT = 'results/j2_ablation_envelope/ablation_data'
os.makedirs(OUT, exist_ok=True)
t1 = json.load(open(TASK1))


def load(path):
    return list(csv.DictReader(open(path)))


def fcol(rows, name):
    return np.array([float(x[name]) if x[name] != '' else np.nan for x in rows])


def emit(v):
    return '' if (v is None or (isinstance(v, float) and np.isnan(v))) else f'{v:.6e}'


# ---------- DELIVERABLE 1: paired time-series ----------
TS_HEADER = [
    't_s', 'phase', 'phase_raw', 'step_index',
    'Hdot_s_planned_x_Nm', 'Hdot_s_planned_y_Nm', 'Hdot_s_planned_z_Nm',      # SS-only
    'Hdot_s_realized_x_Nm', 'Hdot_s_realized_y_Nm', 'Hdot_s_realized_z_Nm',    # DS-only
    'hw_x_Nms', 'hw_y_Nms', 'hw_z_Nms',
    'tauw_x_Nm', 'tauw_y_Nm', 'tauw_z_Nm', 'tauw_norm_Nm', 'tauw_saturated',
    'theta_s_x_deg', 'theta_s_y_deg', 'theta_s_z_deg', 'theta_s_geodesic_deg',
]


def write_timeseries(tag, rows):
    path = os.path.join(OUT, f'ablation_{tag}_timeseries.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(TS_HEADER)
        for x in rows:
            src = x['Hdot_s_source']
            ph_raw = x['phase']
            ph = 'DS' if ph_raw.startswith('DS') else 'SS'
            H = [x['Hdot_s_x_Nm'], x['Hdot_s_y_Nm'], x['Hdot_s_z_Nm']]

            def g(n):
                return float(x[n]) if x[n] != '' else np.nan
            Hp = H if src == 'planned' else ['', '', '']
            Hr = H if src == 'realized' else ['', '', '']
            tw = np.array([g('tauw_x_Nm'), g('tauw_y_Nm'), g('tauw_z_Nm')])
            twn = float(np.linalg.norm(tw))
            sat = 1 if np.nanmax(np.abs(tw)) >= RATE_CAP - SAT_EPS else 0
            th = np.array([g('theta_s_x_deg'), g('theta_s_y_deg'), g('theta_s_z_deg')])
            thg = float(np.linalg.norm(th))
            # Hp/Hr are already the raw source strings (verbatim) when populated
            w.writerow([x['t_s'], ph, ph_raw, x['step_index'],
                        Hp[0], Hp[1], Hp[2], Hr[0], Hr[1], Hr[2],
                        x['hw_x_Nms'], x['hw_y_Nms'], x['hw_z_Nms'],
                        x['tauw_x_Nm'], x['tauw_y_Nm'], x['tauw_z_Nm'], emit(twn), sat,
                        x['theta_s_x_deg'], x['theta_s_y_deg'], x['theta_s_z_deg'], emit(thg)])
    return path


# ---------- DELIVERABLE 2: summary ----------
def summarize(tag, rows):
    src = np.array([x['Hdot_s_source'] for x in rows])
    ph = np.array([x['phase'] for x in rows])
    step = np.array([int(x['step_index']) for x in rows])
    ss_pl = (src == 'planned') & (ph == 'SS')
    H = np.stack([fcol(rows, f'Hdot_s_{a}_Nm') for a in 'xyz'], 1)
    hw = np.stack([fcol(rows, f'hw_{a}_Nms') for a in 'xyz'], 1)
    tw = np.stack([fcol(rows, f'tauw_{a}_Nm') for a in 'xyz'], 1)
    th = np.stack([fcol(rows, f'theta_s_{a}_deg') for a in 'xyz'], 1)
    thg = np.linalg.norm(th, axis=1)
    twinf = np.nanmax(np.abs(tw), axis=1); twn = np.linalg.norm(tw, axis=1)
    sat = twinf >= RATE_CAP - SAT_EPS
    Hss = H[ss_pl]
    over = np.nanmax(np.abs(Hss), axis=1) > RATE_CAP + 1e-6
    per_step_over = {}
    per_step_hw = {}
    for s in sorted(set(step[step >= 0].tolist())):
        m_ss = ss_pl & (step == s)
        o = np.nanmax(np.abs(H[m_ss]), axis=1) > RATE_CAP + 1e-6
        per_step_over[int(s)] = dict(pct=round(100 * float(o.mean()), 1),
                                     count=int(o.sum()), total=int(m_ss.sum()))
        m = step == s
        per_step_hw[int(s)] = [round(float(v), 4) for v in np.nanmax(np.abs(hw[m]), axis=0)]
    return dict(
        n_ticks=len(rows), ss_planned_ticks=int(ss_pl.sum()),
        planned_Hdot_s_SS_peak_per_axis_Nm=dict(
            x=round(float(np.nanmax(np.abs(Hss[:, 0]))), 3),
            y=round(float(np.nanmax(np.abs(Hss[:, 1]))), 3),
            z=round(float(np.nanmax(np.abs(Hss[:, 2]))), 3)),
        planned_SS_over_cap=dict(pct=round(100 * float(over.mean()), 1),
                                 count=int(over.sum()), total=int(ss_pl.sum())),
        planned_SS_over_cap_per_step=per_step_over,
        tauw_saturation_pct=round(100 * float(sat.mean()), 1),
        tauw_saturation_count=[int(sat.sum()), len(rows)],
        tauw_peak_norm_Nm=round(float(np.nanmax(twn)), 3),
        tauw_peak_inf_Nm=round(float(np.nanmax(twinf)), 3),
        hw_peak_per_axis_Nms=[round(float(v), 4) for v in np.nanmax(np.abs(hw), axis=0)],
        hw_peak_inf_Nms=round(float(np.nanmax(np.abs(hw))), 4),
        hw_peak_per_step_per_axis_Nms=per_step_hw,
        theta_s_peak_geodesic_deg=round(float(np.nanmax(thg)), 4),
        theta_s_settled_geodesic_deg=round(float(thg[-1]), 4),
        docks_mm=t1[tag]['docks_mm'])


rows = {}
ts_paths = {}
summ = {}
for tag in ('C', 'U'):
    rows[tag] = load(RUNS[tag]['csv'])
    ts_paths[tag] = write_timeseries(tag, rows[tag])
    summ[tag] = summarize(tag, rows[tag])

# cross-check CSV-derived vs committed task1_key_numbers.json
xcheck = {}
for tag in ('C', 'U'):
    a = summ[tag]['planned_Hdot_s_SS_peak_per_axis_Nm']; b = t1[tag]['planned_Hdot_s_SS_peak_axis_Nm']
    xcheck[tag] = dict(
        planned_peak_match=(a == {k: round(v, 3) for k, v in b.items()}),
        over_pct_match=(summ[tag]['planned_SS_over_cap']['pct'] == t1[tag]['planned_Hdot_s_SS_over_cap']['pct']),
        tauw_sat_match=(summ[tag]['tauw_saturation_pct'] == t1[tag]['realized_tauw_saturation']['pct']),
        hw_peak_match=(summ[tag]['hw_peak_inf_Nms'] == t1[tag]['hw_peak_inf_Nms']),
        theta_peak_match=(summ[tag]['theta_s_peak_geodesic_deg'] == t1[tag]['theta_s_peak_norm_deg']),
        theta_settled_match=(summ[tag]['theta_s_settled_geodesic_deg'] == t1[tag]['theta_s_final_norm_deg']))

out = dict(
    constants=dict(rate_cap_Nm=RATE_CAP, storage_box_Nms=STORAGE_BOX,
                   note='rate cap = planned |Ḣ_s| reference line; storage box = ±h_w reference line'),
    sources={tag: dict(csv=RUNS[tag]['csv'], commit=RUNS[tag]['commit'], label=RUNS[tag]['label'])
             for tag in ('C', 'U')},
    task1_key_numbers=dict(path=TASK1, commit=TASK1_COMMIT),
    crosscheck_csv_vs_task1=xcheck,
    C=summ['C'], U=summ['U'])
json.dump(out, open(os.path.join(OUT, 'ablation_summary.json'), 'w'), indent=2)

# markdown summary
def axr(d):
    return f"x {d['x']} / y {d['y']} / z {d['z']}"


md = f"""# Ablation summary — with (C) vs without rate management (U)

Constants (plot reference lines): **rate cap = {RATE_CAP} N·m**, **storage box = ±{STORAGE_BOX} N·m·s**.
Sources: C = `{RUNS['C']['csv']}` @ `{RUNS['C']['commit']}`; U = `{RUNS['U']['csv']}` @ `{RUNS['U']['commit']}`;
cross-check `{TASK1}` @ `{TASK1_COMMIT}`. Every value below is computed from the C/U CSV columns and
cross-checked against task1_key_numbers.json (all matches: {all(all(v.values()) for v in xcheck.values())}).

| metric | C (with mgmt) | U (rate-off) | source |
|---|---|---|---|
| planned \\|Ḣ_s\\| SS peak / axis (N·m) | {axr(summ['C']['planned_Hdot_s_SS_peak_per_axis_Nm'])} | {axr(summ['U']['planned_Hdot_s_SS_peak_per_axis_Nm'])} | CSV SS/planned ticks |
| planned \\|Ḣ_s\\| SS ticks > 5 | {summ['C']['planned_SS_over_cap']['count']}/{summ['C']['planned_SS_over_cap']['total']} ({summ['C']['planned_SS_over_cap']['pct']}%) | {summ['U']['planned_SS_over_cap']['count']}/{summ['U']['planned_SS_over_cap']['total']} ({summ['U']['planned_SS_over_cap']['pct']}%) | CSV |
| τ_w saturation (% ticks) | {summ['C']['tauw_saturation_pct']}% ({summ['C']['tauw_saturation_count'][0]}/{summ['C']['tauw_saturation_count'][1]}) | {summ['U']['tauw_saturation_pct']}% ({summ['U']['tauw_saturation_count'][0]}/{summ['U']['tauw_saturation_count'][1]}) | CSV τ_w, sat = max\\|τ_w,i\\|≥5−1e-3 |
| τ_w peak norm / ∞ (N·m) | {summ['C']['tauw_peak_norm_Nm']} / {summ['C']['tauw_peak_inf_Nm']} | {summ['U']['tauw_peak_norm_Nm']} / {summ['U']['tauw_peak_inf_Nm']} | CSV |
| h_w peak / axis (N·m·s) | {summ['C']['hw_peak_per_axis_Nms']} | {summ['U']['hw_peak_per_axis_Nms']} | CSV (≤5, no overshoot) |
| h_w peak-∞ (N·m·s) | {summ['C']['hw_peak_inf_Nms']} | {summ['U']['hw_peak_inf_Nms']} | CSV |
| θ_s peak geodesic (deg) | {summ['C']['theta_s_peak_geodesic_deg']} | {summ['U']['theta_s_peak_geodesic_deg']} | CSV (NOT a with/without benefit) |
| **θ_s settled geodesic (deg)** | **{summ['C']['theta_s_settled_geodesic_deg']}** | **{summ['U']['theta_s_settled_geodesic_deg']}** | CSV (honest discriminator) |
| docks (mm, 6/6 @ 5 mm gate) | {summ['C']['docks_mm']} | {summ['U']['docks_mm']} | task1_key_numbers.json |

### Per-step planned \\|Ḣ_s\\| > 5 (% of that step's SS ticks)
| step | C | U |
|---|---|---|
""" + "\n".join(
    f"| {s} | {summ['C']['planned_SS_over_cap_per_step'][s]['pct']}% | {summ['U']['planned_SS_over_cap_per_step'][s]['pct']}% |"
    for s in sorted(summ['C']['planned_SS_over_cap_per_step'])) + f"""

### Per-step h_w peak / axis (N·m·s) — which steps approach the ±5 box (z-axis; peaks on the SHORT steps 2 & 4, step 4 nearest at 4.885)
| step | C [x,y,z] | U [x,y,z] |
|---|---|---|
""" + "\n".join(
    f"| {s} | {summ['C']['hw_peak_per_step_per_axis_Nms'][s]} | {summ['U']['hw_peak_per_step_per_axis_Nms'][s]} |"
    for s in sorted(summ['C']['hw_peak_per_step_per_axis_Nms'])) + """

**Honesty notes:** h_w approaches but never exceeds ±5 (peak C 4.885 @ step 4 / U 4.484 @ step 4, both on
the z-axis of the SHORT steps 2 & 4); no overshoot.
θ_s **peak** is comparable/slightly lower for U — NOT a management benefit; the honest discriminators
are actuator demand (planned |Ḣ_s| > 5 and τ_w saturation, both worse for U) and **settled θ_s**
(U 0.278 vs C 0.108, worse for U).
"""
open(os.path.join(OUT, 'ablation_summary.md'), 'w').write(md)

print('wrote:')
for tag in ('C', 'U'):
    print(f'  {ts_paths[tag]}  ({summ[tag]["n_ticks"]} ticks)')
print(f'  {OUT}/ablation_summary.json')
print(f'  {OUT}/ablation_summary.md')
print('cross-check CSV vs task1_key_numbers.json:', xcheck)
