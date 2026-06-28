#!/usr/bin/env python3
"""J2 step 4a characterization — AOCS re-activation in the inter-step DS loop.

Reads two run dirs (on=<dir> off=<dir>) and reports, for the inter-step DS
settles ONLY (masked via `inter_step_settles` time windows — so DWELL/terminal
DS and SS are excluded), the decisive A/B metrics:
  (1) θ_s drift per inter-step settle  — the invariant check (was uncorrected)
  (2) τ_w during the settles            — confirm non-zero, doing real work
  (3) ω_s (structure rate) during settles — what the AOCS damps
  (4) settle convergence                — n_steps / T_end vs T_settle / exit
  (5) h_w (whole-traversal + inter-step) — the flagged C5 interaction
RAW numbers; no pass/fail. C1-C5/dock/residual come from gate_phase3 separately.

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_aocs_interstep.py \
        on=results/aocs_on off=results/aocs_off
"""
import json
import os
import sys
import numpy as np


def load(run_dir):
    p = os.path.join(run_dir, 'sim_log.json')
    with open(p) as f:
        return json.load(f)


def interstep_mask(sl):
    """Boolean mask over log ticks selecting inter-step DS ticks (the
    _run_ds_passivity_loop settles), via the inter_step_settles windows."""
    t = np.asarray(sl['t'], dtype=float)
    ph = np.asarray(sl['phase'])
    mask = np.zeros(len(t), dtype=bool)
    settles = sl.get('inter_step_settles', []) or []
    eps = 1e-6
    for s in settles:
        lo, hi = float(s['t_start']), float(s['t_end'])
        mask |= (ph == 'DS') & (t > lo + eps) & (t <= hi + eps)
    return mask, settles


def per_settle_groups(sl, mask):
    """Contiguous index groups of inter-step ticks (one per settle)."""
    idx = np.where(mask)[0]
    if not len(idx):
        return []
    return np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)


def analyze(tag, sl):
    t = np.asarray(sl['t'], dtype=float)
    se = np.asarray(sl['struct_euler_deg'], dtype=float)
    tw = np.asarray(sl['tau_w'], dtype=float)
    om = np.asarray(sl.get('omega_struct', np.zeros((len(t), 3))), dtype=float)
    hwp = np.asarray(sl['hw_physical'], dtype=float)
    twn = np.linalg.norm(tw, axis=1)
    hwn = np.linalg.norm(hwp, axis=1)
    omn = np.linalg.norm(om, axis=1)

    mask, settles = interstep_mask(sl)
    groups = per_settle_groups(sl, mask)
    n_is = int(mask.sum())

    print(f"\n========== {tag} ==========")
    print(f"  inter-step ticks: {n_is} ({n_is*0.01:.2f}s) across {len(groups)} settles")

    # (2)+(3) tau_w and omega_s over inter-step ticks
    if n_is:
        print(f"  τ_w  over inter-step ticks: mean={twn[mask].mean():.4f} "
              f"max={twn[mask].max():.4f} Nm")
        print(f"  ω_s  over inter-step ticks: mean={omn[mask].mean():.3e} "
              f"max={omn[mask].max():.3e} rad/s")

    # (1) θ_s drift per inter-step settle + (4) settle convergence
    print("  per-settle:  Δθ_s(max-min)°        | n_steps  dur[s]  "
          "T_end/T_settle  exit")
    worst = 0.0
    # match groups to settles by order (both are time-ordered)
    for gi, g in enumerate(groups):
        dr = se[g].max(0) - se[g].min(0)
        worst = max(worst, float(dr.max()))
        s = settles[gi] if gi < len(settles) else {}
        ratio = (s.get('T_end', float('nan')) /
                 s.get('T_settle', float('nan'))
                 if s.get('T_settle') else float('nan'))
        twmax = twn[g].max()
        print(f"    settle {gi} (step {s.get('step_idx','?')}): "
              f"[{dr[0]:.4f},{dr[1]:.4f},{dr[2]:.4f}] max {dr.max():.4f}  "
              f"|τ_w|max={twmax:.3f}  | {int(s.get('n_steps',len(g))):4d}  "
              f"{len(g)*0.01:.2f}   {ratio:.2e}   {s.get('exit_reason','?')}")
    print(f"  WORST per-settle Δθ_s = {worst:.4f}°")

    # (5) h_w — whole traversal and inter-step window
    print(f"  h_w∞ whole-traversal = {hwn.max():.4f} Nms  "
          f"(C5 cap 4.5; physical ±5)")
    if n_is:
        print(f"  h_w∞ during inter-step settles = {hwn[mask].max():.4f} Nms")
    return {'worst_theta': worst, 'hw_inf': float(hwn.max()),
            'tw_mean': float(twn[mask].mean()) if n_is else 0.0,
            'tw_max': float(twn[mask].max()) if n_is else 0.0,
            'n_is': n_is}


def main(argv):
    runs = {}
    for a in argv:
        if '=' in a:
            k, v = a.split('=', 1)
            runs[k] = v
    if 'on' not in runs or 'off' not in runs:
        print("usage: audit_aocs_interstep.py on=<dir> off=<dir>")
        return 2
    print('=' * 70)
    print('AOCS inter-step re-activation — A/B (inter-step DS settles only)')
    print('=' * 70)
    res = {}
    for tag in ('off', 'on'):
        res[tag] = analyze(tag.upper(), load(runs[tag]))

    print('\n' + '=' * 70)
    print('SUMMARY (off = legacy hardcoded-zero; on = AOCS re-activated)')
    print('=' * 70)
    o, n = res['off'], res['on']
    print(f"  worst Δθ_s/settle:  off {o['worst_theta']:.4f}°  →  "
          f"on {n['worst_theta']:.4f}°   "
          f"({'REDUCED' if n['worst_theta'] < o['worst_theta'] else 'NOT reduced'})")
    print(f"  τ_w inter-step:     off mean {o['tw_mean']:.4f}/max {o['tw_max']:.4f}"
          f"  →  on mean {n['tw_mean']:.4f}/max {n['tw_max']:.4f} Nm")
    print(f"  h_w∞ whole:         off {o['hw_inf']:.4f}  →  on {n['hw_inf']:.4f} Nms"
          f"   (C5 cap 4.5)")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
