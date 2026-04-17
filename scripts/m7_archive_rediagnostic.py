"""Re-compute per-phase metrics on archived M7 sim_logs.

Loads v17, v19, v20, v21 sim_log.json files, recomputes metrics via the
per-phase refactor of compute_metrics, and emits one consolidated table
to stdout + results/archive_rediagnostic.md.

No interpretation — data first.

Usage:
    MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/m7_archive_rediagnostic.py
"""

import os
import sys

from crawlbot.simulation.logging import SimLog
from crawlbot.diagnostics.metrics import compute_metrics


VERSIONS = ['v17', 'v19', 'v20', 'v21']
RESULTS_ROOT = 'results'
OUTPUT_MD = os.path.join(RESULTS_ROOT, 'archive_rediagnostic.md')


def _log_path(v):
    return os.path.join(RESULTS_ROOT, f'M7_1pct_1step_{v}', 'sim_log.json')


def _fetch(results, key):
    entry = results.get(key)
    if entry is None:
        return None
    val, _, _ = entry
    return val


def _fmt(val, prec=4):
    if val is None:
        return '—'
    return f'{val:.{prec}f}'


def _fmt_abort(log):
    if not log.aborted_steps:
        return ('no', '—', '—')
    ev = log.aborted_steps[0]
    return ('yes',
            f"{ev.get('d_mm', float('nan')):.2f}",
            f"{ev.get('ori_deg', float('nan')):.2f}")


def build_row(v, log, results):
    ss_peak = _fetch(results, 'torso_ori_peak_deg_SS')
    ds_peak = _fetch(results, 'torso_ori_peak_deg_DS')
    ss_end = _fetch(results, 'ss_end_torso_ori_deg')
    qref_jump = _fetch(results, 'q_torso_ref_ss_to_ds_jump_deg')
    ee_ss = _fetch(results, 'ee_pos_peak_mm_SS')
    ee_ds = _fetch(results, 'ee_pos_peak_mm_DS')
    abort_flag, d_mm, ori_deg = _fmt_abort(log)

    return {
        'version': v,
        'SS peak ori [°]': _fmt(ss_peak),
        'DS peak ori [°]': _fmt(ds_peak),
        'ori at SS end [°]': _fmt(ss_end),
        'q_ref jump [°]': _fmt(qref_jump),
        'EE pos peak SS [mm]': _fmt(ee_ss, prec=2),
        'EE pos peak DS [mm]': _fmt(ee_ds, prec=2),
        'abort?': abort_flag,
        'abort d_mm': d_mm,
        'abort ori_deg': ori_deg,
    }


def render_table(rows):
    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    header = ' | '.join(c.ljust(widths[c]) for c in cols)
    sep = '-+-'.join('-' * widths[c] for c in cols)
    lines = [header, sep]
    for r in rows:
        lines.append(' | '.join(str(r[c]).ljust(widths[c]) for c in cols))
    return '\n'.join(lines)


def render_markdown(rows):
    cols = list(rows[0].keys())
    lines = ['| ' + ' | '.join(cols) + ' |',
             '|' + '|'.join(['---'] * len(cols)) + '|']
    for r in rows:
        lines.append('| ' + ' | '.join(str(r[c]) for c in cols) + ' |')
    return '\n'.join(lines)


def main():
    rows = []
    missing = []
    for v in VERSIONS:
        p = _log_path(v)
        if not os.path.exists(p):
            missing.append(v)
            continue
        log = SimLog.load(p)
        results = compute_metrics(log)
        rows.append(build_row(v, log, results))

    if missing:
        print(f'ERROR: Missing sim_log.json for versions: {missing}')
        print('Per A6 / the user instruction, do not re-simulate to fill the gap.')
        sys.exit(1)

    table = render_table(rows)
    print()
    print('M7 archive re-diagnostic (per-phase metrics)')
    print('=' * 110)
    print(table)
    print()

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    with open(OUTPUT_MD, 'w') as f:
        f.write('# M7 archive re-diagnostic — per-phase metrics\n\n')
        f.write('Re-computed from archived `sim_log.json` using the per-phase\n')
        f.write('refactor of `crawlbot/diagnostics/metrics.py`. No re-simulation.\n')
        f.write('Metrics are raw `np.max(np.abs(·))` over the corresponding\n')
        f.write('phase mask; `q_ref jump` is the geodesic angle between\n')
        f.write('`q_torso_ref[i_ss_last]` and `q_torso_ref[i_ds_first]`.\n\n')
        f.write(render_markdown(rows))
        f.write('\n')
    print(f'Saved: {OUTPUT_MD}')


if __name__ == '__main__':
    main()
