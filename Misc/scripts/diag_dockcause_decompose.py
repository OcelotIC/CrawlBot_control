#!/usr/bin/env python3
"""Phase DOCK-CAUSE — decompose the dock error in the WBC QP.

READ-ONLY on the committed canonical per-step baselines
(results/figC_sw_s{k}_x1/sim_log.json — the ×1 sweep runs, each truncated
to step k terminal, so step k's dock == the full-run dock).

For each step k at its dock instant (min d_grip_swing over step-k SS ticks):
  dock error  = ||p_ee_realized - anchor||        (physical gripper->anchor)
  WBC residual= ||p_ee_realized - p_ee_ref||       (= e_ee_pos, logged)
  ref gap     = ||p_ee_ref     - anchor||          (planned swing ref -> anchor)
and the exact vector identity  (p_ee - anchor) = (p_ee - p_ee_ref) + (p_ee_ref - anchor).

p_ee / p_ee_ref are in the Pinocchio world = structure frame R_s (state_conversions
expresses everything in R_s). Anchors are static in R_s, so anchor_struct (from
load_anchors_struct_frame at tick 0) is the correct constant anchor coordinate at
every tick. dock_pin = ||p_ee - anchor_struct|| cross-checks against the physical
d_grip_swing (gripper-site metric); any gap = the Pinocchio-tool vs MuJoCo-gripper-site
offset (reported, not hidden).

Arm-config proxy: reach = ||p_ee - p_torso|| at dock (true J_ee conditioning needs
per-tick q, which is NOT logged). standoff = ||r_com|| (step start, matches TSTEP).

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/diag_dockcause_decompose.py
"""
import json
import numpy as np
import mujoco
from scripts.export_figure_data import load_anchors_struct_frame, phase_per_tick, MJCF

model = mujoco.MjModel.from_xml_path(MJCF)
STEPS = [0, 1, 2, 3, 4, 5]
# Canonical dock sequence (swing complement of the stance mapping in
# diag_tstep_sweep_extract.py: STANCE=['a','b','a','b','a','b']@[2,3,3,4,4,5]).
# Swing arm docks its target anchor: b@3, a@3, b@4, a@4, b@5, a@5.
SWING_ARM = ['b', 'a', 'b', 'a', 'b', 'a']
SWING_ANCHOR = [3, 3, 4, 4, 5, 5]


def decompose(sl, k):
    si = np.asarray(sl['step_idx'])
    ph = np.array(phase_per_tick(sl))
    dg = np.asarray(sl['d_grip_swing'], float)
    p_ee = np.asarray(sl['p_ee'], float)
    p_ee_ref = np.asarray(sl['p_ee_ref'], float)
    e_ee = np.asarray(sl['e_ee_pos'], float)
    p_torso = np.asarray(sl['p_torso'], float)
    rc = np.asarray(sl['r_com'], float)
    swing = np.asarray(sl['swing_arm'])

    # dock instant: min d_grip over step-k SS ticks (where the swing ref is live)
    ss = (si == k) & (ph == 'SS') & np.isfinite(dg)
    if not ss.any():
        return None
    ss_idx = np.where(ss)[0]
    i = ss_idx[int(np.argmin(dg[ss_idx]))]
    # all-phase min for cross-check against the sweep table
    allm = (si == k) & np.isfinite(dg)
    dock_all_mm = float(1000 * dg[allm].min())

    aA, aB = load_anchors_struct_frame(
        model, np.asarray(sl['struct_pos'])[0], np.asarray(sl['struct_quat'])[0])
    arm = SWING_ARM[k]
    assert str(swing[i]) == arm, f"step{k}: log swing {swing[i]} != {arm}"
    anchor = (aA if arm == 'a' else aB)[SWING_ANCHOR[k]]

    dock_vec = p_ee[i] - anchor          # gripper(tool) -> anchor  (struct frame)
    wbc_vec = p_ee[i] - p_ee_ref[i]      # realized -> planned ref
    ref_vec = p_ee_ref[i] - anchor       # planned ref -> anchor
    ident = np.linalg.norm(dock_vec - (wbc_vec + ref_vec))   # ~0 by construction

    # step-start standoff (matches TSTEP), reach at dock
    i0 = np.where(si == k)[0][0]
    standoff0 = float(np.linalg.norm(rc[i0]))
    reach = float(np.linalg.norm(p_ee[i] - p_torso[i]))

    return dict(
        step=k, arm=arm, anchor_idx=SWING_ANCHOR[k],
        phase_at_dock=str(ph[i]),
        dock_gate_mm=float(1000 * dg[i]),      # physical gripper-site metric
        dock_all_mm=dock_all_mm,               # min over all phases (table value)
        dock_pin_mm=float(1000 * np.linalg.norm(dock_vec)),   # tool-frame recon
        wbc_resid_mm=float(1000 * np.linalg.norm(wbc_vec)),
        e_ee_pos_logged_mm=float(1000 * e_ee[i]),             # consistency check
        ref_gap_mm=float(1000 * np.linalg.norm(ref_vec)),
        ident_um=float(1e6 * ident),           # identity residual [micron]
        tool_vs_grip_mm=float(1000 * abs(np.linalg.norm(dock_vec) - dg[i])),
        standoff0_m=standoff0, reach_m=reach,
    )


def terminal_readout(sl, k):
    """dock-min-tick vs LAST-SS-tick: is the dock min a converged terminal
    hold (ref at anchor) or a mid-approach transient (ref still short)?"""
    si = np.asarray(sl['step_idx']); ph = np.array(phase_per_tick(sl))
    dg = np.asarray(sl['d_grip_swing'], float)
    p_ee_ref = np.asarray(sl['p_ee_ref'], float)
    swing = np.asarray(sl['swing_arm'])
    aA, aB = load_anchors_struct_frame(
        model, np.asarray(sl['struct_pos'])[0], np.asarray(sl['struct_quat'])[0])
    anchor = (aA if SWING_ARM[k] == 'a' else aB)[SWING_ANCHOR[k]]
    ss = (si == k) & (ph == 'SS') & np.isfinite(dg)
    ss_idx = np.where(ss)[0]
    i_min = ss_idx[int(np.argmin(dg[ss_idx]))]
    i_last = ss_idx[-1]
    rg = lambda j: float(1000 * np.linalg.norm(p_ee_ref[j] - anchor))
    return dict(step=k, n_ss=int(ss.sum()),
                imin_pos=int(np.where(ss_idx == i_min)[0][0]),
                dmin_mm=float(1000 * dg[i_min]), rg_min_mm=rg(i_min),
                dlast_mm=float(1000 * dg[i_last]), rg_last_mm=rg(i_last))


rows = []
terms = []
for k in STEPS:
    sl = json.load(open(f'results/figC_sw_s{k}_x1/sim_log.json'))
    r = decompose(sl, k)
    if r:
        rows.append(r)
    terms.append(terminal_readout(sl, k))

json.dump(rows, open('results/j2_adjconv/dockcause_decomp.json', 'w'), indent=2)

hdr = ('step arm@anch phase  dock_gate dock_pin  WBCresid e_ee(log)  ref_gap '
       'ident_um  tooloff  standoff  reach')
print(hdr)
print('-' * len(hdr))
for r in rows:
    print(f'  {r["step"]}  {r["arm"]}@{r["anchor_idx"]}   {r["phase_at_dock"]:>3}  '
          f'{r["dock_gate_mm"]:8.4f} {r["dock_pin_mm"]:8.4f} '
          f'{r["wbc_resid_mm"]:8.4f} {r["e_ee_pos_logged_mm"]:8.4f}  '
          f'{r["ref_gap_mm"]:7.4f} {r["ident_um"]:7.2f}  {r["tool_vs_grip_mm"]:7.4f}  '
          f'{r["standoff0_m"]:7.3f}  {r["reach_m"]:6.3f}')

# ---- decomposition summary: which term dominates ----
print('\n=== decomposition: dock = WBC-residual + ref-gap (per step) ===')
for r in rows:
    frac_wbc = r['wbc_resid_mm'] / r['dock_pin_mm'] if r['dock_pin_mm'] else float('nan')
    frac_ref = r['ref_gap_mm'] / r['dock_pin_mm'] if r['dock_pin_mm'] else float('nan')
    dom = 'WBC-RESIDUAL' if r['wbc_resid_mm'] > r['ref_gap_mm'] else 'REF-GAP'
    print(f'  step{r["step"]}: dock={r["dock_pin_mm"]:.3f}mm = '
          f'WBC {r["wbc_resid_mm"]:.3f} ({100*frac_wbc:.0f}%) + '
          f'ref {r["ref_gap_mm"]:.4f} ({100*frac_ref:.1f}%)  -> {dom}')

# ---- correlations (task 5) ----
def corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])

dock = [r['dock_pin_mm'] for r in rows]
wbc = [r['wbc_resid_mm'] for r in rows]
reach = [r['reach_m'] for r in rows]
standoff = [r['standoff0_m'] for r in rows]
print('\n=== correlations across 6 steps ===')
print(f'  corr(WBC_resid, reach)     = {corr(wbc, reach):+.3f}')
print(f'  corr(WBC_resid, standoff)  = {corr(wbc, standoff):+.3f}')
print(f'  corr(dock, reach)          = {corr(dock, reach):+.3f}')
print(f'  corr(dock, standoff)       = {corr(dock, standoff):+.3f}   (TSTEP found -0.11)')
print(f'  reach per step   : ' + '  '.join(f's{r["step"]}:{r["reach_m"]:.3f}' for r in rows))
print(f'  standoff per step: ' + '  '.join(f's{r["step"]}:{r["standoff0_m"]:.3f}' for r in rows))

# ---- terminal vs transient: is the dock min a converged hold or mid-approach? ----
print('\n=== dock-min tick vs LAST-SS tick (ref_gap ~0 => converged terminal hold) ===')
print(' step  n_ss  imin/n_ss   d@min  refgap@min   d@last  refgap@last   verdict')
for t in terms:
    frac = t['imin_pos'] / max(1, t['n_ss'] - 1)
    verdict = ('TERMINAL-HOLD' if t['rg_min_mm'] < 0.05
               else 'MID-APPROACH TRANSIENT')
    print(f'   {t["step"]}   {t["n_ss"]:4d}   {frac:5.2f}     '
          f'{t["dmin_mm"]:6.3f}  {t["rg_min_mm"]:8.4f}    '
          f'{t["dlast_mm"]:6.3f}  {t["rg_last_mm"]:8.4f}   {verdict}')
json.dump({'decomp': rows, 'terminal': terms},
          open('results/j2_adjconv/dockcause_decomp.json', 'w'), indent=2)
