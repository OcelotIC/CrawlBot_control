#!/usr/bin/env python3
"""J1 Part B.2 — terminal centroidal-DS metrics: Fix-A (ae0673e) vs pre-Fix-A.

READ-ONLY. Isolates the TERMINAL DS window (the trailing settle after the last
SS swing, phase=='DS', t > last-SS-end) in each run's gate log and reports the
gate-style metrics over THAT window only, plus the residual subtree_angmom[0]
injected across the terminal DS (dock_step4 -> final). Fix-A expected equal or
slightly better.
"""
import json
import os
import numpy as np
import pinocchio as pin
import mujoco

ROOT = os.path.join(os.path.dirname(__file__), '..')
MJCF = os.path.join(ROOT, 'models', 'VISPA_crawling_rwa3.xml')
FIXA = os.path.join(ROOT, 'results', 'fixA_gate')           # Fix-A canonical
PRE = os.path.join(ROOT, 'results', 'p3b_gate_w24000_kp3')  # pre-Fix-A canonical


def terminal_ds(sl):
    t = np.array(sl['t']); ph = np.array(sl['phase'])
    ss_end = t[ph == 'SS'].max()
    mask = (ph == 'DS') & (t > ss_end + 1e-9)
    return mask, ss_end


def theta_s_deg(sq):
    """structure attitude angle vs initial, degrees (per tick)."""
    R0 = pin.Quaternion(sq[0][0], sq[0][1], sq[0][2], sq[0][3]).toRotationMatrix()
    out = []
    for q in sq:
        R = pin.Quaternion(q[0], q[1], q[2], q[3]).toRotationMatrix()
        out.append(np.degrees(np.linalg.norm(pin.log3(R0.T @ R))))
    return np.array(out)


def H_sys_at(label, sl, m, d):
    for (tt, q, v, lbl) in sl['snapshots']:
        if lbl == label:
            d.qpos[:] = np.array(q); d.qvel[:] = np.array(v)
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            mujoco.mj_comVel(m, d); mujoco.mj_subtreeVel(m, d)
            return d.subtree_angmom[0].copy()
    return np.full(3, np.nan)


def metrics(path, m, d):
    sl = json.load(open(os.path.join(path, 'sim_log.json')))
    mask, ss_end = terminal_ds(sl)
    t = np.array(sl['t'])
    e_pos = np.array(sl['e_torso_pos'])[mask] * 1000.0     # mm
    e_ori = np.array(sl['e_torso_ori'])[mask]              # deg (tracking)
    p_torso = np.array(sl['p_torso'])[mask]                # actual torso pos (world)
    hw = np.array(sl['hw'])
    sq = np.array(sl['struct_quat'])
    th = theta_s_deg(sq)
    th_win = th[mask]
    hw_win = hw[mask]
    # subtree_angmom across the terminal DS (dock_step4 -> final)
    H_dock4 = H_sys_at('dock_step4', sl, m, d)
    H_final = H_sys_at('final', sl, m, d)
    return dict(
        n=int(mask.sum()), ss_end=ss_end, t_end=float(t[mask].max()),
        c2_pos_peak=float(np.max(e_pos)), c2_pos_mean=float(np.mean(e_pos)),
        torso_actual_disp_mm=float(np.linalg.norm(p_torso[-1] - p_torso[0]) * 1000.0),
        c2_ori_rms=float(np.sqrt(np.mean(e_ori**2))),
        c4_theta_peak=float(np.max(th_win)), c4_theta_final=float(th[-1]),
        c5_hw_peak=float(np.max(np.abs(hw_win))),
        H_dock4=H_dock4, H_final=H_final,
        dH_terminal=float(np.linalg.norm(H_final - H_dock4)),
        H_final_norm=float(np.linalg.norm(H_final)))


def main():
    m = mujoco.MjModel.from_xml_path(MJCF); d = mujoco.MjData(m)
    fa = metrics(FIXA, m, d)
    pr = metrics(PRE, m, d)
    print('J1 Part B.2 — terminal centroidal-DS metrics: Fix-A vs pre-Fix-A\n')
    print(f'  terminal-DS window: Fix-A t={fa["ss_end"]:.2f}-{fa["t_end"]:.2f}s '
          f'({fa["n"]} ticks) | pre-Fix-A t={pr["ss_end"]:.2f}-{pr["t_end"]:.2f}s ({pr["n"]} ticks)\n')
    rows = [
        ('C2 torso pos PEAK [mm]', 'c2_pos_peak', 'ref-trans', False),
        ('C2 torso pos MEAN [mm]', 'c2_pos_mean', 'small', False),
        ('   torso ACTUAL disp [mm]', 'torso_actual_disp_mm', 'small', False),
        ('C2 torso ori RMS [deg]', 'c2_ori_rms', '<=0.68', False),
        ('C4 theta_s peak [deg]', 'c4_theta_peak', '<=1.9', False),
        ('C4 theta_s final [deg]', 'c4_theta_final', '<=1.65', False),
        ('C5 h_w peak inf-norm', 'c5_hw_peak', '<=4.5', False),
        ('residual dH_sys over term-DS', 'dH_terminal', 'small', False),
        ('|H_sys| final (full run)', 'H_final_norm', '~0', False),
    ]
    print(f'  {"metric":<32} {"Fix-A":>10} {"pre-Fix-A":>11} {"limit":>8}  better?')
    for name, key, lim, _ in rows:
        better = '=' if abs(fa[key]-pr[key]) < 1e-6 else ('YES' if fa[key] < pr[key] else 'worse')
        print(f'  {name:<32} {fa[key]:>10.4f} {pr[key]:>11.4f} {lim:>8}  {better}')
    print(f'\n  terminal-DS subtree_angmom (vec):')
    print(f'    Fix-A:     dock4 {np.round(fa["H_dock4"],4)} -> final {np.round(fa["H_final"],4)}')
    print(f'    pre-Fix-A: dock4 {np.round(pr["H_dock4"],4)} -> final {np.round(pr["H_final"],4)}')
    print(f'  => terminal-DS injects ‖ΔH‖: Fix-A {fa["dH_terminal"]:.4f} vs pre-Fix-A '
          f'{pr["dH_terminal"]:.4f}  ({pr["dH_terminal"]/max(fa["dH_terminal"],1e-9):.0f}x less under Fix-A)')


if __name__ == '__main__':
    main()
