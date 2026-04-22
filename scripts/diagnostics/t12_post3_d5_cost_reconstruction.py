"""T12-post-3 D5 — NMPC cost regime change at k=65."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, '/home/user/CrawlBot_control')

ROOT = '/home/user/CrawlBot_control'

# Cost weights (from CentroidalNMPCConfig defaults, with Wv overridden by
# nmpc_Wv=10.0 from SimConfig — same as default)
Wr = np.diag(100.0 * np.ones(3))
Wv = np.diag(10.0 * np.ones(3))
w_L = 1.0
Wu_f = 0.01
Wu_tau = 0.001
Qf_r = np.diag(1000.0 * np.ones(3))
Qf_v = np.diag(100.0 * np.ones(3))
Qf_L = 10.0
N_horizon = 8

print("=" * 60)
print("D5 — NMPC COST REGIME CHANGE at k=65")
print("=" * 60)
print()
print("Cost function (per stage, summed over N=8 horizon, + terminal):")
print(f"  stage:    ||e_r||²·Wr(100) + ||e_v||²·Wv(10) + w_L·||e_L||²(1)")
print(f"            + ||u||²·(Wu_f=0.01 for forces, Wu_tau=0.001 for torques)")
print(f"  terminal: ||e_r||²·Qf_r(1000) + ||e_v||²·Qf_v(100) + Qf_L·||e_L||²(10)")
print()

for tag, sub in [('T12 unfixed', 'M7_14pct_1step_v22_with_swing_hold'),
                 ('T12 Option A', 'M7_14pct_1step_v22_with_swing_hold_optA')]:
    print()
    print("-" * 60)
    print(f"  {tag}")
    print("-" * 60)
    with open(os.path.join(ROOT, 'results', sub, 'sim_log.json')) as f:
        log = json.load(f)
    t = np.array(log['t'])
    cost = log['nmpc_cost']
    status = log['nmpc_status']
    r_com = np.array(log['r_com'])
    r_com_ref = np.array(log['r_com_ref'])
    v_com = np.array(log['v_com'])
    v_com_ref = np.array(log['v_com_ref'])
    L_com = np.array(log['L_com'])
    L_com_ref = np.array(log['L_com_ref'])
    lam_ref = np.array(log['lambda_ref'])  # 12-vec: 2x(f3 + tau3)

    # Stage-0-only reconstruction
    print(f"\n[per-tick total NMPC cost vs stage-0 reconstruction]")
    print(f"{'k':>4} {'t':>6} {'tot cost':>10} {'st0_recon':>10} "
          f"{'tracking':>10} {'L track':>9} {'wrench reg':>11} "
          f"{'status':>6}")
    for k in [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 70, 80, 100, 150, 200]:
        if k >= len(t):
            continue
        e_r = r_com[k] - r_com_ref[k]
        e_v = v_com[k] - v_com_ref[k]
        e_L = L_com[k] - L_com_ref[k]
        u = lam_ref[k]
        u_f1 = u[0:3]; u_t1 = u[3:6]; u_f2 = u[6:9]; u_t2 = u[9:12]
        c_track = float(e_r @ Wr @ e_r) + float(e_v @ Wv @ e_v)
        c_L = float(w_L * (e_L @ e_L))
        c_u = (Wu_f * (u_f1 @ u_f1 + u_f2 @ u_f2)
               + Wu_tau * (u_t1 @ u_t1 + u_t2 @ u_t2))
        st0 = c_track + c_L + c_u
        print(f"{k:>4} {t[k]:>6.2f} {cost[k]:>10.4f} {st0:>10.4f} "
              f"{c_track:>10.4f} {c_L:>9.4f} {c_u:>11.4f} "
              f"{status[k]:>6d}")

    # Reference fields around the cost regime change
    print(f"\n[reference quantities around k=65]")
    print(f"{'k':>4} {'t':>6} {'r_com':>30} {'r_com_ref':>30} "
          f"{'L_com':>22} {'L_com_ref':>22}")
    for k in [58, 59, 60, 64, 65, 66, 67]:
        if k >= len(t):
            continue
        rc = r_com[k]
        rr = r_com_ref[k]
        Lc = L_com[k]
        Lr = L_com_ref[k]
        print(f"{k:>4} {t[k]:>6.2f} "
              f"({rc[0]:+.3f},{rc[1]:+.3f},{rc[2]:+.3f}) "
              f"({rr[0]:+.3f},{rr[1]:+.3f},{rr[2]:+.3f}) "
              f"({Lc[0]:+.3f},{Lc[1]:+.3f},{Lc[2]:+.3f}) "
              f"({Lr[0]:+.3f},{Lr[1]:+.3f},{Lr[2]:+.3f})")

    # Constraint activity proxies
    print(f"\n[constraint-activity proxies — lambda_ref magnitudes]")
    print(f"{'k':>4} {'t':>6} {'|f1|':>8} {'|tau1|':>8} "
          f"{'|f2|':>8} {'|tau2|':>8} {'|lambda|':>9}")
    for k in [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 70, 80]:
        if k >= len(t):
            continue
        u = lam_ref[k]
        f1 = np.linalg.norm(u[0:3])
        t1 = np.linalg.norm(u[3:6])
        f2 = np.linalg.norm(u[6:9])
        t2 = np.linalg.norm(u[9:12])
        print(f"{k:>4} {t[k]:>6.2f} {f1:>8.4f} {t1:>8.4f} "
              f"{f2:>8.4f} {t2:>8.4f} {np.linalg.norm(u):>9.4f}")

    # Solve time as a proxy for active-set complexity
    print(f"\n[solve time around k=65]")
    nmpc_ms = log.get('nmpc_time_ms', [])
    for k in [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 70, 80]:
        if k < len(nmpc_ms):
            print(f"  k={k:>3} t={t[k]:>6.2f}  nmpc={nmpc_ms[k]:>7.2f} ms  "
                  f"status={status[k]}")

    # h_w trajectory near k=65 — relevant if hw_conservation constraint
    # is binding
    print(f"\n[h_w (component-wise box ±5 Nms) around k=65]")
    hw = np.array(log['hw_physical'])
    print(f"{'k':>4} {'t':>6} {'hx':>8} {'hy':>8} {'hz':>8} {'|h|':>8}")
    for k in [58, 59, 60, 64, 65, 66, 67, 70, 100, 200, 258]:
        if k >= len(t):
            continue
        h = hw[k]
        print(f"{k:>4} {t[k]:>6.2f} {h[0]:>+8.4f} {h[1]:>+8.4f} {h[2]:>+8.4f} "
              f"{np.linalg.norm(h):>8.4f}")
