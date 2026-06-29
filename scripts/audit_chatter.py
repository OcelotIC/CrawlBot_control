#!/usr/bin/env python3
"""J2 DS inter-step settle chatter DIAGNOSIS (offline, no fix).

Reconstructs the exact envelope-box quantity Ḣ_s = Σ_j (r_Cj×f_j + τ_j) per
inter-step tick from the logged settle-QP wrench lambda_qp + structure-frame
anchors (the box is |M_exact·λ|≤τ_w_max with a FIXED RHS, so the active set is
recoverable offline — no crawlbot/ logging needed). Per inter-step segment
(labelled by the just-docked arm), reports the chatter metrics, the period-2
active-set / two-vertex trace, the wrench-rate (limit-cycle vs instability),
and the orbital-term / binding config discriminator (arm-a vs arm-b).

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_chatter.py \
        base=results/chatter_base cfrozen=results/chatter_cfrozen
"""
import csv
import json
import os
import sys
import numpy as np
import mujoco
import pinocchio as pin

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MJCF = os.path.join(ROOT, 'models', 'VISPA_crawling_rwa3.xml')
TAU_W_MAX = 5.0


def anchors_struct_frame(struct_pos0, struct_quat0):
    m = mujoco.MjModel.from_xml_path(MJCF)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    aw, bw = [], []
    for i in range(1, 20):
        sa = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f'anchor_{i}a')
        sb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f'anchor_{i}b')
        if sa < 0 or sb < 0:
            break
        aw.append(d.site_xpos[sa].copy())
        bw.append(d.site_xpos[sb].copy())
    qw, qx, qy, qz = struct_quat0
    R0 = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
    p0 = np.asarray(struct_pos0, float)
    return ([R0.T @ (a - p0) for a in aw], [R0.T @ (b - p0) for b in bw])


def load(run_dir):
    sl = json.load(open(os.path.join(run_dir, 'sim_log.json')))
    pp = list(csv.DictReader(open(os.path.join(run_dir, 'postproc_F3F4.csv'))))
    return sl, pp


def hdot_exact(rA, rB, L):
    return (np.cross(rA, L[0:3]) + L[3:6] + np.cross(rB, L[6:9]) + L[9:12])


def segments(sl):
    """Inter-step DS segments: (step_idx, dock_arm, tick_indices). The dock_arm
    is the arm of the dock immediately PRECEDING the settle in time (the
    just-docked arm) — NOT dock_events[step_idx], which is off-by-one."""
    t = np.asarray(sl['t'], float)
    ph = np.asarray(sl['phase'])
    docks = sorted(sl.get('dock_events', []), key=lambda d: float(d['t']))
    out = []
    for s in (sl.get('inter_step_settles', []) or []):
        lo, hi = float(s['t_start']), float(s['t_end'])
        idx = np.where((ph == 'DS') & (t > lo + 1e-6) & (t <= hi + 1e-6))[0]
        if len(idx) >= 2:
            prev = [d for d in docks if float(d['t']) <= lo + 0.05]
            arm = prev[-1].get('arm', '?') if prev else 'init'
            anc = prev[-1].get('anchor', -1) if prev else -1
            out.append((int(s['step_idx']), f'{arm}{anc if anc >= 0 else ""}', idx))
    return out


def analyze_run(tag, run_dir):
    sl, pp = load(run_dir)
    lq = np.asarray(sl['lambda_qp'], float)
    r_com = np.asarray(sl['r_com'], float)
    sp0 = np.asarray(sl['struct_pos'], float)[0]
    sq0 = np.asarray(sl['struct_quat'], float)[0]
    sa_i = np.array([int(r['stance_a_idx']) for r in pp])
    sb_i = np.array([int(r['stance_b_idx']) for r in pp])
    anchors_a, anchors_b = anchors_struct_frame(sp0, sq0)

    print(f"\n{'='*78}\nRUN: {tag}  ({run_dir})\n{'='*78}")
    print(f"  {'seg(step/arm)':16s} {'nticks':>6s} {'flipfrac':>9s} {'at±5':>6s} "
          f"{'exact‖·‖med':>11s} {'proxy‖·‖med':>11s} {'orbital med':>11s} "
          f"{'wrenchΔ med':>11s} {'Δtrend':>7s} {'vtxdist':>8s}")
    rows = {}
    for (step, arm, idx) in segments(sl):
        if len(idx) < 3:
            continue
        H = np.array([hdot_exact(anchors_a[sa_i[k]], anchors_b[sb_i[k]], lq[k])
                      for k in idx])           # exact Ḣ_s per tick
        fsum = lq[idx, 0:3] + lq[idx, 6:9]
        orb = np.cross(r_com[idx], fsum)        # orbital term r_com×Σf
        proxy = H - orb
        # flip-fraction: per-axis sign flips tick-to-tick; report the max axis.
        flips = [np.mean(np.sign(H[1:, a]) != np.sign(H[:-1, a])) for a in range(3)]
        flipfrac = max(flips)
        at5 = np.mean((np.abs(H) >= TAU_W_MAX - 1e-3).any(axis=1))
        # wrench rate ‖λ_k − λ_{k-1}‖ and trend (2nd-half / 1st-half mean).
        dlam = np.linalg.norm(np.diff(lq[idx], axis=0), axis=1)
        h = len(dlam) // 2
        trend = (dlam[h:].mean() / dlam[:h].mean()) if h > 0 and dlam[:h].mean() > 1e-9 else 1.0
        # two-vertex distance: mean λ on even vs odd ticks within the segment.
        ev = lq[idx[0::2]].mean(0); od = lq[idx[1::2]].mean(0)
        vtx = np.linalg.norm(ev - od)
        print(f"  step{step}/{arm:1s} (n={len(idx):3d})  {len(idx):6d} {flipfrac:9.3f} "
              f"{at5:6.2f} {np.median(np.linalg.norm(H,axis=1)):11.3f} "
              f"{np.median(np.linalg.norm(proxy,axis=1)):11.3f} "
              f"{np.median(np.linalg.norm(orb,axis=1)):11.3f} "
              f"{np.median(dlam):11.3f} {trend:7.2f} {vtx:8.3f}")
        rows[(step, arm)] = dict(flipfrac=flipfrac, at5=at5,
                                 exact=float(np.median(np.linalg.norm(H, axis=1))),
                                 proxy=float(np.median(np.linalg.norm(proxy, axis=1))),
                                 orbital=float(np.median(np.linalg.norm(orb, axis=1))),
                                 dlam=float(np.median(dlam)), trend=float(trend), vtx=float(vtx),
                                 H=H, lam=lq[idx], anchors=(sa_i[idx[0]], sb_i[idx[0]]))
    return rows


def active_set_trace(tag, rows, step):
    """Period-2 active-set trace on one chattering segment: which axes at ±5,
    sign on even vs odd ticks, and the two alternating wrench vertices."""
    key = next((k for k in rows if k[0] == step), None)
    if key is None:
        return
    H = rows[key]['H']; lam = rows[key]['lam']
    print(f"\n  [{tag}] active-set trace, step{step}/{key[1]} (first 8 ticks):")
    for k in range(min(8, len(H))):
        binding = [f"{'xyz'[a]}{'+' if H[k,a]>0 else '-'}" for a in range(3)
                   if abs(H[k, a]) >= TAU_W_MAX - 1e-3]
        print(f"    tick {k} ({'even' if k%2==0 else 'odd '}): Ḣ_s=[{H[k,0]:+.2f},{H[k,1]:+.2f},"
              f"{H[k,2]:+.2f}] at±5: {binding or '—'}")
    ev = lam[0::2].mean(0); od = lam[1::2].mean(0)
    print(f"    vertex A (even) λ≈ [{', '.join(f'{x:+.1f}' for x in ev)}]")
    print(f"    vertex B (odd ) λ≈ [{', '.join(f'{x:+.1f}' for x in od)}]")
    print(f"    ‖A−B‖={np.linalg.norm(ev-od):.2f}  intra-A std={np.linalg.norm(lam[0::2].std(0)):.2f} "
          f"intra-B std={np.linalg.norm(lam[1::2].std(0)):.2f}  "
          f"(‖A−B‖ ≫ intra-std ⇒ two fixed vertices = active-set limit cycle)")


def main(argv):
    runs = {}
    for a in argv:
        if '=' in a:
            k, v = a.split('=', 1); runs[k] = v
    if 'base' not in runs:
        print('usage: audit_chatter.py base=<dir> [cfrozen=<dir>]'); return 2
    res = {tag: analyze_run(tag, d) for tag, d in runs.items()}

    # chatterers = binding segments (at±5 > 0.5); excludes the at-rest initial DS.
    chatters = sorted([k for k, r in res['base'].items() if r['at5'] > 0.5],
                      key=lambda k: -res['base'][k]['flipfrac'])
    print(f"\nchattering segments (at±5>0.5): {chatters}")

    # active-set trace on the worst chattering segment, baseline
    if chatters:
        active_set_trace('base', res['base'], step=chatters[0][0])

    # H1/H2 verdict: did freezing c_curr remove the chatter on the binding segments?
    if 'cfrozen' in res:
        print(f"\n{'='*78}\nH1/H2 — freeze c_curr (test A): chatter metrics base → cfrozen\n{'='*78}")
        for k in res['base']:
            kc = next((c for c in res['cfrozen'] if c[0] == k[0]), None)
            if kc:
                b, c = res['base'][k], res['cfrozen'][kc]
                print(f"  step{k[0]}/{k[1]}: flipfrac {b['flipfrac']:.3f}→{c['flipfrac']:.3f}  "
                      f"at±5 {b['at5']:.2f}→{c['at5']:.2f}  exact‖·‖ {b['exact']:.2f}→{c['exact']:.2f}  "
                      f"vtxdist {b['vtx']:.2f}→{c['vtx']:.2f}")
        chat_b = np.mean([res['base'][k]['flipfrac'] for k in chatters]) if chatters else 0
        chat_c = np.mean([res['cfrozen'][kc]['flipfrac'] for k in chatters
                          for kc in res['cfrozen'] if kc[0] == k[0]]) if chatters else 0
        verdict = ('H1 (c_curr) — freezing removed it' if chat_c < 0.3 * chat_b
                   else 'H2 (structural) — freezing did NOT remove it' if chat_c > 0.7 * chat_b
                   else 'PARTIAL — attenuated but not removed')
        print(f"  chatterer mean flip-frac: base={chat_b:.3f}  cfrozen={chat_c:.3f}  ⇒ {verdict}")

    # config discriminator: arm-a (1,3) vs arm-b (0,2) — orbital & binding
    print(f"\n{'='*78}\nCONFIG — arm-a (steps 1,3) vs arm-b (steps 0,2): why only arm-a binds\n{'='*78}")
    for step in (0, 1, 2, 3, 4):
        k = next((kk for kk in res['base'] if kk[0] == step), None)
        if k:
            r = res['base'][k]
            print(f"  step{step}/{k[1]}: anchors(a,b)={r['anchors']}  orbital‖·‖med={r['orbital']:.2f}  "
                  f"exact‖·‖med={r['exact']:.2f} (cap 5)  at±5={r['at5']:.2f}  flipfrac={r['flipfrac']:.3f}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
