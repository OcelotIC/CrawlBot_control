#!/usr/bin/env python3
"""Phase QP-COND — measure WBC QP Hessian conditioning and test whether a
common-factor weight rescale can reduce it.

Monkeypatches HierarchicalQP._solve_weighted (the WBC QP's ONLY user; the
pre-planner/NMPC use CasADi IPOPT directly) to capture the per-task (A, b, W)
matrices + constraints + bounds at strided ticks during a 1-step canonical C run
(CANON flags, n_steps=1 → step-0 SS + DS). Nothing is committed to crawlbot/; the
patch is applied at runtime from this script only.

Offline it computes, at representative ticks (DS = 2 contacts, SS = 1 contact,
mid-swing vs dock-approach by position):
  - kappa(H) with the shipped 1e-6 reg  (H = Sum_i alpha_i A_i^T A_i + 1e-6 I)
  - decomposition: kappa(H_flat) [all weights=1 -> Jacobian-only] vs kappa(H)
    [weights x Jacobian], and the weight span max(alpha)/min(alpha)
  - RESCALE TEST (all weights / D): re-solve original vs rescaled at each tick,
    compare x* (behaviour) and kappa (conditioning). Two variants:
      (a) reg NOT scaled  -> the ACTUAL effect of dividing config task weights by
          D (hierarchical_qp regularization=1e-6 is a separate absolute term)
      (b) reg scaled too  -> exact H->H/D (pure scalar multiple), the invariance
          baseline: kappa identical, x* identical to machine precision.

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_qpcond_measure.py
"""
import os
import json
import numpy as np

import crawlbot.solvers.hierarchical_qp as hq

STRIDE = 5
CALLS = [0]
CAP = []   # strided full captures
KTRAJ = []  # (idx, n_eq, kappa) for every strided solve

_orig_w = hq.HierarchicalQP._solve_weighted


def _cp(x):
    return None if x is None else np.array(x, dtype=float)


def _assemble(tasks, reg, n):
    """H = Sum W-weighted A^T A + reg I,  g = -Sum A^T W b   (weight_ratio=1)."""
    H = np.zeros((n, n)); g = np.zeros(n)
    for A, b, W, _prio in tasks:
        AtW = A.T @ W
        H += AtW @ A
        g -= AtW @ b
    H = H + reg * np.eye(n)
    H = 0.5 * (H + H.T)
    return H, g


def _patched_w(self, sorted_tasks, x0):
    idx = CALLS[0]; CALLS[0] += 1
    if idx % STRIDE == 0:
        n = self.n_vars
        tasks = [(np.array(t.A, float), np.array(t.b, float),
                  np.array(t.W, float), int(t.priority)) for t in sorted_tasks]
        H, _g = _assemble(tasks, self.regularization, n)
        try:
            kappa = float(np.linalg.cond(H))
        except np.linalg.LinAlgError:
            kappa = float('inf')
        n_eq = 0 if self._C_eq is None else int(np.atleast_2d(self._C_eq).shape[0])
        KTRAJ.append((idx, n_eq, kappa))
        if idx % (STRIDE * 4) == 0:   # heavier full capture, sparser
            CAP.append(dict(
                idx=idx, tasks=tasks, reg=float(self.regularization),
                wr=float(self.weight_ratio), n=n,
                C_eq=_cp(self._C_eq), d_eq=_cp(self._d_eq),
                C_ineq=_cp(self._C_ineq), d_ineq=_cp(self._d_ineq),
                lb=np.array(self._lb, float), ub=np.array(self._ub, float),
                solver=str(self.solver_name), n_eq=n_eq))
    return _orig_w(self, sorted_tasks, x0)


hq.HierarchicalQP._solve_weighted = _patched_w

# ── run 1-step canonical C in-process (CANON flags), MJCF mutate/restore ──
import scripts.diag_cooperative_arms as dca
from scripts.diag_cooperative_arms import _mutate_mjcf, _mjcf_md5, MJCF

SCRATCH = ('/tmp/claude-0/-home-user-CrawlBot-control/'
           '51ef2e0d-fbab-59f4-a937-ec828e0d7eda/scratchpad')
with open(MJCF) as f:
    _original = f.read()
_pre = _mjcf_md5(MJCF)
try:
    _mutate_mjcf(damping=0.0, armature=0.05, anchor_dx=0.8, mass_ratio=0.01)
    try:
        dca.main(
            legacy=False, alpha_torso_lin=0.0, anchor_dx=0.8, mass_ratio=0.01,
            aocs_mode='legacy_pid_numerical', settle_seconds=20.0,
            K_theta=1.0, K_omega=50.0, tau_w_max=5.0,
            n_steps=1, ss_two_task=True, ss_alpha_mom=5000.0,
            alpha_torso_pose=24000.0, ss_alpha_ee=3e3, ss_alpha_posture=2e1,
            ss_alpha_wrench=1e-2, ss_kp_torso=3.0, ss_kd_torso=2.5,
            qp_envelope_exact=True,
            interstep_settle_alpha_wrench=3.0, interstep_settle_epsilon_v=5e-3,
            out_dir_override=f'{SCRATCH}/qpcond_run1')
    except Exception as e:   # post-processing/rendering may fail under disabled GL
        print(f'[note] main() raised after sim (capture already done): '
              f'{type(e).__name__}: {str(e)[:160]}')
finally:
    with open(MJCF, 'w') as f:
        f.write(_original)
    assert _mjcf_md5(MJCF) == _pre, 'MJCF restore failed'

print(f'\ncaptured {CALLS[0]} WBC solves; {len(KTRAJ)} strided kappa; '
      f'{len(CAP)} full problems; solver={CAP[0]["solver"] if CAP else "?"}')

# ── kappa trajectory summary, classified by n_eq (SS=1 contact, DS=2) ──
kt = np.array([(i, ne, k) for (i, ne, k) in KTRAJ], float)
eqvals = sorted(set(int(x) for x in kt[:, 1]))
print(f'\n=== kappa(H) trajectory (n_eq rows -> contacts) ===')
print(f'  distinct n_eq (dyn 20 + 6*nc): {eqvals}  '
      f'(26 => nc=1 SS, 32 => nc=2 DS)')
for ne in eqvals:
    m = kt[:, 1] == ne
    ks = kt[m, 2]
    tag = 'SS(nc=1)' if ne == 26 else ('DS(nc=2)' if ne == 32 else f'nc?({ne})')
    print(f'  {tag:>10} n_eq={ne}: n={int(m.sum()):4d}  '
          f'kappa min={ks.min():.3e} med={np.median(ks):.3e} max={ks.max():.3e}')


def resolve(prob, H, g):
    inst = hq.HierarchicalQP(n_vars=prob['n'], solver=prob['solver'])
    x, info = inst._solve_qp_raw(H, g, prob['C_eq'], prob['d_eq'],
                                 prob['C_ineq'], prob['d_ineq'],
                                 prob['lb'], prob['ub'], None)
    return x, info


# choose representative ticks: first DS, first SS (mid-swing), last SS (dock)
ss = [p for p in CAP if p['n_eq'] == 26]
ds = [p for p in CAP if p['n_eq'] == 32]
reps = []
if ds:
    reps.append(('DS-settle', ds[0]))
if ss:
    reps.append(('SS-mid-swing', ss[len(ss) // 3]))
    reps.append(('SS-dock-approach', ss[-1]))

DIVISORS = [24.0, 240.0, 1e6]   # 24000 -> 1000, 100, 0.024
out = {'ktraj_summary': {}, 'reps': []}
for ne in eqvals:
    m = kt[:, 1] == ne
    out['ktraj_summary'][int(ne)] = dict(
        n=int(m.sum()), kappa_min=float(kt[m, 2].min()),
        kappa_med=float(np.median(kt[m, 2])), kappa_max=float(kt[m, 2].max()))

print('\n=== representative ticks: kappa decomposition + rescale test ===')
for label, prob in reps:
    n = prob['n']; reg = prob['reg']
    tasks = prob['tasks']
    H, g = _assemble(tasks, reg, n)
    H_LS = H - reg * np.eye(n)
    # decomposition
    H_flat, _ = _assemble([(A, b, np.eye(W.shape[0]), pr)
                           for (A, b, W, pr) in tasks], reg, n)
    weights = []
    for _A, _b, W, _pr in tasks:
        d = np.diag(W)
        if d.size:
            weights.append((float(d.min()), float(d.max())))
    wmin = min(w[0] for w in weights); wmax = max(w[1] for w in weights)
    k_H = float(np.linalg.cond(H))
    k_LS = float(np.linalg.cond(H_LS))
    k_flat = float(np.linalg.cond(H_flat))
    x0, _ = resolve(prob, H, g)
    row = dict(label=label, idx=prob['idx'], solver=prob['solver'],
               n_eq=prob['n_eq'], weight_min=wmin, weight_max=wmax,
               weight_span=wmax / wmin, kappa_H=k_H, kappa_H_LS=k_LS,
               kappa_flat_Jacobian_only=k_flat, rescale={})
    print(f'\n [{label}] idx={prob["idx"]} solver={prob["solver"]} '
          f'n_eq={prob["n_eq"]}')
    print(f'   weight span = {wmax:.0f}/{wmin:g} = {wmax/wmin:.3e}')
    print(f'   kappa(H, +1e-6 reg) = {k_H:.4e}')
    print(f'   kappa(H_LS, no reg) = {k_LS:.4e}   '
          f'(reg effect: {abs(k_H-k_LS)/k_LS*100:.2e}% )')
    print(f'   kappa(H_flat, weights=1 = JACOBIAN-ONLY) = {k_flat:.4e}')
    print(f'   => weight span inflates kappa by x{k_H/k_flat:.2f} over Jacobian baseline')
    for D in DIVISORS:
        # (a) reg NOT scaled = real effect of dividing config task weights by D
        Ha = H_LS / D + reg * np.eye(n); ga = g / D
        xa, _ = resolve(prob, Ha, ga)
        ka = float(np.linalg.cond(Ha)); da = float(np.max(np.abs(xa - x0)))
        # (b) reg scaled too = pure scalar multiple H/D (invariance baseline)
        Hb = H / D; gb = g / D
        xb, _ = resolve(prob, Hb, gb)
        kb = float(np.linalg.cond(Hb)); db = float(np.max(np.abs(xb - x0)))
        row['rescale'][f'D={D:g}'] = dict(
            kappa_regUNscaled=ka, dx_regUNscaled=da,
            kappa_regScaled=kb, dx_regScaled=db)
        print(f'   D={D:<8g}: (a reg-fixed) kappa={ka:.4e} dx*={da:.2e} | '
              f'(b reg-scaled/pure cH) kappa={kb:.4e} dx*={db:.2e}')
    out['reps'].append(row)

os.makedirs('results/j2_adjconv', exist_ok=True)
json.dump(out, open('results/j2_adjconv/qpcond_measure.json', 'w'), indent=2)
print('\nwrote results/j2_adjconv/qpcond_measure.json')
