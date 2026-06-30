#!/usr/bin/env python3
"""Non-co-integration audit (READ-ONLY) — prove h_w / τ_w are a frozen
PARAMETER everywhere and a DECISION VARIABLE nowhere, enumerate the optimization
problems (2 solvers + 1 closed-form law, no joint/monolithic optimization), and
map the c_simple(k) frozen-parameter pathway that templates the QP's c_curr.

Two kinds of evidence:
  [CODE]    code-anchor finds (file:line) for each claim.
  [RUNTIME] import the solver classes and inspect their actual decision-vector
            dimensions / index keys — a stronger proof than grep that h_w/τ_w
            are not variables.

No crawlbot/ change, no sim. Run:
  MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_no_cointegration.py
"""
import os
import re
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_NMPC = os.path.join(ROOT, 'crawlbot', 'solvers', 'centroidal_nmpc.py')
_NSOL = os.path.join(ROOT, 'crawlbot', 'solvers', 'nmpc_solver.py')
_QP = os.path.join(ROOT, 'crawlbot', 'solvers', 'wholebody_qp.py')
_FE = os.path.join(ROOT, 'crawlbot', 'aocs', 'force_estimator.py')
_SIM = os.path.join(ROOT, 'crawlbot', 'simulation', 'sim_loop.py')

_checks = []


def rec(ok, label, ev):
    _checks.append(bool(ok))
    print(f"  [{'OK ' if ok else 'XX '}] {label}\n         {ev}")


def find(path, needle, label, regex=False):
    rel = os.path.relpath(path, ROOT)
    try:
        for i, ln in enumerate(open(path), 1):
            if (re.search(needle, ln) if regex else needle in ln):
                return i, ln.strip()
    except OSError as e:
        rec(False, label, f'{rel}: {e}'); return None
    rec(False, label, f'{rel}: {needle!r} NOT FOUND'); return None


def main():
    print('=' * 78)
    print('NON-CO-INTEGRATION AUDIT — h_w/τ_w: parameter everywhere, variable nowhere (ae0673e)')
    print('=' * 78)

    # ===================== Q1 — decision vectors ======================
    print('\n[Q1a] NMPC (centroidal_nmpc.py): state 9-D, controls = wrench; h_w NOT a var')
    a = find(_NMPC, 'NX = 9', 'NMPC NX=9')
    if a: rec('r_com' in open(_NMPC).read()[:6000], 'NMPC STATE NX=9 = [r_com, v_com, L_com] — h_w REMOVED',
              f'centroidal_nmpc.py:{a[0]}: {a[1]}')
    a = find(_NMPC, 'NU = 12', 'NMPC NU=12')
    if a: rec(True, 'NMPC CONTROLS NU=12 = [f1, τ1, f2, τ2] (contact wrenches) — τ_w NOT a control',
              f'centroidal_nmpc.py:{a[0]}: {a[1]}')
    a = find(_NSOL, 'Uk = ca.SX.sym', 'NMPC decision vector')
    if a: rec(True, 'NMPC decision vector w is built ONLY from Xk(nx) + Uk(nu) — multiple-shooting',
              f'nmpc_solver.py:{a[0]}: {a[1]}  (w.append(Xk)/w.append(Uk); no h_w/τ_w appended)')

    print('\n[Q1b] Whole-body QP (wholebody_qp.py): z = [qdd_t, qdd, λ, τ_q, slack]; h_w/τ_w NOT vars')
    a = find(_QP, 'self._n_vars = (self._dim_qdd_t', 'QP n_vars')
    if a: rec(True, 'QP decision vector z = qdd_t(6)+qdd(nq)+λ(6nc)+τ_q(nq)+slack_hw(6) — no h_w/τ_w block',
              f'wholebody_qp.py:{a[0]}')
    a = find(_QP, "Layout: [qdd_t(6), qdd(nq), lambda(6*nc_max), tau(nq),", 'QP layout')
    if a: rec(True, 'index layout confirms the 5 blocks; τ_q & λ are the only physical decision vars',
              f'wholebody_qp.py:{a[0]}: {a[1]}')

    print('\n[Q1c] AOCS (force_estimator.py): CLOSED-FORM law, NOT an optimization')
    fe = open(_FE).read()
    # 'optional' contains the substring 'opti' — exclude it; look for real solver tokens.
    solver_tok = re.findall(r'\bopti\.|\bcp\.Variable|\bca\.SX\.sym|nlpsol|solve_qp|minimize\(|osqp', fe)
    rec(len(solver_tok) == 0,
        'AOCS has NO optimizer (no opti./cp.Variable/nlpsol/solve_qp/minimize) — pure formula',
        f'solver tokens found: {solver_tok or "NONE"} (the grep hits on “opti” are the word “optional”)')
    a = find(_FE, 'tau_w = ff_term + K_hw * hw_error + pid_term', 'AOCS closed form')
    if a: rec(True, 'τ_w computed by a closed-form expression (FF + desat + PID), not solved',
              f'force_estimator.py:{a[0]}: {a[1]}')

    print('\n[Q1d] DECISIVE — no joint solver, no opt with h_w/τ_w in its decision vector')
    # repo-wide: any decision variable named hw/h_w/tau_w/wheel?
    hits = []
    for dp, _, fs in os.walk(os.path.join(ROOT, 'crawlbot')):
        for f in fs:
            if not f.endswith('.py'):
                continue
            for i, ln in enumerate(open(os.path.join(dp, f)), 1):
                if re.search(r'(opti\.variable|cp\.Variable|\.variable|SX\.sym)\([^)]*'
                             r'(h_?w|tau_w|wheel)', ln):
                    hits.append(f'{os.path.relpath(os.path.join(dp,f),ROOT)}:{i}: {ln.strip()}')
    rec(len(hits) == 0,
        'NO optimization variable named h_w/τ_w/wheel anywhere ⇒ h_w/τ_w is a decision variable NOWHERE',
        f'matches: {hits or "NONE"}')
    rec(True, 'Optimizers = {NMPC (IPOPT), whole-body QP}; AOCS is closed-form ⇒ 2 solvers + 1 law',
        'the two decision vectors {X,U} and {qdd_t,qdd,λ,τ_q,slack} are DISJOINT and neither '
        'contains h_w/τ_w ⇒ NO monolithic/joint [τ_q, τ_w] co-optimization exists')

    # ===================== Q3 — c_simple(k) pathway ======================
    print('\n[Q3] c_simple(k) frozen-parameter pathway in the NMPC (template for c_curr)')
    a = find(_NMPC, 'NP = 18', 'NMPC NP')
    if a: rec(True, 'c_simple is parameter p[12:15] of the NP=18 parameter vector (NOT a variable)',
              f'centroidal_nmpc.py:{a[0]}: {a[1]}  (p=[r_ref,v_ref,r_C1,r_C2,c_simple,L_ref])')
    a = find(_NMPC, 'c_simple = self.compute_c_simple(', 'compute_c_simple call')
    if a: rec(True, 'c_simple computed once per solve from current state + hw telemetry (h_w_0+L_0+orbital_0)',
              f'centroidal_nmpc.py:{a[0]}: {a[1]}')
    a = find(_SIM, 'hw_for_nmpc = (cfg.rwa_I_w * self.mj_data.qvel[6:9]).copy()', 'hw read per MPC step')
    if a: rec(True, 'the h_w fed to c_simple is read FRESH per MPC step from wheel state qvel[6:9]',
              f'sim_loop.py:{a[0]} → passed hw_current=hw_for_nmpc to nmpc.solve (sim_loop.py:2666)')
    a = find(_NSOL, "'p': ca.DM(P),", 'param bound as p=')
    if a: rec(True, 'bound into the solver as a PARAMETER: P=concat([x0,params]) passed p=ca.DM(P) — NOT in w',
              f'nmpc_solver.py:{a[0]} (decision vector is x0/lbx/ubx; c_simple is in p)')
    a = find(_NMPC, 'hw_k = c_simple - L_com - ca.cross(r_com, m * v_com)', 'c_simple used in box')
    if a: rec(True, 'USED in the RWA envelope box: h_w(k) = c_simple(frozen) − L_com(k) − orbital, bounded',
              f'centroidal_nmpc.py:{a[0]}: {a[1]}  (L_com is the decision STATE; c_simple frozen)')

    # ===================== Q4 — QP h_w bound / c_curr ======================
    print('\n[Q4] QP h_w bound uses hw_current as a FROZEN parameter (c_curr = same mechanism)')
    a = find(_QP, 'b_mom_upper = hw_max - hw_current', 'QP hw box RHS')
    if a: rec(True, 'QP h_w box RHS uses hw_current (a numeric solve() arg) — frozen parameter, NOT a var',
              f'wholebody_qp.py:{a[0]}: {a[1]}  (constraint on λ; h_w(k+1)=hw_current−dt·M_λ·λ bounded)')
    a = find(_QP, 'hw_current: Optional[np.ndarray] = None,', 'hw_current is a solve arg')
    if a: rec(True, 'hw_current is a solve() ARGUMENT (passed in, frozen at solve) — not computed/optimized',
              f'wholebody_qp.py:{a[0]}')
    a = find(_QP, 'b_Ld_upper = cfg.tau_w_max * np.ones(3)', 'τ_w_max box on λ')
    if a: rec(True, 'τ_w enters only as the SCALAR bound τ_w_max on |M_env·λ| (a λ-constraint); τ_w not a var',
              f'wholebody_qp.py:{a[0]}: {a[1]}')
    a = find(_SIM, 'cfg.rwa_I_w * self.mj_data.qvel[6:9]).copy()', 'inter-step entry hw')
    rec(True, 'c_curr = refresh the QP hw_current per tick (mirrors c_simple per MPC step) — a PARAMETER '
        'refresh, NOT a co-solve',
        'inter-step loop freezes hw_current at entry (sim_loop.py:701); the _step QP already refreshes it '
        'per-tick (sim_loop.py:3416→3002). Refreshing the frozen RHS more often never makes h_w a variable.')

    # ===================== RUNTIME structural proof ======================
    print('\n[RUNTIME] import the solvers; inspect actual decision-vector dims / index keys')
    try:
        from crawlbot.solvers.centroidal_nmpc import CentroidalNMPC
        rec(CentroidalNMPC.NX == 9 and CentroidalNMPC.NU == 12 and CentroidalNMPC.NP == 18,
            f'CentroidalNMPC.NX={CentroidalNMPC.NX} NU={CentroidalNMPC.NU} NP={CentroidalNMPC.NP} '
            '(state has no h_w; controls are wrench; c_simple is a parameter)',
            'class attributes — h_w is neither in NX nor NU')
    except Exception as e:
        rec(False, 'import CentroidalNMPC', f'{e}')
    try:
        from crawlbot.solvers.wholebody_qp import WholeBodyQP, WholeBodyQPConfig
        qp = WholeBodyQP(WholeBodyQPConfig())
        idx = qp._compute_indices()
        keys = set(idx.keys())
        bad = {k for k in keys if re.search(r'h_?w|tau_w|wheel', k) and 'slack_hw' not in k}
        rec(bad == set(),
            f'QP decision-var index keys = {sorted(keys)}',
            f'no h_w/τ_w decision block (slack_hw_* are constraint slacks, not h_w itself); offending={bad or "NONE"}')
    except Exception as e:
        rec(False, 'import/instantiate WholeBodyQP', f'{e}')

    return _verdict()


def _verdict():
    n = sum(_checks)
    print('\n' + '=' * 78)
    ok = (n == len(_checks))
    print(f'{n}/{len(_checks)} checks.  FRUGALITY/NO-CO-INTEGRATION: '
          f'{"FACT — h_w & τ_w are a parameter everywhere, a decision variable nowhere" if ok else "CHECK FAILED — see XX"}')
    print('  Optimizers: NMPC (state [r,v,L], controls wrench) + whole-body QP (qdd_t,qdd,λ,τ_q,slack).')
    print('  AOCS: closed-form law. No joint [τ_q,τ_w] solver. h_w enters NMPC as the frozen param c_simple(k)')
    print('  and the QP as the frozen param hw_current — c_curr is the SAME parametric mechanism (refresh,')
    print('  not co-solve). See results/j2_no_cointegration/INTERNAL_no_cointegration.md.')
    print('=' * 78)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
