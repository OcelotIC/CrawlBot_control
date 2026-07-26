#!/usr/bin/env python3
"""J2 envelope audit — is Piste A (envelope-coupled passivity budget) QP-formulable?

READ-ONLY reproducer for the verdict in Misc/runs/j2_audit/INTERNAL_j2_envelope.md.
Decides whether  dq^T tau_q + 2*alpha*T_kin <= W_budget  is a LINEAR (or SOC) QP
constraint with W_budget a PRE-SOLVE parameter derived from the planned envelope
margin. No crawlbot/ change, no sim run.

Two parts:
  (1) source anchors — where the envelope lives (NMPC exact Hdot_s box + QP M_lambda
      box), that L_com = M_lambda @ lambda is linear in lambda, that the passivity
      row is linear in tau_q only, that lambda_ref/a_com_ff reach the QP, and that
      tau_q & lambda are distinct decision vars coupled only by the linear EoM.
  (2) numeric demo — using the REAL compute_momentum_map: linearity of L_com in
      lambda; the planned margin tau_w_max - ||M_lambda @ lambda_ref||inf is a scalar
      computable pre-solve; and the QP-proxy (lever-from-robot-CoM) vs the NMPC-exact
      (lever-from-origin, Hdot_s) envelope quantities diverge by r_com x sum(f) at a
      non-zero standoff.

Run:  MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_j2_envelope.py
"""
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_SIM = os.path.join(ROOT, 'crawlbot', 'simulation', 'sim_loop.py')
_QP = os.path.join(ROOT, 'crawlbot', 'solvers', 'wholebody_qp.py')
_NMPC = os.path.join(ROOT, 'crawlbot', 'solvers', 'centroidal_nmpc.py')
_CP = os.path.join(ROOT, 'crawlbot', 'solvers', 'contact_phase.py')

_checks = []


def record(ok, label, evidence):
    _checks.append((bool(ok), label, evidence))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print(f"         {evidence}")


def find(path, needle, label):
    rel = os.path.relpath(path, ROOT)
    try:
        with open(path) as f:
            for i, line in enumerate(f, 1):
                if needle in line:
                    return i, line.strip()
    except OSError as e:
        record(False, label, f'{rel}: unreadable ({e})')
        return None
    record(False, label, f'{rel}: anchor {needle!r} NOT FOUND')
    return None


def main():
    print('=' * 78)
    print('J2 ENVELOPE AUDIT — Piste A QP-formulability  (Fix-A canonical ae0673e)')
    print('=' * 78)

    # ---- Q1: where the envelope lives -------------------------------------
    print('\n[Q1] envelope formulation: NMPC (exact Hdot_s) AND QP (M_lambda box)')
    a = find(_NMPC, 'H_dot_s = (ca.cross(r_C1, f1) + tau1 +', 'NMPC exact Hdot_s')
    if a:
        record(True, 'NMPC: Hdot_s = sum(r_Cj x fj + tauj) about ORIGIN, |Hdot_s|<=tau_w_max (linear in u)',
               f'centroidal_nmpc.py:{a[0]}: {a[1]}')
    a = find(_NMPC, 'ca.dot(f1, f1) - f_max_sq', 'NMPC SOC wrench caps')
    if a:
        record(True, 'NMPC: SOC caps ||f||^2<=f_max^2, ||tau||^2<=tau_max^2 (second-order cone)',
               f'centroidal_nmpc.py:{a[0]}: {a[1]}')
    a = find(_NMPC, 'wrong quantity at non-zero', 'NMPC abandoned the L_com proxy')
    if a:
        record(True, 'NMPC EXPLICITLY abandoned the |L_com| proxy (lever-from-robot-CoM) as wrong at standoff',
               f'centroidal_nmpc.py:{a[0]}: {a[1]}')
    a = find(_QP, '|L̇_robot| = |M_λ·λ| ≤ τ_w_max', 'QP momentum-rate box')
    if not a:
        a = find(_QP, 'Momentum rate box', 'QP momentum-rate box')
    if a:
        record(True, 'QP RE-ENFORCES an envelope box |M_lambda @ lambda| <= tau_w_max (linear rows on lambda)',
               f'wholebody_qp.py:{a[0]}: {a[1]}')
    a = find(_QP, 'A_Ld_upper[:, idx[\'lambda\'][0]: idx[\'lambda\'][1]] = M_lambda', 'QP box row = M_lambda on lambda')
    if a:
        record(True, 'QP box rows act on idx[lambda] with the (parameter) matrix M_lambda -> linear in lambda',
               f'wholebody_qp.py:{a[0]}: {a[1]}')

    # ---- Q2: L_com linear in lambda via the momentum map ------------------
    print('\n[Q2] L_com = M_lambda @ lambda  (M_lambda a 3x12 pre-solve parameter)')
    a = find(_CP, 'def compute_momentum_map', 'momentum map fn')
    a2 = find(_QP, 'M_lambda = compute_momentum_map(r_com, contact_config)', 'QP builds M_lambda from r_com')
    if a and a2:
        record(True, 'M_lambda=compute_momentum_map(r_com,contact_config): depends only on r_com+contacts (params)',
               f'contact_phase.py:{a[0]} ; wholebody_qp.py:{a2[0]}')

    # ---- Q3: planned margin is a pre-solve parameter (DECISIVE) -----------
    print('\n[Q3] DECISIVE: is the envelope margin available BEFORE the QP solve?')
    a = find(_SIM, 'rp, vp, _, lr, info_n = self.nmpc.solve', 'NMPC solves first, returns lr=lambda_ref')
    if a:
        record(True, 'NMPC solves first and returns lr (planned lambda_ref) — before the QP',
               f'sim_loop.py:{a[0]}: {a[1]}')
    a = find(_SIM, 'af = self.nmpc.compute_feedforward_acceleration(lr)', 'planned CoM accel af')
    if a:
        record(True, 'af = planned CoM accel = (1/m)sum(f_ref) — gives the orbital term r_com x m*af',
               f'sim_loop.py:{a[0]}: {a[1]}')
    a = find(_SIM, 'lambda_ref=lr, a_com_ff=af', 'lr+af passed into the QP as parameters')
    if a:
        record(True, 'QP receives lambda_ref=lr AND a_com_ff=af as PARAMETERS -> planned margin computable pre-solve',
               f'sim_loop.py:{a[0]}: {a[1]}')

    # ---- Q4: tau_q <-> lambda link is linear (no product) -----------------
    print('\n[Q4] tau_q and lambda are distinct decision vars, coupled only by the LINEAR EoM')
    a = find(_QP, 'Decision vector z = [q̈_t, q̈, λ, τ_q]', 'decision vector layout')
    if a:
        record(True, 'z = [qdd_t, qdd, lambda, tau_q]: lambda and tau_q are SEPARATE decision variables',
               f'wholebody_qp.py:{a[0]}: {a[1]}')
    a = find(_QP, 'A_dyn[6:, idx[\'tau\'][0]: idx[\'tau\'][1]] = -np.eye(nq)', 'tau_q in EoM equality')
    a2 = find(_QP, 'A_dyn[:, idx[\'lambda\'][0]: idx[\'lambda\'][1]] = -J_robot_T', 'lambda in EoM equality')
    if a and a2:
        record(True, 'EoM equality H*qdd+C = B_u*tau_q + J^T*lambda: BOTH linear, SAME equality, NO product',
               f'wholebody_qp.py:{a[0]} (tau_q) ; :{a2[0]} (lambda)')
    a = find(_QP, 'Linear in τ_q only', 'passivity row is linear in tau_q only')
    if a:
        record(True, 'passivity row depends on tau_q ONLY (A_pass[tau]=dq); tau_q does NOT enter the M_lambda box',
               f'wholebody_qp.py:{a[0]}: {a[1]}')

    # ---- Numeric demo: linearity + scalar margin + proxy/exact divergence -
    print('\n[NUMERIC] real compute_momentum_map: linearity, scalar margin, proxy-vs-exact')
    from crawlbot.solvers.contact_phase import compute_momentum_map, ContactConfig, ContactPhase

    # A 2-contact (DS) config at a non-zero standoff (robot CoM 0.35 m below
    # the contacts, matching the canonical CoM-z standoff of -0.35 m).
    r_com = np.array([0.10, -0.05, 0.35])
    r_CA = np.array([0.40, 0.20, 0.00])
    r_CB = np.array([0.40, -0.20, 0.00])
    cc = ContactConfig.from_phase(ContactPhase.DOUBLE, r_CA, r_CB)
    M = compute_momentum_map(r_com, cc)
    record(M.shape == (3, 12), 'M_lambda is 3x12', f'M_lambda.shape = {M.shape}')

    rng = np.random.default_rng(0)
    lam1 = rng.standard_normal(12)
    lam2 = rng.standard_normal(12)
    lhs = M @ (3.0 * lam1 - 2.0 * lam2)
    rhs = 3.0 * (M @ lam1) - 2.0 * (M @ lam2)
    record(np.allclose(lhs, rhs), 'L_com = M_lambda @ lambda is LINEAR in lambda (superposition holds)',
           f'max|M(3a-2b) - (3Ma-2Mb)| = {np.max(np.abs(lhs - rhs)):.2e}')

    # Planned margin from a planned lambda_ref — a pre-solve SCALAR.
    tau_w_max = 5.0
    lambda_ref = np.array([1.0, 0.5, -0.3, 0.2, 0.1, -0.1,   # f1, tau1
                           -0.8, 0.4, 0.6, -0.2, 0.05, 0.1])  # f2, tau2
    Ldot_proxy = M @ lambda_ref                                   # QP box quantity
    planned_usage_proxy = float(np.max(np.abs(Ldot_proxy)))
    W_margin_proxy = tau_w_max - planned_usage_proxy
    record(np.isscalar(W_margin_proxy) or np.ndim(W_margin_proxy) == 0,
           'planned margin tau_w_max - ||M_lambda @ lambda_ref||inf is a SCALAR (pre-solve parameter)',
           f'||M@lambda_ref||inf = {planned_usage_proxy:.4f}, margin = {W_margin_proxy:.4f} N*m')

    # NMPC-exact Hdot_s about the origin from the SAME lambda_ref.
    f1 = lambda_ref[0:3]; t1 = lambda_ref[3:6]
    f2 = lambda_ref[6:9]; t2 = lambda_ref[9:12]
    Hdot_exact = np.cross(r_CA, f1) + t1 + np.cross(r_CB, f2) + t2
    # Identity: Hdot_exact = Ldot_proxy + r_com x sum(f)
    orbital = np.cross(r_com, f1 + f2)
    record(np.allclose(Hdot_exact, Ldot_proxy + orbital),
           'identity Hdot_s(origin) = L_com(robotCoM) + r_com x sum(f)  [transport term]',
           f'max|Hdot_exact - (Ldot_proxy + orbital)| = {np.max(np.abs(Hdot_exact - (Ldot_proxy + orbital))):.2e}')
    divergence = float(np.max(np.abs(Hdot_exact - Ldot_proxy)))
    record(divergence > 1e-6,
           'QP-proxy (lever-from-robot-CoM) DIVERGES from NMPC-exact (origin) at non-zero standoff',
           f'||Hdot_exact - Ldot_proxy||inf = {divergence:.4f} N*m  (= orbital term, nonzero at standoff)')

    # ---- Verdict ----------------------------------------------------------
    n_pass = sum(ok for ok, _, _ in _checks)
    n_tot = len(_checks)
    print('\n' + '=' * 78)
    print(f'VERDICT: {n_pass}/{n_tot} checks confirmed.')
    fails = [lbl for ok, lbl, _ in _checks if not ok]
    if fails:
        print('FAILED:')
        for lbl in fails:
            print(f'  - {lbl}')
        return 1
    print('PISTE A IS FORMULABLE (YES): dq^T tau_q + 2a T_kin <= W_budget is a LINEAR QP row.')
    print('  - W_budget source: tau_w_max - ||Hdot_s(lambda_ref)||inf (planned margin, pre-solve param).')
    print('  - tau_q (passivity) and lambda (envelope) couple only via the linear EoM (no product).')
    print('  FLAG 1 (units): margin is a TORQUE [N*m]; passivity LHS is POWER [W] -> W_budget needs a')
    print('                  units gain [1/s]; a pre-solve scalar, does NOT affect linearity.')
    print('  FLAG 2 (proxy): the QP box uses the L_com proxy the NMPC abandoned; derive W_budget from')
    print('                  the NMPC-exact Hdot_s(lambda_ref) margin, not the QP box, at non-zero standoff.')
    print('  See Misc/runs/j2_audit/INTERNAL_j2_envelope.md.')
    print('=' * 78)
    return 0


if __name__ == '__main__':
    sys.exit(main())
