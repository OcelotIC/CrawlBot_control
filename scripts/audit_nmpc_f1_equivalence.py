"""F1 equivalence + effect proof, at the solver level.

Three checks, in increasing strength:

  1. STRUCTURE — per_stage_refs=True must expand the parameter vector from
     nx + NP to nx + (N+1)·NP and leave the decision-variable and constraint
     counts untouched. Only the parameterization changes, not the problem size.

  2. EQUIVALENCE — feeding the per-stage NLP a single broadcast reference must
     produce the SAME solution as the legacy shared-block NLP, to solver
     tolerance. This is what makes the switch auditable: any behavioural
     difference later is attributable to the reference varying, not to the
     refactor.

  3. EFFECT — feeding it a genuinely varying reference must produce a
     DIFFERENT solution. Without this, check 2 would also pass if the
     per-stage blocks were silently ignored.

Run:
    MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_nmpc_f1_equivalence.py
"""
import numpy as np

from crawlbot.solvers.centroidal_nmpc import CentroidalNMPC, CentroidalNMPCConfig
from crawlbot.solvers.contact_phase import ContactConfig, ContactPhase

N = 8
DT = 0.1
MASS = 71.056


def make(per_stage):
    cfg = CentroidalNMPCConfig(
        robot_mass=MASS, N=N, dt=DT,
        f_max=300.0, tau_max=8.0, L_max=10.0, tau_w_max=2.5, p_max=50.0,
        Wr=100.0 * np.ones(3), Wv=10.0 * np.ones(3),
        Wu_f=0.01, Wu_tau=0.001,
        Qf_r=1000.0 * np.ones(3), Qf_v=100.0 * np.ones(3), Qf_L=10.0,
        w_L=1.0, per_stage_refs=per_stage)
    nmpc = CentroidalNMPC(cfg)
    nmpc.build()
    return nmpc


def scenario():
    cc = ContactConfig.from_phase(
        ContactPhase.DOUBLE,
        np.array([0.6, 0.0, -0.35]), np.array([-0.6, 0.0, -0.35]))
    return dict(
        r_com=np.array([0.02, -0.01, 0.30]),
        v_com=np.array([0.01, 0.0, -0.005]),
        L_com=np.array([0.05, -0.02, 0.01]),
        contact_config=cc,
        hw_current=np.array([0.3, -0.2, 0.1]))


def main():
    legacy = make(False)
    staged = make(True)
    s = scenario()
    NP = CentroidalNMPC.NP

    print('=' * 66)
    print('F1 EQUIVALENCE PROOF')
    print('=' * 66)

    # ---- 1. structure -------------------------------------------------
    lo, st = legacy._nmpc, staged._nmpc
    print('\n[1] STRUCTURE')
    print(f'  parameter blocks     legacy={lo.n_param_blocks}  '
          f'per-stage={st.n_param_blocks}   (expect 1 and {N + 1})')
    print(f'  parameter vector     legacy={lo._np_total}  '
          f'per-stage={st._np_total}   '
          f'(expect {9 + NP} and {9 + (N + 1) * NP})')
    print(f'  decision variables   legacy={len(lo._lbw)}  per-stage={len(st._lbw)}')
    print(f'  constraint rows      legacy={len(lo._lbg)}  per-stage={len(st._lbg)}')
    ok_struct = (lo.n_param_blocks == 1 and st.n_param_blocks == N + 1
                 and lo._np_total == 9 + NP
                 and st._np_total == 9 + (N + 1) * NP
                 and len(lo._lbw) == len(st._lbw)
                 and len(lo._lbg) == len(st._lbg))
    print(f'  -> {"PASS" if ok_struct else "FAIL"}')

    # ---- 2. equivalence under a broadcast reference --------------------
    r_ref = np.array([0.10, 0.02, 0.28])
    v_ref = np.array([0.02, 0.0, -0.01])
    L_ref = np.array([0.02, 0.01, 0.0])

    out_l = legacy.solve(r_com_ref=r_ref, v_com_ref=v_ref, L_com_ref=L_ref,
                         warm_start=False, **s)
    out_s = staged.solve(r_com_ref=r_ref, v_com_ref=v_ref, L_com_ref=L_ref,
                         warm_start=False, **s)
    print('\n[2] EQUIVALENCE  (per-stage NLP fed ONE broadcast reference)')
    print(f'  legacy    success={out_l[4].success}  cost={out_l[4].cost:.9e}')
    print(f'  per-stage success={out_s[4].success}  cost={out_s[4].cost:.9e}')
    d_cost = abs(out_l[4].cost - out_s[4].cost) / max(abs(out_l[4].cost), 1e-12)
    d_r = float(np.max(np.abs(out_l[0] - out_s[0])))
    d_lam = float(np.max(np.abs(out_l[3] - out_s[3])))
    print(f'  |dcost|/cost = {d_cost:.3e}')
    print(f'  max |d r_com_plan| = {d_r:.3e} m')
    print(f'  max |d lambda_0|   = {d_lam:.3e} N')
    ok_equiv = (out_l[4].success and out_s[4].success
                and d_cost < 1e-7 and d_r < 1e-7 and d_lam < 1e-4)
    print(f'  -> {"PASS" if ok_equiv else "FAIL"}')

    # ---- 3. a varying reference must actually change the solution ------
    # Ramp the CoM reference across the horizon instead of holding it.
    r_knots = np.stack([r_ref + np.array([0.02, 0.0, 0.0]) * k
                        for k in range(N + 1)])
    v_knots = np.tile(v_ref, (N + 1, 1))
    L_knots = np.tile(L_ref, (N + 1, 1))
    out_v = staged.solve(r_com_ref=r_knots, v_com_ref=v_knots,
                         L_com_ref=L_knots, warm_start=False, **s)
    print('\n[3] EFFECT  (per-stage NLP fed a RAMPED reference)')
    print(f'  success={out_v[4].success}  cost={out_v[4].cost:.9e}')
    d_r_eff = float(np.max(np.abs(out_v[0] - out_s[0])))
    print(f'  max |d r_com_plan| vs broadcast = {d_r_eff:.6e} m')
    ok_effect = out_v[4].success and d_r_eff > 1e-6
    print(f'  -> {"PASS" if ok_effect else "FAIL"}  '
          f'(must differ, else the per-stage blocks are being ignored)')

    # ---- 4. the guard rails -------------------------------------------
    print('\n[4] GUARDS')
    guards = []
    try:
        legacy.solve(r_com_ref=r_knots, v_com_ref=v_ref, L_com_ref=L_ref,
                     warm_start=False, **s)
        guards.append(('per-knot ref into a legacy NLP', False))
    except ValueError:
        guards.append(('per-knot ref into a legacy NLP', True))
    try:
        staged.solve(r_com_ref=np.zeros((N, 3)), v_com_ref=v_ref,
                     L_com_ref=L_ref, warm_start=False, **s)
        guards.append(('wrong knot count (N instead of N+1)', False))
    except ValueError:
        guards.append(('wrong knot count (N instead of N+1)', True))
    for name, raised in guards:
        print(f'  [{"raises" if raised else "SILENT"}] {name}')
    ok_guard = all(r for _, r in guards)

    ok = ok_struct and ok_equiv and ok_effect and ok_guard
    print(f'\nF1 EQUIVALENCE: {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
