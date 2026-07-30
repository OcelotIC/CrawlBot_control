"""NMPC structural audit — what the NLP actually contains, measured.

Builds `CentroidalNMPC` with the SAME `CentroidalNMPCConfig` that
`sim_loop._setup` constructs from the canonical `SimConfig`, then reports the
realised NLP: decision-variable count, constraint counts by group, which
optional constraint blocks were emitted, and which config fields are inert.

This reads the built solver rather than the source, so a block that is
documented but disabled shows up as absent.
"""
import json
import os

import numpy as np

from crawlbot.simulation.config import SimConfig
from crawlbot.solvers.centroidal_nmpc import CentroidalNMPC, CentroidalNMPCConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANONICAL_MASS = 71.056        # from the two-model consistency check


def build_canonical():
    """Mirror the CANONICAL config, not the SimConfig defaults.

    ⚠ This used to do `cfg = SimConfig()`, which is NOT what the canonical run
    uses and produced a materially wrong audit: it reported the RWA
    conservation box and its terminal set as OFF (ng_path=11, ng_term=0) when
    the canonical has always had both ON (17 / 6).

    `dca.main` builds its config from `run_m7_single_step._make_m7_config()`,
    which passes `enforce_hw_conservation=True`, `h_max_tight=5.0` and
    `kappa_terminal=1.0` as EXPLICIT kwargs — overriding the dataclass
    defaults. This is exactly the trap `centroidal_nmpc.md` §3 already warned
    about: *a dataclass default is not the canonical value*.
    """
    import scripts.run_m7_single_step as _m7
    cfg = _m7._make_m7_config()
    ncfg = CentroidalNMPCConfig(
        robot_mass=CANONICAL_MASS,
        N=cfg.nmpc_N, dt=cfg.nmpc_pred_dt,
        f_max=cfg.nmpc_f_max, tau_max=cfg.nmpc_tau_max,
        L_max=cfg.L_max, tau_w_max=cfg.tau_w_max,
        p_max=cfg.nmpc_p_max,
        Wv=cfg.nmpc_Wv * np.ones(3),
        Wr=cfg.nmpc_Wr * np.ones(3),
        Wu_f=cfg.nmpc_Wu_f, Wu_tau=cfg.nmpc_Wu_tau,
        Qf_r=cfg.nmpc_Qf_r * np.ones(3),
        Qf_v=cfg.nmpc_Qf_v * np.ones(3),
        Qf_L=cfg.nmpc_Qf_L,
        enforce_hw_conservation=cfg.enforce_hw_conservation,
        enforce_hw_terminal=cfg.enforce_hw_terminal,
        h_max_tight=cfg.h_max_tight,
        w_L=cfg.w_L_nmpc,
        kappa_terminal=cfg.kappa_terminal,
        per_stage_refs=cfg.nmpc_per_stage_refs,
        control_period=cfg.nmpc_period)
    return cfg, ncfg


def main():
    cfg, ncfg = build_canonical()
    nmpc = CentroidalNMPC(ncfg)
    nmpc.build()
    s = nmpc._nmpc

    N, nx, nu = s.N, s.nx, s.nu
    n_w = len(s._lbw)
    n_g = len(s._lbg)

    print('=' * 68)
    print('NMPC STRUCTURAL AUDIT — built from the canonical SimConfig')
    print('=' * 68)
    print(f'horizon            N = {N},  dt = {ncfg.dt} s  -> lookahead '
          f'{N * ncfg.dt:.2f} s')
    print(f'control period     nmpc_period = {cfg.nmpc_period} s  ({1/cfg.nmpc_period:.0f} Hz)')
    print(f'  knot spacing == control period? '
          f'{"YES" if abs(ncfg.dt - cfg.nmpc_period) < 1e-12 else "*** NO ***"}')
    print(f'state / control    nx = {nx}, nu = {nu}, np = {nmpc.NP}')
    print(f'integrator         RK4, zero-order hold on u')
    print(f'transcription      multiple shooting')
    print()
    print(f'decision variables n_w = {n_w}   '
          f'(= (N+1)*nx + N*nu = {(N+1)*nx + N*nu})')
    print(f'constraint rows    n_g = {n_g}')

    # Decompose n_g.
    ng_path = s._ng_path
    ng_term = s._ng_terminal
    n_dyn = N * nx
    n_pin = nx
    print(f'  pin x0                 {n_pin:5d}')
    print(f'  dynamics defects       {n_dyn:5d}   (N * nx)')
    print(f'  path constraints       {N * ng_path:5d}   (N * {ng_path})')
    print(f'  terminal constraints   {ng_term:5d}')
    print(f'  --------------------------------')
    print(f'  total                  {n_pin + n_dyn + N*ng_path + ng_term:5d}'
          f'   (reported {n_g})')

    print()
    print('--- path-constraint blocks, as EMITTED ---')
    tau_w_finite = bool(np.isfinite(ncfg.tau_w_max))
    p_max_finite = bool(np.isfinite(ncfg.p_max))
    enforce_hw = bool(ncfg.enforce_hw_conservation)
    blocks = [
        ('SOC ||f_j||^2, ||tau_j||^2', 4, True,
         f'f_max={ncfg.f_max} N, tau_max={ncfg.tau_max} Nm'),
        ('wheel-torque cap |Hdot_s,i|', 6, tau_w_finite,
         f'tau_w_max={ncfg.tau_w_max} Nm'),
        ('linear momentum ||m v||^2', 1, p_max_finite,
         f'p_max={ncfg.p_max} kg m/s'),
        ('RWA conservation box h_w(k)', 6, enforce_hw,
         f'h_max_tight={np.asarray(ncfg.h_max_tight).tolist()}'),
    ]
    for name, rows, on, detail in blocks:
        print(f'  [{"ON " if on else "OFF"}] {name:<30} {rows if on else 0:2d} rows'
              f'   {detail}')
    print(f'  [{"ON " if enforce_hw else "OFF"}] terminal |h_w(N)| <= kappa*h_max  '
          f'{ng_term:2d} rows   kappa={ncfg.kappa_terminal}')

    print()
    print('--- state / control bounds ---')
    print(f'  r_com  unbounded')
    print(f'  v_com  unbounded (norm bounded by the path constraint above)')
    print(f'  L_com  |L_i| <= {ncfg.L_max} Nms   (box on the state)')
    print(f'  u      per-contact; inactive contacts pinned to 0')

    print()
    print('--- cost ---')
    print(f'  stage    Wr={ncfg.Wr[0]}  Wv={ncfg.Wv[0]}  w_L={ncfg.w_L}  '
          f'Wu_f={ncfg.Wu_f}  Wu_tau={ncfg.Wu_tau}')
    print(f'  terminal Qf_r={ncfg.Qf_r[0]}  Qf_v={ncfg.Qf_v[0]}  Qf_L={ncfg.Qf_L}')

    print()
    print('--- solver options actually in force ---')
    for k, v in sorted(s._get_default_solver_options().items()):
        print(f'  {k} = {v}')
    print(f'  (CentroidalNMPCConfig.solver_opts = {ncfg.solver_opts} — never '
          f'overridden by sim_loop, so the defaults above are canonical)')

    print()
    print('--- INERT under this configuration ---')
    inert = []
    if not enforce_hw:
        inert += [
            'c_simple (params p[12:15]) — computed every solve, read by nothing',
            'h_max_tight — only used by the disabled box',
            'kappa_terminal — only used by the disabled terminal constraint',
            'compute_c_simple() — result unused downstream',
        ]
    if not inert:
        print('  (none)')
    for i in inert:
        print(f'  - {i}')

    out = {
        'N': N, 'dt': ncfg.dt, 'lookahead_s': round(N * ncfg.dt, 4),
        'nmpc_period': cfg.nmpc_period,
        'knot_spacing_equals_control_period':
            abs(ncfg.dt - cfg.nmpc_period) < 1e-12,
        'nx': nx, 'nu': nu, 'np': nmpc.NP,
        'n_decision_vars': n_w, 'n_constraint_rows': n_g,
        'ng_path_per_stage': ng_path, 'ng_terminal': ng_term,
        'blocks': {n: {'rows': (r if o else 0), 'active': bool(o)}
                   for n, r, o, _ in blocks},
        'enforce_hw_conservation': enforce_hw,
        'solver_options': s._get_default_solver_options(),
        'inert_fields': inert,
    }
    dest = os.path.join(ROOT, 'results/j2_adjconv/nmpc_structure.json')
    with open(dest, 'w') as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f'\nwrote {dest}')


if __name__ == '__main__':
    main()
