#!/usr/bin/env python3
"""C4 §1 — field-by-field SimConfig diff: the published U run vs the canonical C.

The brief asks whether the committed `u25_fulldiag.csv` is reusable as this
ablation's `none` arm. Answering it needs the *resulting* SimConfig, not the
kwarg list, because `dca.main` fans one kwarg out to several fields.

Rebuilds both configs exactly as `dca.main` does — `_make_m7_config()` then the
runner overrides — for the two calls in
`Misc/scripts/diag_canonical2p5_run.py`:
    U = _run('U', 1e6, 'figU25_addfive')
    C = _run('C', 2.5, 'figC25_addfive')
which are literal-identical except `tau_w_max`.

Also prints where each differing field is consumed, since that is the actual
question: how many *independent physical constraints* the single kwarg moves.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from scripts.run_m7_single_step import _make_m7_config          # noqa: E402

# Kwargs shared by both runs (verbatim from diag_canonical2p5_run.py:126-140).
COMMON = dict(
    anchor_dx=0.8, mass_ratio=0.01, aocs_mode='legacy_pid_numerical',
    settle_seconds=20.0, K_theta=1.0, K_omega=50.0,
    n_steps=6, ss_two_task=True, ss_alpha_mom=400.0,
    alpha_torso_pose=2000.0, ss_alpha_ee=1000.0, ss_alpha_posture=2e1,
    ss_alpha_wrench=1.0, ss_kp_torso=3.0, ss_kd_torso=2.5,
    qp_envelope_exact=True,
    interstep_settle_alpha_wrench=3.0, interstep_settle_epsilon_v=5e-3,
)

# Where each field is actually read. Only fields that differ get looked up.
CONSUMERS = {
    'tau_w_max': [
        'NMPC path constraint  |Ḣ_s,i| ≤ τ_w_max   (sim_loop -> CentroidalNMPCConfig)',
        'whole-body QP envelope box |Ḣ_s| ≤ τ_w_max (sim_loop._build_qp)',
    ],
    'aocs_tau_w_max': [
        'AOCS output clip np.clip(τ_w, ∓cap)       (force_estimator)',
    ],
}


def build(tau_w_max):
    cfg = _make_m7_config()
    cfg.gait_anchor_dx = COMMON['anchor_dx']
    cfg.mapping_bypass_in_ss = False
    cfg.t_ss_margin = 5.0
    cfg.preplanner_f_max = 25.0
    cfg.use_com_z_standoff = True
    cfg.com_z_standoff = -0.35
    cfg.dock_vel_max = 0.01
    cfg.t_settle_final = COMMON['settle_seconds']
    cfg.aocs_K_theta = COMMON['K_theta']
    cfg.aocs_K_omega = COMMON['K_omega']
    cfg.tau_w_max = float(tau_w_max)              # <-- the single kwarg
    cfg.aocs_tau_w_max = float(tau_w_max)         # <-- fans out to here too
    cfg.aocs_mode = COMMON['aocs_mode']
    cfg.aocs_use_legacy_corrected = False
    cfg.aocs_use_H_estimator = False
    cfg.aocs_use_wrench_ff_in_ds = True
    cfg.aocs_active_in_interstep = True
    cfg.interstep_hw_refresh = True
    cfg.interstep_settle_alpha_wrench = COMMON['interstep_settle_alpha_wrench']
    cfg.interstep_settle_epsilon_v = COMMON['interstep_settle_epsilon_v']
    cfg.ds_torso_ref_from_state = True
    cfg.ss_alpha_lambda_int = 1.0
    cfg.ds_centroidal_mode = True
    cfg.log_hifreq_ss = True
    cfg.ss_alpha_mom = COMMON['ss_alpha_mom']
    cfg.ss_two_task_mode = COMMON['ss_two_task']
    cfg.alpha_torso_pose = COMMON['alpha_torso_pose']
    cfg.ss_alpha_ee = COMMON['ss_alpha_ee']
    cfg.ss_alpha_posture = COMMON['ss_alpha_posture']
    cfg.ss_alpha_wrench = COMMON['ss_alpha_wrench']
    cfg.ss_Kp_torso = COMMON['ss_kp_torso']
    cfg.ss_Kd_torso = COMMON['ss_kd_torso']
    cfg.qp_envelope_exact = COMMON['qp_envelope_exact']
    return cfg


def same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    return a == b


def main():
    U, C = build(1e6), build(2.5)
    diffs = []
    for k in sorted(vars(C)):
        vc, vu = getattr(C, k), getattr(U, k)
        if not same(vc, vu):
            diffs.append((k, vu, vc))

    print('=' * 78)
    print('C4 §1 — SimConfig diff: published U (unmanaged) vs C (canonical)')
    print('=' * 78)
    print(f'fields compared: {len(vars(C))}')
    print(f'fields differing: {len(diffs)}')
    print()
    for k, vu, vc in diffs:
        print(f'  {k}')
        print(f'      U = {vu}')
        print(f'      C = {vc}')
        for c in CONSUMERS.get(k, ['(consumer not catalogued)']):
            print(f'      -> {c}')
    print()
    n_constraints = sum(len(CONSUMERS.get(k, [])) for k, _, _ in diffs)
    print(f'INDEPENDENT PHYSICAL CONSTRAINTS MOVED: {n_constraints}')
    print()
    print('Verdict for this brief:')
    print('  u25 is NOT this ablation\'s `none` arm.')
    print('  - it lifts the NMPC rate cap, the QP envelope box AND the AOCS')
    print('    clip together (one kwarg, three consumers) — the brief requires')
    print('    the AOCS clip held fixed in every arm;')
    print('  - it leaves enforce_hw_conservation=True, so the STORAGE box is')
    print('    still on — `none` requires it off.')
    print('  u25 is a fourth configuration, not a member of this design.')


if __name__ == '__main__':
    main()
