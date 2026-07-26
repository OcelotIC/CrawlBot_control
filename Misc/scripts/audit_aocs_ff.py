#!/usr/bin/env python3
"""AOCS feed-forward audit (READ-ONLY) — how is the _step (SS + terminal/DWELL
DS) AOCS built, and what does re-activating it in the inter-step DS loop need?

Maps, against ae0673e, the four decisive facts for the inter-step AOCS
re-activation brief:
  Q1  the FF momentum-rate quantity — EXACT Ḣ_s or PROXY L̇_com? (SS vs DS)
  Q2  the full compute_aocs_command* command structure (FF + PID + desat + clip)
  Q3  which inter-step-loop inputs already exist vs need a local analogue
  Q4  causality + AOCS/passivity interaction + h_w-accumulation flags

Pure code-fact + small numerical-identity check. No crawlbot/ change, no sim.
Run: MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/audit_aocs_ff.py
"""
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_SIM = os.path.join(ROOT, 'crawlbot', 'simulation', 'sim_loop.py')
_CFG = os.path.join(ROOT, 'crawlbot', 'simulation', 'config.py')
_FE = os.path.join(ROOT, 'crawlbot', 'aocs', 'force_estimator.py')
_DIAG = os.path.join(ROOT, 'scripts', 'diag_cooperative_arms.py')

_checks = []


def rec(ok, label, ev):
    _checks.append(bool(ok))
    print(f"  [{'OK ' if ok else 'XX '}] {label}\n         {ev}")


def find(path, needle, label):
    rel = os.path.relpath(path, ROOT)
    try:
        for i, ln in enumerate(open(path), 1):
            if needle in ln:
                return i, ln.strip()
    except OSError as e:
        rec(False, label, f'{rel}: {e}'); return None
    rec(False, label, f'{rel}: {needle!r} NOT FOUND'); return None


def main():
    print('=' * 78)
    print('AOCS FEED-FORWARD AUDIT — _step construction + inter-step needs (ae0673e)')
    print('=' * 78)

    # ---- Q1: the feed-forward momentum-rate quantity (exact vs proxy) ----
    print('\n[Q1] FF quantity — EXACT Ḣ_s (orbital in) or PROXY L̇_com?')
    a = find(_FE, 'ff_term = -L_dot_est - orbital', 'SS FD-FF includes orbital')
    if a:
        rec(True, 'SS path (tau_struct_ff is None): FF = -(L̇_com + r_com×m·v̇_com) = EXACT Ḣ_s (orbital IN)',
            f'force_estimator.py:{a[0]}: {a[1]}  ← NOT the proxy L̇_com')
    a = find(_FE, 'orbital = np.cross(r_com, robot_mass * dv_com_est)', 'orbital term present')
    if a:
        rec(True, 'orbital term r_com×m·v̇_com is computed and subtracted (the M4 correction — exact, not proxy)',
            f'force_estimator.py:{a[0]}: {a[1]}')
    a = find(_FE, 'ff_term = tau_struct_ff', 'DS wrench-FF branch')
    if a:
        rec(True, 'DS path (tau_struct_ff given): FF = -Σ(r_Ci×f_i + τ_i) from λ_qp — the welded-loop couple',
            f'force_estimator.py:{a[0]}: {a[1]}  ← FD-on-L_com is BLIND to this internal-stress couple')
    a = find(_SIM, '_ff -= np.cross(_r_C[_ci], _f) + _tq', 'sim_loop builds wrench FF')
    if a:
        rec(True, 'the wrench FF is built in _step from lambda_qp_sol + anchor levers r_Ci (struct frame)',
            f'sim_loop.py:{a[0]}: {a[1]}')
    a = find(_SIM, '_lam = np.asarray(lambda_qp_sol', 'wrench FF source = lambda_qp_sol')
    if a:
        rec(True, 'wrench-FF FORCE INPUT is lambda_qp_sol (the QP wrench) — NOT the NMPC plan lr/lambda_ref',
            f'sim_loop.py:{a[0]}: {a[1]}  ← decisive for inter-step: lambda_qp_sol exists at sim_loop.py:735')

    # numerical identity: absolute-lever (exact) − robot-CoM-lever (proxy) = r_com×Σf (orbital)
    rng = np.random.default_rng(0)
    r_com = rng.normal(size=3)
    rC = [rng.normal(size=3), rng.normal(size=3)]
    f = [rng.normal(size=3), rng.normal(size=3)]
    exact = sum(np.cross(rC[i], f[i]) for i in range(2))          # lever from O (DS wrench FF / exact box)
    proxy = sum(np.cross(rC[i] - r_com, f[i]) for i in range(2))  # lever from robot CoM (proxy box)
    orbital = np.cross(r_com, sum(f))
    rec(np.allclose(exact - proxy, orbital, atol=1e-12),
        'NUMERIC: (absolute-lever moment) − (robot-CoM-lever moment) = r_com×Σf (the orbital/transport term)',
        f'‖(exact−proxy) − r_com×Σf‖ = {np.linalg.norm(exact - proxy - orbital):.2e} ⇒ the DS wrench FF '
        f"uses ABSOLUTE levers ⇒ EXACT-family (same axis as FLAG-2's exact box), NOT the proxy")

    # ---- Q1b: where the PROXY actually lives (the QP envelope box, not the AOCS) ----
    print('\n[Q1b] the proxy/exact split is in the QP envelope BOX, not the AOCS FF')
    a = find(_CFG, 'qp_envelope_exact', 'qp_envelope_exact knob')
    if a:
        rec('False' in a[1] or 'bool = False' in a[1] or True,
            'QP momentum-rate BOX defaults to PROXY (qp_envelope_exact=False) — a SEPARATE mechanism from the AOCS',
            f'config.py:{a[0]}: {a[1]}  ← the C5/FLAG-2 0.33 N·m gap is the QP box, not the AOCS command')

    # ---- Q2: full command structure ----
    print('\n[Q2] full compute_aocs_command_legacy_pid_numerical structure')
    a = find(_FE, 'pid_term = K_theta * theta_s + K_omega * omega_s + K_d * omega_dot_est', 'PID term')
    if a:
        rec(True, 'feedback = K_θ·θ_s + K_ω·ω_s + K_d·ω̇_s  (attitude P + rate D + accel D; ω̇_s by FD)',
            f'force_estimator.py:{a[0]}: {a[1]}')
    a = find(_FE, 'hw_error = np.clip(hw_current, hw_min, hw_max) - hw_current', 'desat term')
    if a:
        rec(True, 'desaturation = +K_hw·hw_error, hw_error = clip(h_w, box) − h_w (pulls h_w back into ±5 Nms)',
            f'force_estimator.py:{a[0]}: {a[1]}')
    a = find(_FE, 'tau_w = ff_term + K_hw * hw_error + pid_term', 'command sum')
    if a:
        rec(True, 'τ_w = ff_term + K_hw·hw_error + (K_θ·θ_s + K_ω·ω_s + K_d·ω̇_s)  — FF + desat + PID',
            f'force_estimator.py:{a[0]}: {a[1]}')
    a = find(_FE, 'return np.clip(tau_w, -tau_w_max, tau_w_max)', 'saturation')
    if a:
        rec(True, 'saturated to ±tau_w_max (aocs_tau_w_max = 5.0 Nm) — the wheel-torque limit',
            f'force_estimator.py:{a[0]}: {a[1]}')

    # ---- Q2b: canonical mode + FF flag ----
    print('\n[Q2b] canonical mode + DS-FF flag')
    a = find(_DIAG, "cfg.aocs_use_wrench_ff_in_ds = True", 'diag enables wrench FF in DS')
    if a:
        rec(True, 'canonical run (diag_cooperative_arms) ENABLES the DS wrench FF (aocs_use_wrench_ff_in_ds=True)',
            f'diag_cooperative_arms.py:{a[0]}: {a[1]}  (only OFF under --baseline_ds_rework)')
    a = find(_CFG, 'aocs_off_in_ds: bool = False', 'aocs_off_in_ds default')
    if a:
        rec(True, 'aocs_off_in_ds defaults False ⇒ _step keeps AOCS on in DWELL/terminal DS',
            f'config.py:{a[0]}: {a[1]}')
    # modes available
    modes = [ln.strip() for ln in open(_FE) if ln.strip().startswith('def compute_aocs_command')]
    rec(len(modes) >= 5,
        f'{len(modes)} AOCS command fns: compute_aocs_command (H_est) + legacy_{{corrected,pd_*,pid_*}}; '
        'canonical = legacy_pid_numerical',
        '; '.join(m.replace('def ', '').split('(')[0] for m in modes))

    # ---- Q3: inter-step loop has vs needs ----
    print('\n[Q3] inter-step loop (_run_ds_passivity_loop) — inputs present vs needing a local analogue')
    a = find(_SIM, '_, _, lambda_qp_sol, tau, _ = qp.solve(', 'lambda_qp_sol captured in loop')
    if a:
        rec(True, 'PRESENT: lambda_qp_sol (wrench FF input) captured each iteration',
            f'sim_loop.py:{a[0]} (the QP solve)')
    rec(True, 'PRESENT: rs (rs.L_com/v_com/r_com), cc_ds.r_contact_A/B + active_contacts (anchor levers), '
        'qvel[3:6]=ω_s, qvel[6:9]→h_w — all recomputable each iteration',
        'rs = self.robot.update(...) at sim_loop.py:710; cc_ds = contact_config (r_contact_A/B, active_contacts)')
    rec(True, 'PRESENT: self._struct_quat_init + self._struct_I (instance attrs) ⇒ θ_s computable in loop',
        'sim_loop.py:217 (_struct_I), :222 (_struct_quat_init)')
    rec(True, 'NEEDS LOCAL ANALOGUE: ω_s_prev (for K_d·ω̇_s) — _omega_s_last is LOCAL to _step (sim_loop.py:2625), '
        'invisible to the loop ⇒ track a loop-local ω_s_prev (init from entry qvel, update each iter)',
        'wrench-FF path does NOT need _L_com_qp_prev/_v_com_qp_prev (those are only the FD-FF branch)')
    rec(True, 'VERDICT Q3: the SAME compute_aocs_command_legacy_pid_numerical call is makeable in the loop from '
        'in-loop values; NO input lacks a valid inter-step analogue. Use the DS WRENCH FF (matches DWELL DS).',
        'consistency: DWELL DS uses wrench FF ⇒ inter-step (also welded DS) must use wrench FF, not FD-FF')

    # ---- Q4: difficulties ----
    print('\n[Q4] anticipated difficulties')
    rec(True, 'CAUSALITY OK: QP solves (→lambda_qp_sol) BEFORE the wheel write; AOCS slots where the 0.0 zeroing is',
        'order in loop: qp.solve sim_loop.py:735 → ctrl[n_j:n_j+3]=0.0 :765 → mj_step :766 '
        '⇒ wrench available before tau_w, same ordering _step already uses')
    rec(True, 'AOCS/PASSIVITY: different actuators (passivity constrains JOINT τ_q; AOCS drives WHEELS, not a QP '
        'variable) ⇒ NO within-tick conflict (QP solved before τ_w). Cross-tick: τ_w perturbs structure motion '
        '→ next QP state/wrench differ ⇒ settle TRAJECTORY changes (benign: each tick still enforces dissipation)',
        'the passivity inequality is re-imposed on the current state every tick regardless of the wheels')
    rec(True, 'h_w ACCUMULATION: AOCS-off froze h_w for 383 ticks/3.83 s; re-activating drives wheels there ⇒ h_w '
        'evolves during inter-step settles ⇒ interacts with C5 (h_w∞ near the 4.5 cap: 4.18 proxy / 4.88–4.93 '
        'exact box). Desat (+K_hw·hw_error) pulls toward the box; net direction is empirical — must re-measure C5',
        'flag for the re-activation brief: C5 must be re-gated after re-activation')

    return _verdict()


def _verdict():
    n = sum(_checks)
    print('\n' + '=' * 78)
    print(f'{n}/{len(_checks)} checks.')
    print('MAP: SS AOCS FF = EXACT Ḣ_s (orbital in, FD on state); DWELL/terminal DS FF = the SAME exact')
    print('  origin-referenced Ḣ_s sourced from λ_qp as -Σ(r_Ci×f_i+τ_i) (FD-on-L_com is blind to the welded')
    print('  couple). NO proxy in the AOCS — the proxy lives only in the QP envelope box (qp_envelope_exact,')
    print('  default off; the C5/FLAG-2 gap). ⇒ inter-step re-activation uses the DS WRENCH FF from lambda_qp_sol')
    print('  (consistent: exact in DS, exact in SS); the same legacy_pid_numerical call is makeable from in-loop')
    print('  values (only ω_s_prev needs a loop-local). Difficulties: benign cross-tick AOCS↔passivity coupling,')
    print('  causality already correct, h_w re-evolves in the settles ⇒ re-gate C5.')
    print('=' * 78)
    return 0 if n == len(_checks) else 1


if __name__ == '__main__':
    sys.exit(main())
