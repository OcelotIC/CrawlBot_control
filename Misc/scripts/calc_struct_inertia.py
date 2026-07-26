#!/usr/bin/env python3
"""Structure-inertia mini-calc (READ-ONLY) — to convert the dock residual
angular momentum [N·m·s] into a structure rotation rate [rad/s, °/s].

Reads the canonical model's inertial properties (no sim run) and, at one
representative dock configuration (a committed snapshot), reports:
  1. the free-floating STRUCTURE body inertia (mass + tensor) about its CoM,
  2. the COMPOSITE system inertia about the system CoM — the inertia that
     relates subtree_angmom[0] to the welded-system rotation rate,
  3. the per-axis values + the dominant axis of the measured residual
     subtree_angmom[0] vector,
  4. reference angular-momentum / envelope values.

subtree_angmom[0] is the total-system angular momentum about the system CoM,
expressed in the WORLD frame, so the composite inertia is reported in the
WORLD frame at the dock config (≈ structure body frame, attitude error <1°).

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/calc_struct_inertia.py
"""
import json
import os
import numpy as np
import mujoco

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MJCF = os.path.join(ROOT, 'models', 'VISPA_crawling_rwa3.xml')
DOCK_RUN = os.path.join(ROOT, 'Misc', 'runs', 'fixA_gate', 'sim_log.json')


def quat2mat(q_wxyz):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q_wxyz, dtype=float))
    return m.reshape(3, 3)


def principal(I):
    w = np.linalg.eigvalsh(I)
    return np.sort(w)[::-1]


def main():
    m = mujoco.MjModel.from_xml_path(MJCF)
    d = mujoco.MjData(m)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'structure')

    print('=' * 76)
    print('STRUCTURE-INERTIA MINI-CALC  (canonical VISPA_crawling_rwa3, ae0673e)')
    print('=' * 76)

    # ---- 1. STRUCTURE body inertia (config-independent) -----------------
    m_struct = float(m.body_mass[sid])
    Iprin = np.asarray(m.body_inertia[sid], dtype=float)         # principal diag
    Riq = quat2mat(m.body_iquat[sid])                            # principal->body
    I_body = Riq @ np.diag(Iprin) @ Riq.T                        # in structure body frame
    print('\n[1] STRUCTURE body (id=%d "structure")  — about its CoM (body_ipos=%s)'
          % (sid, np.array2string(m.body_ipos[sid], precision=3)))
    print('    mass m_struct          = %.3f kg' % m_struct)
    print('    principal inertia      = [%.1f, %.1f, %.1f] kg·m²  (in the principal frame)'
          % tuple(Iprin))
    print('    body_iquat (wxyz)      = %s  (= 90° about Y ⇒ principal X↔Z vs body)'
          % np.array2string(m.body_iquat[sid], precision=4))
    print('    inertia in BODY frame  = diag([%.1f, %.1f, %.1f]) kg·m²  (Ixx, Iyy, Izz)'
          % (I_body[0, 0], I_body[1, 1], I_body[2, 2]))
    print('      off-diagonals (body) : max|off| = %.2e (≈0 ⇒ body axes are principal)'
          % np.max(np.abs(I_body - np.diag(np.diag(I_body)))))

    # ---- dock configuration (one committed snapshot) --------------------
    sl = json.load(open(DOCK_RUN))
    # 'final' = post-settle traversal end (the 0.003977 headline residual);
    # fall back to the last dock if absent.
    cands = [s for s in sl['snapshots'] if s[3] == 'final'] or \
            [s for s in sl['snapshots'] if s[3].startswith('dock')]
    snap = cands[-1]
    snap_label = snap[3]
    t_d, q_d, v_d = snap[0], np.array(snap[1]), np.array(snap[2])
    d.qpos[:] = q_d
    d.qvel[:] = v_d
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    mujoco.mj_comVel(m, d)
    mujoco.mj_subtreeVel(m, d)
    r_com_sys = np.array(d.subtree_com[0])                       # system CoM (world)

    # ---- 2. COMPOSITE system inertia about system CoM (world frame) -----
    def composite_about(point, bodies):
        I = np.zeros((3, 3))
        for i in bodies:
            mi = float(m.body_mass[i])
            if mi == 0.0:
                continue
            Rw = np.array(d.ximat[i]).reshape(3, 3)              # inertial frame -> world
            Iw = Rw @ np.diag(m.body_inertia[i]) @ Rw.T          # about body CoM, world
            r = np.array(d.xipos[i]) - point                    # body CoM - point
            I += Iw + mi * (float(r @ r) * np.eye(3) - np.outer(r, r))
        return I

    all_bodies = list(range(1, m.nbody))
    I_sys = composite_about(r_com_sys, all_bodies)
    I_struct_about_sys = composite_about(r_com_sys, [sid])       # structure alone, shifted
    off = np.max(np.abs(I_sys - np.diag(np.diag(I_sys))))
    print('\n[2] COMPOSITE SYSTEM inertia about the SYSTEM CoM — WORLD frame, dock config')
    print("    ('%s' snapshot, t=%.2fs; this is the inertia subtree_angmom[0] sees)"
          % (snap_label, t_d))
    print('    system CoM (world)     = %s m' % np.array2string(r_com_sys, precision=4))
    print('    structure CoM (world)  = %s m' % np.array2string(np.array(d.xipos[sid]), precision=4))
    print('    |sysCoM - structCoM|   = %.4f m  (parallel-axis lever)'
          % np.linalg.norm(r_com_sys - np.array(d.xipos[sid])))
    print('    I_sys (3×3) kg·m² =\n' + np.array2string(I_sys, precision=1, suppress_small=True))
    print('    diag(I_sys)            = [%.1f, %.1f, %.1f] kg·m²  (Ixx, Iyy, Izz, world)'
          % (I_sys[0, 0], I_sys[1, 1], I_sys[2, 2]))
    print('    principal(I_sys)       = [%.1f, %.1f, %.1f] kg·m²' % tuple(principal(I_sys)))
    print('    max|off-diagonal|      = %.2f kg·m²' % off)
    print('    (structure-alone about system CoM, diag = [%.1f, %.1f, %.1f] — system adds %.1f%%)'
          % (I_struct_about_sys[0, 0], I_struct_about_sys[1, 1], I_struct_about_sys[2, 2],
             100.0 * (np.trace(I_sys) / np.trace(I_struct_about_sys) - 1.0)))

    # ---- 3. dominant residual axis --------------------------------------
    L = np.array(d.subtree_angmom[0])                            # world frame
    ax = int(np.argmax(np.abs(L)))
    print("\n[3] RESIDUAL subtree_angmom[0] at '%s' (WORLD frame) — direction" % snap_label)
    print('    L_residual             = %s N·m·s' % np.array2string(L, precision=6))
    print('    ‖L_residual‖           = %.6f N·m·s' % np.linalg.norm(L))
    print('    dominant axis          = %s (|L_%s| = %.6f, %.0f%% of norm)'
          % ('XYZ'[ax], 'xyz'[ax], abs(L[ax]),
             100.0 * abs(L[ax]) / (np.linalg.norm(L) + 1e-30)))
    print('    NOTE: from the fixA_gate run (residual 0.003977 = the pose-gated /')
    print('          ε_twist≥0.007 Fix-C residual). The ε_twist=0.005 run (0.001005)')
    print('          retained only its norm, not the vector — direction shown is the')
    print('          0.003977 case; treat the axis split as representative.')

    # ---- 4. reference values --------------------------------------------
    print('\n[4] REFERENCE angular-momentum + envelope (do not re-measure)')
    print('    Fix-A residual              ≈ 0.003977 N·m·s  (ae0673e, traversal-final)')
    print('    Fix-C residual @ε_twist0.005 ≈ 0.001005 N·m·s  (this branch)')
    print('    τ_w,max (wheel-torque cap)   =  5.0 N·m   (per axis)')
    print('    h_max  (wheel momentum)      = ±5.0 N·m·s (per axis; L_max robot = 10)')

    # ---- convenience: proper ω = I⁻¹·L (USER does the final conversion) --
    # The residual is NOT axis-aligned (X≈0.0030, Z≈0.0026), and I_xx≪I_zz,
    # so use the full-tensor solve, not ‖L‖/I_x.
    w_vec = np.linalg.solve(I_sys, L)                           # rad/s, world
    scale = 0.001005 / (np.linalg.norm(L) + 1e-30)             # Fix-C/Fix-A norm ratio
    print('\n[ref only — you do the conversion] proper ω = I_sys⁻¹·L (full tensor):')
    print('    Fix-A (L known):  ω = %s rad/s' % np.array2string(w_vec, precision=3))
    print('                      ω = %s °/s   ‖ω‖=%.3e °/s'
          % (np.array2string(np.degrees(w_vec), precision=4), np.degrees(np.linalg.norm(w_vec))))
    print('    Fix-C (≈same dir, ×%.3f):  ‖ω‖ ≈ %.3e °/s'
          % (scale, scale * np.degrees(np.linalg.norm(w_vec))))
    print('    rate is X-dominated: |L_x|≈|L_z| but I_xx=%.0f ≪ I_zz=%.0f, so ω_x leads.'
          % (I_sys[0, 0], I_sys[2, 2]))
    print('    (Fix-C vector not retained — only its norm; rate uses the Fix-A direction.)')
    print('=' * 76)


if __name__ == '__main__':
    main()
