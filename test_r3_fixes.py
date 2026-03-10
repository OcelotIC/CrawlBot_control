#!/usr/bin/env python3
"""
test_r3_fixes.py — Validation des fixes R3 (1+2) dans _setup_torso_for_step
=============================================================================

Vérifie deux propriétés précises :

  FIX 1 : p_t0 (départ de la trajectoire quintic) coincide avec la vraie
           pose du torso lue depuis MuJoCo, et non une pose fictive recalculée
           par IK sur des ancres nominales.

  FIX 2 : L'IK de fin (q_end) utilise la position live de l'ancre stance,
           et non la position nominale stockée dans ContactScheduler.

Protocole :
  - On initialise la simulation normalement (step 0 = identique avant/après fix).
  - On introduit une dérive artificielle de la structure (+5 cm en x) pour
    simuler l'état réaliste au début du step 2.
  - On compare les valeurs retournées par _setup_torso_for_step avant et après
    la dérive, et on vérifie que les quantités correctives sont bien prises en compte.

Usage :
    python test_r3_fixes.py --urdf <path_to_urdf> --mjcf <path_to_mjcf>

Exit code 0 = tous les checks passent.
"""

import sys
import os
import argparse
import numpy as np

np.set_printoptions(precision=5, suppress=True)

PASS = '\033[92m PASS \033[0m'
FAIL = '\033[91m FAIL \033[0m'
n_pass = n_fail = 0


def check(name, condition, detail=''):
    global n_pass, n_fail
    if condition:
        n_pass += 1
        print(f'  [{PASS}] {name}')
    else:
        n_fail += 1
        print(f'  [{FAIL}] {name}  →  {detail}')


parser = argparse.ArgumentParser()
parser.add_argument('--urdf', default='models/VISPA_crawling_fixed.urdf')
parser.add_argument('--mjcf', default='models/VISPA_crawling.xml')
args = parser.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco
import pinocchio as pin
import pinocchio as _pin   # alias used in inline expressions
from simulation_loop import SimulationLoop, SimConfig, mujoco_to_pinocchio
from contact_scheduler import read_anchors_from_mujoco

# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*65)
print('  SETUP — Initialisation SimulationLoop (n_steps=2)')
print('='*65)

sim = SimulationLoop(mjcf_path=args.mjcf, urdf_path=args.urdf)
sim.setup(n_steps=2, start_a=2, start_b=2)

# Récupérer le premier step SS (swing_arm='b', stance_a=2, stance_b=2, target=3)
plan   = sim.plan
ss_gp  = plan.phases[1]          # phase 1 = première SS
t_ss0  = plan.t_start[1]
t_ss1  = plan.t_end[1]
swing  = ss_gp.swing_arm         # 'b'
stance_a = ss_gp.anchor_a_idx   # 2
stance_b = ss_gp.anchor_b_idx   # 2
target   = ss_gp.swing_to_idx   # 3

print(f'  Step info : swing={swing}, stance=({stance_a}a,{stance_b}b), '
      f'target={target}{swing}')
print(f'  SS timing : [{t_ss0:.2f}, {t_ss1:.2f}] s')

# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*65)
print('  TEST A — Fix 1 : p_t0 = pose torso live')
print('='*65)
print('  (structure à sa position initiale → dérive = 0)')

# --- Appel sans dérive (step 1, structure non dérivée) ---
q_ret = sim._setup_torso_for_step(
    t_ss0, t_ss1, swing, stance_a, stance_b, swing, target)

# Lire la vraie pose torso live
pq_live, _ = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
rs_live = sim.robot.update(pq_live, np.zeros(18))
p_live = rs_live.oMf_torso.translation.copy()
R_live = rs_live.oMf_torso.rotation.copy()

# p_t0 est maintenant dans set_hold et add_phase avec p_start = p_t0
# On le lit depuis le torso_planner (fix 3 : passer la pose structure live)
p_struct_q, v_struct_q, om_struct_q = sim.mj_data.qpos[0:3].copy(), sim.mj_data.qvel[0:3].copy(), sim.mj_data.qvel[3:6].copy()
import pinocchio as _pin; _w,_x,_y,_z = sim.mj_data.qpos[3:7]; R_struct_q = _pin.Quaternion(_w,_x,_y,_z).toRotationMatrix()
ref_hold  = sim.torso_planner.reference_at(t_ss0 - 0.01, p_struct=p_struct_q, R_struct=R_struct_q, v_struct=v_struct_q, omega_struct=om_struct_q)
ref_start = sim.torso_planner.reference_at(t_ss0 + 0.01, p_struct=p_struct_q, R_struct=R_struct_q, v_struct=v_struct_q, omega_struct=om_struct_q)

err_hold  = np.linalg.norm(ref_hold.p  - p_live)
err_start = np.linalg.norm(ref_start.p - p_live)

check('set_hold.p == torso_live (|err| < 0.1 mm)',
      err_hold < 1e-4, f'|err|={err_hold*1000:.3f} mm')
check('add_phase.p_start == torso_live (|err| < 0.1 mm)',
      err_start < 1e-4, f'|err|={err_start*1000:.3f} mm')
check('q_start retourné = config live (non re-calculée par IK)',
      np.allclose(q_ret[:7], pq_live[:7], atol=1e-10),
      f'||Δq_torso||={np.linalg.norm(q_ret[:7]-pq_live[:7]):.2e}')

print(f'\n  p_torso_live  = {p_live}')
print(f'  p_hold        = {ref_hold.p}')
print(f'  err_hold      = {err_hold*1000:.4f} mm')

# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*65)
print('  TEST B — Fix 2 : ancre stance live dans IK end')
print('='*65)
print('  Injection dérive structure +5 cm en x dans mj_data ...')

# Lire ancres live AVANT dérive
mj_a_before, mj_b_before = read_anchors_from_mujoco(sim.mj_model, sim.mj_data)
p_stance_a_before = mj_a_before[stance_a].copy()

# Injecter une dérive de structure artificielle (+5 cm en x)
DRIFT = np.array([0.05, 0.0, 0.0])
sim.mj_data.qpos[0:3] += DRIFT
mujoco.mj_forward(sim.mj_model, sim.mj_data)

# Ancres après dérive
mj_a_after, mj_b_after = read_anchors_from_mujoco(sim.mj_model, sim.mj_data)
p_stance_a_after = mj_a_after[stance_a].copy()
p_target_b_after = mj_b_after[target].copy()

print(f'  Ancre stance A avant dérive : {p_stance_a_before}')
print(f'  Ancre stance A après dérive : {p_stance_a_after}')
print(f'  Dérive mesurée              : {(p_stance_a_after - p_stance_a_before)*100} cm')

check('dérive ancre stance visible (> 4.9 cm)',
      np.linalg.norm(p_stance_a_after - p_stance_a_before) > 0.049,
      f'd={np.linalg.norm(p_stance_a_after-p_stance_a_before)*100:.2f} cm')

# Appel après dérive — le fix doit utiliser p_stance_a_after pour l'IK end
# On re-initialise le planneur pour éviter des phases résiduelles
sim.torso_planner.clear_phases()
q_ret2 = sim._setup_torso_for_step(
    t_ss0, t_ss1, swing, stance_a, stance_b, swing, target)

# --- Vérification Fix 1 après dérive ---
pq_live2, _ = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
rs_live2 = sim.robot.update(pq_live2, np.zeros(18))
p_live2 = rs_live2.oMf_torso.translation.copy()

# fix 3 : passer la pose structure courante (avec dérive)
p_sq2 = sim.mj_data.qpos[0:3].copy(); v_sq2 = sim.mj_data.qvel[0:3].copy(); om_sq2 = sim.mj_data.qvel[3:6].copy()
_w,_x,_y,_z = sim.mj_data.qpos[3:7]; R_sq2 = _pin.Quaternion(_w,_x,_y,_z).toRotationMatrix()
ref_hold2 = sim.torso_planner.reference_at(t_ss0 - 0.01, p_struct=p_sq2, R_struct=R_sq2, v_struct=v_sq2, omega_struct=om_sq2)
err_hold2 = np.linalg.norm(ref_hold2.p - p_live2)

check('Fix 1 tient après dérive : set_hold.p == torso_live (< 0.1 mm)',
      err_hold2 < 1e-4, f'|err|={err_hold2*1000:.3f} mm')

# --- Vérification Fix 2 : p_t1 cohérent avec ancres live ---
# L'IK de fin a été calculé avec se3_a_live = (R_nom, p_stance_a_after)
# On vérifie que le torso end prédit est bien décalé de DRIFT par rapport
# à ce qu'il serait sans dérive (le bras stance a bougé, donc le torso suit)
p_sq_drift = sim.mj_data.qpos[0:3].copy(); _w,_x,_y,_z = sim.mj_data.qpos[3:7]; R_sq_drift = _pin.Quaternion(_w,_x,_y,_z).toRotationMatrix()
ref_end2 = sim.torso_planner.reference_at(t_ss1 - 0.01, p_struct=p_sq_drift, R_struct=R_sq_drift)  # fin de phase
p_end2 = ref_end2.p

# Recomputer IK end without drift to get baseline p_end_nodrift
sim.mj_data.qpos[0:3] -= DRIFT
mujoco.mj_forward(sim.mj_model, sim.mj_data)
sim.torso_planner.clear_phases()
q_ret_nodrift = sim._setup_torso_for_step(
    t_ss0, t_ss1, swing, stance_a, stance_b, swing, target)
p_sq_nd = sim.mj_data.qpos[0:3].copy(); _w,_x,_y,_z = sim.mj_data.qpos[3:7]; R_sq_nd = _pin.Quaternion(_w,_x,_y,_z).toRotationMatrix()
ref_end_nodrift = sim.torso_planner.reference_at(t_ss1 - 0.01, p_struct=p_sq_nd, R_struct=R_sq_nd)
p_end_nodrift = ref_end_nodrift.p

delta_end = p_end2 - p_end_nodrift
print(f'\n  p_t1 sans dérive : {p_end_nodrift}')
print(f'  p_t1 avec dérive : {p_end2}')
print(f'  Δp_t1            : {delta_end*100} cm')
print(f'  Dérive injectée  : {DRIFT*100} cm')

# Le torso end doit se décaler approximativement comme la structure
# (pas exactement à cause du frac < 1, mais dans le même sens et même ordre)
cfg_frac = sim.cfg.torso_frac
check(
    f'Δp_t1 en x dans même sens que DRIFT (dérive × torso_frac ≈ {cfg_frac:.1f})',
    delta_end[0] > 0.0,
    f'Δx={delta_end[0]*100:.2f} cm')

check(
    f'|Δp_t1| > 0 (IK end sensible à la dérive stance)',
    np.linalg.norm(delta_end) > 1e-4,
    f'|Δp_t1|={np.linalg.norm(delta_end)*1000:.2f} mm')

# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*65)
print('  TEST C — Continuité C⁰ : v(τ=0) = 0 et v(τ=1) = 0')
print('='*65)

# Remettre la structure à zéro dérive pour ce test
mujoco.mj_forward(sim.mj_model, sim.mj_data)
sim.torso_planner.clear_phases()
sim._setup_torso_for_step(
    t_ss0, t_ss1, swing, stance_a, stance_b, swing, target)

t_traj_start = t_ss0 + sim.cfg.torso_delay * sim.cfg.t_swing
eps = 1e-4
p_sc = sim.mj_data.qpos[0:3].copy(); _w,_x,_y,_z = sim.mj_data.qpos[3:7]; R_sc = _pin.Quaternion(_w,_x,_y,_z).toRotationMatrix()
ref_begin = sim.torso_planner.reference_at(t_traj_start + eps, p_struct=p_sc, R_struct=R_sc)
ref_end_c  = sim.torso_planner.reference_at(t_ss1 - eps,       p_struct=p_sc, R_struct=R_sc)

check('v_lin(τ≈0) ≈ 0 (départ quintic)',
      np.linalg.norm(ref_begin.v[:3]) < 0.01,
      f'||v||={np.linalg.norm(ref_begin.v[:3]):.4f} m/s')
check('v_lin(τ≈1) ≈ 0 (arrivée quintic)',
      np.linalg.norm(ref_end_c.v[:3]) < 0.01,
      f'||v||={np.linalg.norm(ref_end_c.v[:3]):.4f} m/s')
check('v_ang(τ≈0) ≈ 0',
      np.linalg.norm(ref_begin.v[3:]) < 0.01,
      f'||ω||={np.linalg.norm(ref_begin.v[3:]):.4f} rad/s')
check('v_ang(τ≈1) ≈ 0',
      np.linalg.norm(ref_end_c.v[3:]) < 0.01,
      f'||ω||={np.linalg.norm(ref_end_c.v[3:]):.4f} rad/s')

# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*65)
print(f'  RÉSULTATS : {n_pass} PASS, {n_fail} FAIL')
print('='*65)
sys.exit(0 if n_fail == 0 else 1)
