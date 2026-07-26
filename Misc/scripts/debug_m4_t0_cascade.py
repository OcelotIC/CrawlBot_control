#!/usr/bin/env python3
"""Trace the NMPC → Mapping → QP → MuJoCo cascade at t=0.

Prints, for the very first NMPC+QP call of the single-step M4 sim:
  1. NMPC output (r_com_plan, v_com_plan, a_com_ff, lambda_ref)
  2. CoMToTorsoMapping output (r_b_ref, v_b_ref, a_b_ff, delta)
  3. QP input (what torso reference it receives)
  4. QP output (tau_q commanded joint torques)
  5. Actual MuJoCo torso position after one QP cycle
"""
import os
import sys

_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import numpy as np
import mujoco
import pinocchio as pin

from crawlbot.simulation.sim_loop import SimulationLoop
from crawlbot.simulation.config import SimConfig
from crawlbot.core.com_to_torso_mapping import CoMToTorsoMapping
from crawlbot.core.state_conversions import mujoco_to_pinocchio

URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')
MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')

np.set_printoptions(precision=5, suppress=True, linewidth=100)


def main():
    cfg = SimConfig(
        use_m2_stack=True, alpha_com_soft=5.0, alpha_passivity=1.0,
        enforce_hw_conservation=True, h_max_tight=np.full(3, 5.0),
        w_L_nmpc=1.0, kappa_terminal=0.9,
        aocs_use_legacy_corrected=True,
    )
    sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=cfg)
    sim.setup(n_steps=1, start_a=2, start_b=2)

    # Replicate what sim_loop.run() does before the first _step call:
    # set up the TorsoPlanner for the first step so reference_at(t=0)
    # returns the real dock position, not a zero hold.
    plan = sim.plan
    for i in range(len(plan.phases)):
        if plan.phases[i].phase.value != 'double':
            continue
        if i + 1 < len(plan.phases) and plan.phases[i+1].phase.value != 'double':
            ss_gp = plan.phases[i+1]
            swing_arm = ss_gp.swing_arm
            stance_a = ss_gp.anchor_a_idx
            stance_b = ss_gp.anchor_b_idx
            target_idx = ss_gp.swing_to_idx
            t_ss_start = plan.t_start[i+1]
            t_ss_end = plan.t_end[i+1]
            q_dock = sim._setup_torso_for_step(
                t_ss_start, t_ss_end, swing_arm,
                stance_a, stance_b, swing_arm, target_idx)
            break

    mapping = CoMToTorsoMapping(sim.robot)

    # --- Gather current state from MuJoCo ---
    omega_s = sim.mj_data.qvel[3:6].copy()
    pq, pv = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
    rs = sim.robot.update(pq, pv, omega_struct=omega_s)

    # Reference from TorsoPlanner (same query as sim_loop)
    t = 0.0
    tref = sim.torso_planner.reference_at(t)
    t_horizon = t + cfg.nmpc_N * cfg.nmpc_dt
    cref = sim.torso_planner.com_reference_at(t_horizon)

    print("=" * 76)
    print("  STEP 0: State measured from MuJoCo / Pinocchio")
    print("=" * 76)
    print(f"  r_com (from Pin) :  {rs.r_com}")
    print(f"  v_com (from Pin) :  {rs.v_com}")
    print(f"  L_com (from Pin) :  {rs.L_com}")
    print(f"  total_mass       :  {rs.total_mass:.2f} kg")
    print(f"  torso pos (world):  {rs.oMf_torso.translation}")
    print(f"  omega_s (qvel[3:6]):{omega_s}")
    print()
    print("=" * 76)
    print("  STEP 1: TorsoPlanner references @ t=0")
    print("=" * 76)
    print(f"  tref.p (torso pos ref):  {tref.p}")
    print(f"  tref.v (torso vel ref):  {tref.v}")
    print(f"  tref.a (torso acc ff) :  {tref.a}")
    print(f"  tref.R shape           :  {tref.R.shape}")
    print()
    print(f"  cref.r_com (CoM ref @ t_horizon={t_horizon}):  {cref.r_com}")
    print(f"  cref.v_com                                  :  {cref.v_com}")
    print()

    # --- NMPC solve ---
    if sim.has_rwa:
        hw_for_nmpc = (cfg.rwa_I_w * sim.mj_data.qvel[6:9]).copy()
    else:
        hw_for_nmpc = np.zeros(3)

    from crawlbot.solvers.contact_phase import ContactConfig
    cc_ss = sim.sched.contact_config_at(0.6)  # initial DS config
    cc_nmpc = ContactConfig.from_phase(
        cc_ss.phase,
        sim.sched.anchors_a[2].copy(),
        sim.sched.anchors_b[2].copy())

    rp, vp, Lp, lr, info_n = sim.nmpc.solve(
        r_com=rs.r_com, v_com=rs.v_com, L_com=rs.L_com,
        r_com_ref=cref.r_com, v_com_ref=cref.v_com,
        contact_config=cc_nmpc, warm_start=False,
        hw_current=hw_for_nmpc)
    af = sim.nmpc.compute_feedforward_acceleration(lr)

    print("=" * 76)
    print("  STEP 2: NMPC output (Stage 1)")
    print("=" * 76)
    print(f"  status         :  {info_n.status}  (success={info_n.success})")
    print(f"  cost           :  {info_n.cost:.4f}")
    print(f"  r_com_plan     :  {rp}")
    print(f"  v_com_plan     :  {vp}")
    print(f"  L_com_plan     :  {Lp}")
    print(f"  a_com_ff       :  {af}")
    print(f"  lambda_ref[0:3] (f_A):  {lr[0:3]}")
    print(f"  lambda_ref[3:6] (τ_A):  {lr[3:6]}")
    print(f"  lambda_ref[6:9] (f_B):  {lr[6:9]}")
    print(f"  lambda_ref[9:12](τ_B):  {lr[9:12]}")
    print(f"  |f_A|={np.linalg.norm(lr[0:3]):.3f} N, "
          f"|f_B|={np.linalg.norm(lr[6:9]):.3f} N")
    print()

    # --- Mapping: NMPC CoM reference → torso position reference ---
    r_b_ref, v_b_ref, a_b_ff, delta = mapping.compute(
        r_com_ref=rp, v_com_ref=vp, a_com_ff=af,
        q_current=rs.q, dq_current=rs.v)

    print("=" * 76)
    print("  STEP 3: CoMToTorsoMapping output")
    print("=" * 76)
    print(f"  delta(q)       :  {delta}    (Σ_{{i != torso}} m_i · r_i)")
    print(f"  r_b_ref (mapped):  {r_b_ref}")
    print(f"  v_b_ref (mapped):  {v_b_ref}")
    print(f"  a_b_ff  (mapped):  {a_b_ff}")
    print(f"  m_total/m_b     :  {mapping.ratio:.4f}")
    print()

    # Compare: what torso ref does sim_loop CURRENTLY feed to the QP?
    print("=" * 76)
    print("  STEP 4: What sim_loop CURRENTLY feeds to the QP")
    print("=" * 76)
    print(f"  NOTE: sim_loop.py does NOT use CoMToTorsoMapping.")
    print(f"        It passes tref from TorsoPlanner directly.")
    print(f"  tref.p (passed to QP) :  {tref.p}")
    print(f"  tref.v (passed to QP) :  {tref.v}")
    print(f"  tref.a (passed to QP) :  {tref.a}")
    print()
    print(f"  ==> MAPPING LAYER IS NOT CONNECTED.")
    print(f"  r_com_ref (NMPC)       = {cref.r_com}")
    print(f"  r_b_ref (via mapping)  = {r_b_ref}")
    print(f"  tref.p (actual QP input)= {tref.p}")
    diff_to_mapping = np.linalg.norm(r_b_ref - tref.p)
    diff_to_com_ref = np.linalg.norm(cref.r_com - tref.p)
    print(f"  |r_b_ref - tref.p|   = {diff_to_mapping*1000:.2f} mm")
    print(f"  |cref.r_com - tref.p|= {diff_to_com_ref*1000:.2f} mm")


if __name__ == "__main__":
    main()
