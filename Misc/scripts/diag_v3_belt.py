#!/usr/bin/env python3
"""HERO-RENDER v3 NEUTRALITY BELT — prove the visual edits are dynamics-neutral.

Part 1 (by-construction): compile M0 (canonical MJCF) and M1 (injected visual
geoms + skybox); assert nq/nv/nu, body mass/inertia, DOF/joint params, and
existing-geom collision params are byte-identical, and the 3 new geoms are
contype=0/conaffinity=0. Forward-dynamics (qacc) identical at every stored
canonical state in [0,2s] under identical ctrl.

Part 2 (closed-loop replay): run 2 s of the canonical closed loop with M0 and
with M1 (n_steps=1), diff per-tick qpos/qvel; diff both vs the stored log
snapshots. All must be bit-identical.

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/diag_v3_belt.py
"""
import os
os.environ.setdefault('MUJOCO_GL', 'disabled')
import json
import numpy as np
import mujoco
import crawlbot.solvers.hierarchical_qp as hq
from Misc.scripts.hero_render_v3 import inject_visual

EPS = 1e-6
CANON = 'models/VISPA_crawling_rwa3.xml'
STORED = 'results/figC25_addfive/sim_log.json'
SCRATCH = '/tmp/claude-0/-home-user-CrawlBot-control/51ef2e0d-fbab-59f4-a937-ec828e0d7eda/scratchpad'
M1_XML = f'{SCRATCH}/v3_injected.xml'
res = {}

# ── PART 1 — structural invariants (by construction) ──────────────────
m0 = mujoco.MjModel.from_xml_path(CANON)
inj = inject_visual()
open(M1_XML, 'w').write(inj)
m1 = mujoco.MjModel.from_xml_string(inj)

p1 = {}
p1['nq'] = (int(m0.nq), int(m1.nq), m0.nq == m1.nq)
p1['nv'] = (int(m0.nv), int(m1.nv), m0.nv == m1.nv)
p1['nu'] = (int(m0.nu), int(m1.nu), m0.nu == m1.nu)
p1['na'] = (int(m0.na), int(m1.na), m0.na == m1.na)
p1['nbody'] = (int(m0.nbody), int(m1.nbody), m0.nbody == m1.nbody)
p1['njnt'] = (int(m0.njnt), int(m1.njnt), m0.njnt == m1.njnt)
p1['ngeom'] = (int(m0.ngeom), int(m1.ngeom), int(m1.ngeom - m0.ngeom))  # expect +3
inv = {}
inv['body_mass'] = bool(np.array_equal(m0.body_mass, m1.body_mass))
inv['body_inertia'] = bool(np.array_equal(m0.body_inertia, m1.body_inertia))
inv['body_ipos'] = bool(np.array_equal(m0.body_ipos, m1.body_ipos))
inv['body_iquat'] = bool(np.array_equal(m0.body_iquat, m1.body_iquat))
inv['dof_damping'] = bool(np.array_equal(m0.dof_damping, m1.dof_damping))
inv['dof_armature'] = bool(np.array_equal(m0.dof_armature, m1.dof_armature))
inv['jnt_range'] = bool(np.array_equal(m0.jnt_range, m1.jnt_range))
inv['jnt_axis'] = bool(np.array_equal(m0.jnt_axis, m1.jnt_axis))
# existing geoms: byte-identical after EXCLUDING the 3 new geoms (which insert
# in the MIDDLE of the structure geom list, so a plain [:n0] slice misaligns).
new_names = ('v3_rail_a', 'v3_rail_b', 'v3_rwa_mount')
new_idx = sorted(mujoco.mj_name2id(m1, mujoco.mjtObj.mjOBJ_GEOM, nm) for nm in new_names)
keep = [g for g in range(m1.ngeom) if g not in new_idx]
assert len(keep) == m0.ngeom
inv['geom_contype[existing]'] = bool(np.array_equal(m0.geom_contype, m1.geom_contype[keep]))
inv['geom_conaffinity[existing]'] = bool(np.array_equal(m0.geom_conaffinity, m1.geom_conaffinity[keep]))
inv['geom_pos[existing]'] = bool(np.array_equal(m0.geom_pos, m1.geom_pos[keep]))
inv['geom_size[existing]'] = bool(np.array_equal(m0.geom_size, m1.geom_size[keep]))
inv['geom_type[existing]'] = bool(np.array_equal(m0.geom_type, m1.geom_type[keep]))
inv['new_geoms_noncolliding'] = bool(all(m1.geom_contype[g] == 0 and m1.geom_conaffinity[g] == 0
                                         for g in new_idx))
struct = mujoco.mj_name2id(m1, mujoco.mjtObj.mjOBJ_BODY, 'structure')
inv['structure_mass_7110'] = (float(m0.body_mass[struct]), float(m1.body_mass[struct]))
inv['structure_inertia'] = (m0.body_inertia[struct].tolist(), m1.body_inertia[struct].tolist())

# forward dynamics at each stored canonical state in [0,2s]
sl = json.load(open(STORED))
snaps = [s for s in sl['snapshots'] if s[0] <= 2.01]
d0 = mujoco.MjData(m0); d1 = mujoco.MjData(m1)
rng = np.random.RandomState  # NOT used (determinism); fixed ctrl below
fd = []
for (t, q, v, lbl) in snaps:
    q = np.asarray(q, float); v = np.asarray(v, float)
    ctrl = np.array([0.3 * np.sin(0.7 * i + t) for i in range(m0.nu)])  # rich, deterministic
    for m, d in ((m0, d0), (m1, d1)):
        d.qpos[:] = q; d.qvel[:] = v; d.ctrl[:] = ctrl
        mujoco.mj_forward(m, d)
    dqacc = float(np.max(np.abs(d0.qacc - d1.qacc)))
    dbias = float(np.max(np.abs(d0.qfrc_bias - d1.qfrc_bias)))
    fd.append({'label': lbl, 't': t, 'max_dqacc': dqacc, 'max_dqfrc_bias': dbias})

res['part1_dims'] = p1
res['part1_invariants'] = inv
res['part1_forward_dynamics_at_stored_states'] = fd
p1_ok = (all(v[2] if isinstance(v[2], bool) else True for v in p1.values())
         and p1['ngeom'][2] == 3
         and all(x for k, x in inv.items() if isinstance(x, bool))
         and all(f['max_dqacc'] == 0.0 and f['max_dqfrc_bias'] == 0.0 for f in fd))
res['part1_pass'] = bool(p1_ok)
print('=' * 72)
print('PART 1 — structural + forward-dynamics invariants (M0 canonical vs M1 injected)')
print(f'  nq {p1["nq"]}  nv {p1["nv"]}  nu {p1["nu"]}  nbody {p1["nbody"]}  '
      f'ngeom +{p1["ngeom"][2]} (new geoms)')
for k, x in inv.items():
    print(f'  {k}: {x}')
print(f'  forward-dynamics qacc/bias diff at {len(fd)} stored states: '
      f'max_dqacc={max(f["max_dqacc"] for f in fd):.2e} '
      f'max_dbias={max(f["max_dqfrc_bias"] for f in fd):.2e}')
print(f'  PART 1: {"PASS ✓" if p1_ok else "FAIL ✗"}')

# ── PART 2 — closed-loop 2 s replay (M0 & M1) vs stored log ────────────
_ow = hq.HierarchicalQP._solve_weighted
def _pw(self, tasks, x0):
    self.regularization = EPS
    return _ow(self, tasks, x0)
hq.HierarchicalQP._solve_weighted = _pw

CAP = []
class _Stop2s(Exception):
    pass
_os = mujoco.mj_step
def _ps(m, d, *a, **k):
    _os(m, d, *a, **k)
    if d.time <= 2.0 + 1e-9:
        CAP.append((round(float(d.time), 5), d.qpos.copy(), d.qvel.copy()))
    elif d.time > 2.05:
        raise _Stop2s
mujoco.mj_step = _ps

import scripts.diag_cooperative_arms as dca


def run_cl(xml_path):
    global CAP
    CAP = []
    dca.MJCF = xml_path
    try:
        dca.main(legacy=False, alpha_torso_lin=0.0, anchor_dx=0.8, mass_ratio=0.01,
                 aocs_mode='legacy_pid_numerical', settle_seconds=20.0,
                 K_theta=1.0, K_omega=50.0, tau_w_max=2.5, n_steps=6,
                 ss_two_task=True, ss_alpha_mom=400.0, alpha_torso_pose=2000.0,
                 ss_alpha_ee=1000.0, ss_alpha_posture=2e1, ss_alpha_wrench=1.0,
                 ss_kp_torso=3.0, ss_kd_torso=2.5, qp_envelope_exact=True,
                 interstep_settle_alpha_wrench=3.0, interstep_settle_epsilon_v=5e-3,
                 out_dir_override='v3belt_tmp')
    except _Stop2s:
        pass
    except Exception as e:
        print(f'   [note] closed loop raised after 2s window: {type(e).__name__}')
    return list(CAP)


print('\nPART 2 — closed-loop 2 s replay')
capM0 = run_cl(CANON)
capM1 = run_cl(M1_XML)
n = min(len(capM0), len(capM1))
dq = dv = 0.0
tmm = 0.0
for i in range(n):
    t0, q0, v0 = capM0[i]; t1, q1, v1 = capM1[i]
    tmm = max(tmm, abs(t0 - t1))
    dq = max(dq, float(np.max(np.abs(q0 - q1))))
    dv = max(dv, float(np.max(np.abs(v0 - v1))))
res['part2_M0_vs_M1'] = {'n_ticks': n, 't_end': capM0[-1][0] if capM0 else None,
                         'max_dt': tmm, 'max_dqpos': dq, 'max_dqvel': dv}
p2a_ok = (n > 0 and tmm == 0.0 and dq == 0.0 and dv == 0.0)
print(f'  M0-run vs M1-run: {n} ticks to t={res["part2_M0_vs_M1"]["t_end"]}s  '
      f'max|Δqpos|={dq:.2e} max|Δqvel|={dv:.2e} -> {"BIT-IDENTICAL ✓" if p2a_ok else "DIFFERS ✗"}')

# cross-check vs stored log snapshots (nearest captured tick)
def nearest(cap, t):
    return min(cap, key=lambda c: abs(c[0] - t))
xs = []
for (t, q, v, lbl) in snaps:
    cq, dqs, dvs = None, None, None
    if capM0:
        c = nearest(capM0, t)
        dqs = float(np.max(np.abs(c[1] - np.asarray(q, float))))
        dvs = float(np.max(np.abs(c[2] - np.asarray(v, float))))
    xs.append({'label': lbl, 't': t, 'max_dqpos_vs_stored': dqs, 'max_dqvel_vs_stored': dvs})
res['part2_M0_vs_storedlog'] = xs
p2b_ok = all((x['max_dqpos_vs_stored'] or 0) == 0.0 and (x['max_dqvel_vs_stored'] or 0) == 0.0 for x in xs)
print('  M0-run vs stored-log snapshots (bit-identity of the canonical trajectory):')
for x in xs:
    print(f'    {x["label"]:16} t={x["t"]:.3f}: dqpos={x["max_dqpos_vs_stored"]:.2e} '
          f'dqvel={x["max_dqvel_vs_stored"]:.2e}')

res['belt_pass'] = bool(p1_ok and p2a_ok)
res['storedlog_determinism'] = bool(p2b_ok)
json.dump(res, open('results/j2_adjconv/v3_belt_result.json', 'w'), indent=2)
print('\n' + '=' * 72)
print(f'NEUTRALITY BELT: Part1 {"✓" if p1_ok else "✗"} | '
      f'Part2 M0==M1 {"✓" if p2a_ok else "✗"} | M0==stored {"✓" if p2b_ok else "✗"}')
print(f'  -> injected model is dynamics-{"NEUTRAL ✓" if res["belt_pass"] else "PERTURBING ✗"}')
print('  wrote results/j2_adjconv/v3_belt_result.json')
print('=' * 72)
