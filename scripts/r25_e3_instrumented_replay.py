#!/usr/bin/env python3
"""R25 export E3 — capture the canonical run's LIVE configs and QP diagnostics.

Read-only with respect to `crawlbot/`: nothing in the control path is modified.
The canonical replay is run exactly as `gate/replay_canonical.py` runs it, with
two observers attached:

  1. a wrapper on `WholeBodyQP.solve` that, after each solve, records the
     equality residual, the inequality violation and the storage (h_w) slacks
     from the solution the solver actually returned;
  2. a one-shot dump of the live `SimConfig`, `WholeBodyQPConfig` and
     `CentroidalNMPCConfig` at the moment the loop is set up.

The wrapper reads the solver's inputs and outputs and returns them untouched,
so the control path is byte-identical by construction; the accompanying gate
run is what proves it.

Why a wrapper instead of a logging channel: the brief allows a logging-only
export under the Gate-0 neutrality pattern, but that would mean editing
`crawlbot/` and re-proving 66-column identity. Observing from the outside gets
the same numbers with no source change at all.

Outputs (results/review_closure/r25/):
    e3_live_config.json     every field of the three live config objects
    e3_qp_diagnostics.json  residual / violation / slack statistics
"""
import json
import os
import sys
import dataclasses

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
os.environ.setdefault('MUJOCO_GL', 'disabled')

OUT = 'results/review_closure/r25'
os.makedirs(OUT, exist_ok=True)

import crawlbot.solvers.wholebody_qp as wqp                       # noqa: E402
import crawlbot.simulation.sim_loop as sl                          # noqa: E402
import scripts.diag_cooperative_arms as dca                        # noqa: E402
from scripts.diag_cooperative_arms import _mutate_mjcf, _mjcf_md5, MJCF  # noqa: E402

REC = {'eq_res': [], 'ineq_viol': [], 'bound_viol': [], 'n_solves': 0}
CFG_DUMP = {}


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (bool, int, float, str, type(None))):
        return v
    return repr(v)


def dump_dataclass(obj):
    if obj is None or not dataclasses.is_dataclass(obj):
        return None
    return {f.name: _jsonable(getattr(obj, f.name))
            for f in dataclasses.fields(obj)}


# ── observer 1: wrap the QP solve ────────────────────────────────────────
import crawlbot.solvers.hierarchical_qp as hq                      # noqa: E402

# Wrap HierarchicalQP.solve: it is the only place the assembled equality and
# inequality blocks and the returned solution coexist (the object does not
# retain them afterwards — it returns (x_opt, info) and drops the matrices).
_orig_hq_solve = hq.HierarchicalQP.solve


def hq_solve_observed(self, *a, **kw):
    x_opt, info = _orig_hq_solve(self, *a, **kw)
    REC['n_solves'] += 1
    if x_opt is not None:
        z = np.asarray(x_opt, float)
        C, d = self._C_eq, self._d_eq
        if C is not None and d is not None and len(d):
            REC['eq_res'].append(
                float(np.max(np.abs(np.asarray(C) @ z - np.asarray(d)))))
        Ci, di = self._C_ineq, self._d_ineq
        if Ci is not None and di is not None and len(di):
            # convention: C_ineq z <= d_ineq  -> violation is the positive part
            REC['ineq_viol'].append(
                float(np.max(np.maximum(np.asarray(Ci) @ z - np.asarray(di), 0.0))))
        lo = np.asarray(self._lb, float); up = np.asarray(self._ub, float)
        REC['bound_viol'].append(float(np.max(np.maximum(
            np.maximum(lo - z, z - up), 0.0))))
    return x_opt, info


hq.HierarchicalQP.solve = hq_solve_observed

# ── observer 2: dump the live configs once the loop is built ─────────────
_orig_setup = sl.SimulationLoop.setup


def setup_observed(self, *a, **kw):
    out = _orig_setup(self, *a, **kw)
    if not CFG_DUMP:
        CFG_DUMP['SimConfig_live'] = dump_dataclass(getattr(self, 'cfg', None))
        for attr in ('qp_ss', 'qp', 'qp_ds'):
            q = getattr(self, attr, None)
            if q is not None and dataclasses.is_dataclass(getattr(q, 'config', None)):
                CFG_DUMP[f'WholeBodyQPConfig_live[{attr}]'] = dump_dataclass(q.config)
        nm = getattr(self, 'nmpc', None)
        if nm is not None:
            CFG_DUMP['CentroidalNMPCConfig_live'] = dump_dataclass(
                getattr(nm, 'config', None))
        pp = getattr(self, 'preplanner', None)
        if pp is not None:
            CFG_DUMP['CoarsePrePlannerConfig_live'] = dump_dataclass(
                getattr(pp, 'config', None))
    return out


sl.SimulationLoop.setup = setup_observed

# ── run the canonical C scenario, verbatim from gate/replay_canonical.py ──
C_KWARGS = dict(
    legacy=False, alpha_torso_lin=0.0, anchor_dx=0.8, mass_ratio=0.01,
    aocs_mode='legacy_pid_numerical', settle_seconds=20.0,
    K_theta=1.0, K_omega=50.0, tau_w_max=2.5,
    n_steps=6, ss_two_task=True, ss_alpha_mom=400.0,
    alpha_torso_pose=2000.0, ss_alpha_ee=1000.0, ss_alpha_posture=2e1,
    ss_alpha_wrench=1.0, ss_kp_torso=3.0, ss_kd_torso=2.5,
    qp_envelope_exact=True,
    interstep_settle_alpha_wrench=3.0, interstep_settle_epsilon_v=5e-3,
    out_dir_override='r25_e3_scratch',
)

with open(MJCF) as f:
    _orig_mjcf = f.read()
_pre = _mjcf_md5(MJCF)
try:
    _mutate_mjcf(damping=0.0, armature=0.05, anchor_dx=0.8, mass_ratio=0.01)
    try:
        dca.main(**C_KWARGS)
    except Exception as e:
        print(f'[r25-e3] main() raised: {type(e).__name__}: {str(e)[:160]}',
              flush=True)
finally:
    with open(MJCF, 'w') as f:
        f.write(_orig_mjcf)
    assert _mjcf_md5(MJCF) == _pre, 'MJCF restore failed'


# storage (h_w) slacks: WholeBodyQP.hw_slack_log is populated on every solve
# (wholebody_qp.py:636-650). Harvest from every QP instance the loop built.
SLACK = []
for _obj in list(globals().values()):
    pass
import gc                                                          # noqa: E402
for _o in gc.get_objects():
    if isinstance(_o, wqp.WholeBodyQP):
        SLACK.extend(_o.hw_slack_log)


def stats(v, label):
    if not v:
        return {'available': False,
                'reason': f'{label} not exposed by the solver objects; '
                          f'no attribute among the probed names was populated'}
    a = np.asarray(v, float)
    return {'available': True, 'n': int(a.size),
            'median': float(np.median(a)), 'p95': float(np.percentile(a, 95)),
            'max': float(np.max(a))}


diag = {
    'n_qp_solves': REC['n_solves'],
    'max_equality_residual': stats(REC['eq_res'], 'equality residual'),
    'max_inequality_violation': stats(REC['ineq_viol'], 'inequality violation'),
    'bound_violation': stats(REC['bound_viol'], 'variable-bound violation'),
    'storage_hw_slack': {
        'n_entries': len(SLACK),
        'n_nonzero': int(sum(1 for e in SLACK if max(e['s_up_max'], e['s_lo_max']) > 1e-12)),
        'max_s_up': float(max((e['s_up_max'] for e in SLACK), default=0.0)),
        'max_s_lo': float(max((e['s_lo_max'] for e in SLACK), default=0.0)),
        'max_norm': float(max((e['s_norm'] for e in SLACK), default=0.0)),
        'source': 'WholeBodyQP.hw_slack_log (wholebody_qp.py:636-650)',
    },
}
json.dump(diag, open(f'{OUT}/e3_qp_diagnostics.json', 'w'), indent=2)
json.dump(CFG_DUMP, open(f'{OUT}/e3_live_config.json', 'w'), indent=2)
print(json.dumps(diag, indent=2))
print(f'\nlive config objects dumped: {list(CFG_DUMP)}')
print(f'wrote {OUT}/e3_live_config.json, {OUT}/e3_qp_diagnostics.json')
