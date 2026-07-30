"""F2 §7: log what the NMPC is actually given as `hw_current`.

The bite test (h_max_tight = 3.5) left an unexplained gap: the exported h_w
reached -3.8146 Nms on z during SS while every NMPC solve succeeded, even though
`h_w(0)` == `hw_current` identically and the isolated NLP provably rejects that
state.

Two earlier versions of this probe FAILED to test what they claimed, both
silently:

  1. `SimConfig.h_max_tight = np.full(3, 3.5)` does not override a dataclass
     field built by `field(default_factory=...)` — instances keep the factory
     value.
  2. Patching config.py then `importlib.reload`ing it does not help either:
     `sim_loop` did `from .config import SimConfig` at import time and keeps a
     reference to the ORIGINAL class object.

The only reliable way is the one the sweep driver uses: patch the source, then
run the simulation in a FRESH SUBPROCESS so every import sees the patched file.
This script is its own outer and inner half, selected by an env var.

Run:
    MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_nmpc_f2_hw_probe.py
"""
import json
import os
import re
import subprocess
import sys

import numpy as np

os.environ.setdefault('MUJOCO_GL', 'disabled')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CFG = os.path.join(ROOT, 'crawlbot/simulation/config.py')
DEST = os.path.join(ROOT, 'results/j2_adjconv/nmpc_sweep/f2_hw_probe.json')
INNER = 'CRAWLBOT_F2_PROBE_INNER'

PATCHES = (
    (r'^(\s*enforce_hw_conservation: bool = )(True|False)', 'True'),
    (r'^(\s*enforce_hw_terminal: Optional\[bool\] = )(True|False|None)', 'True'),
    (r'^(\s*h_max_tight: np\.ndarray = field\(default_factory=lambda: np\.full\(3, )([0-9.]+)',
     '3.5'),
)


def outer():
    """Patch config.py, run the inner half in a subprocess, restore."""
    orig = open(CFG).read()
    patched = orig
    for pat, val in PATCHES:
        rx = re.compile(pat, re.M)
        assert rx.search(patched), f'patch target missing: {pat}'
        patched = rx.sub(lambda m, v=val: m.group(1) + v, patched, count=1)
    open(CFG, 'w').write(patched)
    try:
        env = dict(os.environ, MUJOCO_GL='disabled', PYTHONPATH=ROOT,
                   **{INNER: '1'})
        p = subprocess.run([sys.executable, os.path.abspath(__file__)],
                           cwd=ROOT, env=env, timeout=3600)
        rc = p.returncode
    finally:
        open(CFG, 'w').write(orig)
        print('[probe] config.py restored')
    if rc != 0:
        raise SystemExit(rc)
    report()


def report():
    rec = json.load(open(DEST))
    hw = np.array([r['hw_current'] for r in rec])
    hm = np.array([r['h_max'] for r in rec])
    viol = [r for r in rec if r['violates_box']]
    fail = [r for r in rec if not r['success']]
    print('\n' + '=' * 70)
    print('WHAT THE NMPC ACTUALLY RECEIVED AS hw_current')
    print('=' * 70)
    print(f'  solves recorded          : {len(rec)}')
    print(f'  box enforced on          : {sum(r["box_enforced"] for r in rec)}')
    print(f'  h_max seen by the NLP    : {np.unique(hm, axis=0).tolist()}')
    print(f'  hw_current per-axis peak : x={np.abs(hw[:,0]).max():.4f}  '
          f'y={np.abs(hw[:,1]).max():.4f}  z={np.abs(hw[:,2]).max():.4f}')
    print(f'  solves whose hw_current already violates : {len(viol)}')
    print(f'  solves reported infeasible               : {len(fail)}')
    if viol:
        w = max(viol, key=lambda r: r['excess'])
        print(f'  worst: hw={np.round(w["hw_current"],4).tolist()} '
              f'axis={"xyz"[w["worst_axis"]]} excess={w["excess"]:+.4f} '
              f'-> success={w["success"]} status={w["status"]}')
    print(f'\nwrote {DEST}')


def inner():
    import crawlbot.solvers.hierarchical_qp as hq
    from crawlbot.solvers.centroidal_nmpc import CentroidalNMPC
    from crawlbot.simulation.config import SimConfig

    assert SimConfig().h_max_tight[0] == 3.5, 'config patch did not reach the inner process'
    assert SimConfig().enforce_hw_conservation is True
    print('[probe] inner: box ON, terminal ON, h_max_tight = 3.5', flush=True)

    EPS = 1e-6
    _ow = hq.HierarchicalQP._solve_weighted

    def _pw(self, tasks, x0):
        self.regularization = EPS
        return _ow(self, tasks, x0)
    hq.HierarchicalQP._solve_weighted = _pw

    RECORD = []
    _orig_solve = CentroidalNMPC.solve

    def _probe_solve(self, r_com, v_com, L_com, r_com_ref, v_com_ref,
                     contact_config, warm_start=True, hw_current=None,
                     L_com_ref=None):
        hw = (np.asarray(hw_current, dtype=float).reshape(3).copy()
              if hw_current is not None else np.zeros(3))
        h_max = np.asarray(self.config.h_max_tight, dtype=float).reshape(3)
        out = _orig_solve(self, r_com, v_com, L_com, r_com_ref, v_com_ref,
                          contact_config, warm_start, hw_current, L_com_ref)
        info = out[4]
        RECORD.append({
            'hw_current': hw.tolist(),
            'violates_box': bool(np.any(np.abs(hw) > h_max + 1e-12)),
            'worst_axis': int(np.argmax(np.abs(hw))),
            'excess': float(np.max(np.abs(hw) - h_max)),
            'success': bool(info.success), 'status': str(info.status),
            'box_enforced': bool(self.config.enforce_hw_conservation),
            'h_max': h_max.tolist(),
        })
        return out
    CentroidalNMPC.solve = _probe_solve

    import scripts.diag_cooperative_arms as dca
    from scripts.diag_cooperative_arms import _mutate_mjcf, _mjcf_md5, MJCF

    C_KWARGS = dict(
        legacy=False, alpha_torso_lin=0.0, anchor_dx=0.8, mass_ratio=0.01,
        aocs_mode='legacy_pid_numerical', settle_seconds=20.0,
        K_theta=1.0, K_omega=50.0, tau_w_max=2.5,
        n_steps=6, ss_two_task=True, ss_alpha_mom=400.0,
        alpha_torso_pose=2000.0, ss_alpha_ee=1000.0, ss_alpha_posture=2e1,
        ss_alpha_wrench=1.0, ss_kp_torso=3.0, ss_kd_torso=2.5,
        qp_envelope_exact=True,
        interstep_settle_alpha_wrench=3.0, interstep_settle_epsilon_v=5e-3,
        out_dir_override='f2_hw_probe')

    with open(MJCF) as f:
        orig_mjcf = f.read()
    pre = _mjcf_md5(MJCF)
    try:
        _mutate_mjcf(damping=0.0, armature=0.05, anchor_dx=0.8, mass_ratio=0.01)
        try:
            dca.main(**C_KWARGS)
        except Exception as e:
            print(f'[probe] main() raised: {type(e).__name__}: {str(e)[:200]}')
    finally:
        with open(MJCF, 'w') as f:
            f.write(orig_mjcf)
        assert _mjcf_md5(MJCF) == pre, 'MJCF restore failed'

    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    with open(DEST, 'w') as fh:
        json.dump(RECORD, fh)
    print(f'[probe] inner: {len(RECORD)} solves recorded', flush=True)


if __name__ == '__main__':
    (inner if os.environ.get(INNER) else outer)()
