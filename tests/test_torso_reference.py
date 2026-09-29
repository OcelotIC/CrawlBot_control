"""TorsoReferenceShaper — path selection and events on synthetic inputs.

Covers the paths that need no CoMToTorsoMapping (mapping=None): the raw
planner reference, the SS mapping bypass, the SS-entry / dock events, and the
freeze_ref / pure_pd overrides. The δ-mapping path itself is exercised by the
canonical replay (DS) and the legacy_stack scenario (SS, with F-SAT).
"""
from types import SimpleNamespace

import numpy as np

from crawlbot.control.controller import DiagHooks
from crawlbot.control.torso_reference import TorsoReferenceShaper
from crawlbot.simulation.config import SimConfig


def _tr():
    return SimpleNamespace(p=np.array([0.1, 0.2, 0.3]), R=np.eye(3),
                           v=np.arange(6, dtype=float),
                           a=np.arange(6, 12, dtype=float))


def _shape(sh, tr, phase='SS'):
    z = np.zeros(3)
    return sh.shape(tr, None, 0, 0.0, phase, z, z, z)


def test_raw_planner_reference_without_mapping():
    sh = TorsoReferenceShaper(SimConfig(), None, DiagHooks())
    tr = _tr()
    ref = _shape(sh, tr)
    assert ref.p is tr.p and ref.v is tr.v and ref.a is tr.a and ref.R is tr.R


def test_mapping_bypass_holds_the_ss_entry_position():
    cfg = SimConfig()
    cfg.mapping_bypass_in_ss = True
    sh = TorsoReferenceShaper(cfg, None, DiagHooks())
    tr = _tr()
    assert _shape(sh, tr).p is tr.p            # no SS entry yet -> raw
    entry = np.array([1.0, 2.0, 3.0])
    sh.on_ss_entry(entry)
    ref = _shape(sh, tr)
    assert np.array_equal(ref.p, entry) and ref.p is not entry
    assert np.array_equal(ref.v[:3], np.zeros(3))
    assert np.array_equal(ref.v[3:], tr.v[3:])


def test_dock_arms_the_ds_blend_from_the_ss_entry():
    sh = TorsoReferenceShaper(SimConfig(), None, DiagHooks())
    sh.on_dock(5.0)
    assert sh._ds_ramp_t_start == 5.0 and sh._ds_ramp_p_start is None
    entry = np.array([1.0, 2.0, 3.0])
    sh.on_ss_entry(entry)
    assert sh._ds_ramp_t_start is None          # SS entry resets the blend
    sh.on_dock(7.0)
    assert sh._ds_ramp_t_start == 7.0
    assert np.array_equal(sh._ds_ramp_p_start, entry)


def test_freeze_ref_holds_the_first_sample_and_pure_pd_strips_ff():
    sh = TorsoReferenceShaper(SimConfig(), None,
                              DiagHooks(freeze_ref=True, pure_pd=True))
    tr0 = _tr()
    ref0 = _shape(sh, tr0)
    tr1 = _tr()
    tr1.p = tr1.p + 1.0
    tr1.R = -np.eye(3)
    ref1 = _shape(sh, tr1)
    assert np.array_equal(ref1.p, tr0.p) and np.array_equal(ref1.R, np.eye(3))
    assert np.array_equal(ref0.v, np.zeros(6))
    assert np.array_equal(ref1.a, np.zeros(6))
