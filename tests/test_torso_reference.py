"""TorsoReferenceShaper — the raw planner reference and the diagnostic overrides,
on synthetic inputs."""
from types import SimpleNamespace

import numpy as np

from crawlbot.control.controller import DiagHooks
from crawlbot.control.torso_reference import TorsoReferenceShaper
from crawlbot.simulation.config import SimConfig


def _tr():
    return SimpleNamespace(p=np.array([0.1, 0.2, 0.3]), R=np.eye(3),
                           v=np.arange(6, dtype=float),
                           a=np.arange(6, 12, dtype=float))


def test_raw_planner_reference():
    sh = TorsoReferenceShaper(SimConfig(), DiagHooks())
    tr = _tr()
    ref = sh.shape(tr)
    assert ref.p is tr.p and ref.v is tr.v and ref.a is tr.a and ref.R is tr.R


def test_freeze_ref_holds_the_first_sample_and_pure_pd_strips_ff():
    sh = TorsoReferenceShaper(SimConfig(),
                              DiagHooks(freeze_ref=True, pure_pd=True))
    tr0 = _tr()
    ref0 = sh.shape(tr0)
    tr1 = _tr()
    tr1.p = tr1.p + 1.0
    tr1.R = -np.eye(3)
    ref1 = sh.shape(tr1)
    assert np.array_equal(ref1.p, tr0.p) and np.array_equal(ref1.R, np.eye(3))
    assert np.array_equal(ref0.v, np.zeros(6))
    assert np.array_equal(ref1.a, np.zeros(6))
