"""
TorsoReferenceShaper — the torso reference the whole-body QP tracks.

It GENERATES a reference; it does not command. In the ROS 2 layout it belongs on
the reference side, next to the planners — hence a module of its own, out of
``WholeBodyController.track`` (refactor/sim-loop-split, C2).

The reference is the planner's (TorsoPlanner quintic + SLERP, ``tr``), in SS and
DS alike, followed by the ``freeze_ref`` and ``pure_pd`` diagnostic overrides.
The CoM→torso δ-mapping that used to replace the linear part — in the
non-two-task SS path and in DS — was retired (R2a, R2b): the paper's SS stack is
two-task (raw reference) and in DS the QP drops the linear torso task under
``settle_mode``, so the mapped value was never used (NaN probe, C2).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TorsoRef:
    """6-D torso reference for the QP: position ``p`` (3), rotation ``R``
    (3×3), twist ``v`` (6: linear, angular), feedforward acceleration ``a``
    (6). Structure frame."""
    p: np.ndarray
    R: np.ndarray
    v: np.ndarray
    a: np.ndarray


class TorsoReferenceShaper:
    """The planner's torso reference, with the diagnostic overrides."""

    def __init__(self, cfg, diag):
        self._cfg = cfg
        self._diag = diag
        # _diag_freeze_ref: first-sample r_b_ref / R_b_ref held by the hook.
        self._diag_frozen_r_b_ref: Optional[np.ndarray] = None
        self._diag_frozen_R_b_ref: Optional[np.ndarray] = None

    def shape(self, tr):
        """The torso reference for one QP sub-step, from the planner's ``tr``."""
        p_torso_ref_used = tr.p
        v_torso_ref_used = tr.v
        a_torso_ff_used = tr.a

        if self._diag.freeze_ref:
            # Freeze r_b_ref / R_b_ref to the first sample taken.
            # Used to probe PD stability at a STATIC target.
            if self._diag_frozen_r_b_ref is None:
                self._diag_frozen_r_b_ref = p_torso_ref_used.copy()
                self._diag_frozen_R_b_ref = tr.R.copy()
            p_torso_ref_used = self._diag_frozen_r_b_ref.copy()
            v_torso_ref_used = np.zeros(6)
            a_torso_ff_used = np.zeros(6)

        if self._diag.pure_pd:
            # Strip the torso feedforward entering the QP (the controller
            # strips λ_ref and a_com_ff).
            a_torso_ff_used = np.zeros(6)
        R_torso_ref_used = (self._diag_frozen_R_b_ref
                            if self._diag.freeze_ref else tr.R)
        return TorsoRef(p=p_torso_ref_used, R=R_torso_ref_used,
                        v=v_torso_ref_used, a=a_torso_ff_used)
