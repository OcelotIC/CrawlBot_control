"""Locomotion-sequence file loader.

Parses a human-readable text file describing the dock sequence for a
traversal and builds a ``GaitPlan`` that the rest of the stack consumes
through the existing ``ContactScheduler`` interface.

File format
-----------
- Lines starting with ``#`` and blank lines are ignored.
- Exactly one ``START <name_a> <name_b>`` line must appear before the
  first ``DOCK`` line.
- Each ``DOCK <name>`` line names a single MJCF anchor site
  (e.g. ``anchor_4b``). The arm that swings is determined by the
  name suffix (``a`` or ``b``) — the current MJCF declares 12
  explicit welds (``grip_a_to_N<a>`` / ``grip_b_to_N<b>``), so only
  the suffix-matching arm can dock at a given site.

Anchor naming
-------------
Site names match the MJCF: ``anchor_<N><a|b>`` for ``N=1..6``.
Internally we map to 0-indexed array positions (``anchor_1a``
→ ``anchors_a[0]``).

Example
-------
::

    # canonical 5-step forward traversal
    START anchor_3a anchor_3b
    DOCK  anchor_4b
    DOCK  anchor_4a
    DOCK  anchor_5b
    DOCK  anchor_5a
    DOCK  anchor_6b

Policy note (closest-shoulder rule)
-----------------------------------
For any future MJCF with ambidextrous anchors (no suffix preference),
the swing arm should be chosen as the one whose **shoulder** is
geometrically closest to the target anchor — this minimizes the
extension ratio ``‖tool − shoulder‖ / R_max`` and protects σ_min.
Not implemented in v1 since the current MJCF forces the choice by
suffix.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from .contact_scheduler import ContactPhase, ContactScheduler, GaitPhase, GaitPlan


_ANCHOR_RE = re.compile(r'^anchor_(\d+)([ab])$')


@dataclass
class SwingTarget:
    arm: str       # 'a' or 'b'
    anchor_idx: int  # 0-indexed into ContactScheduler.anchors_{a,b}
    dwell_after: float = 0.0  # seconds; >0 ⇒ run a settle DS phase after this dock


@dataclass
class LoadedSequence:
    start_a: int
    start_b: int
    swing_targets: List[SwingTarget]
    source_path: str


def _parse_anchor_name(name: str) -> Tuple[str, int]:
    """Return (arm, 0-indexed_idx) from a name like 'anchor_3b'.

    Raises ValueError on a malformed name.
    """
    m = _ANCHOR_RE.match(name.strip())
    if m is None:
        raise ValueError(
            f"malformed anchor name {name!r} — expected 'anchor_<N><a|b>'")
    n = int(m.group(1))
    arm = m.group(2)
    if n < 1:
        raise ValueError(f"anchor index must be >= 1, got {n} in {name!r}")
    return arm, n - 1


def load_sequence(path: str, n_anchors: int) -> LoadedSequence:
    """Parse a ``.seq`` file into a ``LoadedSequence``.

    Parameters
    ----------
    path : str
        File path to read.
    n_anchors : int
        Number of anchors per arm in the live MJCF (used to validate
        that all referenced indices are in range).

    Returns
    -------
    seq : LoadedSequence
    """
    start_a = start_b = None
    targets: List[SwingTarget] = []

    with open(path) as fh:
        for ln_no, raw in enumerate(fh, start=1):
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue
            tokens = line.split()
            head = tokens[0].upper()

            if head == 'START':
                if start_a is not None:
                    raise ValueError(
                        f"{path}:{ln_no}: duplicate START line")
                if len(tokens) != 3:
                    raise ValueError(
                        f"{path}:{ln_no}: START needs two anchor names, "
                        f"got {len(tokens) - 1}")
                arm0, idx0 = _parse_anchor_name(tokens[1])
                arm1, idx1 = _parse_anchor_name(tokens[2])
                if {arm0, arm1} != {'a', 'b'}:
                    raise ValueError(
                        f"{path}:{ln_no}: START must name one 'a' anchor "
                        f"and one 'b' anchor, got {tokens[1]} {tokens[2]}")
                if arm0 == 'a':
                    start_a, start_b = idx0, idx1
                else:
                    start_a, start_b = idx1, idx0

            elif head == 'DOCK':
                if start_a is None:
                    raise ValueError(
                        f"{path}:{ln_no}: DOCK before START")
                if len(tokens) != 2:
                    raise ValueError(
                        f"{path}:{ln_no}: DOCK needs exactly one anchor name")
                arm, idx = _parse_anchor_name(tokens[1])
                targets.append(SwingTarget(arm=arm, anchor_idx=idx))

            elif head == 'DWELL':
                # Attach a dwell duration (seconds) to the most recent
                # DOCK so the sim runs a settle DS phase after it
                # (modelling external AOCS desat between traversals).
                if start_a is None:
                    raise ValueError(
                        f"{path}:{ln_no}: DWELL before START")
                if not targets:
                    raise ValueError(
                        f"{path}:{ln_no}: DWELL before any DOCK")
                if len(tokens) != 2:
                    raise ValueError(
                        f"{path}:{ln_no}: DWELL needs exactly one duration "
                        f"(seconds)")
                try:
                    dwell_s = float(tokens[1])
                except ValueError:
                    raise ValueError(
                        f"{path}:{ln_no}: DWELL duration must be a number")
                if dwell_s <= 0:
                    raise ValueError(
                        f"{path}:{ln_no}: DWELL duration must be > 0")
                targets[-1].dwell_after = dwell_s

            else:
                raise ValueError(
                    f"{path}:{ln_no}: unknown keyword {tokens[0]!r}; "
                    f"expected START, DOCK or DWELL")

    if start_a is None:
        raise ValueError(f"{path}: missing START line")
    if not targets:
        raise ValueError(f"{path}: no DOCK lines (sequence is empty)")

    # Range check vs the live MJCF.
    if not (0 <= start_a < n_anchors and 0 <= start_b < n_anchors):
        raise ValueError(
            f"{path}: START indices ({start_a}, {start_b}) out of range "
            f"for MJCF n_anchors={n_anchors}")
    for k, st in enumerate(targets):
        if not (0 <= st.anchor_idx < n_anchors):
            raise ValueError(
                f"{path}: DOCK #{k + 1} target index {st.anchor_idx} out "
                f"of range for MJCF n_anchors={n_anchors}")

    return LoadedSequence(
        start_a=start_a, start_b=start_b,
        swing_targets=targets, source_path=path)


def plan_from_sequence(
    sched: ContactScheduler, seq: LoadedSequence
) -> GaitPlan:
    """Build a ``GaitPlan`` from a parsed sequence.

    The ``ContactScheduler`` instance is used only for ``dt_ds`` and
    ``dt_ss`` defaults; its ``plan_traversal`` is not called.

    Validation
    ----------
    - The swing arm at each step must not already be docked at the
      named target (no-op rejection).
    - Sequence must start in the configuration declared by ``START``.
    """
    phases: List[GaitPhase] = []
    ia, ib = seq.start_a, seq.start_b

    phases.append(GaitPhase(
        phase=ContactPhase.DOUBLE, duration=sched.dt_ds,
        anchor_a_idx=ia, anchor_b_idx=ib))

    for k, st in enumerate(seq.swing_targets):
        if st.arm == 'b':
            if st.anchor_idx == ib:
                raise ValueError(
                    f"{seq.source_path}: DOCK #{k + 1} arm B already at "
                    f"anchor_{ib + 1}b (no-op)")
            phases.append(GaitPhase(
                phase=ContactPhase.SINGLE_A, duration=sched.dt_ss,
                anchor_a_idx=ia, anchor_b_idx=ib,
                swing_arm='b', swing_from_idx=ib, swing_to_idx=st.anchor_idx))
            ib = st.anchor_idx
            ds_dur = sched.dt_ds + max(0.0, float(st.dwell_after))
            phases.append(GaitPhase(
                phase=ContactPhase.DOUBLE, duration=ds_dur,
                anchor_a_idx=ia, anchor_b_idx=ib))
        else:  # 'a'
            if st.anchor_idx == ia:
                raise ValueError(
                    f"{seq.source_path}: DOCK #{k + 1} arm A already at "
                    f"anchor_{ia + 1}a (no-op)")
            phases.append(GaitPhase(
                phase=ContactPhase.SINGLE_B, duration=sched.dt_ss,
                anchor_a_idx=ia, anchor_b_idx=ib,
                swing_arm='a', swing_from_idx=ia, swing_to_idx=st.anchor_idx))
            ia = st.anchor_idx
            ds_dur = sched.dt_ds + max(0.0, float(st.dwell_after))
            phases.append(GaitPhase(
                phase=ContactPhase.DOUBLE, duration=ds_dur,
                anchor_a_idx=ia, anchor_b_idx=ib))

    t_start, t_end = [], []
    t = 0.0
    for gp in phases:
        t_start.append(t)
        t += gp.duration
        t_end.append(t)

    plan = GaitPlan(
        phases=phases, t_start=t_start, t_end=t_end, total_duration=t)
    sched._plan = plan
    return plan
