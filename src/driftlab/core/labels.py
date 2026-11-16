"""Drift labels produced by the oracle: d (onset) and h (first harm).

A DriftLabel is the ground truth for one treatment run. Everything downstream
is trained or evaluated against it: benchmark metrics (Paper 1), detector
recall and timing (Paper 2), router decisions (Paper 3), probes (Paper 0),
rewards (Paper 5).
"""

from enum import StrEnum

from pydantic import BaseModel

from driftlab.core.taxonomy import DriftType


class Trigger(StrEnum):
    """Which oracle signal fired first and set d.

    - GOAL_GAP: treatment progress fell behind control (goal or execution drift).
    - INVARIANT: a constraint was violated (constraint drift), even if progress
      looked fine.

    Why it matters: it tells WHY the run counts as drifted, and it fills the
    Target of the drift type.
    """

    GOAL_GAP = "goal_gap"
    INVARIANT = "invariant"


class DriftLabel(BaseModel):
    """Ground truth for one treatment run versus its paired control.

    - d: first step of drift, min(d_G, d_C);
    - h: first irreversible call made while a constraint is violated;
    - tau, k: oracle parameters, stored for reproducibility.

    >>> lab = DriftLabel(trajectory_id="run-0042", control_id="run-0041", dose=3, d=1, h=4)
    >>> lab.drifted, lab.harm_margin
    (True, 3)
    """

    trajectory_id: str
    control_id: str
    dose: int
    d: int | None = None
    h: int | None = None
    trigger: Trigger | None = None
    drift_type: DriftType | None = None
    tau: float | None = None
    k: int | None = None

    @property
    def drifted(self) -> bool:
        """Whether the run drifted at all; input to drift rate and DD50."""
        return self.d is not None

    @property
    def harm_margin(self) -> int | None:
        """h - d: how many steps a monitor has to act before harm.

        Why it matters: it is the time budget of any defense. Margin 0 means
        drift and harm happen in the same step, so only a pre-call block helps;
        a large margin means a cheap reminder or rollback can still save the
        run. Paper 3's router chooses the intervention based on this.
        """
        if self.d is None or self.h is None:
            return None
        return self.h - self.d
