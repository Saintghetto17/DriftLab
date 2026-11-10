"""Drift labels produced by the oracle: d (onset) and h (first harm)."""

from enum import StrEnum

from pydantic import BaseModel

from driftlab.core.taxonomy import DriftType


class Trigger(StrEnum):
    """Which oracle signal fired first and set d."""

    GOAL_GAP = "goal_gap"  # G_t fell behind control: goal or execution drift
    INVARIANT = "invariant"  # a constraint was violated: constraint drift


class DriftLabel(BaseModel):
    trajectory_id: str
    control_id: str
    dose: int
    d: int | None = None  # min(d_G, d_C)
    h: int | None = None  # first irreversible call with a violated constraint
    trigger: Trigger | None = None
    drift_type: DriftType | None = None
    tau: float | None = None
    k: int | None = None

    @property
    def drifted(self) -> bool:
        return self.d is not None

    @property
    def harm_margin(self) -> int | None:
        """h - d: how many steps a monitor has to act before harm."""
        if self.d is None or self.h is None:
            return None
        return self.h - self.d
