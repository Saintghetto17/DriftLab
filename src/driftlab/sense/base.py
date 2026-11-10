"""Detector protocol: rules, probes and decision models all return a Verdict."""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from driftlab.core.taxonomy import DriftType
from driftlab.core.traces import Step


class Verdict(BaseModel):
    risk: float = Field(ge=0.0, le=1.0)
    drift_type: DriftType | None = None
    explanation: str = ""


@runtime_checkable
class Detector(Protocol):
    name: str

    def reset(self) -> None:
        """Clear per-trajectory state."""
        ...

    def observe(self, step: Step) -> Verdict: ...
