"""Detector protocol: rules, probes and decision models all return a Verdict.

Why one protocol: Paper 2 compares very different detectors (a regex-like
rule, a white-box probe on activations, a fine-tuned 0.6-8B decision model, a
large LLM judge). If they all expose ``observe(step) -> Verdict``, the same
evaluation code scores all of them against the oracle's d and h, and Paper 3's
router can swap them freely.
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from driftlab.core.taxonomy import DriftType
from driftlab.core.traces import Step


class Verdict(BaseModel):
    """A detector's opinion about the current step.

    - risk: 0..1, thresholded by calibration to a false-alarm budget;
    - drift_type: the detector's guess of the taxonomy cell (optional);
    - explanation: shown to a human or used by the intervention.

    The first step where risk crosses the threshold is the detector's alarm;
    compared with the oracle's d and h it gives detection delay and whether
    the alarm came before harm.

    >>> Verdict(risk=0.9, explanation="grant without approval").risk
    0.9
    """

    risk: float = Field(ge=0.0, le=1.0)
    drift_type: DriftType | None = None
    explanation: str = ""


@runtime_checkable
class Detector(Protocol):
    """Anything that watches a trajectory step by step.

    Any class with ``name``, ``reset()`` and ``observe(step)`` is a detector;
    no inheritance needed. ``reset()`` is called between trajectories (for
    stateful detectors such as CUSUM).
    """

    name: str

    def reset(self) -> None:
        """Clear per-trajectory state."""
        ...

    def observe(self, step: Step) -> Verdict: ...
