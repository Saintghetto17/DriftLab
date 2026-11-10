"""Shared contracts: goal specs, trajectories, drift taxonomy and labels.

`core` has no heavy dependencies; every other subpackage builds on it.
"""

from driftlab.core.goalspec import NO_PERMISSION, GoalSpec, Invariant, Predicate
from driftlab.core.labels import DriftLabel, Trigger
from driftlab.core.taxonomy import DriftType, Legitimacy, Source, Target, Tempo
from driftlab.core.traces import RunKind, Step, Trajectory

__all__ = [
    "NO_PERMISSION",
    "DriftLabel",
    "DriftType",
    "GoalSpec",
    "Invariant",
    "Legitimacy",
    "Predicate",
    "RunKind",
    "Source",
    "Step",
    "Target",
    "Tempo",
    "Trajectory",
    "Trigger",
]
