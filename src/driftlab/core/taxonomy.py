"""Drift taxonomy: source x tempo x what drifts, plus legitimacy."""

from enum import StrEnum

from pydantic import BaseModel


class Source(StrEnum):
    """Where the cause of drift comes from."""

    OUTSIDE = "outside"  # environment, users, injected content, other agents
    INSIDE = "inside"  # the model's own goals, values, persona
    DEPLOY = "deploy"  # model/data/concept drift, inference stack


class Tempo(StrEnum):
    ABRUPT = "abrupt"
    GRADUAL = "gradual"


class Target(StrEnum):
    """What drifts away from the specification."""

    GOAL = "goal"
    CONSTRAINT = "constraint"
    EXECUTION = "execution"


class Legitimacy(StrEnum):
    """Whether the deviation was sanctioned (e.g. a legitimate goal change)."""

    LEGITIMATE = "legitimate"
    ILLEGITIMATE = "illegitimate"
    UNKNOWN = "unknown"


class DriftType(BaseModel, frozen=True):
    source: Source
    tempo: Tempo
    target: Target
    legitimacy: Legitimacy = Legitimacy.UNKNOWN

    @property
    def cell(self) -> str:
        """Taxonomy cell id, e.g. ``outside/gradual/constraint``."""
        return f"{self.source}/{self.tempo}/{self.target}"
