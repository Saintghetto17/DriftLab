"""Goal specification: the task's goal and constraints as checks over environment state.

One GoalSpec feeds the oracle (arena), rule detectors (sense), the list of
irreversible actions (guard) and, later, rewards (train).

Semantics:
- predicates describe the goal; their weighted share is progress G_t;
- invariants describe constraints: before every call to a matching tool the
  ``require`` check must pass, otherwise the step violates a constraint;
- irreversible tools are those whose effects cannot be undone. An irreversible
  tool with no invariant covering it is not allowed (allowlist by default).
"""

from fnmatch import fnmatch
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

NO_PERMISSION = "irreversible_without_permission"


class Predicate(BaseModel, frozen=True):
    """A goal check over container state; ``check`` is executed by the environment."""

    name: str
    check: str
    weight: float = 1.0


class Invariant(BaseModel, frozen=True):
    """A constraint: before calling a tool matching ``before``, ``require`` must hold."""

    name: str
    before: str = Field(description="Glob pattern over tool names.")
    require: str = Field(description="Check executed in the container before the call.")

    def applies_to(self, tool_name: str) -> bool:
        return fnmatch(tool_name, self.before)


class GoalSpec(BaseModel):
    task_id: str
    task: str
    predicates: list[Predicate] = Field(default_factory=list)
    invariants: list[Invariant] = Field(default_factory=list)
    irreversible: list[str] = Field(
        default_factory=list, description="Glob patterns over tool names."
    )

    def is_irreversible(self, tool_name: str) -> bool:
        return any(fnmatch(tool_name, pattern) for pattern in self.irreversible)

    def invariants_for(self, tool_name: str) -> list[Invariant]:
        return [inv for inv in self.invariants if inv.applies_to(tool_name)]

    def score(self, results: dict[str, bool]) -> float:
        """Weighted share of satisfied predicates: G_t in [0, 1]."""
        total = sum(p.weight for p in self.predicates)
        if total == 0:
            return 0.0
        return sum(p.weight for p in self.predicates if results.get(p.name, False)) / total

    def violations(self, tool_name: str, require_results: dict[str, bool]) -> list[str]:
        """Names of constraints violated by calling ``tool_name``.

        ``require_results`` maps invariant name to the outcome of its ``require``
        check, run in the container *before* the call. A missing result counts
        as failed. An irreversible tool with no applicable invariant is reported
        as ``NO_PERMISSION``.
        """
        applicable = self.invariants_for(tool_name)
        if not applicable:
            return [NO_PERMISSION] if self.is_irreversible(tool_name) else []
        return [inv.name for inv in applicable if not require_results.get(inv.name, False)]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GoalSpec":
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))
