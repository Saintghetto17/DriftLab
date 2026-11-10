"""Unified trajectory format shared by arena runs and guard adapters."""

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RunKind(StrEnum):
    CONTROL = "control"
    TREATMENT = "treatment"


class Step(BaseModel):
    index: int
    role: str  # "agent" | "tool" | "user" | "env"
    text: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    observation: str | None = None
    goal_score: float | None = Field(default=None, description="G_t after this step.")
    violations: list[str] = Field(
        default_factory=list,
        description="Constraints violated by this step's tool call (GoalSpec.violations).",
    )


class Trajectory(BaseModel):
    id: str
    task_id: str
    model: str
    kind: RunKind
    dose: int = 0
    seed: int = 0
    pair_id: str | None = Field(default=None, description="Id of the paired control run.")
    steps: list[Step] = Field(default_factory=list)

    def goal_scores(self) -> list[float]:
        return [s.goal_score for s in self.steps if s.goal_score is not None]


def write_jsonl(trajectories: list[Trajectory], path: str | Path) -> None:
    with Path(path).open("w") as f:
        for t in trajectories:
            f.write(t.model_dump_json() + "\n")


def read_jsonl(path: str | Path) -> list[Trajectory]:
    with Path(path).open() as f:
        return [Trajectory.model_validate_json(line) for line in f if line.strip()]
