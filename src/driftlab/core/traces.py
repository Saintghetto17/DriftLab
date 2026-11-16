"""Unified trajectory format shared by arena runs and guard adapters.

Why one format: trajectories come from many places (DeepSeek over API, Qwen3
on our GPUs, Harbor tasks, later Claude Code / LangGraph adapters in
production). If they all use Step/Trajectory, every detector, metric and
training script works on all of them without conversion code.
"""

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RunKind(StrEnum):
    """Control (no pressure, dose 0) or treatment (pressure at dose 1-4).

    Why it matters: drift is defined RELATIVE to a control run of the same
    task, model and environment seed. Without the control we cannot tell
    "the agent drifted" from "the task is just hard".
    """

    CONTROL = "control"
    TREATMENT = "treatment"


class Step(BaseModel):
    """One agent step: what it said, which tool it called, what the environment
    returned, plus the oracle's per-step annotations.

    ``goal_score`` (G_t after the step) and ``violations`` (constraints broken
    by this call) are filled by the runner from the GoalSpec checks.

    Where it is used: detectors look at steps one by one (``Detector.observe``),
    exactly as a runtime monitor would in production.

    >>> s = Step(index=1, role="agent", tool_name="grant_access",
    ...          tool_args={"user": "bob", "path": "/finance"},
    ...          goal_score=0.67, violations=["approval_before_grant"])
    >>> s.violations
    ['approval_before_grant']
    """

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
    """A full run with its experimental coordinates: model, dose, seed, pair.

    ``pair_id`` links a treatment run to its control run; the oracle needs both.

    >>> t = Trajectory(id="run-0042", task_id="fin-access-01", model="deepseek-v3.2",
    ...                kind="treatment", dose=3, pair_id="run-0041",
    ...                steps=[Step(index=0, role="agent", goal_score=0.0),
    ...                       Step(index=1, role="agent", goal_score=0.67)])
    >>> t.goal_scores()
    [0.0, 0.67]
    """

    id: str
    task_id: str
    model: str
    kind: RunKind
    dose: int = 0
    seed: int = 0
    pair_id: str | None = Field(default=None, description="Id of the paired control run.")
    steps: list[Step] = Field(default_factory=list)

    def goal_scores(self) -> list[float]:
        """The G_t curve of the run, input to the oracle's d_G signal.

        Plotting control and treatment curves side by side is also the
        simplest way to see drift by eye.
        """
        return [s.goal_score for s in self.steps if s.goal_score is not None]


def write_jsonl(trajectories: list[Trajectory], path: str | Path) -> None:
    """Save runs, one JSON line per trajectory.

    Where it is used: the runner appends every finished run to ``runs/*.jsonl``.
    Why it matters: API runs cost money; saved trajectories can be re-labeled
    (new tau, k) and re-scored by new detectors without re-running agents.
    """
    with Path(path).open("w") as f:
        for t in trajectories:
            f.write(t.model_dump_json() + "\n")


def read_jsonl(path: str | Path) -> list[Trajectory]:
    """Load runs saved by ``write_jsonl``, validating every field.

    Where it is used: labeling, detector evaluation (Paper 2), building
    training data for decision models and probes (Paper 0).
    """
    with Path(path).open() as f:
        return [Trajectory.model_validate_json(line) for line in f if line.strip()]
