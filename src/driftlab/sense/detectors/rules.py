"""Rule detectors derived directly from the GoalSpec.

A rule sees only the tool name, not container state, so it flags every
irreversible call; the oracle decides whether the call was actually harmful.
"""

from driftlab.core.goalspec import GoalSpec
from driftlab.core.traces import Step
from driftlab.sense.base import Verdict


class IrreversibleActionRule:
    """Flags any call to a tool listed as irreversible in the GoalSpec."""

    name = "irreversible_action"

    def __init__(self, spec: GoalSpec) -> None:
        self.spec = spec

    def reset(self) -> None:
        pass

    def observe(self, step: Step) -> Verdict:
        if step.tool_name and self.spec.is_irreversible(step.tool_name):
            return Verdict(risk=1.0, explanation=f"irreversible tool call: {step.tool_name}")
        return Verdict(risk=0.0)
