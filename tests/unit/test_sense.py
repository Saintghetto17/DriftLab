from driftlab.core import GoalSpec, Step
from driftlab.sense import Detector
from driftlab.sense.detectors import IrreversibleActionRule


def test_rule_detector():
    det = IrreversibleActionRule(GoalSpec(task_id="t", task="x", irreversible=["grant_access"]))
    assert isinstance(det, Detector)
    assert det.observe(Step(index=0, role="agent", tool_name="grant_access")).risk == 1.0
    assert det.observe(Step(index=1, role="agent", tool_name="ls")).risk == 0.0
