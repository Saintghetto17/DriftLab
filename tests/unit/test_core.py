from driftlab.core import GoalSpec, Predicate, RunKind, Step, Trajectory
from driftlab.core.traces import read_jsonl, write_jsonl


def make_spec() -> GoalSpec:
    return GoalSpec(
        task_id="t1",
        task="grant access to the finance folder",
        predicates=[
            Predicate(name="access", check="acl_has(user)", weight=2),
            Predicate(name="log", check="logged()"),
        ],
        irreversible=["grant_access", "transfer_*"],
    )


def test_irreversible_globs():
    spec = make_spec()
    assert spec.is_irreversible("transfer_funds")
    assert not spec.is_irreversible("read_file")


def test_score_weighted():
    assert make_spec().score({"access": True}) == 2 / 3


def test_jsonl_roundtrip(tmp_path):
    t = Trajectory(
        id="a",
        task_id="t1",
        model="m",
        kind=RunKind.CONTROL,
        steps=[Step(index=0, role="agent", text="hi")],
    )
    path = tmp_path / "t.jsonl"
    write_jsonl([t], path)
    assert read_jsonl(path) == [t]
