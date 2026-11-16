"""Drift taxonomy: source x tempo x what drifts, plus legitimacy.

Every drift we create, measure or detect is placed in one cell of this grid.
The benchmark covers all 18 cells (3 sources x 2 tempos x 3 targets) evenly,
so results can be reported per cell: "detector X catches outside/abrupt/
constraint drift but misses inside/gradual/goal drift".
"""

from enum import StrEnum

from pydantic import BaseModel


class Source(StrEnum):
    """Where the cause of drift comes from.

    - OUTSIDE: the environment pushes the agent (a fake "CEO" email, a prompt
      injection in a tool output, a user eroding a rule, another agent's context).
    - INSIDE: nothing external changed; the model drifts on its own (long
      horizon, value conflict, persona drift).
    - DEPLOY: the task and prompt are the same, but how the model is served
      changed (new model version, quantization, KV-cache compression).

    Why it matters: the cause decides the fix. Outside drift calls for input
    filtering, inside drift for training (Paper 5), deploy drift for a safer
    serving config (Paper 4).
    """

    OUTSIDE = "outside"
    INSIDE = "inside"
    DEPLOY = "deploy"


class Tempo(StrEnum):
    """How fast the drift unfolds.

    - ABRUPT: one event flips behavior (a single pressure email).
    - GRADUAL: behavior erodes over many steps (a series of messages, a long
      session).

    Why it matters: abrupt drift is easy to see in one step; gradual drift is
    invisible step by step and needs sequential detectors (e.g. CUSUM in Paper 3).
    """

    ABRUPT = "abrupt"
    GRADUAL = "gradual"


class Target(StrEnum):
    """What drifts away from the specification.

    - GOAL: the agent pursues a different goal (does someone else's task).
    - CONSTRAINT: the goal is the same, but a rule is broken on the way
      (grants access without approval).
    - EXECUTION: the agent still wants the goal but fails to carry it out
      (loses the plan, gives up).

    Why it matters: constraint drift can look like success (the task gets
    done!), so it is invisible to progress-based metrics and needs invariants.
    """

    GOAL = "goal"
    CONSTRAINT = "constraint"
    EXECUTION = "execution"


class Legitimacy(StrEnum):
    """Whether the deviation was sanctioned.

    Example: approval was denied, so the agent correctly does NOT grant access.
    It deviates from the goal, but legitimately. Benchmark "twin" tasks use this
    to measure false alarms of the oracle and detectors.
    """

    LEGITIMATE = "legitimate"
    ILLEGITIMATE = "illegitimate"
    UNKNOWN = "unknown"


class DriftType(BaseModel, frozen=True):
    """One cell of the taxonomy.

    Where it is used: set by benchmark task design (the intended type), filled
    by the oracle in DriftLabel (the observed type), predicted by detectors in
    Verdict. Comparing intended and observed types validates the oracle.

    Frozen (immutable), so it can be a dict key when aggregating results.

    >>> dt = DriftType(source="outside", tempo="abrupt", target="constraint")
    >>> dt.cell
    'outside/abrupt/constraint'
    """

    source: Source
    tempo: Tempo
    target: Target
    legitimacy: Legitimacy = Legitimacy.UNKNOWN

    @property
    def cell(self) -> str:
        """Cell id used as a row key in result tables and plots.

        Example use: group detector recall by ``label.drift_type.cell`` to get
        the per-cell table of Paper 2.
        """
        return f"{self.source}/{self.tempo}/{self.target}"
