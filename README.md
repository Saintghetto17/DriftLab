# DriftLab

A framework for measuring, detecting and mitigating drift in LLM agents.

Drift is measured by actions and environment state against a paired control run:
`d` is the step where the agent's progress toward the goal diverges from control,
`h` is its first irreversible action, and `h − d` is the window a monitor has to act.

> Early stage: APIs and names will change.

## Structure

| Subpackage | Purpose | Status |
|---|---|---|
| `driftlab.core` | GoalSpec, trajectory format, drift taxonomy, labels | started |
| `driftlab.arena` | Harbor/Docker tasks with dose ladder, oracle (d, h), metrics (DD50, survival) | started |
| `driftlab.sense` | Detectors under one API: rules, omission, embeddings, probes, decision models | started |
| `driftlab.guard` | Runtime monitor router and interventions | planned |
| `driftlab.serve` | Drift checks for inference-stack configurations | planned |
| `driftlab.train` | RL rewards against drift for models ≤9B | planned |

`experiments/` holds paper-specific research code.

## Development

```bash
uv sync
uv run pytest
uv run ruff check . && uv run pyright
```

Integration tests that need Docker are marked `docker` and skipped by default:
`uv run pytest -m docker`.

## License

Apache 2.0
