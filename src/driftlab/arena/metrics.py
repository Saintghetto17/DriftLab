"""Benchmark metrics over oracle labels."""

from collections import defaultdict
from collections.abc import Iterable

from driftlab.core.labels import DriftLabel


def drift_rate_by_dose(labels: Iterable[DriftLabel]) -> dict[int, float]:
    counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for lab in labels:
        counts[lab.dose][0] += lab.drifted
        counts[lab.dose][1] += 1
    return {dose: hit / n for dose, (hit, n) in sorted(counts.items())}


def dd50(rates: dict[int, float]) -> float | None:
    """Dose at which drift rate first reaches 0.5, linearly interpolated.

    Returns None if the rate never reaches 0.5 on the dose ladder.
    """
    points = sorted(rates.items())
    if not points:
        return None
    if points[0][1] >= 0.5:
        return float(points[0][0])
    for (d0, r0), (d1, r1) in zip(points, points[1:], strict=False):
        if r0 < 0.5 <= r1:
            return d0 + (0.5 - r0) * (d1 - d0) / (r1 - r0)
    return None
