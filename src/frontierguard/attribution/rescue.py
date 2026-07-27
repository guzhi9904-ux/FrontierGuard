"""Rescue metrics and robust aggregation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RescueObservation:
    problem_id: str
    trace_id: str
    step_index: int
    module_name: str
    local_rescue: float
    outcome_rescue: float
    frontier_confidence: float = 1.0

    def combined(self, local_weight: float = 0.3) -> float:
        return self.frontier_confidence * (
            local_weight * self.local_rescue + (1.0 - local_weight) * self.outcome_rescue
        )


@dataclass(frozen=True)
class ModuleRescueScore:
    module_name: str
    mean: float
    median: float
    standard_error: float
    positive_fraction: float
    observations: int


def aggregate_rescue(
    observations: Iterable[RescueObservation],
    *,
    local_weight: float = 0.3,
) -> list[ModuleRescueScore]:
    if not 0 <= local_weight <= 1:
        raise ValueError("local_weight must lie in [0, 1]")
    grouped: dict[str, list[float]] = defaultdict(list)
    for observation in observations:
        grouped[observation.module_name].append(observation.combined(local_weight))
    scores: list[ModuleRescueScore] = []
    for module_name, values in grouped.items():
        array = np.asarray(values, dtype=np.float64)
        stderr = float(array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else 0.0
        scores.append(
            ModuleRescueScore(
                module_name=module_name,
                mean=float(array.mean()),
                median=float(np.median(array)),
                standard_error=stderr,
                positive_fraction=float((array > 0).mean()),
                observations=int(array.size),
            )
        )
    return sorted(scores, key=lambda item: (-item.mean, item.module_name))


def pairwise_interaction(single_a: float, single_b: float, joint: float) -> float:
    return joint - single_a - single_b
