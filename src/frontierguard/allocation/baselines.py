"""Cost-matched selector baselines."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable


def select_by_score(
    scores: dict[str, float],
    cost: Callable[[str], float],
    budget: float,
) -> list[str]:
    selected: list[str] = []
    used = 0.0
    order = sorted(scores, key=lambda name: (-scores[name] / max(cost(name), 1e-12), name))
    for name in order:
        item_cost = cost(name)
        if used + item_cost <= budget:
            selected.append(name)
            used += item_cost
    return selected


def select_random(
    candidates: Iterable[str],
    cost: Callable[[str], float],
    budget: float,
    *,
    seed: int,
) -> list[str]:
    values = sorted(set(candidates))
    random.Random(seed).shuffle(values)
    selected: list[str] = []
    used = 0.0
    for name in values:
        item_cost = cost(name)
        if used + item_cost <= budget:
            selected.append(name)
            used += item_cost
    return selected


def structural_scores(candidates: Iterable[str]) -> dict[str, float]:
    priorities = {
        "down_proj": 3.0,
        "o_proj": 2.5,
        "v_proj": 2.0,
        "q_proj": 1.5,
        "k_proj": 1.25,
        "up_proj": 1.0,
        "gate_proj": 1.0,
    }
    return {
        name: next((score for suffix, score in priorities.items() if name.endswith(suffix)), 0.0)
        for name in candidates
    }
