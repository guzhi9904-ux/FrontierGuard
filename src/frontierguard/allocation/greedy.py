"""Measured greedy allocation for non-additive module effects."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass


UtilityFunction = Callable[[frozenset[str]], float]
CostFunction = Callable[[str], float]


@dataclass(frozen=True)
class GreedyStep:
    module_name: str
    marginal_utility: float
    incremental_cost: float
    total_utility: float
    total_cost: float


@dataclass(frozen=True)
class GreedyResult:
    selected: tuple[str, ...]
    utility: float
    cost: float
    steps: tuple[GreedyStep, ...]


def measured_greedy(
    candidates: Iterable[str],
    utility: UtilityFunction,
    incremental_cost: CostFunction,
    budget: float,
    *,
    min_marginal_utility: float = 0.0,
) -> GreedyResult:
    """Greedily remeasure every remaining candidate after each selection."""

    remaining = sorted(set(candidates))
    selected: set[str] = set()
    current_utility = utility(frozenset())
    total_cost = 0.0
    history: list[GreedyStep] = []

    while remaining:
        choices: list[tuple[float, float, str, float, float]] = []
        for module_name in remaining:
            cost = float(incremental_cost(module_name))
            if cost <= 0 or total_cost + cost > budget:
                continue
            candidate_utility = float(utility(frozenset(selected | {module_name})))
            marginal = candidate_utility - current_utility
            ratio = marginal / cost
            choices.append((ratio, marginal, module_name, cost, candidate_utility))
        if not choices:
            break
        ratio, marginal, module_name, cost, candidate_utility = max(
            choices, key=lambda item: (item[0], item[1], item[2])
        )
        if marginal <= min_marginal_utility or ratio <= 0:
            break
        selected.add(module_name)
        remaining.remove(module_name)
        total_cost += cost
        current_utility = candidate_utility
        history.append(
            GreedyStep(
                module_name=module_name,
                marginal_utility=marginal,
                incremental_cost=cost,
                total_utility=current_utility,
                total_cost=total_cost,
            )
        )
    return GreedyResult(
        selected=tuple(step.module_name for step in history),
        utility=current_utility,
        cost=total_cost,
        steps=tuple(history),
    )


def additive_greedy(
    scores: dict[str, float],
    costs: dict[str, float],
    budget: float,
) -> GreedyResult:
    """Convenience allocator when module utilities are assumed additive."""

    def utility(selected: frozenset[str]) -> float:
        return sum(scores[name] for name in selected)

    return measured_greedy(scores, utility, costs.__getitem__, budget)
