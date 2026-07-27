"""Counterfactual prefix rollout estimators."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PrefixOutcome:
    successes: int
    trials: int

    @property
    def probability(self) -> float:
        return self.successes / self.trials if self.trials else float("nan")


RolloutFunction = Callable[[str, int], str]
VerifyFunction = Callable[[str], bool]


def estimate_prefix_success(
    prefix: str,
    seeds: Sequence[int],
    rollout: RolloutFunction,
    verify: VerifyFunction,
) -> PrefixOutcome:
    successes = sum(bool(verify(rollout(prefix, seed))) for seed in seeds)
    return PrefixOutcome(successes=successes, trials=len(seeds))


def bypass_gains(
    fp_outcomes: Sequence[PrefixOutcome],
    quant_outcomes: Sequence[PrefixOutcome],
) -> dict[str, list[float]]:
    if len(fp_outcomes) != len(quant_outcomes):
        raise ValueError("FP and quant outcomes must use the same prefixes")
    if len(fp_outcomes) < 2:
        return {"fp_gain": [], "quant_gain": [], "specific_gain": []}
    fp_prob = [item.probability for item in fp_outcomes]
    quant_prob = [item.probability for item in quant_outcomes]
    fp_gain = [right - left for left, right in zip(fp_prob, fp_prob[1:])]
    quant_gain = [right - left for left, right in zip(quant_prob, quant_prob[1:])]
    return {
        "fp_gain": fp_gain,
        "quant_gain": quant_gain,
        "specific_gain": [q - f for q, f in zip(quant_gain, fp_gain)],
    }
