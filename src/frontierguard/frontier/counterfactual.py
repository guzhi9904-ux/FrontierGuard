"""Counterfactual prefix rollout estimators."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedOutcome:
    seed: int
    success: bool
    continuation: str = ""
    extracted_answer: str | None = None
    output_tokens: int = 0
    truncated: bool = False
    latency_seconds: float | None = None


@dataclass(frozen=True)
class PrefixOutcome:
    successes: int
    trials: int
    outcomes: tuple[SeedOutcome, ...] = ()

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
    outcomes = []
    for seed in seeds:
        output = rollout(prefix, seed)
        outcomes.append(
            SeedOutcome(
                seed=int(seed),
                success=bool(verify(output)),
                continuation=output,
            )
        )
    return PrefixOutcome(
        successes=sum(item.success for item in outcomes),
        trials=len(outcomes),
        outcomes=tuple(outcomes),
    )


@dataclass(frozen=True)
class PairedStepEffect:
    estimate: float
    lower: float
    upper: float
    trials: int
    confidence: float
    seed_effects: tuple[dict[str, float | int | bool], ...]


def paired_specific_effect(
    fp_before: PrefixOutcome,
    fp_after: PrefixOutcome,
    quant_before: PrefixOutcome,
    quant_after: PrefixOutcome,
    *,
    samples: int = 5000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedStepEffect:
    """Bootstrap per-seed difference-in-differences for one injected step."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    groups = [
        {item.seed: item for item in outcome.outcomes}
        for outcome in (fp_before, fp_after, quant_before, quant_after)
    ]
    common_seeds = sorted(set.intersection(*(set(group) for group in groups)))
    if not common_seeds:
        raise ValueError("paired effect needs per-seed outcomes shared by all conditions")

    seed_effects = []
    effects = []
    for paired_seed in common_seeds:
        fp_left = groups[0][paired_seed].success
        fp_right = groups[1][paired_seed].success
        quant_left = groups[2][paired_seed].success
        quant_right = groups[3][paired_seed].success
        fp_delta = int(fp_right) - int(fp_left)
        quant_delta = int(quant_right) - int(quant_left)
        specific_delta = quant_delta - fp_delta
        effects.append(float(specific_delta))
        seed_effects.append(
            {
                "seed": paired_seed,
                "fp_before": fp_left,
                "fp_after": fp_right,
                "quant_before": quant_left,
                "quant_after": quant_right,
                "fp_delta": fp_delta,
                "quant_delta": quant_delta,
                "specific_delta": specific_delta,
            }
        )

    values = np.asarray(effects, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    bootstrap = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return PairedStepEffect(
        estimate=float(values.mean()),
        lower=float(np.quantile(bootstrap, alpha)),
        upper=float(np.quantile(bootstrap, 1.0 - alpha)),
        trials=int(values.size),
        confidence=confidence,
        seed_effects=tuple(seed_effects),
    )


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
