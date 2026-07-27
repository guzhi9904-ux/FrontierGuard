"""Bootstrap estimators that keep generations from one problem together."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairedObservation:
    problem_id: str
    baseline: float
    method: float


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    lower: float
    upper: float
    samples: int
    problems: int


def _problem_means(
    observations: Iterable[PairedObservation],
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for item in observations:
        grouped[item.problem_id].append((item.baseline, item.method))
    baseline: list[float] = []
    method: list[float] = []
    for problem_id in sorted(grouped):
        pairs = np.asarray(grouped[problem_id], dtype=np.float64)
        baseline.append(float(pairs[:, 0].mean()))
        method.append(float(pairs[:, 1].mean()))
    return np.asarray(baseline), np.asarray(method)


def paired_problem_bootstrap(
    observations: Iterable[PairedObservation],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    baseline, method = _problem_means(observations)
    if baseline.size == 0:
        raise ValueError("bootstrap needs at least one problem")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, baseline.size, size=(samples, baseline.size))
    differences = (method[indices] - baseline[indices]).mean(axis=1)
    alpha = (1 - confidence) / 2
    return BootstrapResult(
        estimate=float((method - baseline).mean()),
        lower=float(np.quantile(differences, alpha)),
        upper=float(np.quantile(differences, 1 - alpha)),
        samples=samples,
        problems=int(baseline.size),
    )


def binomial_wilson(successes: int, trials: int, confidence_z: float = 1.96) -> tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("invalid successes/trials")
    proportion = successes / trials
    denominator = 1 + confidence_z**2 / trials
    center = (proportion + confidence_z**2 / (2 * trials)) / denominator
    radius = (
        confidence_z
        * np.sqrt(proportion * (1 - proportion) / trials + confidence_z**2 / (4 * trials**2))
        / denominator
    )
    return float(center - radius), float(center + radius)
