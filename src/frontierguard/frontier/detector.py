"""Difference-in-differences first-error-frontier detector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _zscore(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    deviation = array.std()
    if deviation < 1e-12:
        return np.zeros_like(array)
    return (array - array.mean()) / deviation


@dataclass(frozen=True)
class FrontierDetectorConfig:
    jsd_weight: float = 0.25
    margin_weight: float = 0.25
    bypass_weight: float = 0.50
    combined_threshold: float = 0.5
    bypass_threshold: float = 0.0
    require_persistence: bool = True
    confidence_level: float = 0.95

    def validate(self) -> None:
        weights = self.jsd_weight + self.margin_weight + self.bypass_weight
        if not np.isclose(weights, 1.0):
            raise ValueError(f"frontier weights must sum to one; got {weights}")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")


@dataclass(frozen=True)
class DetectionResult:
    step_index: int | None
    trustworthy: bool
    confidence: float
    evidence_score: float
    combined_score: list[float]


class FrontierDetector:
    def __init__(self, config: FrontierDetectorConfig | None = None):
        self.config = config or FrontierDetectorConfig()
        self.config.validate()

    def score(
        self,
        jsd: list[float],
        margin_drop: list[float],
        bypass_gain: list[float],
    ) -> list[float]:
        if not (len(jsd) == len(margin_drop) == len(bypass_gain)):
            raise ValueError("step-level frontier signals must have equal lengths")
        combined = (
            self.config.jsd_weight * _zscore(jsd)
            + self.config.margin_weight * _zscore(margin_drop)
            + self.config.bypass_weight * _zscore(bypass_gain)
        )
        return combined.tolist()

    def detect(
        self,
        jsd: list[float],
        margin_drop: list[float],
        bypass_gain: list[float],
        *,
        bypass_ci_lower: list[float] | None = None,
    ) -> DetectionResult:
        scores = self.score(jsd, margin_drop, bypass_gain)
        ci = bypass_ci_lower if bypass_ci_lower is not None else [0.0] * len(scores)
        for index, (score, gain, lower) in enumerate(zip(scores, bypass_gain, ci)):
            passes = (
                score > self.config.combined_threshold
                and gain > self.config.bypass_threshold
                and lower > 0
            )
            if not passes:
                continue
            if self.config.require_persistence and index + 1 < len(scores):
                distribution_persists = (
                    scores[index + 1] > 0
                    or jsd[index + 1] >= float(np.median(jsd))
                    or margin_drop[index + 1] >= float(np.median(margin_drop))
                )
                if not distribution_persists:
                    continue
            evidence_score = float(score * max(gain, 0.0))
            return DetectionResult(
                index,
                True,
                self.config.confidence_level,
                evidence_score,
                scores,
            )
        return DetectionResult(None, False, 0.0, 0.0, scores)

    def shortlist(
        self,
        jsd: list[float],
        margin_drop: list[float],
        *,
        top_k: int = 5,
    ) -> list[int]:
        if len(jsd) != len(margin_drop):
            raise ValueError("signals must have equal lengths")
        cheap = self.config.jsd_weight * _zscore(jsd) + self.config.margin_weight * _zscore(
            margin_drop
        )
        order = np.argsort(-cheap, kind="stable")
        return sorted(int(index) for index in order[:top_k])
