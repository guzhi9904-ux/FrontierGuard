"""Problem-level stability summaries for compression-damage attribution."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or right.size != left.size:
        return None
    centered_left = left - left.mean()
    centered_right = right - right.mean()
    denominator = float(
        np.sqrt(
            np.dot(centered_left, centered_left)
            * np.dot(centered_right, centered_right)
        )
    )
    if denominator == 0:
        return None
    return float(np.dot(centered_left, centered_right) / denominator)


def relative_depth_key(row: dict[str, Any], *, bins: int = 4) -> str:
    if bins <= 0:
        raise ValueError("depth bins must be positive")
    depth = float(row["relative_depth"])
    index = min(bins - 1, max(0, int(depth * bins)))
    return (
        f"depth_{index + 1}_of_{bins}."
        f"{row['family']}.{row['projection']}"
    )


def _bootstrap_interval(
    values: np.ndarray,
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    if values.size == 1:
        value = float(values[0])
        return value, value
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def aggregate_damage_rows(
    rows: Iterable[dict[str, Any]],
    *,
    key: Callable[[dict[str, Any]], str],
    score_field: str = "predicted_nll_rescue",
    bootstrap_samples: int = 5000,
    confidence: float = 0.95,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Aggregate repeated traces inside each problem before bootstrapping."""

    by_key_problem: dict[tuple[str, str], list[float]] = defaultdict(list)
    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        group = key(row)
        problem_id = str(row["problem_id"])
        by_key_problem[(group, problem_id)].append(float(row[score_field]))
        metadata.setdefault(
            group,
            {
                "family": row.get("family"),
                "projection": row.get("projection"),
                "layer_index": row.get("layer_index"),
                "relative_depth": row.get("relative_depth"),
                "parameter_count": row.get("parameter_count"),
            },
        )
    grouped: dict[str, list[float]] = defaultdict(list)
    for (group, _problem_id), values in by_key_problem.items():
        grouped[group].append(float(np.mean(values)))

    result = []
    for group, raw_values in grouped.items():
        values = np.asarray(raw_values, dtype=np.float64)
        digest = int(
            hashlib.sha256(group.encode("utf-8")).hexdigest()[:8],
            16,
        )
        lower, upper = _bootstrap_interval(
            values,
            samples=bootstrap_samples,
            confidence=confidence,
            seed=seed ^ digest,
        )
        result.append(
            {
                "key": group,
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "standard_error": (
                    float(values.std(ddof=1) / np.sqrt(values.size))
                    if values.size > 1
                    else 0.0
                ),
                "positive_fraction": float((values > 0).mean()),
                "ci_lower": lower,
                "ci_upper": upper,
                "problems": int(values.size),
                **metadata[group],
            }
        )
    return sorted(result, key=lambda item: (-item["mean"], item["key"]))


def top_k_jaccard(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("top-k must be positive")
    left_keys = {item["key"] for item in left[:k]}
    right_keys = {item["key"] for item in right[:k]}
    union = left_keys | right_keys
    return len(left_keys & right_keys) / len(union) if union else 1.0


def shared_rank_correlation(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> float | None:
    left_rank = {item["key"]: index for index, item in enumerate(left)}
    right_rank = {item["key"]: index for index, item in enumerate(right)}
    shared = sorted(left_rank.keys() & right_rank.keys())
    if len(shared) < 2:
        return None
    left_values = np.asarray([left_rank[item] for item in shared], dtype=np.float64)
    right_values = np.asarray([right_rank[item] for item in shared], dtype=np.float64)
    return _pearson(left_values, right_values)


def exact_patch_diagnostics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    exact = [
        row
        for row in rows
        if row.get("exact_nll_rescue") is not None
        and row.get("exact_role") is not None
    ]
    predicted = np.asarray(
        [float(row["predicted_nll_rescue"]) for row in exact],
        dtype=np.float64,
    )
    observed = np.asarray(
        [float(row["exact_nll_rescue"]) for row in exact],
        dtype=np.float64,
    )
    correlation = _pearson(predicted, observed)

    def role_mean(role: str) -> float | None:
        values = [
            float(row["exact_nll_rescue"])
            for row in exact
            if row["exact_role"] == role
        ]
        return float(np.mean(values)) if values else None

    return {
        "rows": len(exact),
        "pearson_predicted_vs_exact": correlation,
        "sign_agreement": (
            float(np.mean(np.sign(predicted) == np.sign(observed)))
            if predicted.size
            else None
        ),
        "predicted_top_mean_exact_rescue": role_mean("predicted_top"),
        "matched_random_mean_exact_rescue": role_mean("matched_random"),
    }
