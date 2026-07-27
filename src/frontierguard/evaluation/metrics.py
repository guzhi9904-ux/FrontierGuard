"""Task, cost and stability metrics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def accuracy(correct: Iterable[bool]) -> float:
    values = np.asarray(list(correct), dtype=np.float64)
    return float(values.mean()) if values.size else float("nan")


def gap_recovery(bf16_accuracy: float, quant_accuracy: float, method_accuracy: float) -> float:
    denominator = bf16_accuracy - quant_accuracy
    return (method_accuracy - quant_accuracy) / denominator if denominator > 0 else float("nan")


def token_inflation_ratio(
    bf16_tokens: Sequence[int],
    quant_tokens: Sequence[int],
) -> float:
    if len(bf16_tokens) != len(quant_tokens):
        raise ValueError("paired token vectors must have equal length")
    fp = np.asarray(bf16_tokens, dtype=np.float64)
    quant = np.asarray(quant_tokens, dtype=np.float64)
    if fp.size == 0 or fp.mean() == 0:
        return float("nan")
    return float(quant.mean() / fp.mean())


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    left_rank = np.argsort(np.argsort(np.asarray(left), kind="stable"), kind="stable")
    right_rank = np.argsort(np.argsort(np.asarray(right), kind="stable"), kind="stable")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])
