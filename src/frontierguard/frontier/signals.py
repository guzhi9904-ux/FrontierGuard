"""Teacher-forced distribution signals."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class TokenSignals:
    jsd: torch.Tensor
    margin_drop: torch.Tensor
    nll_gap: torch.Tensor
    fp_entropy: torch.Tensor
    quant_entropy: torch.Tensor


@dataclass
class LogitSketch:
    indices: torch.Tensor
    probabilities: torch.Tensor
    rest_probability: torch.Tensor
    target_nll: torch.Tensor
    target_margin: torch.Tensor


def _target_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    target = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    masked = logits.clone()
    masked.scatter_(-1, targets.unsqueeze(-1), float("-inf"))
    alternative = masked.amax(dim=-1)
    return target - alternative


def _entropy_from_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    probabilities = log_probs.exp()
    return -(probabilities * log_probs).sum(dim=-1)


def token_signals(
    fp_logits: torch.Tensor,
    quant_logits: torch.Tensor,
    targets: torch.Tensor,
) -> TokenSignals:
    """Compute aligned signals with shape ``(..., sequence)``."""

    if fp_logits.shape != quant_logits.shape:
        raise ValueError("full-precision and quantized logits must have identical shapes")
    if fp_logits.shape[:-1] != targets.shape:
        raise ValueError("targets must match logits except for the vocabulary axis")

    fp_log = F.log_softmax(fp_logits.float(), dim=-1)
    quant_log = F.log_softmax(quant_logits.float(), dim=-1)
    midpoint_log = torch.logaddexp(fp_log, quant_log) - torch.log(
        torch.tensor(2.0, device=fp_log.device)
    )
    fp_prob = fp_log.exp()
    quant_prob = quant_log.exp()
    jsd = 0.5 * (
        (fp_prob * (fp_log - midpoint_log)).sum(dim=-1)
        + (quant_prob * (quant_log - midpoint_log)).sum(dim=-1)
    )
    fp_margin = _target_margin(fp_logits.float(), targets)
    quant_margin = _target_margin(quant_logits.float(), targets)
    fp_nll = F.cross_entropy(
        fp_logits.float().reshape(-1, fp_logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    quant_nll = F.cross_entropy(
        quant_logits.float().reshape(-1, quant_logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    return TokenSignals(
        jsd=jsd,
        margin_drop=fp_margin - quant_margin,
        nll_gap=quant_nll - fp_nll,
        fp_entropy=_entropy_from_log_probs(fp_log),
        quant_entropy=_entropy_from_log_probs(quant_log),
    )


def aggregate_by_spans(
    values: torch.Tensor,
    spans: list[tuple[int, int]],
    *,
    quantile: float = 0.95,
) -> list[dict[str, float]]:
    """Aggregate a 1D token signal into reasoning steps."""

    if values.ndim != 1:
        raise ValueError("aggregate_by_spans expects a 1D tensor")
    summaries: list[dict[str, float]] = []
    for start, end in spans:
        if start < 0 or end <= start or end > values.numel():
            raise ValueError(f"invalid token span {(start, end)} for {values.numel()} tokens")
        window = values[start:end].float()
        summaries.append(
            {
                "mean": float(window.mean().item()),
                "max": float(window.max().item()),
                "q95": float(torch.quantile(window, quantile).item()),
            }
        )
    return summaries


def sketch_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    top_k: int = 32,
) -> LogitSketch:
    """Compress a distribution to FP top-k categories plus an `other` bin."""

    values = logits.float()
    k = min(top_k, values.shape[-1])
    log_normalizer = torch.logsumexp(values, dim=-1, keepdim=True)
    top_values, top_indices = torch.topk(values, k=k, dim=-1)
    probabilities = torch.exp(top_values - log_normalizer)
    rest = (1.0 - probabilities.sum(dim=-1)).clamp_min(0.0)
    target_logits = values.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    target_nll = log_normalizer.squeeze(-1) - target_logits
    target_margin = _target_margin(values, targets)
    return LogitSketch(
        indices=top_indices,
        probabilities=probabilities,
        rest_probability=rest,
        target_nll=target_nll,
        target_margin=target_margin,
    )


def compare_to_sketch(
    quant_logits: torch.Tensor,
    targets: torch.Tensor,
    fp_sketch: LogitSketch,
    *,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Compare quant logits on the fixed FP top-k partition."""

    values = quant_logits.float()
    log_normalizer = torch.logsumexp(values, dim=-1, keepdim=True)
    quant_probabilities = torch.exp(
        values.gather(-1, fp_sketch.indices) - log_normalizer
    )
    quant_rest = (1.0 - quant_probabilities.sum(dim=-1)).clamp_min(0.0)
    fp_partition = torch.cat(
        (fp_sketch.probabilities, fp_sketch.rest_probability.unsqueeze(-1)), dim=-1
    ).clamp_min(eps)
    quant_partition = torch.cat(
        (quant_probabilities, quant_rest.unsqueeze(-1)), dim=-1
    ).clamp_min(eps)
    fp_partition = fp_partition / fp_partition.sum(dim=-1, keepdim=True)
    quant_partition = quant_partition / quant_partition.sum(dim=-1, keepdim=True)
    midpoint = 0.5 * (fp_partition + quant_partition)
    jsd = 0.5 * (
        (fp_partition * (fp_partition.log() - midpoint.log())).sum(dim=-1)
        + (quant_partition * (quant_partition.log() - midpoint.log())).sum(dim=-1)
    )
    target_logits = values.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    quant_nll = log_normalizer.squeeze(-1) - target_logits
    quant_margin = _target_margin(values, targets)
    return {
        "jsd": jsd,
        "margin_drop": fp_sketch.target_margin - quant_margin,
        "nll_gap": quant_nll - fp_sketch.target_nll,
    }
