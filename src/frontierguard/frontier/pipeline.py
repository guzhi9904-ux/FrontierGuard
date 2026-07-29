"""End-to-end teacher-forced frontier scan on one instrumented model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from frontierguard.frontier.detector import DetectionResult, FrontierDetector
from frontierguard.frontier.signals import (
    TokenSignals,
    aggregate_by_spans,
    compare_to_sketch,
    sketch_logits,
    token_signals,
)
from frontierguard.models.hf_runner import HFRunner


@dataclass
class TeacherForcedScan:
    token_signals: TokenSignals | None
    step_indices: list[int]
    step_jsd: list[float]
    step_margin_drop: list[float]
    step_nll_gap: list[float]
    shortlist: list[int]


ScanProgressCallback = Callable[[str, int, int], None]


@torch.inference_mode()
def scan_teacher_forced(
    runner: HFRunner,
    input_ids: torch.Tensor,
    target_spans: list[tuple[int, int]],
    *,
    detector: FrontierDetector | None = None,
    shortlist_size: int = 5,
    exhaustive_step_threshold: int = 16,
    candidate_neighbor_radius: int = 1,
    progress_callback: ScanProgressCallback | None = None,
) -> TeacherForcedScan:
    """Compare BF16 and quantized logits under exactly the same token prefix.

    ``target_spans`` use positions in the teacher-forcing target vector
    (``input_ids[:, 1:]``), not positions in the raw response string.
    """

    if runner.controller is None:
        raise ValueError("frontier scan needs an instrumented quantization controller")
    with runner.full_precision():
        fp_logits, targets = runner.teacher_forcing(input_ids)
    if progress_callback is not None:
        progress_callback("bf16", 1, 1)
    quant_logits, quant_targets = runner.teacher_forcing(input_ids)
    if progress_callback is not None:
        progress_callback("quantized", 1, 1)
    if not torch.equal(targets, quant_targets):
        raise RuntimeError("teacher-forcing targets changed between precision conditions")
    signals = token_signals(fp_logits, quant_logits, targets)
    jsd = aggregate_by_spans(signals.jsd.squeeze(0), target_spans)
    margin = aggregate_by_spans(signals.margin_drop.squeeze(0), target_spans)
    nll = aggregate_by_spans(signals.nll_gap.squeeze(0), target_spans)
    step_jsd = [item["q95"] for item in jsd]
    step_margin = [item["q95"] for item in margin]
    step_nll = [item["mean"] for item in nll]
    active_detector = detector or FrontierDetector()
    shortlist = active_detector.shortlist(
        step_jsd,
        step_margin,
        nll_gap=step_nll,
        top_k=shortlist_size,
        exhaustive_threshold=exhaustive_step_threshold,
        neighbor_radius=candidate_neighbor_radius,
    )
    return TeacherForcedScan(
        token_signals=signals,
        step_indices=list(range(len(target_spans))),
        step_jsd=step_jsd,
        step_margin_drop=step_margin,
        step_nll_gap=step_nll,
        shortlist=shortlist,
    )


@torch.inference_mode()
def scan_teacher_forced_low_memory(
    runner: HFRunner,
    input_ids: torch.Tensor,
    target_spans: list[tuple[int, int]],
    *,
    detector: FrontierDetector | None = None,
    shortlist_size: int = 5,
    exhaustive_step_threshold: int = 16,
    candidate_neighbor_radius: int = 1,
    top_k: int = 32,
    progress_callback: ScanProgressCallback | None = None,
) -> TeacherForcedScan:
    """Step-window scan using an FP top-k + tail JSD partition.

    Target NLL and margin remain exact. Only JSD is coarsened. Full-precision
    sketches are held on CPU while the quantized pass is evaluated.
    """

    if runner.controller is None:
        raise ValueError("frontier scan needs an instrumented quantization controller")
    sketches = []
    with runner.full_precision():
        for index, (start, end) in enumerate(target_spans, start=1):
            fp_logits, targets = runner.teacher_forcing_window(input_ids, start, end)
            sketch = sketch_logits(fp_logits, targets, top_k=top_k)
            sketches.append(
                type(sketch)(
                    indices=sketch.indices.cpu(),
                    probabilities=sketch.probabilities.cpu(),
                    rest_probability=sketch.rest_probability.cpu(),
                    target_nll=sketch.target_nll.cpu(),
                    target_margin=sketch.target_margin.cpu(),
                )
            )
            del fp_logits
            if progress_callback is not None:
                progress_callback("bf16", index, len(target_spans))

    step_jsd: list[float] = []
    step_margin: list[float] = []
    step_nll: list[float] = []
    for index, ((start, end), cpu_sketch) in enumerate(
        zip(target_spans, sketches),
        start=1,
    ):
        quant_logits, targets = runner.teacher_forcing_window(input_ids, start, end)
        device_sketch = type(cpu_sketch)(
            indices=cpu_sketch.indices.to(quant_logits.device),
            probabilities=cpu_sketch.probabilities.to(quant_logits.device),
            rest_probability=cpu_sketch.rest_probability.to(quant_logits.device),
            target_nll=cpu_sketch.target_nll.to(quant_logits.device),
            target_margin=cpu_sketch.target_margin.to(quant_logits.device),
        )
        comparison = compare_to_sketch(quant_logits, targets, device_sketch)
        step_jsd.append(float(torch.quantile(comparison["jsd"].float(), 0.95).item()))
        step_margin.append(
            float(torch.quantile(comparison["margin_drop"].float(), 0.95).item())
        )
        step_nll.append(float(comparison["nll_gap"].float().mean().item()))
        del quant_logits
        if progress_callback is not None:
            progress_callback("quantized", index, len(target_spans))

    active_detector = detector or FrontierDetector()
    shortlist = active_detector.shortlist(
        step_jsd,
        step_margin,
        nll_gap=step_nll,
        top_k=shortlist_size,
        exhaustive_threshold=exhaustive_step_threshold,
        neighbor_radius=candidate_neighbor_radius,
    )
    return TeacherForcedScan(
        token_signals=None,
        step_indices=list(range(len(target_spans))),
        step_jsd=step_jsd,
        step_margin_drop=step_margin,
        step_nll_gap=step_nll,
        shortlist=shortlist,
    )


def finalize_frontier(
    scan: TeacherForcedScan,
    bypass_gain: list[float],
    *,
    bypass_ci_lower: list[float] | None = None,
    detector: FrontierDetector | None = None,
) -> DetectionResult:
    active_detector = detector or FrontierDetector()
    return active_detector.detect(
        scan.step_jsd,
        scan.step_margin_drop,
        bypass_gain,
        bypass_ci_lower=bypass_ci_lower,
    )
