"""Composable high-level research workflows."""

from __future__ import annotations

import contextlib
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import torch

from frontierguard.frontier.counterfactual import (
    PrefixOutcome,
    SeedOutcome,
    paired_specific_effect,
)
from frontierguard.frontier.detector import FrontierDetector
from frontierguard.frontier.pipeline import (
    ScanProgressCallback,
    TeacherForcedScan,
    finalize_frontier,
    scan_teacher_forced,
    scan_teacher_forced_low_memory,
)
from frontierguard.models.hf_runner import HFRunner, SamplingConfig
from frontierguard.schemas import TraceRecord
from frontierguard.traces.segment import segment_reasoning
from frontierguard.traces.verify import classify_generation, extract_answer_details


WorkflowProgressCallback = Callable[[str, dict[str, Any]], None]


def prepare_trace_steps(runner: HFRunner, trace: TraceRecord) -> None:
    """Rebuild phase-aware spans so legacy trace JSON remains usable."""

    trace.steps = segment_reasoning(trace.response, runner.tokenizer)


def trace_input_ids(runner: HFRunner, trace: TraceRecord) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Reconstruct prompt + verified response and target-vector step spans."""

    prepare_trace_steps(runner, trace)
    prompt_ids = runner.encode_chat(trace.problem)
    response_ids = runner.tokenizer(
        trace.response, add_special_tokens=False, return_tensors="pt"
    )["input_ids"].to(runner.device)
    full_ids = torch.cat((prompt_ids, response_ids), dim=-1)
    prompt_length = prompt_ids.shape[-1]
    spans: list[tuple[int, int]] = []
    for step in trace.steps:
        if step.token_start is None or step.token_end is None:
            raise ValueError("trace steps need token spans for teacher-forcing scan")
        # Logit position j predicts input_ids[j+1]. The response's first token is
        # therefore target position prompt_length - 1.
        spans.append(
            (
                prompt_length - 1 + step.token_start,
                prompt_length - 1 + step.token_end,
            )
        )
    return full_ids, spans


def scan_trace(
    runner: HFRunner,
    trace: TraceRecord,
    *,
    detector: FrontierDetector | None = None,
    shortlist_size: int = 5,
    exhaustive_step_threshold: int = 16,
    candidate_neighbor_radius: int = 1,
    low_memory: bool = True,
    progress_callback: ScanProgressCallback | None = None,
) -> TeacherForcedScan:
    input_ids, all_spans = trace_input_ids(runner, trace)
    eligible_indices = [
        index for index, step in enumerate(trace.steps) if step.eligible
    ]
    if not eligible_indices:
        raise ValueError(f"{trace.problem_id} contains no eligible reasoning steps")
    spans = [all_spans[index] for index in eligible_indices]
    scanner = scan_teacher_forced_low_memory if low_memory else scan_teacher_forced
    scan = scanner(
        runner,
        input_ids,
        spans,
        detector=detector,
        shortlist_size=shortlist_size,
        exhaustive_step_threshold=exhaustive_step_threshold,
        candidate_neighbor_radius=candidate_neighbor_radius,
        progress_callback=progress_callback,
    )
    scan.shortlist = [eligible_indices[index] for index in scan.shortlist]
    scan.step_indices = eligible_indices
    return scan


def counterfactual_trace(
    runner: HFRunner,
    trace: TraceRecord,
    sampling: SamplingConfig,
    *,
    seeds: list[int],
    candidate_steps: list[int] | None = None,
    progress_callback: WorkflowProgressCallback | None = None,
    bootstrap_samples: int = 5000,
    confidence_level: float = 0.95,
    min_trustworthy_seeds: int = 4,
) -> dict[str, Any]:
    """Estimate BF16 and quantized success after verified trace prefixes."""

    prepare_trace_steps(runner, trace)
    prompt_ids = runner.encode_chat(trace.problem)
    response = trace.response
    ends = [0] + [step.char_end for step in trace.steps]
    if candidate_steps is not None:
        needed = {
            prefix
            for step_index in candidate_steps
            for prefix in (step_index, step_index + 1)
        }
        prefix_indices = [index for index in range(len(ends)) if index in needed]
    else:
        prefix_indices = list(range(len(ends)))

    total_rollouts = 2 * len(prefix_indices) * len(seeds)
    rollout_index = 0
    if progress_callback is not None:
        progress_callback(
            "plan",
            {
                "total_rollouts": total_rollouts,
                "prefixes": len(prefix_indices),
                "seeds": len(seeds),
            },
        )

    def evaluate_condition(
        condition: str,
        full_precision: bool,
    ) -> list[PrefixOutcome]:
        nonlocal rollout_index
        outcomes: list[PrefixOutcome] = []
        manager = runner.full_precision() if full_precision else contextlib.nullcontext()
        with manager:
            for prefix_position, prefix_index in enumerate(prefix_indices, start=1):
                response_prefix = response[: ends[prefix_index]]
                prefix_response_ids = runner.tokenizer(
                    response_prefix, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(runner.device)
                input_ids = torch.cat((prompt_ids, prefix_response_ids), dim=-1)
                seed_outcomes = []
                for rollout_seed in seeds:
                    rollout_index += 1
                    config = SamplingConfig(
                        temperature=sampling.temperature,
                        top_p=sampling.top_p,
                        max_new_tokens=sampling.max_new_tokens,
                        seed=rollout_seed,
                    )
                    details = {
                        "condition": condition,
                        "prefix_index": prefix_index,
                        "prefix_position": prefix_position,
                        "prefixes": len(prefix_indices),
                        "seed": rollout_seed,
                        "rollout": rollout_index,
                        "total_rollouts": total_rollouts,
                        "max_new_tokens": config.max_new_tokens,
                    }
                    if progress_callback is not None:
                        progress_callback("rollout_start", details)

                    def generation_progress(completed: int, total: int) -> None:
                        if progress_callback is not None:
                            progress_callback(
                                "token",
                                {
                                    **details,
                                    "completed_tokens": completed,
                                    "max_new_tokens": total,
                                },
                            )

                    generated = runner.generate(
                        input_ids,
                        config,
                        progress_callback=(
                            generation_progress
                            if progress_callback is not None
                            else None
                        ),
                    )
                    full_output = response_prefix + generated["text"]
                    extraction = extract_answer_details(full_output)
                    classification = classify_generation(
                        full_output,
                        extraction,
                        trace.reference_answer,
                        truncated=generated["truncated"],
                    )
                    extracted_answer = extraction.answer
                    success = bool(classification["correct"])
                    if progress_callback is not None:
                        progress_callback(
                            "rollout_end",
                            {
                                **details,
                                "output_tokens": generated["output_tokens"],
                                "truncated": generated["truncated"],
                                "latency_seconds": generated["latency_seconds"],
                                "success": success,
                                "extracted_answer": extracted_answer,
                                "extraction_method": extraction.method,
                                "failure_type": classification["failure_type"],
                            },
                        )
                    seed_outcomes.append(
                        SeedOutcome(
                            seed=rollout_seed,
                            success=success,
                            continuation=generated["text"],
                            extracted_answer=extracted_answer,
                            output_tokens=generated["output_tokens"],
                            truncated=generated["truncated"],
                            latency_seconds=generated["latency_seconds"],
                            extraction_method=extraction.method,
                            answer_candidates=tuple(
                                extraction.to_dict()["candidates"]
                            ),
                            answer_candidate_count=extraction.candidate_count,
                            answer_candidates_truncated=(
                                extraction.candidates_truncated
                            ),
                            failure_type=classification["failure_type"],
                            repetition_fraction=classification[
                                "repetition_fraction"
                            ],
                            eos_reached=classification["eos_reached"],
                        )
                    )
                outcomes.append(
                    PrefixOutcome(
                        successes=sum(item.success for item in seed_outcomes),
                        trials=len(seed_outcomes),
                        outcomes=tuple(seed_outcomes),
                    )
                )
        return outcomes

    fp_outcomes = evaluate_condition("bf16", True)
    quant_outcomes = evaluate_condition("quantized", False)
    fp_by_prefix = dict(zip(prefix_indices, fp_outcomes))
    quant_by_prefix = dict(zip(prefix_indices, quant_outcomes))
    requested_steps = (
        list(range(len(trace.steps)))
        if candidate_steps is None
        else candidate_steps
    )
    fp_gain = []
    quant_gain = []
    specific_gain = []
    bypass_ci_lower = []
    bypass_ci_upper = []
    detection_ci_lower = []
    statistically_eligible = []
    paired_effects = []
    gain_step_indices = []
    for step_index in requested_steps:
        if step_index not in fp_by_prefix or step_index + 1 not in fp_by_prefix:
            continue
        fp_before = fp_by_prefix[step_index]
        fp_after = fp_by_prefix[step_index + 1]
        quant_before = quant_by_prefix[step_index]
        quant_after = quant_by_prefix[step_index + 1]
        effect = paired_specific_effect(
            fp_before,
            fp_after,
            quant_before,
            quant_after,
            samples=bootstrap_samples,
            confidence=confidence_level,
            seed=step_index,
        )
        fp_delta = fp_after.probability - fp_before.probability
        quant_delta = quant_after.probability - quant_before.probability
        eligible = effect.trials >= min_trustworthy_seeds
        gain_step_indices.append(step_index)
        fp_gain.append(fp_delta)
        quant_gain.append(quant_delta)
        specific_gain.append(effect.estimate)
        bypass_ci_lower.append(effect.lower)
        bypass_ci_upper.append(effect.upper)
        detection_ci_lower.append(effect.lower if eligible else 0.0)
        statistically_eligible.append(eligible)
        paired_effects.append(asdict(effect))

    def summarize(outcomes: list[PrefixOutcome]) -> dict[str, Any]:
        items = [item for outcome in outcomes for item in outcome.outcomes]
        failures = Counter(item.failure_type for item in items)
        return {
            "rollouts": len(items),
            "successes": sum(item.success for item in items),
            "truncated": sum(item.truncated for item in items),
            "eos_reached": sum(item.eos_reached for item in items),
            "failure_types": dict(sorted(failures.items())),
            "mean_repetition_fraction": (
                sum(item.repetition_fraction for item in items) / len(items)
                if items
                else 0.0
            ),
        }

    return {
        "prefix_indices": prefix_indices,
        "gain_step_indices": gain_step_indices,
        "fp_outcomes": [asdict(item) for item in fp_outcomes],
        "quant_outcomes": [asdict(item) for item in quant_outcomes],
        "fp_gain": fp_gain,
        "quant_gain": quant_gain,
        "specific_gain": specific_gain,
        "bypass_ci_lower": bypass_ci_lower,
        "bypass_ci_upper": bypass_ci_upper,
        "detection_ci_lower": detection_ci_lower,
        "statistically_eligible": statistically_eligible,
        "paired_effects": paired_effects,
        "condition_summaries": {
            "bf16": summarize(fp_outcomes),
            "quantized": summarize(quant_outcomes),
        },
        "confidence_level": confidence_level,
        "bootstrap_samples": bootstrap_samples,
        "min_trustworthy_seeds": min_trustworthy_seeds,
    }


def complete_frontier(
    scan: TeacherForcedScan,
    counterfactual: dict[str, Any],
    detector: FrontierDetector | None = None,
) -> dict[str, Any]:
    """Map sparse counterfactual gains back to the full step sequence."""

    gain_by_step = dict(
        zip(counterfactual["gain_step_indices"], counterfactual["specific_gain"])
    )
    lower_by_step = dict(
        zip(counterfactual["gain_step_indices"], counterfactual["bypass_ci_lower"])
    )
    upper_by_step = dict(
        zip(counterfactual["gain_step_indices"], counterfactual["bypass_ci_upper"])
    )
    detection_lower_by_step = dict(
        zip(
            counterfactual["gain_step_indices"],
            counterfactual["detection_ci_lower"],
        )
    )
    bypass = [
        float(gain_by_step.get(step_index, 0.0))
        for step_index in scan.step_indices
    ]
    bypass_ci_lower = [
        float(lower_by_step.get(step_index, 0.0))
        for step_index in scan.step_indices
    ]
    bypass_ci_upper = [
        float(upper_by_step.get(step_index, 0.0))
        for step_index in scan.step_indices
    ]
    detection_ci_lower = [
        float(detection_lower_by_step.get(step_index, 0.0))
        for step_index in scan.step_indices
    ]
    result = finalize_frontier(
        scan,
        bypass,
        bypass_ci_lower=detection_ci_lower,
        detector=detector,
    )
    selected_step = (
        scan.step_indices[result.step_index]
        if result.step_index is not None
        else None
    )
    nll_values = torch.tensor(scan.step_nll_gap, dtype=torch.float64)
    nll_median = float(nll_values.median().item()) if nll_values.numel() else 0.0
    teacher_candidates = [
        step_index
        for step_index, nll in zip(scan.step_indices, scan.step_nll_gap)
        if nll > 0.0 and nll >= nll_median
    ]
    teacher_first_error = teacher_candidates[0] if teacher_candidates else None
    positive_causal = [
        step_index
        for step_index, gain in zip(
            counterfactual["gain_step_indices"],
            counterfactual["specific_gain"],
        )
        if gain > 0.0
    ]
    causal_first_error = positive_causal[0] if positive_causal else None
    recovery_step = None
    recovery_gain = 0.0
    recovery_lower = 0.0
    recovery_upper = 0.0
    recovery_trustworthy = False
    if positive_causal:
        recovery_step = max(
            positive_causal,
            key=lambda step_index: gain_by_step[step_index],
        )
        recovery_gain = float(gain_by_step[recovery_step])
        recovery_lower = float(lower_by_step[recovery_step])
        recovery_upper = float(upper_by_step[recovery_step])
        eligible_by_step = dict(
            zip(
                counterfactual["gain_step_indices"],
                counterfactual.get(
                    "statistically_eligible",
                    [False] * len(counterfactual["gain_step_indices"]),
                ),
            )
        )
        recovery_trustworthy = bool(
            eligible_by_step.get(recovery_step, False) and recovery_lower > 0.0
        )
    window_points = (
        [
            value
            for value in (teacher_first_error, causal_first_error, recovery_step)
            if value is not None
        ]
        if recovery_step is not None
        else []
    )
    frontier_window = [min(window_points), max(window_points)] if window_points else None
    return {
        "step_index": selected_step,
        "trustworthy": result.trustworthy,
        "confidence": (
            float(counterfactual.get("confidence_level", result.confidence))
            if result.trustworthy
            else 0.0
        ),
        "confidence_level": float(
            counterfactual.get("confidence_level", result.confidence)
        ),
        "evidence_score": result.evidence_score,
        "step_indices": scan.step_indices,
        "step_jsd": scan.step_jsd,
        "step_margin_drop": scan.step_margin_drop,
        "step_nll_gap": scan.step_nll_gap,
        "bypass_gain": bypass,
        "bypass_ci_lower": bypass_ci_lower,
        "bypass_ci_upper": bypass_ci_upper,
        "combined_score": result.combined_score,
        "first_error_step": teacher_first_error,
        "causal_first_error_step": causal_first_error,
        "recovery_frontier_step": recovery_step,
        "frontier_window": frontier_window,
        "recovery_gain": recovery_gain,
        "recovery_ci_lower": recovery_lower,
        "recovery_ci_upper": recovery_upper,
        "recovery_trustworthy": recovery_trustworthy,
        "frontier_definitions": {
            "first_error_step": (
                "earliest eligible step with positive NLL gap at or above "
                "the per-trace median; diagnostic candidate, not a hypothesis test"
            ),
            "causal_first_error_step": (
                "earliest evaluated step with positive paired "
                "quantization-specific prefix gain"
            ),
            "recovery_frontier_step": (
                "evaluated step with maximum positive paired "
                "quantization-specific prefix gain"
            ),
            "frontier_window": (
                "inclusive span covering teacher-forced onset, earliest causal "
                "recovery and maximum recovery"
            ),
        },
    }
