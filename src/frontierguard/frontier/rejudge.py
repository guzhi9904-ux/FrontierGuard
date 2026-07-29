"""Offline re-judging for saved counterfactual continuations."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any

from frontierguard.frontier.counterfactual import (
    PrefixOutcome,
    SeedOutcome,
    paired_specific_effect,
)
from frontierguard.schemas import ReasoningStep
from frontierguard.traces.verify import classify_generation, extract_answer_details


def _prefix_outcome(
    raw: dict[str, Any],
    *,
    response_prefix: str,
    reference_answer: str,
) -> PrefixOutcome:
    outcomes = []
    for item in raw["outcomes"]:
        full_output = response_prefix + item.get("continuation", "")
        extraction = extract_answer_details(full_output)
        classification = classify_generation(
            full_output,
            extraction,
            reference_answer,
            truncated=bool(item.get("truncated", False)),
        )
        outcomes.append(
            SeedOutcome(
                seed=int(item["seed"]),
                success=bool(classification["correct"]),
                continuation=item.get("continuation", ""),
                extracted_answer=extraction.answer,
                output_tokens=int(item.get("output_tokens", 0)),
                truncated=bool(item.get("truncated", False)),
                latency_seconds=item.get("latency_seconds"),
                extraction_method=extraction.method,
                answer_candidates=tuple(extraction.to_dict()["candidates"]),
                answer_candidate_count=extraction.candidate_count,
                answer_candidates_truncated=extraction.candidates_truncated,
                failure_type=classification["failure_type"],
                repetition_fraction=classification["repetition_fraction"],
                eos_reached=classification["eos_reached"],
            )
        )
    return PrefixOutcome(
        successes=sum(item.success for item in outcomes),
        trials=len(outcomes),
        outcomes=tuple(outcomes),
    )


def _summary(outcomes: list[PrefixOutcome]) -> dict[str, Any]:
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


def rejudge_counterfactual(
    raw: dict[str, Any],
    *,
    response: str,
    steps: list[ReasoningStep],
    reference_answer: str,
    bootstrap_samples: int = 5000,
    confidence_level: float = 0.95,
    min_trustworthy_seeds: int = 4,
) -> dict[str, Any]:
    """Recompute saved rollout judgments and every available adjacent effect."""

    prefix_indices = [int(item) for item in raw["prefix_indices"]]
    ends = [0] + [step.char_end for step in steps]
    fp_outcomes = []
    quant_outcomes = []
    for prefix_index, fp_raw, quant_raw in zip(
        prefix_indices,
        raw["fp_outcomes"],
        raw["quant_outcomes"],
    ):
        if prefix_index >= len(ends):
            raise ValueError(
                f"saved prefix index {prefix_index} exceeds {len(steps)} trace steps"
            )
        response_prefix = response[: ends[prefix_index]]
        fp_outcomes.append(
            _prefix_outcome(
                fp_raw,
                response_prefix=response_prefix,
                reference_answer=reference_answer,
            )
        )
        quant_outcomes.append(
            _prefix_outcome(
                quant_raw,
                response_prefix=response_prefix,
                reference_answer=reference_answer,
            )
        )

    fp_by_prefix = dict(zip(prefix_indices, fp_outcomes))
    quant_by_prefix = dict(zip(prefix_indices, quant_outcomes))
    gain_step_indices = [
        index
        for index in range(len(steps))
        if index in fp_by_prefix and index + 1 in fp_by_prefix
    ]
    fp_gain = []
    quant_gain = []
    specific_gain = []
    lower = []
    upper = []
    detection_lower = []
    eligible = []
    effects = []
    for step_index in gain_step_indices:
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
        is_eligible = effect.trials >= min_trustworthy_seeds
        fp_gain.append(fp_after.probability - fp_before.probability)
        quant_gain.append(quant_after.probability - quant_before.probability)
        specific_gain.append(effect.estimate)
        lower.append(effect.lower)
        upper.append(effect.upper)
        detection_lower.append(effect.lower if is_eligible else 0.0)
        eligible.append(is_eligible)
        effects.append(asdict(effect))

    return {
        **raw,
        "gain_step_indices": gain_step_indices,
        "fp_outcomes": [asdict(item) for item in fp_outcomes],
        "quant_outcomes": [asdict(item) for item in quant_outcomes],
        "fp_gain": fp_gain,
        "quant_gain": quant_gain,
        "specific_gain": specific_gain,
        "bypass_ci_lower": lower,
        "bypass_ci_upper": upper,
        "detection_ci_lower": detection_lower,
        "statistically_eligible": eligible,
        "paired_effects": effects,
        "condition_summaries": {
            "bf16": _summary(fp_outcomes),
            "quantized": _summary(quant_outcomes),
        },
        "confidence_level": confidence_level,
        "bootstrap_samples": bootstrap_samples,
        "min_trustworthy_seeds": min_trustworthy_seeds,
    }
