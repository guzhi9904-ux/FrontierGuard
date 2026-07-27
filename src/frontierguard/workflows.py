"""Composable high-level research workflows."""

from __future__ import annotations

import contextlib
from dataclasses import asdict
from typing import Any

import torch

from frontierguard.frontier.counterfactual import (
    PrefixOutcome,
    bypass_gains,
    estimate_prefix_success,
)
from frontierguard.frontier.detector import FrontierDetector
from frontierguard.frontier.pipeline import (
    TeacherForcedScan,
    finalize_frontier,
    scan_teacher_forced,
    scan_teacher_forced_low_memory,
)
from frontierguard.models.hf_runner import HFRunner, SamplingConfig
from frontierguard.schemas import TraceRecord
from frontierguard.traces.verify import extract_final_answer, verify_math_answer


def trace_input_ids(runner: HFRunner, trace: TraceRecord) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Reconstruct prompt + verified response and target-vector step spans."""

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
    low_memory: bool = True,
) -> TeacherForcedScan:
    input_ids, spans = trace_input_ids(runner, trace)
    scanner = scan_teacher_forced_low_memory if low_memory else scan_teacher_forced
    return scanner(
        runner,
        input_ids,
        spans,
        detector=detector,
        shortlist_size=shortlist_size,
    )


def counterfactual_trace(
    runner: HFRunner,
    trace: TraceRecord,
    sampling: SamplingConfig,
    *,
    seeds: list[int],
    candidate_steps: list[int] | None = None,
) -> dict[str, Any]:
    """Estimate BF16 and quantized success after verified trace prefixes."""

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

    def evaluate_condition(full_precision: bool) -> list[PrefixOutcome]:
        outcomes: list[PrefixOutcome] = []
        manager = runner.full_precision() if full_precision else contextlib.nullcontext()
        with manager:
            for prefix_index in prefix_indices:
                response_prefix = response[: ends[prefix_index]]
                prefix_response_ids = runner.tokenizer(
                    response_prefix, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(runner.device)
                input_ids = torch.cat((prompt_ids, prefix_response_ids), dim=-1)

                def rollout(_: str, seed: int) -> str:
                    config = SamplingConfig(
                        temperature=sampling.temperature,
                        top_p=sampling.top_p,
                        max_new_tokens=sampling.max_new_tokens,
                        seed=seed,
                    )
                    generated = runner.generate(input_ids, config)["text"]
                    return response_prefix + generated

                def verify(output: str) -> bool:
                    return verify_math_answer(
                        extract_final_answer(output), trace.reference_answer
                    )

                outcomes.append(estimate_prefix_success("", seeds, rollout, verify))
        return outcomes

    fp_outcomes = evaluate_condition(True)
    quant_outcomes = evaluate_condition(False)
    if candidate_steps is None:
        gains = bypass_gains(fp_outcomes, quant_outcomes)
        gain_step_indices = list(range(len(trace.steps)))
    else:
        fp_by_prefix = dict(zip(prefix_indices, fp_outcomes))
        quant_by_prefix = dict(zip(prefix_indices, quant_outcomes))
        fp_gain = []
        quant_gain = []
        gain_step_indices = []
        for step_index in candidate_steps:
            if step_index not in fp_by_prefix or step_index + 1 not in fp_by_prefix:
                continue
            fp_delta = (
                fp_by_prefix[step_index + 1].probability
                - fp_by_prefix[step_index].probability
            )
            quant_delta = (
                quant_by_prefix[step_index + 1].probability
                - quant_by_prefix[step_index].probability
            )
            gain_step_indices.append(step_index)
            fp_gain.append(fp_delta)
            quant_gain.append(quant_delta)
        gains = {
            "fp_gain": fp_gain,
            "quant_gain": quant_gain,
            "specific_gain": [q - f for q, f in zip(quant_gain, fp_gain)],
        }
    return {
        "prefix_indices": prefix_indices,
        "gain_step_indices": gain_step_indices,
        "fp_outcomes": [asdict(item) for item in fp_outcomes],
        "quant_outcomes": [asdict(item) for item in quant_outcomes],
        **gains,
    }


def complete_frontier(
    scan: TeacherForcedScan,
    counterfactual: dict[str, Any],
    detector: FrontierDetector | None = None,
) -> dict[str, Any]:
    """Map sparse counterfactual gains back to the full step sequence."""

    bypass = [0.0] * len(scan.step_jsd)
    for step_index, gain in zip(
        counterfactual["gain_step_indices"], counterfactual["specific_gain"]
    ):
        bypass[step_index] = float(gain)
    result = finalize_frontier(scan, bypass, detector=detector)
    return {
        "step_index": result.step_index,
        "trustworthy": result.trustworthy,
        "confidence": result.confidence,
        "step_jsd": scan.step_jsd,
        "step_margin_drop": scan.step_margin_drop,
        "step_nll_gap": scan.step_nll_gap,
        "bypass_gain": bypass,
        "combined_score": result.combined_score,
    }
