"""Build verified trace records from generated responses."""

from __future__ import annotations

import hashlib
from typing import Any

from frontierguard.schemas import TraceRecord
from frontierguard.traces.segment import segment_reasoning
from frontierguard.traces.verify import extract_final_answer, verify_math_answer


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_trace(
    *,
    problem_id: str,
    problem: str,
    response: str,
    reference_answer: str,
    model_id: str,
    model_revision: str | None,
    seed: int,
    generation_config: dict[str, Any],
    tokenizer: Any | None = None,
    token_ids: list[int] | None = None,
    truncated: bool = False,
    dataset_hash: str | None = None,
) -> TraceRecord:
    extracted = extract_final_answer(response)
    return TraceRecord(
        problem_id=problem_id,
        problem=problem,
        response=response,
        reference_answer=reference_answer,
        extracted_answer=extracted,
        correct=verify_math_answer(extracted, reference_answer),
        steps=segment_reasoning(response, tokenizer),
        model_id=model_id,
        model_revision=model_revision,
        seed=seed,
        generation_config=generation_config,
        prompt_hash=stable_hash(problem),
        dataset_hash=dataset_hash,
        token_ids=token_ids,
        truncated=truncated,
    )
