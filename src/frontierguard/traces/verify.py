"""Math-answer extraction, verification and generation-failure diagnostics."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction


_HASH_ANSWER = re.compile(r"####\s*([^\n]+)")
_FINAL_MARKER = re.compile(
    r"(?:final\s+answer|answer\s+is|therefore(?:,\s*)?[^.\n]{0,80}?)"
    r"\s*[:=]?\s*([^\n]+)",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?%?")
_SIMPLE_FRACTION = re.compile(r"[-+]?\d[\d,]*/[-+]?\d[\d,]*")
_LATEX_FRACTION = re.compile(
    r"\\frac\s*\{\s*([-+]?\d[\d,]*)\s*\}\s*\{\s*([-+]?\d[\d,]*)\s*\}"
)
_MARKDOWN = re.compile(r"[*_`]+")
_WORD = re.compile(r"[A-Za-z0-9%]+")


@dataclass(frozen=True)
class AnswerCandidate:
    value: str
    method: str
    raw: str
    start: int
    end: int


@dataclass(frozen=True)
class AnswerExtraction:
    answer: str | None
    method: str | None
    candidates: tuple[AnswerCandidate, ...]
    candidate_count: int
    candidates_truncated: bool

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "method": self.method,
            "candidates": [asdict(item) for item in self.candidates],
            "candidate_count": self.candidate_count,
            "candidates_truncated": self.candidates_truncated,
        }


def _balanced_boxed(text: str) -> list[tuple[str, int, int]]:
    """Extract ``\\boxed{...}`` values while allowing nested braces."""

    results = []
    marker = r"\boxed{"
    cursor = 0
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            return results
        content_start = start + len(marker)
        depth = 1
        index = content_start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            results.append((text[content_start : index - 1], start, index))
            cursor = index
        else:
            cursor = content_start


def _clean_candidate(value: str) -> str:
    cleaned = value.strip().strip("$")
    cleaned = _MARKDOWN.sub("", cleaned)
    cleaned = cleaned.strip(" \t\r\n.:;，。")
    return cleaned


def _values_from_fragment(fragment: str) -> list[str]:
    latex = _LATEX_FRACTION.findall(fragment)
    values = [
        f"{left.replace(',', '')}/{right.replace(',', '')}"
        for left, right in latex
    ]
    without_latex = _LATEX_FRACTION.sub(" ", fragment)
    values.extend(
        item.replace(",", "") for item in _SIMPLE_FRACTION.findall(without_latex)
    )
    without_fractions = _SIMPLE_FRACTION.sub(" ", without_latex)
    values.extend(
        item.replace(",", "") for item in _NUMBER.findall(without_fractions)
    )
    return values


def extract_answer_details(text: str) -> AnswerExtraction:
    """Return the selected answer and inspectable candidates.

    The last explicit boxed, GSM-style or final-answer candidate wins; numeric
    fallback is used only when no explicit candidate exists. Within one
    fragment the final numeric value is preferred, which handles prose such as
    ``Billy helps **240 people**`` without returning Markdown delimiters.
    """

    candidates: list[AnswerCandidate] = []
    for raw, start, end in _balanced_boxed(text):
        values = _values_from_fragment(raw)
        value = values[-1] if values else _clean_candidate(raw)
        if value:
            candidates.append(AnswerCandidate(value, "boxed", raw, start, end))

    for match in _HASH_ANSWER.finditer(text):
        raw = match.group(1)
        values = _values_from_fragment(raw)
        value = values[-1] if values else _clean_candidate(raw)
        if value:
            candidates.append(
                AnswerCandidate(value, "hash_answer", raw, match.start(), match.end())
            )

    for match in _FINAL_MARKER.finditer(text):
        raw = match.group(1)
        values = _values_from_fragment(raw)
        if values:
            candidates.append(
                AnswerCandidate(
                    values[-1],
                    "final_marker",
                    raw,
                    match.start(),
                    match.end(),
                )
            )

    for match in _NUMBER.finditer(text):
        candidates.append(
            AnswerCandidate(
                match.group(0).replace(",", ""),
                "numeric_fallback",
                match.group(0),
                match.start(),
                match.end(),
            )
        )

    explicit = [item for item in candidates if item.method != "numeric_fallback"]
    fallback = [item for item in candidates if item.method == "numeric_fallback"]
    retained = explicit[-64:] + fallback[-32:]
    candidate_count = len(candidates)
    candidates_truncated = len(retained) < candidate_count
    if explicit:
        selected = max(explicit, key=lambda item: (item.start, item.end))
        return AnswerExtraction(
            selected.value,
            selected.method,
            tuple(retained),
            candidate_count,
            candidates_truncated,
        )
    if fallback:
        selected = fallback[-1]
        return AnswerExtraction(
            selected.value,
            selected.method,
            tuple(retained),
            candidate_count,
            candidates_truncated,
        )
    return AnswerExtraction(None, None, (), 0, False)


def extract_final_answer(text: str) -> str | None:
    return extract_answer_details(text).answer


def _normalize_text(value: str) -> str:
    normalized = _clean_candidate(value).lower()
    normalized = normalized.replace("$", "").replace(",", "")
    normalized = normalized.replace("\\%", "%").replace(" ", "")
    normalized = normalized.rstrip(".")
    return normalized


def _numeric(value: str) -> Decimal | None:
    normalized = _normalize_text(value)
    latex = _LATEX_FRACTION.fullmatch(normalized)
    if latex:
        normalized = f"{latex.group(1)}/{latex.group(2)}"
    if normalized.endswith("%"):
        normalized = normalized[:-1]
        try:
            return Decimal(normalized) / Decimal(100)
        except InvalidOperation:
            return None
    if "/" in normalized and normalized.count("/") == 1:
        try:
            fraction = Fraction(normalized)
            return Decimal(fraction.numerator) / Decimal(fraction.denominator)
        except (ValueError, ZeroDivisionError, InvalidOperation):
            return None
    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


def _prediction_value(prediction: str) -> str:
    if _numeric(prediction) is not None:
        return prediction
    extracted = extract_final_answer(prediction)
    return extracted if extracted is not None else prediction


def verify_math_answer(
    prediction: str | None,
    reference: str,
    *,
    relative_tolerance: float = 1e-6,
    absolute_tolerance: float = 1e-8,
) -> bool:
    if prediction is None:
        return False
    prediction = _prediction_value(prediction)
    predicted_text = _normalize_text(prediction)
    reference_text = _normalize_text(extract_final_answer(reference) or reference)
    if predicted_text == reference_text:
        return True
    predicted_number = _numeric(predicted_text)
    reference_number = _numeric(reference_text)
    if predicted_number is None or reference_number is None:
        return False
    return math.isclose(
        float(predicted_number),
        float(reference_number),
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    )


def repetition_fraction(text: str, *, ngram: int = 4) -> float:
    """Fraction of word n-grams accounted for by the most frequent n-gram."""

    words = [item.lower() for item in _WORD.findall(text)]
    if len(words) < ngram:
        return 0.0
    counts: dict[tuple[str, ...], int] = {}
    for index in range(len(words) - ngram + 1):
        key = tuple(words[index : index + ngram])
        counts[key] = counts.get(key, 0) + 1
    total = len(words) - ngram + 1
    return max(counts.values(), default=0) / total


def classify_generation(
    text: str,
    extraction: AnswerExtraction,
    reference: str,
    *,
    truncated: bool,
    repetition_threshold: float = 0.08,
    require_eos: bool = True,
) -> dict:
    """Classify a rollout without conflating parser and reasoning failures."""

    answer_correct = verify_math_answer(extraction.answer, reference)
    selected_correct = answer_correct and (not require_eos or not truncated)
    matching_candidates = [
        item
        for item in extraction.candidates
        if item.method != "numeric_fallback"
        if verify_math_answer(item.value, reference)
    ]
    repetition = repetition_fraction(text)
    if selected_correct:
        failure_type = "none"
    elif truncated and repetition >= repetition_threshold:
        failure_type = "repetition"
    elif truncated:
        failure_type = "truncation"
    elif matching_candidates:
        failure_type = "parser_ambiguity"
    elif extraction.answer is None:
        failure_type = "answer_missing"
    else:
        failure_type = "wrong_answer"
    return {
        "failure_type": failure_type,
        "correct": selected_correct,
        "answer_correct": answer_correct,
        "strict_eos_required": require_eos,
        "truncated": bool(truncated),
        "eos_reached": not truncated,
        "repetition_fraction": repetition,
        "matching_reference_candidates": [asdict(item) for item in matching_candidates],
    }
