"""Math-answer extraction and conservative verification."""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction


_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
_HASH_ANSWER = re.compile(r"####\s*([^\n]+)")
_FINAL_ANSWER = re.compile(
    r"(?:final\s+answer|answer\s+is)\s*[:=]?\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def extract_final_answer(text: str) -> str | None:
    boxed = _BOXED.findall(text)
    if boxed:
        return boxed[-1].strip()
    hashed = _HASH_ANSWER.findall(text)
    if hashed:
        return hashed[-1].strip()
    final = _FINAL_ANSWER.findall(text)
    if final:
        return final[-1].strip().rstrip(".")
    numbers = _NUMBER.findall(text)
    return numbers[-1].replace(",", "") if numbers else None


def _normalize_text(value: str) -> str:
    normalized = value.strip().lower()
    normalized = normalized.replace("$", "").replace(",", "")
    normalized = normalized.replace("\\%", "%").replace(" ", "")
    normalized = normalized.rstrip(".")
    return normalized


def _numeric(value: str) -> Decimal | None:
    normalized = _normalize_text(value)
    if normalized.endswith("%"):
        normalized = normalized[:-1]
        try:
            return Decimal(normalized) / Decimal(100)
        except InvalidOperation:
            return None
    if "/" in normalized and normalized.count("/") == 1:
        try:
            return Decimal(Fraction(normalized).numerator) / Decimal(
                Fraction(normalized).denominator
            )
        except (ValueError, ZeroDivisionError, InvalidOperation):
            return None
    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


def verify_math_answer(
    prediction: str | None,
    reference: str,
    *,
    relative_tolerance: float = 1e-6,
    absolute_tolerance: float = 1e-8,
) -> bool:
    if prediction is None:
        return False
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
