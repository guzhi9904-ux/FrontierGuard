"""Deterministic reasoning-step segmentation."""

from __future__ import annotations

import re
from typing import Any

from frontierguard.schemas import ReasoningStep


_BLANK_LINE = re.compile(r"\n\s*\n+")
_NUMBERED_STEP = re.compile(r"(?m)^(?=\s*(?:step\s*)?\d+[\.\):]\s+)", re.IGNORECASE)
_THINK_END = re.compile(r"</think\s*>", re.IGNORECASE)
_ANSWER_PREFIX = re.compile(
    r"^\s*(?:\*{0,2}|#{1,6}\s*)"
    r"(?P<label>final\s+answer|answer|solution)\s*[:：]?",
    re.IGNORECASE,
)
_MARKUP = re.compile(r"[*_`#>\[\]{}\\]")
_TAG = re.compile(r"<[^>]+>")


def _nonempty_spans(text: str, boundaries: list[int]) -> list[tuple[int, int]]:
    points = sorted({0, len(text), *boundaries})
    spans: list[tuple[int, int]] = []
    for left, right in zip(points, points[1:]):
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if right > left:
            spans.append((left, right))
    return spans


def _step_kind(text: str) -> str:
    stripped = text.strip()
    without_tags = _TAG.sub("", stripped).strip()
    if not without_tags:
        return "format"
    answer_match = _ANSWER_PREFIX.match(without_tags)
    if answer_match is not None:
        remainder = without_tags[answer_match.end() :].strip()
        plain = _MARKUP.sub("", remainder).strip(" :：.-")
        if not plain:
            return "format"
        if answer_match.group("label").lower() == "solution":
            return "content"
        return "answer"
    unmarked = _MARKUP.sub("", without_tags).strip()
    plain = unmarked.strip(" :：.-")
    if not plain:
        return "format"
    words = re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]+|\d+(?:\.\d+)?", plain)
    if unmarked.endswith((":", "：")) and len(words) <= 12 and "=" not in unmarked:
        return "format"
    return "content"


def segment_reasoning(text: str, tokenizer: Any | None = None) -> list[ReasoningStep]:
    """Split a response into stable line/paragraph-level reasoning steps.

    The primary boundary is a blank line. If none exist, line boundaries are
    used. Explicit numbered steps receive their own boundary. The final answer
    remains a step so that its margin can be inspected.
    """

    think_end = _THINK_END.search(text)
    reasoning_end = think_end.start() if think_end is not None else len(text)
    boundaries = [match.end() for match in _BLANK_LINE.finditer(text)]
    boundaries.extend(match.start() for match in _NUMBERED_STEP.finditer(text) if match.start())
    if think_end is not None:
        boundaries.extend((think_end.start(), think_end.end()))
    if not boundaries:
        boundaries.extend(match.end() for match in re.finditer(r"\n+", text))
    if not boundaries and text.strip():
        # Conservative sentence fallback; avoid splitting decimal points.
        boundaries.extend(
            match.end()
            for match in re.finditer(r"(?<!\d)[.!?](?:\s+|$)", text)
            if match.end() < len(text)
        )
    spans = _nonempty_spans(text, boundaries)

    token_offsets: list[tuple[int, int]] | None = None
    if tokenizer is not None:
        try:
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            token_offsets = [tuple(item) for item in encoded["offset_mapping"]]
        except (TypeError, KeyError, NotImplementedError):
            token_offsets = None

    steps: list[ReasoningStep] = []
    for index, (start, end) in enumerate(spans):
        token_start = token_end = None
        if token_offsets is not None:
            overlapping = [
                token_index
                for token_index, (left, right) in enumerate(token_offsets)
                if right > start and left < end
            ]
            if overlapping:
                token_start = overlapping[0]
                token_end = overlapping[-1] + 1
        phase = "reasoning" if start < reasoning_end else "presentation"
        kind = _step_kind(text[start:end])
        steps.append(
            ReasoningStep(
                index=index,
                text=text[start:end],
                char_start=start,
                char_end=end,
                token_start=token_start,
                token_end=token_end,
                phase=phase,
                kind=kind,
                eligible=phase == "reasoning" and kind == "content",
            )
        )
    return steps
