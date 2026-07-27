"""Serializable schemas shared across FrontierGuard stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PrecisionAction:
    """Quantization action applied to a module or layer."""

    weight_bits: int = 4
    activation_bits: int = 4
    kv_bits: int = 4
    weight_group_size: int = 128
    kv_group_size: int = 128
    symmetric_weight: bool = True
    symmetric_activation: bool = True
    symmetric_kv: bool = False
    enabled: bool = True

    def validate(self) -> None:
        for name, value in (
            ("weight_bits", self.weight_bits),
            ("activation_bits", self.activation_bits),
            ("kv_bits", self.kv_bits),
        ):
            if value not in {2, 3, 4, 8, 16}:
                raise ValueError(f"{name} must be one of 2, 3, 4, 8, 16; got {value}")
        if self.weight_group_size == 0 or self.kv_group_size == 0:
            raise ValueError("group sizes must be positive or -1 for the full axis")


@dataclass
class PrecisionMap:
    """Default action plus module-specific overrides."""

    default: PrecisionAction = field(default_factory=PrecisionAction)
    modules: dict[str, PrecisionAction] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def action_for(self, module_name: str) -> PrecisionAction:
        return self.modules.get(module_name, self.default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "default": asdict(self.default),
            "modules": {name: asdict(action) for name, action in self.modules.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PrecisionMap":
        default = PrecisionAction(**value.get("default", {}))
        modules = {
            name: PrecisionAction(**action)
            for name, action in value.get("modules", {}).items()
        }
        result = cls(default=default, modules=modules, metadata=value.get("metadata", {}))
        result.default.validate()
        for action in result.modules.values():
            action.validate()
        return result


@dataclass(frozen=True)
class ReasoningStep:
    index: int
    text: str
    char_start: int
    char_end: int
    token_start: int | None = None
    token_end: int | None = None
    phase: str = "reasoning"
    kind: str = "content"
    eligible: bool = True


@dataclass
class TraceRecord:
    problem_id: str
    problem: str
    response: str
    reference_answer: str
    extracted_answer: str | None
    correct: bool
    steps: list[ReasoningStep]
    model_id: str
    model_revision: str | None
    seed: int
    generation_config: dict[str, Any]
    prompt_hash: str
    dataset_hash: str | None = None
    token_ids: list[int] | None = None
    truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrontierRecord:
    problem_id: str
    trace_id: str
    step_index: int | None
    confidence: float
    jsd: list[float]
    margin_drop: list[float]
    bypass_gain: list[float]
    combined_score: list[float]
    trustworthy: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationRecord:
    run_id: str
    problem_id: str
    condition: str
    seed: int
    output: str
    extracted_answer: str | None
    correct: bool
    prompt_tokens: int
    output_tokens: int
    truncated: bool
    latency_seconds: float | None = None
    peak_memory_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
