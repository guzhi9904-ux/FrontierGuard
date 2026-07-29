"""Architecture normalization for Qwen2 and Llama-family models."""

from __future__ import annotations

import re
from dataclasses import dataclass

from torch import nn

from frontierguard.quant.linear import FakeQuantLinear


PROJECTION_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class ModuleDescriptor:
    name: str
    layer_index: int
    family: str
    projection: str
    parameter_count: int


@dataclass(frozen=True)
class ModelAdapter:
    architecture: str
    layer_pattern: str = r"(?:^|\.)layers\.(\d+)\."

    def describe_modules(self, model: nn.Module) -> list[ModuleDescriptor]:
        pattern = re.compile(self.layer_pattern)
        descriptors: list[ModuleDescriptor] = []
        for name, module in model.named_modules():
            projection = next(
                (suffix for suffix in PROJECTION_SUFFIXES if name.endswith("." + suffix)),
                None,
            )
            if projection is None or not isinstance(module, (nn.Linear, FakeQuantLinear)):
                continue
            linear = module.linear if isinstance(module, FakeQuantLinear) else module
            match = pattern.search(name)
            if not match:
                continue
            family = "attention" if projection in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
            descriptors.append(
                ModuleDescriptor(
                    name=name,
                    layer_index=int(match.group(1)),
                    family=family,
                    projection=projection,
                    parameter_count=sum(parameter.numel() for parameter in linear.parameters()),
                )
            )
        return sorted(descriptors, key=lambda item: (item.layer_index, item.projection))

    def group_names(
        self, model: nn.Module, *, layers_per_block: int = 4
    ) -> dict[str, list[str]]:
        descriptors = self.describe_modules(model)
        groups: dict[str, list[str]] = {}
        for item in descriptors:
            block = item.layer_index // layers_per_block
            groups.setdefault(f"block_{block}.{item.family}", []).append(item.name)
            groups.setdefault(f"projection.{item.projection}", []).append(item.name)
            groups.setdefault(f"layer_{item.layer_index}", []).append(item.name)
            groups.setdefault(
                f"layer_{item.layer_index}.{item.family}", []
            ).append(item.name)
        return groups


def infer_adapter(model: nn.Module) -> ModelAdapter:
    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "")).lower()
    architectures = " ".join(getattr(config, "architectures", []) or []).lower()
    signature = f"{model_type} {architectures}"
    if "qwen2" in signature or "qwen3" in signature:
        return ModelAdapter("qwen")
    if "llama" in signature:
        return ModelAdapter("llama")
    # Most decoder-only Hugging Face models use the same `layers.N` naming.
    return ModelAdapter(model_type or "generic")
