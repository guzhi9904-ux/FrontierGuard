"""Generic teacher-forced intervention measurements."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from frontierguard.quant.controller import QuantizationController
from frontierguard.schemas import PrecisionAction


@dataclass(frozen=True)
class LocalRescueMeasurement:
    baseline_nll: float
    rescued_nll: float

    @property
    def local_rescue(self) -> float:
        return self.baseline_nll - self.rescued_nll


def component_rescue_action(
    low: PrecisionAction,
    *,
    high_bits: int,
    component: str,
) -> PrecisionAction:
    """Raise selected linear components without lowering untouched precision."""

    if component not in {"weight", "activation", "weight_activation"}:
        raise ValueError(f"unsupported rescue component: {component}")
    action = PrecisionAction(
        weight_bits=(
            max(low.weight_bits, high_bits)
            if component in {"weight", "weight_activation"}
            else low.weight_bits
        ),
        activation_bits=(
            max(low.activation_bits, high_bits)
            if component in {"activation", "weight_activation"}
            else low.activation_bits
        ),
        kv_bits=low.kv_bits,
        weight_group_size=low.weight_group_size,
        kv_group_size=low.kv_group_size,
        symmetric_weight=low.symmetric_weight,
        symmetric_activation=low.symmetric_activation,
        symmetric_kv=low.symmetric_kv,
        enabled=low.enabled,
    )
    action.validate()
    if action == low:
        raise ValueError("selected rescue component does not increase precision")
    return action


def sequence_nll(logits: torch.Tensor, targets: torch.Tensor) -> float:
    value = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean",
    )
    return float(value.item())


@torch.inference_mode()
def measure_local_nll_rescue_details(
    model: torch.nn.Module,
    controller: QuantizationController,
    input_ids: torch.Tensor,
    target_start: int,
    target_end: int,
    module_names: Sequence[str],
    *,
    action: PrecisionAction | None = None,
    bf16_rescue: bool = False,
) -> LocalRescueMeasurement:
    """Return inspectable baseline and intervention NLL over a token window."""

    if target_start < 1 or target_end <= target_start or target_end > input_ids.shape[-1]:
        raise ValueError("invalid target token window")
    baseline_logits = model(input_ids=input_ids, use_cache=False).logits
    baseline_slice = baseline_logits[:, target_start - 1 : target_end - 1, :]
    targets = input_ids[:, target_start:target_end]
    baseline_nll = sequence_nll(baseline_slice, targets)

    with controller.intervention(
        module_names, action, disable_quantization=bf16_rescue
    ):
        rescued_logits = model(input_ids=input_ids, use_cache=False).logits
    rescued_slice = rescued_logits[:, target_start - 1 : target_end - 1, :]
    rescued_nll = sequence_nll(rescued_slice, targets)
    return LocalRescueMeasurement(
        baseline_nll=baseline_nll,
        rescued_nll=rescued_nll,
    )


def measure_local_nll_rescue(
    model: torch.nn.Module,
    controller: QuantizationController,
    input_ids: torch.Tensor,
    target_start: int,
    target_end: int,
    module_names: Sequence[str],
    *,
    action: PrecisionAction | None = None,
    bf16_rescue: bool = False,
) -> float:
    """Return baseline NLL minus intervention NLL over a token window."""

    return measure_local_nll_rescue_details(
        model,
        controller,
        input_ids,
        target_start,
        target_end,
        module_names,
        action=action,
        bf16_rescue=bf16_rescue,
    ).local_rescue
