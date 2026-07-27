"""Generic teacher-forced intervention measurements."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional as F

from frontierguard.quant.controller import QuantizationController
from frontierguard.schemas import PrecisionAction


def sequence_nll(logits: torch.Tensor, targets: torch.Tensor) -> float:
    value = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean",
    )
    return float(value.item())


@torch.inference_mode()
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
    return baseline_nll - rescued_nll
