"""Module-level fake quantization wrapper."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from frontierguard.quant.tensor import fake_quantize
from frontierguard.schemas import PrecisionAction


class FakeQuantLinear(nn.Module):
    """Wrap a Linear layer without copying its parameters."""

    def __init__(
        self,
        linear: nn.Linear,
        module_name: str,
        action: PrecisionAction,
        *,
        input_scale: torch.Tensor | None = None,
    ):
        super().__init__()
        self.linear = linear
        self.module_name = module_name
        self.action = action
        self.quantization_enabled = True
        if input_scale is not None and input_scale.numel() != linear.in_features:
            raise ValueError(
                f"input scale width mismatch for {module_name}: "
                f"{input_scale.numel()} != {linear.in_features}"
            )
        scale = (
            input_scale.detach()
            .float()
            .reshape(linear.in_features)
            .to(linear.weight.device)
            if input_scale is not None
            else None
        )
        self.register_buffer("_input_scale", scale, persistent=False)
        self.register_buffer("_fp_weight_cpu", None, persistent=False)
        self._materialized_action: PrecisionAction | None = None

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.linear.weight

    @property
    def bias(self) -> torch.nn.Parameter | None:
        return self.linear.bias

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    @property
    def is_materialized(self) -> bool:
        return self._fp_weight_cpu is not None

    @property
    def has_input_scale(self) -> bool:
        return self._input_scale is not None

    def _uses_input_scale(self, action: PrecisionAction | None = None) -> bool:
        active = action or self.action
        return bool(
            self._input_scale is not None
            and self.quantization_enabled
            and active.enabled
            and (active.weight_bits < 16 or active.activation_bits < 16)
        )

    def _scaled_weight(
        self, weight: torch.Tensor, action: PrecisionAction
    ) -> torch.Tensor:
        if not self._uses_input_scale(action):
            return weight
        scale = self._input_scale.to(device=weight.device, dtype=weight.dtype)
        return weight * scale.unsqueeze(0)

    @torch.no_grad()
    def materialize(self) -> None:
        """Store the BF16 master on CPU and quantize the resident weight once."""

        if self.is_materialized:
            self._write_materialized(self.action)
            return
        self._fp_weight_cpu = self.linear.weight.detach().to("cpu", copy=True)
        self._write_materialized(self.action)

    @torch.no_grad()
    def _write_materialized(self, action: PrecisionAction) -> None:
        if self._fp_weight_cpu is None:
            raise RuntimeError("cannot materialize without a CPU master weight")
        master = self._fp_weight_cpu.to(
            device=self.linear.weight.device,
            dtype=self.linear.weight.dtype,
        )
        quantized = fake_quantize(
            self._scaled_weight(master, action),
            action.weight_bits,
            group_size=action.weight_group_size,
            axis=-1,
            symmetric=action.symmetric_weight,
        )
        self.linear.weight.copy_(quantized)
        self._materialized_action = action

    @torch.no_grad()
    def unmaterialize(self) -> None:
        if self._fp_weight_cpu is None:
            return
        self.linear.weight.copy_(
            self._fp_weight_cpu.to(
                device=self.linear.weight.device,
                dtype=self.linear.weight.dtype,
            )
        )
        self._fp_weight_cpu = None
        self._materialized_action = None

    def _weight_for_forward(self) -> torch.Tensor:
        if self._fp_weight_cpu is None:
            return fake_quantize(
                self._scaled_weight(self.linear.weight, self.action),
                self.action.weight_bits,
                group_size=self.action.weight_group_size,
                axis=-1,
                symmetric=self.action.symmetric_weight,
            )
        if (
            self.quantization_enabled
            and self.action.enabled
            and self.action == self._materialized_action
        ):
            return self.linear.weight
        master = self._fp_weight_cpu.to(
            device=self.linear.weight.device,
            dtype=self.linear.weight.dtype,
        )
        if not self.quantization_enabled or not self.action.enabled:
            return master
        return fake_quantize(
            self._scaled_weight(master, self.action),
            self.action.weight_bits,
            group_size=self.action.weight_group_size,
            axis=-1,
            symmetric=self.action.symmetric_weight,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if (not self.quantization_enabled or not self.action.enabled) and not self.is_materialized:
            return self.linear(inputs)
        transformed_inputs = inputs
        if self._uses_input_scale():
            scale = self._input_scale.to(device=inputs.device, dtype=inputs.dtype)
            transformed_inputs = inputs / scale
        quantized_inputs = fake_quantize(
            transformed_inputs,
            self.action.activation_bits
            if self.quantization_enabled and self.action.enabled
            else 16,
            group_size=-1,
            axis=-1,
            symmetric=self.action.symmetric_activation,
        )
        quantized_weight = self._weight_for_forward()
        return F.linear(quantized_inputs, quantized_weight, self.linear.bias)
