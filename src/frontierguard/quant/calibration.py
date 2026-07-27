"""Calibration utilities for activation-aware reference quantization."""

from __future__ import annotations

import contextlib
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn


@dataclass
class CalibrationArtifact:
    """Per-linear input scales plus provenance needed to reproduce them."""

    scales: dict[str, torch.Tensor]
    metadata: dict[str, str] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        try:
            from safetensors.torch import save_file
        except ImportError as error:  # pragma: no cover - exercised without research extras
            raise RuntimeError(
                "saving calibration artifacts requires `pip install safetensors`"
            ) from error
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        tensors = {
            name: scale.detach().float().cpu().contiguous()
            for name, scale in self.scales.items()
        }
        save_file(tensors, str(destination), metadata=self.metadata)

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationArtifact":
        try:
            from safetensors import safe_open
        except ImportError as error:  # pragma: no cover - exercised without research extras
            raise RuntimeError(
                "loading calibration artifacts requires `pip install safetensors`"
            ) from error
        source = Path(path)
        scales: dict[str, torch.Tensor] = {}
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
            for name in handle.keys():
                scales[name] = handle.get_tensor(name)
        return cls(scales=scales, metadata=metadata)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ActivationMaxCollector(contextlib.AbstractContextManager["ActivationMaxCollector"]):
    """Collect per-input-channel absolute maxima for selected Linear modules."""

    def __init__(
        self,
        model: nn.Module,
        *,
        include: str | None = None,
        exclude: str | None = r"(lm_head|embed_tokens)",
    ):
        self.model = model
        self.include_pattern = re.compile(include) if include else None
        self.exclude_pattern = re.compile(exclude) if exclude else None
        self.maxima: dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []

    def _selected(self, name: str, module: nn.Module) -> bool:
        return (
            bool(name)
            and isinstance(module, nn.Linear)
            and (self.include_pattern is None or self.include_pattern.search(name) is not None)
            and (self.exclude_pattern is None or self.exclude_pattern.search(name) is None)
        )

    def __enter__(self) -> "ActivationMaxCollector":
        if self._handles:
            raise RuntimeError("activation collector is already active")

        for name, module in self.model.named_modules():
            if not self._selected(name, module):
                continue

            def collect(
                _module: nn.Module,
                arguments: tuple[Any, ...],
                *,
                module_name: str = name,
            ) -> None:
                if not arguments or not isinstance(arguments[0], torch.Tensor):
                    return
                inputs = arguments[0].detach().float()
                reduce_dims = tuple(range(inputs.ndim - 1))
                maximum = inputs.abs().amax(dim=reduce_dims)
                previous = self.maxima.get(module_name)
                self.maxima[module_name] = (
                    maximum if previous is None else torch.maximum(previous, maximum)
                )

            self._handles.append(module.register_forward_pre_hook(collect))
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def cpu_maxima(self) -> dict[str, torch.Tensor]:
        return {
            name: maximum.detach().float().cpu()
            for name, maximum in self.maxima.items()
        }


@torch.no_grad()
def build_smoothquant_scales(
    model: nn.Module,
    activation_maxima: dict[str, torch.Tensor],
    *,
    alpha: float = 0.5,
    eps: float = 1e-5,
    minimum: float = 1e-3,
    maximum: float = 1e3,
) -> dict[str, torch.Tensor]:
    """Build SmoothQuant-style channel scales.

    For a linear projection ``Y = X W^T``, the reference backend evaluates
    ``(X / s) (W * s)^T`` before fake quantization. The transformation is
    algebraically equivalent in full precision while balancing activation and
    weight outliers.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    modules = dict(model.named_modules())
    scales: dict[str, torch.Tensor] = {}
    for name, act_max in activation_maxima.items():
        module = modules.get(name)
        if not isinstance(module, nn.Linear):
            raise KeyError(f"calibration module is not a Linear layer: {name}")
        if act_max.numel() != module.in_features:
            raise ValueError(
                f"activation scale width mismatch for {name}: "
                f"{act_max.numel()} != {module.in_features}"
            )
        weight_max = module.weight.detach().float().abs().amax(dim=0).cpu()
        act = act_max.detach().float().cpu()
        valid = (act > eps) & (weight_max > eps)
        scale = torch.ones_like(act)
        scale[valid] = (
            act[valid].pow(alpha) / weight_max[valid].pow(1.0 - alpha)
        )
        scales[name] = scale.clamp(minimum, maximum)
    return scales


@contextlib.contextmanager
def calibration_mode(model: nn.Module) -> Iterator[None]:
    """Run calibration without gradients and restore the model training flag."""

    training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            yield
    finally:
        model.train(training)
