"""Precision-map instrumentation and reversible interventions."""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator, Sequence

from torch import nn

from frontierguard.quant.linear import FakeQuantLinear
from frontierguard.schemas import PrecisionAction, PrecisionMap


def _parent_and_key(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def _set_child(parent: nn.Module, key: str, value: nn.Module) -> None:
    if key.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(key)] = value
    else:
        setattr(parent, key, value)


class QuantizationController:
    def __init__(self, model: nn.Module, wrappers: dict[str, FakeQuantLinear]):
        self.model = model
        self.wrappers = wrappers

    @property
    def module_names(self) -> list[str]:
        return sorted(self.wrappers)

    def set_precision_map(self, precision_map: PrecisionMap) -> None:
        for name, wrapper in self.wrappers.items():
            wrapper.action = precision_map.action_for(name)
            if wrapper.is_materialized:
                wrapper.materialize()

    def set_enabled(self, enabled: bool) -> None:
        for wrapper in self.wrappers.values():
            wrapper.quantization_enabled = enabled

    def materialize_weights(self) -> None:
        """Move master weights to CPU and materialize each configured action."""

        for wrapper in self.wrappers.values():
            wrapper.materialize()

    def unmaterialize_weights(self) -> None:
        for wrapper in self.wrappers.values():
            wrapper.unmaterialize()

    @contextlib.contextmanager
    def disabled(self) -> Iterator[None]:
        previous = {name: wrapper.quantization_enabled for name, wrapper in self.wrappers.items()}
        materialized = {
            name: wrapper.is_materialized for name, wrapper in self.wrappers.items()
        }
        # A full BF16 generation must not copy every CPU master weight at every
        # token. Restore the resident parameters once for the whole context.
        for name, was_materialized in materialized.items():
            if was_materialized:
                self.wrappers[name].unmaterialize()
        self.set_enabled(False)
        try:
            yield
        finally:
            for name, enabled in previous.items():
                self.wrappers[name].quantization_enabled = enabled
            for name, was_materialized in materialized.items():
                if was_materialized:
                    self.wrappers[name].materialize()

    @contextlib.contextmanager
    def intervention(
        self,
        module_names: Sequence[str],
        action: PrecisionAction | None = None,
        *,
        disable_quantization: bool = False,
    ) -> Iterator[None]:
        """Temporarily restore modules to BF16 or another precision action."""

        unknown = sorted(set(module_names) - self.wrappers.keys())
        if unknown:
            raise KeyError(f"unknown intervention modules: {unknown}")
        prior = {
            name: (self.wrappers[name].action, self.wrappers[name].quantization_enabled)
            for name in module_names
        }
        for name in module_names:
            if action is not None:
                self.wrappers[name].action = action
            if disable_quantization:
                self.wrappers[name].quantization_enabled = False
        try:
            yield
        finally:
            for name, (old_action, old_enabled) in prior.items():
                self.wrappers[name].action = old_action
                self.wrappers[name].quantization_enabled = old_enabled

    def unwrap(self) -> None:
        for name, wrapper in list(self.wrappers.items()):
            wrapper.unmaterialize()
            parent, key = _parent_and_key(self.model, name)
            _set_child(parent, key, wrapper.linear)
        self.wrappers.clear()


def instrument_linear_layers(
    model: nn.Module,
    precision_map: PrecisionMap,
    *,
    include: str | None = None,
    exclude: str | None = r"(lm_head|embed_tokens)",
    materialize_weights: bool = False,
) -> QuantizationController:
    """Wrap selected Linear modules in place and return a controller."""

    include_pattern = re.compile(include) if include else None
    exclude_pattern = re.compile(exclude) if exclude else None
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if name
        and isinstance(module, nn.Linear)
        and not isinstance(module, FakeQuantLinear)
        and (include_pattern is None or include_pattern.search(name))
        and (exclude_pattern is None or not exclude_pattern.search(name))
    ]
    wrappers: dict[str, FakeQuantLinear] = {}
    for name, module in candidates:
        wrapper = FakeQuantLinear(module, name, precision_map.action_for(name))
        parent, key = _parent_and_key(model, name)
        _set_child(parent, key, wrapper)
        wrappers[name] = wrapper
    controller = QuantizationController(model, wrappers)
    if materialize_weights:
        controller.materialize_weights()
    return controller
