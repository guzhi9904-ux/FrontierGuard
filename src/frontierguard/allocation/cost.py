"""Separate weight and KV cost accounting."""

from __future__ import annotations

from dataclasses import dataclass

from frontierguard.schemas import PrecisionAction, PrecisionMap


@dataclass(frozen=True)
class ModuleCost:
    name: str
    parameter_count: int
    layer_index: int | None = None
    kv_elements_per_token: int = 0


class CostModel:
    def __init__(self, modules: list[ModuleCost]):
        self.modules = {item.name: item for item in modules}

    def weight_bits(self, precision_map: PrecisionMap) -> int:
        return sum(
            item.parameter_count * precision_map.action_for(name).weight_bits
            for name, item in self.modules.items()
        )

    def effective_weight_bits(self, precision_map: PrecisionMap) -> float:
        parameters = sum(item.parameter_count for item in self.modules.values())
        return self.weight_bits(precision_map) / parameters if parameters else 0.0

    def weight_bytes(self, precision_map: PrecisionMap) -> float:
        return self.weight_bits(precision_map) / 8.0

    def kv_bytes_per_token(self, precision_map: PrecisionMap) -> float:
        total_bits = 0
        by_layer: dict[int, list[tuple[str, ModuleCost]]] = {}
        for name, item in self.modules.items():
            if item.layer_index is None:
                continue
            by_layer.setdefault(item.layer_index, []).append((name, item))
        for items in by_layer.values():
            # A layer stores one K/V cache. If any module action requests a
            # higher KV precision, conservatively charge the whole layer.
            bits = max(precision_map.action_for(name).kv_bits for name, _ in items)
            elements = max(item.kv_elements_per_token for _, item in items)
            total_bits += elements * bits
        return total_bits / 8.0

    def incremental_weight_bytes(
        self,
        module_name: str,
        low_action: PrecisionAction,
        high_action: PrecisionAction,
    ) -> float:
        item = self.modules[module_name]
        return item.parameter_count * (high_action.weight_bits - low_action.weight_bits) / 8.0
