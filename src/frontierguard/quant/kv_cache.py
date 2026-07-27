"""Duck-typed fake quantization for Transformers KV caches."""

from __future__ import annotations

from typing import Any

import torch

from frontierguard.quant.tensor import fake_quantize


def _quantize_pair(
    key: torch.Tensor,
    value: torch.Tensor,
    bits: int,
    group_size: int,
    symmetric: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        fake_quantize(key, bits, group_size=group_size, axis=-1, symmetric=symmetric),
        fake_quantize(value, bits, group_size=group_size, axis=-1, symmetric=symmetric),
    )


def fake_quantize_kv_cache(
    cache: Any,
    bits: int,
    *,
    group_size: int = 128,
    symmetric: bool = False,
) -> Any:
    """Fake-quantize legacy tuples or DynamicCache-like objects.

    The operation returns a new legacy tuple. DynamicCache-like objects are
    updated in place because Transformers cache classes are version-specific.
    """

    if cache is None or bits >= 16:
        return cache
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for index, (key, value) in enumerate(zip(cache.key_cache, cache.value_cache)):
            quantized_key, quantized_value = _quantize_pair(
                key, value, bits, group_size, symmetric
            )
            cache.key_cache[index] = quantized_key
            cache.value_cache[index] = quantized_value
        return cache
    if isinstance(cache, (tuple, list)):
        quantized_layers = []
        for layer in cache:
            if not isinstance(layer, (tuple, list)) or len(layer) < 2:
                raise TypeError("unsupported legacy KV-cache layer")
            key, value = _quantize_pair(layer[0], layer[1], bits, group_size, symmetric)
            quantized_layers.append((key, value, *layer[2:]))
        return tuple(quantized_layers)
    raise TypeError(f"unsupported KV cache type: {type(cache)!r}")
