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
    new_tokens: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if new_tokens is not None:
        if new_tokens <= 0:
            raise ValueError("new_tokens must be positive")
        if key.ndim < 2 or value.ndim < 2:
            raise ValueError("KV tensors need sequence and channel dimensions")
        key_tokens = min(new_tokens, key.shape[-2])
        value_tokens = min(new_tokens, value.shape[-2])
        # Cache objects retain already-quantized history between decoding
        # iterations. Quantizing only the appended suffix avoids both O(T^2)
        # work and repeated quantization error on old KV entries.
        key_suffix = key[..., -key_tokens:, :]
        value_suffix = value[..., -value_tokens:, :]
        key_suffix.copy_(
            fake_quantize(
                key_suffix,
                bits,
                group_size=group_size,
                axis=-1,
                symmetric=symmetric,
            )
        )
        value_suffix.copy_(
            fake_quantize(
                value_suffix,
                bits,
                group_size=group_size,
                axis=-1,
                symmetric=symmetric,
            )
        )
        return key, value
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
    new_tokens: int | None = None,
) -> Any:
    """Fake-quantize legacy tuples or DynamicCache-like objects.

    The operation returns a new legacy tuple. DynamicCache-like objects are
    updated in place because Transformers cache classes are version-specific.
    When ``new_tokens`` is set, only the appended sequence suffix is quantized;
    this is the correct and efficient mode for autoregressive decoding.
    """

    if cache is None or bits >= 16:
        return cache
    # Transformers <=4.56 exposed parallel key_cache/value_cache lists.
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for index, (key, value) in enumerate(zip(cache.key_cache, cache.value_cache)):
            quantized_key, quantized_value = _quantize_pair(
                key, value, bits, group_size, symmetric, new_tokens
            )
            cache.key_cache[index] = quantized_key
            cache.value_cache[index] = quantized_value
        return cache
    # Transformers >=4.57 stores state in CacheLayer objects. Keep the cache
    # object and its layer metadata intact; replacing it with a legacy tuple
    # would discard sliding-window/offloading behavior.
    if hasattr(cache, "layers"):
        for index, layer in enumerate(cache.layers):
            if not hasattr(layer, "keys") or not hasattr(layer, "values"):
                raise TypeError(
                    f"unsupported KV cache layer at index {index}: {type(layer)!r}"
                )
            key = layer.keys
            value = layer.values
            if key is None or value is None:
                continue
            quantized_key, quantized_value = _quantize_pair(
                key, value, bits, group_size, symmetric, new_tokens
            )
            layer.keys = quantized_key
            layer.values = quantized_value
        return cache
    if isinstance(cache, (tuple, list)):
        quantized_layers = []
        for layer in cache:
            if not isinstance(layer, (tuple, list)) or len(layer) < 2:
                raise TypeError("unsupported legacy KV-cache layer")
            key, value = _quantize_pair(
                layer[0],
                layer[1],
                bits,
                group_size,
                symmetric,
                new_tokens,
            )
            quantized_layers.append((key, value, *layer[2:]))
        return tuple(quantized_layers)
    raise TypeError(f"unsupported KV cache type: {type(cache)!r}")
