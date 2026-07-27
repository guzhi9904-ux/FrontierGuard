"""Reference fake-quantization operators.

These functions prioritize transparent research semantics over speed. They
return dequantized floating-point tensors and therefore do not demonstrate
packed-kernel memory or latency gains.
"""

from __future__ import annotations

import torch


def _reshape_groups(
    tensor: torch.Tensor, group_size: int, axis: int
) -> tuple[torch.Tensor, tuple[int, ...], int, int]:
    moved = tensor.movedim(axis, -1)
    original_shape = moved.shape
    width = original_shape[-1]
    size = width if group_size == -1 or group_size >= width else group_size
    if size <= 0:
        raise ValueError("group_size must be positive or -1")
    padding = (-width) % size
    if padding:
        moved = torch.nn.functional.pad(moved, (0, padding))
    grouped = moved.reshape(*moved.shape[:-1], -1, size)
    return grouped, original_shape, width, padding


def fake_quantize(
    tensor: torch.Tensor,
    bits: int,
    *,
    group_size: int = -1,
    axis: int = -1,
    symmetric: bool = True,
    eps: float = 1e-8,
    ste: bool = False,
) -> torch.Tensor:
    """Quantize and dequantize ``tensor`` groupwise along ``axis``.

    ``bits >= 16`` is an identity. ``ste=True`` preserves an identity gradient
    while using the quantized value in the forward pass.
    """

    if bits >= 16 or not tensor.is_floating_point():
        return tensor
    if bits < 2 or bits > 8:
        raise ValueError(f"reference fake quant supports 2..8 bits; got {bits}")
    if tensor.numel() == 0:
        return tensor

    grouped, original_shape, width, _ = _reshape_groups(tensor, group_size, axis)
    compute = grouped.float()

    if symmetric:
        qmax = (1 << (bits - 1)) - 1
        qmin = -(1 << (bits - 1))
        absmax = compute.abs().amax(dim=-1, keepdim=True)
        scale = (absmax / max(qmax, 1)).clamp_min(eps)
        quantized = torch.round(compute / scale).clamp(qmin, qmax)
        dequantized = quantized * scale
    else:
        qmin = 0
        qmax = (1 << bits) - 1
        minimum = compute.amin(dim=-1, keepdim=True)
        maximum = compute.amax(dim=-1, keepdim=True)
        dynamic_range = maximum - minimum
        scale = (dynamic_range / max(qmax - qmin, 1)).clamp_min(eps)
        zero = torch.round(qmin - minimum / scale).clamp(qmin, qmax)
        quantized = torch.round(compute / scale + zero).clamp(qmin, qmax)
        dequantized = (quantized - zero) * scale
        # A constant affine group has no range to encode. It is exactly
        # representable by carrying the constant rather than fabricating a
        # near-zero scale/zero-point pair.
        dequantized = torch.where(dynamic_range <= eps, compute, dequantized)

    flattened = dequantized.reshape(*grouped.shape[:-2], -1)[..., :width]
    restored = flattened.reshape(original_shape).movedim(-1, axis).to(tensor.dtype)
    if ste:
        return tensor + (restored - tensor).detach()
    return restored


def quantization_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    delta = (candidate.float() - reference.float()).reshape(-1)
    denominator = reference.float().reshape(-1).norm().clamp_min(1e-12)
    return {
        "mse": float(delta.square().mean().item()),
        "max_abs": float(delta.abs().max().item()),
        "relative_l2": float((delta.norm() / denominator).item()),
    }
