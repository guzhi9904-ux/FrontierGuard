"""Quantization protocols and kernel-independent fake quantization."""

from frontierguard.quant.controller import QuantizationController, instrument_linear_layers
from frontierguard.quant.kv_cache import fake_quantize_kv_cache
from frontierguard.quant.tensor import fake_quantize

__all__ = [
    "QuantizationController",
    "fake_quantize",
    "fake_quantize_kv_cache",
    "instrument_linear_layers",
]
