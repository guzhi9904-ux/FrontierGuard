"""Quantization protocols and kernel-independent fake quantization."""

from frontierguard.quant.calibration import (
    ActivationMaxCollector,
    CalibrationArtifact,
    build_smoothquant_scales,
)
from frontierguard.quant.controller import QuantizationController, instrument_linear_layers
from frontierguard.quant.factory import instrument_reference_backend
from frontierguard.quant.kv_cache import fake_quantize_kv_cache
from frontierguard.quant.tensor import fake_quantize

__all__ = [
    "QuantizationController",
    "ActivationMaxCollector",
    "CalibrationArtifact",
    "build_smoothquant_scales",
    "fake_quantize",
    "fake_quantize_kv_cache",
    "instrument_linear_layers",
    "instrument_reference_backend",
]
