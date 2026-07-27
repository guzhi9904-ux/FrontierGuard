"""Factories that keep quantization backend selection out of experiment scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn

from frontierguard.quant.calibration import CalibrationArtifact
from frontierguard.quant.controller import QuantizationController, instrument_linear_layers
from frontierguard.schemas import PrecisionMap


REFERENCE_BACKENDS = ("rtn", "smoothquant")
EXTERNAL_BACKENDS = ("flatquant_external", "quarot_external")


def instrument_reference_backend(
    model: nn.Module,
    precision_map: PrecisionMap,
    *,
    backend: str = "rtn",
    calibration_scales: str | Path | None = None,
    materialize_weights: bool = True,
) -> QuantizationController:
    """Instrument a model with an explicitly identified reference backend."""

    if backend not in REFERENCE_BACKENDS:
        if backend in EXTERNAL_BACKENDS:
            raise ValueError(
                f"{backend} requires its locked upstream adapter and transformed checkpoint; "
                "it cannot be emulated by the reference fake-quant backend"
            )
        raise ValueError(f"unknown reference quantization backend: {backend}")

    scales = None
    metadata: dict[str, Any] = {
        "backend": f"{backend}_reference_fake",
        "backend_selector": backend,
        "fake_quant": True,
        "packed_kernel": False,
    }
    if backend == "smoothquant":
        if calibration_scales is None:
            raise ValueError("smoothquant backend requires --calibration-scales")
        artifact = CalibrationArtifact.load(calibration_scales)
        scales = artifact.scales
        metadata.update(
            {
                "calibration_scales": str(Path(calibration_scales).resolve()),
                "calibration": artifact.metadata,
            }
        )
    elif calibration_scales is not None:
        raise ValueError("--calibration-scales is only valid with backend=smoothquant")

    return instrument_linear_layers(
        model,
        precision_map,
        materialize_weights=materialize_weights,
        input_scales=scales,
        backend_metadata=metadata,
    )
