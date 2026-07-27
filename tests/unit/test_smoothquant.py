import pytest
import torch
from torch import nn

from frontierguard.quant.calibration import (
    ActivationMaxCollector,
    CalibrationArtifact,
    build_smoothquant_scales,
)
from frontierguard.quant.controller import instrument_linear_layers
from frontierguard.quant.factory import instrument_reference_backend
from frontierguard.schemas import PrecisionAction, PrecisionMap


def test_activation_collector_and_scale_building():
    torch.manual_seed(17)
    model = nn.Sequential(nn.Linear(4, 3, bias=False)).eval()
    inputs = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
    with ActivationMaxCollector(model, exclude=None) as collector:
        model(inputs)
    maxima = collector.cpu_maxima()
    scales = build_smoothquant_scales(model, maxima, alpha=0.5)

    assert set(scales) == {"0"}
    assert torch.equal(maxima["0"], inputs.abs().squeeze(0))
    assert scales["0"].shape == (4,)
    assert torch.all(scales["0"] > 0)


def test_smooth_scale_keeps_full_precision_path_exact_and_is_reversible():
    torch.manual_seed(19)
    model = nn.Sequential(nn.Linear(4, 3)).eval()
    inputs = torch.randn(3, 4)
    reference = model(inputs)
    action = PrecisionAction(4, 4, 16, weight_group_size=2)
    controller = instrument_linear_layers(
        model,
        PrecisionMap(default=action),
        exclude=None,
        input_scales={"0": torch.tensor([0.5, 2.0, 1.5, 0.75])},
        backend_metadata={"backend": "smoothquant"},
    )

    quantized = model(inputs)
    with controller.disabled():
        restored = model(inputs)

    assert not torch.allclose(quantized, reference)
    assert torch.allclose(restored, reference)
    assert controller.metadata()["smoothed_linear_modules"] == 1


def test_identity_action_does_not_apply_smoothing_roundoff():
    torch.manual_seed(23)
    model = nn.Sequential(nn.Linear(4, 3)).eval()
    inputs = torch.randn(2, 4)
    reference = model(inputs)
    controller = instrument_linear_layers(
        model,
        PrecisionMap(default=PrecisionAction(16, 16, 16)),
        exclude=None,
        input_scales={"0": torch.tensor([0.25, 4.0, 0.5, 2.0])},
    )
    assert torch.equal(model(inputs), reference)
    controller.materialize_weights()
    assert torch.equal(model(inputs), reference)


def test_materialized_smoothquant_matches_dynamic_and_restores_bf16():
    torch.manual_seed(29)
    model = nn.Sequential(nn.Linear(6, 4)).eval()
    inputs = torch.randn(2, 6)
    reference = model(inputs)
    controller = instrument_linear_layers(
        model,
        PrecisionMap(
            default=PrecisionAction(4, 4, 16, weight_group_size=3)
        ),
        exclude=None,
        input_scales={"0": torch.tensor([0.5, 1.0, 2.0, 0.75, 1.5, 3.0])},
    )
    dynamic = model(inputs)
    controller.materialize_weights()
    materialized = model(inputs)
    with controller.disabled():
        restored = model(inputs)

    assert torch.allclose(dynamic, materialized)
    assert torch.allclose(restored, reference)


def test_calibration_artifact_roundtrip(tmp_path):
    pytest.importorskip("safetensors")
    path = tmp_path / "scales.safetensors"
    expected = CalibrationArtifact(
        scales={"layer.0": torch.tensor([1.0, 2.0])},
        metadata={"alpha": "0.5"},
    )
    expected.save(path)
    restored = CalibrationArtifact.load(path)
    assert restored.metadata == expected.metadata
    assert torch.equal(restored.scales["layer.0"], expected.scales["layer.0"])


def test_smoothquant_factory_requires_calibration_artifact():
    model = nn.Sequential(nn.Linear(2, 2))
    with pytest.raises(ValueError, match="calibration-scales"):
        instrument_reference_backend(
            model,
            PrecisionMap(),
            backend="smoothquant",
        )


def test_smoothquant_factory_loads_artifact_and_labels_backend(tmp_path):
    pytest.importorskip("safetensors")
    path = tmp_path / "scales.safetensors"
    CalibrationArtifact(
        scales={"0": torch.ones(2)},
        metadata={"alpha": "0.5"},
    ).save(path)
    model = nn.Sequential(nn.Linear(2, 2))
    controller = instrument_reference_backend(
        model,
        PrecisionMap(),
        backend="smoothquant",
        calibration_scales=path,
        materialize_weights=False,
    )
    metadata = controller.metadata()
    assert metadata["backend"] == "smoothquant_reference_fake"
    assert metadata["calibration"]["alpha"] == "0.5"
    assert metadata["smoothed_linear_modules"] == 1
