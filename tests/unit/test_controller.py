import torch
from torch import nn

from frontierguard.quant.controller import instrument_linear_layers
from frontierguard.schemas import PrecisionAction, PrecisionMap


def test_controller_intervention_is_reversible():
    torch.manual_seed(3)
    model = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 2)).eval()
    low = PrecisionAction(4, 4, 16, weight_group_size=2)
    high = PrecisionAction(8, 8, 16, weight_group_size=2)
    controller = instrument_linear_layers(model, PrecisionMap(default=low), exclude=None)
    inputs = torch.randn(2, 4)

    low_output = model(inputs)
    with controller.disabled():
        fp_output = model(inputs)
    with controller.intervention(["0"], high):
        intervention_output = model(inputs)
    restored_output = model(inputs)

    assert controller.module_names == ["0", "2"]
    assert not torch.allclose(low_output, fp_output)
    assert not torch.allclose(intervention_output, low_output)
    assert torch.allclose(restored_output, low_output)


def test_unwrap_restores_original_modules():
    model = nn.Sequential(nn.Linear(3, 2))
    controller = instrument_linear_layers(model, PrecisionMap(), exclude=None)
    controller.unwrap()
    assert isinstance(model[0], nn.Linear)
    assert controller.module_names == []


def test_materialized_weights_preserve_quant_and_full_precision_outputs():
    torch.manual_seed(11)
    model = nn.Sequential(nn.Linear(4, 3)).eval()
    inputs = torch.randn(2, 4)
    action = PrecisionAction(4, 4, 16, weight_group_size=2)
    controller = instrument_linear_layers(
        model, PrecisionMap(default=action), exclude=None
    )
    quant_before = model(inputs)
    with controller.disabled():
        fp_before = model(inputs)
    controller.materialize_weights()
    quant_after = model(inputs)
    with controller.disabled():
        fp_after = model(inputs)
    assert torch.allclose(quant_before, quant_after)
    assert torch.allclose(fp_before, fp_after)
    controller.unwrap()
    assert torch.allclose(model(inputs), fp_before)
