from torch import nn

from frontierguard.models.adapters import ModelAdapter
from frontierguard.quant.controller import instrument_linear_layers
from frontierguard.schemas import PrecisionMap


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.down_proj = nn.Linear(4, 4)


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Block(), Block()])


def test_adapter_describes_wrapped_and_unwrapped_models():
    model = TinyTransformer()
    adapter = ModelAdapter("tiny")
    before = adapter.describe_modules(model)
    instrument_linear_layers(model, PrecisionMap(), exclude=None)
    after = adapter.describe_modules(model)
    assert [(item.name, item.parameter_count) for item in before] == [
        (item.name, item.parameter_count) for item in after
    ]


def test_adapter_exposes_single_layer_family_groups():
    model = TinyTransformer()
    groups = ModelAdapter("tiny").group_names(model)

    assert groups["layer_0.attention"] == ["layers.0.q_proj"]
    assert groups["layer_0.mlp"] == ["layers.0.down_proj"]
    assert groups["layer_1"] == ["layers.1.down_proj", "layers.1.q_proj"]
