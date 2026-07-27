import torch
from torch import nn

from frontierguard.attribution.measure import measure_local_nll_rescue
from frontierguard.quant.controller import instrument_linear_layers
from frontierguard.schemas import PrecisionAction, PrecisionMap


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(11, 8)
        self.proj = nn.Linear(8, 11)

    def forward(self, input_ids, use_cache=False):
        del use_cache
        logits = self.proj(self.embed(input_ids))
        return type("Output", (), {"logits": logits})


def test_local_nll_rescue_runs_on_instrumented_model():
    torch.manual_seed(5)
    model = TinyLM().eval()
    low = PrecisionAction(4, 4, 16, weight_group_size=4)
    controller = instrument_linear_layers(
        model, PrecisionMap(default=low), include=r"proj", exclude=None
    )
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    rescue = measure_local_nll_rescue(
        model,
        controller,
        ids,
        target_start=2,
        target_end=5,
        module_names=["proj"],
        bf16_rescue=True,
    )
    assert isinstance(rescue, float)
