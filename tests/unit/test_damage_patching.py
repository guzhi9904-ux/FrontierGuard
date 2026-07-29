from types import SimpleNamespace

import pytest
import torch
from torch import nn

from frontierguard.attribution.patching import measure_compression_damage
from frontierguard.models.adapters import ModelAdapter
from frontierguard.quant.controller import instrument_linear_layers
from frontierguard.schemas import PrecisionAction, PrecisionMap


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.down_proj = nn.Linear(8, 8, bias=False)

    def forward(self, hidden):
        return hidden + self.down_proj(torch.tanh(self.q_proj(hidden)))


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8)
        self.layers = nn.ModuleList([TinyBlock(), TinyBlock()])
        self.lm_head = nn.Linear(8, 16, bias=False)

    def forward(
        self,
        input_ids,
        *,
        use_cache=False,
        logits_to_keep=None,
    ):
        del use_cache
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        logits = self.lm_head(hidden)
        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


def _instrumented_model():
    torch.manual_seed(3)
    model = TinyCausalLM()
    low = PrecisionAction(
        weight_bits=3,
        activation_bits=4,
        kv_bits=16,
        weight_group_size=4,
    )
    controller = instrument_linear_layers(
        model,
        PrecisionMap(default=low),
        exclude=r"(lm_head|embed_tokens)",
    )
    descriptors = ModelAdapter("tiny").describe_modules(model)
    return model, controller, descriptors


def test_straight_through_context_changes_gradients_not_forward_values():
    model, controller, _descriptors = _instrumented_model()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    ordinary = model(input_ids=input_ids, use_cache=False).logits.detach()
    with controller.straight_through_gradients():
        surrogate = model(input_ids=input_ids, use_cache=False).logits.detach()

    assert torch.equal(ordinary, surrogate)
    assert not any(
        wrapper.gradient_ste_enabled for wrapper in controller.wrappers.values()
    )


def test_damage_attribution_scores_all_modules_and_exact_controls():
    model, controller, descriptors = _instrumented_model()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    result = measure_compression_damage(
        model,
        controller,
        input_ids,
        target_start=1,
        target_end=4,
        descriptors=descriptors,
        exact_top_k=1,
        exact_random_k=1,
        random_seed=9,
    )

    assert len(result.measurements) == 4
    assert result.bf16_nll >= 0
    assert result.quantized_nll >= 0
    assert all(torch.isfinite(torch.tensor(item.predicted_nll_rescue)) for item in result.measurements)
    exact = [item for item in result.measurements if item.exact_role is not None]
    assert {item.exact_role for item in exact} == {"predicted_top", "matched_random"}
    assert all(item.exact_nll_rescue is not None for item in exact)
    assert controller.wrappers["layers.0.q_proj"].gradient_ste_enabled is False


def test_damage_attribution_rejects_invalid_window():
    model, controller, descriptors = _instrumented_model()
    with pytest.raises(ValueError, match="target window"):
        measure_compression_damage(
            model,
            controller,
            torch.tensor([[1, 2, 3]]),
            target_start=1,
            target_end=3,
            descriptors=descriptors,
        )
