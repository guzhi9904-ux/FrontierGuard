import torch
from torch import nn

from frontierguard.models.hf_runner import HFRunner, SamplingConfig


def test_full_precision_context_restores_kv_bits():
    runner = HFRunner(nn.Linear(2, 2), tokenizer=object(), kv_bits=4)
    with runner.full_precision():
        assert runner.kv_bits == 16
    assert runner.kv_bits == 4


class TinyGenerationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, past_key_values=None, use_cache=True):
        del past_key_values, use_cache
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 3)
        logits[..., 0] = 1.0
        return type(
            "Output",
            (),
            {"logits": logits, "past_key_values": None},
        )


class TinyTokenizer:
    eos_token_id = 2

    @staticmethod
    def decode(tokens, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(map(str, tokens))


def test_generate_reports_token_progress():
    runner = HFRunner(TinyGenerationModel(), TinyTokenizer())
    progress = []

    result = runner.generate(
        torch.tensor([[1, 2]]),
        SamplingConfig(temperature=0.0, max_new_tokens=3),
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )

    assert result["output_tokens"] == 3
    assert progress == [(1, 3), (2, 3), (3, 3)]
