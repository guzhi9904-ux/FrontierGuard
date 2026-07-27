from torch import nn

from frontierguard.models.hf_runner import HFRunner


def test_full_precision_context_restores_kv_bits():
    runner = HFRunner(nn.Linear(2, 2), tokenizer=object(), kv_bits=4)
    with runner.full_precision():
        assert runner.kv_bits == 16
    assert runner.kv_bits == 4
