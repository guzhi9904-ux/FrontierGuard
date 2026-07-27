import torch

from frontierguard.frontier.signals import compare_to_sketch, sketch_logits


def test_logit_sketch_identical_distribution_has_zero_gap():
    torch.manual_seed(17)
    logits = torch.randn(1, 5, 23)
    targets = torch.tensor([[1, 2, 3, 4, 5]])
    sketch = sketch_logits(logits, targets, top_k=7)
    comparison = compare_to_sketch(logits.clone(), targets, sketch)
    assert torch.allclose(comparison["jsd"], torch.zeros_like(comparison["jsd"]), atol=1e-7)
    assert torch.allclose(comparison["margin_drop"], torch.zeros_like(comparison["margin_drop"]))
    assert torch.allclose(comparison["nll_gap"], torch.zeros_like(comparison["nll_gap"]))
