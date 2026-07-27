import torch

from frontierguard.frontier.counterfactual import PrefixOutcome, bypass_gains
from frontierguard.frontier.detector import FrontierDetector, FrontierDetectorConfig
from frontierguard.frontier.signals import token_signals


def test_identical_logits_have_zero_distribution_gap():
    logits = torch.randn(1, 4, 9)
    targets = torch.tensor([[1, 2, 3, 4]])
    signals = token_signals(logits, logits.clone(), targets)
    assert torch.allclose(signals.jsd, torch.zeros_like(signals.jsd), atol=1e-6)
    assert torch.allclose(signals.margin_drop, torch.zeros_like(signals.margin_drop))
    assert torch.allclose(signals.nll_gap, torch.zeros_like(signals.nll_gap))


def test_difference_in_differences_gain():
    fp = [PrefixOutcome(4, 10), PrefixOutcome(5, 10), PrefixOutcome(6, 10)]
    quant = [PrefixOutcome(1, 10), PrefixOutcome(2, 10), PrefixOutcome(7, 10)]
    result = bypass_gains(fp, quant)
    assert abs(result["specific_gain"][0]) < 1e-12
    assert abs(result["specific_gain"][1] - 0.4) < 1e-12


def test_detector_selects_earliest_trustworthy_step():
    detector = FrontierDetector(
        FrontierDetectorConfig(
            combined_threshold=0.1,
            bypass_threshold=0.0,
            require_persistence=False,
        )
    )
    result = detector.detect(
        jsd=[0.1, 0.3, 0.2],
        margin_drop=[0.0, 0.5, 0.1],
        bypass_gain=[0.0, 0.4, 0.1],
        bypass_ci_lower=[0.0, 0.2, 0.0],
    )
    assert result.trustworthy
    assert result.step_index == 1
