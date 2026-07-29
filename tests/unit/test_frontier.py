import torch

from frontierguard.frontier.counterfactual import (
    PrefixOutcome,
    SeedOutcome,
    bypass_gains,
    paired_specific_effect,
)
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
    assert result.confidence == 0.95
    assert result.evidence_score > 0


def _prefix(*values):
    outcomes = tuple(
        SeedOutcome(seed=index, success=value)
        for index, value in enumerate(values)
    )
    return PrefixOutcome(sum(values), len(values), outcomes)


def test_paired_specific_effect_preserves_seed_transitions():
    effect = paired_specific_effect(
        _prefix(True, True, True, True),
        _prefix(True, True, True, True),
        _prefix(False, False, False, False),
        _prefix(True, True, True, False),
        samples=1000,
        seed=3,
    )

    assert effect.trials == 4
    assert effect.estimate == 0.75
    assert effect.lower > 0
    assert [item["specific_delta"] for item in effect.seed_effects] == [1, 1, 1, 0]


def test_detector_never_claims_trust_without_a_confidence_lower_bound():
    detector = FrontierDetector(
        FrontierDetectorConfig(
            combined_threshold=-1.0,
            require_persistence=False,
        )
    )
    result = detector.detect(
        jsd=[0.1, 0.4],
        margin_drop=[0.1, 0.4],
        bypass_gain=[0.0, 1.0],
    )
    assert not result.trustworthy


def test_short_reasoning_traces_use_every_step():
    detector = FrontierDetector()
    assert detector.shortlist(
        [0.1, 0.9, 0.2, 0.3],
        [0.3, 0.2, 0.1, 0.9],
        nll_gap=[0.4, 0.3, 0.2, 0.1],
        top_k=1,
        exhaustive_threshold=4,
    ) == [0, 1, 2, 3]


def test_long_trace_shortlist_adds_nll_candidates_and_neighbors():
    detector = FrontierDetector()
    selected = detector.shortlist(
        [0.0] * 20,
        [0.0] * 20,
        nll_gap=[0.0] * 17 + [5.0, 0.0, 0.0],
        top_k=2,
        exhaustive_threshold=16,
        neighbor_radius=1,
    )

    assert {16, 17, 18}.issubset(selected)
