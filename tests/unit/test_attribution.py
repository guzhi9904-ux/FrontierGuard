import pytest

from frontierguard.attribution.measure import component_rescue_action
from frontierguard.attribution.rescue import RescueObservation
from frontierguard.schemas import PrecisionAction


def test_weight_rescue_preserves_higher_activation_and_kv_precision():
    low = PrecisionAction(weight_bits=4, activation_bits=16, kv_bits=16)
    action = component_rescue_action(low, high_bits=8, component="weight")

    assert action.weight_bits == 8
    assert action.activation_bits == 16
    assert action.kv_bits == 16


def test_activation_rescue_does_not_change_weight_or_kv():
    low = PrecisionAction(weight_bits=4, activation_bits=4, kv_bits=16)
    action = component_rescue_action(low, high_bits=8, component="activation")

    assert action.weight_bits == 4
    assert action.activation_bits == 8
    assert action.kv_bits == 16


def test_noop_rescue_is_rejected():
    low = PrecisionAction(weight_bits=8, activation_bits=8, kv_bits=16)
    with pytest.raises(ValueError, match="does not increase"):
        component_rescue_action(low, high_bits=8, component="weight")


def test_missing_outcome_does_not_dilute_local_rescue():
    observation = RescueObservation(
        problem_id="p1",
        trace_id="p1:0",
        step_index=2,
        module_name="layer_13",
        local_rescue=0.02,
        outcome_rescue=None,
        frontier_confidence=None,
    )

    assert observation.combined(local_weight=0.3) == pytest.approx(0.02)
