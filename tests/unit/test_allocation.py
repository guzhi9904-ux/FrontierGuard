from frontierguard.allocation.cost import CostModel, ModuleCost
from frontierguard.allocation.greedy import additive_greedy, measured_greedy
from frontierguard.schemas import PrecisionAction, PrecisionMap


def test_additive_greedy_obeys_budget():
    result = additive_greedy(
        {"a": 3.0, "b": 2.0, "c": 1.0},
        {"a": 2.0, "b": 1.0, "c": 1.0},
        budget=2.0,
    )
    assert set(result.selected) == {"b", "c"}
    assert result.cost == 2.0


def test_measured_greedy_remeasures_interactions():
    def utility(selected):
        if selected == frozenset({"a", "b"}):
            return 0.5
        return sum({"a": 2.0, "b": 1.8, "c": 1.0}[name] for name in selected)

    result = measured_greedy(["a", "b", "c"], utility, lambda _: 1.0, budget=2.0)
    assert result.selected == ("a", "c")


def test_cost_model_effective_bits():
    costs = CostModel([ModuleCost("a", 100), ModuleCost("b", 100)])
    low = PrecisionAction(weight_bits=4)
    high = PrecisionAction(weight_bits=8)
    precision_map = PrecisionMap(default=low, modules={"a": high})
    assert costs.effective_weight_bits(precision_map) == 6.0
    assert costs.weight_bytes(precision_map) == 150.0


def test_kv_cost_uses_highest_action_once_per_layer():
    costs = CostModel(
        [
            ModuleCost("layer0.q", 10, layer_index=0, kv_elements_per_token=20),
            ModuleCost("layer0.v", 10, layer_index=0, kv_elements_per_token=20),
            ModuleCost("layer1.q", 10, layer_index=1, kv_elements_per_token=20),
        ]
    )
    low = PrecisionAction(4, 4, 4)
    high = PrecisionAction(8, 8, 8)
    precision_map = PrecisionMap(default=low, modules={"layer0.v": high})
    assert costs.kv_bytes_per_token(precision_map) == (20 * 8 + 20 * 4) / 8
