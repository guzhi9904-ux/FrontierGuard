from pathlib import Path

from frontierguard.config import load_experiment
from frontierguard.schemas import PrecisionAction, PrecisionMap


def test_precision_map_roundtrip():
    value = PrecisionMap(
        default=PrecisionAction(4, 4, 4),
        modules={"model.layers.0.mlp.down_proj": PrecisionAction(8, 8, 8)},
        metadata={"budget": 4.4},
    )
    restored = PrecisionMap.from_dict(value.to_dict())
    assert restored == value


def test_experiment_includes_are_resolved():
    root = Path(__file__).parents[2]
    config = load_experiment(root / "configs/experiment/e2_frontier.yaml")
    assert config["model"]["id"].endswith("1.5B")
    assert config["quant"]["default"]["weight_bits"] == 4
    assert config["experiment"]["detector"]["bypass_weight"] == 0.5
