import pytest

from frontierguard.evaluation.selective import (
    ModuleRescueSpec,
    build_module_precision_map,
    build_precision_map,
    matched_random_module_specs,
    paired_success_lift,
    parse_rescue_spec,
    rank_damage_modules,
    random_layer_specs,
    select_module_names,
    summarize_generation_condition,
)
from frontierguard.models.adapters import ModuleDescriptor
from frontierguard.schemas import PrecisionAction


DESCRIPTORS = [
    ModuleDescriptor("layers.0.q_proj", 0, "attention", "q_proj", 10),
    ModuleDescriptor("layers.0.up_proj", 0, "mlp", "up_proj", 30),
    ModuleDescriptor("layers.1.q_proj", 1, "attention", "q_proj", 10),
    ModuleDescriptor("layers.1.up_proj", 1, "mlp", "up_proj", 50),
    ModuleDescriptor("layers.2.q_proj", 2, "attention", "q_proj", 10),
    ModuleDescriptor("layers.2.up_proj", 2, "mlp", "up_proj", 90),
]


def _row(problem, seed, correct, failure="none"):
    return {
        "problem_id": problem,
        "seed": seed,
        "correct": correct,
        "truncated": False,
        "output_tokens": 10,
        "failure": {
            "failure_type": failure,
            "repetition_fraction": 0.01,
        },
    }


def test_parse_and_expand_layer_family_rescue():
    spec = parse_rescue_spec("candidate=0.mlp,1")
    names = select_module_names(DESCRIPTORS, spec.selectors)

    assert spec.label == "candidate=0.mlp,1"
    assert names == (
        "layers.0.up_proj",
        "layers.1.q_proj",
        "layers.1.up_proj",
    )


def test_precision_budget_is_parameter_weighted():
    low = PrecisionAction(4, 8, 16)
    high = PrecisionAction(8, 8, 16)
    precision_map, metadata = build_precision_map(
        DESCRIPTORS,
        parse_rescue_spec("layer1=1"),
        low=low,
        high=high,
    )

    assert precision_map.action_for("layers.1.q_proj") == high
    assert precision_map.action_for("layers.0.q_proj") == low
    assert metadata["selected_parameter_count"] == 60
    assert metadata["instrumented_parameter_count"] == 200
    assert metadata["high_precision_parameter_fraction"] == pytest.approx(0.3)
    assert metadata["effective_weight_bits"] == pytest.approx(5.2)


def test_exact_module_precision_map_and_damage_ranking():
    low = PrecisionAction(4, 8, 16)
    high = PrecisionAction(8, 8, 16)
    scores = []
    for problem_id, values in {
        "p1": {"layers.0.q_proj": 2.0, "layers.1.q_proj": 1.0},
        "p2": {"layers.0.q_proj": 1.0, "layers.1.q_proj": -1.0},
    }.items():
        for module_name, value in values.items():
            scores.append(
                {
                    "problem_id": problem_id,
                    "module_name": module_name,
                    "predicted_nll_rescue": value,
                }
            )

    ranking = rank_damage_modules(
        scores,
        DESCRIPTORS,
        minimum_problem_fraction=1.0,
        minimum_positive_fraction=0.5,
        bootstrap_samples=100,
        seed=3,
    )
    assert [item["key"] for item in ranking] == ["layers.0.q_proj"]
    precision_map, metadata = build_module_precision_map(
        DESCRIPTORS,
        ModuleRescueSpec(
            "ranked_top1",
            ("layers.0.q_proj",),
            {"selection_method": "test"},
        ),
        low=low,
        high=high,
    )
    assert precision_map.action_for("layers.0.q_proj") == high
    assert precision_map.action_for("layers.0.up_proj") == low
    assert metadata["selected_module_count"] == 1
    assert metadata["selection_method"] == "test"


def test_matched_random_module_controls_preserve_projection_mix():
    specs = matched_random_module_specs(
        DESCRIPTORS,
        ("layers.0.q_proj", "layers.0.up_proj"),
        count=2,
        seed=7,
        prefix="ranked_top2",
    )
    by_name = {item.name: item for item in DESCRIPTORS}
    assert len(specs) == 2
    assert len({item.module_names for item in specs}) == 2
    for spec in specs:
        projections = sorted(by_name[name].projection for name in spec.module_names)
        assert projections == ["q_proj", "up_proj"]
        assert not set(spec.module_names) & {"layers.0.q_proj", "layers.0.up_proj"}


def test_random_controls_are_unique_and_reproducible():
    first = random_layer_specs(range(8), budget=2, count=5, seed=7)
    second = random_layer_specs(range(8), budget=2, count=5, seed=7)

    assert first == second
    assert len({item.selectors for item in first}) == 5
    assert all(len(item.selectors) == 2 for item in first)


def test_generation_summary_and_paired_lift():
    baseline = [
        _row("p1", 0, False, "wrong_answer"),
        _row("p1", 1, True),
        _row("p2", 0, False, "wrong_answer"),
        _row("p2", 1, False, "wrong_answer"),
    ]
    method = [
        _row("p1", 0, True),
        _row("p1", 1, True),
        _row("p2", 0, True),
        _row("p2", 1, False, "wrong_answer"),
    ]

    summary = summarize_generation_condition(method)
    lift = paired_success_lift(baseline, method, bootstrap_samples=100, bootstrap_seed=3)

    assert summary["accuracy"] == 0.75
    assert lift["estimate"] == 0.5
    assert lift["improved"] == 2
    assert lift["regressed"] == 0


@pytest.mark.parametrize(
    "value",
    ["bad", "name=", "name=x", "name=1.foo", "bad name=1"],
)
def test_invalid_rescue_specs_are_rejected(value):
    with pytest.raises(ValueError):
        parse_rescue_spec(value)
