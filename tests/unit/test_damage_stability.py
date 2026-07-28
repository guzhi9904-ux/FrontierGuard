import pytest

from frontierguard.attribution.stability import (
    aggregate_damage_rows,
    exact_patch_diagnostics,
    relative_depth_key,
    shared_rank_correlation,
    top_k_jaccard,
)


def _row(problem, module, score, depth, projection="up_proj", **extra):
    return {
        "problem_id": problem,
        "module_name": module,
        "predicted_nll_rescue": score,
        "relative_depth": depth,
        "family": "mlp",
        "projection": projection,
        "layer_index": round(depth * 10),
        "parameter_count": 100,
        **extra,
    }


def test_aggregation_averages_trace_repeats_inside_problem():
    rows = [
        _row("p1", "m1", 1.0, 0.2),
        _row("p1", "m1", 3.0, 0.2),
        _row("p2", "m1", -1.0, 0.2),
    ]
    ranking = aggregate_damage_rows(
        rows,
        key=lambda row: row["module_name"],
        bootstrap_samples=200,
        seed=4,
    )

    assert ranking[0]["mean"] == pytest.approx(0.5)
    assert ranking[0]["problems"] == 2
    assert ranking[0]["positive_fraction"] == pytest.approx(0.5)


def test_relative_depth_key_is_architecture_normalized():
    assert relative_depth_key(_row("p", "m", 1, 0.49), bins=4) == (
        "depth_2_of_4.mlp.up_proj"
    )
    assert relative_depth_key(_row("p", "m", 1, 1.0), bins=4) == (
        "depth_4_of_4.mlp.up_proj"
    )


def test_rank_stability_metrics():
    left = [{"key": "a"}, {"key": "b"}, {"key": "c"}]
    right = [{"key": "a"}, {"key": "c"}, {"key": "b"}]

    assert top_k_jaccard(left, right, k=2) == pytest.approx(1 / 3)
    assert shared_rank_correlation(left, right) == pytest.approx(0.5)


def test_exact_patch_diagnostics_separates_matched_control():
    rows = [
        _row(
            "p1",
            "m1",
            0.2,
            0.2,
            exact_nll_rescue=0.1,
            exact_role="predicted_top",
        ),
        _row(
            "p1",
            "m2",
            -0.2,
            0.5,
            exact_nll_rescue=-0.1,
            exact_role="matched_random",
        ),
    ]
    result = exact_patch_diagnostics(rows)

    assert result["rows"] == 2
    assert result["sign_agreement"] == 1.0
    assert result["predicted_top_mean_exact_rescue"] == pytest.approx(0.1)
    assert result["matched_random_mean_exact_rescue"] == pytest.approx(-0.1)
