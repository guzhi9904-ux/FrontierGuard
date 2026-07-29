import pytest

from frontierguard.schemas import PrecisionAction
from frontierguard.utils.provenance import (
    precision_label,
    stable_fingerprint,
    validate_output_precision_label,
)


def test_precision_label_and_output_validation():
    action = PrecisionAction(weight_bits=4, activation_bits=8, kv_bits=16)
    assert precision_label(action) == "w4a8kv16"
    validate_output_precision_label("results/frontier_w4a8kv16.jsonl", action)
    validate_output_precision_label("results/frontier.jsonl", action)
    with pytest.raises(ValueError, match="filename encodes"):
        validate_output_precision_label("results/frontier_w4a4kv16.jsonl", action)


def test_stable_fingerprint_ignores_mapping_order():
    assert stable_fingerprint({"a": 1, "b": 2}) == stable_fingerprint(
        {"b": 2, "a": 1}
    )
