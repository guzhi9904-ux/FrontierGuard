from frontierguard.frontier.rejudge import rejudge_counterfactual
from frontierguard.traces.segment import segment_reasoning


def _prefix(seed, continuation, *, success=False):
    return {
        "successes": int(success),
        "trials": 1,
        "outcomes": [
            {
                "seed": seed,
                "success": success,
                "continuation": continuation,
                "extracted_answer": None,
                "output_tokens": 4,
                "truncated": False,
                "latency_seconds": 0.1,
            }
        ],
    }


def test_rejudge_repairs_markdown_answer_and_adds_missing_adjacent_effect():
    response = "Compute 1+1=2.\n\nFinal answer: 2"
    steps = segment_reasoning(response)
    raw = {
        "prefix_indices": [0, 1],
        "fp_outcomes": [
            _prefix(0, " Final answer: 2", success=True),
            _prefix(0, " Final answer: 2", success=True),
        ],
        "quant_outcomes": [
            _prefix(0, " Final answer: 3"),
            _prefix(0, " Final Answer: The result is **2 people**."),
        ],
    }

    result = rejudge_counterfactual(
        raw,
        response=response,
        steps=steps,
        reference_answer="2",
        bootstrap_samples=100,
        min_trustworthy_seeds=1,
    )

    assert result["gain_step_indices"] == [0]
    assert result["specific_gain"] == [1.0]
    assert result["quant_outcomes"][1]["successes"] == 1
    outcome = result["quant_outcomes"][1]["outcomes"][0]
    assert outcome["extracted_answer"] == "2"
    assert outcome["extraction_method"] == "final_marker"
    assert outcome["failure_type"] == "none"
