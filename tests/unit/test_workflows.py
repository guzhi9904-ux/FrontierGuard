import contextlib

import torch

from frontierguard.frontier.pipeline import TeacherForcedScan
from frontierguard.models.hf_runner import SamplingConfig
from frontierguard.schemas import TraceRecord
from frontierguard.workflows import complete_frontier, counterfactual_trace


class TinyTokenizer:
    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        return_tensors=None,
        return_offsets_mapping=False,
    ):
        del add_special_tokens
        if return_offsets_mapping:
            return {"offset_mapping": [(index, index + 1) for index in range(len(text))]}
        ids = torch.arange(len(text)).unsqueeze(0)
        return {"input_ids": ids}


class TinyCounterfactualRunner:
    def __init__(self):
        self.tokenizer = TinyTokenizer()
        self.device = torch.device("cpu")
        self._full_precision = False

    def encode_chat(self, problem):
        del problem
        return torch.tensor([[1]])

    @contextlib.contextmanager
    def full_precision(self):
        self._full_precision = True
        try:
            yield
        finally:
            self._full_precision = False

    def generate(self, input_ids, sampling, *, progress_callback=None):
        if progress_callback is not None:
            progress_callback(1, sampling.max_new_tokens)
        repaired = input_ids.shape[-1] > 1
        answer = 2 if self._full_precision or repaired else 3
        text = f" Final answer: {answer}"
        return {
            "text": text,
            "token_ids": [answer],
            "output_tokens": 1,
            "truncated": False,
            "latency_seconds": 0.01,
        }


def _trace():
    return TraceRecord(
        problem_id="p1",
        problem="What is 1+1?",
        response="Compute 1+1=2.\n\nFinal answer: 2",
        reference_answer="2",
        extracted_answer="2",
        correct=True,
        steps=[],
        model_id="tiny",
        model_revision=None,
        seed=0,
        generation_config={},
        prompt_hash="hash",
    )


def test_counterfactual_keeps_raw_paired_seed_evidence():
    result = counterfactual_trace(
        TinyCounterfactualRunner(),
        _trace(),
        SamplingConfig(max_new_tokens=8),
        seeds=[0, 1, 2, 3],
        candidate_steps=[0],
        bootstrap_samples=1000,
        min_trustworthy_seeds=4,
    )

    assert result["specific_gain"] == [1.0]
    assert result["bypass_ci_lower"] == [1.0]
    assert result["detection_ci_lower"] == [1.0]
    assert result["statistically_eligible"] == [True]
    quant_after = result["quant_outcomes"][1]
    assert quant_after["successes"] == 4
    assert quant_after["outcomes"][0]["continuation"] == " Final answer: 2"
    assert result["paired_effects"][0]["seed_effects"][0]["specific_delta"] == 1
    assert result["condition_summaries"]["quantized"]["failure_types"] == {
        "none": 4,
        "wrong_answer": 4,
    }


def test_complete_frontier_maps_filtered_position_to_original_step_index():
    scan = TeacherForcedScan(
        token_signals=None,
        step_indices=[0, 2, 5],
        step_jsd=[0.1, 0.2, 0.9],
        step_margin_drop=[0.1, 0.2, 0.9],
        step_nll_gap=[0.0, 0.0, 1.0],
        shortlist=[5],
    )
    result = complete_frontier(
        scan,
        {
            "gain_step_indices": [5],
            "specific_gain": [1.0],
            "bypass_ci_lower": [1.0],
            "bypass_ci_upper": [1.0],
            "detection_ci_lower": [1.0],
            "statistically_eligible": [True],
        },
    )

    assert result["step_index"] == 5
    assert result["step_indices"] == [0, 2, 5]
    assert result["trustworthy"]
    assert result["recovery_frontier_step"] == 5
    assert result["recovery_trustworthy"]


def test_complete_frontier_does_not_invent_window_without_recovery():
    scan = TeacherForcedScan(
        token_signals=None,
        step_indices=[0, 1],
        step_jsd=[0.2, 0.1],
        step_margin_drop=[0.2, 0.1],
        step_nll_gap=[0.2, 0.1],
        shortlist=[0, 1],
    )
    result = complete_frontier(
        scan,
        {
            "gain_step_indices": [0, 1],
            "specific_gain": [0.0, 0.0],
            "bypass_ci_lower": [0.0, 0.0],
            "bypass_ci_upper": [0.0, 0.0],
            "detection_ci_lower": [0.0, 0.0],
            "statistically_eligible": [True, True],
        },
    )

    assert result["first_error_step"] == 0
    assert result["recovery_frontier_step"] is None
    assert result["frontier_window"] is None
